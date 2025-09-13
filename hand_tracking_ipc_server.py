#!/usr/bin/env python3
"""
MediaPipe Hand Tracking IPC Server

Runs in separate environment with MediaPipe + OpenCV.
Exposes hand tracking via HTTP API for LeRobot to consume.

Usage:
    python hand_tracking_ipc_server.py --port 8888
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HandTracker:
    """MediaPipe-based hand tracker for teleoperation."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None

        # Calibration data - inspired by keyboard_teleop
        self.calibrated = False
        self.palm_bbox_far = None
        self.palm_bbox_near = None
        self.initial_thumb_index_dist = None

        # Enhanced workspace bounds with safety margins
        self.workspace_bounds = {
            'x': (-0.3, 0.3),   # Left/Right limits (meters)
            'y': (-0.3, 0.3),   # Forward/Back limits (meters)
            'z': (0.05, 0.4),   # Up/Down limits (meters)
        }

        # Position smoothing and safety (inspired by absolute_position_control) - DISABLED FOR SPEED
        self.smoothing_factor = 0.0  # Disabled for maximum responsiveness (0 = no smoothing, 1 = ignore new positions)
        self.max_velocity = 10.0     # Very high for testing (m/s)
        self.safety_margin = 0.01    # Reduced safety margin (meters)
        self.max_step_change = 2000  # High for fast servo response

        # Reference position for calibration
        self.reference_position = None
        self.calibration_center = None

        # Previous position for smoothing
        self.previous_position = None
        self.last_update_time = time.time()

        # Current tracking data
        self.current_data = {
            'x': 0.0, 'y': 0.2, 'z': 0.2, 'pinch': 0.0,
            'valid': False, 'timestamp': time.time()
        }
        self.data_lock = threading.Lock()

        # Tracking thread
        self.tracking_thread = None
        self.stop_tracking = False

    def connect(self):
        """Initialize camera and MediaPipe."""
        logger.info(f"Connecting to camera {self.camera_index}...")

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.camera_index}")

        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Initialize MediaPipe
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        logger.info("Hand tracker connected successfully")

    def calibrate(self):
        """Enhanced interactive calibration routine inspired by keyboard_teleop."""
        if not self.cap or not self.hands:
            raise RuntimeError("Must connect before calibrating")

        logger.info("Starting enhanced hand tracking calibration...")

        # Step 1: Calibrate workspace center
        print("\n" + "="*60)
        print("STEP 1: CALIBRATE WORKSPACE CENTER")
        print("="*60)
        print("Place your hand at the CENTER of your desired workspace")
        print("This will be your reference point (origin)")
        print("Press SPACE when ready, ESC to cancel")

        center_data = self._interactive_calibration("CENTER")
        if center_data is None:
            raise RuntimeError("Calibration cancelled")

        # Store reference position
        self.calibration_center = {
            'palm_x': center_data['palm_center'][0],
            'palm_y': center_data['palm_center'][1],
            'palm_size': center_data['palm_bbox_size']
        }

        # Step 2: Calibrate depth range (far)
        print("\n" + "="*60)
        print("STEP 2: CALIBRATE FAR POSITION")
        print("="*60)
        print("Extend your hand FAR from the camera (back edge of workspace)")
        print("Keep your hand at the same X,Y position as center")
        print("Press SPACE when ready, ESC to cancel")

        far_data = self._interactive_calibration("FAR")
        if far_data is None:
            raise RuntimeError("Calibration cancelled")

        self.palm_bbox_far = far_data['palm_bbox_size']

        # Step 3: Calibrate depth range (near)
        print("\n" + "="*60)
        print("STEP 3: CALIBRATE NEAR POSITION")
        print("="*60)
        print("Move your hand CLOSE to the camera (front edge of workspace)")
        print("Keep your hand at the same X,Y position as center")
        print("Press SPACE when ready, ESC to cancel")

        near_data = self._interactive_calibration("NEAR")
        if near_data is None:
            raise RuntimeError("Calibration cancelled")

        self.palm_bbox_near = near_data['palm_bbox_size']

        # Step 4: Calibrate grip
        print("\n" + "="*60)
        print("STEP 4: CALIBRATE GRIP BASELINE")
        print("="*60)
        print("Open your hand completely (relaxed open grip)")
        print("Press SPACE when ready, ESC to cancel")

        grip_data = self._interactive_calibration("GRIP")
        if grip_data is None:
            raise RuntimeError("Calibration cancelled")

        # Average thumb-index distances from all calibration points
        self.initial_thumb_index_dist = (
            center_data['thumb_index_dist'] +
            far_data['thumb_index_dist'] +
            near_data['thumb_index_dist'] +
            grip_data['thumb_index_dist']
        ) / 4

        self.calibrated = True

        # Set reference position for absolute positioning
        self.reference_position = self.calibration_center.copy()

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        print(f"Center position: ({self.calibration_center['palm_x']}, {self.calibration_center['palm_y']})")
        print(f"Depth range: Far={self.palm_bbox_far:.1f}px, Near={self.palm_bbox_near:.1f}px")
        print(f"Grip baseline: {self.initial_thumb_index_dist:.1f}px")
        print("="*60 + "\n")

        cv2.destroyAllWindows()
        self.start_tracking()

    def _interactive_calibration(self, position_name: str) -> Optional[Dict]:
        """Interactive calibration with live preview."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # Mirror
            h, w = frame.shape[:2]

            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Draw instructions
            cv2.putText(frame, f"CALIBRATION - {position_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | ESC: Cancel", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Show palm bbox preview
                    palm_data = self._calculate_palm_bbox(landmarks, w, h)
                    x, y, pw, ph = palm_data['bbox']
                    cv2.rectangle(frame, (x, y), (x + pw, y + ph), (0, 255, 0), 2)
                    cv2.putText(frame, f"Palm: {palm_data['palm_width']:.1f}px", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('Hand Tracking Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE - capture
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    palm_data = self._calculate_palm_bbox(landmarks, w, h)

                    # Calculate thumb-index distance
                    thumb_tip = landmarks.landmark[4]
                    index_tip = landmarks.landmark[8]
                    thumb_index_dist = np.sqrt(
                        (thumb_tip.x - index_tip.x)**2 +
                        (thumb_tip.y - index_tip.y)**2
                    ) * w

                    return {
                        'palm_bbox_size': palm_data['palm_width'],
                        'palm_center': palm_data['center'],
                        'thumb_index_dist': thumb_index_dist
                    }
                else:
                    print("No hand detected! Please try again.")

            elif key == 27:  # ESC - cancel
                return None

    def _calculate_palm_bbox(self, landmarks, frame_width, frame_height):
        """Calculate palm bounding box from hand landmarks."""
        # Key landmarks: wrist, index MCP, pinky MCP
        wrist = landmarks.landmark[0]
        index_mcp = landmarks.landmark[5]
        pinky_mcp = landmarks.landmark[17]

        # Convert to pixels
        points_px = [
            (int(wrist.x * frame_width), int(wrist.y * frame_height)),
            (int(index_mcp.x * frame_width), int(index_mcp.y * frame_height)),
            (int(pinky_mcp.x * frame_width), int(pinky_mcp.y * frame_height))
        ]

        # Bounding box
        x_coords = [p[0] for p in points_px]
        y_coords = [p[1] for p in points_px]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        palm_width = x_max - x_min
        palm_height = y_max - y_min
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        return {
            'bbox': (x_min, y_min, palm_width, palm_height),
            'center': (center_x, center_y),
            'palm_width': palm_width,
            'palm_height': palm_height
        }

    def start_tracking(self):
        """Start background tracking thread."""
        if not self.calibrated:
            # Use default calibration for immediate start
            logger.warning("Using default calibration values")
            self.palm_bbox_far = 80
            self.palm_bbox_near = 120
            self.initial_thumb_index_dist = 50
            self.calibrated = True

        self.stop_tracking = False
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        logger.info("Hand tracking thread started")

    def _clamp_to_workspace(self, position):
        """Ensure position is within safe workspace bounds (inspired by absolute_position_control)."""
        clamped = position.copy()

        for i, axis in enumerate(['x', 'y', 'z']):
            min_val, max_val = self.workspace_bounds[axis]
            min_val += self.safety_margin
            max_val -= self.safety_margin
            clamped[i] = np.clip(position[i], min_val, max_val)

        if not np.array_equal(position, clamped):
            logger.debug(f"Position clamped from {position} to {clamped}")

        return clamped

    def _smooth_position(self, current, target):
        """Apply exponential smoothing to position changes."""
        if current is None:
            return target
        return current * self.smoothing_factor + target * (1 - self.smoothing_factor)

    def _limit_velocity(self, current_pos, target_pos):
        """Limit the velocity of position changes."""
        if current_pos is None:
            return target_pos

        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)

        if distance > self.max_velocity * dt:
            # Scale down movement to respect velocity limit
            delta = delta * (self.max_velocity * dt / distance)
            limited_pos = current_pos + delta
            logger.debug(f"Velocity limited: {distance/dt:.3f} m/s -> {self.max_velocity:.3f} m/s")
            return limited_pos

        return target_pos

    def _calculate_absolute_position(self, palm_data, landmarks, w, h):
        """Calculate absolute position in robot workspace coordinates."""
        if not self.calibrated or self.calibration_center is None:
            # Fallback to old method if not calibrated
            return self._calculate_position_fallback(palm_data, landmarks, w, h)

        # Calculate relative position from calibration center
        palm_center_x, palm_center_y = palm_data['center']

        # Get relative position in pixels
        dx_px = palm_center_x - self.calibration_center['palm_x']
        dy_px = palm_center_y - self.calibration_center['palm_y']

        # Convert to normalized coordinates (-1 to 1)
        # Assuming workspace spans about 300px in each direction
        workspace_span_px = 300
        norm_x = dx_px / workspace_span_px
        norm_y = -dy_px / workspace_span_px  # Invert Y for proper coordinate system

        # Z from palm size (depth)
        palm_size = palm_data['palm_width']
        if self.palm_bbox_near > self.palm_bbox_far:
            norm_z = (palm_size - self.palm_bbox_far) / (self.palm_bbox_near - self.palm_bbox_far)
            norm_z = np.clip(norm_z, 0, 1)
            norm_z = 1.0 - norm_z  # far=1, near=0
        else:
            norm_z = 0.5

        # Convert to absolute robot workspace coordinates
        x_range = self.workspace_bounds['x'][1] - self.workspace_bounds['x'][0]
        y_range = self.workspace_bounds['y'][1] - self.workspace_bounds['y'][0]
        z_range = self.workspace_bounds['z'][1] - self.workspace_bounds['z'][0]

        x = norm_x * x_range * 0.5  # Scale to workspace size
        y = norm_y * y_range * 0.5  # Scale to workspace size
        z = self.workspace_bounds['z'][0] + norm_z * z_range

        return np.array([x, y, z])

    def _calculate_position_fallback(self, palm_data, landmarks, w, h):
        """Fallback position calculation for uncalibrated mode."""
        # Normalized coordinates (0-1)
        norm_x = palm_data['center'][0] / w
        norm_y = 1.0 - (palm_data['center'][1] / h)  # Invert Y

        # Z from palm size
        palm_size = palm_data['palm_width']
        if self.palm_bbox_near and self.palm_bbox_far and self.palm_bbox_near > self.palm_bbox_far:
            norm_z = (palm_size - self.palm_bbox_far) / (self.palm_bbox_near - self.palm_bbox_far)
            norm_z = np.clip(norm_z, 0, 1)
            norm_z = 1.0 - norm_z  # far=1, near=0
        else:
            norm_z = 0.5

        # Convert to robot workspace coordinates
        x = self.workspace_bounds['x'][0] + norm_x * (self.workspace_bounds['x'][1] - self.workspace_bounds['x'][0])
        y = self.workspace_bounds['y'][0] + norm_y * (self.workspace_bounds['y'][1] - self.workspace_bounds['y'][0])
        z = self.workspace_bounds['z'][0] + norm_z * (self.workspace_bounds['z'][1] - self.workspace_bounds['z'][0])

        return np.array([x, y, z])

    def _tracking_loop(self):
        """Main tracking loop with visual feedback."""
        while not self.stop_tracking:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Draw background info panel
            self._draw_info_panel(frame, w, h)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]

                # Draw hand landmarks and connections
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                # Calculate palm data
                palm_data = self._calculate_palm_bbox(landmarks, w, h)

                # Draw palm bounding box
                self._draw_palm_bbox(frame, palm_data)

                # Draw key landmarks
                self._draw_key_landmarks(frame, landmarks, w, h)

                # Calculate absolute position using enhanced method
                raw_position = self._calculate_absolute_position(palm_data, landmarks, w, h)

                # Apply safety bounds
                clamped_position = self._clamp_to_workspace(raw_position)

                # Apply smoothing
                smoothed_position = self._smooth_position(self.previous_position, clamped_position)

                # Apply velocity limiting
                final_position = self._limit_velocity(self.previous_position, smoothed_position)

                # Update previous position for next iteration
                self.previous_position = final_position

                # Extract coordinates
                x, y, z = final_position

                # Pinch calculation
                thumb_tip = landmarks.landmark[4]
                index_tip = landmarks.landmark[8]
                thumb_index_dist = np.sqrt(
                    (thumb_tip.x - index_tip.x)**2 +
                    (thumb_tip.y - index_tip.y)**2
                ) * w

                if self.initial_thumb_index_dist:
                    pinch_ratio = thumb_index_dist / self.initial_thumb_index_dist
                    pinch = (1.0 - np.clip(pinch_ratio, 0, 1)) * 100
                else:
                    # Fallback pinch calculation
                    pinch_ratio = min(1.0, max(0.0, thumb_index_dist / 50.0))
                    pinch = (1.0 - pinch_ratio) * 100

                # Draw pinch visualization
                self._draw_pinch_indicator(frame, thumb_tip, index_tip, thumb_index_dist, pinch, w, h)

                # Draw coordinate overlay
                self._draw_coordinates_overlay(frame, x, y, z, pinch, palm_data['palm_width'])

                # Draw calibration status
                self._draw_calibration_status(frame, raw_position, clamped_position, final_position)

                # Update shared data
                with self.data_lock:
                    self.current_data = {
                        'x': float(x), 'y': float(y), 'z': float(z), 'pinch': float(pinch),
                        'valid': True, 'timestamp': time.time()
                    }

            else:
                # No hand detected - draw warning
                self._draw_no_hand_warning(frame, w, h)

                # No hand detected
                with self.data_lock:
                    self.current_data.update({'valid': False, 'timestamp': time.time()})

            try:
                cv2.imshow("Hand Tracking - Enhanced", frame)
                cv2.moveWindow("Hand Tracking - Enhanced", 100, 100)  # Force window position
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to exit
                    self.stop_tracking = True
                elif key == ord('r') or key == ord('R'):  # R to recalibrate
                    logger.info("Recalibration requested")
                    self._reset_calibration()
                    self.calibrate()
            except cv2.error as e:
                logger.warning(f"CV window error: {e}")
                # Continue without window

            time.sleep(0.001)  # ~1000 FPS max for faster response

    def _draw_info_panel(self, frame, w, h):
        """Draw background information panel."""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Title
        cv2.putText(frame, "Hand Tracking - Enhanced", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Status info
        status_text = f"Calibrated: {'YES' if self.calibrated else 'NO'}"
        cv2.putText(frame, status_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.calibrated else (0, 0, 255), 1)

        # Controls
        cv2.putText(frame, "ESC: Exit | R: Recalibrate", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_palm_bbox(self, frame, palm_data):
        """Draw palm bounding box and center."""
        x, y, pw, ph = palm_data['bbox']
        center_x, center_y = palm_data['center']

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + pw, y + ph), (255, 0, 0), 2)

        # Center point
        cv2.circle(frame, (center_x, center_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), 12, (255, 255, 255), 2)

        # Palm size label
        cv2.putText(frame, f"Palm: {palm_data['palm_width']:.0f}px",
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Center coordinates
        cv2.putText(frame, f"({center_x}, {center_y})",
                   (center_x + 15, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_key_landmarks(self, frame, landmarks, w, h):
        """Draw key landmarks with labels."""
        key_landmarks = {
            'Wrist': (0, (255, 255, 0)),
            'Thumb': (4, (255, 0, 255)),
            'Index': (8, (0, 255, 255)),
            'Middle': (12, (255, 128, 0)),
            'Ring': (16, (128, 255, 0)),
            'Pinky': (20, (0, 128, 255))
        }

        for name, (idx, color) in key_landmarks.items():
            landmark = landmarks.landmark[idx]
            px = int(landmark.x * w)
            py = int(landmark.y * h)

            # Draw larger circle for key points
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)

            # Label
            cv2.putText(frame, name, (px + 10, py - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def _draw_pinch_indicator(self, frame, thumb_tip, index_tip, distance, pinch_percent, w, h):
        """Draw pinch visualization."""
        # Convert to pixel coordinates
        thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_px = (int(index_tip.x * w), int(index_tip.y * h))

        # Line between thumb and index
        line_color = (0, 255, 0) if pinch_percent < 30 else (0, 255, 255) if pinch_percent < 70 else (0, 0, 255)
        cv2.line(frame, thumb_px, index_px, line_color, 3)

        # Distance text
        mid_x = (thumb_px[0] + index_px[0]) // 2
        mid_y = (thumb_px[1] + index_px[1]) // 2
        cv2.putText(frame, f"{distance:.1f}px", (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)

        # Pinch percentage bar
        bar_x, bar_y, bar_w, bar_h = w - 120, 20, 100, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * pinch_percent / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), line_color, -1)
        cv2.putText(frame, f"Pinch: {pinch_percent:.0f}%", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_coordinates_overlay(self, frame, x, y, z, pinch, palm_size):
        """Draw coordinate information overlay."""
        # Position on the left side
        start_y = 100
        line_height = 25

        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, start_y - 10), (300, start_y + line_height * 6), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Robot coordinates
        cv2.putText(frame, "ROBOT COORDINATES:", (10, start_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"X: {x:+.3f}m", (10, start_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Y: {y:+.3f}m", (10, start_y + line_height * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Z: {z:+.3f}m", (10, start_y + line_height * 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Pinch: {pinch:.1f}%", (10, start_y + line_height * 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Palm Size: {palm_size:.1f}px", (10, start_y + line_height * 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_no_hand_warning(self, frame, w, h):
        """Draw warning when no hand is detected."""
        # Center warning text
        text = "NO HAND DETECTED"
        font_scale = 1.2
        thickness = 3
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        # Background rectangle
        cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)

        # Warning text
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 0, 255), thickness)

        # Instruction text
        instruction = "Place your hand in view"
        inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        inst_x = (w - inst_size[0]) // 2
        cv2.putText(frame, instruction, (inst_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)

    def _draw_calibration_status(self, frame, raw_pos, clamped_pos, final_pos):
        """Draw calibration and processing status information."""
        h, w = frame.shape[:2]

        # Position on the right side
        start_x = w - 280
        start_y = 100
        line_height = 20

        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x - 10, start_y - 10), (w - 10, start_y + line_height * 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Calibration status
        cal_status = "CALIBRATED" if self.calibrated else "UNCALIBRATED"
        cal_color = (0, 255, 0) if self.calibrated else (0, 0, 255)
        cv2.putText(frame, f"Status: {cal_status}", (start_x, start_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1)

        # Position processing steps
        cv2.putText(frame, "POSITION PROCESSING:", (start_x, start_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(frame, f"Raw: ({raw_pos[0]:.3f}, {raw_pos[1]:.3f}, {raw_pos[2]:.3f})",
                   (start_x, start_y + line_height * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, f"Clamped: ({clamped_pos[0]:.3f}, {clamped_pos[1]:.3f}, {clamped_pos[2]:.3f})",
                   (start_x, start_y + line_height * 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, f"Final: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})",
                   (start_x, start_y + line_height * 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Safety parameters
        cv2.putText(frame, f"Smoothing: {self.smoothing_factor:.1f}",
                   (start_x, start_y + line_height * 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, f"Max Vel: {self.max_velocity:.2f}m/s",
                   (start_x, start_y + line_height * 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Workspace bounds indicator
        if self.calibrated and self.calibration_center:
            cv2.putText(frame, f"Center: ({self.calibration_center['palm_x']:.0f}, {self.calibration_center['palm_y']:.0f})",
                       (start_x, start_y + line_height * 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def get_position(self) -> Dict[str, Any]:
        """Get current hand position data."""
        with self.data_lock:
            return self.current_data.copy()

    def disconnect(self):
        """Clean up resources."""
        self.stop_tracking = True

        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)

        if self.hands:
            self.hands.close()

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        logger.info("Hand tracker disconnected")

    def _reset_calibration(self):
        """Reset all calibration data."""
        self.calibrated = False
        self.palm_bbox_far = None
        self.palm_bbox_near = None
        self.initial_thumb_index_dist = None
        self.calibration_center = None
        self.reference_position = None
        self.previous_position = None
        logger.info("Calibration data reset")

    def get_calibration_info(self):
        """Get current calibration information."""
        return {
            'calibrated': self.calibrated,
            'palm_bbox_far': self.palm_bbox_far,
            'palm_bbox_near': self.palm_bbox_near,
            'initial_thumb_index_dist': self.initial_thumb_index_dist,
            'calibration_center': self.calibration_center,
            'workspace_bounds': self.workspace_bounds,
            'smoothing_factor': self.smoothing_factor,
            'max_velocity': self.max_velocity
        }


class HandTrackingHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for hand tracking API."""

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/position':
            # Get current hand position
            data = self.server.hand_tracker.get_position()
            self._send_json_response(data)

        elif parsed_path.path == '/status':
            # Get tracker status
            tracker = self.server.hand_tracker
            status = {
                'connected': tracker.cap is not None and tracker.cap.isOpened(),
                'calibrated': tracker.calibrated,
                'tracking': tracker.tracking_thread is not None and tracker.tracking_thread.is_alive(),
                'timestamp': time.time()
            }
            self._send_json_response(status)

        elif parsed_path.path == '/calibrate':
            # Start calibration
            try:
                self.server.hand_tracker.calibrate()
                self._send_json_response({'success': True, 'message': 'Calibration completed'})
            except Exception as e:
                self._send_json_response({'success': False, 'error': str(e)}, 500)

        elif parsed_path.path == '/calibration_info':
            # Get calibration information
            info = self.server.hand_tracker.get_calibration_info()
            self._send_json_response(info)

        elif parsed_path.path == '/reset_calibration':
            # Reset calibration
            try:
                self.server.hand_tracker._reset_calibration()
                self._send_json_response({'success': True, 'message': 'Calibration reset'})
            except Exception as e:
                self._send_json_response({'success': False, 'error': str(e)}, 500)

        else:
            self._send_json_response({'error': 'Endpoint not found'}, 404)

    def _send_json_response(self, data: Dict, status: int = 200):
        """Send JSON response."""
        json_data = json.dumps(data).encode('utf-8')

        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json_data)))
        self.send_header('Access-Control-Allow-Origin', '*')  # CORS
        self.end_headers()
        self.wfile.write(json_data)


class HandTrackingHTTPServer(HTTPServer):
    """HTTP server with embedded hand tracker."""

    def __init__(self, server_address, handler_class, hand_tracker):
        super().__init__(server_address, handler_class)
        self.hand_tracker = hand_tracker


def main():
    parser = argparse.ArgumentParser(description='MediaPipe Hand Tracking IPC Server')
    parser.add_argument('--port', type=int, default=8888, help='Server port (default: 8888)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration on startup')
    args = parser.parse_args()

    # Initialize hand tracker
    tracker = HandTracker(camera_index=args.camera)

    try:
        tracker.connect()

        if args.calibrate:
            tracker.calibrate()
        else:
            tracker.start_tracking()

        # Start HTTP server
        server = HandTrackingHTTPServer(('localhost', args.port), HandTrackingHTTPHandler, tracker)

        logger.info(f"Enhanced Hand tracking server started on http://localhost:{args.port}")
        logger.info("Available endpoints:")
        logger.info("  GET /position         - Get current hand position")
        logger.info("  GET /status           - Get tracker status")
        logger.info("  GET /calibrate        - Start enhanced calibration")
        logger.info("  GET /calibration_info - Get calibration information")
        logger.info("  GET /reset_calibration - Reset calibration data")
        logger.info("\nEnhancements from keyboard_teleop:")
        logger.info("  - Absolute position control (not delta)")
        logger.info("  - Workspace bounds with safety margins")
        logger.info("  - Position smoothing and velocity limiting")
        logger.info("  - 4-step calibration process")
        logger.info("  - Visual feedback with processing steps")
        logger.info("\nPress Ctrl+C to stop, R to recalibrate in video window")

        server.serve_forever()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        tracker.disconnect()


if __name__ == '__main__':
    main()