#!/usr/bin/env python3
"""
Hand Tracking IPC Server with Visual Output
Runs MediaPipe hand tracking in separate process and provides IPC interface.
Shows mirrored visual feedback by saving frames to a file.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HandTracker:
    """Hand tracking using MediaPipe with visual output."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        self.is_connected = False
        self.is_tracking = False
        self.calibrated = False

        # Tracking data
        self.current_data = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'pinch': 0.0,
            'valid': False,
            'timestamp': 0.0
        }
        self.data_lock = threading.Lock()

        # Calibration values (default for typical usage)
        self.palm_bbox_far = 50
        self.palm_bbox_near = 150
        self.initial_thumb_index_dist = 100

        # Frame for visualization
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Visual output file
        self.visual_output_path = "hand_tracking_visual.jpg"

    def connect(self):
        """Connect to camera and initialize MediaPipe."""
        logger.info(f"Connecting to camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Initialize MediaPipe hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.is_connected = True
        logger.info("Hand tracker connected successfully")

    def start_tracking(self):
        """Start the tracking thread with visual output."""
        if not self.is_connected:
            raise RuntimeError("Tracker not connected")

        self.is_tracking = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        logger.info("Hand tracking thread started with visual output")

    def _tracking_loop(self):
        """Main tracking loop with visual feedback."""
        while self.is_tracking:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Draw visual feedback
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Get key points
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                # Calculate positions (normalized to robot workspace)
                x = (thumb_tip.x - 0.5) * 0.6  # -0.3 to 0.3 meters
                y = (0.5 - thumb_tip.y) * 0.6  # Invert Y for intuitive control
                z = 0.2 - thumb_tip.z * 0.3    # 0.05 to 0.35 meters

                # Calculate pinch (distance between thumb and index)
                pinch_dist = np.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2 +
                    (thumb_tip.z - index_tip.z) ** 2
                )

                # Normalize pinch to 0-100 (gripper percentage)
                pinch = max(0, min(100, (1.0 - pinch_dist * 5) * 100))

                # Update tracking data
                with self.data_lock:
                    self.current_data = {
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'pinch': float(pinch),
                        'valid': True,
                        'timestamp': time.time()
                    }

                # Draw position info on frame
                thumb_pixel = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pixel = (int(index_tip.x * w), int(index_tip.y * h))

                # Draw pinch indicator
                cv2.line(display_frame, thumb_pixel, index_pixel, (255, 0, 255), 2)
                cv2.circle(display_frame, thumb_pixel, 8, (0, 255, 0), -1)
                cv2.circle(display_frame, index_pixel, 8, (0, 0, 255), -1)

                # Add text overlay
                cv2.putText(display_frame, f"X: {x:.2f}m", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Y: {y:.2f}m", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Z: {z:.2f}m", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Grip: {pinch:.0f}%", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Draw workspace indicator
                cv2.rectangle(display_frame, (w-150, 10), (w-10, 100), (100, 100, 100), 2)
                workspace_x = int(w-80 + x*100)
                workspace_y = int(55 - y*100)
                cv2.circle(display_frame, (workspace_x, workspace_y), 5, (0, 255, 255), -1)
                cv2.putText(display_frame, "Workspace", (w-140, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            else:
                # No hand detected
                with self.data_lock:
                    self.current_data['valid'] = False

                cv2.putText(display_frame, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Show your hand to the camera", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Add title
            cv2.putText(display_frame, "Hand Tracking Control (Mirrored)", (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Save frame to file for viewing
            with self.frame_lock:
                self.current_frame = display_frame
                cv2.imwrite(self.visual_output_path, display_frame)

            time.sleep(0.01)  # ~100Hz

    def get_current_position(self):
        """Get current hand position and pinch state."""
        with self.data_lock:
            data = self.current_data.copy()

        if not data['valid']:
            return 0.0, 0.0, 0.2, 0.0  # Default position

        return data['x'], data['y'], data['z'], data['pinch']

    def calibrate(self):
        """Skip calibration, use defaults."""
        logger.info("Using default calibration values")
        self.calibrated = True
        self.start_tracking()

    def disconnect(self):
        """Disconnect and cleanup."""
        self.is_tracking = False
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()

        if self.hands:
            self.hands.close()

        # Clean up visual output file
        if os.path.exists(self.visual_output_path):
            try:
                os.remove(self.visual_output_path)
            except:
                pass

        self.is_connected = False
        logger.info("Hand tracker disconnected")

class HandTrackingServer(HTTPServer):
    """HTTP server for hand tracking IPC."""

    def __init__(self, server_address, RequestHandlerClass, hand_tracker):
        super().__init__(server_address, RequestHandlerClass)
        self.hand_tracker = hand_tracker

class RequestHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for hand tracking data."""

    def do_GET(self):
        """Handle GET requests."""
        try:
            if self.path == '/position':
                # Get current position
                x, y, z, pinch = self.server.hand_tracker.get_current_position()
                data = {
                    'x': x, 'y': y, 'z': z,
                    'pinch': pinch,
                    'valid': self.server.hand_tracker.current_data['valid'],
                    'timestamp': time.time()
                }
                self._send_json_response(data)

            elif self.path == '/status':
                # Get tracker status
                data = {
                    'connected': self.server.hand_tracker.is_connected,
                    'tracking': self.server.hand_tracker.is_tracking,
                    'calibrated': self.server.hand_tracker.calibrated
                }
                self._send_json_response(data)

            elif self.path == '/calibrate':
                # Start calibration
                self.server.hand_tracker.calibrate()
                self._send_json_response({'success': True})

            else:
                self.send_error(404, "Not Found")

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self._send_json_response({'success': False, 'error': str(e)}, 500)

    def _send_json_response(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        json_data = json.dumps(data).encode('utf-8')
        self.wfile.write(json_data)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

def main():
    """Main entry point."""
    # Initialize hand tracker
    tracker = HandTracker(camera_index=0)
    tracker.connect()

    # Use default calibration
    logger.warning("Using default calibration values")
    tracker.calibrated = True
    tracker.start_tracking()

    # Start HTTP server
    server_address = ('localhost', 8888)
    httpd = HandTrackingServer(server_address, RequestHandler, tracker)

    logger.info(f"Hand tracking server started on http://{server_address[0]}:{server_address[1]}")
    logger.info("Available endpoints:")
    logger.info("  GET /position  - Get current hand position")
    logger.info("  GET /status    - Get tracker status")
    logger.info("  GET /calibrate - Start calibration")
    logger.info(f"Visual output: {tracker.visual_output_path}")
    logger.info("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        tracker.disconnect()
        httpd.server_close()

if __name__ == "__main__":
    main()