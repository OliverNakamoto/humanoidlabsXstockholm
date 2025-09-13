#!/usr/bin/env python3
"""
Pinch Detection Test - Visual demonstration of hand tracking pinch gestures.

Tests the MediaPipe hand tracking with real-time visualization of:
- Hand landmarks
- Thumb-index finger distance
- Pinch percentage calculation
- Robot gripper position mapping

Usage:
    python test_pinch_detection.py --camera 0
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import math

class PinchDetectionTest:
    """Visual test for pinch gesture detection."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Calibration for pinch detection
        self.baseline_distance = None
        self.max_distance = None
        self.calibration_samples = []

        # Visual feedback
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }

    def start(self):
        """Start the pinch detection test."""
        print("Starting Pinch Detection Test")
        print("=" * 40)
        print("Controls:")
        print("  SPACE - Calibrate open hand (fingers spread)")
        print("  C - Calibrate closed pinch")
        print("  R - Reset calibration")
        print("  Q - Quit")
        print("=" * 40)

        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            self._main_loop()
        finally:
            self._cleanup()

    def _main_loop(self):
        """Main processing loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )

                    # Calculate and visualize pinch
                    self._process_pinch(frame, hand_landmarks, w, h)

            # Draw UI
            self._draw_ui(frame)

            # Show frame
            cv2.imshow('Pinch Detection Test', frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - calibrate open
                if results.multi_hand_landmarks:
                    self._calibrate_open(results.multi_hand_landmarks[0], w)
            elif key == ord('c'):  # C - calibrate closed
                if results.multi_hand_landmarks:
                    self._calibrate_closed(results.multi_hand_landmarks[0], w)
            elif key == ord('r'):  # R - reset
                self._reset_calibration()

    def _process_pinch(self, frame, landmarks, frame_width, frame_height):
        """Process and visualize pinch gesture."""
        # Get thumb tip (landmark 4) and index tip (landmark 8)
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]

        # Convert to pixel coordinates
        thumb_px = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
        index_px = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))

        # Calculate distance
        distance = math.sqrt(
            (thumb_px[0] - index_px[0])**2 +
            (thumb_px[1] - index_px[1])**2
        )

        # Draw connection line between thumb and index
        line_color = self.colors['yellow']
        if self.baseline_distance and self.max_distance:
            # Color based on pinch percentage
            pinch_pct = self._calculate_pinch_percentage(distance)
            if pinch_pct > 80:
                line_color = self.colors['red']  # Closed
            elif pinch_pct > 40:
                line_color = self.colors['yellow']  # Partial
            else:
                line_color = self.colors['green']  # Open

        cv2.line(frame, thumb_px, index_px, line_color, 3)

        # Draw circles on fingertips
        cv2.circle(frame, thumb_px, 8, self.colors['blue'], -1)  # Thumb
        cv2.circle(frame, index_px, 8, self.colors['red'], -1)   # Index

        # Draw distance text
        mid_point = (
            (thumb_px[0] + index_px[0]) // 2,
            (thumb_px[1] + index_px[1]) // 2
        )

        distance_text = f"{distance:.1f}px"
        cv2.putText(frame, distance_text,
                   (mid_point[0] - 30, mid_point[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)

        # Calculate and display pinch percentage
        if self.baseline_distance and self.max_distance:
            pinch_pct = self._calculate_pinch_percentage(distance)

            # Draw pinch bar
            self._draw_pinch_bar(frame, pinch_pct)

            # Display robot gripper position
            gripper_pos = pinch_pct  # 0-100 range
            self._draw_gripper_visualization(frame, gripper_pos)

    def _calculate_pinch_percentage(self, current_distance):
        """Calculate pinch percentage from current thumb-index distance."""
        if not (self.baseline_distance and self.max_distance):
            return 0

        # Normalize distance: baseline=0%, max=100% open
        normalized = (current_distance - self.max_distance) / (self.baseline_distance - self.max_distance)
        pinch_pct = (1.0 - np.clip(normalized, 0, 1)) * 100
        return pinch_pct

    def _draw_pinch_bar(self, frame, pinch_percentage):
        """Draw visual pinch percentage bar."""
        h, w = frame.shape[:2]

        # Bar dimensions
        bar_x, bar_y = 50, h - 100
        bar_width, bar_height = 300, 30

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     self.colors['white'], 2)

        # Fill bar based on pinch percentage
        fill_width = int((pinch_percentage / 100) * bar_width)
        if fill_width > 0:
            # Color gradient: green -> yellow -> red
            if pinch_percentage < 50:
                color = self.colors['green']
            elif pinch_percentage < 80:
                color = self.colors['yellow']
            else:
                color = self.colors['red']

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         color, -1)

        # Text
        cv2.putText(frame, f"Pinch: {pinch_percentage:.1f}%",
                   (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.colors['white'], 2)

    def _draw_gripper_visualization(self, frame, gripper_position):
        """Draw robot gripper visualization."""
        h, w = frame.shape[:2]

        # Gripper drawing area
        gripper_x, gripper_y = w - 200, h - 150
        gripper_width = 100

        # Draw gripper jaws (two rectangles that close based on position)
        max_opening = 50
        current_opening = max_opening * (1 - gripper_position / 100)

        # Left jaw
        left_jaw = (
            gripper_x - int(current_opening),
            gripper_y,
            gripper_x - int(current_opening) + 20,
            gripper_y + 60
        )

        # Right jaw
        right_jaw = (
            gripper_x + int(current_opening) - 20,
            gripper_y,
            gripper_x + int(current_opening),
            gripper_y + 60
        )

        # Draw jaws
        cv2.rectangle(frame, (left_jaw[0], left_jaw[1]), (left_jaw[2], left_jaw[3]),
                     self.colors['blue'], -1)
        cv2.rectangle(frame, (right_jaw[0], right_jaw[1]), (right_jaw[2], right_jaw[3]),
                     self.colors['blue'], -1)

        # Draw gripper label
        cv2.putText(frame, f"Gripper: {gripper_position:.1f}%",
                   (gripper_x - 50, gripper_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.colors['white'], 2)

    def _draw_ui(self, frame):
        """Draw user interface elements."""
        h, w = frame.shape[:2]

        # Status text
        status_lines = [
            "Pinch Detection Test",
            f"Camera: {self.camera_index}",
            "",
            "Controls:",
            "SPACE - Cal. Open Hand",
            "C - Cal. Closed Pinch",
            "R - Reset Calibration",
            "Q - Quit"
        ]

        # Calibration status
        if self.baseline_distance and self.max_distance:
            status_lines.extend([
                "",
                "✓ Calibrated",
                f"Open: {self.max_distance:.1f}px",
                f"Closed: {self.baseline_distance:.1f}px"
            ])
        else:
            status_lines.extend([
                "",
                "⚠ Need Calibration",
                "1. Spread fingers → SPACE",
                "2. Pinch closed → C"
            ])

        # Draw status panel background
        panel_height = len(status_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (300, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, panel_height), self.colors['white'], 2)

        # Draw text
        for i, line in enumerate(status_lines):
            y_pos = 35 + i * 25
            color = self.colors['green'] if line.startswith('✓') else self.colors['white']
            if line.startswith('⚠'):
                color = self.colors['yellow']

            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _calibrate_open(self, landmarks, frame_width):
        """Calibrate open hand position."""
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        ) * frame_width

        self.max_distance = distance
        print(f"Calibrated OPEN hand: {distance:.1f}px")

    def _calibrate_closed(self, landmarks, frame_width):
        """Calibrate closed pinch position."""
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        ) * frame_width

        self.baseline_distance = distance
        print(f"Calibrated CLOSED pinch: {distance:.1f}px")

    def _reset_calibration(self):
        """Reset calibration data."""
        self.baseline_distance = None
        self.max_distance = None
        self.calibration_samples = []
        print("Calibration reset")

    def _cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Test MediaPipe pinch detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()

    try:
        test = PinchDetectionTest(args.camera)
        test.start()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()