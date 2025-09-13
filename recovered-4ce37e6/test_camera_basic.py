#!/usr/bin/env python3
"""
Basic camera test to see if MediaPipe can detect hands.
"""

import cv2
import mediapipe as mp
import time

def test_camera():
    """Test basic camera and hand detection."""
    print("Testing camera and MediaPipe hand detection...")

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,  # Lower threshold
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera 0")
        return False

    print("Camera opened successfully")
    print("Put your hand in front of the camera...")
    print("Press 'q' to quit")

    frame_count = 0
    hand_detected_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from camera")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror image

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Check for hands
            if results.multi_hand_landmarks:
                hand_detected_count += 1
                print(f"Frame {frame_count}: HAND DETECTED! ({hand_detected_count} total)")

                # Draw hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Show hand info
                cv2.putText(frame, "HAND DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                if frame_count % 30 == 0:  # Every 30 frames
                    print(f"Frame {frame_count}: No hand detected")

                cv2.putText(frame, "No hand - wave your hand!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show frame
            cv2.imshow('Camera Test', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Auto-quit after some time if no hands detected
            if frame_count > 300 and hand_detected_count == 0:
                print("No hands detected after 300 frames, stopping...")
                break

    except Exception as e:
        print(f"Error during camera test: {e}")
        return False

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Test complete: {hand_detected_count} hands detected in {frame_count} frames")
    return hand_detected_count > 0

if __name__ == "__main__":
    success = test_camera()
    if success:
        print("✓ Camera and hand detection working!")
    else:
        print("✗ Camera or hand detection not working")