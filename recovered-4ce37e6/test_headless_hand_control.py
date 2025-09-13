#!/usr/bin/env python3
"""
Headless test of hand control without GUI windows.
Tests camera, MediaPipe, and robot command generation.
"""

import cv2
import mediapipe as mp
import time
import sys

def test_headless_hand_detection():
    """Test hand detection without GUI."""
    print("=" * 60)
    print("HEADLESS HAND ROBOT CONTROL TEST")
    print("=" * 60)
    print("Testing camera and MediaPipe without GUI...")

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera 0")
        return False

    print("✓ Camera opened successfully")
    print("✓ MediaPipe initialized")
    print()
    print("INSTRUCTIONS:")
    print("- Put your hand in front of the camera")
    print("- Move hand RIGHT to turn robot LEFT")
    print("- Make partial pinch for 50% gripper")
    print("- Watch for 'TARGET ACHIEVED' message")
    print()

    frame_count = 0
    hand_detected_count = 0
    best_commands = None
    target_achieved = False

    try:
        print("Starting hand detection (20 seconds)...")
        start_time = time.time()

        while time.time() - start_time < 20:  # 20 second test
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror image

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_detected_count += 1
                landmarks = results.multi_hand_landmarks[0]

                # Calculate hand position (simplified)
                h, w = frame.shape[:2]

                # Get palm center (wrist)
                wrist = landmarks.landmark[0]
                palm_x = wrist.x * w
                palm_y = wrist.y * h

                # Calculate thumb-index distance for pinch
                thumb_tip = landmarks.landmark[4]
                index_tip = landmarks.landmark[8]
                thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5 * w

                # Convert to robot coordinates
                # X: 0-640 pixels -> -0.3 to 0.3 meters
                robot_x = (palm_x - 320) / 320 * 0.3

                # Calculate robot commands (simplified mapping)
                shoulder_pan = robot_x * 200  # Scale for joint range
                pinch_percent = max(0, min(100, (50 - thumb_index_dist) * 2))  # Rough pinch estimation

                # Robot commands
                commands = {
                    "shoulder_pan.pos": shoulder_pan,
                    "shoulder_lift.pos": -15.0,
                    "elbow_flex.pos": -15.0,
                    "wrist_flex.pos": -10.0,
                    "wrist_roll.pos": shoulder_pan * 0.15,
                    "gripper.pos": pinch_percent,
                }

                # Check if close to targets
                pan_target = -30.0  # Turn left
                grip_target = 50.0  # 50% gripper

                pan_error = abs(commands["shoulder_pan.pos"] - pan_target)
                grip_error = abs(commands["gripper.pos"] - grip_target)

                if frame_count % 10 == 0:  # Every 10 frames
                    print(f"Frame {frame_count:3d}: Pan={commands['shoulder_pan.pos']:6.1f} (target:-30) | Grip={commands['gripper.pos']:5.1f}% (target:50%)")

                # Check for target achievement
                if pan_error < 8 and grip_error < 15 and not target_achieved:
                    target_achieved = True
                    best_commands = commands.copy()
                    print()
                    print("🎯 TARGET ACHIEVED!")
                    print(f"   Robot will turn LEFT: {commands['shoulder_pan.pos']:.1f}°")
                    print(f"   Gripper will open: {commands['gripper.pos']:.1f}%")
                    print()
                    print("ROBOT COMMANDS:")
                    for joint, value in commands.items():
                        print(f"   {joint:<20}: {value:7.2f}")
                    print()
                    print("✓ This would control your physical robot!")
                    break

            else:
                if frame_count % 60 == 0:  # Every 2 seconds
                    print(f"Frame {frame_count:3d}: No hand detected - wave your hand!")

            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    finally:
        cap.release()
        hands.close()

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Hands detected: {hand_detected_count}")

    if target_achieved:
        print("✅ TARGET MOVEMENTS ACHIEVED!")
        print("✅ Robot control commands generated successfully")
        print("✅ System ready for physical robot")
    else:
        print("⚠️  Target not achieved - try again with:")
        print("   - Hand positioned to the RIGHT")
        print("   - Fingers partially closed (50% pinch)")

    return target_achieved

if __name__ == "__main__":
    success = test_headless_hand_detection()
    sys.exit(0 if success else 1)