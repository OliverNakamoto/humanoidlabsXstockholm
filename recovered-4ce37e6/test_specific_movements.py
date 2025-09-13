#!/usr/bin/env python3
"""
Test specific robot movements with hand gestures.

This test demonstrates how to achieve specific robot positions:
1. Turn robot left
2. Open gripper to 50%

Shows the required hand positions to achieve these commands.
"""

import sys
import time
import math

# Add LeRobot to path
sys.path.insert(0, 'lerobot/src')

def test_specific_movements():
    """Test specific robot movements with hand gestures."""
    print("=" * 60)
    print("SPECIFIC MOVEMENT TEST")
    print("Turn Robot Left + 50% Gripper Opening")
    print("=" * 60)

    try:
        # Import simplified hand leader
        from lerobot.teleoperators.hand_leader.hand_leader_simple import HandLeaderSimple
        from lerobot.teleoperators.hand_leader.config_hand_leader import HandLeaderConfig

        # Create config
        config = HandLeaderConfig(
            camera_index=0,
            urdf_path="dummy_robot.urdf",
            target_frame_name="gripper_frame_link",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        )

        # Create and connect hand leader
        print("1. Connecting to hand tracking...")
        hand_leader = HandLeaderSimple(config)
        hand_leader.connect(calibrate=False)
        print("   [OK] Connected\n")

        # Target movements
        target_shoulder_pan = -30.0  # Turn left (negative = left)
        target_gripper = 50.0        # Half open gripper

        print("TARGET MOVEMENTS:")
        print(f"- Turn robot LEFT: shoulder_pan = {target_shoulder_pan}")
        print(f"- Open gripper 50%: gripper = {target_gripper}%")
        print()

        print("INSTRUCTIONS:")
        print("Move your hand to achieve these targets!")
        print("- Move hand to the RIGHT to turn robot LEFT")
        print("- Partially close your fingers for 50% gripper")
        print("- Watch the values and try to match the targets")
        print()

        print("Current Hand Position → Robot Commands:")
        print("-" * 60)

        best_match_score = float('inf')
        best_action = None

        try:
            for i in range(100):  # 10 seconds at 10Hz
                start_time = time.perf_counter()

                # Get current action
                action = hand_leader.get_action()

                # Extract key values
                shoulder_pan = action.get("shoulder_pan.pos", 0)
                gripper = action.get("gripper.pos", 0)

                # Calculate how close we are to targets
                pan_error = abs(shoulder_pan - target_shoulder_pan)
                gripper_error = abs(gripper - target_gripper)
                total_error = pan_error + gripper_error

                # Visual indicators
                pan_status = "[OK]" if pan_error < 5 else "[--]"
                gripper_status = "[OK]" if gripper_error < 10 else "[--]"

                # Display current values
                print(f"Frame {i+1:2d}: Pan={shoulder_pan:6.1f} {pan_status} | Grip={gripper:5.1f}% {gripper_status} | Error={total_error:5.1f}")

                # Track best match
                if total_error < best_match_score:
                    best_match_score = total_error
                    best_action = action.copy()

                # Check if we hit the target
                if pan_error < 5 and gripper_error < 10:
                    print("\nTARGET ACHIEVED!")
                    print(f"   Shoulder Pan: {shoulder_pan:.1f} (target: {target_shoulder_pan})")
                    print(f"   Gripper: {gripper:.1f}% (target: {target_gripper}%)")

                    print("\nFull robot command:")
                    for joint, value in action.items():
                        print(f"   {joint:<20}: {value:7.2f}")

                    print("\nThis is the command that would be sent to your robot!")
                    break

                # Wait for next cycle
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0, 0.1 - elapsed))

        except KeyboardInterrupt:
            print("\nTest stopped by user")

        # Show best attempt if target wasn't achieved
        if best_action:
            print(f"\nBEST ATTEMPT (Error: {best_match_score:.1f}):")
            print(f"   Shoulder Pan: {best_action.get('shoulder_pan.pos', 0):.1f}")
            print(f"   Gripper: {best_action.get('gripper.pos', 0):.1f}%")
            print("\nBest robot command:")
            for joint, value in best_action.items():
                print(f"   {joint:<20}: {value:7.2f}")

        # Cleanup
        hand_leader.disconnect()
        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

def test_movement_mapping():
    """Show the mapping between hand positions and robot commands."""
    print("\n" + "=" * 60)
    print("MOVEMENT MAPPING GUIDE")
    print("=" * 60)

    mapping = {
        "Turn Robot LEFT": "Move your hand to the RIGHT",
        "Turn Robot RIGHT": "Move your hand to the LEFT",
        "Robot Arm UP": "Move your hand UP",
        "Robot Arm DOWN": "Move your hand DOWN",
        "Robot Forward": "Move your hand BACK (away from camera)",
        "Robot Back": "Move your hand FORWARD (toward camera)",
        "Close Gripper": "Pinch fingers together",
        "Open Gripper": "Spread fingers apart"
    }

    for robot_action, hand_gesture in mapping.items():
        print(f"{robot_action:<15} <- {hand_gesture}")

    print()
    print("Hand Position Values:")
    print("- Workspace center: X=0, Y=0.3m, Z=0.25m")
    print("- Hand left/right: X = -0.3 to +0.3 meters")
    print("- Hand up/down: Y = 0.1 to 0.5 meters")
    print("- Hand forward/back: Z = 0.1 to 0.4 meters")
    print("- Gripper: 0% = open, 100% = closed")

def main():
    print("Hand-to-Robot Movement Test")

    # Show mapping first
    test_movement_mapping()

    # Run the specific movement test
    try:
        return test_specific_movements()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 0

if __name__ == '__main__':
    sys.exit(main())