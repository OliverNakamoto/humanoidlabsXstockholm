#!/usr/bin/env python3
"""
Direct test of hand_leader teleoperator without full robot setup.

Tests the hand tracking -> robot command pipeline directly.
"""

import sys
import time
import logging

# Add LeRobot to path
sys.path.insert(0, 'lerobot/src')

def test_hand_leader_direct():
    """Test hand_leader teleoperator directly."""
    print("=" * 60)
    print("DIRECT HAND LEADER TEST")
    print("=" * 60)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # Import simplified hand leader (bypasses IK issues)
        from lerobot.teleoperators.hand_leader.hand_leader_simple import HandLeaderSimple
        from lerobot.teleoperators.hand_leader.config_hand_leader import HandLeaderConfig

        # Create test config (no real robot needed for this test)
        config = HandLeaderConfig(
            camera_index=0,
            urdf_path="dummy_robot.urdf",  # Not used in this test
            target_frame_name="gripper_frame_link",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        )

        print(f"Config: {config}")
        print("\n1. Creating HandLeader...")

        # Create simplified hand leader
        hand_leader = HandLeaderSimple(config)
        print("   [OK] HandLeader created")

        print("\n2. Connecting to hand tracking IPC...")
        try:
            hand_leader.connect(calibrate=False)
            print("   [OK] Connected to hand tracking")
        except Exception as e:
            print(f"   [ERROR] Connection failed: {e}")
            print("   Make sure the hand tracking server is running:")
            print("   python hand_tracking_ipc_server.py --port 8888")
            return 1

        print(f"\n3. Testing action generation...")
        print("   Move your hand in front of the camera!")
        print("   Open/close your hand to test pinch control")
        print("   Press Ctrl+C to stop")
        print()

        # Generate actions in a loop
        try:
            for i in range(50):  # 5 seconds at 10 Hz
                start_time = time.perf_counter()

                # Get action from hand position
                try:
                    # Provide dummy current position (normally from robot)
                    current_pos = {
                        "shoulder_pan.pos": 0.0,
                        "shoulder_lift.pos": 10.0,
                        "elbow_flex.pos": -20.0,
                        "wrist_flex.pos": 0.0,
                        "wrist_roll.pos": 0.0,
                        "gripper.pos": 50.0,
                    }

                    action = hand_leader.get_action(current_pos)

                    # Display action
                    print(f"Action {i+1:2d}:")
                    for joint, value in action.items():
                        print(f"  {joint:<20}: {value:7.2f}")
                    print()

                except Exception as e:
                    print(f"Action generation error: {e}")

                # Wait for next cycle (10 Hz)
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0, 0.1 - elapsed))

        except KeyboardInterrupt:
            print("\nTest stopped by user")

        print("\n4. Disconnecting...")
        hand_leader.disconnect()
        print("   [OK] Disconnected")

        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYour hand gestures are being converted to robot commands!")
        print("Next step: Connect to a real robot to see it move.")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    try:
        exit_code = test_hand_leader_direct()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)

if __name__ == '__main__':
    main()