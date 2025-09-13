#!/usr/bin/env python3
"""
Test pinch control with robot communication.

Tests the complete pipeline:
1. Hand tracking via IPC
2. Hand position -> robot commands
3. Pinch gesture -> gripper control

Usage:
    python test_hand_robot_pinch.py
"""

import sys
import time
import requests
import json
from typing import Dict, Any

# Add LeRobot to path
sys.path.insert(0, 'lerobot/src')

def test_ipc_connection():
    """Test if hand tracking server is responding."""
    print("Testing IPC connection...")

    try:
        # Test status endpoint
        response = requests.get('http://localhost:8888/status', timeout=2)
        if response.status_code == 200:
            status = response.json()
            print(f"[OK] Server status: {status}")
            return True
        else:
            print(f"[ERROR] Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] IPC connection failed: {e}")
        return False

def get_hand_position():
    """Get current hand position from IPC server."""
    try:
        response = requests.get('http://localhost:8888/position', timeout=0.5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def test_hand_leader_import():
    """Test if hand_leader teleoperator can be imported."""
    print("Testing hand_leader import...")

    try:
        from lerobot.teleoperators.hand_leader.hand_leader import HandLeader
        from lerobot.teleoperators.hand_leader.config_hand_leader import HandLeaderConfig
        print("[OK] HandLeader imported successfully")
        return True
    except Exception as e:
        print(f"[ERROR] HandLeader import failed: {e}")
        return False

def create_test_config():
    """Create a test configuration for hand_leader."""
    from lerobot.teleoperators.hand_leader.config_hand_leader import HandLeaderConfig

    config = HandLeaderConfig(
        camera_index=0,
        urdf_path="dummy_robot.urdf",  # We'll use dummy for testing
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    )
    return config

def test_hand_position_streaming():
    """Test real-time hand position streaming."""
    print("\nTesting hand position streaming...")
    print("Move your hand in front of the camera!")
    print("Open/close your hand to test pinch detection")
    print("Press Ctrl+C to stop\n")

    try:
        for i in range(100):  # 10 seconds at 10Hz
            data = get_hand_position()

            if data and data.get('valid'):
                x, y, z = data['x'], data['y'], data['z']
                pinch = data['pinch']

                # Format output
                pos_str = f"Position: ({x:6.3f}, {y:6.3f}, {z:6.3f})"
                pinch_str = f"Pinch: {pinch:5.1f}%"

                # Visual indicators
                pinch_bar = "#" * int(pinch/10) + "." * (10 - int(pinch/10))

                print(f"\r{pos_str} | {pinch_str} [{pinch_bar}]", end='', flush=True)
            else:
                print(f"\rNo hand detected - wave your hand in front of camera!", end='', flush=True)

            time.sleep(0.1)  # 10Hz

    except KeyboardInterrupt:
        print("\n\nStreaming stopped by user")

    print("\n")

def test_robot_command_generation():
    """Test generating robot commands from hand positions."""
    print("Testing robot command generation...")

    try:
        from lerobot.teleoperators.hand_leader.hand_leader import HandLeader
        config = create_test_config()

        # Create hand leader (will fail on robot connection, but we can test IPC)
        try:
            hand_leader = HandLeader(config)
            print("[OK] HandLeader created")

            # Test IPC connection
            hand_leader.connect(calibrate=False)
            print("[OK] HandLeader IPC connected")

            # Get a few commands
            print("Getting robot commands from hand positions...")
            for i in range(5):
                try:
                    # Get action (this calls IPC internally)
                    action = hand_leader.get_action()
                    print(f"Command {i+1}: {action}")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Command generation error: {e}")

            hand_leader.disconnect()
            print("[OK] HandLeader disconnected")

        except Exception as e:
            print(f"HandLeader test error (expected if no robot): {e}")

    except Exception as e:
        print(f"[ERROR] Robot command generation failed: {e}")

def main():
    print("=" * 60)
    print("HAND-ROBOT PINCH CONTROL TEST")
    print("=" * 60)

    # Test 1: IPC Connection
    if not test_ipc_connection():
        print("\n❌ Hand tracking server not running!")
        print("Please start it first with:")
        print("  python hand_tracking_ipc_server.py --port 8888")
        return 1

    # Test 2: LeRobot Import
    if not test_hand_leader_import():
        print("\n❌ LeRobot hand_leader import failed!")
        return 1

    # Test 3: Hand Position Streaming
    test_hand_position_streaming()

    # Test 4: Robot Command Generation
    test_robot_command_generation()

    print("=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\nTo run with actual robot:")
    print("1. Connect your robot via USB")
    print("2. Update config with correct URDF path")
    print("3. Run: lerobot-teleoperate --teleoperator hand_leader")

    return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)