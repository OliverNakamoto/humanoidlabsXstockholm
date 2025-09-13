#!/usr/bin/env python

import sys
import time
from pathlib import Path

# Add lerobot to path
sys.path.append("lerobot/src")

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

def test_robot_connection(port="COM6"):
    """Test SO101 robot connection on specified port"""
    print(f"Testing SO101 robot connection on {port}...")

    try:
        # Create robot config for SO101
        config = SO101FollowerConfig(
            port=port,
            id="follower"
        )

        print(f"Config: {config}")

        # Try to connect to robot
        robot = make_robot_from_config(config)
        print("Robot object created successfully")

        # Connect
        robot.connect()
        print("Robot connected successfully!")

        # Get current position
        current_pos = robot.read()
        print(f"Current robot position: {current_pos}")

        # Disconnect
        robot.disconnect()
        print("Robot disconnected")

        return True

    except Exception as e:
        print(f"Robot connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    ports_to_try = ["COM3", "COM4", "COM6"]

    for port in ports_to_try:
        print(f"\n{'='*50}")
        success = test_robot_connection(port)
        if success:
            print(f"✅ SUCCESS: Robot found on {port}")
            break
        else:
            print(f"❌ FAILED: Robot not found on {port}")
    else:
        print("\n❌ Robot not found on any available port")
        sys.exit(1)