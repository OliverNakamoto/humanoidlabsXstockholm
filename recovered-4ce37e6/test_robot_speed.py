#!/usr/bin/env python3
"""
Test script to check robot movement speed
"""

import time
import numpy as np
from lerobot.robots.so101.so101_follower import SO101FollowerConfig, SO101Follower

def test_robot_speed():
    print("Testing robot movement speed...")

    # Initialize robot
    config = SO101FollowerConfig(port="COM6", robot_id="follower")
    robot = SO101Follower(config)

    try:
        robot.connect()
        print("Robot connected")

        # Get current position
        current_obs = robot.get_observation()
        current_action = {key: current_obs[key] for key in robot.action_features}

        print(f"Starting position: {current_action}")

        # Test small movement
        target_action = current_action.copy()
        target_action["shoulder_pan.pos"] += 200  # Small movement

        print(f"Moving to: {target_action}")

        start_time = time.time()
        robot.send_action(target_action)

        # Wait for movement to complete
        time.sleep(1)

        # Check new position
        new_obs = robot.get_observation()
        new_action = {key: new_obs[key] for key in robot.action_features}

        end_time = time.time()
        movement_time = end_time - start_time

        print(f"Final position: {new_action}")
        print(f"Movement took: {movement_time:.2f} seconds")

        # Calculate movement speed
        position_change = abs(new_action["shoulder_pan.pos"] - current_action["shoulder_pan.pos"])
        speed = position_change / movement_time
        print(f"Movement speed: {speed:.1f} steps/second")

    finally:
        robot.disconnect()

if __name__ == "__main__":
    test_robot_speed()