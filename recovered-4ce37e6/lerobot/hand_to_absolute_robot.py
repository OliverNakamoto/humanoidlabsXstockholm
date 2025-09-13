#!/usr/bin/env python3
"""
Hand Tracking to Absolute Robot Position Control

This integrates hand tracking with absolute position control.
Your hand position directly sets the robot's end effector position.

Key features:
- Direct 1:1 mapping from hand to robot workspace
- No incremental movements - hand position = robot position
- Smooth interpolation for natural movement
- Safety bounds to prevent collisions
"""

import numpy as np
import time
import requests
import sys
import os

# Add keyboard_teleop directory to path for IK imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'keyboard_teleop'))

# ==================== CONFIGURATION ====================

# Hand tracking server
HAND_TRACKING_URL = "http://localhost:8888"

# Mapping from hand space to robot space
HAND_TO_ROBOT_SCALE = {
    'x': 1.0,   # Direct mapping
    'y': 1.0,   # Direct mapping
    'z': 1.0,   # Direct mapping
}

HAND_TO_ROBOT_OFFSET = {
    'x': 0.0,   # No offset
    'y': 0.0,   # No offset
    'z': 0.0,   # No offset
}

# Robot workspace limits (safety)
ROBOT_WORKSPACE = {
    'x': (-0.25, 0.25),  # meters
    'y': (0.05, 0.35),   # meters
    'z': (0.05, 0.35),   # meters
}

# Control parameters
UPDATE_RATE = 30  # Hz
SMOOTHING = 0.7   # 0=no smoothing, 1=full smoothing
DEADZONE = 0.005  # Ignore movements smaller than this (meters)

# ==================== HAND TRACKING CLIENT ====================

class HandTrackingClient:
    """Client for getting hand position from IPC server."""

    def __init__(self, server_url=HAND_TRACKING_URL):
        self.server_url = server_url
        self.last_valid_position = None

    def get_hand_position(self):
        """
        Get current hand position from tracking server.

        Returns:
            dict: {'x': float, 'y': float, 'z': float, 'pinch': float, 'valid': bool}
            None: If connection failed
        """
        try:
            response = requests.get(
                f"{self.server_url}/position",
                timeout=0.1  # Fast timeout for responsiveness
            )

            if response.status_code == 200:
                data = response.json()

                if data.get('valid', False):
                    self.last_valid_position = data
                    return data

                # Return last valid position if current is invalid
                return self.last_valid_position

        except requests.exceptions.RequestException:
            return None

    def is_server_running(self):
        """Check if hand tracking server is running."""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=1)
            return response.status_code == 200
        except:
            return False

# ==================== ABSOLUTE POSITION ROBOT CONTROLLER ====================

class AbsoluteRobotController:
    """
    Controller that sets robot to absolute positions based on hand tracking.
    """

    def __init__(self, use_real_robot=False, robot_port=None):
        """
        Initialize the controller.

        Args:
            use_real_robot: If True, connect to physical robot
            robot_port: Serial port for robot connection
        """
        self.use_real_robot = use_real_robot
        self.current_position = np.array([0.15, 0.15, 0.15])  # Default center position
        self.target_position = self.current_position.copy()
        self.current_gripper = 50.0  # Default 50% open

        if use_real_robot:
            # Import robot control modules
            from keyboard_teleop.forward_kinematics import forward_kinematics
            from keyboard_teleop.inverse_kinematics import iterative_ik
            from keyboard_teleop.absolute_position_control import AbsolutePositionController

            self.robot = AbsolutePositionController(port=robot_port)
            self.forward_kinematics = forward_kinematics
            self.iterative_ik = iterative_ik

            print("Connected to real robot")
        else:
            self.robot = None
            print("Running in simulation mode (no real robot)")

    def set_absolute_position(self, x, y, z, gripper_percent):
        """
        Set robot to absolute position.

        Args:
            x, y, z: Target position in meters
            gripper_percent: Gripper opening (0-100%)
        """

        # Create target array
        target = np.array([x, y, z])

        # Apply workspace limits
        for i, axis in enumerate(['x', 'y', 'z']):
            target[i] = np.clip(target[i], *ROBOT_WORKSPACE[axis])

        # Check if movement is significant (deadzone)
        delta = np.linalg.norm(target - self.current_position)
        if delta < DEADZONE and abs(gripper_percent - self.current_gripper) < 5:
            return  # Skip insignificant movements

        # Apply smoothing
        smoothed_target = (
            self.current_position * SMOOTHING +
            target * (1 - SMOOTHING)
        )

        # Update state
        self.target_position = smoothed_target
        self.current_position = smoothed_target
        self.current_gripper = gripper_percent

        # Send to robot (real or simulated)
        if self.use_real_robot and self.robot:
            success = self.robot.set_absolute_position(
                smoothed_target[0],
                smoothed_target[1],
                smoothed_target[2],
                gripper_percent=gripper_percent
            )
            if not success:
                print("Warning: Robot command failed")
        else:
            # Simulation output
            print(f"Robot position: [{smoothed_target[0]:.3f}, {smoothed_target[1]:.3f}, {smoothed_target[2]:.3f}] "
                  f"Gripper: {gripper_percent:.1f}%")

    def shutdown(self):
        """Safely shutdown the controller."""
        if self.robot:
            self.robot.shutdown()

# ==================== MAIN CONTROL LOOP ====================

def main():
    """Main control loop for hand-to-robot absolute position control."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Hand tracking to absolute robot position control"
    )
    parser.add_argument(
        '--robot',
        action='store_true',
        help='Connect to real robot (default: simulation)'
    )
    parser.add_argument(
        '--port',
        default="/dev/tty.usbmodem58CD1774031",
        help='Serial port for robot'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Run hand tracking calibration'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HAND TRACKING TO ABSOLUTE ROBOT CONTROL")
    print("=" * 60)

    # Initialize hand tracking client
    hand_client = HandTrackingClient()

    # Check if server is running
    if not hand_client.is_server_running():
        print("ERROR: Hand tracking server not running!")
        print("Please start it with: python hand_tracking_ipc_server.py")
        return 1

    print("✓ Hand tracking server connected")

    # Run calibration if requested
    if args.calibrate:
        print("\nStarting calibration...")
        try:
            response = requests.get(f"{HAND_TRACKING_URL}/calibrate")
            print("Calibration completed")
        except:
            print("Calibration failed")

    # Initialize robot controller
    robot_controller = AbsoluteRobotController(
        use_real_robot=args.robot,
        robot_port=args.port
    )

    print("\nControl mapping:")
    print("- Hand position → Robot end effector position")
    print("- Hand pinch → Robot gripper")
    print("- Direct 1:1 absolute position mapping")
    print("\nPress Ctrl+C to stop\n")

    # Main control loop
    try:
        loop_count = 0
        last_update_time = time.time()

        while True:
            loop_start = time.time()

            # Get hand position
            hand_data = hand_client.get_hand_position()

            if hand_data and hand_data.get('valid'):
                # Map hand coordinates to robot coordinates
                robot_x = (hand_data['x'] * HAND_TO_ROBOT_SCALE['x'] +
                          HAND_TO_ROBOT_OFFSET['x'])
                robot_y = (hand_data['y'] * HAND_TO_ROBOT_SCALE['y'] +
                          HAND_TO_ROBOT_OFFSET['y'])
                robot_z = (hand_data['z'] * HAND_TO_ROBOT_SCALE['z'] +
                          HAND_TO_ROBOT_OFFSET['z'])

                gripper = hand_data['pinch']  # Already 0-100%

                # Set robot to absolute position
                robot_controller.set_absolute_position(
                    robot_x, robot_y, robot_z, gripper
                )

                # Display status every second
                if time.time() - last_update_time > 1.0:
                    print(f"Hand: ({hand_data['x']:.3f}, {hand_data['y']:.3f}, {hand_data['z']:.3f}) "
                          f"→ Robot: ({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f}) "
                          f"Grip: {gripper:.1f}%")
                    last_update_time = time.time()

            else:
                if loop_count % 30 == 0:  # Every second at 30Hz
                    print("Waiting for hand detection...")

            loop_count += 1

            # Maintain update rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / UPDATE_RATE) - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nControl stopped by user")

    finally:
        robot_controller.shutdown()
        print("Shutdown complete")

    return 0

# ==================== TESTING FUNCTIONS ====================

def test_mapping():
    """Test the coordinate mapping without robot."""

    print("Testing coordinate mapping...")
    print("-" * 40)

    test_positions = [
        {'x': 0.0, 'y': 0.3, 'z': 0.25, 'pinch': 50},   # Center
        {'x': -0.15, 'y': 0.2, 'z': 0.2, 'pinch': 0},   # Left
        {'x': 0.15, 'y': 0.4, 'z': 0.3, 'pinch': 100},  # Right
    ]

    for i, hand_pos in enumerate(test_positions, 1):
        # Apply mapping
        robot_x = hand_pos['x'] * HAND_TO_ROBOT_SCALE['x'] + HAND_TO_ROBOT_OFFSET['x']
        robot_y = hand_pos['y'] * HAND_TO_ROBOT_SCALE['y'] + HAND_TO_ROBOT_OFFSET['y']
        robot_z = hand_pos['z'] * HAND_TO_ROBOT_SCALE['z'] + HAND_TO_ROBOT_OFFSET['z']

        # Apply limits
        robot_x = np.clip(robot_x, *ROBOT_WORKSPACE['x'])
        robot_y = np.clip(robot_y, *ROBOT_WORKSPACE['y'])
        robot_z = np.clip(robot_z, *ROBOT_WORKSPACE['z'])

        print(f"Test {i}:")
        print(f"  Hand: ({hand_pos['x']:.3f}, {hand_pos['y']:.3f}, {hand_pos['z']:.3f})")
        print(f"  Robot: ({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f})")
        print(f"  Gripper: {hand_pos['pinch']:.1f}%")
        print()

if __name__ == "__main__":
    # Run test if --test flag is provided
    if '--test' in sys.argv:
        test_mapping()
    else:
        sys.exit(main())