#!/usr/bin/env python3
"""
Absolute Position Control for Robot Arm

Instead of incremental movements (+1, -1), this directly sets the
end effector to absolute coordinates (x, y, z).

Key differences from keyboard_teleop.py:
1. Direct position setting instead of increments
2. Hand tracking provides absolute coordinates
3. Smooth interpolation between positions
4. Safety bounds checking
"""

import numpy as np
from forward_kinematics import forward_kinematics
from inverse_kinematics import iterative_ik
from ..lerobot.src.lerobot.motors.feetech import FeetechMotorsBus, TorqueMode
import time
import sys
sys.path.append('..')  # Add parent directory to path

# ==================== CONFIGURATION ====================

# Robot workspace boundaries (in meters)
WORKSPACE_BOUNDS = {
    'x': (-0.3, 0.3),   # Left/Right limits
    'y': (-0.3, 0.3),   # Forward/Back limits
    'z': (0.05, 0.4),   # Up/Down limits (avoid hitting table)
}

# Smoothing parameters
SMOOTHING_FACTOR = 0.3  # 0 = no smoothing, 1 = ignore new positions
MAX_VELOCITY = 0.05      # Max movement per timestep (m/s)

# Safety parameters
MAX_STEP_CHANGE = 500   # Max servo step change per cycle
SAFETY_MARGIN = 0.02     # Keep this far from boundaries (meters)

# ==================== HELPER FUNCTIONS ====================

def servo_steps_to_angles(steps):
    """Convert servo steps to joint angles in degrees."""
    if len(steps) != 4:
        raise ValueError("Expected 4 steps for main joints.")

    calibration = [
        {"zero_step": 2047, "direction": -1},
        {"zero_step": 54,   "direction": 1},
        {"zero_step": 25,   "direction": 1},
        {"zero_step": 2095, "direction": 1},
    ]

    degrees_per_step = 360.0 / 4096.0
    angle_values = []

    for i, step in enumerate(steps):
        zero_step = calibration[i]["zero_step"]
        direction = calibration[i]["direction"]
        angle_value = (step - zero_step) * direction * degrees_per_step
        angle_values.append(angle_value % 360)

    return angle_values

def angles_to_servo_steps(angles):
    """Convert joint angles in degrees to servo steps."""
    if len(angles) != 4:
        raise ValueError("Expected 4 angles for main joints.")

    calibration = [
        {"zero_step": 2047, "direction": -1},
        {"zero_step": 54,   "direction": 1},
        {"zero_step": 25,   "direction": 1},
        {"zero_step": 2095, "direction": 1},
    ]

    steps_per_degree = 4096 / 360.0
    step_values = []

    for i, angle in enumerate(angles):
        zero_step = calibration[i]["zero_step"]
        direction = calibration[i]["direction"]
        step_value = int(zero_step + direction * angle * steps_per_degree)
        step_values.append(step_value % 4096)

    return step_values

def clamp_to_workspace(position):
    """Ensure position is within safe workspace bounds."""
    clamped = position.copy()

    for i, axis in enumerate(['x', 'y', 'z']):
        min_val, max_val = WORKSPACE_BOUNDS[axis]
        min_val += SAFETY_MARGIN
        max_val -= SAFETY_MARGIN
        clamped[i] = np.clip(position[i], min_val, max_val)

    if not np.array_equal(position, clamped):
        print(f"Warning: Position clamped from {position} to {clamped}")

    return clamped

def smooth_position(current, target, factor=SMOOTHING_FACTOR):
    """Apply exponential smoothing to position changes."""
    return current * factor + target * (1 - factor)

def limit_velocity(current_pos, target_pos, max_vel=MAX_VELOCITY, dt=0.1):
    """Limit the velocity of position changes."""
    delta = target_pos - current_pos
    distance = np.linalg.norm(delta)

    if distance > max_vel * dt:
        # Scale down movement to respect velocity limit
        delta = delta * (max_vel * dt / distance)
        limited_pos = current_pos + delta
        print(f"Velocity limited: {distance/dt:.3f} m/s -> {max_vel:.3f} m/s")
        return limited_pos

    return target_pos

# ==================== ABSOLUTE POSITION CONTROLLER ====================

class AbsolutePositionController:
    """
    Controller for setting absolute end effector positions.
    """

    def __init__(self, port="/dev/tty.usbmodem58CD1774031"):
        """Initialize the robot arm controller."""

        # Initialize motor bus
        self.follower_arm = FeetechMotorsBus(
            port=port,
            motors={
                "shoulder_pan": (6, "sts3215"),
                "shoulder_lift": (5, "sts3215"),
                "elbow_flex": (4, "sts3215"),
                "wrist_flex": (3, "sts3215"),
                "wrist_roll": (2, "sts3215"),
                "gripper": (1, "sts3215"),
            },
        )

        # Connect to robot
        self.follower_arm.connect()
        self.follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)
        time.sleep(1)

        # Get initial positions
        current_positions = self.follower_arm.read("Present_Position")
        self.positions = current_positions[0:4]
        self.wrist_roll_pos = current_positions[4]
        self.gripper_pos = current_positions[5]

        # Convert to angles and get end effector position
        self.angles = servo_steps_to_angles(self.positions)
        self.ef_position, self.ef_angles = forward_kinematics(*self.angles)

        # Initialize target position to current position
        self.target_position = self.ef_position.copy()

        print(f"Controller initialized")
        print(f"Initial servo positions: {self.positions}")
        print(f"Initial joint angles: {self.angles}")
        print(f"Initial EF position: {self.ef_position}")
        print(f"Initial EF orientation: {self.ef_angles}")

    def set_absolute_position(self, x, y, z, gripper_percent=None, wrist_roll_deg=None):
        """
        Set the end effector to an absolute position.

        Args:
            x, y, z: Target position in meters
            gripper_percent: Gripper opening (0=closed, 100=open)
            wrist_roll_deg: Wrist roll angle in degrees
        """

        # Create target position array
        target = np.array([x, y, z])

        # Safety checks
        target = clamp_to_workspace(target)

        # Apply smoothing
        smoothed_target = smooth_position(self.ef_position, target)

        # Apply velocity limiting
        limited_target = limit_velocity(self.ef_position, smoothed_target)

        # Store as new target
        self.target_position = limited_target

        print(f"Setting absolute position to: {self.target_position}")

        # Compute inverse kinematics
        try:
            updated_angles = iterative_ik(
                self.target_position,
                90,  # Default pitch angle
                self.angles,  # Use current angles as initial guess
                max_iter=100,  # Reduce iterations for speed
                alpha=0.7  # Slightly more aggressive convergence
            )

            print(f"IK solution angles: {updated_angles}")

            # Verify with forward kinematics
            final_pos, _ = forward_kinematics(*updated_angles)
            error = self.target_position - final_pos
            print(f"Position error: {error}, Norm: {np.linalg.norm(error):.4f}")

            # Convert to servo steps
            updated_steps = angles_to_servo_steps(updated_angles)

            # Safety check for large jumps
            current_positions = self.follower_arm.read("Present_Position")
            current_motors = current_positions[0:4]

            for i, (current, target) in enumerate(zip(current_motors, updated_steps)):
                if abs(target - current) > MAX_STEP_CHANGE:
                    print(f"WARNING: Large jump on joint {i}: {current} -> {target}")
                    print("Movement cancelled for safety")
                    return False

            # Add wrist roll and gripper
            if wrist_roll_deg is not None:
                # Convert angle to servo steps (simplified)
                self.wrist_roll_pos = int(2048 + wrist_roll_deg * 11.375)

            if gripper_percent is not None:
                # Convert percentage to servo steps (0-100% -> servo range)
                # Assuming gripper range is 1000-3000 steps
                self.gripper_pos = int(1000 + gripper_percent * 20)

            updated_steps.append(self.wrist_roll_pos)
            updated_steps.append(self.gripper_pos)

            # Send to robot
            self.follower_arm.write("Goal_Position", np.array(updated_steps))

            # Update internal state
            self.angles = updated_angles[:]
            self.ef_position = self.target_position.copy()

            return True

        except Exception as e:
            print(f"Error in IK computation: {e}")
            return False

    def get_current_position(self):
        """Get the current end effector position."""
        return self.ef_position.copy()

    def shutdown(self):
        """Safely shutdown the controller."""
        print("Shutting down controller...")
        self.follower_arm.write("Torque_Enable", TorqueMode.DISABLED.value)
        self.follower_arm.disconnect()

# ==================== HAND TRACKING INTEGRATION ====================

def hand_tracking_control(controller):
    """
    Control robot using hand tracking absolute positions.
    """
    import requests

    print("Starting hand tracking control...")
    print("Make sure hand_tracking_ipc_server.py is running!")

    try:
        while True:
            try:
                # Get hand position from IPC server
                response = requests.get('http://localhost:8888/position', timeout=0.5)
                if response.status_code == 200:
                    data = response.json()

                    if data['valid']:
                        # Hand detected - set absolute position
                        x = data['x']
                        y = data['y']
                        z = data['z']
                        gripper = data['pinch']  # 0-100%

                        # Set robot to absolute position
                        success = controller.set_absolute_position(
                            x, y, z,
                            gripper_percent=gripper
                        )

                        if success:
                            print(f"Hand position: ({x:.3f}, {y:.3f}, {z:.3f}), Grip: {gripper:.1f}%")
                    else:
                        print("No hand detected")

            except requests.exceptions.RequestException:
                print("Cannot connect to hand tracking server")

            time.sleep(0.1)  # 10 Hz update rate

    except KeyboardInterrupt:
        print("\nControl stopped by user")

# ==================== MAIN PROGRAM ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Absolute position control for robot arm")
    parser.add_argument('--port', default="/dev/tty.usbmodem58CD1774031",
                       help='Serial port for robot')
    parser.add_argument('--mode', choices=['test', 'hand'], default='test',
                       help='Control mode: test or hand tracking')

    args = parser.parse_args()

    # Initialize controller
    controller = AbsolutePositionController(port=args.port)

    try:
        if args.mode == 'test':
            # Test mode - set specific positions
            print("\nTest mode - Setting absolute positions")

            # Move to position 1
            print("\nMoving to position 1: (0.2, 0.1, 0.15)")
            controller.set_absolute_position(0.2, 0.1, 0.15, gripper_percent=50)
            time.sleep(2)

            # Move to position 2
            print("\nMoving to position 2: (0.15, -0.1, 0.2)")
            controller.set_absolute_position(0.15, -0.1, 0.2, gripper_percent=0)
            time.sleep(2)

            # Move to position 3
            print("\nMoving to position 3: (0.25, 0.0, 0.1)")
            controller.set_absolute_position(0.25, 0.0, 0.1, gripper_percent=100)
            time.sleep(2)

        else:
            # Hand tracking mode
            hand_tracking_control(controller)

    finally:
        controller.shutdown()