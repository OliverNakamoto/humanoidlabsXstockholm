#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import numpy as np
import requests
import os
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# Import IK functions from keyboard_teleop
import sys
keyboard_teleop_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'keyboard_teleop')
if os.path.exists(keyboard_teleop_path):
    sys.path.insert(0, keyboard_teleop_path)
    try:
        from forward_kinematics import forward_kinematics
        from inverse_kinematics import iterative_ik
        HAS_KEYBOARD_IK = True
        logging.info("Loaded keyboard_teleop IK functions")
    except ImportError as e:
        HAS_KEYBOARD_IK = False
        logging.warning(f"Could not load keyboard_teleop IK functions: {e}")
else:
    HAS_KEYBOARD_IK = False
    logging.warning("keyboard_teleop folder not found")

try:
    from lerobot.model.kinematics import RobotKinematics
    HAS_IK = True
except ImportError:
    HAS_IK = False
    logging.warning("RobotKinematics not available, using simple geometric mapping")

from ..teleoperator import Teleoperator
from .config_phone_gyro import PhoneGyroConfig

logger = logging.getLogger(__name__)


class PhoneGyro(Teleoperator):
    """
    Phone Gyroscope Teleoperator for robot control using phone's gyroscope and accelerometer.

    Controls end-effector position and orientation using phone movements:
    - Phone tilt (pitch/roll) -> End-effector position
    - Phone orientation -> End-effector orientation
    - Uses inverse kinematics for precise control
    """

    config_class = PhoneGyroConfig
    name = "phone_gyro"

    def __init__(self, config: PhoneGyroConfig):
        super().__init__(config)
        self.config = config

        # Initialize IK solver if available
        if HAS_IK:
            try:
                self.kinematics = RobotKinematics(
                    urdf_path=config.urdf_path,
                    target_frame_name=config.target_frame_name,
                    joint_names=config.joint_names
                )
                self.use_ik = True
                logger.info("Using inverse kinematics for phone gyro control")
            except Exception as e:
                logger.warning(f"IK initialization failed: {e}, using simple mapping")
                self.use_ik = False
        else:
            self.use_ik = False

        # HTTP session for requests
        self.session = requests.Session()

        # Current end-effector position for incremental IK (like keyboard_teleop)
        self.ef_position = np.array([0.2, 0.0, 0.15])  # Default start position
        self.ef_pitch = 90.0  # Default pitch angle
        self.current_angles = [0.0, 0.0, 0.0, 0.0]  # Current joint angles

        # Movement step size (same as keyboard_teleop)
        self.step_size = 0.005

        # Define joint limits (in normalized range -100 to 100)
        self.joint_limits = {
            'shoulder_pan': (-100, 100),
            'shoulder_lift': (-100, 100),
            'elbow_flex': (-100, 100),
            'wrist_flex': (-100, 100),
            'wrist_roll': (-100, 100),
        }

        # Workspace parameters for simple mapping
        self.workspace_center = {'x': 0.0, 'y': 0.2, 'z': 0.15}
        self.workspace_scale = {'x': 200, 'y': 150, 'z': 100}

        self._is_connected = False

    @property
    def action_features(self) -> dict[str, type]:
        """Define the action space for the robot."""
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """No feedback features for phone gyro control."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = False) -> None:
        """Connect to the phone gyroscope server."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting to phone gyroscope server...")

        try:
            # Test connection to server
            response = self.session.get(f"{self.config.server_url}/status", timeout=2.0)
            if response.status_code == 200:
                logger.info(f"Connected to phone gyro server at {self.config.server_url}")
                self._is_connected = True
            else:
                raise ConnectionError(f"Server returned status {response.status_code}")

        except Exception as e:
            raise ConnectionError(f"Failed to connect to phone gyro server: {e}")

        if calibrate:
            self.calibrate()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Phone gyroscope is always considered calibrated when connected."""
        return self.is_connected

    def configure(self) -> None:
        """Apply any one-time configuration to the phone gyroscope teleoperator."""
        logger.info("Configuring phone gyroscope teleoperator...")
        # No specific configuration needed for phone gyro

    def calibrate(self) -> None:
        """Calibrate the phone gyroscope (reset to center position)."""
        logger.info("Calibrating phone gyroscope...")
        try:
            response = self.session.get(f"{self.config.server_url}/calibrate", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info("Phone gyroscope calibrated")
                else:
                    logger.warning(f"Calibration failed: {data.get('message', 'Unknown error')}")
            else:
                logger.warning(f"Calibration request failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Calibration error: {e}")

    def get_phone_pose(self) -> Tuple[float, float, float, float, float, float, bool, float, float, float]:
        """
        Get current phone pose, acceleration, and joystick from server.

        Returns:
            Tuple of (x, y, z, roll, pitch, yaw, gripper_closed, accel_x, joystick_x, joystick_y) in robot coordinates
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            response = self.session.get(f"{self.config.server_url}/status", timeout=0.1)
            if response.status_code == 200:
                data = response.json()
                if data.get('valid'):
                    pos = data['position']
                    ori = data['orientation']

                    # Apply scaling
                    x = pos['x'] * self.config.position_scale
                    y = pos['y'] * self.config.position_scale
                    z = pos['z'] * self.config.position_scale

                    # Transform orientation for portrait phone mode (phone held upright)
                    # When phone is held upright (portrait), we need to offset the pitch by -90 degrees
                    # so that "flat" phone orientation corresponds to neutral robot orientation
                    roll = np.radians(ori['roll']) * self.config.orientation_scale
                    pitch = np.radians(ori['pitch'] + 90.0) * self.config.orientation_scale  # Add 90° offset for portrait mode
                    yaw = np.radians(ori['yaw']) * self.config.orientation_scale

                    # Get gripper state
                    gripper_closed = data.get('gripper_closed', False)

                    # Get acceleration data
                    accel = data.get('acceleration', {'x': 0.0})
                    accel_x = accel.get('x', 0.0)

                    # Get joystick data
                    joystick = data.get('joystick', {'x': 0.0, 'y': 0.0})
                    joystick_x = joystick.get('x', 0.0)
                    joystick_y = joystick.get('y', 0.0)

                    return (x, y, z, roll, pitch, yaw, gripper_closed, accel_x, joystick_x, joystick_y)

        except Exception as e:
            logger.debug(f"Error getting phone pose: {e}")

        # Return default pose if no data (gripper open by default, no acceleration, no joystick)
        return (0.0, 0.2, 0.15, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0)

    def compute_ik(self, current_pos: Dict[str, float]) -> Dict[str, float]:
        """
        Compute incremental IK control from joystick and phone data.

        Args:
            current_pos: Current joint positions (used for initialization)

        Returns:
            Dictionary of joint positions
        """
        # Get phone pose, gripper state, acceleration, and joystick
        x, y, z, roll, pitch, yaw, gripper_closed, accel_x, joystick_x, joystick_y = self.get_phone_pose()

        # Use incremental IK control (like keyboard_teleop)
        return self.incremental_ik_control(
            joystick_x=joystick_x,
            joystick_y=joystick_y,
            pitch=pitch,
            roll=roll,
            gripper_closed=gripper_closed
        )

    def incremental_ik_control(self, joystick_x: float = 0.0, joystick_y: float = 0.0,
                              pitch: float = 0.0, roll: float = 0.0, gripper_closed: bool = False) -> Dict[str, float]:
        """
        Incremental IK control exactly like keyboard_teleop.

        Joystick mapping:
        - X axis (left/right) -> Y movement in end-effector space (like 'a'/'d' keys)
        - Y axis (forward/back) -> X movement in end-effector space (like 'w'/'s' keys)
        - Phone pitch -> wrist_flex (when joystick not active)
        - Phone roll -> wrist_roll (always)

        Args:
            joystick_x, joystick_y: Normalized joystick position (-1 to 1)
            pitch, roll: Phone orientation in radians
            gripper_closed: Gripper state

        Returns:
            Dictionary of joint positions in normalized range
        """

        if not HAS_KEYBOARD_IK:
            logger.warning("Keyboard IK not available, using fallback")
            return self.fallback_control(joystick_x, joystick_y, pitch, roll, gripper_closed)

        # Movement deltas based on joystick (same logic as keyboard_teleop)
        movement = {
            'x': joystick_y * self.step_size,   # Forward/back from joystick Y
            'y': -joystick_x * self.step_size,  # Left/right from joystick X (negative to fix direction)
            'z': 0.0  # No Z control from joystick
        }

        # Debug logging
        if abs(joystick_x) > 0.01 or abs(joystick_y) > 0.01:
            logger.info(f"Joystick input: x={joystick_x:.3f}, y={joystick_y:.3f}")
            logger.info(f"Movement delta: x={movement['x']:.6f}, y={movement['y']:.6f}")
            logger.info(f"EF position: {self.ef_position}")

        # Update end-effector position incrementally (like keyboard_teleop)
        self.ef_position[0] += movement['x']
        self.ef_position[1] += movement['y']
        self.ef_position[2] += movement['z']

        # Clamp to workspace limits
        self.ef_position[0] = np.clip(self.ef_position[0], 0.05, 0.35)
        self.ef_position[1] = np.clip(self.ef_position[1], -0.25, 0.25)
        self.ef_position[2] = np.clip(self.ef_position[2], 0.05, 0.3)

        try:
            # Compute IK with current angles as seed (like keyboard_teleop)
            updated_angles = iterative_ik(
                self.ef_position,
                self.ef_pitch,
                self.current_angles,
                max_iter=100,
                alpha=0.5
            )

            # Check for sudden jumps (safety check from keyboard_teleop)
            max_angle_change = 30  # degrees
            angle_changes = np.abs(np.array(updated_angles) - np.array(self.current_angles))

            if np.all(angle_changes < max_angle_change):
                self.current_angles = list(updated_angles)
                logger.debug(f"IK solved: EF pos={self.ef_position}, angles={self.current_angles}")
            else:
                logger.warning("IK jump detected, keeping previous angles")

        except Exception as e:
            logger.warning(f"IK computation failed: {e}")

        # Convert angles to servo positions (-100 to 100 range)
        joint_positions = {}

        # Map 4 main joints from IK solution
        # Note: These mappings may need adjustment based on your specific robot calibration
        joint_positions["shoulder_pan.pos"] = float(np.clip(self.current_angles[0] * 100/180, -100, 100))
        joint_positions["shoulder_lift.pos"] = float(np.clip(self.current_angles[1] * 100/180, -100, 100))
        joint_positions["elbow_flex.pos"] = float(np.clip(self.current_angles[2] * 100/180, -100, 100))

        # Wrist flex: IK result when joystick active, phone pitch when inactive
        joystick_active = abs(joystick_x) > 0.05 or abs(joystick_y) > 0.05
        if joystick_active:
            joint_positions["wrist_flex.pos"] = float(np.clip(self.current_angles[3] * 100/180, -100, 100))
        else:
            # Use phone pitch for wrist control when joystick is centered
            # Invert pitch direction: forward tilt (positive pitch) -> negative wrist_flex (down)
            wrist_flex = np.clip(np.degrees(pitch) * 2, -100, 100)  # Removed negative sign to fix direction
            joint_positions["wrist_flex.pos"] = float(wrist_flex)

        # Wrist roll: Always controlled by phone roll (NOT linked to joystick as requested)
        wrist_roll = np.clip(np.degrees(roll) * 2, -100, 100)
        joint_positions["wrist_roll.pos"] = float(wrist_roll)

        # Gripper
        joint_positions["gripper.pos"] = 100.0 if gripper_closed else 0.0

        return joint_positions

    def fallback_control(self, joystick_x: float, joystick_y: float,
                        pitch: float, roll: float, gripper_closed: bool) -> Dict[str, float]:
        """Fallback control when IK not available"""
        joint_positions = {}

        # Simple direct mapping
        joint_positions["shoulder_pan.pos"] = float(np.clip(-joystick_x * 50, -100, 100))
        joint_positions["shoulder_lift.pos"] = float(np.clip(joystick_y * 50, -100, 100))
        joint_positions["elbow_flex.pos"] = float(np.clip(-joystick_y * 30, -100, 100))
        joint_positions["wrist_flex.pos"] = float(np.clip(np.degrees(pitch) * 2, -100, 100))  # Fixed direction
        joint_positions["wrist_roll.pos"] = float(np.clip(np.degrees(roll) * 2, -100, 100))
        joint_positions["gripper.pos"] = 100.0 if gripper_closed else 0.0

        return joint_positions

    def get_action(self, current_pos: Dict[str, float] = None) -> Dict[str, float]:
        """
        Get action by computing IK from phone pose.

        Args:
            current_pos: Current robot joint positions (used as IK seed)

        Returns:
            Dictionary of target joint positions
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Use provided current position or default
        if current_pos is None:
            current_pos = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
            }

        # Compute action from phone pose
        action = self.compute_ik(current_pos)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} computed action: {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Force feedback not implemented for phone gyro control."""
        raise NotImplementedError("Force feedback not available for phone gyro control")

    def disconnect(self) -> None:
        """Disconnect from the phone gyro server."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
        logger.info(f"{self} disconnected.")