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
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

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

    def get_phone_pose(self) -> Tuple[float, float, float, float, float, float, bool]:
        """
        Get current phone pose from server.

        Returns:
            Tuple of (x, y, z, roll, pitch, yaw, gripper_closed) in robot coordinates
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

                    roll = np.radians(ori['roll']) * self.config.orientation_scale
                    pitch = np.radians(ori['pitch']) * self.config.orientation_scale
                    yaw = np.radians(ori['yaw']) * self.config.orientation_scale

                    # Get gripper state
                    gripper_closed = data.get('gripper_closed', False)

                    return (x, y, z, roll, pitch, yaw, gripper_closed)

        except Exception as e:
            logger.debug(f"Error getting phone pose: {e}")

        # Return default pose if no data (gripper open by default)
        return (0.0, 0.2, 0.15, 0.0, 0.0, 0.0, False)

    def compute_ik(self, current_pos: Dict[str, float]) -> Dict[str, float]:
        """
        Compute inverse kinematics from phone pose to joint angles.

        Args:
            current_pos: Current joint positions (used as IK seed)

        Returns:
            Dictionary of joint positions
        """
        # Get phone pose and gripper state
        x, y, z, roll, pitch, yaw, gripper_closed = self.get_phone_pose()

        if self.use_ik:
            # Create target transformation matrix for IK
            target_pose = np.eye(4)

            # Set position
            target_pose[0, 3] = x
            target_pose[1, 3] = y
            target_pose[2, 3] = z

            # Set orientation using rotation matrix
            rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])
            target_pose[:3, :3] = rotation.as_matrix()

            # Extract current joint values for IK seed
            current_joints = []
            for joint_name in self.config.joint_names:
                key = f"{joint_name}.pos"
                if key in current_pos:
                    # Convert from normalized range to degrees if needed
                    if self.config.use_degrees:
                        current_joints.append(current_pos[key])
                    else:
                        # Convert from [-100, 100] to degrees
                        min_deg, max_deg = (-90, 90)  # Default limits
                        norm_val = current_pos[key]
                        deg_val = min_deg + (norm_val + 100) * (max_deg - min_deg) / 200
                        current_joints.append(deg_val)
                else:
                    current_joints.append(0.0)  # Default position

            current_joints = np.array(current_joints)

            # Compute IK
            try:
                joint_solution = self.kinematics.inverse_kinematics(
                    current_joints,
                    target_pose,
                    position_weight=1.0,
                    orientation_weight=0.5
                )

                # Convert joint angles back to normalized range
                actions = {}
                for i, joint_name in enumerate(self.config.joint_names):
                    if i < len(joint_solution):
                        if self.config.use_degrees:
                            actions[f"{joint_name}.pos"] = float(joint_solution[i])
                        else:
                            # Convert from degrees to [-100, 100] range
                            deg_val = joint_solution[i]
                            min_deg, max_deg = (-90, 90)  # Default limits
                            norm_val = 200 * (deg_val - min_deg) / (max_deg - min_deg) - 100
                            norm_val = np.clip(norm_val, -100, 100)
                            actions[f"{joint_name}.pos"] = float(norm_val)

                # Gripper control from phone button
                actions["gripper.pos"] = 100.0 if gripper_closed else 0.0

                return actions

            except Exception as e:
                logger.warning(f"IK failed: {e}, using simple mapping")
                return self.simple_position_mapping(x, y, z, roll, pitch, yaw, gripper_closed)
        else:
            return self.simple_position_mapping(x, y, z, roll, pitch, yaw, gripper_closed)

    def simple_position_mapping(self, x: float, y: float, z: float,
                               roll: float, pitch: float, yaw: float, gripper_closed: bool = False) -> Dict[str, float]:
        """
        Joystick-style position mapping using simplified inverse kinematics.

        Joystick Mode:
        - Phone X position (left/right tilt) -> End-effector X position
        - Phone Y position (forward/back tilt) -> End-effector Y position
        - Phone Z position (twist) -> End-effector Z position
        - Wrist orientation remains neutral for stability

        Args:
            x, y, z: Target position in robot workspace (meters)
            roll, pitch, yaw: Phone orientation (used for wrist if needed)
            gripper_closed: Gripper state

        Returns:
            Dictionary of joint positions in normalized range
        """
        # Clamp positions to workspace limits
        x_clamped = np.clip(x, self.workspace_limits['x'][0], self.workspace_limits['x'][1])
        y_clamped = np.clip(y, self.workspace_limits['y'][0], self.workspace_limits['y'][1])
        z_clamped = np.clip(z, self.workspace_limits['z'][0], self.workspace_limits['z'][1])

        # Simple geometric IK approximation for SO101 arm
        joint_positions = {}

        # Calculate joint angles using basic geometric relationships
        # This is a simplified IK - real IK would be more complex

        # Shoulder pan (base rotation) - maps to X position
        shoulder_pan_rad = np.arctan2(x_clamped, y_clamped)
        shoulder_pan_deg = np.degrees(shoulder_pan_rad)
        shoulder_pan_norm = np.clip(shoulder_pan_deg * 1.5, -100, 100)
        joint_positions["shoulder_pan.pos"] = float(shoulder_pan_norm)

        # Distance from base
        r = np.sqrt(x_clamped**2 + y_clamped**2)

        # Shoulder lift - affects Y position and height
        target_reach = np.sqrt(r**2 + (z_clamped - 0.1)**2)  # Offset for base height
        shoulder_lift_rad = np.arctan2(z_clamped - 0.1, r) + np.arccos(np.clip(target_reach / 0.6, 0, 1))
        shoulder_lift_deg = np.degrees(shoulder_lift_rad) - 90  # Offset for neutral position
        shoulder_lift_norm = np.clip(shoulder_lift_deg * 2, -100, 100)
        joint_positions["shoulder_lift.pos"] = float(shoulder_lift_norm)

        # Elbow flex - compensates for shoulder to maintain end-effector position
        elbow_angle_rad = np.pi - 2 * np.arccos(np.clip(target_reach / 0.6, 0, 1))
        elbow_angle_deg = np.degrees(elbow_angle_rad)
        elbow_flex_norm = np.clip((elbow_angle_deg - 120) * 1.5, -100, 100)
        joint_positions["elbow_flex.pos"] = float(elbow_flex_norm)

        # Wrist flex - keep relatively neutral for stability (slight downward angle)
        wrist_flex_norm = -20.0  # Slight downward angle
        joint_positions["wrist_flex.pos"] = float(wrist_flex_norm)

        # Wrist roll - keep neutral for joystick mode
        wrist_roll_norm = 0.0
        joint_positions["wrist_roll.pos"] = float(wrist_roll_norm)

        # Gripper: controlled by phone button
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