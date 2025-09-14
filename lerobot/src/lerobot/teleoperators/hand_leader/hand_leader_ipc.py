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

"""
Hand Leader using IPC Communication

Hand tracking teleoperator that uses Unix domain sockets to communicate
with a separate MediaPipe process, avoiding dependency conflicts.
"""

import logging
import time
import numpy as np
from typing import Dict, Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics

from ..teleoperator import Teleoperator
from .config_hand_leader import HandLeaderIPCConfig
from .ipc_client import HandTrackingClientManager
from .ipc_protocol import SOCKET_PATH
from .process_manager import HandTrackingProcessManager

logger = logging.getLogger(__name__)


class HandLeaderIPC(Teleoperator):
    """
    Hand Leader using IPC communication with separate MediaPipe process.
    Provides reliable hand tracking control without dependency conflicts.
    """

    config_class = HandLeaderIPCConfig
    name = "hand_leader_ipc"

    def __init__(self, config: HandLeaderIPCConfig):
        super().__init__(config)
        self.config = config
        
        # IPC client for receiving hand tracking data
        self.client_manager = HandTrackingClientManager(
            socket_path=getattr(config, 'socket_path', SOCKET_PATH),
            auto_reconnect=True
        )
        
        # Process manager for MediaPipe tracking
        self.process_manager = HandTrackingProcessManager(
            camera_index=getattr(config, 'camera_index', 0),
            socket_path=getattr(config, 'socket_path', SOCKET_PATH),
            verbose=logger.level <= logging.DEBUG,
            show_window=getattr(config, 'show_window', False)
        )
        
        # Initialize kinematics solver if URDF is provided
        self.kinematics = None
        logger.info("config is")
        logger.info(config)
        if hasattr(config, 'urdf_path') and config.urdf_path:
            try:
                self.kinematics = RobotKinematics(
                    urdf_path=config.urdf_path,
                    target_frame_name=getattr(config, 'target_frame_name', 'gripper_frame_link'),
                    joint_names=getattr(config, 'joint_names', [
                        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
                    ])
                )
                logger.info("Kinematics solver initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize kinematics solver: {e}")
                self.kinematics = None
        # logger.info(1/0)
        
        # Define joint limits (in degrees)
        self.joint_limits = {
            'shoulder_pan': (-90, 90),
            'shoulder_lift': (-90, 90),
            'elbow_flex': (-90, 90),
            'wrist_flex': (-90, 90),
            'wrist_roll': (-180, 180), # this seems suss, shouldn't be more than 180 no?
        }

        self.workspace_bounds = {
            'x_min': 0.10, 'x_max': 0.30,  # 5cm to 30cm forward
            'y_min': -0.10, 'y_max': 0.10, # ±10cm left/right
            'z_min': 0.10, 'z_max': 0.40   # 10cm to 40cm height
        }

        # Test actual robot reach using URDF (if kinematics available)
        if self.kinematics is not None:
            logger.info("🔍 Testing actual robot reach using URDF...")
            actual_bounds = self._test_robot_reach()
            logger.info(f"📊 URDF-based workspace bounds: {actual_bounds}")
            logger.info(f"📊 Current workspace bounds:    {self.workspace_bounds}")

            # Compare and warn about mismatches
            self._compare_workspace_bounds(actual_bounds)

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
        """No feedback features for hand tracking."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect by starting the tracking process and IPC client."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting hand leader (IPC mode)...")
        
        try:
            # Start the MediaPipe tracking process
            if not self.process_manager.start():
                raise RuntimeError("Failed to start tracking process")
            
            # Wait a bit for the process to initialize
            time.sleep(2.0)
            
            # Start the IPC client
            if not self.client_manager.start():
                raise RuntimeError("Failed to start IPC client")
            
            # Wait for connection to establish
            max_wait = 10.0  # 10 seconds
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait:
                client = self.client_manager.get_client()
                if client and client.is_connected():
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("Timeout waiting for IPC connection")
            
            # Wait for calibration to complete
            logger.info("Waiting for hand tracking calibration to complete...")
            client = self.client_manager.get_client()
            if not client.wait_for_calibration(timeout=30.0):
                raise RuntimeError("Timeout waiting for hand tracking calibration")
            
            self._is_connected = True
            logger.info(f"{self} connected successfully - calibration complete")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._cleanup_on_failure()
            raise


    def _cleanup_on_failure(self):
        """Clean up resources after connection failure."""
        if self.client_manager:
            self.client_manager.stop()
        
        if self.process_manager:
            self.process_manager.stop()

    @property
    def is_calibrated(self) -> bool:
        """Check if the hand tracking is calibrated."""
        client = self.client_manager.get_client()
        return client and client.is_calibrated() if client else False

    def calibrate(self) -> None:
        """Request calibration from the tracking process."""
        client = self.client_manager.get_client()
        if client:
            if client.request_calibration():
                logger.info("Calibration request sent to tracking process")
            else:
                logger.error("Failed to send calibration request")
        else:
            logger.error("No active client to request calibration")

    def configure(self) -> None:
        """No configuration needed for IPC-based tracking."""
        pass

    def setup_motors(self) -> None:
        """No motor setup needed for IPC-based tracking."""
        pass

    def get_endpos(self):
        """
        Get current end effector position, orientation, and second hand curl from IPC client.

        Returns:
            Tuple of (x, y, z, pinch, qx, qy, qz, qw, second_hand_curl) where:
            - x, y, z are in meters (robot workspace coordinates)
            - pinch is gripper percentage (0-100)
            - qx, qy, qz, qw are quaternion components for orientation
            - second_hand_curl is the second hand curl percentage (0-100)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        client = self.client_manager.get_client()
        if not client:
            logger.warning("No active IPC client")
            return (0.175, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # Identity quaternion + uncurled hand
        
        return client.get_current_position_orientation_and_pitch()

    def compute_ik(self, current_pos: Dict[str, float]) -> Dict[str, float]:
        """
        Compute inverse kinematics from end effector position to joint angles.
        
        Args:
            current_pos: Current joint positions (used as IK seed)
            
        Returns:
            Dictionary of joint positions including gripper
        """
        # Get end effector target from IPC including second hand curl
        x, y, z, pinch, qx, qy, qz, qw, second_hand_curl = self.get_endpos()

        # DEBUG: Log IK input coordinates
        logger.info(f"🎯 IK INPUT: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")
        logger.info(f"📐 Workspace bounds: {self.workspace_bounds}")
        center_x = (self.workspace_bounds['x_min'] + self.workspace_bounds['x_max']) / 2
        center_y = (self.workspace_bounds['y_min'] + self.workspace_bounds['y_max']) / 2
        center_z = (self.workspace_bounds['z_min'] + self.workspace_bounds['z_max']) / 2
        logger.info(f"🎯 Robot CENTER should be: x={center_x:.3f}m, y={center_y:.3f}m, z={center_z:.3f}m")
        logger.info(f"🔧 Using kinematics solver: {self.kinematics is not None}")
        
        # If no kinematics solver available, use simple mapping
        if self.kinematics is None:
            actions = self._simple_position_mapping(x, y, z, pinch, qx, qy, qz, qw)
            # Override joint 4 (wrist_flex) with second hand curl
            actions["wrist_flex.pos"] = self._convert_curl_to_joint_value(second_hand_curl)
            return actions
        
        # Create target transformation matrix for IK
        target_pose = np.eye(4)
        target_pose[0, 3] = x  # X position
        target_pose[1, 3] = y  # Y position  
        target_pose[2, 3] = z  # Z position
        
        # Add orientation from quaternion
        target_pose[:3, :3] = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
        
        # Extract current joint values for IK seed
        current_joints = []
        joint_names = getattr(self.config, 'joint_names', [
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
        ])
        
        for joint_name in joint_names:
            key = f"{joint_name}.pos"
            if key in current_pos:
                # Convert from normalized (-100 to 100) to degrees
                norm_val = current_pos[key]
                min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
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
                orientation_weight=0.01  # Low weight on orientation
            )
            
            # Convert joint angles back to normalized range
            actions = {}
            for i, joint_name in enumerate(joint_names):
                if i < len(joint_solution):
                    deg_val = joint_solution[i]
                    min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                    # Convert from degrees to [-100, 100] range
                    norm_val = 200 * (deg_val - min_deg) / (max_deg - min_deg) - 100
                    norm_val = np.clip(norm_val, -100, 100)
                    actions[f"{joint_name}.pos"] = float(norm_val)
            
            # Add gripper position (inverted: 100 - pinch)
            actions["gripper.pos"] = float(100.0 - pinch)

            # Override joint 4 (wrist_flex) with second hand curl
            # actions["wrist_flex.pos"] = self._convert_curl_to_joint_value(second_hand_curl)

            # DEBUG: Log IK solver output
            logger.info(f"📤 IK solver OUTPUT: {actions}")
            return actions
            
        except Exception as e:
            logger.warning(f"IK failed: {e}, using simple mapping")
            actions = self._simple_position_mapping(x, y, z, pinch, qx, qy, qz, qw)
            # Override joint 4 (wrist_flex) with second hand curl
            actions["wrist_flex.pos"] = self._convert_curl_to_joint_value(second_hand_curl)
            return actions
        
    def _simple_position_mapping(
        self,
        x: float,
        y: float,
        z: float,
        pinch: float,
        qx: float | None = None,
        qy: float | None = None,
        qz: float | None = None,
        qw: float | None = None,
    ) -> Dict[str, float]:
        """Simple position mapping when IK is not available.

        If orientation (quaternion) is provided, map yaw to `wrist_roll`.
        """
        # DEBUG: Log simple mapping input
        logger.info(f"🤖 Simple mapping INPUT: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")

        # Base mapping from position only
        shoulder_pan = x * 300  # X controls pan
        shoulder_lift = -y * 200 + 50  # Y controls lift
        elbow_flex = z * 150 - 30  # Z affects elbow
        wrist_flex = y * 100  # Y also affects wrist

        # DEBUG: Log joint calculations before clipping
        logger.info(f"🔧 Raw joint values: pan={shoulder_pan:.1f}, lift={shoulder_lift:.1f}, elbow={elbow_flex:.1f}, wrist_flex={wrist_flex:.1f}")
        
        # Default wrist roll from X if no orientation
        wrist_roll = x * 180
        
        # If quaternion provided, use its yaw for wrist_roll
        if None not in (qx, qy, qz, qw):
            try:
                roll, pitch, yaw = self._quaternion_to_euler(qx, qy, qz, qw)
                # Map yaw (radians) to [-100, 100] via degrees/180*100
                wrist_roll = np.degrees(yaw) / 180.0 * 100.0
            except Exception as e:
                logger.debug(f"Quaternion->euler conversion failed, fallback to position roll: {e}")
        
        actions = {
            "shoulder_pan.pos": float(np.clip(shoulder_pan, -100, 100)),
            "shoulder_lift.pos": float(np.clip(shoulder_lift, -100, 100)),
            "elbow_flex.pos": float(np.clip(elbow_flex, -100, 100)),
            "wrist_flex.pos": float(np.clip(wrist_flex, -100, 100)),
            "wrist_roll.pos": float(np.clip(wrist_roll, -100, 100)),
            "gripper.pos": float(100.0 - pinch),
        }

        # DEBUG: Log final simple mapping output
        logger.info(f"📤 Simple mapping OUTPUT: {actions}")
        return actions
    
    def _convert_curl_to_joint_value(self, curl_percentage: float) -> float:
        """Convert second hand curl percentage to joint 4 (wrist_flex) value.

        Args:
            curl_percentage: Curl from second hand (0-100)
                - 0% = hand fully open/uncurled (maps to -10° joint angle)
                - 100% = hand fully curled (maps to +90° joint angle)

        Returns:
            Joint value in normalized range (-100 to 100)
        """
        # Get wrist_flex joint limits for safety clamping
        min_deg, max_deg = self.joint_limits.get('wrist_flex', (-90, 90))

        # Convert curl percentage to degrees using custom range
        # 0% -> -10°, 100% -> +90°
        # Linear mapping: curl_degrees = -10 + (curl_percentage / 100) * (90 - (-10))
        curl_degrees = -10.0 + (curl_percentage / 100.0) * (90.0 - (-10.0))

        # Clamp to joint limits for safety
        curl_degrees = np.clip(curl_degrees, min_deg, max_deg)

        # Convert to normalized range (-100 to 100)
        joint_value = 200 * (curl_degrees - min_deg) / (max_deg - min_deg) - 100
        joint_value = np.clip(joint_value, -100, 100)

        return float(joint_value)

    @staticmethod
    def _quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        # Normalize to be safe
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm == 0:
            return np.eye(3)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        
        # Precompute values
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        
        R = np.array([
            [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
            [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
            [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)],
        ])
        return R

    @staticmethod
    def _quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def get_action(self, current_pos: Dict[str, float] = None) -> Dict[str, float]:
        """
        Get action by computing IK from IPC-tracked hand position.
        
        Args:
            current_pos: Current robot joint positions (used as IK seed)
            
        Returns:
            Dictionary of target joint positions
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        start = time.perf_counter()
        
        # Check client health
        client = self.client_manager.get_client()
        if not client or not client.is_connected():
            logger.warning("IPC client not connected, returning safe position")
            return self._get_safe_position()
        
        # Check data age - LOG ONLY, DON'T REJECT
        data_age = client.get_data_age()
        if data_age > 2.0:  # 2 second timeout instead of 500ms
            logger.debug(f"Hand tracking data age: {data_age:.3f}s - continuing anyway")
        
        # Use provided current position or default
        if current_pos is None:
            current_pos = self._get_safe_position()

        # Store current position for IK seed consistency
        self._last_current_pos = current_pos

        # Compute IK to get joint positions
        action = self.compute_ik(current_pos)

        # DEBUG: Log final action being sent to robot
        logger.info(f"🚀 FINAL ACTION to robot: {action}")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} computed action: {dt_ms:.1f}ms")

        return self._get_fixed_position(x=0.30, y=0.30, z=0.35, gripper_open=True) 
    
        return action
    

    def _get_safe_position(self) -> Dict[str, float]:
        """Get safe default position."""
        return {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        }

    def _get_fixed_position(self, x: float, y: float, z: float, gripper_open: bool = True) -> Dict[str, float]:
        """
        Calculate joint positions to move end effector to a specific 3D location.

        Args:
            x, y, z: Target position in meters (robot workspace coordinates)
            gripper_open: Whether gripper should be open (True) or closed (False)

        Returns:
            Dictionary of joint positions to reach that target
        """
        logger.info(f"🎯 FIXED POSITION TARGET: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")

        # Option 1: Use IK Solver if available
        if self.kinematics is not None:
            try:
                # Create target transformation matrix for IK
                target_pose = np.eye(4)
                target_pose[0, 3] = x  # X position
                target_pose[1, 3] = y  # Y position
                target_pose[2, 3] = z  # Z position
                # Keep identity rotation (no orientation constraints)

                # Use actual current robot position as IK seed (from get_action parameter)
                current_joints = []
                joint_names = getattr(self.config, 'joint_names', [
                    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
                ])

                # Get current position from robot (passed from get_action)
                if hasattr(self, '_last_current_pos') and self._last_current_pos:
                    current_robot_pos = self._last_current_pos
                else:
                    current_robot_pos = self._get_safe_position()  # Fallback only

                for joint_name in joint_names:
                    key = f"{joint_name}.pos"
                    norm_val = current_robot_pos.get(key, 0.0)
                    min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                    deg_val = min_deg + (norm_val + 100) * (max_deg - min_deg) / 200
                    current_joints.append(deg_val)

                current_joints = np.array(current_joints)

                # Compute IK
                joint_solution = self.kinematics.inverse_kinematics(
                    current_joints,
                    target_pose,
                    position_weight=1.0,
                    orientation_weight=0.01  # Low weight on orientation
                )

                # Convert joint angles back to normalized range
                actions = {}
                for i, joint_name in enumerate(joint_names):
                    if i < len(joint_solution):
                        deg_val = joint_solution[i]
                        min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                        # Convert from degrees to [-100, 100] range
                        norm_val = 200 * (deg_val - min_deg) / (max_deg - min_deg) - 100
                        norm_val = np.clip(norm_val, -100, 100)
                        actions[f"{joint_name}.pos"] = float(norm_val)

                # Add gripper position
                actions["gripper.pos"] = float(100.0 if gripper_open else 0.0)

                logger.info(f"📤 FIXED POSITION IK OUTPUT: {actions}")
                return actions

            except Exception as e:
                logger.warning(f"Fixed position IK failed: {e}, using simple mapping")

        # Option 2: Fallback to simple mapping
        logger.info("🤖 Using simple mapping for fixed position")

        # Use the same mapping as _simple_position_mapping()
        shoulder_pan = x * 300  # X controls pan
        shoulder_lift = -y * 200 + 50  # Y controls lift
        elbow_flex = z * 150 - 30  # Z affects elbow
        wrist_flex = y * 100  # Y also affects wrist
        wrist_roll = 0.0  # Fixed orientation

        actions = {
            "shoulder_pan.pos": float(np.clip(shoulder_pan, -100, 100)),
            "shoulder_lift.pos": float(np.clip(shoulder_lift, -100, 100)),
            "elbow_flex.pos": float(np.clip(elbow_flex, -100, 100)),
            "wrist_flex.pos": float(np.clip(wrist_flex, -100, 100)),
            "wrist_roll.pos": float(np.clip(wrist_roll, -100, 100)),
            "gripper.pos": float(100.0 if gripper_open else 0.0),
        }

        logger.info(f"📤 FIXED POSITION SIMPLE OUTPUT: {actions}")
        return actions

    def _test_robot_reach(self) -> dict:
        """
        Test the actual reach limits of the robot using URDF forward kinematics.

        Returns:
            Dictionary with actual workspace bounds based on joint limits
        """
        if self.kinematics is None:
            return self.workspace_bounds

        joint_names = getattr(self.config, 'joint_names', [
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
        ])

        # Test points by exploring joint limit combinations
        test_positions = []

        logger.info("🔍 Testing robot reach at joint limits...")

        # Test different joint configurations to find workspace bounds
        test_configs = [
            # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]

            # Maximum forward reach
            [0, -90, -90, 0, 0],    # Shoulder down, elbow extended
            [0, 45, -45, 0, 0],     # Shoulder up, elbow extended

            # Maximum left/right reach
            [90, 0, -45, 0, 0],     # Pan right, elbow extended
            [-90, 0, -45, 0, 0],    # Pan left, elbow extended
            [90, -45, -90, 0, 0],   # Pan right, shoulder down, elbow extended
            [-90, -45, -90, 0, 0],  # Pan left, shoulder down, elbow extended

            # Maximum up reach
            [0, 90, 45, -45, 0],    # Shoulder up, elbow up
            [0, 90, 90, -90, 0],    # Shoulder up, elbow fully up

            # Maximum down reach
            [0, -90, 90, 90, 0],    # Shoulder down, elbow down
            [0, -45, -90, 45, 0],   # Shoulder mid-down, elbow extended

            # Corner positions
            [90, 90, 45, -45, 0],   # Right-up
            [-90, 90, 45, -45, 0],  # Left-up
            [90, -90, -90, 0, 0],   # Right-down-forward
            [-90, -90, -90, 0, 0],  # Left-down-forward

            # Additional test positions
            [45, 0, -90, 0, 0],     # Mid-right, extended
            [-45, 0, -90, 0, 0],    # Mid-left, extended
            [0, 0, -90, 0, 0],      # Center, extended
            [0, 0, 0, 0, 0],        # Center, neutral
        ]

        for config in test_configs:
            try:
                # Clamp joint angles to defined limits
                clamped_config = []
                for i, (joint_name, angle) in enumerate(zip(joint_names, config)):
                    if i < len(config):
                        min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                        clamped_angle = np.clip(angle, min_deg, max_deg)
                        clamped_config.append(clamped_angle)

                # Use forward kinematics to get end effector position
                pose_matrix = self.kinematics.forward_kinematics(np.array(clamped_config))

                # Extract position (x, y, z) from transformation matrix
                x, y, z = pose_matrix[0, 3], pose_matrix[1, 3], pose_matrix[2, 3]
                test_positions.append((x, y, z))

                logger.debug(f"Config {clamped_config} → Position ({x:.3f}, {y:.3f}, {z:.3f})")

            except Exception as e:
                logger.debug(f"Failed to test config {config}: {e}")
                continue

        if not test_positions:
            logger.warning("Could not determine workspace bounds from URDF")
            return self.workspace_bounds

        # Calculate actual workspace bounds from test positions
        xs, ys, zs = zip(*test_positions)

        actual_bounds = {
            'x_min': float(np.min(xs)),
            'x_max': float(np.max(xs)),
            'y_min': float(np.min(ys)),
            'y_max': float(np.max(ys)),
            'z_min': float(np.min(zs)),
            'z_max': float(np.max(zs))
        }

        logger.info(f"✅ Tested {len(test_positions)} robot configurations")
        logger.info(f"📏 X range: {actual_bounds['x_min']:.3f}m to {actual_bounds['x_max']:.3f}m (span: {actual_bounds['x_max']-actual_bounds['x_min']:.3f}m)")
        logger.info(f"📏 Y range: {actual_bounds['y_min']:.3f}m to {actual_bounds['y_max']:.3f}m (span: {actual_bounds['y_max']-actual_bounds['y_min']:.3f}m)")
        logger.info(f"📏 Z range: {actual_bounds['z_min']:.3f}m to {actual_bounds['z_max']:.3f}m (span: {actual_bounds['z_max']-actual_bounds['z_min']:.3f}m)")

        return actual_bounds

    def _compare_workspace_bounds(self, actual_bounds: dict):
        """Compare actual URDF-based bounds with configured bounds and warn about issues."""

        current = self.workspace_bounds
        actual = actual_bounds

        logger.info("🔍 WORKSPACE BOUNDS COMPARISON:")
        logger.info("="*50)

        # Check each dimension
        for dim in ['x', 'y', 'z']:
            current_min = current[f'{dim}_min']
            current_max = current[f'{dim}_max']
            actual_min = actual[f'{dim}_min']
            actual_max = actual[f'{dim}_max']

            current_span = current_max - current_min
            actual_span = actual_max - actual_min

            logger.info(f"{dim.upper()} axis:")
            logger.info(f"  Current: {current_min:.3f} to {current_max:.3f} (span: {current_span:.3f}m)")
            logger.info(f"  Actual:  {actual_min:.3f} to {actual_max:.3f} (span: {actual_span:.3f}m)")

            # Check for issues
            if current_min < actual_min:
                logger.warning(f"  ⚠️  Current {dim}_min ({current_min:.3f}) is BELOW robot's reach ({actual_min:.3f})")
            if current_max > actual_max:
                logger.warning(f"  ⚠️  Current {dim}_max ({current_max:.3f}) is BEYOND robot's reach ({actual_max:.3f})")
            if current_span > actual_span * 1.1:  # 10% tolerance
                logger.warning(f"  ⚠️  Current span ({current_span:.3f}) is much larger than actual ({actual_span:.3f})")
            if current_span < actual_span * 0.8:  # 20% tolerance
                logger.info(f"  ℹ️   Current span ({current_span:.3f}) is conservative vs actual ({actual_span:.3f})")

        logger.info("="*50)

        # Suggest better bounds
        safety_margin = 0.02  # 2cm safety margin
        suggested_bounds = {
            'x_min': max(actual['x_min'] + safety_margin, 0.05),  # At least 5cm forward
            'x_max': actual['x_max'] - safety_margin,
            'y_min': actual['y_min'] + safety_margin,
            'y_max': actual['y_max'] - safety_margin,
            'z_min': max(actual['z_min'] + safety_margin, 0.05),  # Above ground
            'z_max': actual['z_max'] - safety_margin,
        }

        logger.info("💡 SUGGESTED workspace bounds (with 2cm safety margin):")
        logger.info(f"   x_min: {suggested_bounds['x_min']:.3f}, x_max: {suggested_bounds['x_max']:.3f}")
        logger.info(f"   y_min: {suggested_bounds['y_min']:.3f}, y_max: {suggested_bounds['y_max']:.3f}")
        logger.info(f"   z_min: {suggested_bounds['z_min']:.3f}, z_max: {suggested_bounds['z_max']:.3f}")

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Force feedback not implemented for hand tracking."""
        raise NotImplementedError("Force feedback not available for hand tracking")

    def disconnect(self) -> None:
        """Disconnect the IPC client and stop tracking process."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Disconnecting hand leader...")
        
        # Stop IPC client
        if self.client_manager:
            self.client_manager.stop()
        
        # Stop tracking process
        if self.process_manager:
            self.process_manager.stop()
        
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics for debugging."""
        stats = {
            'connected': self.is_connected,
            'process_manager_info': self.process_manager.get_process_info() if self.process_manager else {},
            'ipc_client_stats': {}
        }
        
        client = self.client_manager.get_client()
        if client:
            stats['ipc_client_stats'] = client.get_connection_stats()
        
        return stats
