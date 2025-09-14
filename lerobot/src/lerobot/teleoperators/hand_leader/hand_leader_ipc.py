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

        # Initialize live workspace mapping variables
        self._workspace_mapping_active = False
        self._mapped_positions = []
        self._mapping_bounds = {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'z_min': float('inf'), 'z_max': float('-inf')
        }

        # Configuration for startup workspace mapping
        self.enable_startup_mapping = getattr(config, 'enable_startup_mapping', True)
        self.startup_mapping_duration = getattr(config, 'startup_mapping_duration', 60)
        self._robot_instance = None  # Will be set by teleoperation system

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

            # Run automatic workspace mapping if enabled and kinematics available
            if self.enable_startup_mapping and self.kinematics is not None:
                logger.info("🗺️ AUTOMATIC WORKSPACE MAPPING ENABLED")
                self._run_startup_workspace_mapping()
            elif self.enable_startup_mapping and self.kinematics is None:
                logger.warning("⚠️  Workspace mapping disabled: No URDF/kinematics available")
                logger.info("💡 Add --teleop.urdf_path to enable workspace mapping")

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

    def start_live_workspace_mapping(self, duration_seconds=60):
        """
        Start live workspace mapping by tracking robot positions as user moves it.

        Args:
            duration_seconds: How long to run the mapping session (0 = manual stop)
        """
        if self.kinematics is None:
            logger.error("❌ Cannot start live mapping: No kinematics solver available")
            logger.info("💡 Add --teleop.urdf_path to your command to enable live mapping")
            return False

        if not self.is_connected:
            logger.error("❌ Cannot start live mapping: Robot not connected")
            return False

        logger.info("🚀 STARTING LIVE WORKSPACE MAPPING")
        logger.info("="*60)
        logger.info("📋 Instructions:")
        logger.info("   • Move the robot arm to explore its full workspace")
        logger.info("   • Reach as far as safely possible in all directions:")
        logger.info("     - Forward/backward (X-axis)")
        logger.info("     - Left/right (Y-axis)")
        logger.info("     - Up/down (Z-axis)")
        logger.info("   • Press Ctrl+C to stop mapping early")
        logger.info(f"   • Mapping will run for {duration_seconds}s if not stopped")
        logger.info("="*60)

        # Disable torque on the robot motors so they can be moved manually
        try:
            if hasattr(self._robot_instance, 'bus') and hasattr(self._robot_instance.bus, 'disable_torque'):
                logger.info("🔓 Disabling motor torque for manual movement...")
                self._robot_instance.bus.disable_torque()
                time.sleep(0.5)  # Give motors time to release
                logger.info("✅ Motors are now free to move manually")
            else:
                logger.warning("⚠️  Could not find disable_torque method on robot")
        except Exception as e:
            logger.warning(f"Could not disable torque: {e}")
            logger.info("⚠️  Motors may still be powered - be careful when moving the arm")

        # Reset mapping data
        self._workspace_mapping_active = True
        self._mapped_positions = []
        self._mapping_bounds = {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'z_min': float('inf'), 'z_max': float('-inf')
        }

        import threading
        import time

        start_time = time.time()
        last_display_time = 0
        sample_count = 0
        display_interval = 2.0  # Display update every 2 seconds

        # try:
        logger.info("tessthsi")

        while self._workspace_mapping_active:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Stop if duration exceeded
            if duration_seconds > 0 and elapsed_time >= duration_seconds:
                logger.info("⏰ Mapping duration reached, stopping...")
                break

            # Get current robot position (same way as teleoperation loop)
            if self._robot_instance and hasattr(self._robot_instance, 'get_pos'):
                # Read current robot position directly - just like teleop_loop does
                current_pos = self._robot_instance.get_pos()
                logger.info(f"Current robot position: {current_pos}")

                if current_pos:
                    # Convert robot joint positions to end effector position using forward kinematics
                    ee_position = self._calculate_end_effector_position_from_robot_pos(current_pos)

                    # if ee_position:
                    #     # Update boundaries
                    #     self._update_mapping_bounds(ee_position)
                    #     sample_count += 1
                    #     logger.debug(f"Mapped position: {ee_position}")
                    # else:
                    #     logger.debug("Failed to calculate end effector position")
                else:
                    logger.debug("robot.get_pos() returned None")
            else:
                logger.debug("No robot instance available")

            # except Exception as e:
            #     logger.debug(f"Failed to read robot position: {e}")

            # Display progress periodically
            if current_time - last_display_time >= display_interval:
                self._display_mapping_progress(elapsed_time, duration_seconds, sample_count)
                last_display_time = current_time

            time.sleep(0.5)  # 2 Hz sampling rate

        # except KeyboardInterrupt:
        #     logger.info("\n🛑 Mapping stopped by user")

        # Finish mapping
        self._workspace_mapping_active = False

        # Re-enable torque on the robot motors
        try:
            if hasattr(self._robot_instance, 'bus') and hasattr(self._robot_instance.bus, 'enable_torque'):
                logger.info("🔒 Re-enabling motor torque...")
                self._robot_instance.bus.enable_torque()
                time.sleep(0.5)  # Give motors time to engage
        except Exception as e:
            logger.warning(f"Could not re-enable torque: {e}")

        final_bounds = self._finalize_workspace_mapping(sample_count)

        logger.info("✅ Live workspace mapping completed!")
        return final_bounds

    def stop_live_workspace_mapping(self):
        """Stop the live workspace mapping session."""
        self._workspace_mapping_active = False
        logger.info("🛑 Stopping live workspace mapping...")

    def _get_current_robot_position(self):
        """Get current robot joint positions from the robot instance or last known position."""
        try:
            # Method 1: Read directly from robot instance if available
            if self._robot_instance and hasattr(self._robot_instance, 'get_pos'):
                try:
                    robot_pos = self._robot_instance.get_pos()
                    if robot_pos:
                        # Extract joint positions in order
                        joint_names = getattr(self.config, 'joint_names', [
                            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
                        ])

                        joint_positions = []
                        for joint_name in joint_names:
                            key = f"{joint_name}.pos"
                            if key in robot_pos:
                                joint_positions.append(robot_pos[key])
                            else:
                                joint_positions.append(0.0)  # Default if missing

                        return joint_positions
                except Exception as e:
                    logger.debug(f"Failed to read from robot instance: {e}")

            # Method 2: Use last known position from get_action()
            if hasattr(self, '_last_current_pos') and self._last_current_pos:
                # Extract joint positions in order
                joint_names = getattr(self.config, 'joint_names', [
                    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
                ])

                joint_positions = []
                for joint_name in joint_names:
                    key = f"{joint_name}.pos"
                    if key in self._last_current_pos:
                        joint_positions.append(self._last_current_pos[key])
                    else:
                        joint_positions.append(0.0)  # Default if missing

                return joint_positions

            return None

        except Exception as e:
            logger.debug(f"Failed to get current robot position: {e}")
            return None

    def _calculate_end_effector_position(self, joint_positions):
        """Calculate end effector position from joint positions using forward kinematics."""
        try:
            joint_names = getattr(self.config, 'joint_names', [
                "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
            ])

            # Convert normalized positions to degrees
            joint_angles_deg = []
            for i, joint_name in enumerate(joint_names):
                if i < len(joint_positions):
                    norm_val = joint_positions[i]
                    min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                    deg_val = min_deg + (norm_val + 100) * (max_deg - min_deg) / 200
                    joint_angles_deg.append(deg_val)

            # Use forward kinematics
            pose_matrix = self.kinematics.forward_kinematics(np.array(joint_angles_deg))
            x, y, z = pose_matrix[0, 3], pose_matrix[1, 3], pose_matrix[2, 3]

            return (x, y, z)

        except Exception as e:
            logger.debug(f"Failed to calculate end effector position: {e}")
            return None

    def _calculate_end_effector_position_from_robot_pos(self, robot_pos):
        """Calculate end effector position from robot position dict (same format as robot.get_pos())."""
        try:
            if not self.kinematics:
                logger.error("Kinematics not initialized - check URDF path")
                return 1/0
            joint_names = getattr(self.config, 'joint_names', [
                "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
            ])

            # Extract joint angles from robot position dict
            joint_angles_deg = []
            for joint_name in joint_names:
                key = f"{joint_name}.pos"
                if key in robot_pos:
                    norm_val = robot_pos[key]  # Robot returns normalized [-100, 100] values
                    min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                    # Convert normalized value to degrees
                    deg_val = min_deg + (norm_val + 100) * (max_deg - min_deg) / 200
                    joint_angles_deg.append(deg_val)
                    logger.info(f"{joint_name}: norm={norm_val:.1f} -> deg={deg_val:.1f}")
                else:
                    logger.info(f"Missing joint {key} in robot position")
                    joint_angles_deg.append(0.0)  # Default

            # Use forward kinematics
            pose_matrix = self.kinematics.forward_kinematics(np.array(joint_angles_deg))
            x, y, z = pose_matrix[0, 3], pose_matrix[1, 3], pose_matrix[2, 3]

            logger.info(f"Forward kinematics: joints={joint_angles_deg} -> pos=({x:.3f}, {y:.3f}, {z:.3f})")
            return (x, y, z)

        except Exception as e:
            logger.error(f"Failed to calculate end effector position from robot pos: {e}")
            return None

    def _update_mapping_bounds(self, position):
        """Update the running workspace boundaries with new position."""
        x, y, z = position

        # Validate position is reasonable (basic sanity check)
        if abs(x) > 2.0 or abs(y) > 2.0 or abs(z) > 2.0:
            logger.debug(f"Ignoring outlier position: ({x:.3f}, {y:.3f}, {z:.3f})")
            return

        # Update boundaries
        self._mapping_bounds['x_min'] = min(self._mapping_bounds['x_min'], x)
        self._mapping_bounds['x_max'] = max(self._mapping_bounds['x_max'], x)
        self._mapping_bounds['y_min'] = min(self._mapping_bounds['y_min'], y)
        self._mapping_bounds['y_max'] = max(self._mapping_bounds['y_max'], y)
        self._mapping_bounds['z_min'] = min(self._mapping_bounds['z_min'], z)
        self._mapping_bounds['z_max'] = max(self._mapping_bounds['z_max'], z)

        # Store position for analysis
        self._mapped_positions.append((x, y, z))

    def _display_mapping_progress(self, elapsed_time, duration_seconds, sample_count):
        """Display current mapping progress."""
        bounds = self._mapping_bounds

        # Handle case where no valid positions recorded yet
        if bounds['x_min'] == float('inf'):
            logger.info(f"⏱️  Mapping progress: {elapsed_time:.1f}s - No valid positions recorded yet")
            return

        time_str = f"{elapsed_time:.1f}s"
        if duration_seconds > 0:
            time_str += f"/{duration_seconds}s"

        logger.info(f"⏱️  LIVE WORKSPACE MAPPING ({time_str}) - Samples: {sample_count}")
        logger.info(f"📏 X: {bounds['x_min']:.3f}m to {bounds['x_max']:.3f}m (span: {bounds['x_max']-bounds['x_min']:.3f}m)")
        logger.info(f"📏 Y: {bounds['y_min']:.3f}m to {bounds['y_max']:.3f}m (span: {bounds['y_max']-bounds['y_min']:.3f}m)")
        logger.info(f"📏 Z: {bounds['z_min']:.3f}m to {bounds['z_max']:.3f}m (span: {bounds['z_max']-bounds['z_min']:.3f}m)")

    def _finalize_workspace_mapping(self, sample_count):
        """Finalize workspace mapping and return results."""
        bounds = self._mapping_bounds

        if sample_count == 0 or bounds['x_min'] == float('inf'):
            logger.warning("❌ No valid positions recorded during mapping")
            return self.workspace_bounds

        logger.info("="*60)
        logger.info("📊 FINAL WORKSPACE MAPPING RESULTS")
        logger.info(f"✅ Recorded {sample_count} valid positions")
        logger.info(f"📏 X range: {bounds['x_min']:.3f}m to {bounds['x_max']:.3f}m (span: {bounds['x_max']-bounds['x_min']:.3f}m)")
        logger.info(f"📏 Y range: {bounds['y_min']:.3f}m to {bounds['y_max']:.3f}m (span: {bounds['y_max']-bounds['y_min']:.3f}m)")
        logger.info(f"📏 Z range: {bounds['z_min']:.3f}m to {bounds['z_max']:.3f}m (span: {bounds['z_max']-bounds['z_min']:.3f}m)")

        # Add safety margin
        safety_margin = 0.02  # 2cm
        safe_bounds = {
            'x_min': bounds['x_min'] + safety_margin,
            'x_max': bounds['x_max'] - safety_margin,
            'y_min': bounds['y_min'] + safety_margin,
            'y_max': bounds['y_max'] - safety_margin,
            'z_min': max(bounds['z_min'] + safety_margin, 0.05),  # At least 5cm above ground
            'z_max': bounds['z_max'] - safety_margin,
        }

        logger.info("💡 SUGGESTED workspace bounds (with 2cm safety margin):")
        logger.info(f"   x_min: {safe_bounds['x_min']:.3f}, x_max: {safe_bounds['x_max']:.3f}")
        logger.info(f"   y_min: {safe_bounds['y_min']:.3f}, y_max: {safe_bounds['y_max']:.3f}")
        logger.info(f"   z_min: {safe_bounds['z_min']:.3f}, z_max: {safe_bounds['z_max']:.3f}")
        logger.info("="*60)

        # Compare with current bounds
        self._compare_workspace_bounds(bounds)

        return safe_bounds

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

    def map_workspace_live(self, duration_seconds=60):
        """
        Convenience method to start live workspace mapping.

        Usage:
            teleop.map_workspace_live(60)  # Map for 60 seconds
            teleop.map_workspace_live(0)   # Map until Ctrl+C

        Args:
            duration_seconds: Mapping duration (0 = manual stop)

        Returns:
            Dictionary with measured workspace bounds
        """
        return self.start_live_workspace_mapping(duration_seconds)

    def _run_startup_workspace_mapping(self):
        """Run automatic workspace mapping at startup."""
        logger.info("")
        logger.info("🚀 STARTING AUTOMATIC WORKSPACE MAPPING")
        logger.info("="*60)
        logger.info("📋 INSTRUCTIONS:")
        logger.info("   🎮 Use your hand tracking or leader arm to move the robot")
        logger.info("   📐 Explore the robot's full workspace:")
        logger.info("     • Move as far FORWARD/BACKWARD as safely possible")
        logger.info("     • Move as far LEFT/RIGHT as safely possible")
        logger.info("     • Move as HIGH/LOW as safely possible")
        logger.info("   ⏱️  You have {} seconds to explore".format(self.startup_mapping_duration))
        logger.info("   🛑 Press Ctrl+C to stop mapping early")
        logger.info("   📊 Real-time boundaries will be displayed")
        logger.info("="*60)
        logger.info("")

        # try:
        # Wait a moment for user to read instructions
        import time
        logger.info("⏳ Starting workspace mapping in 3 seconds...")
        time.sleep(1)
        logger.info("⏳ 2...")
        time.sleep(1)
        logger.info("⏳ 1...")
        time.sleep(1)
        logger.info("🟢 GO! Move the robot to map workspace!")
        logger.info("")

        # Run the live mapping
        mapped_bounds = self.start_live_workspace_mapping(self.startup_mapping_duration)

        if mapped_bounds and mapped_bounds != self.workspace_bounds:
            logger.info("")
            logger.info("🔄 UPDATING WORKSPACE BOUNDS")
            logger.info(f"   Old bounds: {self.workspace_bounds}")
            logger.info(f"   New bounds: {mapped_bounds}")

            # Update the workspace bounds with the mapped results
            self.workspace_bounds.update(mapped_bounds)

            logger.info("✅ Workspace bounds updated for this session!")
            logger.info("💡 Consider adding these bounds to your configuration")
        else:
            logger.info("ℹ️  No workspace bounds changes needed")

        # except KeyboardInterrupt:
        #     logger.info("")
        #     logger.info("🛑 Startup workspace mapping interrupted by user")
        #     logger.info("   Exiting teleoperation...")
        #     # Re-raise KeyboardInterrupt to exit the entire program
        #     raise

        # except Exception as e:
        #     logger.error(f"❌ Startup workspace mapping failed: {e}")
        #     logger.info("   Continuing with default workspace bounds...")

        logger.info("")
        logger.info("🎯 STARTING NORMAL TELEOPERATION")
        logger.info("="*60)
        logger.info("")

    def set_robot_instance(self, robot):
        """Set the robot instance for position reading during mapping."""
        self._robot_instance = robot
