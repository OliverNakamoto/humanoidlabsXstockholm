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
        
        # Define joint limits (in degrees)
        self.joint_limits = {
            'shoulder_pan': (-90, 90),
            'shoulder_lift': (-90, 90),
            'elbow_flex': (-90, 90),
            'wrist_flex': (-90, 90),
            'wrist_roll': (-180, 180),
        }
        
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
        Get current end effector position from IPC client.
        
        Returns:
            Tuple of (x, y, z, pinch) where:
            - x, y, z are in meters (robot workspace coordinates)
            - pinch is gripper percentage (0-100)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        client = self.client_manager.get_client()
        if not client:
            logger.warning("No active IPC client")
            return (0.0, 0.2, 0.2, 0.0)
        
        return client.get_current_position()

    def compute_ik(self, current_pos: Dict[str, float]) -> Dict[str, float]:
        """
        Compute inverse kinematics from end effector position to joint angles.
        
        Args:
            current_pos: Current joint positions (used as IK seed)
            
        Returns:
            Dictionary of joint positions including gripper
        """
        # Get end effector target from IPC
        x, y, z, pinch = self.get_endpos()
        
        # If no kinematics solver available, use simple mapping
        if self.kinematics is None:
            return self._simple_position_mapping(x, y, z, pinch)
        
        # Create target transformation matrix for IK
        target_pose = np.eye(4)
        target_pose[0, 3] = x  # X position
        target_pose[1, 3] = y  # Y position  
        target_pose[2, 3] = z  # Z position
        
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
            
            # Add gripper position
            actions["gripper.pos"] = float(pinch)
            
            return actions
            
        except Exception as e:
            logger.warning(f"IK failed: {e}, using simple mapping")
            return self._simple_position_mapping(x, y, z, pinch)

    def _simple_position_mapping(self, x: float, y: float, z: float, pinch: float) -> Dict[str, float]:
        """Simple position mapping when IK is not available."""
        # Map hand position to joint angles (simplified)
        shoulder_pan = x * 300  # X controls pan
        shoulder_lift = -y * 200 + 50  # Y controls lift
        elbow_flex = z * 150 - 30  # Z affects elbow
        wrist_flex = y * 100  # Y also affects wrist
        wrist_roll = x * 180  # X controls wrist roll
        
        return {
            "shoulder_pan.pos": np.clip(shoulder_pan, -100, 100),
            "shoulder_lift.pos": np.clip(shoulder_lift, -100, 100),
            "elbow_flex.pos": np.clip(elbow_flex, -100, 100),
            "wrist_flex.pos": np.clip(wrist_flex, -100, 100),
            "wrist_roll.pos": np.clip(wrist_roll, -100, 100),
            "gripper.pos": float(pinch),
        }

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
        
        # Check data age
        data_age = client.get_data_age()
        if data_age > 0.5:  # 500ms timeout
            logger.warning(f"Hand tracking data too old ({data_age:.3f}s), returning safe position")
            return self._get_safe_position()
        
        # Use provided current position or default
        if current_pos is None:
            current_pos = self._get_safe_position()
        
        # Compute IK to get joint positions
        action = self.compute_ik(current_pos)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} computed action: {dt_ms:.1f}ms")
        
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