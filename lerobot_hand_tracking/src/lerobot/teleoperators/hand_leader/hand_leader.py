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
from typing import Dict, Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics

from ..teleoperator import Teleoperator
from .config_hand_leader import HandLeaderConfig
from .cv_hand_tracker import get_tracker_instance

logger = logging.getLogger(__name__)


class HandLeader(Teleoperator):
    """
    Hand Leader using computer vision (MediaPipe) for teleoperation control.
    Tracks hand position and pinch gesture to control robot arm.
    """

    config_class = HandLeaderConfig
    name = "hand_leader"

    def __init__(self, config: HandLeaderConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize CV tracker
        self.tracker = get_tracker_instance(config.camera_index)
        
        # Initialize kinematics solver
        self.kinematics = RobotKinematics(
            urdf_path=config.urdf_path,
            target_frame_name=config.target_frame_name,
            joint_names=config.joint_names
        )
        
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
        """No feedback features for CV-based control."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the CV tracker and run calibration if needed."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting hand leader (CV tracker)...")
        
        # Initialize CV tracker
        self.tracker.connect()
        
        # Run calibration if requested
        if calibrate:
            self.calibrate()
        else:
            # If not calibrating, we need to start tracking manually
            if not self.tracker.calibrated:
                logger.warning("Tracker not calibrated, using default calibration values")
                # Set some reasonable defaults
                self.tracker.palm_bbox_far = 50
                self.tracker.palm_bbox_near = 150
                self.tracker.initial_thumb_index_dist = 100
                self.tracker.calibrated = True
            self.tracker.start_tracking()
        
        self._is_connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if the CV tracker is calibrated."""
        return self.tracker.calibrated if self.tracker else False

    def calibrate(self) -> None:
        """Run the CV calibration routine."""
        logger.info(f"\nRunning calibration of {self}")
        self.tracker.calibrate()
        logger.info("Calibration complete")

    def configure(self) -> None:
        """No motor configuration needed for CV control."""
        pass

    def setup_motors(self) -> None:
        """No motor setup needed for CV control."""
        pass

    def get_endpos(self):
        """
        Get current end effector position from CV tracker.
        
        Returns:
            Tuple of (x, y, z, pinch) where:
            - x, y, z are in meters (robot workspace coordinates)
            - pinch is gripper percentage (0-100)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        return self.tracker.get_current_position()

    def compute_ik(self, current_pos: Dict[str, float]) -> Dict[str, float]:
        """
        Compute inverse kinematics from end effector position to joint angles.
        
        Args:
            current_pos: Current joint positions (used as IK seed)
            
        Returns:
            Dictionary of joint positions including gripper
        """
        # Get end effector target from CV
        x, y, z, pinch = self.get_endpos()
        
        # Create target transformation matrix for IK
        # Simple approach: keep orientation fixed, only change position
        target_pose = np.eye(4)
        target_pose[0, 3] = x  # X position
        target_pose[1, 3] = y  # Y position  
        target_pose[2, 3] = z  # Z position
        
        # Extract current joint values for IK seed
        # Convert from normalized (-100 to 100) to degrees if needed
        current_joints = []
        for joint_name in self.config.joint_names:
            key = f"{joint_name}.pos"
            if key in current_pos:
                # Assuming current_pos values are in normalized range -100 to 100
                # Convert to degrees based on joint limits
                norm_val = current_pos[key]
                min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                # Convert from [-100, 100] to [min_deg, max_deg]
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
            for i, joint_name in enumerate(self.config.joint_names):
                if i < len(joint_solution):
                    deg_val = joint_solution[i]
                    min_deg, max_deg = self.joint_limits.get(joint_name, (-90, 90))
                    # Convert from degrees to [-100, 100] range
                    norm_val = 200 * (deg_val - min_deg) / (max_deg - min_deg) - 100
                    norm_val = np.clip(norm_val, -100, 100)
                    actions[f"{joint_name}.pos"] = float(norm_val)
            
            # Add gripper position (already in 0-100 range)
            actions["gripper.pos"] = float(pinch)
            
            return actions
            
        except Exception as e:
            logger.warning(f"IK failed: {e}, returning safe position")
            # Return safe default position on IK failure
            return {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": float(pinch),
            }

    def get_action(self, current_pos: Dict[str, float] = None) -> Dict[str, float]:
        """
        Get action by computing IK from CV-tracked hand position.
        
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
        
        # Compute IK to get joint positions
        action = self.compute_ik(current_pos)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} computed action: {dt_ms:.1f}ms")
        
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Force feedback not implemented for CV control."""
        raise NotImplementedError("Force feedback not available for CV hand tracking")

    def disconnect(self) -> None:
        """Disconnect the CV tracker."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.tracker.disconnect()
        self._is_connected = False
        logger.info(f"{self} disconnected.")