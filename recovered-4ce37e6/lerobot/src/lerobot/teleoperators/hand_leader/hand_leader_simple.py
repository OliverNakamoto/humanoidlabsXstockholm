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

from ..teleoperator import Teleoperator
from .config_hand_leader import HandLeaderConfig
from .cv_hand_tracker_ipc import get_tracker_instance

logger = logging.getLogger(__name__)


class HandLeaderSimple(Teleoperator):
    """
    Simplified Hand Leader using computer vision (MediaPipe) for teleoperation control.

    This version bypasses inverse kinematics and directly maps hand position to joint angles
    using simple geometric transformations. Good for testing and demonstrations.
    """

    config_class = HandLeaderConfig
    name = "hand_leader_simple"

    def __init__(self, config: HandLeaderConfig):
        super().__init__(config)
        self.config = config

        # Initialize CV tracker
        self.tracker = get_tracker_instance(config.camera_index)

        # Minimal smoothing + deadband to reduce jitter while staying responsive
        self._prev_xyz = None
        self._prev_action: Dict[str, float] | None = None
        self._pos_smoothing_alpha = 0.2
        self._deadband = 1.0

        # Define joint limits (in normalized range -100 to 100)
        self.joint_limits = {
            'shoulder_pan': (-100, 100),     # Left-right movement
            'shoulder_lift': (-100, 100),    # Up-down movement
            'elbow_flex': (-100, 100),       # Forward-back movement
            'wrist_flex': (-100, 100),       # Wrist pitch
            'wrist_roll': (-100, 100),       # Wrist roll
        }

        # Workspace mapping parameters
        self.workspace_center = {'x': 0.0, 'y': 0.3, 'z': 0.25}  # Robot workspace center
        # Slightly higher scaling for snappier mapping
        self.workspace_scale = {'x': 220, 'y': 170, 'z': 120}    # Scale factors

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

    def simple_position_to_joints(self, x: float, y: float, z: float) -> Dict[str, float]:
        """
        Simple geometric mapping from hand position to joint angles.

        This is a simplified approach that doesn't use inverse kinematics.
        Instead, it maps hand position directly to joint space using geometric rules.

        Args:
            x, y, z: Hand position in robot workspace (meters)

        Returns:
            Dictionary of joint positions in normalized range (-100 to 100)
        """
        # Center position relative to workspace center
        dx = x - self.workspace_center['x']
        # Invert Y so that moving hand down moves robot down
        dy = self.workspace_center['y'] - y
        dz = z - self.workspace_center['z']

        # Map to joint space using simple geometric rules
        joint_positions = {}

        # Shoulder pan: left-right movement
        # Positive X (right) -> positive shoulder pan
        shoulder_pan = np.clip(dx * self.workspace_scale['x'], -100, 100)
        joint_positions["shoulder_pan.pos"] = float(shoulder_pan)

        # Shoulder lift: up-down movement
        # Higher Y (up) -> positive shoulder lift
        shoulder_lift = np.clip(dy * self.workspace_scale['y'], -100, 100)
        joint_positions["shoulder_lift.pos"] = float(shoulder_lift)

        # Elbow flex: forward-back movement
        # Further Z (forward) -> negative elbow flex (extension)
        elbow_flex = np.clip(-dz * self.workspace_scale['z'], -100, 100)
        joint_positions["elbow_flex.pos"] = float(elbow_flex)

        # Wrist flex: slight downward tilt based on distance
        # Further away -> more downward tilt
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        wrist_flex = np.clip(-distance * 50, -100, 100)
        joint_positions["wrist_flex.pos"] = float(wrist_flex)

        # Wrist roll: slight rotation based on X position
        wrist_roll = np.clip(dx * 30, -100, 100)
        joint_positions["wrist_roll.pos"] = float(wrist_roll)

        return joint_positions

    def get_action(self, current_pos: Dict[str, float] = None) -> Dict[str, float]:
        """
        Get action by mapping hand position to joint space.

        Args:
            current_pos: Current robot joint positions (not used in this simple version)

        Returns:
            Dictionary of target joint positions
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Get hand position and pinch
        x, y, z, pinch = self.get_endpos()

        # Light smoothing on position only
        if self._prev_xyz is None:
            smoothed_xyz = (x, y, z)
        else:
            px, py, pz = self._prev_xyz
            smoothed_xyz = (
                px * self._pos_smoothing_alpha + x * (1 - self._pos_smoothing_alpha),
                py * self._pos_smoothing_alpha + y * (1 - self._pos_smoothing_alpha),
                pz * self._pos_smoothing_alpha + z * (1 - self._pos_smoothing_alpha),
            )
        self._prev_xyz = smoothed_xyz

        # Simple position to joint mapping
        joint_positions = self.simple_position_to_joints(*smoothed_xyz)

        # Invert pinch so that closing fingers closes gripper (mirroring fix)
        joint_positions["gripper.pos"] = float(100.0 - pinch)

        # Apply small deadband to reduce micro-oscillations
        if self._prev_action is not None:
            for k in joint_positions.keys():
                prev = self._prev_action.get(k, joint_positions[k])
                if abs(joint_positions[k] - prev) < self._deadband:
                    joint_positions[k] = prev
        self._prev_action = dict(joint_positions)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} computed action: {dt_ms:.1f}ms")

        return joint_positions

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
