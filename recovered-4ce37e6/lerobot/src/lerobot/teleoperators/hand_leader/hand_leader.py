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

# Try to import IPC client first, fall back to direct MediaPipe if server not running
try:
    from .cv_hand_tracker_ipc import get_tracker_instance
    USE_IPC = True
except ImportError:
    from .cv_hand_tracker import get_tracker_instance
    USE_IPC = False

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
        # Lightweight smoothing + deadband to reduce jitter without adding latency
        self._prev_xyz = None
        self._prev_action: Dict[str, float] | None = None
        self._pos_smoothing_alpha = 0.2  # small smoothing for stability
        self._deadband = 1.0  # ignore tiny normalized changes to avoid servo buzz
        
        # Automatically use laptop camera (index 0)
        camera_index = 0  # Always use default camera
        logger.info(f"Using default camera (index {camera_index}) for hand tracking")

        # Initialize CV tracker with automatic camera selection
        self.tracker = get_tracker_instance(camera_index)

        # Motor calibration data from LeRobot
        # Format: {name: {'min': min_pos, 'center': center_pos, 'max': max_pos}}
        self.motor_calibration = {
            'shoulder_pan': {'min': 1027, 'center': 2192, 'max': 3060},
            'shoulder_lift': {'min': 800, 'center': 1059, 'max': 3158},
            'elbow_flex': {'min': 910, 'center': 3138, 'max': 3163},
            'wrist_flex': {'min': 858, 'center': 2607, 'max': 3204},
            'wrist_roll': {'min': 126, 'center': 1927, 'max': 3960},
            'gripper': {'min': 2015, 'center': 2043, 'max': 3502},
        }

        # Convert to normalized ranges for safety
        self.joint_limits = {}
        for joint_name, calib in self.motor_calibration.items():
            # Calculate safe normalized range (-100 to 100)
            # where 0 is center position
            self.joint_limits[joint_name] = self._calculate_safe_range(calib)

        self._is_connected = False

    def _calculate_safe_range(self, calib):
        """Calculate safe normalized range from motor calibration."""
        # Apply safety margin of 90% to avoid hitting hard limits
        safety_factor = 0.9

        # For most joints, we use symmetric range around center
        # But respect the actual min/max limits
        return {
            'min_norm': -90,  # Safe normalized minimum
            'max_norm': 90,   # Safe normalized maximum
            'min_pos': calib['min'],
            'center_pos': calib['center'],
            'max_pos': calib['max']
        }

    def _normalize_to_motor_position(self, joint_name, normalized_value):
        """Convert normalized value (-100 to 100) to actual motor position."""
        calib = self.motor_calibration[joint_name]
        limits = self.joint_limits[joint_name]

        # Clamp to safe range
        normalized_value = np.clip(normalized_value, limits['min_norm'], limits['max_norm'])

        # Convert to motor position
        if normalized_value >= 0:
            # Positive: interpolate between center and max
            ratio = normalized_value / 100.0
            motor_pos = calib['center'] + ratio * (calib['max'] - calib['center'])
        else:
            # Negative: interpolate between min and center
            ratio = -normalized_value / 100.0
            motor_pos = calib['center'] - ratio * (calib['center'] - calib['min'])

        # Final safety clamp
        motor_pos = np.clip(motor_pos, calib['min'], calib['max'])
        return int(motor_pos)

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

    def connect(self, calibrate: bool = False) -> None:
        """Connect to the CV tracker and run calibration if needed."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting hand leader (CV tracker)...")

        # Initialize CV tracker
        self.tracker.connect()

        # Skip calibration on Windows - use defaults
        logger.info("Skipping calibration, using default values for hand tracking")
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
        Simple geometric mapping from hand position to joint angles with safety limits.

        This bypasses complex IK and uses direct geometric relationships.

        Args:
            x, y, z: Hand position in robot workspace (meters)

        Returns:
            Dictionary of joint positions with motor safety limits applied
        """
        # Workspace center for relative positioning
        workspace_center = {'x': 0.0, 'y': 0.2, 'z': 0.15}
        # Slightly higher scaling for a snappier feel (range maps to -100..100)
        workspace_scale = {'x': 180, 'y': 150, 'z': 100}

        # Calculate relative position from center
        dx = x - workspace_center['x']
        # Invert Y so that moving hand down moves robot down
        dy = workspace_center['y'] - y
        dz = z - workspace_center['z']

        # Simple geometric mapping to joints
        joint_positions = {}

        # Shoulder pan: left-right movement (X maps to pan)
        shoulder_pan_norm = dx * workspace_scale['x']
        shoulder_pan_safe = self._apply_joint_limits('shoulder_pan', shoulder_pan_norm)
        joint_positions["shoulder_pan.pos"] = float(shoulder_pan_safe)

        # Shoulder lift: up-down movement (Y maps to lift)
        shoulder_lift_norm = dy * workspace_scale['y']
        shoulder_lift_safe = self._apply_joint_limits('shoulder_lift', shoulder_lift_norm)
        joint_positions["shoulder_lift.pos"] = float(shoulder_lift_safe)

        # Elbow flex: forward-back movement (Z maps to elbow)
        elbow_flex_norm = -dz * workspace_scale['z']
        elbow_flex_safe = self._apply_joint_limits('elbow_flex', elbow_flex_norm)
        joint_positions["elbow_flex.pos"] = float(elbow_flex_safe)

        # Wrist flex: slight downward tilt based on distance
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        wrist_flex_norm = -distance * 40
        wrist_flex_safe = self._apply_joint_limits('wrist_flex', wrist_flex_norm)
        joint_positions["wrist_flex.pos"] = float(wrist_flex_safe)

        # Wrist roll: slight rotation based on X position
        wrist_roll_norm = dx * 25
        wrist_roll_safe = self._apply_joint_limits('wrist_roll', wrist_roll_norm)
        joint_positions["wrist_roll.pos"] = float(wrist_roll_safe)

        return joint_positions

    def _apply_joint_limits(self, joint_name, normalized_value):
        """Apply safety limits to a joint position."""
        if joint_name not in self.joint_limits:
            logger.warning(f"No limits defined for joint {joint_name}")
            return np.clip(normalized_value, -100, 100)

        limits = self.joint_limits[joint_name]
        safe_value = np.clip(normalized_value, limits['min_norm'], limits['max_norm'])

        if normalized_value != safe_value:
            logger.debug(f"Joint {joint_name} limited: {normalized_value:.1f} -> {safe_value:.1f}")

        return safe_value

    def get_action(self, current_pos: Dict[str, float] = None) -> Dict[str, float]:
        """
        Get action by mapping hand position to joint positions.

        Uses simple geometric mapping instead of complex IK.

        Args:
            current_pos: Current robot joint positions (not used in simple mapping)

        Returns:
            Dictionary of target joint positions
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Get hand position and pinch
        x, y, z, pinch = self.get_endpos()

        # Apply very light smoothing on XYZ only (keeps latency low)
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

        # Use simple geometric mapping instead of IK
        action = self.simple_position_to_joints(*smoothed_xyz)

        # Add gripper position with safety limits
        # Invert pinch so that closing fingers closes gripper (mirroring fix)
        # Pinch is 0-100% -> map to -100..100 normalized
        gripper_norm = (50 - pinch) * 2
        gripper_safe = self._apply_joint_limits('gripper', gripper_norm)
        action["gripper.pos"] = float(gripper_safe)

        # Apply a small deadband to reduce micro-oscillations on all joints
        if self._prev_action is not None:
            for k in action.keys():
                prev = self._prev_action.get(k, action[k])
                if abs(action[k] - prev) < self._deadband:
                    action[k] = prev
        self._prev_action = dict(action)

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
