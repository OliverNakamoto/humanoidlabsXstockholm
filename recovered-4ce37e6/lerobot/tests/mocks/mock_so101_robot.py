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

import numpy as np
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict
import logging

from lerobot.cameras import CameraConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots import Robot, RobotConfig

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("mock_so101")
@dataclass
class MockSO101Config(RobotConfig):
    """Configuration for Mock SO101 Robot."""
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    calibrated: bool = True
    use_degrees: bool = False
    smooth_movement: bool = True  # Enable smooth interpolation
    movement_speed: float = 0.1  # Speed of interpolation (0-1)
    add_noise: bool = False  # Add realistic noise to positions
    noise_level: float = 0.5  # Noise amplitude


class MockSO101Robot(Robot):
    """Mock SO101 Robot for simulation and testing without hardware."""

    config_class = MockSO101Config
    name = "mock_so101"

    def __init__(self, config: MockSO101Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = config.calibrated
        
        # SO101 specific motors
        self.motors = [
            "shoulder_pan",
            "shoulder_lift", 
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]
        
        # Current positions (normalized -100 to 100 for joints, 0 to 100 for gripper)
        self.current_positions = {
            f"{motor}.pos": 0.0 if motor != "gripper" else 50.0 
            for motor in self.motors
        }
        
        # Target positions for smooth movement
        self.target_positions = self.current_positions.copy()
        
        # Last update time for interpolation
        self.last_update_time = time.time()
        
        # Store last received action for debugging
        self.last_action = None
        
        logger.info(f"Initialized {self.name} with motors: {self.motors}")

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Return empty dict since we're not using cameras in simulation
        return {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect the mock robot."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._is_connected = True
        logger.info(f"Mock SO101 robot connected (simulation mode)")
        
        if calibrate and not self.is_calibrated:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        """Simulate calibration."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Simulating SO101 calibration...")
        time.sleep(0.5)  # Simulate calibration time
        
        # Reset to home position
        self.current_positions = {
            f"{motor}.pos": 0.0 if motor != "gripper" else 50.0 
            for motor in self.motors
        }
        self.target_positions = self.current_positions.copy()
        
        self._is_calibrated = True
        logger.info("Mock SO101 calibration complete")

    def configure(self) -> None:
        """No configuration needed for mock robot."""
        pass

    def get_pos(self) -> dict[str, Any]:
        """Get current position (for teleoperation loop compatibility)."""
        return self.get_observation()

    def get_observation(self) -> dict[str, Any]:
        """Get current robot state with smooth interpolation."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Update positions with interpolation if enabled
        if self.config.smooth_movement:
            self._update_positions()
        else:
            # Instant movement to target
            self.current_positions = self.target_positions.copy()
        
        # Add noise if configured
        obs = self.current_positions.copy()
        if self.config.add_noise:
            for key in obs:
                if key != "gripper.pos":  # Less noise on gripper
                    noise = np.random.normal(0, self.config.noise_level)
                    obs[key] = np.clip(obs[key] + noise, -100, 100)
        
        return obs

    def _update_positions(self):
        """Smoothly interpolate current positions toward targets."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Interpolate each motor position
        for key in self.current_positions:
            current = self.current_positions[key]
            target = self.target_positions[key]
            
            # Calculate interpolation step
            diff = target - current
            step = diff * self.config.movement_speed * min(dt * 10, 1.0)  # Cap dt effect
            
            # Update position
            if abs(diff) > 0.1:  # Only update if not close enough
                self.current_positions[key] = current + step
            else:
                self.current_positions[key] = target

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Receive action and update target positions."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Validate action keys
        expected_keys = set(self.action_features.keys())
        received_keys = set(action.keys())
        
        if received_keys != expected_keys:
            missing = expected_keys - received_keys
            extra = received_keys - expected_keys
            if missing:
                logger.warning(f"Missing action keys: {missing}")
            if extra:
                logger.warning(f"Extra action keys: {extra}")
        
        # Update target positions
        for key, value in action.items():
            if key in self.target_positions:
                # Clip values to valid ranges
                if key == "gripper.pos":
                    self.target_positions[key] = np.clip(value, 0, 100)
                else:
                    self.target_positions[key] = np.clip(value, -100, 100)
        
        self.last_action = action.copy()
        
        # Log received action for debugging
        logger.debug(f"Received action: {action}")
        
        return action

    def disconnect(self) -> None:
        """Disconnect the mock robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Mock SO101 robot disconnecting...")
        self._is_connected = False
        logger.info("Mock SO101 robot disconnected")

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the robot state."""
        return {
            "current_positions": self.current_positions.copy(),
            "target_positions": self.target_positions.copy(),
            "last_action": self.last_action.copy() if self.last_action else None,
            "is_connected": self.is_connected,
            "is_calibrated": self.is_calibrated,
        }