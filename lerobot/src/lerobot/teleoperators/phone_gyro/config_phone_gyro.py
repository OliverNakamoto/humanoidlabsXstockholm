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

from dataclasses import dataclass
from pathlib import Path

from ..teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("phone_gyro")
@dataclass
class PhoneGyroConfig(TeleoperatorConfig):
    """Configuration for phone gyroscope teleoperator"""

    # URL of the phone gyro server
    server_url: str = "http://localhost:8889"

    # URDF path for inverse kinematics
    urdf_path: str | Path = "path/to/robot.urdf"

    # Target frame for IK (end effector)
    target_frame_name: str = "gripper_frame_link"

    # Joint names to control
    joint_names: list[str] = None

    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll"
            ]

    # Whether to use degrees (True) or normalized range (False)
    use_degrees: bool = False

    # Control sensitivity
    position_scale: float = 1.0
    orientation_scale: float = 1.0