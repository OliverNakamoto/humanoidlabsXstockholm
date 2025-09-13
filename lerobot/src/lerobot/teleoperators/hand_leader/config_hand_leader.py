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

from dataclasses import dataclass, field
from typing import List

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("hand_leader")
@dataclass
class HandLeaderConfig(TeleoperatorConfig):
    # Camera index for OpenCV (default 0 for primary camera)
    camera_index: int = 0
    
    # Path to robot URDF file for kinematics
    urdf_path: str = "path/to/robot.urdf"
    
    # Target frame name in URDF for end effector
    target_frame_name: str = "gripper_frame_link"
    
    # Joint names for IK solver (in order)
    joint_names: List[str] = field(default_factory=lambda: [
        "shoulder_pan",
        "shoulder_lift", 
        "elbow_flex",
        "wrist_flex",
        "wrist_roll"
    ])
    
    # Whether to use degrees (True) or normalized range (False)
    use_degrees: bool = False
    
    # Control inversions
    invert_y: bool = True        # Move hand down -> robot down
    invert_pinch: bool = True    # Pinch closes gripper


@TeleoperatorConfig.register_subclass("hand_leader_simple")
@dataclass
class HandLeaderSimpleConfig(TeleoperatorConfig):
    # Camera index for OpenCV (default 0 for primary camera)
    camera_index: int = 0
