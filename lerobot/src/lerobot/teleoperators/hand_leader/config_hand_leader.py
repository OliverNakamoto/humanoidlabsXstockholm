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


@TeleoperatorConfig.register_subclass("hand_leader_ipc")
@dataclass
class HandLeaderIPCConfig(TeleoperatorConfig):
    """Configuration for Hand Leader with IPC (MediaPipe in separate process)."""
    
    # Camera index for MediaPipe tracking
    camera_index: int = 0
    
    # Unix socket path for IPC communication
    socket_path: str = "/tmp/lerobot_hand_tracking.sock"
    
    # Path to robot URDF file for kinematics (optional)
    urdf_path: str = ""
    
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
    
    # Whether to show the live tracking visualization window
    show_window: bool = False

    # Workspace mapping configuration
    enable_startup_mapping: bool = True  # Run workspace mapping at startup
    startup_mapping_duration: int = 300   # Duration in seconds for startup mapping