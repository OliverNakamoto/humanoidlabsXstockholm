# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
from typing import Any
from pathlib import Path

import numpy as np
import rerun as rr


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit, port=9902)


def log_robot_urdf(robot_name: str, urdf_path: str) -> bool:
    """Log robot URDF to Rerun for 3D visualization.
    
    Args:
        robot_name: Name of the robot for logging path
        urdf_path: Path to the URDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        urdf_file = Path(urdf_path)
        if not urdf_file.exists():
            print(f"URDF file not found: {urdf_path}")
            return False
            
        # Read URDF content
        with open(urdf_file, 'r') as f:
            urdf_content = f.read()
        
        # Log the URDF as text asset and as 3D representation
        rr.log(f"robot/{robot_name}/urdf", rr.TextDocument(urdf_content, media_type=rr.MediaType.XML))
        
        # For actual 3D visualization, we'd need to parse the URDF and create meshes
        # For now, we'll create a simple representation
        print(f"Logged URDF for {robot_name} from {urdf_path}")
        return True
        
    except Exception as e:
        print(f"Failed to log URDF: {e}")
        return False


def log_robot_state(robot_name: str, joint_positions: dict[str, float]):
    """Log robot joint states for 3D visualization.
    
    Args:
        robot_name: Name of the robot
        joint_positions: Dictionary of joint names to positions (in radians)
    """
    try:
        # Log individual joint positions as scalars
        for joint_name, position in joint_positions.items():
            clean_name = joint_name.replace('.pos', '')
            rr.log(f"robot/{robot_name}/joints/{clean_name}", rr.Scalar(float(position)))
        
        # Create a simple 3D representation of the robot
        # This is a simplified visualization - for full 3D robot, we'd need URDF parsing
        _log_simple_robot_visualization(robot_name, joint_positions)
        
    except Exception as e:
        print(f"Failed to log robot state: {e}")


def _log_simple_robot_visualization(robot_name: str, joint_positions: dict[str, float]):
    """Create a simple 3D visualization of the robot arm."""
    
    # Extract joint angles, converting from normalized values to radians if needed
    joints = {}
    for key, value in joint_positions.items():
        joint_name = key.replace('.pos', '')
        if joint_name in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']:
            # Convert from normalized (-100 to 100) to radians (-π/2 to π/2)
            joints[joint_name] = (value / 100.0) * (np.pi / 2)
        elif joint_name == 'gripper':
            # Gripper is typically 0-100, keep as is
            joints[joint_name] = value / 100.0
    
    # Simple forward kinematics for visualization
    # Base position
    base_pos = np.array([0.0, 0.0, 0.0])
    
    # Link lengths (approximated from URDF)
    link_lengths = {
        'shoulder_pan': 0.05,
        'shoulder_lift': 0.08,
        'upper_arm': 0.15,
        'forearm': 0.12,
        'wrist': 0.08
    }
    
    # Calculate link positions using simple forward kinematics
    positions = []
    current_pos = base_pos.copy()
    
    # Base
    positions.append(current_pos.copy())
    
    # Shoulder pan (rotation around Z)
    current_pos[2] += link_lengths['shoulder_pan']
    positions.append(current_pos.copy())
    
    # Shoulder lift (rotation around Y, then move along new Z)
    shoulder_lift_angle = joints.get('shoulder_lift', 0)
    current_pos[0] += link_lengths['upper_arm'] * np.sin(shoulder_lift_angle)
    current_pos[2] += link_lengths['upper_arm'] * np.cos(shoulder_lift_angle)
    positions.append(current_pos.copy())
    
    # Elbow flex
    elbow_angle = joints.get('elbow_flex', 0) + shoulder_lift_angle
    current_pos[0] += link_lengths['forearm'] * np.sin(elbow_angle)
    current_pos[2] += link_lengths['forearm'] * np.cos(elbow_angle)
    positions.append(current_pos.copy())
    
    # Wrist
    wrist_flex_angle = joints.get('wrist_flex', 0) + elbow_angle
    current_pos[0] += link_lengths['wrist'] * np.sin(wrist_flex_angle)
    current_pos[2] += link_lengths['wrist'] * np.cos(wrist_flex_angle)
    positions.append(current_pos.copy())
    
    # Log the robot arm as connected line segments
    positions_array = np.array(positions)
    
    # Create line segments for the robot arm
    for i in range(len(positions) - 1):
        segment_name = f"robot/{robot_name}/arm/segment_{i}"
        points = np.array([positions[i], positions[i + 1]])
        rr.log(segment_name, rr.LineStrips3D([points], colors=[0xFF0000FF], radii=[0.01]))
    
    # Log joint positions as spheres
    for i, pos in enumerate(positions):
        joint_name = f"robot/{robot_name}/arm/joint_{i}"
        rr.log(joint_name, rr.Points3D([pos], colors=[0x00FF00FF], radii=[0.015]))


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any], robot_name: str = None):
    """Enhanced logging that includes robot visualization."""
    
    # Log scalar observations and actions as before
    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation.{obs}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"observation.{obs}_{i}", rr.Scalar(float(v)))
            else:
                rr.log(f"observation.{obs}", rr.Image(val), static=True)
    
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}_{i}", rr.Scalar(float(v)))
    
    # Log robot state if robot name is provided
    if robot_name and action:
        log_robot_state(robot_name, action)
