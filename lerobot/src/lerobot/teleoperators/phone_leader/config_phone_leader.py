#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("phone_leader")
@dataclass
class PhoneLeaderConfig(TeleoperatorConfig):
    # Phone IMU server URL
    server_url: str = "http://localhost:8899"

    # Workspace center and bounds (meters)
    center_x: float = 0.0
    center_y: float = 0.2
    center_z: float = 0.15
    min_x: float = -0.3
    max_x: float = 0.3
    min_y: float = 0.05
    max_y: float = 0.45
    min_z: float = 0.05
    max_z: float = 0.40

    # Velocity integration gain
    # target_pos += [vx,vy,vz] * dt * vel_gain
    vel_gain: float = 1.0

    # Orientation mapping (degrees)
    # yaw -> shoulder_pan, pitch/roll -> EE orientation (if IK), otherwise wrist joints
    yaw_range_deg: float = 60.0
    roll_range_deg: float = 90.0
    pitch_range_deg: float = 90.0

    # Optional IK mode using placo RobotKinematics if available
    use_ik: bool = False
    urdf_path: str = ""
    target_frame_name: str = "gripper_frame_link"
    invert_pinch: bool = True

