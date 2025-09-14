#!/usr/bin/env python

import time
import numpy as np
from typing import Dict, Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_phone_leader import PhoneLeaderConfig
from .phone_imu_ipc import PhoneIMUIPC


def _rpy_deg_to_matrix(roll, pitch, yaw):
    # ZYX convention (yaw around Z, pitch around Y, roll around X)
    rz = np.array([[np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw)), 0],
                   [np.sin(np.deg2rad(yaw)),  np.cos(np.deg2rad(yaw)), 0],
                   [0, 0, 1]])
    ry = np.array([[ np.cos(np.deg2rad(pitch)), 0, np.sin(np.deg2rad(pitch))],
                   [ 0, 1, 0],
                   [-np.sin(np.deg2rad(pitch)), 0, np.cos(np.deg2rad(pitch))]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(np.deg2rad(roll)), -np.sin(np.deg2rad(roll))],
                   [0, np.sin(np.deg2rad(roll)),  np.cos(np.deg2rad(roll))]])
    return rz @ ry @ rx


class PhoneLeader(Teleoperator):
    """Teleoperator using phone orientation + velocity inputs over HTTP."""

    config_class = PhoneLeaderConfig
    name = "phone_leader"

    def __init__(self, config: PhoneLeaderConfig):
        super().__init__(config)
        self.config = config
        self.ipc = PhoneIMUIPC(server_url=config.server_url)
        self._is_connected = False

        # Target position state (meters)
        self._pos = np.array([config.center_x, config.center_y, config.center_z], dtype=float)
        self._last_t = time.perf_counter()

        # Optional IK
        self._kin = None
        if config.use_ik and config.urdf_path:
            try:
                from lerobot.model.kinematics import RobotKinematics
                self._kin = RobotKinematics(config.urdf_path, config.target_frame_name)
            except Exception:
                self._kin = None

        # Simple smoothing
        self._prev_action: Dict[str, float] | None = None
        self._deadband = 1.0

        # Simple 1D Kalman filters for orientation and velocity (per-axis)
        # x_k = x_{k-1}  (random walk); update with measurement z
        class _KF:
            def __init__(self, q=0.5, r=4.0, x0=0.0):  # process and measurement noise (deg^2)
                self.q = float(q)
                self.r = float(r)
                self.x = float(x0)
                self.p = 1.0
            def update(self, z):
                # predict
                self.p = self.p + self.q
                # update
                k = self.p / (self.p + self.r)
                self.x = self.x + k * (float(z) - self.x)
                self.p = (1.0 - k) * self.p
                return self.x

        self._kf_yaw = _KF(q=0.5, r=8.0)
        self._kf_pitch = _KF(q=0.5, r=8.0)
        self._kf_roll = _KF(q=0.5, r=8.0)
        # velocities are slider-controlled; still lightly filter to avoid jitter
        self._kf_vx = _KF(q=0.05, r=0.5)
        self._kf_vy = _KF(q=0.05, r=0.5)
        self._kf_vz = _KF(q=0.05, r=0.5)

    @property
    def action_features(self) -> dict[str, type]:
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
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.ipc.connect()
        self._is_connected = True

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.ipc.disconnect()
        self._is_connected = False

    def _clamp_workspace(self, p: np.ndarray) -> np.ndarray:
        p[0] = np.clip(p[0], self.config.min_x, self.config.max_x)
        p[1] = np.clip(p[1], self.config.min_y, self.config.max_y)
        p[2] = np.clip(p[2], self.config.min_z, self.config.max_z)
        return p

    def _simple_map(self, pos: np.ndarray, yaw: float, pitch: float, roll: float) -> Dict[str, float]:
        # Map yaw to shoulder_pan
        pan = np.clip((yaw / self.config.yaw_range_deg) * 100.0, -100.0, 100.0)
        # Map position around center to lift/elbow
        dx = pos[0] - self.config.center_x
        dy = self.config.center_y - pos[1]  # invert so down phone moves arm down
        dz = pos[2] - self.config.center_z
        sx = 200.0; sy = 180.0; sz = 120.0
        lift = np.clip(dy * sy, -100.0, 100.0)
        elbow = np.clip(-dz * sz, -100.0, 100.0)
        # Wrist from phone roll/pitch
        wrist_roll = np.clip((roll / self.config.roll_range_deg) * 100.0, -100.0, 100.0)
        wrist_flex = np.clip((pitch / self.config.pitch_range_deg) * 100.0, -100.0, 100.0)
        return {
            "shoulder_pan.pos": float(pan),
            "shoulder_lift.pos": float(lift),
            "elbow_flex.pos": float(elbow),
            "wrist_flex.pos": float(wrist_flex),
            "wrist_roll.pos": float(wrist_roll),
        }

    def get_action(self, current_pos: Dict[str, float] = None) -> Dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        now = time.perf_counter()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now

        data = self.ipc.get()
        yaw = self._kf_yaw.update(data.get("yaw", 0.0))
        pitch = self._kf_pitch.update(data.get("pitch", 0.0))
        roll = self._kf_roll.update(data.get("roll", 0.0))
        vx = self._kf_vx.update(data.get("vx", 0.0))
        vy = self._kf_vy.update(data.get("vy", 0.0))
        vz = self._kf_vz.update(data.get("vz", 0.0))
        grip = float(data.get("gripper", 50.0))

        # Integrate velocity into target position
        self._pos += np.array([vx, vy, vz]) * dt * self.config.vel_gain
        self._pos = self._clamp_workspace(self._pos)

        if self._kin is not None:
            # Build pose and solve IK
            T = np.eye(4)
            T[:3, :3] = _rpy_deg_to_matrix(roll, pitch, yaw)
            T[0, 3], T[1, 3], T[2, 3] = self._pos.tolist()

            # Seed from current_pos if given
            seed = []
            if current_pos is not None:
                # Order is robot-dependent; we assume same as phone mapping order
                for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]:
                    key = f"{name}.pos"
                    seed.append(float(current_pos.get(key, 0.0)))
            else:
                seed = [0, 0, 0, 0, 0]

            # Convert seed [-100..100] to degrees; simple linear map in [-90,90] except wrist_roll [-180,180]
            deg_seed = np.array([
                seed[0] * 0.9,   # ~[-90,90]
                seed[1] * 0.9,
                seed[2] * 0.9,
                seed[3] * 0.9,
                seed[4] * 1.8,   # ~[-180,180]
            ], dtype=float)

            try:
                sol_deg = self._kin.inverse_kinematics(deg_seed, T, position_weight=1.0, orientation_weight=0.1)
                # Map degrees back to normalized
                action = {
                    "shoulder_pan.pos": float(np.clip(sol_deg[0] / 0.9, -100.0, 100.0)),
                    "shoulder_lift.pos": float(np.clip(sol_deg[1] / 0.9, -100.0, 100.0)),
                    "elbow_flex.pos": float(np.clip(sol_deg[2] / 0.9, -100.0, 100.0)),
                    "wrist_flex.pos": float(np.clip(sol_deg[3] / 0.9, -100.0, 100.0)),
                    "wrist_roll.pos": float(np.clip(sol_deg[4] / 1.8, -100.0, 100.0)),
                }
            except Exception:
                action = self._simple_map(self._pos, yaw, pitch, roll)
        else:
            action = self._simple_map(self._pos, yaw, pitch, roll)

        # Gripper mapping
        action["gripper.pos"] = float(100.0 - grip) if self.config.invert_pinch else float(grip)

        # Deadband to reduce micro jitter
        if self._prev_action is not None:
            for k in action.keys():
                prev = self._prev_action.get(k, action[k])
                if abs(action[k] - prev) < self._deadband:
                    action[k] = prev
        self._prev_action = dict(action)
        return action
