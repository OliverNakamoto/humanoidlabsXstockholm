#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation
import time
from typing import Tuple, Optional

class PhoneKalmanFilter:
    """
    Extended Kalman Filter for phone sensor fusion.

    State vector: [qw, qx, qy, qz, wx, wy, wz, ax, ay, az]
    - qw, qx, qy, qz: quaternion orientation
    - wx, wy, wz: angular velocity (rad/s)
    - ax, ay, az: linear acceleration (m/s²)
    """

    def __init__(self):
        # State dimension
        self.state_dim = 10

        # Initialize state: [qw, qx, qy, qz, wx, wy, wz, ax, ay, az]
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion (identity)

        # State covariance matrix
        self.P = np.eye(self.state_dim) * 0.1

        # Process noise covariance
        self.Q = np.eye(self.state_dim)
        self.Q[0:4, 0:4] *= 0.001   # Quaternion process noise
        self.Q[4:7, 4:7] *= 0.01    # Angular velocity process noise
        self.Q[7:10, 7:10] *= 0.1   # Acceleration process noise

        # Measurement noise covariance
        self.R_gyro = np.eye(3) * 0.01      # Gyroscope noise
        self.R_accel = np.eye(3) * 0.1      # Accelerometer noise
        self.R_mag = np.eye(3) * 0.05       # Magnetometer noise (if available)

        # Gravity vector
        self.gravity = np.array([0, 0, -9.81])

        # Last update time
        self.last_time = None

        # Calibration offsets
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.is_calibrated = False

    def normalize_quaternion(self):
        """Normalize the quaternion part of the state."""
        q_norm = np.linalg.norm(self.x[0:4])
        if q_norm > 0:
            self.x[0:4] /= q_norm

    def quaternion_to_euler(self, q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        qw, qx, qy, qz = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def predict(self, dt: float):
        """Prediction step of the Kalman filter."""
        # Current state
        q = self.x[0:4]  # quaternion
        w = self.x[4:7]  # angular velocity
        a = self.x[7:10] # acceleration

        # Predict quaternion using angular velocity
        # q_new = q + 0.5 * dt * quaternion_multiply(q, [0, wx, wy, wz])
        w_quat = np.array([0, w[0], w[1], w[2]])
        q_dot = 0.5 * self.quaternion_multiply(q, w_quat)
        q_new = q + dt * q_dot

        # Normalize quaternion
        q_new = q_new / np.linalg.norm(q_new)

        # Update state (angular velocity and acceleration assumed constant)
        self.x[0:4] = q_new
        # self.x[4:7] = w (unchanged)
        # self.x[7:10] = a (unchanged)

        # Jacobian of the process model
        F = np.eye(self.state_dim)

        # Quaternion derivatives with respect to angular velocity
        F[0:4, 4:7] = 0.5 * dt * np.array([
            [-q[1], -q[2], -q[3]],
            [ q[0], -q[3],  q[2]],
            [ q[3],  q[0], -q[1]],
            [-q[2],  q[1],  q[0]]
        ])

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

        # Normalize quaternion
        self.normalize_quaternion()

    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def update_gyro(self, gyro_raw: np.ndarray):
        """Update with gyroscope measurement."""
        # Apply bias correction
        gyro = gyro_raw - self.gyro_bias

        # Measurement function: h(x) = angular_velocity_from_state
        h = self.x[4:7]

        # Innovation
        y = gyro - h

        # Measurement Jacobian
        H = np.zeros((3, self.state_dim))
        H[0:3, 4:7] = np.eye(3)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gyro

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        # Normalize quaternion
        self.normalize_quaternion()

    def update_accel(self, accel_raw: np.ndarray):
        """Update with accelerometer measurement."""
        # Apply bias correction
        accel = accel_raw - self.accel_bias

        # Current quaternion
        q = self.x[0:4]

        # Expected gravity in body frame (measurement function)
        # Rotate gravity vector from world to body frame
        gravity_body = self.rotate_vector_by_quaternion_inverse(self.gravity, q)

        # Innovation (measured acceleration should equal rotated gravity + linear acceleration)
        expected_accel = gravity_body + self.x[7:10]
        y = accel - expected_accel

        # Measurement Jacobian
        H = np.zeros((3, self.state_dim))

        # Partial derivatives with respect to quaternion
        H[0:3, 0:4] = self.gravity_jacobian_wrt_quaternion(q)

        # Partial derivatives with respect to linear acceleration
        H[0:3, 7:10] = np.eye(3)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_accel

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        # Normalize quaternion
        self.normalize_quaternion()

    def rotate_vector_by_quaternion_inverse(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector by inverse of quaternion."""
        qw, qx, qy, qz = q
        vx, vy, vz = v

        # q^-1 * v * q where q^-1 = [qw, -qx, -qy, -qz] / |q|^2
        # Since quaternion is normalized, |q|^2 = 1
        return np.array([
            vx * (qw*qw + qx*qx - qy*qy - qz*qz) + 2*vy*(qx*qy + qw*qz) + 2*vz*(qx*qz - qw*qy),
            2*vx*(qx*qy - qw*qz) + vy*(qw*qw - qx*qx + qy*qy - qz*qz) + 2*vz*(qy*qz + qw*qx),
            2*vx*(qx*qz + qw*qy) + 2*vy*(qy*qz - qw*qx) + vz*(qw*qw - qx*qx - qy*qy + qz*qz)
        ])

    def gravity_jacobian_wrt_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Compute Jacobian of gravity rotation with respect to quaternion."""
        qw, qx, qy, qz = q
        gx, gy, gz = self.gravity

        # Partial derivatives of rotated gravity with respect to quaternion components
        J = np.zeros((3, 4))

        # df/dqw
        J[0, 0] = 2*(qw*gx + qz*gy - qy*gz)
        J[1, 0] = 2*(-qz*gx + qw*gy + qx*gz)
        J[2, 0] = 2*(qy*gx - qx*gy + qw*gz)

        # df/dqx
        J[0, 1] = 2*(qx*gx + qy*gy + qz*gz)
        J[1, 1] = 2*(qy*gx - qx*gy - qw*gz)
        J[2, 1] = 2*(qz*gx + qw*gy - qx*gz)

        # df/dqy
        J[0, 2] = 2*(-qy*gx + qx*gy + qw*gz)
        J[1, 2] = 2*(qx*gx + qy*gy + qz*gz)
        J[2, 2] = 2*(-qw*gx + qz*gy - qy*gz)

        # df/dqz
        J[0, 3] = 2*(-qz*gx - qw*gy + qx*gz)
        J[1, 3] = 2*(qw*gx - qz*gy + qy*gz)
        J[2, 3] = 2*(qx*gx + qy*gy + qz*gz)

        return J

    def calibrate(self, gyro_samples: list, accel_samples: list, duration: float = 2.0):
        """Calibrate sensor biases by averaging samples while stationary."""
        print(f"Calibrating for {duration} seconds...")

        if len(gyro_samples) > 10 and len(accel_samples) > 10:
            # Calculate bias as mean of samples (assuming stationary)
            self.gyro_bias = np.mean(gyro_samples, axis=0)

            # For accelerometer, bias is deviation from expected gravity
            accel_mean = np.mean(accel_samples, axis=0)
            expected_gravity_mag = 9.81
            actual_gravity_mag = np.linalg.norm(accel_mean)

            # Simple bias correction (more sophisticated methods exist)
            if actual_gravity_mag > 0:
                self.accel_bias = accel_mean - (accel_mean / actual_gravity_mag) * expected_gravity_mag

            self.is_calibrated = True
            print(f"Calibration complete:")
            print(f"  Gyro bias: {self.gyro_bias}")
            print(f"  Accel bias: {self.accel_bias}")
        else:
            print("Not enough samples for calibration")

    def update(self, gyro: np.ndarray, accel: np.ndarray, timestamp: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Update filter with new sensor measurements.

        Args:
            gyro: Angular velocity in rad/s [wx, wy, wz]
            accel: Linear acceleration in m/s² [ax, ay, az]
            timestamp: Timestamp of measurement (optional)

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        current_time = timestamp if timestamp is not None else time.time()

        if self.last_time is not None:
            dt = current_time - self.last_time
            dt = max(dt, 0.001)  # Minimum dt to prevent numerical issues

            # Prediction step
            self.predict(dt)

        # Update steps
        self.update_gyro(gyro)
        self.update_accel(accel)

        # Update angular velocity and acceleration estimates
        self.x[4:7] = gyro - self.gyro_bias
        self.x[7:10] = accel - self.accel_bias

        self.last_time = current_time

        # Convert quaternion to Euler angles
        return self.quaternion_to_euler(self.x[0:4])

    def get_state(self) -> dict:
        """Get current filter state."""
        roll, pitch, yaw = self.quaternion_to_euler(self.x[0:4])

        return {
            'quaternion': self.x[0:4].copy(),
            'angular_velocity': self.x[4:7].copy(),
            'acceleration': self.x[7:10].copy(),
            'euler': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
            'is_calibrated': self.is_calibrated,
            'covariance_trace': np.trace(self.P)
        }


class SimpleKalmanFilter:
    """
    Simplified Kalman filter for orientation tracking.
    State: [roll, pitch, yaw, gyro_x, gyro_y, gyro_z]
    """

    def __init__(self):
        # State: [roll, pitch, yaw, wx, wy, wz]
        self.x = np.zeros(6)

        # Covariance matrix
        self.P = np.eye(6) * 0.1

        # Process noise
        self.Q = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])

        # Measurement noise
        self.R_accel = np.diag([0.1, 0.1])  # Roll, pitch from accelerometer
        self.R_gyro = np.diag([0.01, 0.01, 0.01])  # Angular velocities

        self.last_time = None

    def predict(self, dt: float):
        """Predict step."""
        # State transition: angles integrate angular velocities
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

        # Keep angles in reasonable range
        self.x[0:3] = np.mod(self.x[0:3] + np.pi, 2*np.pi) - np.pi

    def update_accel(self, accel: np.ndarray):
        """Update with accelerometer (roll and pitch only)."""
        # Calculate roll and pitch from accelerometer
        ax, ay, az = accel

        accel_roll = np.arctan2(ay, az)
        accel_pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az))

        # Measurement
        z = np.array([accel_roll, accel_pitch])

        # Measurement function: H maps state to measurement
        H = np.zeros((2, 6))
        H[0, 0] = 1  # roll
        H[1, 1] = 1  # pitch

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_accel

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_gyro(self, gyro: np.ndarray):
        """Update with gyroscope."""
        # Measurement function: direct measurement of angular velocities
        H = np.zeros((3, 6))
        H[0:3, 3:6] = np.eye(3)

        # Innovation
        y = gyro - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gyro

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def update(self, gyro: np.ndarray, accel: np.ndarray, timestamp: Optional[float] = None) -> Tuple[float, float, float]:
        """Update with sensor measurements."""
        current_time = timestamp if timestamp is not None else time.time()

        if self.last_time is not None:
            dt = current_time - self.last_time
            dt = max(dt, 0.001)
            self.predict(dt)

        # Update with measurements
        self.update_accel(accel)
        self.update_gyro(gyro)

        self.last_time = current_time

        return self.x[0], self.x[1], self.x[2]  # roll, pitch, yaw

    def get_state(self) -> dict:
        """Get current state."""
        return {
            'euler': {'roll': self.x[0], 'pitch': self.x[1], 'yaw': self.x[2]},
            'angular_velocity': self.x[3:6].copy(),
            'covariance_trace': np.trace(self.P)
        }


def test_kalman_filter():
    """Test the Kalman filter with simulated data."""
    print("Testing Kalman Filter...")

    # Create filter
    kf = SimpleKalmanFilter()

    # Simulate some data
    dt = 0.01  # 100 Hz
    time_steps = 1000

    # True orientation (sinusoidal motion)
    true_roll = []
    true_pitch = []
    true_yaw = []

    for i in range(time_steps):
        t = i * dt
        roll = 0.5 * np.sin(0.5 * t)
        pitch = 0.3 * np.cos(0.3 * t)
        yaw = 0.2 * t

        true_roll.append(roll)
        true_pitch.append(pitch)
        true_yaw.append(yaw)

    # Simulate sensor data with noise
    estimated_angles = []

    for i in range(time_steps):
        # True angular velocities (derivatives)
        t = i * dt
        wx = 0.5 * 0.5 * np.cos(0.5 * t)  # d(roll)/dt
        wy = -0.3 * 0.3 * np.sin(0.3 * t)  # d(pitch)/dt
        wz = 0.2  # d(yaw)/dt

        # Add noise
        gyro_noise = np.random.normal(0, 0.01, 3)
        accel_noise = np.random.normal(0, 0.1, 3)

        gyro = np.array([wx, wy, wz]) + gyro_noise

        # Simulate accelerometer (gravity + noise)
        # Perfect accelerometer would measure rotated gravity
        roll_true = true_roll[i]
        pitch_true = true_pitch[i]

        # Gravity components in body frame
        ax = 9.81 * np.sin(pitch_true)
        ay = -9.81 * np.cos(pitch_true) * np.sin(roll_true)
        az = -9.81 * np.cos(pitch_true) * np.cos(roll_true)

        accel = np.array([ax, ay, az]) + accel_noise

        # Update filter
        roll_est, pitch_est, yaw_est = kf.update(gyro, accel, t)
        estimated_angles.append([roll_est, pitch_est, yaw_est])

        if i % 100 == 0:
            print(f"Step {i}: True=[{true_roll[i]:.3f}, {true_pitch[i]:.3f}, {true_yaw[i]:.3f}], "
                  f"Est=[{roll_est:.3f}, {pitch_est:.3f}, {yaw_est:.3f}]")

    # Calculate RMS error
    estimated_angles = np.array(estimated_angles)
    true_angles = np.array([true_roll, true_pitch, true_yaw]).T

    rms_error = np.sqrt(np.mean((estimated_angles - true_angles)**2, axis=0))
    print(f"\nRMS Errors: Roll={rms_error[0]:.4f}, Pitch={rms_error[1]:.4f}, Yaw={rms_error[2]:.4f}")

    print("Kalman filter test completed!")


if __name__ == "__main__":
    test_kalman_filter()