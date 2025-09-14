#!/usr/bin/env python

"""
Kalman Filter Playground for Phone Gyroscope Data

This playground allows you to test different Kalman filter configurations
with real phone gyroscope and accelerometer data to improve robot control.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
import time
import threading
from collections import deque
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Extended Kalman Filter for sensor fusion of gyroscope and accelerometer data.

    State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]

    This filter fuses:
    - Gyroscope data (angular velocities) - high frequency, drifts over time
    - Accelerometer data (orientation estimates) - low frequency, noisy but stable
    """

    def __init__(self, dt=0.01):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step in seconds
        """
        self.dt = dt

        # State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.x = np.zeros(6)

        # State covariance matrix
        self.P = np.eye(6) * 0.1

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        # Process noise covariance
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]) * dt

        # Measurement noise covariance
        self.R_gyro = np.diag([0.1, 0.1, 0.1])  # Gyroscope noise
        self.R_accel = np.diag([0.5, 0.5])      # Accelerometer noise (roll, pitch only)

    def predict(self):
        """Prediction step using gyroscope data"""
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gyro(self, gyro_rates: np.ndarray):
        """
        Update step using gyroscope measurements.

        Args:
            gyro_rates: [roll_rate, pitch_rate, yaw_rate] in rad/s
        """
        # Measurement matrix for gyroscope (measures angular rates)
        H = np.array([
            [0, 0, 0, 1, 0, 0],  # roll_rate
            [0, 0, 0, 0, 1, 0],  # pitch_rate
            [0, 0, 0, 0, 0, 1]   # yaw_rate
        ])

        # Innovation
        z = gyro_rates
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gyro

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P

    def update_accel(self, accel_orientation: np.ndarray):
        """
        Update step using accelerometer-derived orientation.

        Args:
            accel_orientation: [roll, pitch] in radians from accelerometer
        """
        # Measurement matrix for accelerometer (measures roll and pitch)
        H = np.array([
            [1, 0, 0, 0, 0, 0],  # roll
            [0, 1, 0, 0, 0, 0]   # pitch
        ])

        # Innovation
        z = accel_orientation
        y = z - H @ self.x

        # Handle angle wrapping
        y[0] = self._wrap_angle(y[0])
        y[1] = self._wrap_angle(y[1])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_accel

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P

    def get_orientation(self) -> Tuple[float, float, float]:
        """Get current orientation estimate (roll, pitch, yaw) in radians"""
        return self.x[0], self.x[1], self.x[2]

    def get_angular_rates(self) -> Tuple[float, float, float]:
        """Get current angular rate estimate (roll_rate, pitch_rate, yaw_rate) in rad/s"""
        return self.x[3], self.x[4], self.x[5]

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi


class PhoneDataCollector:
    """Collects real-time data from phone gyro server"""

    def __init__(self, server_url="http://localhost:8889", buffer_size=1000):
        self.server_url = server_url
        self.buffer_size = buffer_size

        # Data buffers
        self.timestamps = deque(maxlen=buffer_size)
        self.gyro_data = deque(maxlen=buffer_size)
        self.accel_data = deque(maxlen=buffer_size)
        self.raw_orientation = deque(maxlen=buffer_size)
        self.filtered_orientation = deque(maxlen=buffer_size)

        # Threading
        self.collecting = False
        self.collection_thread = None

        # Kalman filter
        self.kalman = KalmanFilter(dt=0.01)

        # Session for requests
        self.session = requests.Session()

    def start_collection(self):
        """Start collecting data from phone"""
        if self.collecting:
            return

        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started phone data collection")

    def stop_collection(self):
        """Stop collecting data"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
        logger.info("Stopped phone data collection")

    def _collection_loop(self):
        """Main collection loop"""
        last_time = time.time()

        while self.collecting:
            try:
                # Get data from phone server
                response = self.session.get(f"{self.server_url}/status", timeout=0.1)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('valid'):
                        current_time = time.time()
                        dt = current_time - last_time

                        # Extract data
                        gyro = np.array([
                            np.radians(data.get('gyro', {}).get('x', 0.0)),
                            np.radians(data.get('gyro', {}).get('y', 0.0)),
                            np.radians(data.get('gyro', {}).get('z', 0.0))
                        ])

                        accel = np.array([
                            data.get('accel', {}).get('x', 0.0),
                            data.get('accel', {}).get('y', 0.0),
                            data.get('accel', {}).get('z', 0.0)
                        ])

                        # Calculate orientation from accelerometer
                        accel_roll = np.arctan2(accel[1], accel[2])
                        accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
                        accel_orientation = np.array([accel_roll, accel_pitch])

                        raw_ori = np.array([
                            np.radians(data['orientation']['roll']),
                            np.radians(data['orientation']['pitch']),
                            np.radians(data['orientation']['yaw'])
                        ])

                        # Update Kalman filter
                        if dt > 0:
                            self.kalman.dt = dt
                            self.kalman.predict()
                            self.kalman.update_gyro(gyro)
                            self.kalman.update_accel(accel_orientation)

                        # Store data
                        self.timestamps.append(current_time)
                        self.gyro_data.append(gyro)
                        self.accel_data.append(accel)
                        self.raw_orientation.append(raw_ori)
                        self.filtered_orientation.append(np.array(self.kalman.get_orientation()))

                        last_time = current_time

            except Exception as e:
                logger.debug(f"Collection error: {e}")

            time.sleep(0.01)  # 100Hz collection rate


class KalmanPlayground:
    """Interactive playground for testing Kalman filter parameters"""

    def __init__(self):
        self.collector = PhoneDataCollector()

        # Setup matplotlib
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 10))
        self.fig.suptitle('Kalman Filter Playground - Phone Gyroscope Data', fontsize=16)

        # Plot setup
        self.setup_plots()

        # Animation
        self.ani = None

    def setup_plots(self):
        """Setup all subplots"""
        # Row 1: Orientation comparison
        self.axes[0, 0].set_title('Roll Angle')
        self.axes[0, 0].set_ylabel('Angle (degrees)')
        self.axes[0, 0].legend(['Raw', 'Kalman Filtered'], loc='upper right')
        self.axes[0, 0].grid(True)

        self.axes[0, 1].set_title('Pitch Angle')
        self.axes[0, 1].set_ylabel('Angle (degrees)')
        self.axes[0, 1].legend(['Raw', 'Kalman Filtered'], loc='upper right')
        self.axes[0, 1].grid(True)

        # Row 2: Angular rates
        self.axes[1, 0].set_title('Roll Rate')
        self.axes[1, 0].set_ylabel('Rate (deg/s)')
        self.axes[1, 0].legend(['Gyroscope', 'Kalman Estimate'], loc='upper right')
        self.axes[1, 0].grid(True)

        self.axes[1, 1].set_title('Pitch Rate')
        self.axes[1, 1].set_ylabel('Rate (deg/s)')
        self.axes[1, 1].legend(['Gyroscope', 'Kalman Estimate'], loc='upper right')
        self.axes[1, 1].grid(True)

        # Row 3: Motor outputs and noise analysis
        self.axes[2, 0].set_title('Motor 3 (Pitch) & Motor 4 (Roll) Commands')
        self.axes[2, 0].set_ylabel('Motor Command (-100 to 100)')
        self.axes[2, 0].legend(['Motor 3 (Pitch)', 'Motor 4 (Roll)'], loc='upper right')
        self.axes[2, 0].grid(True)

        self.axes[2, 1].set_title('Kalman Filter Uncertainty')
        self.axes[2, 1].set_ylabel('Standard Deviation')
        self.axes[2, 1].legend(['Roll Uncertainty', 'Pitch Uncertainty'], loc='upper right')
        self.axes[2, 1].grid(True)

    def update_plots(self, frame):
        """Update all plots with latest data"""
        if len(self.collector.timestamps) < 10:
            return

        # Get recent data
        n_points = min(200, len(self.collector.timestamps))
        times = np.array(list(self.collector.timestamps))[-n_points:]
        raw_ori = np.array(list(self.collector.raw_orientation))[-n_points:]
        filt_ori = np.array(list(self.collector.filtered_orientation))[-n_points:]
        gyro = np.array(list(self.collector.gyro_data))[-n_points:]

        # Normalize time
        times = times - times[0]

        # Clear and update plots
        for ax in self.axes.flat:
            ax.clear()
        self.setup_plots()

        # Plot orientation comparison
        self.axes[0, 0].plot(times, np.degrees(raw_ori[:, 0]), 'r-', alpha=0.7, label='Raw')
        self.axes[0, 0].plot(times, np.degrees(filt_ori[:, 0]), 'b-', linewidth=2, label='Kalman')
        self.axes[0, 0].set_title('Roll Angle')
        self.axes[0, 0].legend()

        self.axes[0, 1].plot(times, np.degrees(raw_ori[:, 1]), 'r-', alpha=0.7, label='Raw')
        self.axes[0, 1].plot(times, np.degrees(filt_ori[:, 1]), 'b-', linewidth=2, label='Kalman')
        self.axes[0, 1].set_title('Pitch Angle')
        self.axes[0, 1].legend()

        # Plot angular rates
        kalman_rates = np.array([self.collector.kalman.get_angular_rates() for _ in range(len(times))])

        self.axes[1, 0].plot(times, np.degrees(gyro[:, 0]), 'g-', alpha=0.7, label='Gyro')
        self.axes[1, 0].plot(times, np.degrees(kalman_rates[:, 0]), 'b-', linewidth=2, label='Kalman')
        self.axes[1, 0].set_title('Roll Rate')
        self.axes[1, 0].legend()

        self.axes[1, 1].plot(times, np.degrees(gyro[:, 1]), 'g-', alpha=0.7, label='Gyro')
        self.axes[1, 1].plot(times, np.degrees(kalman_rates[:, 1]), 'b-', linewidth=2, label='Kalman')
        self.axes[1, 1].set_title('Pitch Rate')
        self.axes[1, 1].legend()

        # Calculate motor commands (pitch -> motor 3, roll -> motor 4)
        motor3_cmd = np.clip(np.degrees(filt_ori[:, 1]) * 2, -100, 100)  # Pitch -> Motor 3
        motor4_cmd = np.clip(np.degrees(filt_ori[:, 0]) * 2, -100, 100)  # Roll -> Motor 4

        self.axes[2, 0].plot(times, motor3_cmd, 'purple', linewidth=2, label='Motor 3 (Pitch)')
        self.axes[2, 0].plot(times, motor4_cmd, 'orange', linewidth=2, label='Motor 4 (Roll)')
        self.axes[2, 0].set_title('Motor Commands')
        self.axes[2, 0].legend()

        # Plot uncertainty
        P_diag = np.sqrt(np.diag(self.collector.kalman.P))
        uncertainty_data = np.array([[P_diag[0], P_diag[1]] for _ in range(len(times))])

        self.axes[2, 1].plot(times, np.degrees(uncertainty_data[:, 0]), 'r-', label='Roll σ')
        self.axes[2, 1].plot(times, np.degrees(uncertainty_data[:, 1]), 'b-', label='Pitch σ')
        self.axes[2, 1].set_title('Filter Uncertainty')
        self.axes[2, 1].legend()

        # Set common properties
        for ax in self.axes.flat:
            ax.grid(True)
            ax.set_xlabel('Time (s)')

    def start(self):
        """Start the playground"""
        print("🎮 Kalman Filter Playground")
        print("=" * 50)
        print("1. Make sure phone gyro server is running: python phone_gyro_server.py")
        print("2. Open http://localhost:8889 on your phone and tap 'Start Control'")
        print("3. Move your phone to see real-time filtering!")
        print("\n📊 Plots:")
        print("• Top row: Raw vs Kalman filtered orientation")
        print("• Middle row: Gyroscope vs Kalman estimated rates")
        print("• Bottom left: Motor 3 (pitch) & Motor 4 (roll) commands")
        print("• Bottom right: Kalman filter uncertainty")
        print("\n🎯 Goal: Tune filter to reduce noise while maintaining responsiveness")

        # Start data collection
        self.collector.start_collection()

        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=50, blit=False)

        try:
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.collector.stop_collection()


def main():
    """Run the Kalman filter playground"""
    logging.basicConfig(level=logging.INFO)

    playground = KalmanPlayground()
    playground.start()


if __name__ == "__main__":
    main()