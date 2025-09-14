#!/usr/bin/env python

"""
Phone Gyroscope Server with Kalman Filtering

Enhanced version of the phone gyro server that includes a Kalman filter
for better sensor fusion of gyroscope and accelerometer data.
"""

import logging
import json
import time
import threading
from typing import Dict, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Extended Kalman Filter for gyroscope and accelerometer fusion"""

    def __init__(self, dt=0.01):
        self.dt = dt

        # State: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1

        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]) * dt

        # Measurement noise
        self.R_gyro = np.diag([0.1, 0.1, 0.1])
        self.R_accel = np.diag([0.3, 0.3])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gyro(self, gyro_rates):
        H = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        y = gyro_rates - H @ self.x
        S = H @ self.P @ H.T + self.R_gyro
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P

    def update_accel(self, accel_orientation):
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        y = accel_orientation - H @ self.x
        y[0] = self._wrap_angle(y[0])
        y[1] = self._wrap_angle(y[1])

        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P

    def get_orientation(self):
        return self.x[0], self.x[1], self.x[2]

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


class PhoneGyroData:
    """Thread-safe storage for phone gyroscope data with Kalman filtering"""

    def __init__(self):
        self.lock = threading.Lock()
        self.use_kalman = True  # Enable/disable Kalman filtering
        self.kalman = KalmanFilter(dt=0.01)
        self.reset()

    def reset(self):
        """Reset all gyro data to defaults"""
        with self.lock:
            # Position control (integrated from gyro rates)
            self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}

            # Raw orientation from phone
            self.raw_orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

            # Filtered orientation from Kalman filter
            self.filtered_orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

            # Raw sensor data
            self.gyro = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}

            # Control parameters
            self.valid = False
            self.timestamp = time.time()
            self.calibrated = False
            self.reference_orientation = None

            # Reset Kalman filter
            self.kalman = KalmanFilter(dt=0.01)

    def set_kalman_enabled(self, enabled: bool):
        """Enable or disable Kalman filtering"""
        with self.lock:
            self.use_kalman = enabled
            if enabled:
                # Reset filter when enabling
                self.kalman = KalmanFilter(dt=0.01)

    def update(self, gyro_data: dict):
        """Update gyroscope data from phone with Kalman filtering"""
        with self.lock:
            current_time = time.time()
            dt = current_time - self.timestamp

            if 'gyroscope' in gyro_data:
                self.gyro = gyro_data['gyroscope']

            if 'accelerometer' in gyro_data:
                self.accel = gyro_data['accelerometer']

            if 'orientation' in gyro_data:
                self.raw_orientation = gyro_data['orientation']

            # Apply Kalman filtering if enabled
            if self.use_kalman and dt > 0:
                # Update Kalman filter timing
                self.kalman.dt = min(dt, 0.1)  # Cap dt to prevent instability

                # Predict step
                self.kalman.predict()

                # Update with gyroscope data
                gyro_rates = np.array([
                    np.radians(self.gyro['x']),
                    np.radians(self.gyro['y']),
                    np.radians(self.gyro['z'])
                ])
                self.kalman.update_gyro(gyro_rates)

                # Update with accelerometer-derived orientation
                accel_roll = np.arctan2(self.accel['y'], self.accel['z'])
                accel_pitch = np.arctan2(-self.accel['x'],
                                       np.sqrt(self.accel['y']**2 + self.accel['z']**2))
                accel_orientation = np.array([accel_roll, accel_pitch])
                self.kalman.update_accel(accel_orientation)

                # Get filtered orientation
                roll, pitch, yaw = self.kalman.get_orientation()
                self.filtered_orientation = {
                    'roll': np.degrees(roll),
                    'pitch': np.degrees(pitch),
                    'yaw': np.degrees(yaw)
                }
            else:
                # Use raw orientation if Kalman is disabled
                self.filtered_orientation = self.raw_orientation.copy()

            # Integrate gyroscope for position control
            if dt > 0 and self.calibrated:
                position_scale = 0.01
                self.position['x'] += self.gyro['x'] * dt * position_scale
                self.position['y'] += self.gyro['z'] * dt * position_scale
                self.position['z'] += -self.gyro['y'] * dt * position_scale

                # Keep within workspace bounds
                self.position['x'] = np.clip(self.position['x'], -0.3, 0.3)
                self.position['y'] = np.clip(self.position['y'], 0.1, 0.4)
                self.position['z'] = np.clip(self.position['z'], 0.05, 0.3)

            self.timestamp = current_time
            self.valid = True

    def get_current_pose(self) -> Tuple[float, float, float, float, float, float]:
        """Get current pose using filtered orientation"""
        with self.lock:
            if not self.valid:
                return (0.0, 0.2, 0.15, 0.0, 0.0, 0.0)

            return (
                self.position['x'],
                self.position['y'],
                self.position['z'],
                self.filtered_orientation['roll'],
                self.filtered_orientation['pitch'],
                self.filtered_orientation['yaw']
            )

    def get_motor_commands(self) -> Dict[str, float]:
        """Get motor commands with pitch->motor3, roll->motor4 mapping"""
        with self.lock:
            if not self.valid:
                return {f"motor_{i}": 0.0 for i in range(5)}

            # Map orientation to motor commands
            pitch_deg = self.filtered_orientation['pitch']
            roll_deg = self.filtered_orientation['roll']

            return {
                "motor_0": np.clip(self.position['x'] * 200, -100, 100),  # X position
                "motor_1": np.clip((self.position['y'] - 0.2) * 200, -100, 100),  # Y position
                "motor_2": np.clip((self.position['z'] - 0.15) * 200, -100, 100),  # Z position
                "motor_3": np.clip(pitch_deg * 2, -100, 100),  # Pitch -> Motor 3
                "motor_4": np.clip(roll_deg * 2, -100, 100),   # Roll -> Motor 4
            }

    def calibrate(self):
        """Calibrate the gyroscope"""
        with self.lock:
            self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}
            self.reference_orientation = self.filtered_orientation.copy()
            self.calibrated = True
            logger.info("Phone gyroscope calibrated with Kalman filter")


class PhoneGyroHandler(BaseHTTPRequestHandler):
    """HTTP request handler with Kalman filter support"""

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self.serve_phone_interface()
        elif parsed_path.path == '/status':
            self.serve_status()
        elif parsed_path.path == '/motor_commands':
            self.serve_motor_commands()
        elif parsed_path.path == '/kalman_toggle':
            # Toggle Kalman filter
            enabled = not self.server.gyro_data.use_kalman
            self.server.gyro_data.set_kalman_enabled(enabled)
            self.send_json_response({'success': True, 'kalman_enabled': enabled})
        elif parsed_path.path == '/calibrate':
            self.server.gyro_data.calibrate()
            self.send_json_response({'success': True, 'message': 'Calibrated'})
        else:
            self.send_error(404)

    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/gyro':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                gyro_data = json.loads(post_data.decode('utf-8'))

                self.server.gyro_data.update(gyro_data)
                self.send_json_response({'success': True})

            except Exception as e:
                logger.error(f"Error processing gyro data: {e}")
                self.send_json_response({'success': False, 'error': str(e)}, 400)
        else:
            self.send_error(404)

    def serve_status(self):
        """Serve current gyroscope status with Kalman info"""
        x, y, z, roll, pitch, yaw = self.server.gyro_data.get_current_pose()

        with self.server.gyro_data.lock:
            raw_ori = self.server.gyro_data.raw_orientation
            filt_ori = self.server.gyro_data.filtered_orientation

        status = {
            'position': {'x': x, 'y': y, 'z': z},
            'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
            'raw_orientation': raw_ori,
            'filtered_orientation': filt_ori,
            'gyro': self.server.gyro_data.gyro,
            'accel': self.server.gyro_data.accel,
            'kalman_enabled': self.server.gyro_data.use_kalman,
            'calibrated': self.server.gyro_data.calibrated,
            'valid': self.server.gyro_data.valid,
            'timestamp': self.server.gyro_data.timestamp
        }

        self.send_json_response(status)

    def serve_motor_commands(self):
        """Serve motor commands"""
        commands = self.server.gyro_data.get_motor_commands()
        self.send_json_response(commands)

    def serve_phone_interface(self):
        """Serve enhanced HTML interface with Kalman controls"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Phone Control with Kalman Filter</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: white;
            text-align: center;
        }

        .container {
            max-width: 400px;
            margin: 0 auto;
        }

        .status {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }

        .controls {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }

        button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 20px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            margin: 5px;
            cursor: pointer;
        }

        button:hover {
            background-color: #45a049;
        }

        .stop-button {
            background-color: #f44336;
        }

        .kalman-button {
            background-color: #2196F3;
        }

        .motor-displays {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 10px 0;
        }

        .motor-display {
            background-color: #333;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }

        .data {
            font-family: monospace;
            font-size: 12px;
            text-align: left;
            margin: 10px 0;
        }

        .connected {
            color: #4CAF50;
        }

        .disconnected {
            color: #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Robot Control + Kalman Filter</h1>

        <div class="status">
            <h3>Status: <span id="status" class="disconnected">Disconnected</span></h3>
            <div id="kalman-status" class="data">Kalman Filter: Enabled</div>
            <div id="orientation" class="data">
                Raw: roll=0, pitch=0<br>
                Filtered: roll=0, pitch=0
            </div>
        </div>

        <div class="motor-displays">
            <div class="motor-display">
                <strong>Motor 3 (Pitch)</strong><br>
                <span id="motor3">0.0</span>
            </div>
            <div class="motor-display">
                <strong>Motor 4 (Roll)</strong><br>
                <span id="motor4">0.0</span>
            </div>
        </div>

        <div class="controls">
            <button onclick="startControl()">Start Control</button>
            <button onclick="calibrate()">Calibrate</button>
            <button onclick="toggleKalman()" class="kalman-button" id="kalmanBtn">Toggle Kalman</button>
            <button onclick="stopControl()" class="stop-button">Stop</button>
        </div>

        <div class="status">
            <h4>📋 Controls:</h4>
            <div style="text-align: left; font-size: 12px;">
                • Pitch → Motor 3<br>
                • Roll → Motor 4<br>
                • Kalman filter reduces noise
            </div>
        </div>
    </div>

    <script>
        let isControlling = false;
        let kalmanEnabled = true;

        function startControl() {
            if ('DeviceMotionEvent' in window) {
                if (typeof DeviceMotionEvent.requestPermission === 'function') {
                    DeviceMotionEvent.requestPermission()
                        .then(response => {
                            if (response === 'granted') {
                                beginControl();
                            } else {
                                alert('Motion permission denied');
                            }
                        })
                        .catch(console.error);
                } else {
                    beginControl();
                }
            } else {
                alert('Device motion not supported');
            }
        }

        function beginControl() {
            isControlling = true;
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('status').className = 'connected';

            window.addEventListener('devicemotion', handleMotion);
            setInterval(updateStatus, 100);
        }

        function stopControl() {
            isControlling = false;
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').className = 'disconnected';

            window.removeEventListener('devicemotion', handleMotion);
        }

        function handleMotion(event) {
            if (!isControlling) return;

            const acceleration = event.accelerationIncludingGravity;
            const rotationRate = event.rotationRate;

            const roll = Math.atan2(acceleration.y, acceleration.z) * 180 / Math.PI;
            const pitch = Math.atan2(-acceleration.x, Math.sqrt(acceleration.y * acceleration.y + acceleration.z * acceleration.z)) * 180 / Math.PI;

            const data = {
                gyroscope: {
                    x: rotationRate ? rotationRate.beta : 0,
                    y: rotationRate ? rotationRate.alpha : 0,
                    z: rotationRate ? rotationRate.gamma : 0
                },
                accelerometer: {
                    x: acceleration.x,
                    y: acceleration.y,
                    z: acceleration.z
                },
                orientation: {
                    roll: roll,
                    pitch: pitch,
                    yaw: 0
                },
                timestamp: Date.now()
            };

            fetch('/gyro', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            }).catch(console.error);
        }

        function calibrate() {
            fetch('/calibrate')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Calibrated!');
                    }
                })
                .catch(console.error);
        }

        function toggleKalman() {
            fetch('/kalman_toggle')
                .then(response => response.json())
                .then(data => {
                    kalmanEnabled = data.kalman_enabled;
                    updateKalmanStatus();
                })
                .catch(console.error);
        }

        function updateKalmanStatus() {
            document.getElementById('kalman-status').textContent =
                `Kalman Filter: ${kalmanEnabled ? 'Enabled' : 'Disabled'}`;
            document.getElementById('kalmanBtn').textContent =
                kalmanEnabled ? 'Disable Kalman' : 'Enable Kalman';
        }

        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const raw = data.raw_orientation;
                    const filt = data.filtered_orientation;

                    document.getElementById('orientation').innerHTML =
                        `Raw: roll=${raw.roll.toFixed(1)}, pitch=${raw.pitch.toFixed(1)}<br>` +
                        `Filtered: roll=${filt.roll.toFixed(1)}, pitch=${filt.pitch.toFixed(1)}`;

                    // Update motor displays
                    document.getElementById('motor3').textContent = (filt.pitch * 2).toFixed(1);
                    document.getElementById('motor4').textContent = (filt.roll * 2).toFixed(1);

                    kalmanEnabled = data.kalman_enabled;
                    updateKalmanStatus();
                })
                .catch(console.error);
        }
    </script>
</body>
</html>'''

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-Length', len(html.encode()))
        self.end_headers()
        self.wfile.write(html.encode())

    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        response = json.dumps(data).encode()
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response)

    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")


class PhoneGyroServer(HTTPServer):
    """HTTP server with Kalman filtering"""

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.gyro_data = PhoneGyroData()


def start_phone_gyro_server_with_kalman(port=8889):
    """Start the phone gyroscope server with Kalman filtering"""
    logging.basicConfig(level=logging.INFO)

    server = PhoneGyroServer(('0.0.0.0', port), PhoneGyroHandler)

    logger.info(f"Phone gyro server with Kalman filter starting on port {port}")
    logger.info(f"Open http://localhost:{port} on your phone")
    logger.info("Features: Kalman filtering, Motor 3 (pitch), Motor 4 (roll)")
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopping...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    start_phone_gyro_server_with_kalman()