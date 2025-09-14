#!/usr/bin/env python

"""
Phone Gyroscope Server with Built-in Ngrok Integration

This version uses the ngrok Python SDK to automatically create a public tunnel
for your phone gyroscope control system with Kalman filtering.
"""

import os
import logging
import json
import time
import threading
from typing import Dict, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# Import ngrok Python SDK
try:
    import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("⚠️  ngrok Python package not found. Install with: python -m pip install ngrok")

logger = logging.getLogger(__name__)

# Your ngrok authtoken
NGROK_AUTHTOKEN = "32f9HuuEhm2h4ndUBTE23MZYjgn_2arH7dL4j47gJLPYTLgd5"


class KalmanFilter:
    """Extended Kalman Filter for gyroscope and accelerometer fusion"""

    def __init__(self, dt=0.01):
        self.dt = dt
        self.x = np.zeros(6)  # [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.P = np.eye(6) * 0.1

        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]) * dt
        self.R_gyro = np.diag([0.1, 0.1, 0.1])
        self.R_accel = np.diag([0.3, 0.3])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gyro(self, gyro_rates):
        H = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        y = gyro_rates - H @ self.x
        S = H @ self.P @ H.T + self.R_gyro
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P

    def update_accel(self, accel_orientation):
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
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
        self.use_kalman = True
        self.kalman = KalmanFilter(dt=0.01)
        self.reset()

    def reset(self):
        with self.lock:
            self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}
            self.raw_orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            self.filtered_orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            self.gyro = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.valid = False
            self.timestamp = time.time()
            self.calibrated = False
            self.kalman = KalmanFilter(dt=0.01)

    def update(self, gyro_data: dict):
        with self.lock:
            current_time = time.time()
            dt = current_time - self.timestamp

            if 'gyroscope' in gyro_data:
                self.gyro = gyro_data['gyroscope']
            if 'accelerometer' in gyro_data:
                self.accel = gyro_data['accelerometer']
            if 'orientation' in gyro_data:
                self.raw_orientation = gyro_data['orientation']

            if self.use_kalman and dt > 0:
                self.kalman.dt = min(dt, 0.1)
                self.kalman.predict()

                gyro_rates = np.array([
                    np.radians(self.gyro['x']),
                    np.radians(self.gyro['y']),
                    np.radians(self.gyro['z'])
                ])
                self.kalman.update_gyro(gyro_rates)

                accel_roll = np.arctan2(self.accel['y'], self.accel['z'])
                accel_pitch = np.arctan2(-self.accel['x'],
                                       np.sqrt(self.accel['y']**2 + self.accel['z']**2))
                self.kalman.update_accel(np.array([accel_roll, accel_pitch]))

                roll, pitch, yaw = self.kalman.get_orientation()
                self.filtered_orientation = {
                    'roll': np.degrees(roll),
                    'pitch': np.degrees(pitch),
                    'yaw': np.degrees(yaw)
                }
            else:
                self.filtered_orientation = self.raw_orientation.copy()

            if dt > 0 and self.calibrated:
                position_scale = 0.01
                self.position['x'] += self.gyro['x'] * dt * position_scale
                self.position['y'] += self.gyro['z'] * dt * position_scale
                self.position['z'] += -self.gyro['y'] * dt * position_scale

                self.position['x'] = np.clip(self.position['x'], -0.3, 0.3)
                self.position['y'] = np.clip(self.position['y'], 0.1, 0.4)
                self.position['z'] = np.clip(self.position['z'], 0.05, 0.3)

            self.timestamp = current_time
            self.valid = True

    def get_current_pose(self):
        with self.lock:
            if not self.valid:
                return (0.0, 0.2, 0.15, 0.0, 0.0, 0.0)
            return (
                self.position['x'], self.position['y'], self.position['z'],
                self.filtered_orientation['roll'],
                self.filtered_orientation['pitch'],
                self.filtered_orientation['yaw']
            )

    def calibrate(self):
        with self.lock:
            self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}
            self.calibrated = True
            logger.info("Phone gyroscope calibrated")


class PhoneGyroHandler(BaseHTTPRequestHandler):
    """HTTP request handler for phone gyroscope data"""

    def do_GET(self):
        if self.path == '/':
            self.serve_phone_interface()
        elif self.path == '/status':
            self.serve_status()
        elif self.path == '/calibrate':
            self.server.gyro_data.calibrate()
            self.send_json_response({'success': True, 'message': 'Calibrated'})
        else:
            self.send_error(404)

    def do_POST(self):
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

    def serve_phone_interface(self):
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Robot Phone Control via Ngrok</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0; padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; text-align: center; min-height: 100vh;
        }
        .container { max-width: 400px; margin: 0 auto; }
        .header { margin: 20px 0; }
        .status, .controls {
            background: rgba(255,255,255,0.1);
            padding: 20px; border-radius: 15px;
            margin: 15px 0; backdrop-filter: blur(10px);
        }
        button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white; padding: 15px 25px; border: none;
            border-radius: 10px; font-size: 16px; margin: 5px;
            cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
        .stop-button { background: linear-gradient(45deg, #f44336, #da190b); }
        .motor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }
        .motor-display {
            background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px;
            font-family: 'Courier New', monospace; font-size: 14px;
        }
        .data { font-family: 'Courier New', monospace; font-size: 12px; text-align: left; margin: 10px 0; }
        .connected { color: #4CAF50; font-weight: bold; }
        .disconnected { color: #ff6b6b; font-weight: bold; }
        .ngrok-badge {
            background: rgba(0,0,0,0.2); padding: 10px; border-radius: 10px;
            font-size: 12px; margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Robot Control</h1>
            <div class="ngrok-badge">🌐 Connected via Ngrok</div>
        </div>

        <div class="status">
            <h3>Status: <span id="status" class="disconnected">Disconnected</span></h3>
            <div id="orientation" class="data">
                Raw: roll=0°, pitch=0°<br>
                Kalman: roll=0°, pitch=0°
            </div>
        </div>

        <div class="motor-grid">
            <div class="motor-display">
                <strong>Motor 3 (Pitch)</strong><br>
                <span id="motor3" style="font-size: 18px; color: #4CAF50;">0.0</span>
            </div>
            <div class="motor-display">
                <strong>Motor 4 (Roll)</strong><br>
                <span id="motor4" style="font-size: 18px; color: #2196F3;">0.0</span>
            </div>
        </div>

        <div class="controls">
            <button onclick="startControl()">🚀 Start Control</button>
            <button onclick="calibrate()">🎯 Calibrate</button>
            <button onclick="stopControl()" class="stop-button">🛑 Stop</button>
        </div>

        <div class="status">
            <h4>📱 Instructions:</h4>
            <div style="text-align: left; font-size: 14px;">
                • <strong>Pitch phone forward/back</strong> → Motor 3<br>
                • <strong>Roll phone left/right</strong> → Motor 4<br>
                • Kalman filter smooths out noise<br>
                • Public access via Ngrok tunnel
            </div>
        </div>
    </div>

    <script>
        let isControlling = false;

        function startControl() {
            if ('DeviceMotionEvent' in window) {
                if (typeof DeviceMotionEvent.requestPermission === 'function') {
                    DeviceMotionEvent.requestPermission()
                        .then(response => {
                            if (response === 'granted') beginControl();
                            else alert('❌ Motion permission denied');
                        }).catch(console.error);
                } else {
                    beginControl();
                }
            } else {
                alert('❌ Device motion not supported');
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
            const pitch = Math.atan2(-acceleration.x,
                Math.sqrt(acceleration.y * acceleration.y + acceleration.z * acceleration.z)) * 180 / Math.PI;

            fetch('/gyro', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gyroscope: {
                        x: rotationRate ? rotationRate.beta : 0,
                        y: rotationRate ? rotationRate.alpha : 0,
                        z: rotationRate ? rotationRate.gamma : 0
                    },
                    accelerometer: { x: acceleration.x, y: acceleration.y, z: acceleration.z },
                    orientation: { roll: roll, pitch: pitch, yaw: 0 },
                    timestamp: Date.now()
                })
            }).catch(console.error);
        }

        function calibrate() {
            fetch('/calibrate')
                .then(response => response.json())
                .then(data => {
                    if (data.success) alert('✅ Calibrated! Robot reset to center.');
                }).catch(console.error);
        }

        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const raw = data.raw_orientation;
                    const filt = data.filtered_orientation;

                    document.getElementById('orientation').innerHTML =
                        `Raw: roll=${raw.roll.toFixed(1)}°, pitch=${raw.pitch.toFixed(1)}°<br>` +
                        `Kalman: roll=${filt.roll.toFixed(1)}°, pitch=${filt.pitch.toFixed(1)}°`;

                    document.getElementById('motor3').textContent = (filt.pitch * 2).toFixed(1);
                    document.getElementById('motor4').textContent = (filt.roll * 2).toFixed(1);
                }).catch(console.error);
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
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")


class PhoneGyroServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.gyro_data = PhoneGyroData()


def start_server_with_ngrok():
    """Start the server with automatic ngrok tunneling"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create server
    server = PhoneGyroServer(("localhost", 0), PhoneGyroHandler)  # Use port 0 for auto-assign
    port = server.server_address[1]

    print("🤖 Starting Phone Gyroscope Control with Ngrok")
    print("="*60)

    if not NGROK_AVAILABLE:
        print("❌ ngrok Python package not found!")
        print("Install with: python -m pip install ngrok")
        return

    try:
        # Set authtoken as environment variable (required by ngrok SDK)
        os.environ['NGROK_AUTHTOKEN'] = NGROK_AUTHTOKEN

        # Start ngrok tunnel
        print(f"🔗 Creating ngrok tunnel for localhost:{port}...")
        listener = ngrok.forward(port, authtoken=NGROK_AUTHTOKEN)
        public_url = listener.url()

        print(f"✅ Server running locally: http://localhost:{port}")
        print(f"🌐 Public URL: {public_url}")
        print()
        print("📱 PHONE SETUP:")
        print(f"   Open this URL on your phone: {public_url}")
        print("   (Works from anywhere with internet connection)")
        print()
        print("🎮 PHONE CONTROLS:")
        print("   1. Tap '🚀 Start Control' and allow motion permissions")
        print("   2. Tap '🎯 Calibrate' to reset robot position")
        print("   3. Move phone to see Motor 3 (pitch) and Motor 4 (roll)")
        print()
        print("🤖 ROBOT TELEOPERATION:")
        print("   cd lerobot")
        print("   py -m lerobot.teleoperate \\")
        print("     --robot.type=so101_follower \\")
        print("     --robot.port=COM6 \\")
        print("     --robot.id=follower \\")
        print("     --teleop.type=phone_gyro \\")
        print(f"     --teleop.server_url=http://localhost:{port} \\")
        print("     --display_data=true \\")
        print("     --fps=60")
        print()
        print("🎯 MOTOR MAPPING:")
        print("   • Phone PITCH (forward/back) → Motor 3 (wrist_flex)")
        print("   • Phone ROLL (left/right)    → Motor 4 (wrist_roll)")
        print()
        print("🔬 FEATURES:")
        print("   ✅ Kalman filter for smooth control")
        print("   ✅ Real-time motor command display")
        print("   ✅ Global access via ngrok tunnel")
        print("   ✅ Beautiful responsive web interface")
        print()
        print("⌨️  Press Ctrl+C to stop")
        print("="*60)

        # Keep server running
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            server.server_close()
            ngrok.kill()
            print("✅ Server stopped cleanly")

    except Exception as e:
        print(f"❌ Failed to start ngrok tunnel: {e}")
        print("💡 Fallback: Server running locally only")
        print(f"   Open http://localhost:{port} if on same network")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            server.server_close()


if __name__ == "__main__":
    start_server_with_ngrok()