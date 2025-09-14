#!/usr/bin/env python

"""
Phone Gyroscope Server with Ngrok Integration
Clean version without emojis for Windows compatibility
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
    print("WARNING: ngrok Python package not found. Install with: python -m pip install ngrok")

logger = logging.getLogger(__name__)

# Your ngrok authtoken
NGROK_AUTHTOKEN = "32f9HuuEhm2h4ndUBTE23MZYjgn_2arH7dL4j47gJLPYTLgd5"


# Import the new Kalman filter
from kalman_filter import SimpleKalmanFilter

class KalmanFilter:
    """Wrapper for the new SimpleKalmanFilter"""

    def __init__(self, dt=0.01):
        self.filter = SimpleKalmanFilter()
        self.last_time = time.time()

    def update_with_sensors(self, gyro, accel):
        """Update with gyroscope and accelerometer data"""
        current_time = time.time()
        roll, pitch, yaw = self.filter.update(gyro, accel, current_time)
        self.last_time = current_time
        return roll, pitch, yaw

    def get_state(self):
        state = self.filter.get_state()
        return {
            'roll': state['euler']['roll'],
            'pitch': state['euler']['pitch'],
            'yaw': state['euler']['yaw'],
            'roll_rate': state['angular_velocity'][0],
            'pitch_rate': state['angular_velocity'][1],
            'yaw_rate': state['angular_velocity'][2]
        }


class PhoneGyroData:
    def __init__(self, use_kalman=True):
        self.lock = threading.Lock()
        self.orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}
        self.valid = False
        self.last_update = 0
        self.calibration_offset = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.use_kalman = use_kalman
        self.kalman = KalmanFilter() if use_kalman else None

    def update(self, alpha, beta, gamma, gyro=None, accel=None):
        with self.lock:
            if gamma is None or beta is None or alpha is None:
                return

            # Convert device orientation to robot orientation
            raw_roll = gamma
            raw_pitch = beta
            raw_yaw = alpha

            if self.use_kalman and self.kalman and gyro is not None and accel is not None:
                # Use new Kalman filter with actual sensor data
                try:
                    roll, pitch, yaw = self.kalman.update_with_sensors(gyro, accel)
                    self.orientation = {
                        'roll': roll - self.calibration_offset['roll'],
                        'pitch': pitch - self.calibration_offset['pitch'],
                        'yaw': yaw - self.calibration_offset['yaw']
                    }
                except Exception as e:
                    print(f"Kalman filter error: {e}")
                    # Fallback to direct mapping
                    self.orientation = {
                        'roll': raw_roll - self.calibration_offset['roll'],
                        'pitch': raw_pitch - self.calibration_offset['pitch'],
                        'yaw': raw_yaw - self.calibration_offset['yaw']
                    }
            else:
                # Direct mapping without filtering
                self.orientation = {
                    'roll': raw_roll - self.calibration_offset['roll'],
                    'pitch': raw_pitch - self.calibration_offset['pitch'],
                    'yaw': raw_yaw - self.calibration_offset['yaw']
                }

            # Simple position mapping
            roll_rad = np.radians(self.orientation['roll'])
            pitch_rad = np.radians(self.orientation['pitch'])

            self.position = {
                'x': np.sin(roll_rad) * 0.1,
                'y': 0.2 + np.sin(pitch_rad) * 0.1,
                'z': 0.15 + np.cos(pitch_rad) * 0.05
            }

            self.valid = True
            self.last_update = time.time()

    def calibrate(self):
        with self.lock:
            self.calibration_offset = self.orientation.copy()
            if self.kalman:
                # Reset Kalman filter
                self.kalman = KalmanFilter()
            return True

    def toggle_kalman(self):
        with self.lock:
            self.use_kalman = not self.use_kalman
            if self.use_kalman and not self.kalman:
                self.kalman = KalmanFilter()
            return self.use_kalman

    def get_status(self):
        with self.lock:
            age = time.time() - self.last_update
            return {
                'valid': self.valid and age < 1.0,
                'orientation': self.orientation.copy(),
                'position': self.position.copy(),
                'age': age,
                'kalman_enabled': self.use_kalman
            }


class PhoneGyroHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.serve_html()
        elif self.path == '/status':
            self.serve_status()
        elif self.path == '/calibrate':
            self.serve_calibrate()
        elif self.path == '/toggle_kalman':
            self.serve_toggle_kalman()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == '/update':
            self.handle_update()
        else:
            self.send_error(404, "Not Found")

    def serve_html(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phone Gyro Robot Control</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .button {
            padding: 15px 30px;
            margin: 10px;
            font-size: 18px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .start-button { background: #4CAF50; color: white; }
        .stop-button { background: #f44336; color: white; }
        .calibrate-button { background: #2196F3; color: white; }
        .kalman-button { background: #FF9800; color: white; }
        .button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
        .data-box {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            margin: 15px 0;
            border-radius: 10px;
            text-align: left;
        }
        .motor-display {
            display: inline-block;
            margin: 5px;
            padding: 15px;
            background: rgba(255,255,255,0.3);
            border-radius: 10px;
            min-width: 120px;
            font-weight: bold;
        }
        .status { font-size: 16px; margin: 15px 0; font-weight: bold; }
        .connected { color: #4CAF50; }
        .disconnected { color: #f44336; }
        .kalman-status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            background: rgba(255,255,255,0.2);
            font-weight: bold;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
        h3 { color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
    </style>
</head>
<body>
    <div class="container">
        <h1>Robot Phone Control</h1>
        <p style="font-size: 18px;">Control your robot using phone gyroscope with Kalman filtering!</p>

        <div class="status" id="status">Ready to connect...</div>
        <div class="kalman-status" id="kalman-status">Kalman Filter: Loading...</div>

        <button class="button start-button" onclick="startControl()">START CONTROL</button>
        <button class="button stop-button" onclick="stopControl()">STOP CONTROL</button>
        <button class="button calibrate-button" onclick="calibrate()">CALIBRATE</button>
        <button class="button kalman-button" onclick="toggleKalman()">TOGGLE KALMAN</button>

        <div class="data-box">
            <h3>Phone Orientation</h3>
            <p><strong>Roll:</strong> <span id="roll">0.0</span>&deg; (left/right tilt)</p>
            <p><strong>Pitch:</strong> <span id="pitch">0.0</span>&deg; (forward/back tilt)</p>
            <p><strong>Yaw:</strong> <span id="yaw">0.0</span>&deg; (rotation)</p>
        </div>

        <div class="data-box">
            <h3>Robot Motor Commands</h3>
            <div class="motor-display">
                <strong>MOTOR 3</strong><br>
                Wrist Flex (Pitch)<br>
                <span id="motor3" style="font-size: 24px;">0</span>
            </div>
            <div class="motor-display">
                <strong>MOTOR 4</strong><br>
                Wrist Roll (Roll)<br>
                <span id="motor4" style="font-size: 24px;">0</span>
            </div>
        </div>

        <p><strong>Controls:</strong> Phone PITCH controls Motor 3, Phone ROLL controls Motor 4</p>
        <p><small>Tilt phone forward/back for Motor 3, roll left/right for Motor 4</small></p>
    </div>

    <script>
        let isRunning = false;
        let updateInterval;
        let currentData = { alpha: null, beta: null, gamma: null };
        let motionData = {
            gyro: { x: 0, y: 0, z: 0 },
            accel: { x: 0, y: 0, z: 0 }
        };

        function updateDisplay() {
            if (!isRunning) return;

            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('roll').textContent = data.orientation.roll.toFixed(1);
                    document.getElementById('pitch').textContent = data.orientation.pitch.toFixed(1);
                    document.getElementById('yaw').textContent = data.orientation.yaw.toFixed(1);

                    // Motor mapping: pitch -> motor 3, roll -> motor 4
                    const motor3 = Math.round(Math.max(-100, Math.min(100, data.orientation.pitch * 2)));
                    const motor4 = Math.round(Math.max(-100, Math.min(100, data.orientation.roll * 2)));

                    document.getElementById('motor3').textContent = motor3;
                    document.getElementById('motor4').textContent = motor4;

                    const statusEl = document.getElementById('status');
                    if (data.valid) {
                        statusEl.textContent = 'CONNECTED - Robot receiving commands';
                        statusEl.className = 'status connected';
                    } else {
                        statusEl.textContent = 'DISCONNECTED - No recent data';
                        statusEl.className = 'status disconnected';
                    }

                    const kalmanEl = document.getElementById('kalman-status');
                    kalmanEl.textContent = `Kalman Filter: ${data.kalman_enabled ? 'ENABLED (Smooth)' : 'DISABLED (Raw)'}`;
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('status').textContent = 'ERROR - Cannot connect to server';
                    document.getElementById('status').className = 'status disconnected';
                });
        }

        function sendUpdate() {
            if (!isRunning) return;

            fetch('/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    alpha: currentData.alpha,
                    beta: currentData.beta,
                    gamma: currentData.gamma,
                    gyro: motionData.gyro,
                    accel: motionData.accel
                })
            }).catch(error => console.error('Update error:', error));
        }

        function startControl() {
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                DeviceOrientationEvent.requestPermission()
                    .then(response => {
                        if (response == 'granted') {
                            startGyroscope();
                        } else {
                            alert('Motion permission denied. Please enable in browser settings.');
                        }
                    })
                    .catch(console.error);
            } else {
                startGyroscope();
            }
        }

        function startGyroscope() {
            isRunning = true;
            document.getElementById('status').textContent = 'STARTING gyroscope...';
            document.getElementById('status').className = 'status';

            window.addEventListener('deviceorientation', handleOrientation);
            window.addEventListener('devicemotion', handleMotion);
            updateInterval = setInterval(updateDisplay, 50); // 20Hz update rate

            setTimeout(() => {
                if (isRunning) {
                    document.getElementById('status').textContent = 'ACTIVE - Move phone to control robot';
                    document.getElementById('status').className = 'status connected';
                }
            }, 1000);
        }

        function stopControl() {
            isRunning = false;
            window.removeEventListener('deviceorientation', handleOrientation);
            if (updateInterval) clearInterval(updateInterval);
            document.getElementById('status').textContent = 'STOPPED - Control disabled';
            document.getElementById('status').className = 'status disconnected';
        }

        function handleOrientation(event) {
            currentData.alpha = event.alpha;
            currentData.beta = event.beta;
            currentData.gamma = event.gamma;
            sendUpdate();
        }

        function handleMotion(event) {
            if (event.rotationRate) {
                motionData.gyro.x = event.rotationRate.alpha || 0;
                motionData.gyro.y = event.rotationRate.beta || 0;
                motionData.gyro.z = event.rotationRate.gamma || 0;
            }
            if (event.acceleration) {
                motionData.accel.x = event.acceleration.x || 0;
                motionData.accel.y = event.acceleration.y || 0;
                motionData.accel.z = event.acceleration.z || 0;
            }
        }

        function calibrate() {
            fetch('/calibrate')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Robot position CALIBRATED!');
                    } else {
                        alert('Calibration failed: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Calibration error:', error);
                    alert('Calibration error - check console');
                });
        }

        function toggleKalman() {
            fetch('/toggle_kalman')
                .then(response => response.json())
                .then(data => {
                    const status = data.kalman_enabled ? 'ENABLED' : 'DISABLED';
                    alert(`Kalman Filter ${status}`);
                })
                .catch(error => {
                    console.error('Toggle error:', error);
                    alert('Toggle error - check console');
                });
        }

        // Start updating display immediately
        setInterval(updateDisplay, 200);
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def serve_status(self):
        status = gyro_data.get_status()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def serve_calibrate(self):
        success = gyro_data.calibrate()
        response = {'success': success, 'message': 'Calibrated successfully' if success else 'Calibration failed'}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_toggle_kalman(self):
        kalman_enabled = gyro_data.toggle_kalman()
        response = {'kalman_enabled': kalman_enabled}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def handle_update(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode())
            alpha = data.get('alpha')
            beta = data.get('beta')
            gamma = data.get('gamma')

            # Extract gyro and accel data
            gyro = None
            accel = None
            if 'gyro' in data and data['gyro']:
                gyro_data_raw = data['gyro']
                gyro = np.array([
                    np.radians(gyro_data_raw.get('x', 0)),  # Convert to rad/s
                    np.radians(gyro_data_raw.get('y', 0)),
                    np.radians(gyro_data_raw.get('z', 0))
                ])

            if 'accel' in data and data['accel']:
                accel_data_raw = data['accel']
                accel = np.array([
                    accel_data_raw.get('x', 0),
                    accel_data_raw.get('y', 0),
                    accel_data_raw.get('z', 0) + 9.81  # Add gravity offset
                ])

            gyro_data.update(alpha, beta, gamma, gyro, accel)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())

        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def log_message(self, format, *args):
        pass  # Suppress default HTTP server logging


# Global data storage
gyro_data = PhoneGyroData(use_kalman=True)


def start_ngrok_tunnel(port):
    """Start ngrok tunnel and return public URL"""
    if not NGROK_AVAILABLE:
        return None

    try:
        # Connect to ngrok with authtoken
        session = ngrok.Session()
        session.authtoken = NGROK_AUTHTOKEN

        # Create HTTP tunnel
        tunnel = session.http_endpoint().listen()

        print(f"[NGROK] Tunnel created: {tunnel.url()}")
        print(f"[NGROK] Forwarding to: http://localhost:{port}")

        return tunnel.url()

    except Exception as e:
        print(f"[NGROK ERROR] Failed to create tunnel: {e}")
        return None


def main():
    port = 8889
    print("=" * 50)
    print("PHONE GYROSCOPE ROBOT CONTROL SERVER")
    print("=" * 50)

    # Start ngrok tunnel
    public_url = start_ngrok_tunnel(port)

    # Start HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, PhoneGyroHandler)

    print(f"\n[SERVER] Starting on port {port}...")
    print(f"[LOCAL] http://localhost:{port}")

    if public_url:
        print(f"[PUBLIC] {public_url}")
        print(f"\n*** OPEN THIS URL ON YOUR PHONE: {public_url} ***")
    else:
        print("[WARNING] Ngrok tunnel failed - using localhost only")
        print("Motion sensors may not work without HTTPS")

    print("\n[CONTROLS]")
    print("- Phone PITCH (forward/back) -> Motor 3 (wrist_flex)")
    print("- Phone ROLL (left/right) -> Motor 4 (wrist_roll)")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.server_close()


if __name__ == '__main__':
    main()