#!/usr/bin/env python

"""
Horizontal Phone Control Server
Optimized for landscape/horizontal phone orientation with better grip
Gyroscope rotated 90 degrees left for horizontal use
"""

import os
import logging
import json
import time
import threading
import numpy as np
from typing import Dict, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Import ngrok
try:
    import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("WARNING: ngrok not found. Install with: pip install ngrok")

logger = logging.getLogger(__name__)
NGROK_AUTHTOKEN = "32fWpNeDorrKVGxx7V4FVaXSK8Q_6T6r23CNo4PZwvMk67cCH"


class PhoneGyroData:
    def __init__(self):
        self.lock = threading.Lock()
        self.orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}
        self.acceleration = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.joystick = {'x': 0.0, 'y': 0.0}
        self.valid = False
        self.last_update = 0
        self.calibration_offset = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.accel_calibration_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.gripper_closed = False
        self.gripper_last_update = 0

    def update_orientation(self, alpha, beta, gamma):
        """Update with rotated gyroscope data (90 degrees left rotation)"""
        with self.lock:
            if gamma is None or beta is None or alpha is None:
                return

            # Apply 90-degree left rotation transformation
            # Original: roll=gamma, pitch=beta, yaw=alpha
            # Rotated: roll=beta, pitch=-gamma, yaw=alpha
            raw_roll = beta      # pitch becomes roll
            raw_pitch = -gamma   # -roll becomes pitch
            raw_yaw = alpha      # yaw stays yaw

            # Apply calibration offsets
            self.orientation = {
                'roll': raw_roll - self.calibration_offset['roll'],
                'pitch': raw_pitch - self.calibration_offset['pitch'],
                'yaw': raw_yaw - self.calibration_offset['yaw']
            }

            # Simple position mapping from orientation
            roll_rad = np.radians(self.orientation['roll'])
            pitch_rad = np.radians(self.orientation['pitch'])

            self.position = {
                'x': np.sin(pitch_rad) * 0.1 + 0.0,
                'y': -np.sin(roll_rad) * 0.1 + 0.2,  # Negative for intuitive control
                'z': np.cos(pitch_rad) * 0.05 + 0.15
            }

            self.valid = True
            self.last_update = time.time()

    def update_acceleration(self, accel_x, accel_y, accel_z):
        """Update acceleration with rotation applied"""
        with self.lock:
            if accel_x is None or accel_y is None or accel_z is None:
                return

            # Apply same 90-degree left rotation to acceleration
            # Original: x, y, z
            # Rotated: x=y, y=-x, z=z
            rotated_x = accel_y
            rotated_y = -accel_x
            rotated_z = accel_z

            self.acceleration = {
                'x': rotated_x - self.accel_calibration_offset['x'],
                'y': rotated_y - self.accel_calibration_offset['y'],
                'z': rotated_z - self.accel_calibration_offset['z']
            }

    def update_joystick(self, joystick_x, joystick_y):
        """Update joystick data"""
        with self.lock:
            if joystick_x is not None and joystick_y is not None:
                # Normalize joystick values to -1 to 1 range
                self.joystick = {
                    'x': np.clip(joystick_x / 15.0, -1.0, 1.0),
                    'y': np.clip(joystick_y / 15.0, -1.0, 1.0)
                }

    def update_gripper(self, closed):
        """Update gripper state"""
        with self.lock:
            self.gripper_closed = bool(closed)
            self.gripper_last_update = time.time()

    def calibrate(self):
        """Calibrate by setting current orientation as zero"""
        with self.lock:
            self.calibration_offset = self.orientation.copy()
            self.accel_calibration_offset = self.acceleration.copy()
            return True

    def get_status(self):
        """Get current status"""
        with self.lock:
            age = time.time() - self.last_update
            gripper_age = time.time() - self.gripper_last_update
            return {
                'valid': self.valid and age < 1.0,
                'orientation': self.orientation.copy(),
                'position': self.position.copy(),
                'acceleration': self.acceleration.copy(),
                'joystick': self.joystick.copy(),
                'age': age,
                'gripper_closed': self.gripper_closed,
                'gripper_age': gripper_age
            }


# Global data instance
gyro_data = PhoneGyroData()


class HorizontalPhoneHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.serve_html()
        elif self.path == '/status':
            self.serve_status()
        elif self.path == '/calibrate':
            self.serve_calibrate()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == '/update':
            self.handle_update()
        elif self.path == '/gripper':
            self.handle_gripper()
        elif self.path == '/joystick':
            self.handle_joystick()
        else:
            self.send_error(404, "Not Found")

    def serve_html(self):
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Horizontal Robot Control</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            overflow: hidden;
            touch-action: none;
            height: 100vh;
            width: 100vw;
        }

        .container {
            display: flex;
            flex-direction: row;
            height: 100vh;
            width: 100vw;
            padding: 10px;
            gap: 10px;
        }

        .left-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .right-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-items: center;
            justify-content: center;
        }

        .header {
            text-align: center;
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .header h1 {
            font-size: 18px;
            margin-bottom: 5px;
        }

        .status {
            font-size: 12px;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 20px;
            background: #e74c3c;
            color: white;
        }

        .status.connected {
            background: #27ae60;
        }

        .joystick-container {
            position: relative;
            width: 200px;
            height: 200px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border: 3px solid rgba(255, 255, 255, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
        }

        .joystick-knob {
            position: absolute;
            width: 60px;
            height: 60px;
            background: radial-gradient(circle, #4CAF50, #2E7D32);
            border-radius: 50%;
            border: 3px solid white;
            cursor: grab;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            z-index: 10;
        }

        .joystick-knob.dragging {
            cursor: grabbing;
            box-shadow: 0 6px 12px rgba(0,0,0,0.5);
        }

        .crosshair {
            position: absolute;
            background: rgba(255, 255, 255, 0.2);
        }

        .crosshair.horizontal {
            width: 100%;
            height: 1px;
            top: 50%;
        }

        .crosshair.vertical {
            width: 1px;
            height: 100%;
            left: 50%;
        }

        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .control-group {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
        }

        .control-group h3 {
            font-size: 14px;
            margin-bottom: 10px;
            color: #ecf0f1;
        }

        .gyro-display {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 10px;
        }

        .gyro-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 8px 12px;
            border-radius: 8px;
            text-align: center;
            font-size: 11px;
        }

        .gyro-value {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            font-weight: bold;
            color: #3498db;
        }

        .button {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
        }

        .button:hover {
            background: #2980b9;
            transform: scale(1.05);
        }

        .button:active {
            transform: scale(0.95);
        }

        .button.danger {
            background: #e74c3c;
        }

        .button.danger:hover {
            background: #c0392b;
        }

        .button.success {
            background: #27ae60;
        }

        .button.success:hover {
            background: #229954;
        }

        .gripper-controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 10px;
        }

        .orientation-hint {
            text-align: center;
            font-size: 11px;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 15px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }

        .landscape-indicator {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.5);
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 10px;
        }

        @media (max-width: 600px) {
            .container {
                flex-direction: column;
            }

            .joystick-container {
                width: 150px;
                height: 150px;
            }

            .joystick-knob {
                width: 45px;
                height: 45px;
            }
        }
    </style>
</head>
<body>
    <div class="landscape-indicator">📱 LANDSCAPE MODE</div>

    <div class="container">
        <div class="left-panel">
            <div class="header">
                <h1>🤖 Horizontal Robot Control</h1>
                <div class="status" id="status">DISCONNECTED</div>
            </div>

            <div class="control-group">
                <h3>📊 Phone Orientation (Rotated 90° Left)</h3>
                <div class="gyro-display">
                    <div class="gyro-item">
                        <div>ROLL</div>
                        <div class="gyro-value" id="rollValue">0.0°</div>
                    </div>
                    <div class="gyro-item">
                        <div>PITCH</div>
                        <div class="gyro-value" id="pitchValue">0.0°</div>
                    </div>
                    <div class="gyro-item">
                        <div>YAW</div>
                        <div class="gyro-value" id="yawValue">0.0°</div>
                    </div>
                    <div class="gyro-item">
                        <div>ACCEL X</div>
                        <div class="gyro-value" id="accelValue">0.0</div>
                    </div>
                </div>
                <button class="button" onclick="calibrate()">🎯 CALIBRATE</button>
            </div>

            <div class="control-group">
                <h3>🦾 Gripper Control</h3>
                <div class="gripper-controls">
                    <button class="button success" onmousedown="setGripper(true)" onmouseup="setGripper(false)" ontouchstart="setGripper(true)" ontouchend="setGripper(false)">
                        🤏 CLOSE
                    </button>
                    <button class="button danger" onmousedown="setGripper(false)" onmouseup="setGripper(false)">
                        ✋ OPEN
                    </button>
                </div>
            </div>
        </div>

        <div class="right-panel">
            <div class="control-group">
                <h3>🕹️ Movement Joystick</h3>
                <div class="joystick-container" id="joystickContainer">
                    <div class="crosshair horizontal"></div>
                    <div class="crosshair vertical"></div>
                    <div class="joystick-knob" id="joystickKnob"></div>
                </div>
                <div class="gyro-display">
                    <div class="gyro-item">
                        <div>X POS</div>
                        <div class="gyro-value" id="joyX">0.0</div>
                    </div>
                    <div class="gyro-item">
                        <div>Y POS</div>
                        <div class="gyro-value" id="joyY">0.0</div>
                    </div>
                </div>
            </div>

            <div class="orientation-hint">
                📱 Hold phone horizontally (landscape)<br>
                🔄 Gyroscope rotated 90° left for better grip<br>
                🕹️ Drag joystick for precise movement
            </div>
        </div>
    </div>

    <script>
        let isRunning = false;
        let updateInterval;
        let currentData = { alpha: null, beta: null, gamma: null };
        let motionData = {
            gyro: { x: 0, y: 0, z: 0 },
            accel: { x: 0, y: 0, z: 0 }
        };

        // Joystick variables
        const joystickKnob = document.getElementById('joystickKnob');
        const joystickContainer = document.getElementById('joystickContainer');
        let isDragging = false;
        let containerRect = joystickContainer.getBoundingClientRect();
        let containerCenterX = containerRect.left + containerRect.width / 2;
        let containerCenterY = containerRect.top + containerRect.height / 2;
        let maxRadius = containerRect.width / 2 - 30;
        let joystickX = 0;
        let joystickY = 0;

        function updateDisplay() {
            if (!isRunning) return;

            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.valid) {
                        document.getElementById('status').textContent = 'CONNECTED';
                        document.getElementById('status').className = 'status connected';

                        document.getElementById('rollValue').textContent = data.orientation.roll.toFixed(1) + '°';
                        document.getElementById('pitchValue').textContent = data.orientation.pitch.toFixed(1) + '°';
                        document.getElementById('yawValue').textContent = data.orientation.yaw.toFixed(1) + '°';
                        document.getElementById('accelValue').textContent = data.acceleration.x.toFixed(2);

                        document.getElementById('joyX').textContent = data.joystick.x.toFixed(2);
                        document.getElementById('joyY').textContent = data.joystick.y.toFixed(2);
                    } else {
                        document.getElementById('status').textContent = 'DISCONNECTED';
                        document.getElementById('status').className = 'status';
                    }
                })
                .catch(error => {
                    document.getElementById('status').textContent = 'ERROR';
                    document.getElementById('status').className = 'status';
                });
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
                        if (response === 'granted') {
                            startGyroscope();
                        } else {
                            alert('Motion sensor permission denied');
                        }
                    })
                    .catch(error => {
                        console.error('Permission error:', error);
                        alert('Error requesting motion sensor permission');
                    });
            } else {
                startGyroscope();
            }
        }

        function startGyroscope() {
            isRunning = true;
            document.getElementById('status').textContent = 'STARTING...';

            window.addEventListener('deviceorientation', handleOrientation);
            window.addEventListener('devicemotion', handleMotion);
            updateInterval = setInterval(updateDisplay, 50);

            setTimeout(() => {
                if (isRunning) {
                    document.getElementById('status').textContent = 'ACTIVE';
                    document.getElementById('status').className = 'status connected';
                }
            }, 1000);
        }

        function stopControl() {
            isRunning = false;
            window.removeEventListener('deviceorientation', handleOrientation);
            window.removeEventListener('devicemotion', handleMotion);
            if (updateInterval) clearInterval(updateInterval);

            document.getElementById('status').textContent = 'STOPPED';
            document.getElementById('status').className = 'status';
        }

        function calibrate() {
            fetch('/calibrate')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('🎯 Robot position CALIBRATED!');
                    } else {
                        alert('❌ Calibration failed');
                    }
                })
                .catch(error => {
                    console.error('Calibration error:', error);
                    alert('❌ Calibration error');
                });
        }

        function setGripper(closed) {
            fetch('/gripper', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ closed: closed })
            }).catch(error => console.error('Gripper error:', error));
        }

        // Joystick functions
        function updateJoystickPosition(clientX, clientY) {
            const dx = clientX - containerCenterX;
            const dy = clientY - containerCenterY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance <= maxRadius) {
                joystickX = dx;
                joystickY = -dy;
                joystickKnob.style.left = (containerRect.width / 2 + dx - 30) + 'px';
                joystickKnob.style.top = (containerRect.height / 2 + dy - 30) + 'px';
            } else {
                const angle = Math.atan2(dy, dx);
                joystickX = Math.cos(angle) * maxRadius;
                joystickY = -Math.sin(angle) * maxRadius;
                joystickKnob.style.left = (containerRect.width / 2 + joystickX - 30) + 'px';
                joystickKnob.style.top = (containerRect.height / 2 - joystickY - 30) + 'px';
            }

            // Convert to -15 to +15 range for server
            const joyXCm = (joystickX / maxRadius) * 15;
            const joyYCm = (joystickY / maxRadius) * 15;

            sendJoystickData(joyXCm, joyYCm);
        }

        function resetJoystick() {
            joystickX = 0;
            joystickY = 0;
            joystickKnob.style.left = (containerRect.width / 2 - 30) + 'px';
            joystickKnob.style.top = (containerRect.height / 2 - 30) + 'px';
            sendJoystickData(0, 0);
        }

        function sendJoystickData(x, y) {
            fetch('/joystick', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: x, y: y })
            }).catch(error => console.error('Joystick error:', error));
        }

        // Mouse events
        joystickKnob.addEventListener('mousedown', (e) => {
            isDragging = true;
            joystickKnob.classList.add('dragging');
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                updateJoystickPosition(e.clientX, e.clientY);
            }
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                joystickKnob.classList.remove('dragging');
                resetJoystick();
            }
        });

        // Touch events
        joystickKnob.addEventListener('touchstart', (e) => {
            isDragging = true;
            joystickKnob.classList.add('dragging');
            e.preventDefault();
        });

        document.addEventListener('touchmove', (e) => {
            if (isDragging && e.touches.length > 0) {
                updateJoystickPosition(e.touches[0].clientX, e.touches[0].clientY);
            }
        });

        document.addEventListener('touchend', () => {
            if (isDragging) {
                isDragging = false;
                joystickKnob.classList.remove('dragging');
                resetJoystick();
            }
        });

        // Update container dimensions on resize
        window.addEventListener('resize', () => {
            containerRect = joystickContainer.getBoundingClientRect();
            containerCenterX = containerRect.left + containerRect.width / 2;
            containerCenterY = containerRect.top + containerRect.height / 2;
            maxRadius = containerRect.width / 2 - 30;
        });

        // Initialize
        resetJoystick();
        startControl();
    </script>
</body>
</html>'''

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
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

        response = {'success': success}
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

            # Extract accelerometer data
            accel = data.get('accel', {})
            accel_x = accel.get('x', 0)
            accel_y = accel.get('y', 0)
            accel_z = accel.get('z', 0)

            gyro_data.update_orientation(alpha, beta, gamma)
            gyro_data.update_acceleration(accel_x, accel_y, accel_z)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())

        except Exception as e:
            print(f"Update error: {e}")
            self.send_error(400, str(e))

    def handle_gripper(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode())
            closed = data.get('closed', False)

            gyro_data.update_gripper(closed)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())

        except Exception as e:
            print(f"Gripper error: {e}")
            self.send_error(400, str(e))

    def handle_joystick(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode())
            x = float(data.get('x', 0))
            y = float(data.get('y', 0))

            gyro_data.update_joystick(x, y)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())

        except Exception as e:
            print(f"Joystick error: {e}")
            self.send_error(400, str(e))

    def log_message(self, format, *args):
        pass


def main():
    # Kill any existing servers on port 8889
    print("Starting Horizontal Phone Control Server...")

    port = 8889
    server_address = ('', port)

    print(f"Server starting on port {port}...")
    print(f"Local URL: http://localhost:{port}")
    print("Press Ctrl+C to stop")

    # Start HTTP server
    httpd = HTTPServer(server_address, HorizontalPhoneHandler)

    # Start ngrok if available
    if NGROK_AVAILABLE:
        try:
            ngrok.set_auth_token(NGROK_AUTHTOKEN)
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.url()
            print(f"Public URL: {public_url}")
            print("\n" + "="*60)
            print("🤖 HORIZONTAL ROBOT CONTROL")
            print("="*60)
            print(f"📱 Phone URL: {public_url}")
            print("🔄 Gyroscope rotated 90° left for horizontal grip")
            print("🕹️ Optimized for landscape orientation")
            print("="*60)
        except Exception as e:
            print(f"Ngrok error: {e}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()
        if NGROK_AVAILABLE:
            ngrok.disconnect(public_url)


if __name__ == "__main__":
    main()