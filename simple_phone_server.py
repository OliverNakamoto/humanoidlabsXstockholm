#!/usr/bin/env python
"""
Simple Phone Gyro Server for Testing
A basic HTTP server that provides phone gyroscope data for robot control.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import numpy as np


class PhoneGyroData:
    def __init__(self):
        self.lock = threading.Lock()
        self.orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}  # Default robot workspace center
        self.acceleration = {'x': 0.0, 'y': 0.0, 'z': 0.0}  # Accelerometer data
        self.joystick = {'x': 0.0, 'y': 0.0}  # Virtual joystick position
        self.valid = False
        self.last_update = 0
        self.calibration_offset = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.accel_calibration_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.gripper_closed = False  # Gripper state: True = closed, False = open
        self.gripper_last_update = 0

    def update(self, alpha, beta, gamma, accel_x=None, accel_y=None, accel_z=None, joystick_x=None, joystick_y=None):
        with self.lock:
            # Convert device orientation to robot orientation
            # Device: alpha=yaw, beta=pitch, gamma=roll
            raw_roll = gamma if gamma is not None else 0.0
            raw_pitch = beta if beta is not None else 0.0
            raw_yaw = alpha if alpha is not None else 0.0

            # Apply calibration offset
            self.orientation = {
                'roll': raw_roll - self.calibration_offset['roll'],
                'pitch': raw_pitch - self.calibration_offset['pitch'],
                'yaw': raw_yaw - self.calibration_offset['yaw']
            }

            # Update acceleration data if provided
            if accel_x is not None and accel_y is not None and accel_z is not None:
                self.acceleration = {
                    'x': accel_x - self.accel_calibration_offset['x'],
                    'y': accel_y - self.accel_calibration_offset['y'],
                    'z': accel_z - self.accel_calibration_offset['z']
                }

            # Update joystick data if provided
            if joystick_x is not None and joystick_y is not None:
                # Normalize joystick values to -1 to 1 range
                # Input is in cm (-15 to +15), output should be -1 to 1
                self.joystick = {
                    'x': np.clip(joystick_x / 15.0, -1.0, 1.0),
                    'y': np.clip(joystick_y / 15.0, -1.0, 1.0)
                }

            # Simple position mapping from orientation
            # This is a basic mapping - roll affects X, pitch affects Y
            roll_rad = np.radians(self.orientation['roll'])
            pitch_rad = np.radians(self.orientation['pitch'])

            self.position = {
                'x': np.sin(roll_rad) * 0.1,  # Small movement range
                'y': 0.2 + np.sin(pitch_rad) * 0.1,
                'z': 0.15 + np.cos(pitch_rad) * 0.05
            }

            self.valid = True
            self.last_update = time.time()

    def calibrate(self):
        with self.lock:
            # Store current orientation and acceleration as calibration offset
            self.calibration_offset = self.orientation.copy()
            self.accel_calibration_offset = self.acceleration.copy()
            return True

    def set_gripper(self, closed):
        with self.lock:
            self.gripper_closed = closed
            self.gripper_last_update = time.time()
            return True

    def get_status(self):
        with self.lock:
            age = time.time() - self.last_update
            gripper_age = time.time() - self.gripper_last_update
            return {
                'valid': self.valid and age < 1.0,  # Data is stale after 1 second
                'orientation': self.orientation.copy(),
                'position': self.position.copy(),
                'acceleration': self.acceleration.copy(),
                'joystick': self.joystick.copy(),
                'age': age,
                'gripper_closed': self.gripper_closed,
                'gripper_age': gripper_age
            }


class PhoneGyroHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/':
            self.serve_html()
        elif path == '/status':
            self.serve_status()
        elif path == '/calibrate':
            self.serve_calibrate()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/update':
            self.handle_update()
        elif path == '/gripper':
            self.handle_gripper()
        else:
            self.send_error(404, "Not Found")

    def serve_html(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phone Gyro Control</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 20px; background-color: #f0f0f0; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .button { padding: 15px 30px; margin: 10px; font-size: 18px; border: none; border-radius: 5px; cursor: pointer; }
        .start-button { background-color: #4CAF50; color: white; }
        .stop-button { background-color: #f44336; color: white; }
        .calibrate-button { background-color: #2196F3; color: white; }
        .gripper-button { background-color: #FF9800; color: white; font-weight: bold; }
        .gripper-button:active { background-color: #E65100; transform: scale(0.95); }
        .gripper-closed { background-color: #E65100; }
        .gripper-open { background-color: #FF9800; }
        .joystick-container {
            position: relative;
            width: 200px;
            height: 200px;
            margin: 20px auto;
            background: radial-gradient(circle, #ddd 40%, #bbb 100%);
            border-radius: 50%;
            border: 3px solid #888;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
        }
        .joystick-knob {
            position: absolute;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle, #4CAF50 0%, #2E7D32 100%);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            cursor: pointer;
            border: 3px solid #1B5E20;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transition: transform 0.1s ease;
        }
        .joystick-knob:active { transform: translate(-50%, -50%) scale(1.1); }
        .joystick-display { text-align: center; margin-top: 10px; font-weight: bold; }
        .data-box { background-color: #e7e7e7; padding: 15px; margin: 10px 0; border-radius: 5px; text-align: left; }
        .motor-display { display: inline-block; margin: 5px; padding: 10px; background-color: #ddd; border-radius: 5px; min-width: 120px; }
        .status { font-size: 14px; color: #666; margin: 10px 0; }
        .connected { color: green; }
        .disconnected { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Phone Gyro Robot Control</h1>
        <p>Control your robot using phone gyroscope with gripper button!</p>

        <div class="status" id="status">Waiting for motion permission...</div>

        <button class="button start-button" onclick="startControl()">Start Control</button>
        <button class="button stop-button" onclick="stopControl()">Stop Control</button>
        <button class="button calibrate-button" onclick="calibrate()">Calibrate</button>

        <div style="margin: 20px 0;">
            <button class="button gripper-button gripper-open" id="gripperButton"
                    ontouchstart="closeGripper()" ontouchend="openGripper()"
                    onmousedown="closeGripper()" onmouseup="openGripper()">
                GRIPPER - HOLD TO CLOSE
            </button>
            <p><small>Hold button to close gripper, release to open</small></p>
        </div>

        <div class="data-box">
            <h3>Position Joystick (Inverse Kinematics)</h3>
            <div class="joystick-container" id="joystickContainer">
                <div class="joystick-knob" id="joystickKnob"></div>
            </div>
            <div class="joystick-display">
                <p>X: <span id="joystickX">0.0</span> cm | Y: <span id="joystickY">0.0</span> cm</p>
                <p><small>Drag joystick to control robot X,Y position with IK</small></p>
            </div>
        </div>

        <div class="data-box">
            <h3>Orientation Data</h3>
            <p><strong>Roll:</strong> <span id="roll">0.0</span>&deg; (left/right tilt)</p>
            <p><strong>Pitch:</strong> <span id="pitch">0.0</span>&deg; (forward/back tilt)</p>
            <p><strong>Yaw:</strong> <span id="yaw">0.0</span>&deg; (rotation)</p>
        </div>

        <div class="data-box">
            <h3>Acceleration Data</h3>
            <p><strong>Accel X:</strong> <span id="accelX">0.0</span> m/s² (left/right movement)</p>
            <p><strong>Accel Y:</strong> <span id="accelY">0.0</span> m/s² (forward/back movement)</p>
            <p><strong>Accel Z:</strong> <span id="accelZ">0.0</span> m/s² (up/down movement)</p>
        </div>

        <div class="data-box">
            <h3>Robot Motors</h3>
            <div class="motor-display">
                <strong>Motor 0</strong><br>
                (Shoulder Pan)<br>
                <span id="motor0">0.0</span>
            </div>
            <div class="motor-display">
                <strong>Motor 3</strong><br>
                (Wrist Flex)<br>
                <span id="motor3">0.0</span>
            </div>
            <div class="motor-display">
                <strong>Motor 4</strong><br>
                (Wrist Roll)<br>
                <span id="motor4">0.0</span>
            </div>
            <div class="motor-display">
                <strong>Gripper</strong><br>
                (Close/Open)<br>
                <span id="gripper" style="font-weight: bold;">OPEN</span>
            </div>
        </div>

        <p><small>Acceleration X → Motor 0, Pitch → Motor 3, Roll → Motor 4, Button → Gripper</small></p>
    </div>

    <script>
        let isRunning = false;
        let updateInterval;
        let currentAcceleration = { x: 0, y: 0, z: 0 };
        let joystickPosition = { x: 0, y: 0 }; // Joystick position in cm
        let isJoystickActive = false;
        let joystickContainer = null;
        let joystickKnob = null;

        function updateDisplay() {
            if (!isRunning) return;

            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('roll').textContent = data.orientation.roll.toFixed(1);
                    document.getElementById('pitch').textContent = data.orientation.pitch.toFixed(1);
                    document.getElementById('yaw').textContent = data.orientation.yaw.toFixed(1);

                    // Display acceleration data
                    document.getElementById('accelX').textContent = data.acceleration.x.toFixed(2);
                    document.getElementById('accelY').textContent = data.acceleration.y.toFixed(2);
                    document.getElementById('accelZ').textContent = data.acceleration.z.toFixed(2);

                    // Motor mapping: accel X -> motor 0, pitch -> motor 3 (inverted), roll -> motor 4
                    const motor0 = Math.round(data.acceleration.x * 10); // Acceleration X controls shoulder pan
                    const motor3 = Math.round(-data.orientation.pitch * 2); // Inverted: forward tilt = negative
                    const motor4 = Math.round(data.orientation.roll * 2);

                    document.getElementById('motor0').textContent = motor0;
                    document.getElementById('motor3').textContent = motor3;
                    document.getElementById('motor4').textContent = motor4;

                    // Update gripper display
                    const gripperEl = document.getElementById('gripper');
                    if (data.gripper_closed) {
                        gripperEl.textContent = 'CLOSED';
                        gripperEl.style.color = '#E65100';
                    } else {
                        gripperEl.textContent = 'OPEN';
                        gripperEl.style.color = '#FF9800';
                    }

                    const statusEl = document.getElementById('status');
                    if (data.valid) {
                        statusEl.textContent = 'Connected - Sending data to robot';
                        statusEl.className = 'status connected';
                    } else {
                        statusEl.textContent = 'Disconnected - No recent data';
                        statusEl.className = 'status disconnected';
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    document.getElementById('status').textContent = 'Error connecting to server';
                });
        }

        function sendUpdate(alpha, beta, gamma) {
            if (!isRunning) return;

            fetch('/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    alpha: alpha,
                    beta: beta,
                    gamma: gamma,
                    accel_x: currentAcceleration.x,
                    accel_y: currentAcceleration.y,
                    accel_z: currentAcceleration.z,
                    joystick_x: joystickPosition.x,
                    joystick_y: joystickPosition.y
                })
            }).catch(error => console.error('Error sending update:', error));
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
            document.getElementById('status').textContent = 'Starting gyroscope and accelerometer...';

            window.addEventListener('deviceorientation', handleOrientation);
            window.addEventListener('devicemotion', handleMotion);
            updateInterval = setInterval(updateDisplay, 100); // Update display 10 times per second

            setTimeout(() => {
                if (isRunning) {
                    document.getElementById('status').textContent = 'Gyroscope & accelerometer active - Move phone to control robot';
                }
            }, 1000);
        }

        function stopControl() {
            isRunning = false;
            window.removeEventListener('deviceorientation', handleOrientation);
            window.removeEventListener('devicemotion', handleMotion);
            if (updateInterval) clearInterval(updateInterval);
            document.getElementById('status').textContent = 'Control stopped';
            document.getElementById('status').className = 'status disconnected';
        }

        function handleOrientation(event) {
            sendUpdate(event.alpha, event.beta, event.gamma);
        }

        function handleMotion(event) {
            if (event.accelerationIncludingGravity) {
                currentAcceleration.x = event.accelerationIncludingGravity.x || 0;
                currentAcceleration.y = event.accelerationIncludingGravity.y || 0;
                currentAcceleration.z = event.accelerationIncludingGravity.z || 0;
            }
        }

        function initJoystick() {
            joystickContainer = document.getElementById('joystickContainer');
            joystickKnob = document.getElementById('joystickKnob');

            // Mouse events
            joystickKnob.addEventListener('mousedown', startJoystick);
            document.addEventListener('mousemove', moveJoystick);
            document.addEventListener('mouseup', stopJoystick);

            // Touch events for mobile
            joystickKnob.addEventListener('touchstart', startJoystick);
            document.addEventListener('touchmove', moveJoystick);
            document.addEventListener('touchend', stopJoystick);
        }

        function startJoystick(event) {
            event.preventDefault();
            isJoystickActive = true;
            joystickKnob.style.transition = 'none';
        }

        function moveJoystick(event) {
            if (!isJoystickActive) return;
            event.preventDefault();

            const rect = joystickContainer.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            let clientX, clientY;
            if (event.touches) {
                clientX = event.touches[0].clientX;
                clientY = event.touches[0].clientY;
            } else {
                clientX = event.clientX;
                clientY = event.clientY;
            }

            const deltaX = clientX - centerX;
            const deltaY = clientY - centerY;
            const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
            const maxRadius = 60; // Maximum movement radius

            let finalX = deltaX;
            let finalY = deltaY;

            if (distance > maxRadius) {
                finalX = (deltaX / distance) * maxRadius;
                finalY = (deltaY / distance) * maxRadius;
            }

            // Update knob position
            joystickKnob.style.transform = `translate(-50%, -50%) translate(${finalX}px, ${finalY}px)`;

            // Convert to robot coordinates (cm)
            joystickPosition.x = (finalX / maxRadius) * 15; // ±15cm range
            joystickPosition.y = -(finalY / maxRadius) * 15; // Invert Y for robot coords

            // Update display
            document.getElementById('joystickX').textContent = joystickPosition.x.toFixed(1);
            document.getElementById('joystickY').textContent = joystickPosition.y.toFixed(1);
        }

        function stopJoystick(event) {
            if (!isJoystickActive) return;
            event.preventDefault();
            isJoystickActive = false;

            // Return to center with smooth transition
            joystickKnob.style.transition = 'transform 0.3s ease';
            joystickKnob.style.transform = 'translate(-50%, -50%)';

            // Reset position
            joystickPosition.x = 0;
            joystickPosition.y = 0;

            // Update display
            document.getElementById('joystickX').textContent = '0.0';
            document.getElementById('joystickY').textContent = '0.0';
        }

        function calibrate() {
            fetch('/calibrate')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Robot position calibrated!');
                    } else {
                        alert('Calibration failed: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Calibration error:', error);
                    alert('Calibration error - see console');
                });
        }

        function closeGripper() {
            const button = document.getElementById('gripperButton');
            button.className = 'button gripper-button gripper-closed';
            button.textContent = 'GRIPPER - CLOSING';

            fetch('/gripper', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ closed: true })
            }).catch(error => console.error('Gripper error:', error));
        }

        function openGripper() {
            const button = document.getElementById('gripperButton');
            button.className = 'button gripper-button gripper-open';
            button.textContent = 'GRIPPER - HOLD TO CLOSE';

            fetch('/gripper', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ closed: false })
            }).catch(error => console.error('Gripper error:', error));
        }

        // Initialize joystick and start updating display
        initJoystick();
        setInterval(updateDisplay, 500);
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

    def handle_update(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode())
            alpha = data.get('alpha')
            beta = data.get('beta')
            gamma = data.get('gamma')
            accel_x = data.get('accel_x')
            accel_y = data.get('accel_y')
            accel_z = data.get('accel_z')
            joystick_x = data.get('joystick_x')
            joystick_y = data.get('joystick_y')

            gyro_data.update(alpha, beta, gamma, accel_x, accel_y, accel_z, joystick_x, joystick_y)

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

    def handle_gripper(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode())
            closed = data.get('closed', False)

            gyro_data.set_gripper(closed)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'gripper_closed': closed}).encode())

        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def log_message(self, format, *args):
        # Suppress default HTTP server logging
        pass


# Global data storage
gyro_data = PhoneGyroData()


def main():
    port = 8889
    server_address = ('', port)

    print(f"Starting Phone Gyro Server on port {port}...")
    print(f"Open http://localhost:{port} on your phone")
    print("Press Ctrl+C to stop")

    httpd = HTTPServer(server_address, PhoneGyroHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == '__main__':
    main()