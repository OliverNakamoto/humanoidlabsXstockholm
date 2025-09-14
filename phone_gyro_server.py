#!/usr/bin/env python

import logging
import json
import time
import threading
from typing import Dict, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

logger = logging.getLogger(__name__)

class PhoneGyroData:
    """Thread-safe storage for phone gyroscope data"""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        """Reset all gyro data to defaults"""
        with self.lock:
            # Position control (integrated from gyro rates)
            self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}  # Robot workspace center

            # Orientation control (direct from phone orientation)
            self.orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

            # Raw gyroscope data (angular velocities)
            self.gyro = {'x': 0.0, 'y': 0.0, 'z': 0.0}

            # Raw accelerometer (for orientation reference)
            self.accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}

            # Joystick data
            self.joystick = {'x': 0.0, 'y': 0.0}

            # Gripper state
            self.gripper_closed = False

            # Control parameters
            self.valid = False
            self.timestamp = time.time()
            self.calibrated = False
            self.reference_orientation = None

    def update(self, gyro_data: dict):
        """Update gyroscope data from phone"""
        with self.lock:
            if 'gyroscope' in gyro_data:
                self.gyro = gyro_data['gyroscope']

            if 'accelerometer' in gyro_data:
                self.accel = gyro_data['accelerometer']

            if 'orientation' in gyro_data:
                self.orientation = gyro_data['orientation']

            if 'joystick' in gyro_data:
                self.joystick = gyro_data['joystick']

            if 'gripper_closed' in gyro_data:
                self.gripper_closed = gyro_data['gripper_closed']

            # Integrate gyroscope for position control
            dt = time.time() - self.timestamp
            if dt > 0 and self.calibrated:
                # Use gyro Y (phone tilt) for Z movement (up/down)
                # Use gyro X (phone roll) for X movement (left/right)
                # Use gyro Z (phone yaw) for Y movement (forward/back)

                position_scale = 0.01  # Scale factor for integration
                self.position['x'] += self.gyro['x'] * dt * position_scale
                self.position['y'] += self.gyro['z'] * dt * position_scale
                self.position['z'] += -self.gyro['y'] * dt * position_scale  # Invert Y

                # Keep within reasonable workspace bounds
                self.position['x'] = np.clip(self.position['x'], -0.3, 0.3)
                self.position['y'] = np.clip(self.position['y'], 0.1, 0.4)
                self.position['z'] = np.clip(self.position['z'], 0.05, 0.3)

            self.timestamp = time.time()
            self.valid = True

    def get_current_pose(self) -> Tuple[float, float, float, float, float, float]:
        """Get current pose as (x, y, z, roll, pitch, yaw)"""
        with self.lock:
            if not self.valid:
                # Return default pose
                return (0.0, 0.2, 0.15, 0.0, 0.0, 0.0)

            return (
                self.position['x'],
                self.position['y'],
                self.position['z'],
                self.orientation['roll'],
                self.orientation['pitch'],
                self.orientation['yaw']
            )

    def calibrate(self):
        """Calibrate the gyroscope (reset position to center)"""
        with self.lock:
            # Reset position to workspace center
            self.position = {'x': 0.0, 'y': 0.2, 'z': 0.15}

            # Store reference orientation
            self.reference_orientation = self.orientation.copy()

            self.calibrated = True
            logger.info("Phone gyroscope calibrated")


class PhoneGyroHandler(BaseHTTPRequestHandler):
    """HTTP request handler for phone gyroscope data"""

    def do_GET(self):
        """Handle GET requests - serve the phone control interface"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self.serve_phone_interface()
        elif parsed_path.path == '/status':
            self.serve_status()
        elif parsed_path.path == '/calibrate':
            self.server.gyro_data.calibrate()
            self.send_json_response({'success': True, 'message': 'Calibrated'})
        else:
            self.send_error(404)

    def do_POST(self):
        """Handle POST requests - receive gyroscope data"""
        if self.path == '/gyro':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                gyro_data = json.loads(post_data.decode('utf-8'))

                # Update gyro data
                self.server.gyro_data.update(gyro_data)

                self.send_json_response({'success': True})

            except Exception as e:
                logger.error(f"Error processing gyro data: {e}")
                self.send_json_response({'success': False, 'error': str(e)}, 400)
        elif self.path == '/joystick':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                joystick_data = json.loads(post_data.decode('utf-8'))

                # Update joystick data
                self.server.gyro_data.update(joystick_data)

                self.send_json_response({'success': True})

            except Exception as e:
                logger.error(f"Error processing joystick data: {e}")
                self.send_json_response({'success': False, 'error': str(e)}, 400)
        else:
            self.send_error(404)

    def serve_phone_interface(self):
        """Serve the HTML interface for phone control"""
        html = self.get_phone_interface_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-Length', len(html.encode()))
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_status(self):
        """Serve current gyroscope status"""
        x, y, z, roll, pitch, yaw = self.server.gyro_data.get_current_pose()

        status = {
            'position': {'x': x, 'y': y, 'z': z},
            'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
            'joystick': self.server.gyro_data.joystick,
            'gripper_closed': self.server.gyro_data.gripper_closed,
            'calibrated': self.server.gyro_data.calibrated,
            'valid': self.server.gyro_data.valid,
            'timestamp': self.server.gyro_data.timestamp
        }

        self.send_json_response(status)

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

    def get_phone_interface_html(self):
        """Generate HTML interface for phone control"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Phone Control</title>
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
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            margin: 5px;
            cursor: pointer;
        }

        button:hover {
            background-color: #45a049;
        }

        .stop-button {
            background-color: #f44336;
        }

        .stop-button:hover {
            background-color: #da190b;
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

        .instructions {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: left;
            font-size: 14px;
        }

        .joystick-container {
            position: relative;
            width: 200px;
            height: 200px;
            margin: 20px auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border: 3px solid rgba(255, 255, 255, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .joystick-knob {
            position: absolute;
            width: 60px;
            height: 60px;
            background: #4CAF50;
            border-radius: 50%;
            border: 3px solid white;
            cursor: grab;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transition: none;
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

        .joystick-labels {
            position: absolute;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.8);
        }

        .joystick-labels.top { top: 10px; left: 50%; transform: translateX(-50%); }
        .joystick-labels.bottom { bottom: 10px; left: 50%; transform: translateX(-50%); }
        .joystick-labels.left { left: 10px; top: 50%; transform: translateY(-50%); }
        .joystick-labels.right { right: 10px; top: 50%; transform: translateY(-50%); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Robot Phone Control</h1>

        <div class="status">
            <h3>Status: <span id="status" class="disconnected">Disconnected</span></h3>
            <div id="position" class="data">Position: x=0, y=0, z=0</div>
            <div id="orientation" class="data">Orientation: roll=0, pitch=0, yaw=0</div>
            <div id="joystick-pos" class="data">Joystick: x=0.00, y=0.00</div>
        </div>

        <div class="joystick-container" id="joystickContainer">
            <div class="crosshair horizontal"></div>
            <div class="crosshair vertical"></div>
            <div class="joystick-labels top">FORWARD</div>
            <div class="joystick-labels bottom">BACK</div>
            <div class="joystick-labels left">LEFT</div>
            <div class="joystick-labels right">RIGHT</div>
            <div class="joystick-knob" id="joystickKnob"></div>
        </div>

        <div class="instructions">
            <h4>🕹️ Joystick Control:</h4>
            <p>Drag the joystick to move robot left/right and forward/back</p>
            <h4>📱 Gyro Control:</h4>
            <ul>
                <li>Phone <strong>pitch/roll</strong> controls wrist orientation</li>
                <li>Tap to toggle gripper (green = closed, red = open)</li>
            </ul>
        </div>

        <div class="controls">
            <button onclick="startControl()">🚀 Start Control</button>
            <button onclick="calibrate()">🎯 Calibrate</button>
            <button onclick="toggleGripper()" id="gripperBtn">✋ Gripper: Open</button>
            <button onclick="stopControl()" class="stop-button">⛔ Stop</button>
        </div>

        <div id="debug" class="data"></div>
    </div>

    <script>
        let isControlling = false;
        let gyroInterval = null;
        let gripperClosed = false;

        // Joystick variables
        let isDragging = false;
        let joystickX = 0;
        let joystickY = 0;
        let containerRect = null;
        let containerCenterX = 0;
        let containerCenterY = 0;
        let maxRadius = 0;

        function startControl() {
            if ('DeviceMotionEvent' in window) {
                // Request permission on iOS
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

            // Initialize joystick
            initJoystick();

            // Start listening to device motion
            window.addEventListener('devicemotion', handleMotion);

            // Start periodic updates
            gyroInterval = setInterval(updateStatus, 100);
        }

        function stopControl() {
            isControlling = false;
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').className = 'disconnected';

            window.removeEventListener('devicemotion', handleMotion);

            if (gyroInterval) {
                clearInterval(gyroInterval);
                gyroInterval = null;
            }
        }

        function handleMotion(event) {
            if (!isControlling) return;

            const acceleration = event.accelerationIncludingGravity;
            const rotationRate = event.rotationRate;

            // Calculate orientation from accelerometer
            const roll = Math.atan2(acceleration.y, acceleration.z) * 180 / Math.PI;
            const pitch = Math.atan2(-acceleration.x, Math.sqrt(acceleration.y * acceleration.y + acceleration.z * acceleration.z)) * 180 / Math.PI;

            const data = {
                gyroscope: {
                    x: rotationRate ? rotationRate.beta : 0,  // Roll rate
                    y: rotationRate ? rotationRate.alpha : 0, // Yaw rate
                    z: rotationRate ? rotationRate.gamma : 0  // Pitch rate
                },
                accelerometer: {
                    x: acceleration.x,
                    y: acceleration.y,
                    z: acceleration.z
                },
                orientation: {
                    roll: roll,
                    pitch: pitch,
                    yaw: 0  // Can't get yaw from accelerometer alone
                },
                timestamp: Date.now()
            };

            // Send data to server
            fetch('/gyro', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            }).catch(error => {
                console.error('Error sending gyro data:', error);
            });
        }

        function calibrate() {
            fetch('/calibrate')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Calibrated! Robot reset to center position.');
                    }
                })
                .catch(error => {
                    console.error('Error calibrating:', error);
                    alert('Error calibrating');
                });
        }

        function initJoystick() {
            const joystickContainer = document.getElementById('joystickContainer');
            const joystickKnob = document.getElementById('joystickKnob');

            containerRect = joystickContainer.getBoundingClientRect();
            containerCenterX = containerRect.left + containerRect.width / 2;
            containerCenterY = containerRect.top + containerRect.height / 2;
            maxRadius = containerRect.width / 2 - 30;

            // Position knob at center initially
            joystickKnob.style.left = (containerRect.width / 2 - 30) + 'px';
            joystickKnob.style.top = (containerRect.height / 2 - 30) + 'px';

            // Mouse events
            joystickKnob.addEventListener('mousedown', startDrag);
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);

            // Touch events
            joystickKnob.addEventListener('touchstart', startDrag);
            document.addEventListener('touchmove', drag);
            document.addEventListener('touchend', stopDrag);

            // Update on window resize
            window.addEventListener('resize', initJoystick);
        }

        function startDrag(e) {
            isDragging = true;
            document.getElementById('joystickKnob').classList.add('dragging');
            e.preventDefault();
        }

        function drag(e) {
            if (!isDragging) return;

            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;

            const dx = clientX - containerCenterX;
            const dy = clientY - containerCenterY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance <= maxRadius) {
                joystickX = dx / maxRadius;
                joystickY = -dy / maxRadius; // Invert Y for intuitive control
                document.getElementById('joystickKnob').style.left = (containerRect.width / 2 + dx - 30) + 'px';
                document.getElementById('joystickKnob').style.top = (containerRect.height / 2 + dy - 30) + 'px';
            } else {
                const angle = Math.atan2(dy, dx);
                joystickX = Math.cos(angle);
                joystickY = -Math.sin(angle); // Invert Y
                document.getElementById('joystickKnob').style.left = (containerRect.width / 2 + Math.cos(angle) * maxRadius - 30) + 'px';
                document.getElementById('joystickKnob').style.top = (containerRect.height / 2 + Math.sin(angle) * maxRadius - 30) + 'px';
            }

            // Update display and send data
            document.getElementById('joystick-pos').textContent = `Joystick: x=${joystickX.toFixed(2)}, y=${joystickY.toFixed(2)}`;
            sendJoystickData();
        }

        function stopDrag() {
            if (!isDragging) return;

            isDragging = false;
            document.getElementById('joystickKnob').classList.remove('dragging');

            // Reset joystick to center
            joystickX = 0;
            joystickY = 0;
            document.getElementById('joystickKnob').style.left = (containerRect.width / 2 - 30) + 'px';
            document.getElementById('joystickKnob').style.top = (containerRect.height / 2 - 30) + 'px';
            document.getElementById('joystick-pos').textContent = 'Joystick: x=0.00, y=0.00';
            sendJoystickData();
        }

        function sendJoystickData() {
            const data = {
                joystick: {
                    x: joystickX,
                    y: joystickY
                },
                gripper_closed: gripperClosed
            };

            fetch('/joystick', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            }).catch(error => {
                console.error('Error sending joystick data:', error);
            });
        }

        function toggleGripper() {
            gripperClosed = !gripperClosed;
            const btn = document.getElementById('gripperBtn');
            if (gripperClosed) {
                btn.textContent = '✊ Gripper: Closed';
                btn.style.backgroundColor = '#4CAF50';
            } else {
                btn.textContent = '✋ Gripper: Open';
                btn.style.backgroundColor = '#f44336';
            }
            sendJoystickData();
        }

        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('position').textContent =
                        `Position: x=${data.position.x.toFixed(3)}, y=${data.position.y.toFixed(3)}, z=${data.position.z.toFixed(3)}`;

                    document.getElementById('orientation').textContent =
                        `Orientation: roll=${data.orientation.roll.toFixed(1)}°, pitch=${data.orientation.pitch.toFixed(1)}°, yaw=${data.orientation.yaw.toFixed(1)}°`;
                })
                .catch(error => {
                    console.error('Error getting status:', error);
                });
        }
    </script>
</body>
</html>'''


class PhoneGyroServer(HTTPServer):
    """HTTP server for phone gyroscope control"""

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.gyro_data = PhoneGyroData()


def start_phone_gyro_server(port=8889):
    """Start the phone gyroscope server"""
    logging.basicConfig(level=logging.INFO)

    server = PhoneGyroServer(('0.0.0.0', port), PhoneGyroHandler)

    logger.info(f"Phone gyro server starting on port {port}")
    logger.info(f"Open http://localhost:{port} on your phone to control the robot")
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopping...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    start_phone_gyro_server()