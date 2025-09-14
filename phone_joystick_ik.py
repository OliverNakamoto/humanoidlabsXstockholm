#!/usr/bin/env python

"""
Phone Joystick Server with Incremental IK Control
Based on keyboard_teleop approach for smooth robot control
"""

import os
import logging
import json
import time
import threading
import numpy as np
from typing import Dict, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import IK functions from keyboard_teleop
import sys
sys.path.append('keyboard_teleop')
from forward_kinematics import forward_kinematics
from inverse_kinematics import iterative_ik

# Import ngrok
try:
    import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("WARNING: ngrok not found. Install with: pip install ngrok")

logger = logging.getLogger(__name__)
NGROK_AUTHTOKEN = "32fKQf5vxHcZUZeBtDB6ADWsUuz_6ywRPi5gHBgebtet7CGdS"


class RobotIKController:
    """Handles IK calculations and robot state"""

    def __init__(self):
        self.lock = threading.Lock()

        # Current joint angles (degrees)
        self.current_angles = [0.0, 0.0, 0.0, 0.0]  # 4 main joints

        # Current end-effector position
        self.ef_position = np.array([0.2, 0.0, 0.15])  # Default start position
        self.ef_pitch = 90.0  # Default pitch

        # Movement velocities (m/s)
        self.velocity = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # Joystick input (-1 to 1)
        self.joystick = {'x': 0.0, 'y': 0.0}

        # Movement parameters
        self.max_velocity = 0.05  # m/s
        self.step_size = 0.002  # meters per update

        # Initialize with forward kinematics
        self.update_from_angles()

    def update_from_angles(self):
        """Update end-effector position from current angles"""
        with self.lock:
            pos, rpy = forward_kinematics(*self.current_angles)
            self.ef_position = pos
            self.ef_pitch = rpy[1]  # pitch component

    def set_joystick(self, x: float, y: float):
        """Set joystick position (-1 to 1 range)"""
        with self.lock:
            self.joystick['x'] = np.clip(x, -1, 1)
            self.joystick['y'] = np.clip(y, -1, 1)

            # Convert joystick to velocity
            # Joystick Y -> Robot X (forward/back)
            # Joystick X -> Robot Y (left/right)
            self.velocity['x'] = self.joystick['y'] * self.step_size
            self.velocity['y'] = -self.joystick['x'] * self.step_size  # Negative for intuitive left/right
            self.velocity['z'] = 0.0  # No Z control from joystick

    def update_position(self) -> Dict[str, float]:
        """Update end-effector position based on velocity and compute IK"""
        with self.lock:
            # Update end-effector position with velocity
            self.ef_position[0] += self.velocity['x']
            self.ef_position[1] += self.velocity['y']
            self.ef_position[2] += self.velocity['z']

            # Clamp to workspace limits
            self.ef_position[0] = np.clip(self.ef_position[0], 0.05, 0.35)
            self.ef_position[1] = np.clip(self.ef_position[1], -0.2, 0.2)
            self.ef_position[2] = np.clip(self.ef_position[2], 0.05, 0.25)

            # Compute IK
            try:
                new_angles = iterative_ik(
                    self.ef_position,
                    self.ef_pitch,
                    self.current_angles,
                    max_iter=100,
                    alpha=0.5
                )

                # Check for sudden jumps
                max_angle_change = 30  # degrees
                angle_changes = np.abs(np.array(new_angles) - np.array(self.current_angles))

                if np.all(angle_changes < max_angle_change):
                    self.current_angles = list(new_angles)
                else:
                    print(f"IK jump detected, skipping update")

            except Exception as e:
                print(f"IK error: {e}")

            # Convert angles to servo positions (-100 to 100 range)
            servo_positions = {}

            # Map angles to servo positions
            # These mappings may need adjustment based on your robot
            servo_positions['shoulder_pan.pos'] = float(np.clip(self.current_angles[0] * 100/90, -100, 100))
            servo_positions['shoulder_lift.pos'] = float(np.clip(self.current_angles[1] * 100/90, -100, 100))
            servo_positions['elbow_flex.pos'] = float(np.clip(self.current_angles[2] * 100/90, -100, 100))
            servo_positions['wrist_flex.pos'] = float(np.clip(self.current_angles[3] * 100/90, -100, 100))
            servo_positions['wrist_roll.pos'] = 0.0  # Not controlled by IK
            servo_positions['gripper.pos'] = 0.0  # Not controlled by IK

            return servo_positions

    def get_status(self) -> Dict:
        """Get current controller status"""
        with self.lock:
            return {
                'ef_position': self.ef_position.tolist(),
                'ef_pitch': self.ef_pitch,
                'angles': self.current_angles,
                'velocity': self.velocity.copy(),
                'joystick': self.joystick.copy()
            }


# Global controller instance
ik_controller = RobotIKController()


class PhoneJoystickHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.serve_html()
        elif self.path == '/status':
            self.serve_status()
        elif self.path == '/robot_action':
            self.serve_robot_action()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == '/joystick':
            self.handle_joystick()
        else:
            self.send_error(404, "Not Found")

    def serve_html(self):
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Robot IK Joystick Control</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            touch-action: none;
        }

        .container {
            width: 100%;
            max-width: 400px;
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 24px;
        }

        .joystick-container {
            position: relative;
            width: 300px;
            height: 300px;
            margin: 0 auto 30px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border: 3px solid rgba(255, 255, 255, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .joystick-knob {
            position: absolute;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle, #4CAF50, #2E7D32);
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

        .axis-label {
            position: absolute;
            font-size: 14px;
            font-weight: bold;
            color: rgba(255, 255, 255, 0.8);
        }

        .axis-label.top { top: 10px; left: 50%; transform: translateX(-50%); }
        .axis-label.bottom { bottom: 10px; left: 50%; transform: translateX(-50%); }
        .axis-label.left { left: 10px; top: 50%; transform: translateY(-50%); }
        .axis-label.right { right: 10px; top: 50%; transform: translateY(-50%); }

        .status-panel {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
        }

        .status-value {
            font-family: 'Courier New', monospace;
            color: #4CAF50;
        }

        .info-text {
            text-align: center;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Robot IK Joystick Control</h1>

        <div class="joystick-container" id="joystickContainer">
            <div class="crosshair horizontal"></div>
            <div class="crosshair vertical"></div>
            <div class="axis-label top">FORWARD</div>
            <div class="axis-label bottom">BACKWARD</div>
            <div class="axis-label left">LEFT</div>
            <div class="axis-label right">RIGHT</div>
            <div class="joystick-knob" id="joystickKnob"></div>
        </div>

        <div class="status-panel">
            <div class="status-item">
                <span>X Position:</span>
                <span class="status-value" id="xPos">0.00</span>
            </div>
            <div class="status-item">
                <span>Y Position:</span>
                <span class="status-value" id="yPos">0.00</span>
            </div>
            <div class="status-item">
                <span>End Effector:</span>
                <span class="status-value" id="efPos">[0.00, 0.00, 0.00]</span>
            </div>
            <div class="status-item">
                <span>Joint Angles:</span>
                <span class="status-value" id="angles">[0, 0, 0, 0]</span>
            </div>
        </div>

        <div class="info-text">
            Drag the joystick to control robot movement<br>
            Uses Inverse Kinematics for smooth end-effector control
        </div>
    </div>

    <script>
        const joystickKnob = document.getElementById('joystickKnob');
        const joystickContainer = document.getElementById('joystickContainer');

        let isDragging = false;
        let containerRect = joystickContainer.getBoundingClientRect();
        let containerCenterX = containerRect.left + containerRect.width / 2;
        let containerCenterY = containerRect.top + containerRect.height / 2;
        let maxRadius = containerRect.width / 2 - 40;

        // Current joystick position
        let joystickX = 0;
        let joystickY = 0;

        // Update container dimensions on resize
        window.addEventListener('resize', () => {
            containerRect = joystickContainer.getBoundingClientRect();
            containerCenterX = containerRect.left + containerRect.width / 2;
            containerCenterY = containerRect.top + containerRect.height / 2;
            maxRadius = containerRect.width / 2 - 40;
        });

        function updateJoystickPosition(clientX, clientY) {
            const dx = clientX - containerCenterX;
            const dy = clientY - containerCenterY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance <= maxRadius) {
                joystickX = dx / maxRadius;
                joystickY = -dy / maxRadius;  // Invert Y for intuitive control
                joystickKnob.style.left = (containerRect.width / 2 + dx - 40) + 'px';
                joystickKnob.style.top = (containerRect.height / 2 + dy - 40) + 'px';
            } else {
                const angle = Math.atan2(dy, dx);
                joystickX = Math.cos(angle);
                joystickY = -Math.sin(angle);  // Invert Y
                joystickKnob.style.left = (containerRect.width / 2 + Math.cos(angle) * maxRadius - 40) + 'px';
                joystickKnob.style.top = (containerRect.height / 2 + Math.sin(angle) * maxRadius - 40) + 'px';
            }

            // Update display
            document.getElementById('xPos').textContent = joystickX.toFixed(2);
            document.getElementById('yPos').textContent = joystickY.toFixed(2);

            // Send to server
            sendJoystickData();
        }

        function resetJoystick() {
            joystickX = 0;
            joystickY = 0;
            joystickKnob.style.left = (containerRect.width / 2 - 40) + 'px';
            joystickKnob.style.top = (containerRect.height / 2 - 40) + 'px';
            document.getElementById('xPos').textContent = '0.00';
            document.getElementById('yPos').textContent = '0.00';
            sendJoystickData();
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

        // Send joystick data to server
        function sendJoystickData() {
            fetch('/joystick', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    x: joystickX,
                    y: joystickY
                })
            }).catch(error => console.error('Joystick update error:', error));
        }

        // Update status display
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.ef_position) {
                        const ef = data.ef_position;
                        document.getElementById('efPos').textContent =
                            `[${ef[0].toFixed(3)}, ${ef[1].toFixed(3)}, ${ef[2].toFixed(3)}]`;
                    }
                    if (data.angles) {
                        const angles = data.angles.map(a => a.toFixed(1));
                        document.getElementById('angles').textContent =
                            `[${angles.join(', ')}]`;
                    }
                })
                .catch(error => console.error('Status update error:', error));
        }

        // Initialize joystick position
        resetJoystick();

        // Update status periodically
        setInterval(updateStatus, 100);
    </script>
</body>
</html>'''

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def serve_status(self):
        status = ik_controller.get_status()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def serve_robot_action(self):
        """Get robot action for LeRobot teleoperator"""
        action = ik_controller.update_position()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(action).encode())

    def handle_joystick(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode())
            x = float(data.get('x', 0))
            y = float(data.get('y', 0))

            ik_controller.set_joystick(x, y)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())

        except Exception as e:
            print(f"Joystick error: {e}")
            self.send_error(400, str(e))

    def log_message(self, format, *args):
        pass  # Suppress request logging


def start_server_with_ngrok():
    """Start the server with ngrok tunnel"""

    # Start HTTP server
    server_port = 8080
    server = HTTPServer(('0.0.0.0', server_port), PhoneJoystickHandler)

    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    print(f"Server started on http://localhost:{server_port}")

    # Setup ngrok
    if NGROK_AVAILABLE:
        try:
            # Configure ngrok
            ngrok.set_auth_token(NGROK_AUTHTOKEN)

            # Create tunnel
            tunnel = ngrok.connect(server_port, "http")
            public_url = tunnel.url()

            print("\n" + "="*50)
            print("Robot IK Joystick Control")
            print("="*50)
            print(f"Local URL: http://localhost:{server_port}")
            print(f"Public URL: {public_url}")
            print("="*50)
            print("\nOpen the URL on your phone to control the robot")
            print("Drag the joystick for smooth IK-based movement")
            print("\nPress Ctrl+C to stop")

        except Exception as e:
            print(f"Ngrok error: {e}")
            print("Server running on localhost only")
    else:
        print("Ngrok not available - localhost only")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        if NGROK_AVAILABLE:
            ngrok.disconnect(public_url)


if __name__ == "__main__":
    start_server_with_ngrok()