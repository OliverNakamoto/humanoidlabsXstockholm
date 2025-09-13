#!/usr/bin/env python3
"""
Hand Tracking Server - Runs MediaPipe in separate environment
Serves hand tracking data via HTTP API to avoid protobuf conflicts
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import cv2

# This runs in MediaPipe environment with protobuf 4.x
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("❌ MediaPipe not available in this environment")

class HandTracker:
    """MediaPipe hand tracking wrapper."""

    def __init__(self, show_video=False):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        self.latest_data = None
        self.running = False
        self.show_video = show_video
        self.calibrated = True  # Always calibrated for now
        
    def start_tracking(self):
        """Start continuous hand tracking in background thread."""
        self.running = True
        self.tracking_thread = Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
    def _tracking_loop(self):
        """Continuous tracking loop."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Flip frame horizontally for selfie-view
            frame = cv2.flip(frame, 1)

            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Draw hand annotations on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.show_video:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Extract hand data from first hand
                landmarks = results.multi_hand_landmarks[0]

                # Get key landmarks
                thumb_tip = landmarks.landmark[4]
                index_tip = landmarks.landmark[8]
                middle_tip = landmarks.landmark[12]
                wrist = landmarks.landmark[0]

                # Calculate grip (distance between thumb and index)
                grip_distance = ((thumb_tip.x - index_tip.x) ** 2 +
                               (thumb_tip.y - index_tip.y) ** 2) ** 0.5

                # Normalize grip (0 = closed, 1 = open)
                grip_normalized = min(1.0, max(0.0, grip_distance * 10))

                # Convert to robot coordinate system (wrist position)
                x = wrist.x  # Left-right (0-1, flipped due to mirror)
                y = 1.0 - wrist.y  # Up-down (inverted: 0=bottom, 1=top)
                z = wrist.z  # Forward-back (relative depth)

                # Convert grip to pinch percentage (0=open, 100=closed)
                pinch = (1.0 - grip_normalized) * 100.0

                self.latest_data = {
                    "hand_detected": True,
                    "x": x,
                    "y": y,
                    "z": z,
                    "pinch": pinch,
                    "valid": True,
                    "wrist": {"x": wrist.x, "y": wrist.y, "z": wrist.z},
                    "thumb": {"x": thumb_tip.x, "y": thumb_tip.y, "z": thumb_tip.z},
                    "index": {"x": index_tip.x, "y": index_tip.y, "z": index_tip.z},
                    "middle": {"x": middle_tip.x, "y": middle_tip.y, "z": middle_tip.z},
                    "grip": grip_normalized,
                    "timestamp": time.time()
                }

                # Add status text to frame
                if self.show_video:
                    cv2.putText(frame, f"Hand: ({x:.2f}, {y:.2f}, {z:.2f})",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pinch: {pinch:.1f}%",
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.latest_data = {
                    "hand_detected": False,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "pinch": 0.0,
                    "valid": False,
                    "timestamp": time.time()
                }

                # Add "No hand detected" text
                if self.show_video:
                    cv2.putText(frame, "No hand detected",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show video window
            if self.show_video:
                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                    self.running = False

            time.sleep(1/30)  # 30 FPS
    
    def get_latest_data(self):
        """Get latest hand tracking data."""
        return self.latest_data
    
    def stop(self):
        """Stop tracking."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

class HandTrackingHandler(BaseHTTPRequestHandler):
    """HTTP handler for hand tracking requests."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/hand_data":
            # Return latest hand data
            data = tracker.get_latest_data() if tracker else {"error": "tracker not initialized"}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # CORS
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        elif self.path == "/position":
            # Return position data (for IPC client compatibility)
            data = tracker.get_latest_data() if tracker else {"error": "tracker not initialized"}

            if data and "error" not in data:
                # Format for IPC client
                position_data = {
                    "x": data.get("x", 0.0),
                    "y": data.get("y", 0.0),
                    "z": data.get("z", 0.0),
                    "pinch": data.get("pinch", 0.0),
                    "valid": data.get("valid", False)
                }
            else:
                position_data = {"x": 0.0, "y": 0.0, "z": 0.0, "pinch": 0.0, "valid": False}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # CORS
            self.end_headers()
            self.wfile.write(json.dumps(position_data).encode())

        elif self.path == "/status":
            # Health check
            status = {
                "status": "running",
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "calibrated": tracker.calibrated if tracker else False,
                "timestamp": time.time()
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())

        elif self.path == "/calibrate":
            # Calibration endpoint (always successful for now)
            response = {"success": True, "message": "Calibration completed"}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

def main():
    """Start hand tracking server."""
    global tracker
    
    print("Hand Tracking Server Starting...")
    print("=" * 40)

    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not available!")
        print("Run in MediaPipe environment:")
        print("pip install mediapipe opencv-python")
        return 1

    # Initialize tracker
    try:
        tracker = HandTracker()
        tracker.start_tracking()
        print("Hand tracking started")
    except Exception as e:
        print(f"Failed to start hand tracking: {e}")
        return 1

    # Start HTTP server
    server_address = ('localhost', 8888)
    httpd = HTTPServer(server_address, HandTrackingHandler)

    print(f"Server running on http://{server_address[0]}:{server_address[1]}")
    print("Endpoints:")
    print("  GET /hand_data  - Get detailed hand data")
    print("  GET /position   - Get position data (IPC client)")
    print("  GET /status     - Server health check")
    print("  GET /calibrate  - Trigger calibration")
    print("\nPress Ctrl+C to stop or ESC in video window")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        tracker.stop()
        httpd.shutdown()

if __name__ == "__main__":
    tracker = None
    main()