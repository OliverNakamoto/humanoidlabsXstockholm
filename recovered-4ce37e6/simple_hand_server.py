#!/usr/bin/env python3
"""
Simple Hand Tracking Server - Basic HTTP server for hand tracking
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Simple mock data for testing
class SimpleHandTracker:
    def __init__(self):
        self.running = False
        self.calibrated = True

    def start_tracking(self):
        self.running = True
        print("Hand tracking started (mock mode)")

    def get_latest_data(self):
        # Return mock data that moves slightly
        t = time.time()
        return {
            "x": 0.5 + 0.1 * (t % 2 - 1),  # Oscillate between 0.4 and 0.6
            "y": 0.5,
            "z": 0.0,
            "pinch": 50.0,
            "valid": True,
            "hand_detected": True,
            "timestamp": t
        }

    def stop(self):
        self.running = False
        print("Hand tracking stopped")

class HandTrackingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/position":
            data = tracker.get_latest_data()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        elif self.path == "/status":
            status = {
                "status": "running",
                "calibrated": True,
                "timestamp": time.time()
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())

        elif self.path == "/calibrate":
            response = {"success": True, "message": "Calibration completed"}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def main():
    global tracker

    print("Simple Hand Tracking Server Starting...")
    print("=" * 40)

    tracker = SimpleHandTracker()
    tracker.start_tracking()

    server_address = ('localhost', 8888)
    httpd = HTTPServer(server_address, HandTrackingHandler)

    print(f"Server running on http://{server_address[0]}:{server_address[1]}")
    print("Endpoints:")
    print("  GET /position   - Get position data")
    print("  GET /status     - Server health check")
    print("  GET /calibrate  - Trigger calibration")
    print("\nPress Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        tracker.stop()
        httpd.shutdown()

if __name__ == "__main__":
    tracker = None
    main()