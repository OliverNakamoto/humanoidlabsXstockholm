#!/usr/bin/env python

"""
Simple runner script for phone joystick with ngrok
"""

import subprocess
import time
import ngrok

def main():
    print("Starting Phone Joystick Server with ngrok...")

    # Set ngrok auth token
    ngrok_token = "32fKQf5vxHcZUZeBtDB6ADWsUuz_6ywRPi5gHBgebtet7CGdS"
    ngrok.get_default().auth_token = ngrok_token

    # Start the server in background
    print("Starting HTTP server...")
    server_process = subprocess.Popen([
        "python", "phone_joystick_ik.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait a moment for server to start
    time.sleep(3)

    # Create ngrok tunnel
    print("Creating ngrok tunnel...")
    try:
        tunnel = ngrok.connect(addr=8080, proto="http")
        public_url = tunnel.public_url

        print("\n" + "="*60)
        print("PHONE JOYSTICK READY!")
        print("="*60)
        print(f"Open this URL on your phone: {public_url}")
        print(f"Local URL: http://localhost:8080")
        print("="*60)
        print("Instructions:")
        print("1. Open the URL on your phone")
        print("2. Drag the joystick to control robot movement")
        print("3. Uses Inverse Kinematics for smooth control")
        print("4. Press Ctrl+C to stop")
        print("="*60)

        # Keep running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server_process.terminate()
            ngrok.disconnect(public_url)

    except Exception as e:
        print(f"Failed to create ngrok tunnel: {e}")
        print("Server is running locally at: http://localhost:8080")
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server_process.terminate()

if __name__ == "__main__":
    main()