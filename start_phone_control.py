#!/usr/bin/env python

"""
Script to start phone gyroscope robot control with ngrok tunnel.

This script:
1. Starts the phone gyro server on localhost:8889
2. Creates an ngrok tunnel to expose it to the internet
3. Prints the public URL to access from your phone

Usage:
    python start_phone_control.py

Then open the printed URL on your phone to control the robot.
"""

import subprocess
import time
import threading
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from phone_gyro_server import start_phone_gyro_server

logger = logging.getLogger(__name__)


def start_ngrok_tunnel(port=8889):
    """Start ngrok tunnel and return the public URL"""
    try:
        # Start ngrok tunnel
        logger.info(f"Starting ngrok tunnel for port {port}...")
        process = subprocess.Popen(
            ['ngrok', 'http', str(port), '--log=stdout'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a moment for ngrok to start
        time.sleep(3)

        # Get the tunnel URL
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:4040/api/tunnels'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                import json
                tunnels = json.loads(result.stdout)
                if tunnels.get('tunnels'):
                    public_url = tunnels['tunnels'][0]['public_url']
                    logger.info(f"✅ Ngrok tunnel created: {public_url}")
                    return process, public_url

        except Exception as e:
            logger.warning(f"Could not get tunnel URL automatically: {e}")
            logger.info("Check ngrok dashboard at http://localhost:4040")

        return process, None

    except FileNotFoundError:
        logger.error("❌ ngrok not found! Please install ngrok:")
        logger.error("1. Download from https://ngrok.com/download")
        logger.error("2. Extract to PATH or current directory")
        logger.error("3. Run: ngrok authtoken <your_token>")
        return None, None
    except Exception as e:
        logger.error(f"❌ Failed to start ngrok: {e}")
        return None, None


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    port = 8889

    print("🤖 Starting Phone Gyroscope Robot Control")
    print("=" * 50)

    # Start ngrok tunnel
    ngrok_process, public_url = start_ngrok_tunnel(port)

    if ngrok_process is None:
        print("❌ Failed to start ngrok tunnel")
        print("You can still use localhost if phone and computer are on same network")
        public_url = f"http://localhost:{port}"

    # Start the phone gyro server in a separate thread
    server_thread = threading.Thread(
        target=start_phone_gyro_server,
        args=(port,),
        daemon=True
    )
    server_thread.start()

    # Give server time to start
    time.sleep(2)

    print("\n🎉 Phone Control System Ready!")
    print("=" * 50)

    if public_url:
        print(f"📱 Open this URL on your phone: {public_url}")
    else:
        print(f"📱 Local URL: http://localhost:{port}")
        print("   (if ngrok failed, use this if phone is on same network)")

    print("\n📋 Instructions:")
    print("1. Open the URL on your phone browser")
    print("2. Tap 'Start Control' and allow motion permissions")
    print("3. Tap 'Calibrate' to reset robot to center")
    print("4. Tilt your phone to control the robot!")

    print("\n🎮 Controls:")
    print("• Tilt phone LEFT/RIGHT → Robot moves left/right")
    print("• Tilt phone FORWARD/BACK → Robot moves forward/back")
    print("• Roll phone → Robot moves up/down")
    print("• Phone orientation → End-effector orientation")

    print("\n🚀 Now start the robot teleoperation:")
    print("cd lerobot")
    print("py -m lerobot.teleoperate \\")
    print("  --robot.type=so101_follower \\")
    print("  --robot.port=COM6 \\")  # You'll need to find correct port
    print("  --robot.id=follower \\")
    print("  --teleop.type=phone_gyro \\")
    print("  --display_data=true \\")
    print("  --fps=60")

    print("\n⌨️  Press Ctrl+C to stop")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping phone control system...")

        if ngrok_process:
            ngrok_process.terminate()

        print("✅ Stopped")


if __name__ == "__main__":
    main()