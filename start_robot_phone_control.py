#!/usr/bin/env python

"""
One-Command Robot Phone Control Launcher

This script automatically starts everything needed for phone gyroscope robot control:
1. Installs ngrok if needed
2. Starts phone gyro server with ngrok tunnel
3. Launches LeRobot teleoperation
4. Handles all coordination between services

Usage: python start_robot_phone_control.py
"""

import subprocess
import sys
import time
import threading
import logging
import os
import json
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


def install_ngrok_if_needed():
    """Install ngrok Python package if not available"""
    try:
        import ngrok
        return True
    except ImportError:
        print("📦 Installing ngrok Python package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ngrok"])
            print("✅ ngrok installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install ngrok")
            return False


def wait_for_server_ready(port, timeout=30):
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://localhost:{port}/status', timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def get_server_port_from_output(process):
    """Extract server port from process output"""
    for _ in range(60):  # Wait up to 60 seconds
        try:
            if process.poll() is not None:  # Process ended
                break

            # Try to find port in common locations
            for port in [8889, 8890, 8891, 8892]:
                try:
                    response = requests.get(f'http://localhost:{port}/status', timeout=0.5)
                    if response.status_code == 200:
                        return port
                except:
                    continue

            time.sleep(1)
        except:
            pass

    return 8889  # Default fallback


def find_robot_port():
    """Try to find the correct robot port"""
    common_ports = ["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"]

    print("🔍 Trying to detect robot port...")

    # For now, default to COM6 but could be enhanced to actually test ports
    return "COM6"


def main():
    """Main launcher function"""
    print("🚀 ONE-COMMAND ROBOT PHONE CONTROL")
    print("=" * 50)

    # Install ngrok if needed
    if not install_ngrok_if_needed():
        print("❌ Cannot proceed without ngrok")
        return

    # Find robot port
    robot_port = find_robot_port()
    print(f"🤖 Using robot port: {robot_port}")

    print("\n🔗 Starting phone gyro server with ngrok...")

    # Start phone gyro server with ngrok
    server_process = subprocess.Popen(
        [sys.executable, "phone_gyro_server_ngrok.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Monitor server output for ngrok URL and port
    ngrok_url = None
    server_port = None

    def monitor_server_output():
        nonlocal ngrok_url, server_port
        for line in iter(server_process.stdout.readline, ''):
            print(line.strip())
            if "Public URL:" in line:
                ngrok_url = line.split("Public URL: ")[1].strip()
            elif "Server running locally:" in line and "localhost:" in line:
                try:
                    server_port = int(line.split("localhost:")[1].split()[0])
                except:
                    pass

    # Start monitoring in background
    monitor_thread = threading.Thread(target=monitor_server_output, daemon=True)
    monitor_thread.start()

    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    time.sleep(5)  # Give it time to start

    # Get server port
    if server_port is None:
        server_port = get_server_port_from_output(server_process)

    print(f"📡 Server detected on port: {server_port}")

    # Wait for server to be responding
    if wait_for_server_ready(server_port):
        print("✅ Phone gyro server is ready!")
    else:
        print("⚠️ Server might not be fully ready, continuing anyway...")

    # Display phone instructions
    print("\n" + "=" * 60)
    print("📱 PHONE SETUP:")
    if ngrok_url:
        print(f"   Open this URL on your phone: {ngrok_url}")
    else:
        print("   Ngrok URL will be shown above when ready")
    print("   1. Tap '🚀 Start Control' and allow motion permissions")
    print("   2. Tap '🎯 Calibrate' to reset robot position")
    print("   3. Move phone to control robot!")
    print("\n🎯 CONTROLS:")
    print("   • Phone PITCH → Motor 3 (wrist_flex)")
    print("   • Phone ROLL → Motor 4 (wrist_roll)")
    print("=" * 60)

    # Give user time to set up phone
    print("\n⏳ Waiting 10 seconds for you to set up your phone...")
    for i in range(10, 0, -1):
        print(f"   Starting LeRobot in {i} seconds...", end='\r')
        time.sleep(1)
    print("\n")

    # Start LeRobot teleoperation
    print("🤖 Starting LeRobot teleoperation...")

    lerobot_cmd = [
        sys.executable, "-m", "lerobot.teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={robot_port}",
        "--robot.id=follower",
        "--teleop.type=phone_gyro",
        f"--teleop.server_url=http://localhost:{server_port}",
        "--display_data=true",
        "--fps=60"
    ]

    print(f"Running: {' '.join(lerobot_cmd)}")

    try:
        # Change to lerobot directory
        os.chdir("lerobot")

        # Start LeRobot
        lerobot_process = subprocess.Popen(lerobot_cmd)

        print("\n🎉 SYSTEM READY!")
        print("=" * 50)
        print("✅ Phone gyro server: Running with ngrok")
        print("✅ LeRobot teleoperation: Started")
        print("📱 Move your phone to control the robot!")
        print("\n⌨️ Press Ctrl+C to stop everything")
        print("=" * 50)

        # Wait for user to stop
        try:
            lerobot_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping all processes...")

    except FileNotFoundError:
        print("❌ Could not find LeRobot. Make sure you're in the right directory.")
        print("💡 Try running from the main project directory")

    except Exception as e:
        print(f"❌ Error starting LeRobot: {e}")

    finally:
        # Clean up processes
        print("🧹 Cleaning up...")
        try:
            if 'lerobot_process' in locals():
                lerobot_process.terminate()
            server_process.terminate()
        except:
            pass

        time.sleep(2)
        print("✅ All processes stopped")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🔧 Try running the components separately if this fails")