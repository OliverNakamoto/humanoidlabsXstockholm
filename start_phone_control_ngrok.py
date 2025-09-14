#!/usr/bin/env python

"""
Enhanced Phone Control Startup with Ngrok

Now that you have ngrok configured with an authtoken, this script will:
1. Start the Kalman-enhanced phone gyro server
2. Create a reliable ngrok tunnel
3. Display the public URL for your phone
4. Show real-time connection status
"""

import subprocess
import time
import threading
import logging
import sys
import json
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


def get_ngrok_tunnel_info():
    """Get ngrok tunnel information from the local API"""
    try:
        response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
        if response.status_code == 200:
            tunnels = response.json()
            if tunnels.get('tunnels'):
                tunnel = tunnels['tunnels'][0]
                return {
                    'public_url': tunnel['public_url'],
                    'name': tunnel['name'],
                    'proto': tunnel['proto'],
                    'requests': tunnel.get('metrics', {}).get('http', {}).get('count', 0)
                }
    except Exception as e:
        logger.debug(f"Could not get tunnel info: {e}")
    return None


def start_enhanced_gyro_server(port=8889):
    """Start the Kalman-enhanced gyro server"""
    logger.info(f"Starting enhanced gyro server on port {port}...")

    process = subprocess.Popen(
        [sys.executable, 'phone_gyro_server_with_kalman.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Give server time to start
    time.sleep(2)

    # Check if server is responding
    try:
        response = requests.get(f'http://localhost:{port}/status', timeout=2)
        if response.status_code == 200:
            logger.info("✅ Enhanced gyro server started successfully")
            return process
    except:
        pass

    logger.error("❌ Failed to start gyro server")
    return None


def start_ngrok_tunnel(port=8889):
    """Start ngrok tunnel with authenticated token"""
    logger.info(f"Creating ngrok tunnel for port {port}...")

    try:
        # Start ngrok tunnel with your authenticated token
        process = subprocess.Popen(
            ['ngrok', 'http', str(port), '--log=stdout'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for tunnel to establish
        logger.info("Waiting for tunnel to establish...")
        for i in range(10):
            time.sleep(1)
            tunnel_info = get_ngrok_tunnel_info()
            if tunnel_info:
                logger.info(f"✅ Ngrok tunnel created: {tunnel_info['public_url']}")
                return process, tunnel_info['public_url']
            print(f"  Attempt {i+1}/10...")

        logger.warning("⚠️  Tunnel created but couldn't get URL automatically")
        logger.info("Check the ngrok dashboard at http://localhost:4040")
        return process, None

    except FileNotFoundError:
        logger.error("❌ ngrok not found in PATH")
        logger.error("Make sure ngrok is installed and accessible")
        return None, None
    except Exception as e:
        logger.error(f"❌ Failed to start ngrok: {e}")
        return None, None


def monitor_connections(public_url):
    """Monitor connection status"""
    if not public_url:
        return

    logger.info("📊 Starting connection monitor...")
    last_request_count = 0

    while True:
        time.sleep(10)  # Check every 10 seconds

        tunnel_info = get_ngrok_tunnel_info()
        if tunnel_info:
            current_requests = tunnel_info['requests']
            if current_requests > last_request_count:
                new_requests = current_requests - last_request_count
                logger.info(f"📱 {new_requests} new phone connection(s) - Total: {current_requests}")
                last_request_count = current_requests


def print_instructions(public_url, local_url):
    """Print user instructions"""
    print("\n" + "="*60)
    print("🎉 PHONE GYROSCOPE CONTROL SYSTEM READY!")
    print("="*60)

    print("\n📱 PHONE SETUP:")
    if public_url:
        print(f"   Open this URL on your phone: {public_url}")
        print("   (Works from anywhere with internet connection)")
    else:
        print(f"   Fallback URL: {local_url}")
        print("   (Only works if phone is on same WiFi network)")

    print("\n🎮 PHONE CONTROLS:")
    print("   1. Tap 'Start Control' and allow motion permissions")
    print("   2. Tap 'Toggle Kalman' to enable/disable filtering")
    print("   3. Tap 'Calibrate' to reset robot position")
    print("   4. Move phone to see Motor 3 (pitch) and Motor 4 (roll)")

    print("\n🤖 ROBOT CONTROL:")
    print("   cd lerobot")
    print("   py -m lerobot.teleoperate \\")
    print("     --robot.type=so101_follower \\")
    print("     --robot.port=COM6 \\")
    print("     --robot.id=follower \\")
    print("     --teleop.type=phone_gyro \\")
    print("     --display_data=true \\")
    print("     --fps=60")

    print("\n🎯 MOTOR MAPPING:")
    print("   • Phone PITCH ↔ Motor 3 (wrist_flex)")
    print("   • Phone ROLL  ↔ Motor 4 (wrist_roll)")
    print("   • Phone TILT  ↔ Position control (X/Y/Z)")

    print("\n🔬 KALMAN FILTER FEATURES:")
    print("   • Real-time sensor fusion")
    print("   • Noise reduction")
    print("   • Drift compensation")
    print("   • Toggle on/off for comparison")

    if public_url:
        print("\n🌐 NGROK STATUS:")
        print(f"   Public URL: {public_url}")
        print(f"   Dashboard: http://localhost:4040")
        print("   (Monitor connections and requests)")

    print("\n⌨️  Press Ctrl+C to stop all services")
    print("="*60)


def main():
    """Main function to start everything"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("🚀 Starting Phone Gyroscope Control with Ngrok")
    print("Using Kalman-enhanced server for better filtering")

    port = 8889
    local_url = f"http://localhost:{port}"

    # Start the enhanced gyro server
    server_process = start_enhanced_gyro_server(port)
    if not server_process:
        print("❌ Failed to start server. Exiting.")
        return

    # Start ngrok tunnel
    ngrok_process, public_url = start_ngrok_tunnel(port)

    # Print instructions
    print_instructions(public_url, local_url)

    # Start connection monitoring
    if public_url:
        monitor_thread = threading.Thread(
            target=monitor_connections,
            args=(public_url,),
            daemon=True
        )
        monitor_thread.start()

    try:
        # Keep everything running
        print("\n📡 System running... Move your phone to test!")
        server_process.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")

        # Clean up processes
        if server_process:
            server_process.terminate()
        if ngrok_process:
            ngrok_process.terminate()

        # Wait for cleanup
        time.sleep(2)
        print("✅ All services stopped")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Clean up on error
        if server_process:
            server_process.terminate()
        if ngrok_process:
            ngrok_process.terminate()


if __name__ == "__main__":
    main()