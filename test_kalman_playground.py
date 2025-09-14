#!/usr/bin/env python

"""
Test script for Kalman Filter Playground

This script demonstrates:
1. How to run the Kalman filter playground
2. The new motor mapping (pitch -> motor 3, roll -> motor 4)
3. Real-time filtering of phone gyroscope data
"""

import sys
import subprocess
import time
import threading
from pathlib import Path

def run_servers():
    """Run both the Kalman server and playground"""
    print("🎮 Kalman Filter Playground Test")
    print("=" * 50)

    # Start the enhanced gyro server
    print("1. Starting enhanced gyro server with Kalman filtering...")
    server_process = subprocess.Popen([
        sys.executable,
        "phone_gyro_server_with_kalman.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Give server time to start
    time.sleep(3)

    print("2. Server started! Open http://localhost:8889 on your phone")
    print("   - Tap 'Start Control' to begin")
    print("   - Use 'Toggle Kalman' to enable/disable filtering")
    print("   - Watch Motor 3 (pitch) and Motor 4 (roll) displays")

    print("\n3. Starting Kalman Filter Playground...")

    # Start the playground in a separate process
    playground_process = subprocess.Popen([
        sys.executable,
        "kalman_filter_playground.py"
    ])

    print("\n🎯 What to Test:")
    print("• Compare raw vs filtered orientation plots")
    print("• Watch Motor 3 respond to phone PITCH movements")
    print("• Watch Motor 4 respond to phone ROLL movements")
    print("• Toggle Kalman filter on/off to see difference")
    print("• Observe filter uncertainty in bottom-right plot")

    print("\n⌨️  Press Ctrl+C to stop both processes")

    try:
        # Wait for user to stop
        playground_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping processes...")

    # Clean up
    server_process.terminate()
    playground_process.terminate()

    # Wait a moment for cleanup
    time.sleep(1)

    print("✅ Test completed")

def main():
    """Main test function"""
    print("This test will:")
    print("1. Start enhanced gyro server with Kalman filter")
    print("2. Open the visual playground")
    print("3. Show real-time filtering of your phone's gyroscope data")
    print("\nMake sure you have matplotlib installed:")
    print("pip install matplotlib scipy")

    response = input("\nReady to start? (y/n): ")
    if response.lower() != 'y':
        return

    try:
        run_servers()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Install dependencies: pip install matplotlib scipy numpy")
        print("2. Make sure no other process is using port 8889")
        print("3. Check that all Python files are in the current directory")

if __name__ == "__main__":
    main()