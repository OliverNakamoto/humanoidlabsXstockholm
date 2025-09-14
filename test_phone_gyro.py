#!/usr/bin/env python

"""
Test script for phone gyroscope control system.
Tests the phone gyro teleoperator without requiring a physical robot.
"""

import sys
import time
import logging
from pathlib import Path

# Add lerobot to path
sys.path.append("lerobot/src")

from lerobot.teleoperators.phone_gyro import PhoneGyro, PhoneGyroConfig

def test_phone_gyro():
    """Test the phone gyroscope teleoperator"""
    logging.basicConfig(level=logging.INFO)

    print("🤖 Testing Phone Gyroscope Teleoperator")
    print("=" * 50)

    # Create config
    config = PhoneGyroConfig(
        server_url="http://localhost:8889",
        use_degrees=False,  # Use normalized range
        position_scale=1.0,
        orientation_scale=1.0
    )

    print(f"Config: {config}")

    # Create teleoperator
    teleop = PhoneGyro(config)

    try:
        # Connect (without calibration)
        print("\n📡 Connecting to phone gyro server...")
        teleop.connect(calibrate=False)
        print("✅ Connected!")

        print("\n📱 Open http://localhost:8889 on your phone")
        print("   Tap 'Start Control' and move your phone around")

        print("\n🎮 Reading phone gyro data...")
        print("   (Press Ctrl+C to stop)")

        # Test loop
        for i in range(100):  # Run for ~10 seconds
            try:
                # Get action from phone
                action = teleop.get_action()

                # Print action
                print(f"\rStep {i+1:3d} | ", end="")
                for motor, value in action.items():
                    print(f"{motor.split('.')[0][:8]:<8}: {value:+6.1f} | ", end="")

                time.sleep(0.1)  # 10Hz update rate

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error getting action: {e}")
                time.sleep(1)

        print(f"\n\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure phone gyro server is running: python phone_gyro_server.py")
        print("2. Open http://localhost:8889 on your phone")
        print("3. Tap 'Start Control' to begin sending data")

    finally:
        if teleop.is_connected:
            teleop.disconnect()
            print("📡 Disconnected")


if __name__ == "__main__":
    test_phone_gyro()