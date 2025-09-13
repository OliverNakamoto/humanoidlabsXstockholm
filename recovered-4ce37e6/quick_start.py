#!/usr/bin/env python3
"""
Quick Start Script for LeRobot SO101 + HandCV
Run this first to get everything working
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show results."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout:
                print(result.stdout[:500])  # First 500 chars
        else:
            print("❌ Failed")
            print(result.stderr[:500])
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 LeRobot SO101 + HandCV Quick Start")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("lerobot").exists():
        print("❌ lerobot directory not found")
        print("Run this script from the main project directory")
        return 1
    
    print("📁 Project structure looks good")
    
    # Step 1: Install dependencies
    print("\n📦 Installing dependencies...")
    dependencies = [
        "pip install opencv-python",
        "pip install mediapipe", 
        "pip install pyserial",
        "pip install numpy",
        "pip install matplotlib"
    ]
    
    for dep in dependencies:
        run_command(dep, f"Installing {dep.split()[-1]}")
    
    # Step 2: Install LeRobot
    run_command("cd lerobot && pip install -e .", "Installing LeRobot")
    
    # Step 3: Test basic functionality
    print("\n🧪 Running basic tests...")
    
    # Test camera
    run_command("python -c \"import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()\"", 
                "Testing camera")
    
    # Test hand tracking
    if Path("cv/cv_hand_tracker.py").exists():
        run_command("cd cv && python -c \"from cv_hand_tracker import HandTracker; print('HandTracker OK')\"",
                   "Testing hand tracker")
    
    # Test LeRobot imports
    run_command("python -c \"from lerobot.teleoperators.hand_cv import HandCVTeleop; print('HandCVTeleop OK')\"",
               "Testing LeRobot integration")
    
    # Step 4: Run calibration helper
    print("\n🔧 Ready for calibration!")
    print("\nNext steps:")
    print("1. Connect your SO101 robot via USB")
    print("2. Run: python calibration_helper.py")
    print("3. Follow the calibration prompts")
    print("4. Test with: python test_hand_cv_teleop.py")
    
    print("\n📖 For detailed setup: see lerobot_setup_guide.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())