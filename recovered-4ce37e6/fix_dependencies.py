#!/usr/bin/env python3
"""
Fix dependency conflicts for LeRobot + HandCV setup
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show results."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success")
        else:
            print("❌ Failed")
            if result.stderr:
                print(result.stderr[:500])
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔧 Fixing LeRobot + HandCV Dependencies")
    print("=" * 40)
    
    # Step 1: Downgrade NumPy to compatible version
    print("\n📦 Installing compatible NumPy version...")
    run_command(
        "pip install \"numpy>=1.26.4,<2.0\"", 
        "Installing NumPy 1.x (compatible with MediaPipe)"
    )
    
    # Step 2: Install MediaPipe with compatible numpy
    run_command(
        "pip install mediapipe", 
        "Reinstalling MediaPipe"
    )
    
    # Step 3: Install other CV dependencies
    run_command(
        "pip install opencv-python", 
        "Installing OpenCV"
    )
    
    # Step 4: Reinstall LeRobot to ensure compatibility
    run_command(
        'pip install -e ".[feetech]" --no-deps',
        "Reinstalling LeRobot without dependency resolution"
    )
    
    # Step 5: Test critical imports
    print("\n🧪 Testing critical imports...")
    
    test_imports = [
        ("numpy", "import numpy; print(f'NumPy version: {numpy.__version__}')"),
        ("mediapipe", "import mediapipe; print('MediaPipe: OK')"),
        ("cv2", "import cv2; print('OpenCV: OK')"),
        ("lerobot", "from lerobot.robots.so101_follower import SO101Follower; print('LeRobot: OK')"),
        ("hand_cv", "from lerobot.teleoperators.hand_cv import HandCVTeleop; print('HandCV: OK')")
    ]
    
    for name, test_code in test_imports:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: {result.stderr.strip()[:100]}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print("\n📋 Next steps:")
    print("1. If all imports work, you're ready to go!")
    print("2. Run: python calibration_helper.py")
    print("3. If still issues, try creating a fresh virtual environment")
    
    return 0

if __name__ == "__main__":
    main()