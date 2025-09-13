#!/usr/bin/env python3
"""
Quick test to verify all components are working
"""

import sys
from pathlib import Path

# Add LeRobot to path
sys.path.append(str(Path(__file__).parent / "lerobot" / "src"))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    # Test basic imports
    try:
        import mediapipe
        print(f"OK MediaPipe {mediapipe.__version__}")
    except ImportError as e:
        print(f"FAIL MediaPipe: {e}")
        return False
    
    try:
        import cv2
        print(f"OK OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"FAIL OpenCV: {e}")
        return False
    
    try:
        from lerobot.robots.so101_follower import SO101Follower
        print("OK SO101Follower")
    except ImportError as e:
        print(f"FAIL SO101Follower: {e}")
        return False
    
    try:
        from lerobot.teleoperators.hand_cv import HandCVTeleop
        print("OK HandCVTeleop")
    except ImportError as e:
        print(f"FAIL HandCVTeleop: {e}")
        return False
    
    return True

def test_robot_connection():
    """Test robot connection without calibration."""
    print("\nTesting robot connection...")
    
    try:
        import serial
        ser = serial.Serial('COM6', 115200, timeout=1)
        print(f"OK Serial connection to COM6 successful")
        ser.close()
        return True
    except Exception as e:
        print(f"FAIL Serial connection failed: {e}")
        return False

def test_camera():
    """Test camera access."""
    print("\nTesting camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"OK Camera working: {frame.shape}")
                cap.release()
                return True
            else:
                print("FAIL Camera can't capture frames")
        else:
            print("FAIL Camera not accessible")
        cap.release()
        return False
    except Exception as e:
        print(f"FAIL Camera test failed: {e}")
        return False

def main():
    print("Quick System Test")
    print("=" * 30)
    
    # Test all components
    imports_ok = test_imports()
    robot_ok = test_robot_connection()
    camera_ok = test_camera()
    
    print(f"\nResults:")
    print(f"  Imports: {'OK' if imports_ok else 'FAIL'}")
    print(f"  Robot:   {'OK' if robot_ok else 'FAIL'}")
    print(f"  Camera:  {'OK' if camera_ok else 'FAIL'}")
    
    if imports_ok and robot_ok and camera_ok:
        print("\nAll tests passed!")
        print("\nYou're ready for:")
        print("1. Robot calibration")
        print("2. Hand tracking tests")
        print("3. Full system integration")
        
        print("\nNext commands:")
        print("# Interactive calibration (follow prompts)")
        print("python calibration_helper.py")
        print("")
        print("# Or manual calibration")
        print("python -m lerobot.calibrate --robot.type=so101_follower --robot.port=COM6 --robot.id=follower")
        
        return 0
    else:
        print("\nSome tests failed - fix issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())