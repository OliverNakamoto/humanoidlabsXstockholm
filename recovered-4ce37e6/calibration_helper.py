#!/usr/bin/env python3
"""
LeRobot SO101 Calibration Helper
Guided calibration process for SO101 robot arm
"""

import sys
import time
import argparse
from pathlib import Path

# Add LeRobot to path
sys.path.append(str(Path(__file__).parent / "lerobot" / "src"))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=== Checking Dependencies ===")
    
    missing = []
    
    try:
        import serial
        print("✓ pyserial")
    except ImportError:
        missing.append("pyserial")
        
    try:
        import cv2
        print("✓ opencv-python")
    except ImportError:
        missing.append("opencv-python")
        
    try:
        import mediapipe
        print("✓ mediapipe")
    except ImportError:
        missing.append("mediapipe")
        
    try:
        from lerobot.robots.so101_follower import SO101Follower
        print("✓ lerobot")
    except ImportError:
        missing.append("lerobot")
        print("✗ lerobot not found - run: cd lerobot && pip install -e .")
        
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def find_robot_port():
    """Find available serial ports for robot connection."""
    print("\n=== Finding Robot Port ===")
    
    import serial.tools.list_ports
    
    ports = list(serial.tools.list_ports.comports())
    available_ports = []
    
    for port in ports:
        try:
            ser = serial.Serial(port.device, 115200, timeout=1)
            ser.close()
            available_ports.append(port.device)
            print(f"✓ {port.device} - {port.description}")
        except:
            print(f"✗ {port.device} - {port.description} (in use or inaccessible)")
    
    if not available_ports:
        print("❌ No available serial ports found")
        print("Check robot connection and USB cables")
        return None
    
    return available_ports

def test_camera():
    """Test camera availability and basic functionality."""
    print("\n=== Testing Camera ===")
    
    import cv2
    
    for camera_id in range(4):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {camera_id}: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return camera_id
            else:
                print(f"✗ Camera {camera_id}: Can't read frames")
        cap.release()
    
    print("❌ No working cameras found")
    return None

def calibrate_robot(port: str, robot_id: str):
    """Guide user through robot calibration process."""
    print(f"\n=== Calibrating Robot on {port} ===")
    
    try:
        from lerobot.robots.so101_follower import SO101Follower
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        
        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=False  # Use normalized values
        )
        
        print("🤖 Connecting to robot...")
        robot = SO101Follower(config)
        
        # Check if robot is already calibrated
        if robot.calibration:
            print("ℹ️  Existing calibration found")
            user_input = input("Use existing calibration? (y/n): ")
            if user_input.lower() == 'y':
                robot.connect(calibrate=False)
                print("✅ Using existing calibration")
                return test_robot_movement(robot)
        
        print("\n🔧 Starting calibration process...")
        print("⚠️  ENSURE ROBOT HAS CLEAR MOVEMENT SPACE")
        input("Press Enter when robot is in safe position...")
        
        # This will trigger the calibration process
        robot.connect(calibrate=True)
        
        print("✅ Robot calibration complete!")
        return test_robot_movement(robot)
        
    except Exception as e:
        print(f"❌ Robot calibration failed: {e}")
        return False

def test_robot_movement(robot):
    """Test basic robot movement after calibration."""
    print("\n=== Testing Robot Movement ===")
    
    try:
        print("📍 Getting current position...")
        current_pos = robot.get_pos()
        print("Current position:")
        for joint, pos in current_pos.items():
            print(f"  {joint}: {pos:.3f}")
        
        print("\n🤏 Testing small movement...")
        test_action = current_pos.copy()
        
        # Small test movements
        test_action['shoulder_pan.pos'] += 0.05  # Small pan
        robot.send_action(test_action)
        time.sleep(1)
        
        # Return to original position
        robot.send_action(current_pos)
        time.sleep(1)
        
        print("✅ Robot movement test successful")
        robot.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Robot movement test failed: {e}")
        try:
            robot.disconnect()
        except:
            pass
        return False

def test_hand_tracking():
    """Test hand tracking functionality."""
    print("\n=== Testing Hand Tracking ===")
    
    # Test your CV hand tracker
    try:
        sys.path.append(str(Path(__file__).parent / "cv"))
        from cv_hand_tracker import HandTracker
        
        tracker = HandTracker()
        print("✓ HandTracker initialized")
        
        # Test with camera
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not available for hand tracking test")
            return False
        
        print("🖐️  Testing hand detection (10 seconds)...")
        print("Move your hand in front of the camera")
        
        detection_count = 0
        total_frames = 0
        start_time = time.time()
        
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                continue
                
            result = tracker.process_frame(frame)
            total_frames += 1
            
            if result.get('detected', False):
                detection_count += 1
                
            # Show progress every 30 frames
            if total_frames % 30 == 0:
                detection_rate = detection_count / total_frames * 100
                print(f"Detection rate: {detection_rate:.1f}%")
        
        cap.release()
        tracker.close()
        
        detection_rate = detection_count / total_frames * 100
        print(f"✅ Hand tracking test complete: {detection_rate:.1f}% detection rate")
        
        if detection_rate < 30:
            print("⚠️  Low detection rate - check lighting and background")
        
        return detection_rate > 30
        
    except Exception as e:
        print(f"❌ Hand tracking test failed: {e}")
        return False

def test_integration():
    """Test HandCV teleoperator integration."""
    print("\n=== Testing HandCV Integration ===")
    
    try:
        from lerobot.teleoperators.hand_cv import HandCVTeleop
        from lerobot.teleoperators.hand_cv.config_hand_cv import HandCVTeleopConfig
        
        config = HandCVTeleopConfig(
            id="test_integration",
            camera_id=0,
            enable_visualization=False  # Don't show window in test
        )
        
        teleop = HandCVTeleop(config)
        print("✓ HandCVTeleop created")
        
        teleop.connect()
        print("✓ HandCVTeleop connected")
        
        # Test action generation
        dummy_current_pos = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0
        }
        
        action = teleop.get_action(dummy_current_pos)
        print("✓ Action generated:", list(action.keys()))
        
        status = teleop.get_status()
        print(f"✓ Status: Connected={status['connected']}, FPS={status['fps']:.1f}")
        
        teleop.disconnect()
        print("✅ HandCV integration test successful")
        return True
        
    except Exception as e:
        print(f"❌ HandCV integration test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='LeRobot SO101 Calibration Helper')
    parser.add_argument('--robot-id', default='my_so101', help='Robot ID for calibration')
    parser.add_argument('--port', help='Serial port (will auto-detect if not specified)')
    parser.add_argument('--skip-robot', action='store_true', help='Skip robot calibration (CV only)')
    args = parser.parse_args()
    
    print("🤖 LeRobot SO101 + HandCV Calibration Helper")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return 1
    
    # Step 2: Find hardware
    if not args.skip_robot:
        available_ports = find_robot_port()
        if not available_ports and not args.port:
            print("❌ No robot ports found and none specified")
            return 1
        
        robot_port = args.port or available_ports[0]
        print(f"Using robot port: {robot_port}")
    
    camera_id = test_camera()
    if camera_id is None:
        print("❌ No working camera found")
        return 1
    
    # Step 3: Calibrate robot
    if not args.skip_robot:
        print(f"\n🔧 Robot calibration starting...")
        if not calibrate_robot(robot_port, args.robot_id):
            print("❌ Robot calibration failed")
            return 1
    else:
        print("⏭️  Skipping robot calibration")
    
    # Step 4: Test hand tracking
    if not test_hand_tracking():
        print("❌ Hand tracking test failed")
        return 1
    
    # Step 5: Test integration
    if not test_integration():
        print("❌ Integration test failed")
        return 1
    
    print("\n🎉 All calibration tests passed!")
    print("\nNext steps:")
    if not args.skip_robot:
        print(f"1. Test full system: python -m lerobot.teleoperate --robot.type=so101_follower --robot.port={robot_port} --teleop.type=hand_cv")
    print("2. Run workspace calibration: python cv/calibrate_workspace.py")
    print("3. Fine-tune configuration in config_hand_cv.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())