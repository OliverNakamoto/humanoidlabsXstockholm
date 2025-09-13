#!/usr/bin/env python3
"""
Test script for HandCVTeleop integration with LeRobot.
Demonstrates how to use computer vision hand tracking as a teleoperation device.
"""

import sys
import time
import argparse
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.teleoperators.hand_cv.hand_cv_teleop import HandCVTeleop
from lerobot.teleoperators.hand_cv.config_hand_cv import HandCVTeleopConfig


def test_basic_functionality():
    """Test basic HandCVTeleop functionality without robot hardware."""
    print("=== Testing HandCVTeleop Basic Functionality ===")
    
    # Create configuration
    config = HandCVTeleopConfig(
        id="test_hand_cv",
        camera_id=0,
        camera_width=640,
        camera_height=480,
        confidence_threshold=0.7,
        smoothing_window=5,
        enable_visualization=True,
        update_rate=30.0,
        use_degrees=False
    )
    
    print(f"Configuration created:")
    print(f"  Camera ID: {config.camera_id}")
    print(f"  Resolution: {config.camera_width}x{config.camera_height}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print(f"  Use degrees: {config.use_degrees}")
    
    # Create teleoperator
    try:
        teleop = HandCVTeleop(config)
        print("✓ HandCVTeleop created successfully")
    except Exception as e:
        print(f"✗ Failed to create HandCVTeleop: {e}")
        return False
    
    # Test connection
    try:
        teleop.connect()
        print("✓ Camera connected successfully")
        print(f"  Connected: {teleop.is_connected}")
        print(f"  Calibrated: {teleop.is_calibrated}")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    # Test action features
    features = teleop.action_features
    print(f"✓ Action features: {list(features.keys())}")
    
    # Test getting actions
    print("\n=== Testing Action Retrieval ===")
    print("Move your hand in front of the camera...")
    print("Press Ctrl+C to stop, or wait 30 seconds")
    
    try:
        start_time = time.time()
        action_count = 0
        
        while time.time() - start_time < 30.0:  # Test for 30 seconds
            action = teleop.get_action()
            status = teleop.get_status()
            
            action_count += 1
            
            # Print status every 30 actions (~1 second at 30Hz)
            if action_count % 30 == 0:
                print(f"Time: {time.time() - start_time:.1f}s | "
                      f"FPS: {status['fps']:.1f} | "
                      f"Detection: {status['detection_active']} | "
                      f"Connected: {status['connected']}")
                
                # Print action values
                print(f"  Actions: ", end="")
                for key, val in action.items():
                    print(f"{key}: {val:.2f} ", end="")
                print()
            
            time.sleep(1/30)  # ~30 Hz
            
    except KeyboardInterrupt:
        print("\n✓ Test interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during action retrieval: {e}")
        return False
    
    # Cleanup
    try:
        teleop.disconnect()
        print("✓ Teleoperator disconnected successfully")
    except Exception as e:
        print(f"✗ Error during disconnect: {e}")
        return False
    
    print("\n=== Basic Functionality Test Complete ===")
    return True


def test_lerobot_integration():
    """Test integration with LeRobot teleoperation system."""
    print("=== Testing LeRobot Integration ===")
    
    # Test with LeRobot's teleoperation command
    print("To test with actual SO101 follower robot, use:")
    print()
    print("python -m lerobot.teleoperate \\")
    print("    --robot.type=so101_follower \\")
    print("    --robot.port=/dev/ttyUSB0 \\")
    print("    --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \\")
    print("    --robot.use_degrees=false \\")
    print("    --teleop.type=hand_cv \\")
    print("    --teleop.camera_id=0 \\")
    print("    --teleop.use_degrees=false \\")
    print("    --teleop.enable_visualization=true \\")
    print("    --display_data=true")
    print()
    
    # Test configuration registration
    try:
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.teleoperators.hand_cv.config_hand_cv import HandCVTeleopConfig
        
        config = HandCVTeleopConfig(
            id="integration_test",
            camera_id=0
        )
        
        teleop = make_teleoperator_from_config(config)
        print("✓ Teleoperator created via LeRobot factory")
        print(f"  Type: {type(teleop).__name__}")
        print(f"  Name: {teleop.name}")
        
    except Exception as e:
        print(f"✗ LeRobot integration test failed: {e}")
        return False
    
    print("✓ LeRobot integration test passed")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test HandCVTeleop implementation')
    parser.add_argument('--test', choices=['basic', 'integration', 'all'], 
                       default='all', help='Which test to run')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use')
    args = parser.parse_args()
    
    print("HandCVTeleop Test Script")
    print("======================")
    print()
    
    if args.test in ['basic', 'all']:
        success = test_basic_functionality()
        if not success:
            print("Basic functionality test failed!")
            return 1
    
    if args.test in ['integration', 'all']:
        success = test_lerobot_integration()  
        if not success:
            print("LeRobot integration test failed!")
            return 1
    
    print()
    print("🎉 All tests passed!")
    print()
    print("Next steps:")
    print("1. Calibrate your workspace: python cv/calibrate_workspace.py")
    print("2. Get a robot URDF file for accurate inverse kinematics")
    print("3. Test with your SO101 robot using the command shown above")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())