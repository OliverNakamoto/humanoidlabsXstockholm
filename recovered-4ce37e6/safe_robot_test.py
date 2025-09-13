#!/usr/bin/env python3
"""
Safe robot connection test - handles connection issues gracefully
"""

import sys
import time
from pathlib import Path

# Add LeRobot to path
sys.path.append(str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

def safe_robot_test():
    """Test robot connection with error handling."""
    print("🤖 Safe Robot Connection Test")
    print("=" * 40)
    
    config = SO101FollowerConfig(
        port="COM6",
        id="follower",
        disable_torque_on_disconnect=True  # This might help with torque issues
    )
    
    robot = SO101Follower(config)
    
    try:
        print("🔌 Attempting to connect...")
        robot.connect(calibrate=False)  # Connect without immediate calibration
        print("✅ Connection successful!")
        
        # Get current position (this will tell us if motors are responding)
        print("\n📍 Reading current position...")
        try:
            current_pos = robot.get_pos()
            print("✅ Position read successful!")
            
            joint_names = list(current_pos.keys())
            print(f"Available joints: {joint_names}")
            
            for joint_name, position in current_pos.items():
                print(f"  {joint_name}: {position:.3f}")
                
        except Exception as pos_error:
            print(f"❌ Could not read position: {pos_error}")
            return False
        
        # Test if we can send a command (just return to current position)
        print("\n🧪 Testing command sending...")
        try:
            robot.send_action(current_pos)  # Send current position (should be safe)
            print("✅ Command sending successful!")
            
            # Try a small gripper movement
            if joint_names:
                gripper_joint = joint_names[-1]  # Assume last joint is gripper
                print(f"\n🦾 Testing small {gripper_joint} movement...")
                
                # Very small movement
                test_pos = current_pos.copy()
                current_gripper = current_pos[gripper_joint]
                
                # Move slightly towards open (but not fully)
                test_pos[gripper_joint] = min(1.0, current_gripper + 0.1)
                
                robot.send_action(test_pos)
                time.sleep(1)
                
                # Return to original position
                robot.send_action(current_pos)
                print("✅ Gripper test successful!")
                
        except Exception as cmd_error:
            print(f"❌ Command sending failed: {cmd_error}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
        # Try to diagnose the issue
        if "Torque_Enable" in str(e):
            print("\n🔧 Torque Enable Error Detected!")
            print("Possible solutions:")
            print("1. Check if robot is properly powered on")
            print("2. Verify all servo connections")
            print("3. Try restarting the robot")
            print("4. Check if another program is using the robot")
            print("5. Run calibration first to initialize motors")
        
        elif "status packet" in str(e):
            print("\n🔧 Communication Error Detected!")
            print("Possible solutions:")
            print("1. Check USB cable connection")
            print("2. Verify COM port (currently using COM6)")
            print("3. Try a different USB port")
            print("4. Check if other programs are using COM6")
        
        return False
        
    finally:
        try:
            print("\n🔌 Disconnecting...")
            robot.disconnect()
            print("👋 Disconnected safely")
        except:
            print("⚠️ Disconnect had issues (this is usually OK)")

def main():
    """Main function."""
    success = safe_robot_test()
    
    if success:
        print("\n🎉 Robot test successful!")
        print("\nYou can now try:")
        print("1. python simple_robot_api.py")
        print("2. python test_robot_control.py") 
        print("3. Complete robot calibration")
    else:
        print("\n❌ Robot test failed")
        print("\nNext steps:")
        print("1. Complete robot calibration first:")
        print("   python -m lerobot.calibrate --robot.type=so101_follower --robot.port=COM6 --robot.id=follower")
        print("2. Check hardware connections")
        print("3. Try restarting the robot")

if __name__ == "__main__":
    main()