#!/usr/bin/env python3
"""
Test script to control SO101 robot programmatically
Shows how to move joints, open/close gripper, etc.
"""

import sys
import time
from pathlib import Path

# Add LeRobot to path
sys.path.append(str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

def test_robot_control():
    """Test basic robot control functions."""
    print("🤖 Testing SO101 Robot Control")
    print("=" * 40)
    
    # Create robot configuration
    config = SO101FollowerConfig(
        port="COM6",
        id="follower"
    )
    
    # Create and connect robot
    robot = SO101Follower(config)
    
    try:
        print("🔌 Connecting to robot...")
        robot.connect()
        print("✅ Connected!")
        
        # Get current position
        print("\n📍 Current robot position:")
        current_pos = robot.get_pos()
        for joint_name, position in current_pos.items():
            print(f"  {joint_name}: {position:.3f}")
        
        # Test gripper control
        print("\n🦾 Testing gripper control...")
        
        # Open gripper (assuming gripper is last joint)
        gripper_joint = list(current_pos.keys())[-1]  # Usually the last joint
        print(f"Opening gripper ({gripper_joint})...")
        
        # Create action to open gripper
        open_action = current_pos.copy()
        open_action[gripper_joint] = 1.0  # Fully open (normalized 0-1)
        
        robot.send_action(open_action)
        time.sleep(2)
        
        # Close gripper
        print(f"Closing gripper ({gripper_joint})...")
        close_action = current_pos.copy()
        close_action[gripper_joint] = 0.0  # Fully closed
        
        robot.send_action(close_action)
        time.sleep(2)
        
        # Return to original position
        print("↩️ Returning to original position...")
        robot.send_action(current_pos)
        time.sleep(1)
        
        print("✅ Gripper test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🔌 Disconnecting...")
        robot.disconnect()

def test_joint_movement():
    """Test moving individual joints."""
    print("\n🦾 Testing individual joint movement...")
    
    config = SO101FollowerConfig(port="COM6", id="follower")
    robot = SO101Follower(config)
    
    try:
        robot.connect()
        
        # Get current position
        current_pos = robot.get_pos()
        joint_names = list(current_pos.keys())
        
        print(f"Available joints: {joint_names}")
        
        # Test moving each joint slightly
        for joint_name in joint_names:
            print(f"\n🎯 Testing {joint_name}...")
            
            # Small movement
            test_action = current_pos.copy()
            test_action[joint_name] = min(1.0, current_pos[joint_name] + 0.1)
            
            robot.send_action(test_action)
            time.sleep(1)
            
            # Return to center
            robot.send_action(current_pos)
            time.sleep(1)
        
        print("✅ Joint movement test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        robot.disconnect()

def interactive_control():
    """Interactive robot control interface."""
    print("\n🎮 Interactive Robot Control")
    print("Commands:")
    print("  'o' - Open gripper")
    print("  'c' - Close gripper") 
    print("  'r' - Reset to current position")
    print("  'q' - Quit")
    
    config = SO101FollowerConfig(port="COM6", id="follower")
    robot = SO101Follower(config)
    
    try:
        robot.connect()
        initial_pos = robot.get_pos()
        joint_names = list(initial_pos.keys())
        gripper_joint = joint_names[-1]  # Assume last joint is gripper
        
        print(f"\nRobot connected. Gripper joint: {gripper_joint}")
        print("Current position saved as reset point.")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'q':
                break
            elif command == 'o':
                print("Opening gripper...")
                action = initial_pos.copy()
                action[gripper_joint] = 1.0
                robot.send_action(action)
            elif command == 'c':
                print("Closing gripper...")
                action = initial_pos.copy()
                action[gripper_joint] = 0.0
                robot.send_action(action)
            elif command == 'r':
                print("Resetting to initial position...")
                robot.send_action(initial_pos)
            else:
                print("Unknown command. Use 'o', 'c', 'r', or 'q'")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        robot.disconnect()

def main():
    """Main function to run tests."""
    print("🚀 SO101 Robot Control Test")
    print("=" * 30)
    
    try:
        # Test basic control
        test_robot_control()
        
        # Ask user what to do next
        print("\n🤔 What would you like to test next?")
        print("1. Individual joint movement")
        print("2. Interactive control")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            test_joint_movement()
        elif choice == "2":
            interactive_control()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()