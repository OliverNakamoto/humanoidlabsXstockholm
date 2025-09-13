#!/usr/bin/env python3
"""
Simple API for controlling SO101 robot
Easy-to-use functions for common robot operations
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add LeRobot to path
sys.path.append(str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

class SimpleRobotAPI:
    """Simple API wrapper for SO101 robot control."""
    
    def __init__(self, port="COM6", robot_id="follower"):
        """Initialize robot connection."""
        self.config = SO101FollowerConfig(port=port, id=robot_id)
        self.robot = SO101Follower(self.config)
        self.connected = False
        self.home_position = None
        
    def connect(self):
        """Connect to the robot."""
        if not self.connected:
            print(f"🔌 Connecting to robot on {self.config.port}...")
            self.robot.connect()
            self.connected = True
            
            # Save current position as home
            self.home_position = self.robot.get_pos()
            print("✅ Robot connected!")
            print(f"📍 Home position saved: {list(self.home_position.keys())}")
        return self
    
    def disconnect(self):
        """Disconnect from the robot."""
        if self.connected:
            print("🔌 Disconnecting...")
            self.robot.disconnect()
            self.connected = False
            print("👋 Disconnected")
    
    def __enter__(self):
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def get_joint_names(self):
        """Get list of joint names."""
        if not self.connected:
            self.connect()
        return list(self.robot.get_pos().keys())
    
    def get_current_position(self) -> Dict[str, float]:
        """Get current joint positions."""
        if not self.connected:
            self.connect()
        return self.robot.get_pos()
    
    def move_to_position(self, positions: Dict[str, float], duration: float = 1.0):
        """
        Move robot to specified joint positions.
        
        Args:
            positions: Dict of {joint_name: position_value}
            duration: Time to wait after sending command
        """
        if not self.connected:
            self.connect()
        
        print(f"🎯 Moving to: {positions}")
        self.robot.send_action(positions)
        if duration > 0:
            time.sleep(duration)
    
    def open_gripper(self, duration: float = 1.0):
        """Open the gripper."""
        joints = self.get_joint_names()
        gripper_joint = joints[-1]  # Assume last joint is gripper
        
        current = self.get_current_position()
        current[gripper_joint] = 1.0  # Fully open
        
        print("✋ Opening gripper...")
        self.move_to_position(current, duration)
    
    def close_gripper(self, duration: float = 1.0):
        """Close the gripper."""
        joints = self.get_joint_names()
        gripper_joint = joints[-1]  # Assume last joint is gripper
        
        current = self.get_current_position()
        current[gripper_joint] = 0.0  # Fully closed
        
        print("✊ Closing gripper...")
        self.move_to_position(current, duration)
    
    def go_home(self, duration: float = 2.0):
        """Return to home position."""
        if self.home_position is None:
            print("❌ No home position saved")
            return
        
        print("🏠 Returning home...")
        self.move_to_position(self.home_position, duration)
    
    def move_joint(self, joint_name: str, position: float, duration: float = 1.0):
        """
        Move a specific joint.
        
        Args:
            joint_name: Name of the joint
            position: Target position (0.0 to 1.0 for most joints)
            duration: Time to wait after movement
        """
        current = self.get_current_position()
        current[joint_name] = position
        self.move_to_position(current, duration)
    
    def wave_hello(self):
        """Make the robot wave hello."""
        print("👋 Waving hello...")
        
        # Get joints (this is robot-specific, may need adjustment)
        joints = self.get_joint_names()
        current = self.get_current_position()
        
        # Simple wave motion (adjust joint names as needed)
        if len(joints) >= 2:
            wave_joint = joints[1]  # Second joint for waving
            
            # Wave motion
            for _ in range(3):
                # Wave up
                current[wave_joint] = min(1.0, current[wave_joint] + 0.3)
                self.move_to_position(current, 0.5)
                
                # Wave down
                current[wave_joint] = max(0.0, current[wave_joint] - 0.3)
                self.move_to_position(current, 0.5)
        
        self.go_home()

def demo_api():
    """Demonstrate the simple robot API."""
    print("🤖 Simple Robot API Demo")
    print("=" * 30)
    
    # Use context manager for automatic connection/disconnection
    with SimpleRobotAPI(port="COM6") as robot:
        
        # Show available joints
        joints = robot.get_joint_names()
        print(f"Available joints: {joints}")
        
        # Show current position
        pos = robot.get_current_position()
        print(f"Current position: {pos}")
        
        # Test gripper
        print("\n🦾 Testing gripper...")
        robot.open_gripper(1)
        robot.close_gripper(1)
        
        # Test individual joint movement
        if len(joints) > 0:
            print(f"\n🎯 Testing {joints[0]} movement...")
            robot.move_joint(joints[0], 0.7, 1)
            robot.move_joint(joints[0], 0.3, 1)
        
        # Return home
        robot.go_home()
        
        # Wave hello
        robot.wave_hello()
        
        print("✅ Demo complete!")

if __name__ == "__main__":
    demo_api()