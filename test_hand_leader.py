#!/usr/bin/env python

"""
Test script for hand_leader teleoperation.
This tests the CV hand tracking without needing the actual robot connected.
"""

import sys
import time
from pathlib import Path

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.teleoperators.hand_leader import HandLeader, HandLeaderConfig


def test_hand_leader():
    """Test the hand leader CV tracking."""
    
    print("="*60)
    print("Testing Hand Leader CV Tracking")
    print("="*60)
    
    # Create config
    config = HandLeaderConfig(
        type="hand_leader",
        id="test_hand_leader",
        camera_index=0,
        urdf_path="lerobot/src/lerobot/robots/so101_follower/so101.urdf",  
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    )
    
    # Create hand leader
    print("\nCreating hand leader...")
    leader = HandLeader(config)
    
    # Connect and calibrate
    print("\nConnecting and calibrating...")
    try:
        leader.connect(calibrate=True)
    except Exception as e:
        print(f"Error during connection: {e}")
        return
    
    print("\n" + "="*60)
    print("Starting teleoperation test")
    print("Move your hand to control the virtual robot")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Test loop
    try:
        while True:
            # Get action from hand tracking
            action = leader.get_action()
            
            # Display the action
            print("\r" + " "*80, end="")  # Clear line
            print(f"\rActions: ", end="")
            for key, value in action.items():
                print(f"{key}: {value:6.1f} ", end="")
            
            time.sleep(0.05)  # 20 Hz update rate
            
    except KeyboardInterrupt:
        print("\n\nStopping test...")
    finally:
        leader.disconnect()
        print("Test complete!")


if __name__ == "__main__":
    test_hand_leader()