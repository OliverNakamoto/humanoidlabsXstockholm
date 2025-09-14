#!/usr/bin/env python

"""
Standalone test for hand tracking with visual feedback.
Simple script to test hand tracking without the full teleoperation pipeline.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "cv"))

from lerobot.teleoperators.hand_leader.cv_hand_tracker import CVHandTracker


def draw_robot_visualization(frame, joint_positions, hand_pos):
    """Draw a simple robot arm visualization on the frame."""
    h, w = frame.shape[:2]
    
    # Create visualization area on the right side
    viz_x = w - 300
    viz_y = 50
    viz_w = 250
    viz_h = 300
    
    # Draw background
    cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_w, viz_y + viz_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_w, viz_y + viz_h), (255, 255, 255), 2)
    cv2.putText(frame, "Robot Simulation", (viz_x + 10, viz_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw joint positions as bars
    joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist F", "Wrist R", "Gripper"]
    bar_height = 20
    bar_spacing = 40
    
    for i, (name, key) in enumerate(zip(joint_names, joint_positions.keys())):
        y_pos = viz_y + 60 + i * bar_spacing
        value = joint_positions[key]
        
        # Normalize value to bar width (-100 to 100 -> 0 to bar_width)
        if "gripper" in key:
            bar_width = int((value / 100) * 200)
            color = (0, 200, 0)  # Green for gripper
        else:
            bar_width = int((value + 100) / 200 * 200)
            color = (100, 100, 255)  # Light blue for joints
        
        # Draw bar background
        cv2.rectangle(frame, (viz_x + 20, y_pos), 
                     (viz_x + 220, y_pos + bar_height), 
                     (100, 100, 100), -1)
        
        # Draw bar
        if bar_width > 0:
            cv2.rectangle(frame, (viz_x + 20, y_pos), 
                         (viz_x + 20 + bar_width, y_pos + bar_height), 
                         color, -1)
        
        # Draw label
        label = f"{name[:8]:8s}: {value:6.1f}"
        cv2.putText(frame, label, (viz_x + 20, y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw hand position info
    x, y, z, pinch = hand_pos
    info_y = viz_y + viz_h - 60
    cv2.putText(frame, f"Hand X: {x:.2f}", (viz_x + 20, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Hand Y: {y:.2f}", (viz_x + 20, info_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Hand Z: {z:.2f}", (viz_x + 20, info_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw pinch indicator
    pinch_radius = int(10 + pinch / 10)
    pinch_color = (0, int(255 - pinch * 2.5), int(pinch * 2.5))
    cv2.circle(frame, (viz_x + 180, info_y + 20), pinch_radius, pinch_color, -1)
    cv2.putText(frame, f"Pinch: {pinch:.0f}%", (viz_x + 120, info_y + 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def simple_ik(hand_x, hand_y, hand_z):
    """Very simple pseudo-IK for demonstration."""
    # Map hand position to joint angles (simplified)
    shoulder_pan = hand_x * 300  # X controls pan
    shoulder_lift = -hand_y * 200 + 50  # Y controls lift
    elbow_flex = hand_z * 150 - 30  # Z affects elbow
    wrist_flex = hand_y * 100  # Y also affects wrist
    wrist_roll = hand_x * 180  # X controls wrist roll
    
    return {
        "shoulder_pan.pos": np.clip(shoulder_pan, -100, 100),
        "shoulder_lift.pos": np.clip(shoulder_lift, -100, 100),
        "elbow_flex.pos": np.clip(elbow_flex, -100, 100),
        "wrist_flex.pos": np.clip(wrist_flex, -100, 100),
        "wrist_roll.pos": np.clip(wrist_roll, -100, 100),
    }


def main():
    print("="*60)
    print("Standalone Hand Tracking Test")
    print("="*60)
    print("This test shows hand tracking with simulated robot control")
    print("No hardware or complex setup required!")
    print("="*60 + "\n")
    
    # Initialize tracker
    tracker = CVHandTracker(camera_index=0)
    tracker.connect()
    
    print("Starting calibration...")
    tracker.calibrate()
    
    print("\n" + "="*60)
    print("TRACKING STARTED")
    print("="*60)
    print("Move your hand to control the simulated robot")
    print("The visualization shows joint positions on the right")
    print("Press 'q' to quit")
    print("="*60 + "\n")
    
    # Initialize camera for display
    cap = cv2.VideoCapture(0)
    
    # Smooth joint positions
    smooth_joints = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }
    
    smoothing_factor = 0.2  # How quickly to follow target
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Get hand position
            x, y, z, pinch = tracker.get_current_position()
            hand_pos = (x, y, z, pinch)
            
            # Compute target joint positions
            target_joints = simple_ik(x, y, z)
            target_joints["gripper.pos"] = pinch
            
            # Smooth the joint movements
            for key in smooth_joints:
                if key in target_joints:
                    smooth_joints[key] += (target_joints[key] - smooth_joints[key]) * smoothing_factor
            
            # Draw visualization
            draw_robot_visualization(frame, smooth_joints, hand_pos)
            
            # Add status text
            status = "TRACKING" if tracker.current_data['valid'] else "NO HAND DETECTED"
            color = (0, 255, 0) if tracker.current_data['valid'] else (0, 0, 255)
            cv2.putText(frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Hand Tracking Robot Control", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tracker.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Test complete!")


if __name__ == "__main__":
    main()