#!/usr/bin/env python3

"""
Simple test to verify hand tracking works without lerobot dependencies.
This bypasses the dependency conflicts by using a minimal implementation.
"""

import sys
import os
import time

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    print("✓ All CV dependencies available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Please install: pip install mediapipe opencv-python")
    sys.exit(1)


class SimpleHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Calculate palm center (simplified)
            palm_x = hand_landmarks.landmark[9].x * w  # Middle finger base
            palm_y = hand_landmarks.landmark[9].y * h
            
            # Calculate pinch (thumb tip to index tip distance)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            pinch_dist = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            ) * w
            
            # Normalize pinch (smaller distance = more closed)
            pinch_percent = max(0, min(100, 100 * (1 - pinch_dist / 100)))
            
            # Draw info
            cv2.circle(frame, (int(palm_x), int(palm_y)), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Palm: ({palm_x:.0f}, {palm_y:.0f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pinch: {pinch_percent:.0f}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw robot simulation bars
            self.draw_robot_sim(frame, palm_x, palm_y, pinch_percent, w, h)
            
            return True
        
        return False

    def draw_robot_sim(self, frame, palm_x, palm_y, pinch, w, h):
        """Draw simulated robot joint positions."""
        # Simulate joint angles based on hand position
        joints = {
            "Shoulder Pan": (palm_x - w/2) / w * 200,  # -100 to 100
            "Shoulder Lift": -(palm_y - h/2) / h * 200,  # -100 to 100  
            "Elbow": palm_x / w * 200 - 100,  # -100 to 100
            "Wrist": palm_y / h * 200 - 100,  # -100 to 100
            "Gripper": pinch  # 0 to 100
        }
        
        # Draw bars
        bar_y = h - 200
        bar_width = 150
        bar_height = 20
        
        for i, (name, value) in enumerate(joints.items()):
            y = bar_y + i * 30
            
            # Background bar
            cv2.rectangle(frame, (10, y), (10 + bar_width, y + bar_height), (50, 50, 50), -1)
            
            # Value bar
            if name == "Gripper":
                fill_width = int(bar_width * value / 100)
                color = (0, 255, 0)
            else:
                fill_width = int(bar_width * (value + 100) / 200)
                color = (100, 100, 255)
            
            if fill_width > 0:
                cv2.rectangle(frame, (10, y), (10 + fill_width, y + bar_height), color, -1)
            
            # Label
            cv2.putText(frame, f"{name}: {value:.0f}", (180, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    print("="*60)
    print("Simple Hand Tracking Test")
    print("="*60)
    print("This tests hand tracking without lerobot dependencies")
    print("Move your hand to see simulated robot control")
    print("Press 'q' to quit")
    print("="*60)
    
    tracker = SimpleHandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully")
    print("Starting hand tracking...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)  # Mirror image
            
            # Process hand tracking
            hand_detected = tracker.process_frame(frame)
            
            # Add status
            status = "HAND DETECTED" if hand_detected else "NO HAND"
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, status, (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}")
            
            cv2.imshow('Simple Hand Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.hands.close()
        print("Test complete!")


if __name__ == "__main__":
    main()