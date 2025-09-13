#!/usr/bin/env python3
"""
Lightweight hand tracking fallback using only OpenCV
For situations where MediaPipe conflicts arise
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

class LightweightHandTracker:
    """Simple hand tracking using OpenCV contours and convex hull."""
    
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect_hand_simple(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """
        Simple hand detection using skin color and contours.
        Returns: (x, y, width, height) of hand bounding box or None
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color range (adjust for lighting)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find largest contour (assume it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours
        if cv2.contourArea(largest_contour) < 1000:
            return None
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    
    def get_hand_center(self, frame) -> Optional[Tuple[float, float]]:
        """Get normalized hand center position (0-1 range)."""
        hand_box = self.detect_hand_simple(frame)
        if hand_box is None:
            return None
            
        x, y, w, h = hand_box
        center_x = (x + w/2) / frame.shape[1]  # Normalize to 0-1
        center_y = (y + h/2) / frame.shape[0]
        
        return (center_x, center_y)
    
    def estimate_gesture(self, frame) -> str:
        """Basic gesture recognition - open/closed hand."""
        hand_box = self.detect_hand_simple(frame)
        if hand_box is None:
            return "none"
            
        x, y, w, h = hand_box
        
        # Simple heuristic: if hand is more square, it's likely closed
        aspect_ratio = w / h if h > 0 else 0
        
        if 0.7 < aspect_ratio < 1.3:
            return "closed"  # More square = fist
        else:
            return "open"    # More rectangular = open hand

def test_lightweight_tracker():
    """Test the lightweight tracker."""
    print("Testing lightweight hand tracker...")
    
    cap = cv2.VideoCapture(0)
    tracker = LightweightHandTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get hand position
        center = tracker.get_hand_center(frame)
        gesture = tracker.estimate_gesture(frame)
        
        # Draw results
        if center:
            x_pixel = int(center[0] * frame.shape[1])
            y_pixel = int(center[1] * frame.shape[0])
            cv2.circle(frame, (x_pixel, y_pixel), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Pos: {center[0]:.2f}, {center[1]:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Lightweight Hand Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_lightweight_tracker()