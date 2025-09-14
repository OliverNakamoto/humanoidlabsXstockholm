#!/usr/bin/env python3

"""
Minimal test to check if computer vision works at all.
This uses only basic OpenCV without MediaPipe to avoid dependency conflicts.
"""

import sys

try:
    import cv2
    print("✓ OpenCV available")
    
    # Test camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Camera not available")
        sys.exit(1)
    else:
        print("✓ Camera available")
        
    print("\nStarting basic camera test...")
    print("Press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        frame = cv2.flip(frame, 1)
        
        # Add simple text overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Move your hand here", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw a simple rectangle where hand should be
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        
        cv2.imshow('Basic Camera Test', frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed successfully!")
    
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Please install: pip install opencv-python")
    sys.exit(1)