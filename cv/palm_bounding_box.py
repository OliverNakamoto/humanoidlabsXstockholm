import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_palm_bounding_box(landmarks, frame_width, frame_height, padding=5):
    """
    Create a precise palm area bounding box:
    - X-axis: stretches from point 5 (index MCP) to point 17 (pinky MCP)
    - Y-axis: spans from point 0 (wrist) to average Y of points 5 and 17
    
    Args:
        landmarks: MediaPipe hand landmarks
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        padding: Small padding in pixels around the palm area (default 5px)
    
    Returns:
        dict: Contains precise palm bounding box coordinates and metrics
            - 'bbox': (x, y, width, height) - tight palm bounding box
            - 'center': (x, y) - palm center point
            - 'palm_width': palm width in pixels (x-axis span)
            - 'palm_height': palm height in pixels (y-axis span)
            - 'aspect_ratio': palm width/height ratio
    """
    # MediaPipe hand landmark indices for palm area
    WRIST = 0
    INDEX_MCP = 5  # Index finger base knuckle
    PINKY_MCP = 17  # Pinky finger base knuckle
    
    # Get palm landmark coordinates in pixel space
    wrist = landmarks.landmark[WRIST]
    index_mcp = landmarks.landmark[INDEX_MCP]
    pinky_mcp = landmarks.landmark[PINKY_MCP]
    
    # Convert normalized coordinates to pixel coordinates
    wrist_px = (int(wrist.x * frame_width), int(wrist.y * frame_height))
    index_mcp_px = (int(index_mcp.x * frame_width), int(index_mcp.y * frame_height))
    pinky_mcp_px = (int(pinky_mcp.x * frame_width), int(pinky_mcp.y * frame_height))
    
    # X-axis: from index MCP to pinky MCP
    x_min = min(index_mcp_px[0], pinky_mcp_px[0])
    x_max = max(index_mcp_px[0], pinky_mcp_px[0])
    
    # Y-axis: from wrist to average of index/pinky MCP Y positions
    avg_mcp_y = (index_mcp_px[1] + pinky_mcp_px[1]) // 2
    y_min = min(wrist_px[1], avg_mcp_y)
    y_max = max(wrist_px[1], avg_mcp_y)
    
    # Apply small padding
    bbox_x = max(0, x_min - padding)
    bbox_y = max(0, y_min - padding)
    bbox_width = min(frame_width - bbox_x, x_max - x_min + 2 * padding)
    bbox_height = min(frame_height - bbox_y, y_max - y_min + 2 * padding)
    
    # Calculate palm center
    palm_center_x = (x_min + x_max) // 2
    palm_center_y = (y_min + y_max) // 2
    
    # Calculate actual palm dimensions
    palm_width = x_max - x_min
    palm_height = y_max - y_min
    
    # Calculate aspect ratio for palm analysis
    aspect_ratio = palm_width / palm_height if palm_height > 0 else 1.0
    
    return {
        'bbox': (bbox_x, bbox_y, bbox_width, bbox_height),
        'center': (palm_center_x, palm_center_y),
        'palm_width': palm_width,
        'palm_height': palm_height,
        'aspect_ratio': aspect_ratio,
        'palm_points': {
            'wrist': wrist_px,
            'index_mcp': index_mcp_px,
            'pinky_mcp': pinky_mcp_px,
            'avg_mcp_y': avg_mcp_y
        }
    }

class PalmHandTracker:
    """
    Enhanced hand tracker using palm-based bounding box for more accurate ratio detection.
    """
    
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    
    def process_frame(self, frame):
        """
        Process frame using palm-based bounding box detection.
        
        Args:
            frame: OpenCV frame
            
        Returns:
            frame: Annotated frame with palm bounding box
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw MediaPipe landmarks
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Calculate palm-based bounding box
            palm_data = calculate_palm_bounding_box(hand_landmarks, w, h)
            
            # Draw palm bounding box
            x, y, bbox_w, bbox_h = palm_data['bbox']
            cv2.rectangle(frame, (x, y), (x + bbox_w, y + bbox_h), (0, 255, 0), 2)
            cv2.putText(frame, "Palm Box", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw palm center
            center_x, center_y = palm_data['center']
            cv2.circle(frame, (center_x, center_y), 8, (255, 0, 0), -1)
            
            # Draw palm key points
            palm_points = palm_data['palm_points']
            cv2.circle(frame, palm_points['wrist'], 6, (0, 0, 255), -1)  # Red for wrist
            cv2.circle(frame, palm_points['index_mcp'], 6, (255, 255, 0), -1)  # Cyan for index MCP
            cv2.circle(frame, palm_points['pinky_mcp'], 6, (255, 0, 255), -1)  # Magenta for pinky MCP
            
            # Draw the average MCP Y line (horizontal line between index and pinky MCPs)
            avg_y = palm_points['avg_mcp_y']
            cv2.line(frame, (palm_points['index_mcp'][0], avg_y), 
                    (palm_points['pinky_mcp'][0], avg_y), (0, 255, 255), 2)  # Yellow line
            
            # Display palm metrics
            cv2.putText(frame, f"Palm W: {palm_data['palm_width']:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Palm H: {palm_data['palm_height']:.1f}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Ratio: {palm_data['aspect_ratio']:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Compare with traditional bounding box
            # Traditional method: all landmarks bounding box
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Draw traditional bounding box for comparison
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
            cv2.putText(frame, "Traditional Box", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            return palm_data
        
        return None
    
    def close(self):
        """Clean up MediaPipe resources."""
        self.hands.close()

def main():
    """
    Demo application showing palm-based bounding box tracking.
    """
    cap = cv2.VideoCapture(0)
    tracker = PalmHandTracker()
    
    print("Precise Palm-Based Hand Tracker")
    print("Green box = Precise palm area (X: point 5 to 17, Y: wrist to avg MCP)")
    print("Yellow box = Traditional bounding box")
    print("Red dot = Wrist (point 0)")
    print("Cyan dot = Index MCP (point 5)")  
    print("Magenta dot = Pinky MCP (point 17)")
    print("Yellow line = Average Y between MCPs")
    print("Blue dot = Palm center")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Mirror the frame for better user experience
        frame = cv2.flip(frame, 1)
        
        # Process frame
        palm_data = tracker.process_frame(frame)
        
        cv2.imshow('Palm-Based Hand Tracker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

if __name__ == "__main__":
    main()