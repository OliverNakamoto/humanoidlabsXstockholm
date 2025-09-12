import cv2
import mediapipe as mp
import numpy as np

class SimpleHandTracker:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands now
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Robot workspace box (big box)
        self.workspace_box = None
        
        # Z-axis calibration points
        self.z_calibration = {
            'table_hand_size': None,    # Hand size when touching table (0cm)
            'high_hand_size': None,     # Hand size when 15cm above table
            'is_calibrated': False
        }
        
        # Pinch calibration for left hand (thumb tip to index tip)
        self.pinch_calibration = {
            'closed_distance': None,    # Normalized distance when pinch is fully closed
            'open_distance': None,      # Normalized distance when pinch is fully open
            'is_calibrated': False
        }
    
    def is_hand_open(self, landmarks):
        """
        Detect if hand is open (True) or closed (False)
        Uses finger tip positions relative to their PIP joints
        """
        # Finger tip and PIP joint landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        fingers_up = 0
        
        # Check each finger
        for tip, pip in zip(finger_tips, finger_pips):
            tip_y = landmarks.landmark[tip].y
            pip_y = landmarks.landmark[pip].y
            
            # For most fingers, tip should be above PIP when extended
            if tip == 4:  # Thumb - check x coordinate instead
                if landmarks.landmark[tip].x > landmarks.landmark[pip].x:
                    fingers_up += 1
            else:  # Other fingers - check y coordinate
                if tip_y < pip_y:  # tip above pip means finger extended
                    fingers_up += 1
        
        # Hand is "open" if 3 or more fingers are extended
        return fingers_up >= 3
    
    def get_hand_size(self, landmarks, w, h):
        """Calculate hand size from landmarks"""
        x_coords = [int(lm.x * w) for lm in landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Return diagonal size as hand size measure
        return np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    
    def calculate_palm_bounding_box(self, landmarks, w, h, padding=5):
        """
        Calculate precise palm area bounding box:
        - X-axis: stretches from point 5 (index MCP) to point 17 (pinky MCP)
        - Y-axis: spans from point 0 (wrist) to average Y of points 5 and 17
        
        Args:
            landmarks: MediaPipe hand landmarks
            w, h: Frame width and height
            padding: Small padding in pixels around the palm area
            
        Returns:
            dict: Palm bounding box data with bbox coordinates, center, and dimensions
        """
        # MediaPipe hand landmark indices for palm area
        WRIST = 0
        INDEX_MCP = 5  # Index finger base knuckle
        PINKY_MCP = 17  # Pinky finger base knuckle
        
        # Get palm landmark coordinates in pixel space
        wrist = landmarks.landmark[WRIST]
        index_mcp = landmarks.landmark[INDEX_MCP]
        pinky_mcp = landmarks.landmark[PINKY_MCP]
        
        # Convert to pixel coordinates
        wrist_px = (int(wrist.x * w), int(wrist.y * h))
        index_mcp_px = (int(index_mcp.x * w), int(index_mcp.y * h))
        pinky_mcp_px = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))
        
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
        bbox_width = min(w - bbox_x, x_max - x_min + 2 * padding)
        bbox_height = min(h - bbox_y, y_max - y_min + 2 * padding)
        
        # Calculate palm center
        palm_center_x = (x_min + x_max) // 2
        palm_center_y = (y_min + y_max) // 2
        
        # Calculate actual palm dimensions
        palm_width = x_max - x_min
        palm_height = y_max - y_min
        
        return {
            'bbox': (bbox_x, bbox_y, bbox_width, bbox_height),
            'center': (palm_center_x, palm_center_y),
            'palm_width': palm_width,
            'palm_height': palm_height,
            'aspect_ratio': palm_width / palm_height if palm_height > 0 else 1.0,
            'key_points': {
                'wrist': wrist_px,
                'index_mcp': index_mcp_px,
                'pinky_mcp': pinky_mcp_px,
                'avg_mcp_y': avg_mcp_y
            }
        }
    
    def get_normalized_pinch_distance(self, landmarks):
        """
        Calculate normalized pinch distance between thumb tip (4) and index finger tip (8).
        Normalizes using palm width to handle camera distance variations.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            float: Normalized pinch distance
        """
        # Get thumb tip (4) and index finger tip (8)
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        # Calculate raw distance between thumb and index tips
        raw_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        # Normalize by palm width to handle distance variations
        # Use distance between index MCP (5) and pinky MCP (17) as reference
        index_mcp = landmarks.landmark[5]
        pinky_mcp = landmarks.landmark[17]
        palm_width = np.sqrt((index_mcp.x - pinky_mcp.x)**2 + (index_mcp.y - pinky_mcp.y)**2)
        
        # Normalize the pinch distance by palm width
        if palm_width > 0:
            normalized_distance = raw_distance / palm_width
        else:
            normalized_distance = 0
        
        return normalized_distance
    
    def calibrate_pinch(self, landmarks, state):
        """
        Calibrate pinch states for left hand.
        
        Args:
            landmarks: MediaPipe hand landmarks
            state: 'closed' for fully closed pinch, 'open' for fully open pinch
        """
        normalized_distance = self.get_normalized_pinch_distance(landmarks)
        
        if state == 'closed':
            self.pinch_calibration['closed_distance'] = normalized_distance
            print(f"Pinch closed calibration: {normalized_distance:.3f}")
        elif state == 'open':
            self.pinch_calibration['open_distance'] = normalized_distance
            print(f"Pinch open calibration: {normalized_distance:.3f}")
        
        # Check if both states are calibrated
        if (self.pinch_calibration['closed_distance'] is not None and 
            self.pinch_calibration['open_distance'] is not None):
            self.pinch_calibration['is_calibrated'] = True
            print("Pinch calibration complete!")
    
    def get_pinch_value(self, landmarks):
        """
        Get pinch value scaled from 0-100 (0=closed, 100=fully open).
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            int: Pinch value from 0-100, or None if not calibrated
        """
        if not self.pinch_calibration['is_calibrated']:
            return None
        
        current_distance = self.get_normalized_pinch_distance(landmarks)
        closed_dist = self.pinch_calibration['closed_distance']
        open_dist = self.pinch_calibration['open_distance']
        
        # Linear interpolation between closed and open states
        if open_dist != closed_dist:
            # Scale from 0 to 100
            pinch_ratio = (current_distance - closed_dist) / (open_dist - closed_dist)
            pinch_value = int(np.clip(pinch_ratio * 100, 0, 100))
        else:
            pinch_value = 50  # Default middle value if distances are equal
        
        return pinch_value
    
    def detect_hand_side(self, landmarks):
        """
        Detect if hand is left or right based on thumb position relative to other fingers.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            str: 'Left' or 'Right'
        """
        # Compare thumb tip (4) x-coordinate with pinky MCP (17) x-coordinate
        thumb_x = landmarks.landmark[4].x
        pinky_x = landmarks.landmark[17].x
        
        # When viewing the mirrored camera feed:
        # If thumb is to the left of pinky, it's a right hand (from user's perspective)
        # If thumb is to the right of pinky, it's a left hand (from user's perspective)
        if thumb_x < pinky_x:
            return 'Right'
        else:
            return 'Left'
    
    def calibrate_z_axis(self, landmarks, w, h, position):
        """
        Calibrate Z-axis depth estimation
        position: 'table' for table level, 'high' for 15cm above
        """
        hand_size = self.get_hand_size(landmarks, w, h)
        
        if position == 'table':
            self.z_calibration['table_hand_size'] = hand_size
            print(f"Table calibration: hand size = {hand_size:.1f}")
        elif position == 'high':
            self.z_calibration['high_hand_size'] = hand_size
            print(f"High calibration: hand size = {hand_size:.1f}")
        
        # Check if both points are calibrated
        if (self.z_calibration['table_hand_size'] is not None and 
            self.z_calibration['high_hand_size'] is not None):
            self.z_calibration['is_calibrated'] = True
            print("Z-axis calibration complete!")
    
    def get_z_position(self, landmarks, w, h):
        """
        Get Z position (0 = table, 1 = 15cm above)
        Returns None if not calibrated
        """
        if not self.z_calibration['is_calibrated']:
            return None
        
        current_hand_size = self.get_hand_size(landmarks, w, h)
        table_size = self.z_calibration['table_hand_size']
        high_size = self.z_calibration['high_hand_size']
        
        # Linear interpolation between the two calibration points
        # Larger hand = closer to table (lower Z)
        if table_size != high_size:
            z_ratio = (current_hand_size - high_size) / (table_size - high_size)
            z_position = np.clip(z_ratio, 0, 1)  # 0 = high, 1 = table
            return 1 - z_position  # Flip so 0 = table, 1 = high
        
        return 0.5  # Default middle if sizes are equal
    
    def set_workspace_box(self, frame_shape):
        """Set the robot workspace box based on frame size"""
        h, w = frame_shape[:2]
        margin = 50
        self.workspace_box = {
            'x1': margin,
            'y1': margin, 
            'x2': w - margin,
            'y2': h - margin
        }
    
    def process_frame(self, frame):
        """Process frame and detect hand"""
        h, w = frame.shape[:2]
        
        # Set workspace box if not set
        if self.workspace_box is None:
            self.set_workspace_box(frame.shape)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Draw big workspace box
        cv2.rectangle(frame, 
                     (self.workspace_box['x1'], self.workspace_box['y1']),
                     (self.workspace_box['x2'], self.workspace_box['y2']),
                     (0, 255, 255), 3)  # Yellow box
        
        cv2.putText(frame, "Robot Workspace", 
                   (self.workspace_box['x1'], self.workspace_box['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Process all detected hands
        right_hand_data = None
        left_hand_pinch = None
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Detect which hand this is
                hand_side = self.detect_hand_side(hand_landmarks)
                
                # Process differently based on hand side
                if hand_side == 'Right':
                    # RIGHT HAND: Position tracking
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                              self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),  # Green
                                              self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1))
                    
                    # Calculate palm-based bounding box
                    palm_data = self.calculate_palm_bounding_box(hand_landmarks, w, h)
                    palm_x, palm_y, palm_w, palm_h = palm_data['bbox']
                    
                    # Draw palm-based bounding box (GREEN)
                    cv2.rectangle(frame, (palm_x, palm_y), (palm_x + palm_w, palm_y + palm_h), (0, 255, 0), 2)
                    cv2.putText(frame, "Right Hand (Position)", (palm_x, palm_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Use palm center as tracking point
                    center_x, center_y = palm_data['center']
                    
                    # Check if hand is open or closed
                    hand_open = self.is_hand_open(hand_landmarks)
                    
                    # Get Z position if calibrated
                    z_pos = self.get_z_position(hand_landmarks, w, h)
                    
                    # Draw palm center point
                    cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)  # Green dot
                    cv2.putText(frame, f"R: ({center_x}, {center_y})", 
                               (center_x + 15, center_y - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Store right hand data
                    right_hand_data = {
                        'center': (center_x, center_y),
                        'open': hand_open,
                        'z_pos': z_pos,
                        'palm_data': palm_data
                    }
                    
                else:
                    # LEFT HAND: Pinch detection
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                              self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2),  # Magenta
                                              self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=1))
                    
                    # Get pinch value
                    pinch_value = self.get_pinch_value(hand_landmarks)
                    
                    # Draw pinch visualization
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    
                    thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    index_px = (int(index_tip.x * w), int(index_tip.y * h))
                    
                    # Draw line between thumb and index tips
                    cv2.line(frame, thumb_px, index_px, (255, 0, 255), 2)
                    cv2.circle(frame, thumb_px, 6, (255, 0, 255), -1)  # Thumb tip
                    cv2.circle(frame, index_px, 6, (255, 0, 255), -1)  # Index tip
                    
                    # Draw pinch area label
                    mid_x = (thumb_px[0] + index_px[0]) // 2
                    mid_y = (thumb_px[1] + index_px[1]) // 2
                    cv2.putText(frame, "Left Hand (Pinch)", (mid_x - 50, mid_y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    left_hand_pinch = pinch_value
        
        # Display information for both hands
        if right_hand_data is not None:
            center_x, center_y = right_hand_data['center']
            hand_open = right_hand_data['open']
            z_pos = right_hand_data['z_pos']
            palm_data = right_hand_data['palm_data']
            
            # Display right hand state
            state_text = "OPEN" if hand_open else "CLOSED"
            state_color = (0, 255, 0) if hand_open else (0, 0, 255)
            cv2.putText(frame, f"Right Hand: {state_text}", 
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            
            # Display palm metrics
            cv2.putText(frame, f"Palm Ratio: {palm_data['aspect_ratio']:.2f}", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display Z position if calibrated
            if z_pos is not None:
                z_cm = z_pos * 15
                cv2.putText(frame, f"Z: {z_cm:.1f}cm", 
                           (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Print right hand coordinates
            if z_pos is not None:
                z_cm = z_pos * 15
                print(f"Right Hand - Palm center: x={center_x}, y={center_y}, Z={z_cm:.1f}cm, Open: {hand_open}")
            else:
                print(f"Right Hand - Palm center: x={center_x}, y={center_y}, Open: {hand_open}")
        
        # Display left hand pinch information
        if left_hand_pinch is not None:
            cv2.putText(frame, f"Left Pinch: {left_hand_pinch}", 
                       (w - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            print(f"Left Hand - Pinch value: {left_hand_pinch}")
        else:
            # Show calibration instructions if not calibrated
            if not self.pinch_calibration['is_calibrated']:
                cv2.putText(frame, "Press 'c' pinch closed, 'o' pinch open", 
                           (w - 400, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show general calibration instructions
        if right_hand_data is not None and self.z_calibration['is_calibrated'] == False:
            cv2.putText(frame, "Press 't' for table, 'h' for high (15cm)", 
                       (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    tracker = SimpleHandTracker()
    
    print("Dual Hand Tracker: Right Hand Position + Left Hand Pinch")
    print("Yellow box = Robot workspace")
    print("Green = Right hand (position tracking with palm-based bounding box)")
    print("Magenta = Left hand (pinch detection)")
    print("Right hand Z-axis calibration: Press 't' for table, 'h' for high (15cm)")
    print("Left hand pinch calibration: Press 'c' for closed pinch, 'o' for open pinch")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process frame
        frame = tracker.process_frame(frame)
        
        cv2.imshow('Simple Hand Tracker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):  # Calibrate table position for right hand
            # Process frame to get hands data
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_side = tracker.detect_hand_side(hand_landmarks)
                    if hand_side == 'Right':
                        tracker.calibrate_z_axis(hand_landmarks, frame.shape[1], frame.shape[0], 'table')
                        break
        elif key == ord('h'):  # Calibrate high position (15cm) for right hand
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_side = tracker.detect_hand_side(hand_landmarks)
                    if hand_side == 'Right':
                        tracker.calibrate_z_axis(hand_landmarks, frame.shape[1], frame.shape[0], 'high')
                        break
        elif key == ord('c'):  # Calibrate closed pinch for left hand
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_side = tracker.detect_hand_side(hand_landmarks)
                    if hand_side == 'Left':
                        tracker.calibrate_pinch(hand_landmarks, 'closed')
                        break
        elif key == ord('o'):  # Calibrate open pinch for left hand
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_side = tracker.detect_hand_side(hand_landmarks)
                    if hand_side == 'Left':
                        tracker.calibrate_pinch(hand_landmarks, 'open')
                        break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()