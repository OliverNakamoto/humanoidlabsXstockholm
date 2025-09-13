import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional

class RobotHandController:
    """
    Computer vision pipeline for ARM-SO101 robot control using hand tracking.
    Replaces expensive leader arms with MediaPipe hand detection.
    """
    
    def __init__(self):
        # MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Robot workspace bounds (ARM-SO101 specifications)
        self.robot_workspace = {
            'x_range': (-0.4, 0.4),  # meters
            'y_range': (-0.4, 0.4),  # meters
            'z_range': (0.1, 0.6)    # meters above base
        }
        
        # Camera calibration parameters
        self.camera_bounds = None
        self.depth_reference = 0.15  # Reference hand size at 0.5m depth
        
        # Smoothing buffer
        self.position_history = []
        self.gripper_history = []
        self.smooth_window = 3
        
    def detect_hand(self, frame: np.ndarray) -> Optional[dict]:
        """
        Detect hand landmarks and extract bounding box.
        
        Args:
            frame: Camera frame
            
        Returns:
            Dict with hand data or None if no hand detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Get first detected hand
        landmarks = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract landmark coordinates
        x_coords = [int(lm.x * w) for lm in landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in landmarks.landmark]
        
        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Hand center point
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Hand size (diagonal of bounding box)
        hand_size = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        
        return {
            'center': (center_x, center_y),
            'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
            'size': hand_size,
            'landmarks': landmarks
        }
    
    def estimate_3d_position(self, hand_data: dict, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Convert hand position to 3D robot coordinates.
        
        Args:
            hand_data: Hand detection data
            frame_shape: (height, width) of frame
            
        Returns:
            (x, y, z) normalized coordinates [-1, 1]
        """
        h, w = frame_shape
        center_x, center_y = hand_data['center']
        hand_size = hand_data['size']
        
        # Auto-calibrate camera bounds if not set
        if self.camera_bounds is None:
            self.camera_bounds = {
                'x_min': int(w * 0.1), 'x_max': int(w * 0.9),
                'y_min': int(h * 0.1), 'y_max': int(h * 0.9)
            }
        
        # Map X-Y to robot workspace
        x_norm = 2 * (center_x - self.camera_bounds['x_min']) / (self.camera_bounds['x_max'] - self.camera_bounds['x_min']) - 1
        y_norm = 2 * (center_y - self.camera_bounds['y_min']) / (self.camera_bounds['y_max'] - self.camera_bounds['y_min']) - 1
        
        # Invert Y (camera Y increases downward, robot Y upward)
        y_norm = -y_norm
        
        # Estimate Z from hand size (larger hand = closer/lower Z)
        size_ratio = hand_size / (w * 0.2)  # Normalize by frame width
        z_norm = 1 - np.clip(size_ratio * 2, 0, 2)  # Larger size = lower Z
        
        # Clamp to [-1, 1] range
        x_norm = np.clip(x_norm, -1, 1)
        y_norm = np.clip(y_norm, -1, 1)
        z_norm = np.clip(z_norm, -1, 1)
        
        return x_norm, y_norm, z_norm
    
    def detect_gripper_state(self, landmarks) -> float:
        """
        Detect if hand is open (True) or closed (False).
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Float [0,1] where 1=open, 0=closed
        """
        # Key finger tip and joint landmarks
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        
        # Calculate finger extensions
        extensions = []
        for tip, pip in zip(finger_tips, finger_pips):
            tip_pos = landmarks.landmark[tip]
            pip_pos = landmarks.landmark[pip]
            wrist_pos = landmarks.landmark[0]
            
            # Distance from wrist to tip vs pip
            tip_dist = np.sqrt((tip_pos.x - wrist_pos.x)**2 + (tip_pos.y - wrist_pos.y)**2)
            pip_dist = np.sqrt((pip_pos.x - wrist_pos.x)**2 + (pip_pos.y - wrist_pos.y)**2)
            
            # Extension ratio
            if pip_dist > 0:
                extensions.append(tip_dist / pip_dist)
        
        # Average extension (open hand = higher values)
        avg_extension = np.mean(extensions)
        
        # Convert to 0-1 range (tune threshold as needed)
        gripper_open = np.clip((avg_extension - 1.0) / 0.3, 0, 1)
        
        return gripper_open
    
    def smooth_output(self, position: Tuple[float, float, float], gripper: float) -> Tuple[Tuple[float, float, float], float]:
        """
        Apply smoothing filter to reduce jitter.
        
        Args:
            position: Current (x, y, z)
            gripper: Current gripper state
            
        Returns:
            Smoothed (position, gripper)
        """
        # Add to history buffers
        self.position_history.append(position)
        self.gripper_history.append(gripper)
        
        # Keep only recent history
        if len(self.position_history) > self.smooth_window:
            self.position_history.pop(0)
        if len(self.gripper_history) > self.smooth_window:
            self.gripper_history.pop(0)
        
        # Calculate smoothed values
        if len(self.position_history) > 0:
            smooth_pos = tuple(np.mean([p[i] for p in self.position_history]) for i in range(3))
        else:
            smooth_pos = position
            
        if len(self.gripper_history) > 0:
            smooth_gripper = np.mean(self.gripper_history)
        else:
            smooth_gripper = gripper
            
        return smooth_pos, smooth_gripper
    
    def process_frame(self, frame: np.ndarray) -> List[float]:
        """
        Main processing function - returns LeRobot compatible action.
        
        Args:
            frame: Camera frame
            
        Returns:
            [x, y, z, gripper_open] normalized coordinates
        """
        # Detect hand
        hand_data = self.detect_hand(frame)
        
        if hand_data is None:
            return [0.0, 0.0, 0.0, 0.0]  # No hand detected
        
        # Get 3D position
        x, y, z = self.estimate_3d_position(hand_data, frame.shape[:2])
        
        # Get gripper state
        gripper = self.detect_gripper_state(hand_data['landmarks'])
        
        # Apply smoothing
        (x_smooth, y_smooth, z_smooth), gripper_smooth = self.smooth_output((x, y, z), gripper)
        
        return [x_smooth, y_smooth, z_smooth, gripper_smooth]
    
    def get_confidence_score(self) -> float:
        """
        Return confidence score based on detection stability.
        
        Returns:
            Confidence score [0,1]
        """
        if len(self.position_history) < 2:
            return 0.5
        
        # Calculate position variance as confidence measure
        recent_positions = np.array(self.position_history[-3:])
        if len(recent_positions) > 1:
            variance = np.mean(np.var(recent_positions, axis=0))
            confidence = 1.0 / (1.0 + variance * 10)  # Lower variance = higher confidence
            return np.clip(confidence, 0.0, 1.0)
        
        return 0.5
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """
        Add visualization overlay to frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with visualization overlay
        """
        vis_frame = frame.copy()
        hand_data = self.detect_hand(frame)
        
        if hand_data is None:
            cv2.putText(vis_frame, "No hand detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_frame
        
        # Draw bounding box
        x, y, w, h = hand_data['bbox']
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw center point
        center_x, center_y = hand_data['center']
        cv2.circle(vis_frame, (center_x, center_y), 8, (255, 0, 0), -1)
        
        # Get current action
        action = self.process_frame(frame)
        confidence = self.get_confidence_score()
        
        # Display action values
        cv2.putText(vis_frame, f"Position: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Gripper: {action[3]:.2f} ({'Open' if action[3] > 0.5 else 'Closed'})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Confidence: {confidence:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame