import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, Optional, List, Dict
from collections import deque
import math

class HandTracker:
    """
    Computer Vision pipeline for robot arm control using hand tracking.
    Replaces expensive leader arms with MediaPipe-based hand detection.
    """
    
    def __init__(self, 
                 smoothing_window: int = 5,
                 confidence_threshold: float = 0.7,
                 workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize the hand tracker.
        
        Args:
            smoothing_window: Number of frames for smoothing filter
            confidence_threshold: Minimum confidence for hand detection
            workspace_bounds: Dict with 'x', 'y', 'z' keys mapping to (min, max) tuples
        """
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only track one hand
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # Workspace calibration - default values
        self.workspace_bounds = workspace_bounds or {
            'x': (-0.5, 0.5),  # meters
            'y': (-0.5, 0.5),  # meters  
            'z': (0.0, 0.8)    # meters (height above table)
        }
        
        # Smoothing buffers
        self.smoothing_window = smoothing_window
        self.position_buffer = deque(maxlen=smoothing_window)
        self.gripper_buffer = deque(maxlen=smoothing_window)
        
        # Calibration parameters
        self.camera_bounds = None  # Will be set during calibration
        self.depth_calibration = {
            'reference_hand_size': 0.15,  # Reference hand size at 1m distance
            'size_to_depth_factor': 0.3   # Scaling factor for depth estimation
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def calibrate_workspace(self, camera_frame_size: Tuple[int, int]) -> None:
        """
        Calibrate camera bounds to workspace mapping.
        
        Args:
            camera_frame_size: (width, height) of camera frame
        """
        frame_w, frame_h = camera_frame_size
        
        # Map camera frame to workspace bounds
        # Assume camera captures the full workspace with some padding
        self.camera_bounds = {
            'x': (int(frame_w * 0.1), int(frame_w * 0.9)),  # 10% padding on sides
            'y': (int(frame_h * 0.1), int(frame_h * 0.9)),  # 10% padding top/bottom
        }
        
    def _normalize_coordinates(self, x: int, y: int, hand_size: float, 
                             frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to normalized robot workspace coordinates.
        
        Args:
            x, y: Pixel coordinates of hand center
            hand_size: Relative size of detected hand
            frame_shape: (height, width) of camera frame
            
        Returns:
            (x_norm, y_norm, z_norm): Normalized coordinates in [-1, 1] range
        """
        frame_h, frame_w = frame_shape
        
        if self.camera_bounds is None:
            self.calibrate_workspace((frame_w, frame_h))
        
        # Map pixel coordinates to workspace
        x_min, x_max = self.camera_bounds['x']
        y_min, y_max = self.camera_bounds['y']
        
        # Normalize to [-1, 1] range
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
        
        # Clamp to valid range
        x_norm = np.clip(x_norm, -1, 1)
        y_norm = np.clip(y_norm, -1, 1)
        
        # Estimate depth from hand size (larger = closer/lower)
        # Invert Y axis to match robot coordinates (up is positive)
        y_norm = -y_norm
        
        # Calculate z based on hand size relative to reference
        size_ratio = hand_size / self.depth_calibration['reference_hand_size']
        z_norm = np.clip(1 - size_ratio * self.depth_calibration['size_to_depth_factor'], -1, 1)
        
        return x_norm, y_norm, z_norm
    
    def _detect_gripper_state(self, landmarks) -> float:
        """
        Detect if hand is open or closed for gripper control.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Float between 0 (closed) and 1 (open)
        """
        # Get key landmark points
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8] 
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate finger extensions
        wrist = landmarks[0]
        
        # Distance from wrist to fingertips
        distances = []
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
            dist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            distances.append(dist)
        
        # Average finger extension (normalized)
        avg_extension = np.mean(distances)
        
        # Convert to gripper openness (0 = closed, 1 = open)
        # This threshold may need tuning based on hand size
        gripper_openness = np.clip((avg_extension - 0.15) / 0.1, 0, 1)
        
        return gripper_openness
    
    def _calculate_hand_size(self, landmarks) -> float:
        """
        Calculate relative hand size from landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Relative hand size
        """
        # Calculate bounding box of hand landmarks
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Return diagonal size as hand size measure
        return math.sqrt(width**2 + height**2)
    
    def _smooth_output(self, position: Tuple[float, float, float], 
                      gripper: float) -> Tuple[Tuple[float, float, float], float]:
        """
        Apply smoothing filter to reduce jitter.
        
        Args:
            position: Current (x, y, z) position
            gripper: Current gripper state
            
        Returns:
            Smoothed (position, gripper) tuple
        """
        # Add to buffers
        self.position_buffer.append(position)
        self.gripper_buffer.append(gripper)
        
        # Calculate smoothed values
        if len(self.position_buffer) > 0:
            positions_array = np.array(list(self.position_buffer))
            smooth_pos = np.mean(positions_array, axis=0)
            smooth_pos = tuple(smooth_pos)
        else:
            smooth_pos = position
            
        if len(self.gripper_buffer) > 0:
            smooth_gripper = np.mean(list(self.gripper_buffer))
        else:
            smooth_gripper = gripper
            
        return smooth_pos, smooth_gripper
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single camera frame and extract robot control commands.
        
        Args:
            frame: Camera frame as numpy array
            
        Returns:
            Dict containing:
            - 'action': [x, y, z, gripper_open] - normalized coordinates
            - 'confidence': float - detection confidence
            - 'detected': bool - whether hand was detected
            - 'fps': float - current processing FPS
            - 'visualization': annotated frame for display
        """
        # Update FPS counter
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # Default output
        output = {
            'action': [0.0, 0.0, 0.0, 0.0],
            'confidence': 0.0,
            'detected': False,
            'fps': self.current_fps,
            'visualization': vis_frame
        }
        
        if results.multi_hand_landmarks:
            # Use the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on visualization
            self.mp_draw.draw_landmarks(vis_frame, hand_landmarks, 
                                      self.mp_hands.HAND_CONNECTIONS)
            
            # Calculate hand center (using wrist and middle finger MCP)
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            
            center_x = int((wrist.x + middle_mcp.x) / 2 * frame.shape[1])
            center_y = int((wrist.y + middle_mcp.y) / 2 * frame.shape[0])
            
            # Calculate hand size
            hand_size = self._calculate_hand_size(hand_landmarks.landmark)
            
            # Convert to robot coordinates
            x_norm, y_norm, z_norm = self._normalize_coordinates(
                center_x, center_y, hand_size, frame.shape[:2]
            )
            
            # Detect gripper state
            gripper_state = self._detect_gripper_state(hand_landmarks.landmark)
            
            # Apply smoothing
            smooth_pos, smooth_gripper = self._smooth_output(
                (x_norm, y_norm, z_norm), gripper_state
            )
            
            # Update output
            output.update({
                'action': [smooth_pos[0], smooth_pos[1], smooth_pos[2], smooth_gripper],
                'confidence': 1.0,  # MediaPipe doesn't provide explicit confidence
                'detected': True
            })
            
            # Add visualization markers
            cv2.circle(vis_frame, (center_x, center_y), 10, (0, 255, 0), -1)
            cv2.putText(vis_frame, f'Pos: ({x_norm:.2f}, {y_norm:.2f}, {z_norm:.2f})',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f'Gripper: {gripper_state:.2f}',
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add FPS to visualization
        cv2.putText(vis_frame, f'FPS: {self.current_fps:.1f}',
                   (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        output['visualization'] = vis_frame
        return output
        
    def close(self):
        """Clean up resources."""
        self.hands.close()