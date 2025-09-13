#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class CVHandTracker:
    """
    Computer vision hand tracker using MediaPipe for teleoperation control.
    Tracks palm position (X, Y, Z) and pinch gesture for gripper control.
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize the CV hand tracker.
        
        Args:
            camera_index: Index of the camera to use (default 0)
        """
        self.camera_index = camera_index
        self.cap = None
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        
        # Calibration data
        self.calibrated = False
        self.palm_bbox_far = None  # Palm bounding box when hand is far (top Z)
        self.palm_bbox_near = None  # Palm bounding box when hand is near (bottom Z)
        self.initial_thumb_index_dist = None  # Initial distance for pinch normalization
        self.workspace_bounds = {
            'x_min': -0.3, 'x_max': 0.3,  # meters
            'y_min': 0.1, 'y_max': 0.5,   # meters
            'z_min': 0.1, 'z_max': 0.4    # meters
        }
        
        # Current tracking data
        self.current_data = {
            'x': 0.0,
            'y': 0.2,
            'z': 0.2,
            'pinch': 0.0,
            'valid': False
        }
        self.data_lock = threading.Lock()
        
        # Tracking thread
        self.tracking_thread = None
        self.stop_tracking = False
        
    def connect(self):
        """Initialize camera and MediaPipe."""
        logger.info("Initializing CV hand tracker...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize MediaPipe hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        logger.info("CV hand tracker connected")
        
    def calibrate(self):
        """Run calibration routine to establish hand tracking bounds."""
        if not self.cap or not self.hands:
            raise RuntimeError("Must connect before calibrating")
        
        logger.info("Starting hand tracking calibration...")
        
        # Calibrate far position
        print("\n" + "="*50)
        print("CALIBRATION - FAR POSITION")
        print("="*50)
        print("Extend your hand FAR from the camera")
        print("Keep your hand flat with fingers spread")
        print("Hold steady when countdown reaches 0")
        
        far_data = self._capture_calibration_point(countdown_from=3)
        if far_data is None:
            raise RuntimeError("Failed to capture far position calibration")
        
        self.palm_bbox_far = far_data['palm_bbox_size']
        far_thumb_index = far_data['thumb_index_dist']
        
        # Calibrate near position
        print("\n" + "="*50)
        print("CALIBRATION - NEAR POSITION")
        print("="*50)
        print("Move your hand CLOSE to the camera")
        print("Keep your hand flat with fingers spread")
        print("Hold steady when countdown reaches 0")
        
        near_data = self._capture_calibration_point(countdown_from=3)
        if near_data is None:
            raise RuntimeError("Failed to capture near position calibration")
        
        self.palm_bbox_near = near_data['palm_bbox_size']
        near_thumb_index = near_data['thumb_index_dist']
        
        # Use average of far and near thumb-index distance for normalization
        self.initial_thumb_index_dist = (far_thumb_index + near_thumb_index) / 2
        
        self.calibrated = True
        
        print("\n" + "="*50)
        print("CALIBRATION COMPLETE!")
        print(f"Far bbox size: {self.palm_bbox_far:.1f}")
        print(f"Near bbox size: {self.palm_bbox_near:.1f}")
        print(f"Initial thumb-index distance: {self.initial_thumb_index_dist:.1f}")
        print("="*50 + "\n")
        
        # Start tracking thread after calibration
        self.start_tracking()
        
    def _capture_calibration_point(self, countdown_from: int = 3) -> Optional[Dict]:
        """
        Capture a calibration point with countdown display.
        
        Returns:
            Dictionary with calibration data or None if failed
        """
        for i in range(countdown_from, 0, -1):
            print(f"  {i}...")
            
            # Show preview with countdown
            start_time = time.time()
            while time.time() - start_time < 1.0:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)  # Mirror for intuitive control
                
                # Add countdown text
                cv2.putText(frame, str(i), (300, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                
                cv2.imshow('Calibration', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return None
        
        print("  CAPTURE!")
        
        # Capture multiple frames for averaging
        samples = []
        for _ in range(10):
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # Calculate palm bounding box
                palm_data = self._calculate_palm_bbox(landmarks, w, h)
                
                # Calculate thumb-index distance
                thumb_tip = landmarks.landmark[4]
                index_tip = landmarks.landmark[8]
                thumb_index_dist = np.sqrt(
                    (thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2
                ) * w  # Convert to pixels
                
                samples.append({
                    'palm_bbox_size': palm_data['palm_width'],
                    'thumb_index_dist': thumb_index_dist
                })
                
                # Show captured frame
                cv2.rectangle(frame, 
                             (palm_data['bbox'][0], palm_data['bbox'][1]),
                             (palm_data['bbox'][0] + palm_data['bbox'][2],
                              palm_data['bbox'][1] + palm_data['bbox'][3]),
                             (0, 255, 0), 3)
                cv2.putText(frame, "CAPTURED", (200, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow('Calibration', frame)
            cv2.waitKey(50)
        
        cv2.destroyAllWindows()
        
        if len(samples) < 5:
            print("  WARNING: Not enough valid samples captured")
            return None
        
        # Average the samples
        avg_bbox_size = np.mean([s['palm_bbox_size'] for s in samples])
        avg_thumb_index = np.mean([s['thumb_index_dist'] for s in samples])
        
        return {
            'palm_bbox_size': avg_bbox_size,
            'thumb_index_dist': avg_thumb_index
        }
    
    def _calculate_palm_bbox(self, landmarks, frame_width, frame_height):
        """Calculate palm bounding box from hand landmarks."""
        # MediaPipe hand landmark indices
        WRIST = 0
        INDEX_MCP = 5
        PINKY_MCP = 17
        
        wrist = landmarks.landmark[WRIST]
        index_mcp = landmarks.landmark[INDEX_MCP]
        pinky_mcp = landmarks.landmark[PINKY_MCP]
        
        # Convert to pixel coordinates
        wrist_px = (int(wrist.x * frame_width), int(wrist.y * frame_height))
        index_mcp_px = (int(index_mcp.x * frame_width), int(index_mcp.y * frame_height))
        pinky_mcp_px = (int(pinky_mcp.x * frame_width), int(pinky_mcp.y * frame_height))
        
        # Calculate bounding box
        x_min = min(index_mcp_px[0], pinky_mcp_px[0])
        x_max = max(index_mcp_px[0], pinky_mcp_px[0])
        
        avg_mcp_y = (index_mcp_px[1] + pinky_mcp_px[1]) // 2
        y_min = min(wrist_px[1], avg_mcp_y)
        y_max = max(wrist_px[1], avg_mcp_y)
        
        palm_width = x_max - x_min
        palm_height = y_max - y_min
        
        # Calculate center
        palm_center_x = (x_min + x_max) // 2
        palm_center_y = (y_min + y_max) // 2
        
        return {
            'bbox': (x_min, y_min, palm_width, palm_height),
            'center': (palm_center_x, palm_center_y),
            'palm_width': palm_width,
            'palm_height': palm_height
        }
    
    def start_tracking(self):
        """Start the tracking thread."""
        if not self.calibrated:
            raise RuntimeError("Must calibrate before starting tracking")
        
        self.stop_tracking = False
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        logger.info("Hand tracking started")
    
    def _tracking_loop(self):
        """Main tracking loop running in separate thread."""
        while not self.stop_tracking:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # Calculate palm data
                palm_data = self._calculate_palm_bbox(landmarks, w, h)
                
                # Calculate normalized X, Y position (0 to 1)
                norm_x = palm_data['center'][0] / w
                norm_y = 1.0 - (palm_data['center'][1] / h)  # Invert Y
                
                # Calculate Z from palm size (near = larger bbox)
                palm_size = palm_data['palm_width']
                if self.palm_bbox_near > self.palm_bbox_far:
                    # Normalize Z: far=0, near=1
                    norm_z = (palm_size - self.palm_bbox_far) / (self.palm_bbox_near - self.palm_bbox_far)
                    norm_z = np.clip(norm_z, 0, 1)
                    norm_z = 1.0 - norm_z  # Invert so far=1, near=0 for robot coords
                else:
                    norm_z = 0.5
                
                # Calculate pinch percentage
                thumb_tip = landmarks.landmark[4]
                index_tip = landmarks.landmark[8]
                thumb_index_dist = np.sqrt(
                    (thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2
                ) * w
                
                # Normalize pinch (0 = open, 100 = closed)
                pinch_ratio = thumb_index_dist / self.initial_thumb_index_dist
                pinch = (1.0 - np.clip(pinch_ratio, 0, 1)) * 100
                
                # Convert normalized coordinates to robot workspace
                x = self.workspace_bounds['x_min'] + norm_x * (self.workspace_bounds['x_max'] - self.workspace_bounds['x_min'])
                y = self.workspace_bounds['y_min'] + norm_y * (self.workspace_bounds['y_max'] - self.workspace_bounds['y_min'])
                z = self.workspace_bounds['z_min'] + norm_z * (self.workspace_bounds['z_max'] - self.workspace_bounds['z_min'])
                
                # Update current data thread-safely
                with self.data_lock:
                    self.current_data = {
                        'x': x,
                        'y': y,
                        'z': z,
                        'pinch': pinch,
                        'valid': True
                    }
            else:
                # No hand detected
                with self.data_lock:
                    self.current_data['valid'] = False
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
    
    def get_current_position(self) -> Tuple[float, float, float, float]:
        """
        Get current hand position and pinch value.
        
        Returns:
            Tuple of (x, y, z, pinch) in robot coordinates
            x, y, z in meters, pinch in percentage (0-100)
        """
        with self.data_lock:
            if not self.current_data['valid']:
                # Return safe default position if no hand detected
                return (0.0, 0.2, 0.2, 0.0)
            
            return (
                self.current_data['x'],
                self.current_data['y'],
                self.current_data['z'],
                self.current_data['pinch']
            )
    
    def disconnect(self):
        """Clean up resources."""
        self.stop_tracking = True
        
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        
        if self.hands:
            self.hands.close()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        logger.info("CV hand tracker disconnected")


# Global instance for shared access
_tracker_instance = None


def get_tracker_instance(camera_index: int = 0) -> CVHandTracker:
    """Get or create the global tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CVHandTracker(camera_index)
    return _tracker_instance