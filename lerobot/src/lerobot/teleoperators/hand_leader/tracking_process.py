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

"""
MediaPipe Hand Tracking Process

Isolated process for hand tracking using MediaPipe.
Communicates via Unix domain sockets to avoid dependency conflicts.
"""

import cv2
import mediapipe as mp
import numpy as np
import socket
import time
import signal
import sys
import os
import argparse
import logging
from typing import Optional, Dict, Any, Tuple
import threading

from ipc_protocol import (
    HandTrackingData, 
    HandTrackingProtocol, 
    MessageValidator,
    SOCKET_PATH, 
    HEARTBEAT_INTERVAL,
    MSG_TYPE_CALIBRATION,
    MSG_TYPE_CALIBRATION_COMPLETE,
    MSG_TYPE_SHUTDOWN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MediaPipeHandTracker:
    """MediaPipe-based hand tracking with palm bounding box detection."""
    
    def __init__(self, camera_index: int = 0, show_window: bool = False):
        self.camera_index = camera_index
        self.show_window = show_window
        self.cap = None
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        
        # Calibration data
        self.calibrated = False
        self.palm_bbox_far = None  # Palm bounding box when hand is far
        self.palm_bbox_near = None  # Palm bounding box when hand is near
        self.initial_thumb_index_dist = None  # Initial distance for pinch normalization
        
        # Workspace bounds (robot coordinate system) - FIXED PHYSICAL DIMENSIONS
        self.workspace_bounds = {
            'x_min': 0.10, 'x_max': 0.30,  # 5cm to 30cm forward (25cm range)
            'y_min': -0.10, 'y_max': 0.10, # ±10cm left/right (20cm range)
            'z_min': 0.10, 'z_max': 0.40   # 10cm to 40cm height (30cm range)
        }
        
        # Current tracking state
        self.current_data = HandTrackingData.create_default()
        self.frame_count = 0
        
        # Orientation tracking
        self.neutral_orientation = None  # Neutral quaternion from calibration
        self.prev_euler = np.array([0.0, 0.0, 0.0])  # Previous Euler angles for EMA smoothing
        self.orientation_ema_alpha = 0.3  # EMA smoothing factor (lower = more smoothing)
        
        # Second hand tracking (for curl control)
        self.second_hand_curl = 0.0  # Default uncurled (hand open)
        self.prev_second_hand_curl = 0.0  # Previous curl for EMA smoothing
        self.second_hand_curl_ema_alpha = 0.4  # EMA smoothing for curl

    
    @staticmethod
    def _build_orthonormal_basis(wrist, index_mcp, pinky_mcp):
        """Build orthonormal basis from 3 hand landmarks.
        
        Args:
            wrist, index_mcp, pinky_mcp: 3D points as numpy arrays
            
        Returns:
            3x3 rotation matrix as numpy array
        """
        # Vector from wrist to index MCP (forward direction)
        forward = index_mcp - wrist
        forward = forward / np.linalg.norm(forward)
        
        # Vector from index MCP to pinky MCP (side direction)
        side_raw = pinky_mcp - index_mcp
        
        # Make side orthogonal to forward using Gram-Schmidt
        side = side_raw - np.dot(side_raw, forward) * forward
        side = side / np.linalg.norm(side)
        
        # Up direction (cross product)
        up = np.cross(forward, side)
        up = up / np.linalg.norm(up)
        
        # Build rotation matrix [right, up, forward] (or adjust axes as needed)
        rotation_matrix = np.column_stack([side, up, forward])
        return rotation_matrix
    
    @staticmethod
    def _rotation_matrix_to_quaternion(R):
        """Convert rotation matrix to quaternion.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion as (x, y, z, w)
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])
    
    @staticmethod
    def _quaternion_to_euler(q):
        """Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            q: Quaternion as [qx, qy, qz, qw]
            
        Returns:
            Euler angles as [roll, pitch, yaw] in radians
        """
        qx, qy, qz, qw = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    @staticmethod
    def _euler_to_quaternion(euler):
        """Convert Euler angles to quaternion.
        
        Args:
            euler: Euler angles as [roll, pitch, yaw] in radians
            
        Returns:
            Quaternion as [qx, qy, qz, qw]
        """
        roll, pitch, yaw = euler
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw])
    
    def _calculate_hand_orientation(self, landmarks, frame_width, frame_height):
        """Calculate hand orientation from landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks
            frame_width, frame_height: Frame dimensions
            
        Returns:
            Smoothed quaternion as [qx, qy, qz, qw]
        """
        # Extract key landmarks (wrist, index MCP, pinky MCP)
        wrist = landmarks.landmark[0]  # Wrist
        index_mcp = landmarks.landmark[5]  # Index MCP
        pinky_mcp = landmarks.landmark[17]  # Pinky MCP
        
        # Convert to 3D coordinates (using MediaPipe's z coordinate)
        wrist_3d = np.array([wrist.x * frame_width, wrist.y * frame_height, wrist.z * frame_width])
        index_mcp_3d = np.array([index_mcp.x * frame_width, index_mcp.y * frame_height, index_mcp.z * frame_width])
        pinky_mcp_3d = np.array([pinky_mcp.x * frame_width, pinky_mcp.y * frame_height, pinky_mcp.z * frame_width])
        
        # Build orthonormal basis
        rotation_matrix = self._build_orthonormal_basis(wrist_3d, index_mcp_3d, pinky_mcp_3d)
        
        # Convert to quaternion
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        # Apply camera-to-robot coordinate transformation
        # This mapping may need adjustment based on your camera setup
        # Example: flip Y and Z axes for typical camera mounting
        qx, qy, qz, qw = quaternion
        robot_quaternion = np.array([qx, -qz, -qy, qw])  # Example mapping
        
        # Convert to Euler for smoothing
        euler = self._quaternion_to_euler(robot_quaternion)
        
        # Apply EMA smoothing
        self.prev_euler = self.orientation_ema_alpha * euler + (1 - self.orientation_ema_alpha) * self.prev_euler
        
        # Convert smoothed Euler back to quaternion
        smoothed_quaternion = self._euler_to_quaternion(self.prev_euler)
        
        # Apply neutral orientation offset if calibrated
        if self.neutral_orientation is not None:
            # Subtract neutral orientation (quaternion multiplication)
            smoothed_quaternion = self._quaternion_multiply(smoothed_quaternion, self._quaternion_conjugate(self.neutral_orientation))
        
        return smoothed_quaternion
    
    @staticmethod
    def _quaternion_multiply(q1, q2):
        """Multiply two quaternions."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([x, y, z, w])
    
    @staticmethod
    def _quaternion_conjugate(q):
        """Get quaternion conjugate."""
        qx, qy, qz, qw = q
        return np.array([-qx, -qy, -qz, qw])
        
    def initialize(self) -> bool:
        """Initialize camera and MediaPipe."""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize MediaPipe hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,  # Enable detection of 2 hands
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            logger.info("MediaPipe hand tracker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hand tracker: {e}")
            return False
    
    def calibrate(self) -> bool:
        """Run calibration routine to establish hand tracking bounds."""
        logger.info("Starting hand tracking calibration...")
        
        try:
            # Calibrate far position
            print("\n" + "="*50)
            print("CALIBRATION - FAR POSITION")
            print("="*50)
            print("Extend your hand FAR from the camera")
            print("Keep your hand flat with fingers spread")
            print("Hold steady when countdown reaches 0")
            
            far_data = self._capture_calibration_point(countdown_from=3)
            if far_data is None:
                logger.error("Failed to capture far position calibration")
                return False
            
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
                logger.error("Failed to capture near position calibration")
                return False
            
            self.palm_bbox_near = near_data['palm_bbox_size']
            near_thumb_index = near_data['thumb_index_dist']
            # Capture neutral orientation from near position
            if 'neutral_quaternion' in near_data:
                self.neutral_orientation = near_data['neutral_quaternion']
            
            # Use average of far and near thumb-index distance for normalization
            self.initial_thumb_index_dist = (far_thumb_index + near_thumb_index) / 2
            
            self.calibrated = True
            
            print("\n" + "="*50)
            print("CALIBRATION COMPLETE!")
            print(f"Far bbox size: {self.palm_bbox_far:.1f}")
            print(f"Near bbox size: {self.palm_bbox_near:.1f}")
            print(f"Initial thumb-index distance: {self.initial_thumb_index_dist:.1f}")
            if self.neutral_orientation is not None:
                qx, qy, qz, qw = self.neutral_orientation
                print(f"Neutral orientation (quat): [{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}]")
            print("="*50 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def _capture_calibration_point(self, countdown_from: int = 3) -> Optional[Dict[str, float]]:
        """Capture a calibration point with countdown display."""
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
                
                cv2.imshow('Calibration - MediaPipe Process', frame)
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
                
                # Calculate orientation quaternion for calibration snapshot
                try:
                    q_cal = self._calculate_hand_orientation(landmarks, w, h)
                    qx, qy, qz, qw = float(q_cal[0]), float(q_cal[1]), float(q_cal[2]), float(q_cal[3])
                except Exception as e:
                    logger.debug(f"Calibration orientation calculation failed: {e}")
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                
                samples.append({
                    'palm_bbox_size': palm_data['palm_width'],
                    'thumb_index_dist': thumb_index_dist,
                    'quat': (qx, qy, qz, qw)
                })
                
                # Show captured frame
                cv2.rectangle(frame, 
                             (palm_data['bbox'][0], palm_data['bbox'][1]),
                             (palm_data['bbox'][0] + palm_data['bbox'][2],
                              palm_data['bbox'][1] + palm_data['bbox'][3]),
                             (0, 255, 0), 3)
                cv2.putText(frame, "CAPTURED", (200, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow('Calibration - MediaPipe Process', frame)
            cv2.waitKey(50)
        
        cv2.destroyAllWindows()
        
        if len(samples) < 5:
            logger.warning("Not enough valid samples captured")
            return None
        
        # Average the samples
        avg_bbox_size = np.mean([s['palm_bbox_size'] for s in samples])
        avg_thumb_index = np.mean([s['thumb_index_dist'] for s in samples])
        # Average quaternion components then renormalize
        quats = np.array([s['quat'] for s in samples if 'quat' in s])
        if len(quats) > 0:
            avg_quat = np.mean(quats, axis=0)
            norm = np.linalg.norm(avg_quat)
            if norm > 0:
                avg_quat = avg_quat / norm
            else:
                avg_quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            avg_quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        return {
            'palm_bbox_size': float(avg_bbox_size),
            'thumb_index_dist': float(avg_thumb_index),
            'neutral_quaternion': avg_quat
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
    
    def _calculate_hand_curl(self, landmarks) -> float:
        """Calculate hand curl percentage from MediaPipe landmarks.

        Args:
            landmarks: MediaPipe hand landmarks

        Returns:
            Curl percentage (0-100) where:
            - 0% = hand fully open (fingers extended)
            - 100% = hand fully curled (fingers curled into fist)
            - Maps to wrist_flex joint range (-10° to +90°)
        """
        try:
            # MediaPipe hand landmark indices for finger joints
            # Index finger: MCP=5, PIP=6, DIP=7, TIP=8
            # Middle finger: MCP=9, PIP=10, DIP=11, TIP=12
            # Ring finger: MCP=13, PIP=14, DIP=15, TIP=16
            # Pinky finger: MCP=17, PIP=18, DIP=19, TIP=20

            finger_curl_scores = []

            # Calculate curl for each finger (excluding thumb)
            fingers = [
                [5, 6, 7, 8],    # Index finger
                [9, 10, 11, 12], # Middle finger
                [13, 14, 15, 16], # Ring finger
                [17, 18, 19, 20]  # Pinky finger
            ]

            for finger_indices in fingers:
                mcp_idx, pip_idx, dip_idx, tip_idx = finger_indices

                # Get landmark positions
                mcp = np.array([landmarks.landmark[mcp_idx].x, landmarks.landmark[mcp_idx].y])
                pip = np.array([landmarks.landmark[pip_idx].x, landmarks.landmark[pip_idx].y])
                dip = np.array([landmarks.landmark[dip_idx].x, landmarks.landmark[dip_idx].y])
                tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])

                # Calculate vectors between joints
                mcp_to_pip = pip - mcp
                pip_to_dip = dip - pip
                dip_to_tip = tip - dip

                # Calculate angles between joint segments
                angle1 = self._calculate_angle_between_vectors(mcp_to_pip, pip_to_dip)
                angle2 = self._calculate_angle_between_vectors(pip_to_dip, dip_to_tip)

                # Convert angles to curl score (0 = straight, 1 = fully curled)
                # Smaller angles (more bent) = higher curl score
                curl_score1 = max(0, (np.pi - angle1) / np.pi)  # 0 when straight (π), 1 when bent (0)
                curl_score2 = max(0, (np.pi - angle2) / np.pi)

                # Average the two joint curl scores for this finger
                finger_curl = (curl_score1 + curl_score2) / 2.0
                finger_curl_scores.append(finger_curl)

            # Calculate overall curl as average of all fingers
            overall_curl = np.mean(finger_curl_scores) * 100.0  # Convert to percentage

            # Clamp to 0-100 range
            overall_curl = np.clip(overall_curl, 0.0, 100.0)

            # Apply EMA smoothing
            self.prev_second_hand_curl = (
                self.second_hand_curl_ema_alpha * overall_curl +
                (1.0 - self.second_hand_curl_ema_alpha) * self.prev_second_hand_curl
            )

            return self.prev_second_hand_curl

        except Exception as e:
            logger.debug(f"Failed to calculate hand curl: {e}")
            return 0.0  # Default to uncurled (open hand)

    @staticmethod
    def _calculate_angle_between_vectors(v1, v2):
        """Calculate angle between two 2D vectors."""
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

        # Calculate angle using dot product
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(dot_product)

        return angle
    
    def process_frame(self) -> HandTrackingData:
        """Process single frame and return dual hand tracking data."""
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return HandTrackingData.create_default()
        
        frame = cv2.flip(frame, 1)  # Mirror for intuitive control
        h, w = frame.shape[:2]
        
        # Store original frame for visualization
        display_frame = frame.copy() if self.show_window else None
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Initialize default values
        primary_hand = None
        second_hand = None
        second_hand_curl = 0.0  # Default uncurled (open hand)
        second_hand_detected = False
        
        # Store handedness info for visualization
        primary_hand_info = {"confidence": 0.0}
        second_hand_info = {"confidence": 0.0}
        
        if results.multi_hand_landmarks and self.calibrated:
            # Classify hands by MediaPipe handedness (Left/Right)
            for i, landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness classification for this hand
                handedness_label = "Unknown"
                handedness_confidence = 0.0
                
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness = results.multi_handedness[i]
                    if handedness.classification:
                        handedness_label = handedness.classification[0].label
                        handedness_confidence = handedness.classification[0].score
                
                # Right hand = Primary hand (position + orientation)
                # Left hand = Secondary hand (curl control)
                if handedness_label == "Right":
                    primary_hand = landmarks
                    primary_hand_info = {"confidence": handedness_confidence, "landmarks": landmarks}
                elif handedness_label == "Left":
                    second_hand = landmarks
                    second_hand_detected = True
                    second_hand_info = {"confidence": handedness_confidence, "landmarks": landmarks}
                else:
                    # Fallback: if handedness unclear, treat as primary hand
                    if primary_hand is None:
                        primary_hand = landmarks
                        primary_hand_info = {"confidence": handedness_confidence, "landmarks": landmarks}
                    
                logger.debug(f"Hand {i}: {handedness_label} (confidence: {handedness_confidence:.2f})")
        
        # Process primary hand (position + orientation control)
        if primary_hand is not None:
            # Calculate palm data
            palm_data = self._calculate_palm_bbox(primary_hand, w, h)
            
            # Calculate normalized X, Y position (0 to 1)
            norm_x = palm_data['center'][0] / w
            norm_y = 1.0 - (palm_data['center'][1] / h)  # Invert Y
            
            # Calculate Z from palm size (near = larger bbox) - CALIBRATED
            palm_size = palm_data['palm_width']
            if self.palm_bbox_near > self.palm_bbox_far:
                # Normalize Z: far=0, near=1
                norm_z = (palm_size - self.palm_bbox_far) / (self.palm_bbox_near - self.palm_bbox_far)
                norm_z = np.clip(norm_z, 0, 1)
                norm_z = 1.0 - norm_z  # Invert so far=1, near=0 for robot coords
            else:
                norm_z = 0.5
            
            # Calculate pinch percentage
            thumb_tip = primary_hand.landmark[4]
            index_tip = primary_hand.landmark[8]
            thumb_index_dist = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            ) * w
            
            # Normalize pinch (0 = open, 100 = closed)
            if self.initial_thumb_index_dist > 0:
                pinch_ratio = thumb_index_dist / self.initial_thumb_index_dist
                pinch = (1.0 - np.clip(pinch_ratio, 0, 1)) * 100
            else:
                pinch = 0.0
            
            # Convert normalized coordinates to robot workspace with correct axis mapping:
            # CV X (horizontal) → Robot Y (left/right)
            # CV Y (vertical) → Robot X (forward/backward)
            # CV Z (depth) → Robot Z (height)
            x = self.workspace_bounds['x_min'] + norm_y * (self.workspace_bounds['x_max'] - self.workspace_bounds['x_min'])
            y = self.workspace_bounds['y_min'] + norm_x * (self.workspace_bounds['y_max'] - self.workspace_bounds['y_min'])
            z = self.workspace_bounds['z_min'] + norm_z * (self.workspace_bounds['z_max'] - self.workspace_bounds['z_min'])
            
            # Calculate hand orientation
            try:
                orientation_quat = self._calculate_hand_orientation(primary_hand, w, h)
                qx, qy, qz, qw = orientation_quat
            except Exception as e:
                logger.warning(f"Failed to calculate orientation: {e}")
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # Identity quaternion
            
            hand_detected = True
            
        else:
            # No primary hand detected - use defaults (center of workspace)
            x, y, z = 0.175, 0.0, 0.25  # Mid-range for each axis
            pinch = 0.0
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # Identity quaternion
            palm_data = {'palm_width': 0.0, 'palm_height': 0.0, 'bbox': (0, 0, 0, 0), 'center': (0, 0)}
            hand_detected = False
        
        # Process second hand (curl control)
        if second_hand is not None:
            second_hand_curl = self._calculate_hand_curl(second_hand)
        
        # Draw visualization if requested
        if self.show_window and display_frame is not None:
            # Draw primary hand (RIGHT hand in green)
            if primary_hand is not None:
                self.mp_draw.draw_landmarks(
                    display_frame, primary_hand, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2))
                
                # Draw palm bounding box
                bbox = palm_data['bbox']
                cv2.rectangle(display_frame, 
                             (bbox[0], bbox[1]), 
                             (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                             (0, 255, 0), 2)
                
                # Draw center point
                center = palm_data['center']
                cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
                
                # Add RIGHT HAND label
                label_text = f"RIGHT HAND ({primary_hand_info['confidence']:.1f})"
                cv2.putText(display_frame, label_text, (center[0] - 50, center[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw second hand (LEFT hand in red)
            if second_hand is not None:
                self.mp_draw.draw_landmarks(
                    display_frame, second_hand, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2))
                
                # Get wrist position for label placement
                wrist = second_hand.landmark[0]
                wrist_x = int(wrist.x * w)
                wrist_y = int(wrist.y * h)
                
                # Add LEFT HAND label
                label_text = f"LEFT HAND ({second_hand_info['confidence']:.1f})"
                cv2.putText(display_frame, label_text, (wrist_x - 50, wrist_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display detailed tracking info
            y_offset = 25
            font_scale = 0.5
            thickness = 1

            # Frame info
            frame_text = f"Frame: {w}x{h} pixels"
            cv2.putText(display_frame, frame_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            y_offset += 25

            if hand_detected:
                # Right hand detailed info
                palm_center = palm_data['center']
                cv2.putText(display_frame, "RIGHT HAND (Position Control):", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                y_offset += 20

                pixel_text = f"  Pixel: ({palm_center[0]}, {palm_center[1]})"
                cv2.putText(display_frame, pixel_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                y_offset += 20

                norm_text = f"  Normalized: ({norm_x:.3f}, {norm_y:.3f}, {norm_z:.3f})"
                cv2.putText(display_frame, norm_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                y_offset += 20

                robot_text = f"  Robot: ({x:.3f}, {y:.3f}, {z:.3f})m"
                cv2.putText(display_frame, robot_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                y_offset += 20

                pinch_text = f"  Pinch: {pinch:.1f}%"
                cv2.putText(display_frame, pinch_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                y_offset += 25

            if second_hand_detected:
                # Left hand detailed info
                cv2.putText(display_frame, "LEFT HAND (Curl Control):", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                y_offset += 20

                curl_text = f"  Curl: {second_hand_curl:.1f}%"
                cv2.putText(display_frame, curl_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                y_offset += 25

            # Draw frame center crosshair for reference
            center_x, center_y = w // 2, h // 2
            crosshair_size = 20
            # Horizontal line
            cv2.line(display_frame, (center_x - crosshair_size, center_y),
                    (center_x + crosshair_size, center_y), (255, 255, 255), 1)
            # Vertical line
            cv2.line(display_frame, (center_x, center_y - crosshair_size),
                    (center_x, center_y + crosshair_size), (255, 255, 255), 1)

            # Label the center
            cv2.putText(display_frame, f"Center ({center_x},{center_y})",
                       (center_x + 25, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Status messages
            if not self.calibrated:
                cv2.putText(display_frame, "NOT CALIBRATED - Right hand for position, Left hand for curl", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif not hand_detected and not second_hand_detected:
                cv2.putText(display_frame, "Show hands: Right=Position/Grip, Left=Curl", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Add coordinate mapping explanation in bottom right
            mapping_lines = [
                "Coordinate Mapping:",
                "X (left/right) → Robot Y (±10cm)",
                "Y (up/down) → Robot X (5-30cm)",
                "Z (depth) → Robot Z (height)"
            ]

            for i, line in enumerate(mapping_lines):
                cv2.putText(display_frame, line, (w - 300, h - 80 + i*15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Create result with all data including second hand curl
        result = HandTrackingData(
            timestamp=time.time(),
            hand_detected=hand_detected,
            x=x,
            y=y,
            z=z,
            pinch=pinch,
            palm_width=palm_data['palm_width'],
            palm_height=palm_data['palm_height'],
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            second_hand_pitch=second_hand_curl,
            calibrated=self.calibrated,
            second_hand_detected=second_hand_detected
        )
        
        # Show visualization window if requested
        if self.show_window and display_frame is not None:
            cv2.imshow('Hand Tracking - MediaPipe Process', display_frame)
            # Non-blocking key check
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed, stopping visualization")
                self.show_window = False
                cv2.destroyAllWindows()
        
        return result
    
    def cleanup(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("MediaPipe hand tracker cleaned up")


class HandTrackingServer:
    """Unix domain socket server for hand tracking communication."""
    
    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.socket = None
        self.running = False
        self.tracker = None
        self.heartbeat_thread = None
        self.client_addresses = set()
        
    def start(self, camera_index: int = 0, show_window: bool = False) -> bool:
        """Start the hand tracking server."""
        try:
            # Force remove existing socket file (cleanup from previous runs)
            if os.path.exists(self.socket_path):
                try:
                    os.unlink(self.socket_path)
                    logger.info(f"Removed stale socket file: {self.socket_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove stale socket: {e}")
            
            # Create Unix domain socket
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.socket.bind(self.socket_path)
            self.socket.settimeout(0.1)  # Non-blocking with timeout
            
            # Initialize hand tracker
            self.tracker = MediaPipeHandTracker(camera_index, show_window)
            if not self.tracker.initialize():
                logger.error("Failed to initialize hand tracker")
                return False
            
            # Run calibration
            if not self.tracker.calibrate():
                logger.error("Failed to calibrate hand tracker")
                return False
            
            self.running = True
            
            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            logger.info(f"Hand tracking server started on {self.socket_path}")
            logger.info("Calibration completed - ready for teleoperation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def run(self):
        """Main server loop."""
        logger.info("Starting hand tracking server loop")
        frame_time = 1.0 / 30.0  # Target 30 FPS
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Process hand tracking
                hand_data = self.tracker.process_frame()
                
                # Validate and sanitize data
                hand_data = MessageValidator.sanitize_hand_data(hand_data)
                
                # Pack and broadcast to all known clients
                message = HandTrackingProtocol.pack_hand_data(hand_data)
                if message:
                    self._broadcast_message(message)
                
                # Check for incoming messages (calibration requests, etc.)
                self._handle_incoming_messages()
                
                # Send calibration complete signal to new clients
                self._send_calibration_status_to_new_clients()
                
                # Maintain frame rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Server loop error: {e}")
        finally:
            self.stop()
    
    def _send_calibration_status_to_new_clients(self):
        """Send calibration complete status to clients that just connected."""
        if self.tracker and self.tracker.calibrated:
            calibration_msg = HandTrackingProtocol.pack_calibration_complete()
            if calibration_msg:
                # Send to all known clients - they can ignore if they already received it
                self._broadcast_message(calibration_msg)
    
    def _broadcast_message(self, message: bytes):
        """Broadcast message to all known clients."""
        for addr in self.client_addresses.copy():
            try:
                self.socket.sendto(message, addr)
            except Exception as e:
                logger.warning(f"Failed to send to client {addr}: {e}")
                self.client_addresses.discard(addr)
    
    def _handle_incoming_messages(self):
        """Handle incoming messages from clients."""
        try:
            while True:
                try:
                    data, addr = self.socket.recvfrom(1024)
                    is_new_client = addr not in self.client_addresses
                    self.client_addresses.add(addr)
                    
                    # Send calibration status to new clients immediately
                    if is_new_client and self.tracker and self.tracker.calibrated:
                        calibration_msg = HandTrackingProtocol.pack_calibration_complete()
                        if calibration_msg:
                            try:
                                self.socket.sendto(calibration_msg, addr)
                                logger.info(f"Sent calibration complete status to new client {addr}")
                            except Exception as e:
                                logger.warning(f"Failed to send calibration status to {addr}: {e}")
                    
                    msg_type = HandTrackingProtocol.get_message_type(data)
                    if msg_type == MSG_TYPE_CALIBRATION:
                        logger.info("Received calibration request")
                        # TODO: Implement runtime calibration
                    elif msg_type == MSG_TYPE_SHUTDOWN:
                        logger.info("Received shutdown request")
                        self.running = False
                        break
                        
                except socket.timeout:
                    break  # No more messages
                except Exception as e:
                    logger.warning(f"Error handling message: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in message handling: {e}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        while self.running:
            try:
                heartbeat = HandTrackingProtocol.pack_heartbeat()
                self._broadcast_message(heartbeat)
                time.sleep(HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
    
    def stop(self):
        """Stop the server."""
        logger.info("Stopping hand tracking server")
        self.running = False
        
        if self.tracker:
            self.tracker.cleanup()
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except:
                pass
        
        logger.info("Hand tracking server stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    global server
    if server:
        server.stop()
    sys.exit(0)


# Global server instance for signal handling
server = None


def main():
    """Main entry point for hand tracking process."""
    parser = argparse.ArgumentParser(description="MediaPipe Hand Tracking Process")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--socket", type=str, default=SOCKET_PATH, help="Socket path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--show-window", action="store_true", help="Show live tracking window")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start server
    global server
    server = HandTrackingServer(args.socket)
    
    if not server.start(args.camera, args.show_window):
        logger.error("Failed to start hand tracking server")
        sys.exit(1)
    
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
