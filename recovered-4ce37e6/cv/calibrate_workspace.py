import cv2
import numpy as np
import json
import time
from typing import Dict, List, Tuple
from cv_hand_tracker import HandTracker

class WorkspaceCalibrator:
    """
    Interactive calibration tool for mapping camera space to robot workspace.
    Guides user through calibration process to establish coordinate mapping.
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.calibration_points = []
        self.robot_workspace_bounds = {
            'x': (-0.5, 0.5),  # meters
            'y': (-0.5, 0.5),  # meters
            'z': (0.0, 0.8)    # meters
        }
        
    def start_calibration(self) -> Dict:
        """
        Interactive calibration process.
        
        Returns:
            Dict containing calibration parameters
        """
        print("Starting workspace calibration...")
        print("This will help map your camera view to the robot's workspace.")
        print("\nInstructions:")
        print("1. Position your hand at each corner of the robot's reachable workspace")
        print("2. Press SPACE when your hand is in the correct position")
        print("3. Press 'q' to quit at any time")
        print("4. We'll calibrate 4 corner points + center position")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize hand tracker for visualization
        tracker = HandTracker()
        
        calibration_sequence = [
            ("Top-Left corner", (-0.5, 0.5, 0.4)),
            ("Top-Right corner", (0.5, 0.5, 0.4)),
            ("Bottom-Right corner", (0.5, -0.5, 0.4)),
            ("Bottom-Left corner", (-0.5, -0.5, 0.4)),
            ("Center position", (0.0, 0.0, 0.4))
        ]
        
        calibration_data = {
            'camera_points': [],
            'robot_points': [],
            'frame_size': None
        }
        
        for i, (position_name, robot_coords) in enumerate(calibration_sequence):
            print(f"\n[{i+1}/{len(calibration_sequence)}] Move hand to: {position_name}")
            print(f"Target robot coordinates: {robot_coords}")
            print("Press SPACE when ready, 'q' to quit")
            
            point_captured = False
            while not point_captured:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Process frame with hand tracker
                result = tracker.process_frame(frame)
                vis_frame = result['visualization']
                
                # Add calibration instructions
                cv2.putText(vis_frame, f"Calibrating: {position_name}", 
                           (10, vis_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis_frame, f"Target: {robot_coords}", 
                           (10, vis_frame.shape[0] - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis_frame, "Press SPACE when ready", 
                           (10, vis_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show current detection status
                if result['detected']:
                    cv2.putText(vis_frame, "Hand Detected - Ready to Capture", 
                               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_frame, "No Hand Detected", 
                               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow('Workspace Calibration', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and result['detected']:
                    # Capture current hand position
                    camera_point = self._extract_camera_coordinates(result, frame.shape[:2])
                    calibration_data['camera_points'].append(camera_point)
                    calibration_data['robot_points'].append(robot_coords)
                    
                    if calibration_data['frame_size'] is None:
                        calibration_data['frame_size'] = frame.shape[:2]
                    
                    print(f"✓ Captured point: camera{camera_point} -> robot{robot_coords}")
                    point_captured = True
                    
                elif key == ord('q'):
                    print("Calibration cancelled.")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return None
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Calculate transformation matrix
        calibration_params = self._calculate_transformation(calibration_data)
        
        # Save calibration
        self._save_calibration(calibration_params)
        
        print("\n✓ Calibration completed successfully!")
        print(f"Calibration saved to: workspace_calibration.json")
        
        return calibration_params
    
    def _extract_camera_coordinates(self, tracker_result: Dict, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Extract camera coordinates from tracker result.
        
        Args:
            tracker_result: Output from HandTracker.process_frame()
            frame_shape: (height, width) of camera frame
            
        Returns:
            (x, y) coordinates in camera pixel space
        """
        # For now, use the normalized coordinates from the tracker
        # In a more sophisticated system, we'd extract the actual pixel coordinates
        action = tracker_result['action']
        
        # Convert normalized coordinates back to pixel coordinates
        # This is a simplified approach - in practice you'd want the raw pixel data
        frame_h, frame_w = frame_shape
        
        # Estimate pixel coordinates from normalized values
        # Note: This is approximate - ideally we'd modify HandTracker to return raw pixels
        x_pixel = int((action[0] + 1) * frame_w / 2)
        y_pixel = int((-action[1] + 1) * frame_h / 2)  # Flip Y back
        
        return (x_pixel, y_pixel)
    
    def _calculate_transformation(self, calibration_data: Dict) -> Dict:
        """
        Calculate transformation matrix from camera to robot coordinates.
        
        Args:
            calibration_data: Dict with camera_points, robot_points, frame_size
            
        Returns:
            Dict with transformation parameters
        """
        camera_points = np.array(calibration_data['camera_points'], dtype=np.float32)
        robot_points = np.array(calibration_data['robot_points'], dtype=np.float32)
        
        # Calculate affine transformation for X-Y plane
        # Use first 3 points for affine transform (need minimum 3 points)
        camera_xy = camera_points[:3, :2]  # First 3 points, X-Y coordinates
        robot_xy = robot_points[:3, :2]    # First 3 points, X-Y coordinates
        
        # Calculate affine transformation matrix
        transform_matrix = cv2.getAffineTransform(camera_xy, robot_xy)
        
        # Calculate depth calibration parameters
        # Use all points to establish size-to-depth relationship
        depth_params = self._calculate_depth_calibration(calibration_data)
        
        calibration_params = {
            'transform_matrix': transform_matrix.tolist(),
            'depth_calibration': depth_params,
            'frame_size': calibration_data['frame_size'],
            'robot_workspace_bounds': self.robot_workspace_bounds,
            'calibration_timestamp': time.time()
        }
        
        return calibration_params
    
    def _calculate_depth_calibration(self, calibration_data: Dict) -> Dict:
        """
        Calculate depth estimation parameters based on hand size variations.
        
        Args:
            calibration_data: Calibration data with camera and robot points
            
        Returns:
            Dict with depth calibration parameters
        """
        # For now, use default depth calibration
        # In a more sophisticated system, you'd analyze hand size variations
        # during calibration to establish size-to-depth relationship
        
        return {
            'reference_hand_size': 0.15,
            'size_to_depth_factor': 0.3,
            'depth_offset': 0.0
        }
    
    def _save_calibration(self, calibration_params: Dict) -> None:
        """
        Save calibration parameters to file.
        
        Args:
            calibration_params: Calibration parameters to save
        """
        with open('workspace_calibration.json', 'w') as f:
            json.dump(calibration_params, f, indent=2)
    
    @staticmethod
    def load_calibration(filename: str = 'workspace_calibration.json') -> Dict:
        """
        Load calibration parameters from file.
        
        Args:
            filename: Path to calibration file
            
        Returns:
            Dict with calibration parameters
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Calibration file {filename} not found.")
            return None
    
    def test_calibration(self, calibration_params: Dict = None) -> None:
        """
        Test the calibration by showing real-time coordinate mapping.
        
        Args:
            calibration_params: Optional calibration parameters, loads from file if None
        """
        if calibration_params is None:
            calibration_params = self.load_calibration()
            if calibration_params is None:
                print("No calibration found. Run calibration first.")
                return
        
        print("Testing calibration...")
        print("Move your hand around to see coordinate mapping.")
        print("Press 'q' to quit.")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        # Initialize tracker with calibrated parameters
        tracker = HandTracker()
        
        # Apply calibration to tracker
        self._apply_calibration_to_tracker(tracker, calibration_params)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            result = tracker.process_frame(frame)
            vis_frame = result['visualization']
            
            # Add calibration test info
            cv2.putText(vis_frame, "Calibration Test Mode", 
                       (10, vis_frame.shape[0] - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(vis_frame, "Press 'q' to quit", 
                       (10, vis_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if result['detected']:
                action = result['action']
                cv2.putText(vis_frame, f"Robot Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Calibration Test', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _apply_calibration_to_tracker(self, tracker: HandTracker, calibration_params: Dict) -> None:
        """
        Apply calibration parameters to hand tracker.
        
        Args:
            tracker: HandTracker instance
            calibration_params: Calibration parameters
        """
        # Update workspace bounds
        if 'robot_workspace_bounds' in calibration_params:
            tracker.workspace_bounds = calibration_params['robot_workspace_bounds']
        
        # Update depth calibration
        if 'depth_calibration' in calibration_params:
            tracker.depth_calibration.update(calibration_params['depth_calibration'])
        
        # Set frame size for camera bounds calculation
        if 'frame_size' in calibration_params:
            frame_h, frame_w = calibration_params['frame_size']
            tracker.calibrate_workspace((frame_w, frame_h))

def main():
    """Main calibration script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Workspace Calibration Tool')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--test', action='store_true', help='Test existing calibration')
    args = parser.parse_args()
    
    calibrator = WorkspaceCalibrator(camera_id=args.camera)
    
    if args.test:
        calibrator.test_calibration()
    else:
        calibrator.start_calibration()

if __name__ == "__main__":
    main()