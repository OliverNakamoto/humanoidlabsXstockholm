import cv2
import numpy as np
import threading
import time
from typing import Dict, Optional, Callable, Any
from abc import ABC, abstractmethod
from cv_hand_tracker import HandTracker
from calibrate_workspace import WorkspaceCalibrator

class LeRobotTeleopInterface(ABC):
    """
    Abstract interface that mimics LeRobot's teleoperation interface.
    This ensures drop-in compatibility with existing LeRobot pipelines.
    """
    
    @abstractmethod
    def get_action(self) -> np.ndarray:
        """
        Get current action from teleoperation device.
        
        Returns:
            Action array in LeRobot format: [x, y, z, roll, pitch, yaw, gripper]
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if teleoperation device is connected."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

class CVHandTeleop(LeRobotTeleopInterface):
    """
    Computer Vision Hand Tracking Teleoperation for LeRobot.
    Drop-in replacement for hardware leader arms using hand tracking.
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 calibration_file: str = 'workspace_calibration.json',
                 action_format: str = 'lerobot',  # 'lerobot' or 'simple'
                 enable_orientation: bool = False,
                 update_rate: float = 30.0):
        """
        Initialize CV hand teleoperation.
        
        Args:
            camera_id: Camera device ID
            calibration_file: Path to workspace calibration file
            action_format: Output format ('lerobot' or 'simple')
            enable_orientation: Whether to estimate hand orientation (experimental)
            update_rate: Target update rate in Hz
        """
        self.camera_id = camera_id
        self.calibration_file = calibration_file
        self.action_format = action_format
        self.enable_orientation = enable_orientation
        self.update_rate = update_rate
        
        # State variables
        self.current_action = np.zeros(7 if action_format == 'lerobot' else 4)
        self.is_running = False
        self.camera_thread = None
        self.last_update_time = 0
        
        # Initialize components
        self.cap = None
        self.tracker = None
        self.calibration = None
        
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize camera, tracker, and calibration."""
        # Load calibration
        self.calibration = WorkspaceCalibrator.load_calibration(self.calibration_file)
        if self.calibration is None:
            print("Warning: No calibration found. Using default workspace mapping.")
            print(f"Run 'python {self.calibration_file}' to calibrate workspace.")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.update_rate)
        
        # Initialize hand tracker
        self.tracker = HandTracker(
            smoothing_window=5,
            confidence_threshold=0.7
        )
        
        # Apply calibration if available
        if self.calibration:
            self._apply_calibration()
            
    def _apply_calibration(self) -> None:
        """Apply loaded calibration to tracker."""
        if 'robot_workspace_bounds' in self.calibration:
            self.tracker.workspace_bounds = self.calibration['robot_workspace_bounds']
        
        if 'depth_calibration' in self.calibration:
            self.tracker.depth_calibration.update(self.calibration['depth_calibration'])
        
        if 'frame_size' in self.calibration:
            frame_h, frame_w = self.calibration['frame_size']
            self.tracker.calibrate_workspace((frame_w, frame_h))
    
    def start(self) -> None:
        """Start the teleoperation system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        
        print("CV Hand Teleoperation started")
        print(f"Action format: {self.action_format}")
        print(f"Update rate: {self.update_rate} Hz")
    
    def stop(self) -> None:
        """Stop the teleoperation system."""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
    
    def _camera_loop(self) -> None:
        """Main camera processing loop running in separate thread."""
        target_interval = 1.0 / self.update_rate
        
        while self.is_running:
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame with hand tracker
            result = self.tracker.process_frame(frame)
            
            # Convert to LeRobot action format
            self.current_action = self._convert_to_action_format(result)
            self.last_update_time = time.time()
            
            # Maintain update rate
            loop_time = time.time() - loop_start
            if loop_time < target_interval:
                time.sleep(target_interval - loop_time)
    
    def _convert_to_action_format(self, tracker_result: Dict) -> np.ndarray:
        """
        Convert tracker result to specified action format.
        
        Args:
            tracker_result: Output from HandTracker.process_frame()
            
        Returns:
            Action array in specified format
        """
        if not tracker_result['detected']:
            # Return zero action when no hand detected
            return np.zeros(7 if self.action_format == 'lerobot' else 4)
        
        action_data = tracker_result['action']  # [x, y, z, gripper]
        
        if self.action_format == 'simple':
            # Simple format: [x, y, z, gripper]
            return np.array(action_data, dtype=np.float32)
        
        elif self.action_format == 'lerobot':
            # LeRobot format: [x, y, z, roll, pitch, yaw, gripper]
            x, y, z, gripper = action_data
            
            # Map position to robot workspace (convert from [-1,1] to actual meters)
            workspace = self.tracker.workspace_bounds
            x_robot = x * (workspace['x'][1] - workspace['x'][0]) / 2 + (workspace['x'][1] + workspace['x'][0]) / 2
            y_robot = y * (workspace['y'][1] - workspace['y'][0]) / 2 + (workspace['y'][1] + workspace['y'][0]) / 2
            z_robot = z * (workspace['z'][1] - workspace['z'][0]) / 2 + (workspace['z'][1] + workspace['z'][0]) / 2
            
            # Orientation (simplified - could be enhanced with hand pose estimation)
            if self.enable_orientation:
                # Placeholder for orientation estimation
                roll, pitch, yaw = 0.0, 0.0, 0.0  # Could estimate from hand landmarks
            else:
                roll, pitch, yaw = 0.0, 0.0, 0.0
            
            # Convert gripper from [0,1] to [-1,1] range (LeRobot convention)
            gripper_lerobot = gripper * 2 - 1
            
            return np.array([x_robot, y_robot, z_robot, roll, pitch, yaw, gripper_lerobot], dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown action format: {self.action_format}")
    
    def get_action(self) -> np.ndarray:
        """
        Get current action from hand tracking.
        
        Returns:
            Action array in specified format
        """
        if not self.is_running:
            self.start()
        
        return self.current_action.copy()
    
    def is_connected(self) -> bool:
        """Check if camera and tracking are working."""
        if not self.is_running or self.cap is None:
            return False
        
        # Check if we've received recent updates
        current_time = time.time()
        return (current_time - self.last_update_time) < 1.0  # 1 second timeout
    
    def get_status(self) -> Dict:
        """
        Get detailed status information.
        
        Returns:
            Dict with status information
        """
        return {
            'connected': self.is_connected(),
            'running': self.is_running,
            'action_format': self.action_format,
            'last_update': self.last_update_time,
            'current_action': self.current_action.tolist(),
            'calibrated': self.calibration is not None,
            'camera_id': self.camera_id
        }
    
    def close(self) -> None:
        """Clean up resources."""
        self.stop()
        
        if self.cap:
            self.cap.release()
        
        if self.tracker:
            self.tracker.close()

class LeRobotCVAdapter:
    """
    Adapter class that provides seamless integration with existing LeRobot code.
    Allows switching between CV control and hardware leader arms.
    """
    
    def __init__(self, use_cv_control: bool = True, **kwargs):
        """
        Initialize the adapter.
        
        Args:
            use_cv_control: Whether to use CV control or fall back to hardware
            **kwargs: Arguments passed to the teleoperation backend
        """
        self.use_cv_control = use_cv_control
        
        if use_cv_control:
            self.teleop = CVHandTeleop(**kwargs)
        else:
            # Placeholder for hardware leader arm
            # In real integration, this would initialize the hardware leader arm
            print("Hardware leader arm not implemented. Using CV control.")
            self.teleop = CVHandTeleop(**kwargs)
    
    def __getattr__(self, name):
        """Delegate all attribute access to the underlying teleoperation system."""
        return getattr(self.teleop, name)
    
    def switch_to_cv(self, **kwargs) -> None:
        """Switch to CV control."""
        if hasattr(self.teleop, 'close'):
            self.teleop.close()
        
        self.teleop = CVHandTeleop(**kwargs)
        self.use_cv_control = True
        print("Switched to CV hand tracking control")
    
    def switch_to_hardware(self, **kwargs) -> None:
        """Switch to hardware leader arm (placeholder)."""
        print("Hardware leader arm switching not implemented")
        # In real implementation, would initialize hardware here

def create_lerobot_teleop(backend: str = 'cv', **kwargs) -> LeRobotTeleopInterface:
    """
    Factory function to create teleoperation interface.
    
    Args:
        backend: 'cv' for computer vision, 'hardware' for leader arm
        **kwargs: Arguments passed to teleoperation backend
        
    Returns:
        Teleoperation interface
    """
    if backend == 'cv':
        return CVHandTeleop(**kwargs)
    elif backend == 'hardware':
        # Placeholder for hardware backend
        print("Hardware backend not implemented, using CV")
        return CVHandTeleop(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

# Example usage and integration patterns
class LeRobotDataCollector:
    """
    Example data collector showing how to integrate with LeRobot pipelines.
    This demonstrates the drop-in replacement pattern.
    """
    
    def __init__(self, teleop_backend: str = 'cv'):
        """
        Initialize data collector.
        
        Args:
            teleop_backend: Teleoperation backend to use
        """
        # Create teleoperation interface
        self.teleop = create_lerobot_teleop(teleop_backend, action_format='lerobot')
        
        # Storage for collected data
        self.actions = []
        self.timestamps = []
    
    def collect_episode(self, duration: float = 10.0) -> Dict:
        """
        Collect a demonstration episode.
        
        Args:
            duration: Episode duration in seconds
            
        Returns:
            Dict with collected data
        """
        print(f"Collecting episode for {duration} seconds...")
        print("Move your hand to demonstrate the task")
        
        start_time = time.time()
        episode_actions = []
        episode_timestamps = []
        
        # Start teleoperation
        self.teleop.start()
        
        try:
            while time.time() - start_time < duration:
                # Get current action
                action = self.teleop.get_action()
                timestamp = time.time() - start_time
                
                episode_actions.append(action.copy())
                episode_timestamps.append(timestamp)
                
                # Print status
                if len(episode_actions) % 30 == 0:  # Every ~1 second at 30Hz
                    print(f"t={timestamp:.1f}s, action={action[:4]}, connected={self.teleop.is_connected()}")
                
                time.sleep(1/30)  # 30Hz collection rate
                
        except KeyboardInterrupt:
            print("\nEpisode collection interrupted by user")
        
        # Store data
        self.actions.extend(episode_actions)
        self.timestamps.extend(episode_timestamps)
        
        episode_data = {
            'actions': np.array(episode_actions),
            'timestamps': np.array(episode_timestamps),
            'duration': duration,
            'teleop_status': self.teleop.get_status()
        }
        
        print(f"Collected {len(episode_actions)} action samples")
        return episode_data
    
    def save_data(self, filename: str) -> None:
        """Save collected data to file."""
        data = {
            'actions': np.array(self.actions),
            'timestamps': np.array(self.timestamps)
        }
        np.savez(filename, **data)
        print(f"Data saved to {filename}")
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.teleop, 'close'):
            self.teleop.close()

# Example LeRobot-style integration
def main():
    """Example usage of the CV teleoperation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LeRobot CV Teleoperation')
    parser.add_argument('--backend', choices=['cv', 'hardware'], default='cv',
                       help='Teleoperation backend')
    parser.add_argument('--format', choices=['simple', 'lerobot'], default='lerobot',
                       help='Action format')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--collect', action='store_true', help='Collect demonstration episode')
    parser.add_argument('--duration', type=float, default=10.0, help='Collection duration')
    args = parser.parse_args()
    
    if args.collect:
        # Demonstration data collection
        collector = LeRobotDataCollector(teleop_backend=args.backend)
        
        try:
            episode_data = collector.collect_episode(duration=args.duration)
            collector.save_data('demonstration_episode.npz')
        finally:
            collector.close()
    
    else:
        # Real-time teleoperation
        teleop = create_lerobot_teleop(
            backend=args.backend,
            camera_id=args.camera,
            action_format=args.format
        )
        
        teleop.start()
        
        try:
            print("Real-time teleoperation active")
            print("Press Ctrl+C to quit")
            
            while True:
                action = teleop.get_action()
                status = teleop.get_status()
                
                print(f"Action: {action[:4]}, Connected: {status['connected']}")
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            teleop.close()

if __name__ == "__main__":
    main()