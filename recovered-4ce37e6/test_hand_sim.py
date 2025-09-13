#!/usr/bin/env python

"""
Test script for hand tracking teleoperation with simulated robot and Rerun visualization.
This allows testing the complete pipeline without any physical hardware.
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path
import numpy as np

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after path setup
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from tests.mocks.mock_so101_robot import MockSO101Config
from lerobot.teleoperators.hand_leader import HandLeaderConfig
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
import rerun as rr


def run_simulation_teleoperation(
    camera_index: int = 0,
    urdf_path: str = None,
    fps: int = 30,
    duration: float = None,
    display_debug: bool = True
):
    """
    Run hand tracking teleoperation with simulated robot.
    
    Args:
        camera_index: Camera index for hand tracking
        urdf_path: Path to robot URDF file
        fps: Target frames per second
        duration: Duration in seconds (None for infinite)
        display_debug: Show debug information
    """
    
    print("="*60)
    print("Hand Tracking Teleoperation Simulation")
    print("="*60)
    print("This test allows you to control a simulated robot using hand gestures")
    print("without any physical hardware connected.")
    print("="*60 + "\n")
    
    # Set default URDF path
    if urdf_path is None:
        urdf_path = "lerobot/src/lerobot/robots/so101_follower/so101.urdf"
    
    # Initialize Rerun for visualization
    logger.info("Initializing Rerun visualization...")
    _init_rerun(session_name="hand_tracking_simulation")
    
    # Create mock robot configuration
    robot_config = MockSO101Config(
        type="mock_so101",
        id="simulated_so101",
        calibrated=True,
        smooth_movement=True,
        movement_speed=0.15,  # Smooth movement speed
        add_noise=True,
        noise_level=0.2
    )
    
    # Create hand leader configuration
    teleop_config = HandLeaderConfig(
        type="hand_leader",
        id="cv_hand_tracker",
        camera_index=camera_index,
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    )
    
    # Create robot and teleoperator
    logger.info("Creating simulated robot...")
    robot = make_robot_from_config(robot_config)
    
    logger.info("Creating hand tracking teleoperator...")
    teleop = make_teleoperator_from_config(teleop_config)
    
    # Connect devices
    try:
        logger.info("Connecting devices...")
        robot.connect(calibrate=False)  # Skip calibration for mock robot
        teleop.connect(calibrate=True)  # Run hand tracking calibration
        
        print("\n" + "="*60)
        print("SIMULATION STARTED")
        print("="*60)
        print("Move your hand to control the simulated robot")
        print("Watch the Rerun visualization at http://localhost:9902")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Main control loop
        start_time = time.time()
        loop_count = 0
        
        while True:
            loop_start = time.perf_counter()
            
            # Get current robot position (for IK seed)
            current_pos = robot.get_observation()
            
            # Get action from hand tracking
            action = teleop.get_action(current_pos)
            
            # Send action to robot
            robot.send_action(action)
            
            # Get updated observation
            observation = robot.get_observation()
            
            # Log to Rerun
            log_rerun_data(observation, action)
            
            # Log hand tracking specific data
            if hasattr(teleop, 'tracker') and teleop.tracker:
                x, y, z, pinch = teleop.get_endpos()
                rr.log("hand_tracking/position/x", rr.Scalar(x))
                rr.log("hand_tracking/position/y", rr.Scalar(y))
                rr.log("hand_tracking/position/z", rr.Scalar(z))
                rr.log("hand_tracking/pinch", rr.Scalar(pinch))
                
                # Log 3D position as point
                rr.log("hand_tracking/position_3d", 
                      rr.Points3D([[x, y, z]], 
                                 colors=[[255, 0, 0]],
                                 radii=[0.01]))
            
            # Display debug info
            if display_debug and loop_count % 10 == 0:  # Update every 10 loops
                print("\r" + " "*100, end="")  # Clear line
                print(f"\rLoop: {loop_count:5d} | ", end="")
                print(f"FPS: {1/(time.perf_counter() - loop_start):.1f} | ", end="")
                
                # Show subset of joint positions
                for key in ["shoulder_pan.pos", "elbow_flex.pos", "gripper.pos"]:
                    if key in observation:
                        print(f"{key}: {observation[key]:6.1f} ", end="")
            
            loop_count += 1
            
            # Check duration
            if duration is not None and (time.time() - start_time) > duration:
                break
            
            # Maintain target FPS
            dt = time.perf_counter() - loop_start
            sleep_time = max(0, 1/fps - dt)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        raise
    finally:
        # Disconnect devices
        logger.info("Disconnecting devices...")
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        
        # Shutdown Rerun
        rr.disconnect()
        
        print("\nSimulation complete!")


def main():
    parser = argparse.ArgumentParser(description="Hand tracking teleoperation simulation")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--urdf", type=str, default=None, help="Path to robot URDF")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    
    args = parser.parse_args()
    
    run_simulation_teleoperation(
        camera_index=args.camera,
        urdf_path=args.urdf,
        fps=args.fps,
        duration=args.duration,
        display_debug=args.debug
    )


if __name__ == "__main__":
    main()