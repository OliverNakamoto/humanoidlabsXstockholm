#!/usr/bin/env python3
"""
Computer Vision Robot Control Demo
==================================

Comprehensive demonstration and testing script for the CV-based robot arm control system.
This script provides various modes to test and demonstrate the hand tracking capabilities.
"""

import cv2
import numpy as np
import time
import sys
import argparse
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt

from cv_hand_tracker import HandTracker
from calibrate_workspace import WorkspaceCalibrator
from lerobot_integration import CVHandTeleop, LeRobotDataCollector

class CVRobotDemo:
    """Comprehensive demo system for CV robot control."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.tracker = None
        
    def run_basic_tracking_demo(self) -> None:
        """Basic hand tracking demonstration with visualization."""
        print("=== Basic Hand Tracking Demo ===")
        print("This demo shows real-time hand tracking with position and gripper detection.")
        print("Move your hand around to see the tracking in action.")
        print("Press 'q' to quit, 's' to save a screenshot")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        self.tracker = HandTracker()
        screenshot_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame
            result = self.tracker.process_frame(frame)
            vis_frame = result['visualization']
            
            # Add demo info
            cv2.putText(vis_frame, "Basic Hand Tracking Demo", 
                       (10, vis_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(vis_frame, "Press 'q' to quit, 's' for screenshot", 
                       (10, vis_frame.shape[0] - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show detailed info when hand is detected
            if result['detected']:
                action = result['action']
                cv2.putText(vis_frame, f"Normalized Pos: ({action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f})", 
                           (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Gripper Open: {action[3]:.3f}", 
                           (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Confidence: {result['confidence']:.3f}", 
                           (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Basic Hand Tracking Demo', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_filename = f"demo_screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_filename, vis_frame)
                print(f"Screenshot saved: {screenshot_filename}")
                screenshot_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run_calibration_demo(self) -> None:
        """Demonstrate the calibration process."""
        print("=== Workspace Calibration Demo ===")
        print("This will guide you through the calibration process.")
        print("Make sure you have enough space to move your hand around.")
        
        calibrator = WorkspaceCalibrator(camera_id=self.camera_id)
        calibration_params = calibrator.start_calibration()
        
        if calibration_params:
            print("\n=== Testing Calibration ===")
            input("Press Enter to test the calibration...")
            calibrator.test_calibration(calibration_params)
    
    def run_lerobot_demo(self) -> None:
        """Demonstrate LeRobot integration."""
        print("=== LeRobot Integration Demo ===")
        print("This shows how the system integrates with LeRobot pipelines.")
        print("Actions are output in LeRobot format: [x, y, z, roll, pitch, yaw, gripper]")
        
        # Initialize CV teleoperation
        teleop = CVHandTeleop(
            camera_id=self.camera_id,
            action_format='lerobot',
            update_rate=30.0
        )
        
        teleop.start()
        
        print("\nTeleoperation started. Move your hand to generate robot actions.")
        print("Press Ctrl+C to quit")
        
        try:
            while True:
                action = teleop.get_action()
                status = teleop.get_status()
                
                if status['connected']:
                    print(f"Robot Action: [{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}, "
                          f"{action[3]:6.3f}, {action[4]:6.3f}, {action[5]:6.3f}, {action[6]:6.3f}]")
                else:
                    print("No hand detected - waiting for input...")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            teleop.close()
    
    def run_data_collection_demo(self) -> None:
        """Demonstrate data collection for robot learning."""
        print("=== Data Collection Demo ===")
        print("This shows how to collect demonstration data for robot learning.")
        
        duration = float(input("Enter collection duration (seconds, default=10): ") or "10")
        
        collector = LeRobotDataCollector(teleop_backend='cv')
        
        try:
            print(f"\nStarting data collection in 3 seconds...")
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            episode_data = collector.collect_episode(duration=duration)
            
            # Save data
            timestamp = int(time.time())
            filename = f"demo_episode_{timestamp}.npz"
            collector.save_data(filename)
            
            # Show statistics
            actions = episode_data['actions']
            print(f"\n=== Collection Statistics ===")
            print(f"Duration: {episode_data['duration']:.1f} seconds")
            print(f"Samples collected: {len(actions)}")
            print(f"Sample rate: {len(actions)/episode_data['duration']:.1f} Hz")
            print(f"Position range X: [{actions[:,0].min():.3f}, {actions[:,0].max():.3f}]")
            print(f"Position range Y: [{actions[:,1].min():.3f}, {actions[:,1].max():.3f}]")
            print(f"Position range Z: [{actions[:,2].min():.3f}, {actions[:,2].max():.3f}]")
            print(f"Gripper range: [{actions[:,6].min():.3f}, {actions[:,6].max():.3f}]")
            
        finally:
            collector.close()
    
    def run_performance_test(self) -> None:
        """Test system performance and measure latency."""
        print("=== Performance Test ===")
        print("Testing system performance and measuring latency.")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        self.tracker = HandTracker()
        
        # Performance metrics
        frame_times = []
        detection_counts = 0
        total_frames = 0
        test_duration = 10.0  # seconds
        
        print(f"Running performance test for {test_duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < test_duration:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame
            result = self.tracker.process_frame(frame)
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            if result['detected']:
                detection_counts += 1
            
            total_frames += 1
            
            # Show progress
            if total_frames % 30 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {elapsed:.1f}s, FPS: {result['fps']:.1f}")
        
        self.cap.release()
        
        # Calculate statistics
        frame_times = np.array(frame_times)
        avg_fps = 1.0 / np.mean(frame_times)
        detection_rate = detection_counts / total_frames
        
        print(f"\n=== Performance Results ===")
        print(f"Total frames processed: {total_frames}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Frame time - Mean: {np.mean(frame_times)*1000:.1f}ms")
        print(f"Frame time - Std: {np.std(frame_times)*1000:.1f}ms")
        print(f"Frame time - Max: {np.max(frame_times)*1000:.1f}ms")
        print(f"Detection rate: {detection_rate:.1%}")
        print(f"Latency estimate: {np.mean(frame_times)*1000:.1f}ms per frame")
    
    def run_robustness_test(self) -> None:
        """Test system robustness under various conditions."""
        print("=== Robustness Test ===")
        print("Testing system behavior under various conditions.")
        print("Follow the instructions to test different scenarios.")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        self.tracker = HandTracker()
        
        test_scenarios = [
            ("Normal operation", "Move your hand normally", 10),
            ("Fast movements", "Move your hand quickly", 10),
            ("Slow movements", "Move your hand very slowly", 10),
            ("Partial occlusion", "Cover part of your hand", 10),
            ("Edge of frame", "Move hand to edges of camera view", 10),
            ("No hand", "Remove your hand from view", 5),
            ("Multiple hands", "Show both hands (only one will be tracked)", 10)
        ]
        
        results = {}
        
        for scenario_name, instruction, duration in test_scenarios:
            print(f"\n--- {scenario_name} ---")
            print(f"Instructions: {instruction}")
            input("Press Enter when ready...")
            
            detection_count = 0
            total_frames = 0
            confidence_scores = []
            
            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                result = self.tracker.process_frame(frame)
                vis_frame = result['visualization']
                
                # Add scenario info
                cv2.putText(vis_frame, f"Test: {scenario_name}", 
                           (10, vis_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(vis_frame, instruction, 
                           (10, vis_frame.shape[0] - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                remaining = duration - (time.time() - start_time)
                cv2.putText(vis_frame, f"Time remaining: {remaining:.1f}s", 
                           (10, vis_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow('Robustness Test', vis_frame)
                
                if result['detected']:
                    detection_count += 1
                    confidence_scores.append(result['confidence'])
                
                total_frames += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Store results
            results[scenario_name] = {
                'detection_rate': detection_count / total_frames if total_frames > 0 else 0,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'total_frames': total_frames
            }
            
            print(f"Detection rate: {results[scenario_name]['detection_rate']:.1%}")
            print(f"Average confidence: {results[scenario_name]['avg_confidence']:.3f}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        print(f"\n=== Robustness Test Summary ===")
        for scenario, data in results.items():
            print(f"{scenario:20s}: Detection {data['detection_rate']:5.1%}, "
                  f"Confidence {data['avg_confidence']:.3f}")
    
    def run_interactive_demo(self) -> None:
        """Interactive demo mode with multiple visualization options."""
        print("=== Interactive Demo ===")
        print("Interactive mode with multiple visualization options.")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  '1' - Basic visualization")
        print("  '2' - Detailed overlay")
        print("  '3' - Robot coordinate display")
        print("  'c' - Toggle coordinate trail")
        print("  'r' - Reset trail")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        self.tracker = HandTracker()
        
        # Visualization state
        viz_mode = 1
        show_trail = False
        coordinate_trail = []
        max_trail_length = 50
        screenshot_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            result = self.tracker.process_frame(frame)
            vis_frame = result['visualization']
            
            # Add coordinates to trail
            if result['detected'] and show_trail:
                action = result['action']
                # Convert to pixel coordinates for trail
                h, w = vis_frame.shape[:2]
                x_pixel = int((action[0] + 1) * w / 2)
                y_pixel = int((-action[1] + 1) * h / 2)
                coordinate_trail.append((x_pixel, y_pixel))
                
                if len(coordinate_trail) > max_trail_length:
                    coordinate_trail.pop(0)
            
            # Draw trail
            if show_trail and len(coordinate_trail) > 1:
                for i in range(1, len(coordinate_trail)):
                    alpha = i / len(coordinate_trail)
                    color = (int(255 * alpha), int(255 * alpha), 0)
                    cv2.line(vis_frame, coordinate_trail[i-1], coordinate_trail[i], color, 2)
            
            # Visualization modes
            if viz_mode == 2 and result['detected']:
                # Detailed overlay
                action = result['action']
                info_lines = [
                    f"Hand Position: ({action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f})",
                    f"Gripper State: {action[3]:.3f} ({'Open' if action[3] > 0.5 else 'Closed'})",
                    f"Detection FPS: {result['fps']:.1f}",
                    f"Trail Points: {len(coordinate_trail)}" if show_trail else "Trail: Disabled"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(vis_frame, line, (10, 200 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            elif viz_mode == 3 and result['detected']:
                # Robot coordinate display
                action = result['action']
                workspace = self.tracker.workspace_bounds
                
                # Convert to robot coordinates
                x_robot = action[0] * (workspace['x'][1] - workspace['x'][0]) / 2
                y_robot = action[1] * (workspace['y'][1] - workspace['y'][0]) / 2
                z_robot = action[2] * (workspace['z'][1] - workspace['z'][0]) / 2
                
                cv2.putText(vis_frame, f"Robot Coordinates (m):", 
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(vis_frame, f"X: {x_robot:+.3f}", 
                           (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(vis_frame, f"Y: {y_robot:+.3f}", 
                           (10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(vis_frame, f"Z: {z_robot:+.3f}", 
                           (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Control instructions
            cv2.putText(vis_frame, "Controls: q=quit, s=screenshot, 1-3=viz modes, c=trail", 
                       (10, vis_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Interactive Demo', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_filename = f"interactive_demo_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_filename, vis_frame)
                print(f"Screenshot saved: {screenshot_filename}")
                screenshot_count += 1
            elif key == ord('1'):
                viz_mode = 1
                print("Visualization: Basic")
            elif key == ord('2'):
                viz_mode = 2
                print("Visualization: Detailed")
            elif key == ord('3'):
                viz_mode = 3
                print("Visualization: Robot coordinates")
            elif key == ord('c'):
                show_trail = not show_trail
                print(f"Coordinate trail: {'Enabled' if show_trail else 'Disabled'}")
                if not show_trail:
                    coordinate_trail.clear()
            elif key == ord('r'):
                coordinate_trail.clear()
                print("Trail reset")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main demo script entry point."""
    parser = argparse.ArgumentParser(description='CV Robot Control Demo System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    
    # Demo mode selection
    demo_group = parser.add_mutually_exclusive_group()
    demo_group.add_argument('--basic', action='store_true', 
                           help='Run basic hand tracking demo')
    demo_group.add_argument('--calibrate', action='store_true', 
                           help='Run calibration demo')
    demo_group.add_argument('--lerobot', action='store_true', 
                           help='Run LeRobot integration demo')
    demo_group.add_argument('--collect', action='store_true', 
                           help='Run data collection demo')
    demo_group.add_argument('--performance', action='store_true', 
                           help='Run performance test')
    demo_group.add_argument('--robustness', action='store_true', 
                           help='Run robustness test')
    demo_group.add_argument('--interactive', action='store_true', 
                           help='Run interactive demo')
    demo_group.add_argument('--all', action='store_true', 
                           help='Run all demos in sequence')
    
    args = parser.parse_args()
    
    print("Computer Vision Robot Control Demo System")
    print("=" * 50)
    
    demo = CVRobotDemo(camera_id=args.camera)
    
    try:
        if args.basic:
            demo.run_basic_tracking_demo()
        elif args.calibrate:
            demo.run_calibration_demo()
        elif args.lerobot:
            demo.run_lerobot_demo()
        elif args.collect:
            demo.run_data_collection_demo()
        elif args.performance:
            demo.run_performance_test()
        elif args.robustness:
            demo.run_robustness_test()
        elif args.interactive:
            demo.run_interactive_demo()
        elif args.all:
            # Run all demos in sequence
            demos = [
                ("Basic Tracking", demo.run_basic_tracking_demo),
                ("Performance Test", demo.run_performance_test),
                ("LeRobot Integration", demo.run_lerobot_demo),
                ("Interactive Demo", demo.run_interactive_demo)
            ]
            
            for name, demo_func in demos:
                print(f"\n{'='*20} {name} {'='*20}")
                input("Press Enter to continue to next demo...")
                demo_func()
        else:
            # Interactive menu
            while True:
                print("\nAvailable Demos:")
                print("1. Basic hand tracking demo")
                print("2. Workspace calibration")
                print("3. LeRobot integration demo")
                print("4. Data collection demo")
                print("5. Performance test")
                print("6. Robustness test")
                print("7. Interactive demo")
                print("8. Run all demos")
                print("0. Exit")
                
                choice = input("\nSelect demo (0-8): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    demo.run_basic_tracking_demo()
                elif choice == '2':
                    demo.run_calibration_demo()
                elif choice == '3':
                    demo.run_lerobot_demo()
                elif choice == '4':
                    demo.run_data_collection_demo()
                elif choice == '5':
                    demo.run_performance_test()
                elif choice == '6':
                    demo.run_robustness_test()
                elif choice == '7':
                    demo.run_interactive_demo()
                elif choice == '8':
                    # Run selected demos
                    demos_to_run = [
                        demo.run_basic_tracking_demo,
                        demo.run_performance_test,
                        demo.run_lerobot_demo
                    ]
                    for demo_func in demos_to_run:
                        demo_func()
                        input("\nPress Enter for next demo...")
                else:
                    print("Invalid choice. Please select 0-8.")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
    finally:
        print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()