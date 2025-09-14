# #!/usr/bin/env python

# import logging
# import time
# from dataclasses import dataclass
# from pathlib import Path

# import cv2
# import draccus
# import numpy as np

# # LeRobot imports
# from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
# from lerobot.robots import RobotConfig, make_robot_from_config
# from lerobot.robots import so101_follower  # noqa: F401
# from lerobot.utils.utils import move_cursor_up

# # Optional imports for YOLO
# try:
#     from ultralytics import YOLO
#     YOLO_AVAILABLE = True
# except ImportError:
#     YOLO_AVAILABLE = False
#     print("Warning: YOLO not installed. Using basic motion detection instead.")


# @dataclass
# class FlyFeederConfig:
#     """Configuration for the autonomous fly feeder."""
#     robot: RobotConfig
#     camera: OpenCVCameraConfig
#     # Position of the gripper when waiting for a fly (not squeezing)
#     gripper_open_pos_deg: float = 25.0
#     # Position of the gripper when squeezing
#     gripper_squeeze_pos_deg: float = 20.0
#     # How long to squeeze for, in seconds
#     squeeze_duration_s: float = 3.0
#     # How long to wait after a squeeze before detecting again
#     cooldown_duration_s: float = 2.0
#     # Optional: path to a YOLO model
#     yolo_model_path: str = "yolov8n.pt"
#     headless: bool = False


# class FlyFeeder:
#     def __init__(self, cfg: FlyFeederConfig):
#         self.cfg = cfg
#         self.headless = cfg.headless
        
#         print("Initializing robot...")
#         self.robot = make_robot_from_config(cfg.robot)

#         print("Initializing camera...")
#         self.camera = OpenCVCamera(cfg.camera)

#         self.detector = None
#         if YOLO_AVAILABLE:
#             model_path = Path(cfg.yolo_model_path)
#             if model_path.exists():
#                 self.detector = YOLO(model_path)
#                 print(f"Loaded YOLO model from {model_path}")
#             else:
#                 print(f"YOLO model '{model_path}' not found. Using motion detection.")
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

#         # --- State Machine Variables ---
#         # The robot can be in one of these states: "IDLE", "SQUEEZING", "COOLDOWN"
#         self.state = "IDLE"
#         self.state_change_time = 0

#     def connect(self):
#         print("Connecting to robot motors...")
#         self.robot.connect()
#         print("Connecting to camera...")
#         self.camera.connect()
#         print("All devices connected!")
#         # Start the robot in the open position
#         self.move_gripper_to(self.cfg.gripper_open_pos_deg)

#     def disconnect(self):
#         if hasattr(self.robot, "bus") and self.robot.bus:
#             num_motors = len(self.robot.bus.motors)
#             print("\n" * (num_motors + 4))

#         if self.robot.is_connected:
#             self.move_gripper_to(self.cfg.gripper_open_pos_deg)
#             time.sleep(0.5)
#             self.robot.disconnect()
#             print("Robot motors disconnected.")
#         if self.camera.is_connected:
#             self.camera.disconnect()
#             print("Camera disconnected.")
            
#     def move_gripper_to(self, target_degrees: float):
#         """Sends an action to move the gripper, keeping other joints stationary."""
#         current_state = self.robot.get_observation()
#         action = {k: v for k, v in current_state.items() if k.endswith('.pos')}

#         if not action:
#             return

#         action["gripper.pos"] = target_degrees
#         self.robot.send_action(action)

#     def detect_fly(self, frame):
#         """Detects a fly and draws a bounding box."""
#         detected = False
#         if self.detector:
#             results = self.detector(frame, conf=0.9, verbose=False)
#             for r in results:
#                 if len(r.boxes) > 0:
#                     detected = True
#                     for box in r.boxes:
#                         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#         else: # Fallback motion detection
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             fgmask = self.bg_subtractor.apply(gray)
#             contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for contour in contours:
#                 if cv2.contourArea(contour) > 100:
#                     detected = True
#                     x, y, w, h = cv2.boundingRect(contour)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         return detected, frame

#     def run(self):
#         """Main loop for fly detection and gripper control using a state machine."""
#         print("Starting autonomous control loop. Press Ctrl+C to exit.")
#         if not self.headless:
#             cv2.namedWindow("Fly Feeder", cv2.WINDOW_NORMAL)

#         num_motors = len(self.robot.bus.motors)
#         print("\n" * (num_motors + 4))

#         while True:
#             try:
#                 # --- Always print live motor positions ---
#                 robot_obs = self.robot.get_observation()
#                 motor_positions = {k: v for k, v in robot_obs.items() if k.endswith(".pos")}
#                 print("--- LIVE MOTOR POSITIONS ---")
#                 print(f"{'MOTOR':<20} | {'POSITION':>7}")
#                 print("-" * 32)
#                 for name, pos in motor_positions.items():
#                     motor_name = name.removesuffix(".pos")
#                     print(f"{motor_name:<20} | {pos:>7.2f}°")
#                 print("-" * 32)
                
#                 # --- Get camera image and detect fly ---
#                 frame = self.camera.read()
#                 fly_detected, annotated_frame = self.detect_fly(frame.copy())
                
#                 # --- State Machine Logic ---
#                 current_time = time.time()
                
#                 if self.state == "IDLE":
#                     if fly_detected:
#                         print("Fly detected! Squeezing...")
#                         self.move_gripper_to(self.cfg.gripper_squeeze_pos_deg)
#                         self.state = "SQUEEZING"
#                         self.state_change_time = current_time

#                 elif self.state == "SQUEEZING":
#                     if current_time - self.state_change_time > self.cfg.squeeze_duration_s:
#                         print("Squeeze complete. Releasing and starting cooldown.")
#                         self.move_gripper_to(self.cfg.gripper_open_pos_deg)
#                         self.state = "COOLDOWN"
#                         self.state_change_time = current_time

#                 elif self.state == "COOLDOWN":
#                     if current_time - self.state_change_time > self.cfg.cooldown_duration_s:
#                         print("Cooldown finished. Ready to detect.")
#                         self.state = "IDLE"

#                 # --- Update Display ---
#                 move_cursor_up(len(motor_positions) + 5)
                
#                 if not self.headless:
#                     color = (0, 255, 0)
#                     if self.state == "SQUEEZING":
#                         color = (0, 0, 255)
#                     elif self.state == "COOLDOWN":
#                         color = (0, 255, 255) # Yellow for cooldown
                        
#                     cv2.putText(annotated_frame, f"Status: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                     cv2.imshow("Fly Feeder", annotated_frame)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break

#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 logging.error(f"Error in main loop: {e}", exc_info=True)
#                 break
        
#         if not self.headless:
#             cv2.destroyAllWindows()

# @draccus.wrap()
# def main(cfg: FlyFeederConfig):
#     feeder = FlyFeeder(cfg)
#     try:
#         feeder.connect()
#         feeder.run()
#     finally:
#         feeder.disconnect()
#     print("Shutdown complete.")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python

import logging
import time
from dataclasses import dataclass

import cv2
import draccus

# LeRobot imports
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots import so101_follower  # noqa: F401
from lerobot.utils.utils import move_cursor_up

# Import our new VLM class
from vlm import FlyVLM


@dataclass
class FlyFeederConfig:
    """Configuration for the autonomous fly feeder."""
    robot: RobotConfig
    camera: OpenCVCameraConfig
    gripper_open_pos_deg: float = 25.0
    gripper_squeeze_pos_deg: float = 15.0
    squeeze_duration_s: float = 3.0
    cooldown_duration_s: float = 2.0
    headless: bool = False


class FlyFeeder:
    def __init__(self, cfg: FlyFeederConfig):
        self.cfg = cfg
        self.headless = cfg.headless
        
        print("Initializing robot...")
        self.robot = make_robot_from_config(cfg.robot)

        print("Initializing camera...")
        self.camera = OpenCVCamera(cfg.camera)
        
        # Initialize our VLM detector
        self.vlm_detector = FlyVLM()

        self.state = "IDLE"
        self.state_change_time = 0

    def connect(self):
        print("Connecting to robot motors...")
        self.robot.connect()
        print("Connecting to camera...")
        self.camera.connect()
        print("All devices connected!")
        self.move_gripper_to(self.cfg.gripper_open_pos_deg)

    def disconnect(self):
        if hasattr(self.robot, "bus") and self.robot.bus:
            num_motors = len(self.robot.bus.motors)
            print("\n" * (num_motors + 4))

        if self.robot.is_connected:
            self.move_gripper_to(self.cfg.gripper_open_pos_deg)
            time.sleep(0.5)
            self.robot.disconnect()
            print("Robot motors disconnected.")
        if self.camera.is_connected:
            self.camera.disconnect()
            print("Camera disconnected.")
            
    def move_gripper_to(self, target_degrees: float):
        current_state = self.robot.get_observation()
        action = {k: v for k, v in current_state.items() if k.endswith('.pos')}
        if not action: return
        action["gripper.pos"] = target_degrees
        self.robot.send_action(action)

    def run(self):
        print("Starting autonomous control loop. Press Ctrl+C to exit.")
        if not self.headless: cv2.namedWindow("Fly Feeder", cv2.WINDOW_NORMAL)

        num_motors = len(self.robot.bus.motors)
        print("\n" * (num_motors + 4))

        while True:
            try:
                robot_obs = self.robot.get_observation()
                motor_positions = {k: v for k, v in robot_obs.items() if k.endswith(".pos")}
                
                # --- Print live motor positions ---
                print("--- LIVE MOTOR POSITIONS ---")
                print(f"{'MOTOR':<20} | {'POSITION':>7}")
                print("-" * 32)
                for name, pos in motor_positions.items():
                    motor_name = name.removesuffix(".pos")
                    print(f"{motor_name:<20} | {pos:>7.2f}°")
                print("-" * 32)
                
                # Get camera image and use the VLM to check for a fly
                frame = self.camera.read()
                fly_detected = self.vlm_detector.check_for_fly(frame)
                
                # --- State Machine Logic ---
                current_time = time.time()
                
                if self.state == "IDLE":
                    if fly_detected:
                        print("Fly detected by VLM! Squeezing...")
                        self.move_gripper_to(self.cfg.gripper_squeeze_pos_deg)
                        self.state = "SQUEEZING"
                        self.state_change_time = current_time

                elif self.state == "SQUEEZING":
                    if current_time - self.state_change_time > self.cfg.squeeze_duration_s:
                        print("Squeeze complete. Releasing and starting cooldown.")
                        self.move_gripper_to(self.cfg.gripper_open_pos_deg)
                        self.state = "COOLDOWN"
                        self.state_change_time = current_time

                elif self.state == "COOLDOWN":
                    if current_time - self.state_change_time > self.cfg.cooldown_duration_s:
                        print("Cooldown finished. Ready to detect.")
                        self.state = "IDLE"

                # --- Update Display ---
                move_cursor_up(len(motor_positions) + 5)
                
                if not self.headless:
                    color = (0, 255, 0)
                    if self.state == "SQUEEZING": color = (0, 0, 255)
                    elif self.state == "COOLDOWN": color = (0, 255, 255)
                        
                    cv2.putText(frame, f"Status: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if fly_detected:
                         cv2.putText(frame, "VLM: YES", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Fly Feeder", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

            except KeyboardInterrupt: break
            except Exception as e:
                logging.error(f"Error in main loop: {e}", exc_info=True)
                break
        
        if not self.headless: cv2.destroyAllWindows()

@draccus.wrap()
def main(cfg: FlyFeederConfig):
    feeder = FlyFeeder(cfg)
    try:
        feeder.connect()
        feeder.run()
    finally:
        feeder.disconnect()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()