# #!/usr/bin/env python

# import logging
# import time
# from dataclasses import dataclass

# import draccus

# # LeRobot imports for creating a robot from configuration
# from lerobot.robots import RobotConfig, make_robot_from_config
# # Import to register the robot type you want to use
# from lerobot.robots import so101_follower  # noqa: F401
# # Import the utility for a clean terminal display
# from lerobot.utils.utils import move_cursor_up


# @dataclass
# class MonitorConfig:
#     """A simple configuration that only contains the robot settings."""
#     robot: RobotConfig


# class RobotMonitor:
#     def __init__(self, cfg: MonitorConfig):
#         """Initializes the monitor with just a robot object."""
#         print("Initializing robot...")
#         # This robot object is created without any camera configuration
#         self.robot = make_robot_from_config(cfg.robot)

#     def connect(self):
#         """Connects to the robot hardware."""
#         print("Connecting to robot motors...")
#         # When prompted, just press Enter to use existing calibration, or 'c' to re-calibrate.
#         self.robot.connect()
#         print("Robot connected!")

#     def disconnect(self):
#         """Disconnects from the robot and cleans up the console."""
#         if hasattr(self.robot, "bus") and self.robot.bus:
#             num_motors = len(self.robot.bus.motors)
#             print("\n" * (num_motors + 4)) # Move cursor below the live display

#         if self.robot.is_connected:
#             self.robot.disconnect()
#             print("Robot disconnected.")
            
#     def run(self):
#         """Main loop to read and print motor states."""
#         print("Starting real-time monitoring. Press Ctrl+C to exit.")
        
#         # Prepare the console for the live display
#         num_motors = len(self.robot.bus.motors)
#         print("\n" * (num_motors + 4))

#         while True:
#             try:
#                 # 1. Get observation from the robot to read motor positions
#                 robot_obs = self.robot.get_observation()
                
#                 # 2. Filter for motor positions (keys ending in ".pos")
#                 motor_positions = {k: v for k, v in robot_obs.items() if k.endswith(".pos")}

#                 # 3. Print a formatted, updating table to the console
#                 print("--- LIVE MOTOR POSITIONS ---")
#                 print(f"{'MOTOR':<20} | {'POSITION':>7}")
#                 print("-" * 32)
#                 for name, pos in motor_positions.items():
#                     motor_name = name.removesuffix(".pos")
#                     print(f"{motor_name:<20} | {pos:>7.2f}°")
#                 print("-" * 32)
                
#                 # Move the cursor up so the next print overwrites this one
#                 move_cursor_up(len(motor_positions) + 4)

#                 # Wait a moment to control the refresh rate
#                 time.sleep(0.1) # Refresh rate of 10 Hz

#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 logging.error(f"Error in main loop: {e}", exc_info=True)
#                 break
        

# @draccus.wrap()
# def main(cfg: MonitorConfig):
#     """Initializes and runs the RobotMonitor."""
#     monitor = RobotMonitor(cfg)
#     try:
#         monitor.connect()
#         monitor.run()
#     finally:
#         monitor.disconnect()
#     print("Monitoring stopped.")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python

import logging
import time
from dataclasses import dataclass

import draccus

# LeRobot imports
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots import so101_follower  # noqa: F401
from lerobot.utils.utils import move_cursor_up


@dataclass
class GripperTestConfig:
    """Configuration for the gripper test script."""
    robot: RobotConfig
    # The two target positions for the gripper
    gripper_pos_1: float = 15.0
    gripper_pos_2: float = 25.0
    # Time in seconds to wait before moving the gripper again
    move_interval_s: float = 2.0


class GripperTester:
    def __init__(self, cfg: GripperTestConfig):
        """Initializes the tester with a robot object and gripper targets."""
        self.cfg = cfg
        
        print("Initializing robot...")
        self.robot = make_robot_from_config(cfg.robot)

        # State variables to track the gripper's target
        self.last_move_time = 0
        self.next_target_is_pos_2 = True

    def connect(self):
        """Connects to the robot hardware."""
        print("Connecting to robot motors...")
        self.robot.connect()
        print("All devices connected!")

    def disconnect(self):
        """Disconnects from the robot and cleans up the console."""
        if hasattr(self.robot, "bus") and self.robot.bus:
            num_motors = len(self.robot.bus.motors)
            print("\n" * (num_motors + 4))

        if self.robot.is_connected:
            self.robot.disconnect()
            print("Robot disconnected.")
            
    def move_gripper_to(self, target_degrees: float):
        """
        Sends an action to move the gripper to a specific angle, keeping other joints stationary.
        """
        # 1. Get the current state of all motors to use as a template.
        current_state = self.robot.get_observation()
        action = {k: v for k, v in current_state.items() if k.endswith('.pos')}

        if not action:
            print("Warning: Could not get motor positions to create an action.")
            return

        # 2. Set the desired gripper position in the action dictionary.
        action["gripper.pos"] = target_degrees

        # 3. Send the complete action. Other joints are commanded to their current
        #    position, so they won't move.
        print(f"\nSending command: Move gripper to {target_degrees:.2f}°")
        self.robot.send_action(action)

    def run(self):
        """Main loop to print states and move the gripper."""
        print("Starting gripper test. Press Ctrl+C to exit.")
        
        num_motors = len(self.robot.bus.motors)
        print("\n" * (num_motors + 4))

        while True:
            try:
                # --- Block for printing live motor positions ---
                robot_obs = self.robot.get_observation()
                motor_positions = {k: v for k, v in robot_obs.items() if k.endswith(".pos")}
                print("--- LIVE MOTOR POSITIONS ---")
                print(f"{'MOTOR':<20} | {'POSITION':>7}")
                print("-" * 32)
                for name, pos in motor_positions.items():
                    motor_name = name.removesuffix(".pos")
                    print(f"{motor_name:<20} | {pos:>7.2f}°")
                print("-" * 32)
                
                # --- Block for moving the gripper ---
                current_time = time.time()
                if current_time - self.last_move_time > self.cfg.move_interval_s:
                    if self.next_target_is_pos_2:
                        target = self.cfg.gripper_pos_2
                    else:
                        target = self.cfg.gripper_pos_1
                    
                    self.move_gripper_to(target)
                    
                    # Update state for the next move
                    self.next_target_is_pos_2 = not self.next_target_is_pos_2
                    self.last_move_time = current_time

                move_cursor_up(len(motor_positions) + 5) # +5 for header and extra print line

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}", exc_info=True)
                break


@draccus.wrap()
def main(cfg: GripperTestConfig):
    """Initializes and runs the GripperTester."""
    tester = GripperTester(cfg)
    try:
        tester.connect()
        tester.run()
    finally:
        tester.disconnect()
    print("Test stopped.")


if __name__ == "__main__":
    main()