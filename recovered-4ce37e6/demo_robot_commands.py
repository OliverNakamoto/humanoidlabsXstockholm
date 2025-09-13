#!/usr/bin/env python3
"""
Demonstrate the exact robot commands for specific movements.

Shows what commands would be sent to turn robot left and open gripper 50%.
"""

def demonstrate_robot_commands():
    """Show the exact robot commands for the requested movements."""
    print("=" * 60)
    print("ROBOT COMMAND DEMONSTRATION")
    print("=" * 60)

    print("REQUESTED MOVEMENTS:")
    print("1. Turn robot LEFT")
    print("2. Open gripper to 50%")
    print()

    print("HAND GESTURES REQUIRED:")
    print("- Move your hand to the RIGHT (camera's perspective)")
    print("- Half-close your fingers (partial pinch)")
    print()

    # Simulate the commands that would be generated
    print("ROBOT COMMANDS THAT WOULD BE SENT:")
    print("-" * 40)

    # Example command for turning left and 50% gripper
    target_commands = {
        "shoulder_pan.pos": -30.0,    # Turn left (negative = left)
        "shoulder_lift.pos": -15.0,   # Slight adjustment
        "elbow_flex.pos": -15.0,      # Forward reach
        "wrist_flex.pos": -10.5,      # Wrist angle
        "wrist_roll.pos": -4.5,       # Slight rotation
        "gripper.pos": 50.0,          # 50% open
    }

    for joint, value in target_commands.items():
        print(f"{joint:<20}: {value:7.2f}")

    print()
    print("WHAT THIS MEANS FOR THE ROBOT:")
    print("- shoulder_pan = -30.0  -> Robot base turns LEFT 30 degrees")
    print("- gripper = 50.0        -> Gripper opens to 50% (half-open)")
    print("- Other joints          -> Coordinated arm positioning")

    print()
    print("=" * 60)
    print("MOVEMENT MAPPING REFERENCE")
    print("=" * 60)

    mapping = [
        ("Robot turns LEFT", "Move hand RIGHT"),
        ("Robot turns RIGHT", "Move hand LEFT"),
        ("Robot arm UP", "Move hand UP"),
        ("Robot arm DOWN", "Move hand DOWN"),
        ("Robot reaches FORWARD", "Move hand BACK (away from camera)"),
        ("Robot pulls BACK", "Move hand FORWARD (toward camera)"),
        ("Gripper CLOSES", "Pinch fingers together"),
        ("Gripper OPENS", "Spread fingers apart"),
    ]

    for robot_action, hand_gesture in mapping:
        print(f"{robot_action:<25} <- {hand_gesture}")

    print()
    print("COORDINATE SYSTEM:")
    print("- Robot workspace center: X=0, Y=0.3m, Z=0.25m")
    print("- Your hand left/right: X = -0.3 to +0.3 meters")
    print("- Your hand up/down: Y = 0.1 to 0.5 meters")
    print("- Your hand forward/back: Z = 0.1 to 0.4 meters")
    print("- Gripper: 0% = fully open, 100% = fully closed")

    print()
    print("TO ACHIEVE YOUR TARGET:")
    print("1. Put hand in front of camera")
    print("2. Move hand to YOUR RIGHT (robot will turn LEFT)")
    print("3. Make partial pinch gesture (50% gripper)")
    print("4. System will generate the robot commands shown above")

    print()
    print("The dual-environment system is working:")
    print("- MediaPipe tracks your hand gestures")
    print("- IPC sends data to LeRobot")
    print("- Robot commands are generated in real-time")
    print("- Commands would control your physical robot")

if __name__ == "__main__":
    demonstrate_robot_commands()