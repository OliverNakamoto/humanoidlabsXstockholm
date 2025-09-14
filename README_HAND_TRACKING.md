# Hand Tracking Teleoperation - Testing Guide

This guide explains how to test the hand tracking teleoperation system without any physical robot hardware.

## Overview

The hand tracking system uses MediaPipe to detect your hand movements via webcam and converts them into robot control commands. You can test this in simulation mode without connecting any physical robot.

## Available Test Methods

### Method 1: Standalone Visual Test
The simplest way to test - shows hand tracking with on-screen robot visualization:

```bash
python test_hand_standalone.py
```

Features:
- Real-time hand tracking visualization
- Simulated robot joint positions displayed on screen
- No complex dependencies
- Immediate visual feedback

### Method 2: Full Simulation with Rerun
Complete teleoperation pipeline with Rerun 3D visualization:

```bash
python test_hand_sim.py
```

Features:
- Full teleoperation loop
- Rerun visualization at http://localhost:9902
- Realistic robot simulation with smooth movements
- Hand position logging and visualization

Options:
- `--camera 0` - Camera index (default: 0)
- `--fps 30` - Target frame rate (default: 30)
- `--debug` - Show debug output
- `--duration 60` - Run for specific duration in seconds

### Method 3: Using lerobot-teleoperate Command
Run the full teleoperation system with mock robot:

```bash
./run_sim_teleoperate.sh
```

Or manually:
```bash
cd lerobot
python -m lerobot.teleoperate \
    --robot.type=mock_so101 \
    --robot.id=simulated_follower \
    --teleop.type=hand_leader \
    --teleop.camera_index=0 \
    --display_data=true
```

## Hand Tracking Calibration

When you start any of these tests, you'll go through a calibration process:

1. **Far Position Calibration**
   - Extend your hand FAR from the camera
   - Keep hand flat with fingers spread
   - Hold steady when countdown reaches 0

2. **Near Position Calibration**
   - Move hand CLOSE to camera
   - Keep hand flat with fingers spread
   - Hold steady when countdown reaches 0

## Hand Control Mapping

- **X-axis (left/right)**: Controls shoulder pan and wrist roll
- **Y-axis (up/down)**: Controls shoulder lift and wrist flex
- **Z-axis (forward/back)**: Controls elbow flex (via palm size)
- **Pinch gesture**: Controls gripper (thumb to index finger distance)

## Visualization in Rerun

When using Method 2 or 3 with `--display_data=true`:

1. Open browser to http://localhost:9902
2. You'll see:
   - Real-time joint positions (observation.*)
   - Target joint positions (action.*)
   - Hand tracking data (hand_tracking/*)
   - 3D position of hand target

## Troubleshooting

### Camera not found
- Check camera index (try 0, 1, 2)
- Ensure webcam permissions are granted
- Close other applications using camera

### Hand not detected
- Ensure good lighting
- Keep hand in camera view
- Show full hand (not just fingers)
- Try recalibrating

### Slow performance
- Reduce FPS with `--fps 15`
- Close other applications
- Check CPU usage

### Import errors
Ensure all dependencies are installed:
```bash
pip install mediapipe opencv-python rerun-sdk
```

## Mock Robot Configuration

The mock SO101 robot supports these parameters:

- `smooth_movement`: Enable smooth interpolation (default: true)
- `movement_speed`: Speed of interpolation 0-1 (default: 0.2)
- `add_noise`: Add realistic noise to positions (default: true)
- `noise_level`: Amplitude of noise (default: 0.3)

## Development Notes

### File Structure
```
/home/oliverz/Documents/AlignedRobotics/ARM/
├── lerobot/
│   ├── src/lerobot/
│   │   ├── teleoperators/
│   │   │   └── hand_leader/        # Hand tracking implementation
│   │   └── robots/
│   │       └── so101_follower/     # Robot config and URDF
│   └── tests/mocks/
│       └── mock_so101_robot.py     # Simulated robot
├── cv/
│   └── palm_bounding_box.py        # Original CV code
├── test_hand_standalone.py         # Simple visual test
├── test_hand_sim.py                # Full simulation test
└── run_sim_teleoperate.sh         # Command-line script
```

### Key Components

1. **CVHandTracker** (`cv_hand_tracker.py`): Core hand tracking with MediaPipe
2. **HandLeader** (`hand_leader.py`): Teleoperator interface implementation
3. **MockSO101Robot** (`mock_so101_robot.py`): Simulated robot for testing
4. **Inverse Kinematics**: Converts hand position to joint angles

## Next Steps

Once comfortable with simulation:
1. Connect physical SO101 robot
2. Replace `mock_so101` with `so101_follower` 
3. Add proper USB port configuration
4. Run with real hardware!

## Support

For issues or questions, check the main LeRobot documentation or the hand tracking implementation in `/lerobot/src/lerobot/teleoperators/hand_leader/`.