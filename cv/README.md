# Computer Vision Pipeline for Robot Arm Control

A complete computer vision system that replaces expensive leader arms in LeRobot setups by using hand tracking to control follower robot arms. Features real-time hand detection, 3D position estimation, gripper control, and seamless LeRobot integration.

## Features

- **Real-time Hand Tracking**: MediaPipe-based hand detection at 30+ FPS
- **3D Position Estimation**: X-Y mapping from hand center, Z-depth from hand size
- **Gripper Control**: Open/closed detection from hand landmarks
- **Workspace Calibration**: Interactive calibration system for camera-robot mapping
- **LeRobot Integration**: Drop-in replacement for hardware leader arms
- **Smoothing & Filtering**: Reduces jitter for stable robot control
- **Visualization**: Real-time overlay showing hand position and robot targets

## Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run Basic Demo**:
```bash
python demo.py --basic
```

3. **Calibrate Workspace** (recommended):
```bash
python calibrate_workspace.py
```

4. **Test LeRobot Integration**:
```bash
python demo.py --lerobot
```

## Usage Examples

### Basic Hand Tracking
```python
from cv_hand_tracker import HandTracker
import cv2

tracker = HandTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = tracker.process_frame(frame)
    
    # result['action'] = [x, y, z, gripper_open]
    # result['detected'] = True/False
    # result['visualization'] = annotated frame
    
    cv2.imshow('Tracking', result['visualization'])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### LeRobot Integration
```python
from lerobot_integration import CVHandTeleop

# Create teleoperation interface
teleop = CVHandTeleop(action_format='lerobot')
teleop.start()

# Get robot actions
action = teleop.get_action()  # [x, y, z, roll, pitch, yaw, gripper]
```

### Data Collection
```python
from lerobot_integration import LeRobotDataCollector

collector = LeRobotDataCollector()
episode_data = collector.collect_episode(duration=10.0)
collector.save_data('demonstration.npz')
```

## File Structure

- `cv_hand_tracker.py` - Core hand tracking pipeline
- `calibrate_workspace.py` - Workspace calibration system
- `lerobot_integration.py` - LeRobot compatibility layer
- `demo.py` - Comprehensive demonstration script
- `requirements.txt` - Python dependencies

## Action Format

The system outputs normalized coordinates compatible with LeRobot:

**Simple Format**: `[x, y, z, gripper_open]`
- x, y, z: Normalized coordinates [-1, 1]
- gripper_open: 0 (closed) to 1 (open)

**LeRobot Format**: `[x, y, z, roll, pitch, yaw, gripper]`
- x, y, z: Robot workspace coordinates (meters)
- roll, pitch, yaw: Orientation (currently 0)
- gripper: -1 (closed) to 1 (open)

## Calibration

Run interactive calibration to map camera view to robot workspace:

```bash
python calibrate_workspace.py
```

Follow on-screen instructions to calibrate 5 key positions. Calibration data is saved to `workspace_calibration.json`.

## Demo Modes

```bash
# All available demos
python demo.py

# Specific demos
python demo.py --basic           # Basic hand tracking
python demo.py --calibrate       # Calibration process
python demo.py --lerobot         # LeRobot integration
python demo.py --collect         # Data collection
python demo.py --performance     # Performance testing
python demo.py --interactive     # Interactive mode with controls
```

## Performance

- **Latency**: ~33ms per frame (30 FPS)
- **Detection Range**: 10% frame padding for reliable tracking
- **Smoothing**: 5-frame moving average filter
- **Accuracy**: Sub-pixel hand center detection

## Integration with Existing LeRobot Code

Replace hardware leader arm initialization:

```python
# Before (hardware leader arm)
# leader_arm = HardwareLeaderArm()

# After (CV hand tracking)
from lerobot_integration import CVHandTeleop
leader_arm = CVHandTeleop(action_format='lerobot')
```

The interface maintains compatibility with existing LeRobot teleoperation patterns.

## Requirements

- Python 3.8+
- USB camera or webcam
- OpenCV 4.8+
- MediaPipe 0.10+
- NumPy 1.24+

## Troubleshooting

**No camera detected**: Check camera ID with `ls /dev/video*` or try different camera IDs
**Poor tracking**: Ensure good lighting and clear background
**Calibration issues**: Make sure full robot workspace is visible in camera view
**Low FPS**: Reduce camera resolution or close other camera applications