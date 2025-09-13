# LeRobot SO101 + HandCV Setup Guide

Complete guide for setting up your SO101 robot arm with computer vision hand tracking control.

## Prerequisites

### Hardware Required
- ✅ SO101 follower robot arm
- ✅ USB camera/webcam  
- ✅ USB-to-serial adapter for robot communication
- ✅ Computer with good lighting setup

### Software Requirements
```bash
# Basic dependencies
pip install opencv-python mediapipe numpy

# LeRobot (if not installed)
cd lerobot
pip install -e .

# Optional: IK solver for better control
pip install placo

# Optional: 3D visualization
pip install matplotlib
```

## Step 1: Hardware Connection & Testing

### 1.1 Connect SO101 Robot
```bash
# Find your robot's serial port
# Windows: Check Device Manager for COM ports
# Linux/Mac: ls /dev/tty*

# Test basic connection
python -c "
import serial
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200)  # Adjust port
    print('✓ Robot connected')
    ser.close()
except:
    print('✗ Connection failed - check port and cables')
"
```

### 1.2 Test Camera
```bash
# Test camera access
python -c "
import cv2
cap = cv2.VideoCapture(0)  # Try 0, 1, 2... 
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f'✓ Camera working: {frame.shape}')
    else:
        print('✗ Camera read failed')
    cap.release()
else:
    print('✗ Camera not found')
"
```

## Step 2: Robot Calibration

### 2.1 Motor Calibration (Critical!)

**⚠️ SAFETY: Ensure robot has clear movement space before calibrating**

```bash
# Calibrate SO101 motors - this is REQUIRED for proper operation
python -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=your_robot_id

# Follow the interactive prompts:
# 1. Move robot to middle position manually
# 2. Move all joints through full range of motion
# 3. Calibration data will be saved automatically
```

**What this does:**
- Records motor position ranges
- Sets homing offsets
- Creates calibration file for your specific robot
- **Without this, your robot movements will be incorrect!**

### 2.2 Test Basic Robot Control

```bash
# Test basic robot movement (AFTER calibration)
python -c "
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

config = SO101FollowerConfig(
    port='/dev/ttyUSB0',  # Your port
    id='your_robot_id'
)

robot = SO101Follower(config)
robot.connect()

print('Robot connected. Testing small movement...')

# Get current position
current = robot.get_pos()
print(f'Current position: {current}')

# Small test movement (SAFE)
test_action = current.copy()
test_action['shoulder_pan.pos'] += 0.1  # Small movement

robot.send_action(test_action)
print('✓ Robot movement test complete')

robot.disconnect()
"
```

## Step 3: Hand Tracking Calibration

### 3.1 Test Hand Tracking
```bash
# Test your existing CV hand tracker
cd cv
python demo.py --basic

# Verify hand detection works well:
# - Good lighting
# - Clear background  
# - Hand fully visible
# - Smooth tracking
```

### 3.2 Workspace Calibration
```bash
# Run your workspace calibration
cd cv
python calibrate_workspace.py

# This will:
# 1. Map camera view to robot workspace
# 2. Calibrate depth estimation
# 3. Set workspace boundaries
# 4. Save calibration file
```

### 3.3 Test HandCV Teleoperator
```bash
# Test the integrated HandCV teleoperator
python test_hand_cv_teleop.py --test basic

# Check for:
# ✓ Camera connection
# ✓ Hand detection  
# ✓ Coordinate mapping
# ✓ Action generation
```

## Step 4: Integrated System Testing

### 4.1 Test Without Robot (Simulation)
```bash
# Test teleoperation without robot hardware
python test_hand_cv_teleop.py --test integration

# This tests:
# - LeRobot integration
# - Configuration loading
# - Action format compatibility
```

### 4.2 Test With Robot (CAREFUL!)

**⚠️ START WITH ROBOT IN SAFE POSITION**

```bash
# Test with actual robot - START SLOW
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=your_robot_id \
    --teleop.type=hand_cv \
    --teleop.camera_id=0 \
    --teleop.enable_visualization=true \
    --display_data=true \
    --fps=10  # Start with low FPS for safety

# What to watch for:
# - Smooth hand tracking
# - Robot follows hand movement
# - No jerky or dangerous movements
# - Emergency stop ready (Ctrl+C)
```

### 4.3 Safety Testing Checklist

- [ ] Robot moves in same direction as hand
- [ ] Movements are smooth and proportional  
- [ ] Gripper opens/closes with hand
- [ ] Robot stops when hand not detected
- [ ] Emergency stop (Ctrl+C) works immediately
- [ ] No collision with workspace boundaries

## Step 5: Configuration Tuning

### 5.1 Adjust Workspace Bounds
```python
# Edit your config for optimal workspace
# File: lerobot/src/lerobot/teleoperators/hand_cv/config_hand_cv.py

# Workspace bounds (in meters, relative to robot base)
workspace_x_min: float = -0.3  # Adjust based on your setup
workspace_x_max: float = 0.3
workspace_y_min: float = -0.3  
workspace_y_max: float = 0.3
workspace_z_min: float = 0.05  # Above table surface
workspace_z_max: float = 0.5   # Max reach height

# Tracking sensitivity
confidence_threshold: float = 0.8  # Higher = more strict detection
smoothing_window: int = 7          # Higher = smoother but slower
```

### 5.2 Tune Movement Speed & Responsiveness
```python
# In config_hand_cv.py
update_rate: float = 15.0  # Start low, increase gradually

# For IK settings (if using URDF)
position_weight: float = 1.0     # Focus on position accuracy
orientation_weight: float = 0.01 # Low weight on orientation
```

## Step 6: Advanced Features

### 6.1 Get SO101 URDF for Better IK
```bash
# You'll need the URDF file for accurate inverse kinematics
# Check with robot manufacturer or LeRobot community
# Once you have it, update config:

# In your test command:
--teleop.urdf_path=/path/to/so101.urdf \
--teleop.end_effector_frame="gripper_link"
```

### 6.2 Data Collection for Training
```bash
# Once system works well, collect demonstration data
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --teleop.type=hand_cv \
    --episode-time-s=30 \
    --num-episodes=10 \
    --dataset-name=hand_tracking_demos

# This records your hand movements for later policy training
```

## Troubleshooting Common Issues

### Robot Connection Issues
```bash
# Check port permissions (Linux)
sudo chmod 666 /dev/ttyUSB0

# Find correct port
python -m lerobot.find_port

# Test serial communication
python -c "
import serial
ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', 'COM3', 'COM4']
for port in ports:
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        print(f'✓ {port} works')
        ser.close()
    except:
        print(f'✗ {port} failed')
"
```

### Hand Tracking Issues
```bash
# Improve lighting and background
# Try different camera indices
python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"

# Adjust confidence threshold
# Edit config_hand_cv.py: confidence_threshold = 0.5  # Lower = more sensitive
```

### Movement Issues
```bash
# If robot moves wrong direction:
# 1. Check calibration was completed
# 2. Verify workspace bounds in config
# 3. Test coordinate mapping with visualization

# If movements are jerky:
# 1. Increase smoothing_window
# 2. Decrease update_rate  
# 3. Improve lighting for stable tracking
```

## Quick Start Commands

### Complete Test Sequence
```bash
# 1. Hardware test
python test_3d_workspace.py

# 2. Hand tracking test  
python test_hand_cv_teleop.py

# 3. Robot only test (no hand tracking)
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB1

# 4. Full integrated test
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --teleop.type=hand_cv \
    --teleop.camera_id=0 \
    --fps=10
```

## Success Criteria

You know the system is working when:
- ✅ Robot calibration completes without errors
- ✅ Hand tracking shows stable detection
- ✅ Robot follows hand movements smoothly
- ✅ Gripper responds to hand open/close
- ✅ System handles hand disappearing gracefully
- ✅ Emergency stop works reliably

## Next Steps for Hackathon

Once basic system works:
1. **Web Integration**: Add WebSocket server for remote control
2. **Multiple Users**: Queue system for website visitors
3. **Gesture Commands**: Specific gestures trigger actions
4. **Visual Feedback**: Live video stream to website
5. **Safety Features**: Collision detection, workspace limits

The foundation is solid - now you can build awesome demo features on top! 🚀