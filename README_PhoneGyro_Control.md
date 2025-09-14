# 📱 Phone Gyroscope Robot Control

Control your SO101 robot using your phone's gyroscope and accelerometer! This system uses inverse kinematics to translate phone movements into precise robot control.

## 🎮 How It Works

- **Phone Tilt (Pitch/Roll)** → Robot end-effector position (X/Y/Z)
- **Phone Orientation** → End-effector orientation (roll/pitch/yaw)
- **Real-time IK** → Converts poses to joint angles
- **Ngrok Tunnel** → Exposes control interface to your phone anywhere

## 🚀 Quick Start

### 1. Install Ngrok (Optional but Recommended)

```bash
# Download from https://ngrok.com/download
# Extract to your PATH or project directory
# Sign up and get auth token, then run:
ngrok authtoken YOUR_TOKEN_HERE
```

### 2. Start the Phone Control System

```bash
python start_phone_control.py
```

This will:
- Start the gyro server on port 8889
- Create an ngrok tunnel (if installed)
- Show you the URL to open on your phone

### 3. Connect Your Phone

1. Open the printed URL on your phone browser
2. Tap **"Start Control"** and allow motion permissions
3. Tap **"Calibrate"** to reset the robot to center position
4. Move your phone to see the position values change

### 4. Start Robot Teleoperation

In a new terminal:

```bash
cd lerobot
py -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=COM6 \
  --robot.id=follower \
  --teleop.type=phone_gyro \
  --display_data=true \
  --fps=60
```

## 🎯 Controls Mapping

| Phone Movement | Robot Action |
|----------------|--------------|
| Tilt **LEFT/RIGHT** | End-effector moves **LEFT/RIGHT** (X-axis) |
| Tilt **FORWARD/BACK** | End-effector moves **FORWARD/BACK** (Y-axis) |
| Roll **CLOCKWISE/COUNTER** | End-effector moves **UP/DOWN** (Z-axis) |
| Phone **ROLL** | End-effector **ROLL** orientation |
| Phone **PITCH** | End-effector **PITCH** orientation |
| Phone **YAW** | End-effector **YAW** orientation |

## 🛠️ Manual Setup (Without start_phone_control.py)

### Step 1: Start Gyro Server

```bash
python phone_gyro_server.py
```

Server will start on `http://localhost:8889`

### Step 2: Create Ngrok Tunnel (Optional)

```bash
ngrok http 8889
```

Copy the public URL (e.g., `https://abc123.ngrok.io`)

### Step 3: Test System

```bash
python test_phone_gyro.py
```

Open the URL on your phone, start control, and watch the terminal output.

### Step 4: Run Robot Control

```bash
cd lerobot
py -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=CORRECT_PORT \
  --robot.id=follower \
  --teleop.type=phone_gyro \
  --teleop.server_url=http://localhost:8889 \
  --display_data=true \
  --fps=60
```

## ⚙️ Configuration Options

You can customize the phone gyro behavior:

```bash
py -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=COM6 \
  --robot.id=follower \
  --teleop.type=phone_gyro \
  --teleop.server_url=http://localhost:8889 \
  --teleop.position_scale=1.5 \
  --teleop.orientation_scale=0.8 \
  --teleop.use_degrees=false \
  --display_data=true \
  --fps=60
```

### Configuration Parameters

- `server_url`: Phone gyro server URL (default: http://localhost:8889)
- `position_scale`: Position sensitivity (default: 1.0)
- `orientation_scale`: Orientation sensitivity (default: 1.0)
- `use_degrees`: Use degrees instead of normalized range (default: false)

## 🔧 Troubleshooting

### Phone Not Connecting

1. **Same Network**: If not using ngrok, ensure phone and computer are on the same WiFi
2. **Firewall**: Check if Windows Firewall is blocking port 8889
3. **HTTPS**: Some browsers require HTTPS for motion sensors (ngrok provides this)

### Robot Not Moving

1. **Robot Connection**: Verify robot is connected and on correct COM port
2. **Calibration**: Tap "Calibrate" on phone interface to reset center position
3. **Permissions**: Ensure browser has motion sensor permissions

### No Gyroscope Data

1. **Permissions**: Allow motion access when prompted
2. **HTTPS**: Motion sensors often require secure connection (use ngrok)
3. **Browser**: Try Chrome or Safari (better motion sensor support)

### IK Errors

1. **URDF Path**: Update `urdf_path` in config if using custom robot model
2. **Joint Names**: Verify joint names match your robot configuration
3. **Workspace Limits**: Movements might be outside robot's reachable workspace

## 📱 Phone Interface Features

### Status Display
- **Position**: Current end-effector position (x, y, z in meters)
- **Orientation**: Current orientation (roll, pitch, yaw in degrees)
- **Connection**: Shows if data is being received

### Controls
- **Start Control**: Begin sending gyroscope data
- **Calibrate**: Reset robot to center position
- **Stop**: Stop sending data

### Instructions
- Built-in help showing control mapping
- Real-time data display

## 🏗️ Architecture

```
Phone Browser
    ↓ (Gyroscope/Accelerometer data via HTTP)
Phone Gyro Server (port 8889)
    ↓ (Position/Orientation data)
Phone Gyro Teleoperator
    ↓ (Inverse Kinematics)
SO101 Robot (via LeRobot)
```

## 🚀 Advanced Features

### Custom IK Configuration

Create custom config for your robot:

```python
from lerobot.teleoperators.phone_gyro import PhoneGyroConfig

config = PhoneGyroConfig(
    server_url="http://localhost:8889",
    urdf_path="path/to/your/robot.urdf",
    target_frame_name="custom_end_effector",
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5"],
    position_scale=1.2,
    orientation_scale=0.9
)
```

### Position vs Velocity Control

The system currently uses **position integration** - gyroscope rates are integrated to get positions. This provides:

✅ **Advantages**:
- Absolute positioning
- Stable control
- Natural feel

❌ **Disadvantages**:
- Can drift over time
- Requires calibration

### Alternative: Direct Velocity Control

For velocity control, modify the integration in `phone_gyro_server.py`:

```python
# Instead of integrating position, use rates directly
# This would require changes to the teleoperator
```

## 🎯 Tips for Best Control

1. **Hold phone steadily** - sudden movements can cause jumpy motion
2. **Calibrate often** - use the calibrate button to reset center position
3. **Smooth movements** - gradual tilts work better than quick jerks
4. **Landscape mode** - often provides better control axis mapping
5. **Test first** - use `test_phone_gyro.py` before connecting robot

## 📊 Performance

- **Update Rate**: ~100Hz from phone, ~60Hz to robot
- **Latency**: ~50-100ms over local network, ~150-300ms over internet
- **Accuracy**: Depends on phone's gyroscope quality and calibration

Enjoy controlling your robot with your phone! 🤖📱