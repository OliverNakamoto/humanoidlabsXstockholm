# 🎮 Kalman Filter Playground

A real-time visual playground for testing and tuning Kalman filters with phone gyroscope data for robot control.

## 🎯 Features

✅ **Real-time Kalman Filtering** - Fuse gyroscope and accelerometer data
✅ **Live Visualization** - 6 plots showing raw vs filtered data
✅ **Motor Mapping** - Pitch controls Motor 3, Roll controls Motor 4
✅ **Interactive Tuning** - Toggle filtering on/off to see difference
✅ **Uncertainty Display** - View filter confidence in real-time

## 📊 What You'll See

### **6 Real-time Plots:**

1. **Roll Angle** - Raw phone roll vs Kalman filtered roll
2. **Pitch Angle** - Raw phone pitch vs Kalman filtered pitch
3. **Roll Rate** - Gyroscope vs Kalman estimated angular velocity
4. **Pitch Rate** - Gyroscope vs Kalman estimated angular velocity
5. **Motor Commands** - Motor 3 (pitch) & Motor 4 (roll) outputs
6. **Filter Uncertainty** - Kalman filter confidence levels

## 🚀 Quick Start

### Method 1: Automated Test

```bash
python test_kalman_playground.py
```

This will automatically start both the server and playground.

### Method 2: Manual Setup

**Terminal 1: Start Enhanced Server**
```bash
python phone_gyro_server_with_kalman.py
```

**Terminal 2: Start Playground**
```bash
python kalman_filter_playground.py
```

**Phone:** Open `http://localhost:8889` and tap "Start Control"

## 🎮 Controls & Features

### **Phone Interface**
- **Start Control** - Begin sending gyroscope data
- **Toggle Kalman** - Enable/disable Kalman filtering
- **Calibrate** - Reset robot to center position
- **Motor Displays** - Real-time Motor 3 & Motor 4 values

### **Playground Plots**
- **Red lines** - Raw sensor data (noisy)
- **Blue lines** - Kalman filtered data (smooth)
- **Green lines** - Gyroscope measurements
- **Purple/Orange** - Motor 3 (pitch) & Motor 4 (roll)

## ⚙️ Motor Mapping

| Phone Movement | Motor | Joint | Control |
|----------------|-------|--------|---------|
| **Tilt LEFT/RIGHT** | Motor 0 | shoulder_pan | Position X |
| **Tilt FORWARD/BACK** | Motor 1 | shoulder_lift | Position Y |
| **Roll PHONE** | Motor 2 | elbow_flex | Position Z |
| **PITCH PHONE** | **Motor 3** | wrist_flex | **Pitch** |
| **ROLL PHONE** | **Motor 4** | wrist_roll | **Roll** |

## 🔬 Kalman Filter Details

### **State Vector**
```
x = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
```

### **Sensor Fusion**
- **Gyroscope** → High-frequency angular rates (drifts over time)
- **Accelerometer** → Low-frequency orientation (noisy but stable)
- **Kalman Filter** → Optimal fusion of both sensors

### **Noise Parameters**
```python
Process noise: [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
Gyro noise:    [0.1, 0.1, 0.1]
Accel noise:   [0.3, 0.3]
```

## 🧪 Testing & Tuning

### **What to Test:**

1. **Move phone slowly** - Watch filtering smooth out noise
2. **Move phone quickly** - See how filter tracks fast movements
3. **Hold phone steady** - Observe noise reduction
4. **Toggle Kalman on/off** - Compare raw vs filtered
5. **Watch Motor 3** - Tilt phone forward/back (pitch)
6. **Watch Motor 4** - Roll phone left/right (roll)

### **Tuning Parameters:**

Edit `kalman_filter_playground.py` to adjust:

```python
# Process noise (how much we trust prediction)
self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]) * dt

# Gyroscope noise (measurement trust)
self.R_gyro = np.diag([0.1, 0.1, 0.1])

# Accelerometer noise (measurement trust)
self.R_accel = np.diag([0.3, 0.3])
```

**Higher values** = Trust sensor less (more smoothing)
**Lower values** = Trust sensor more (less smoothing)

## 📈 Understanding the Plots

### **Good Filtering Signs:**
- ✅ Blue line smoother than red line
- ✅ Filter tracks fast movements
- ✅ Uncertainty decreases over time
- ✅ Motor commands are stable

### **Poor Filtering Signs:**
- ❌ Blue line too smooth (misses movements)
- ❌ Blue line too noisy (not filtering enough)
- ❌ High uncertainty values
- ❌ Jerky motor commands

## 🔧 Troubleshooting

### **No Data Showing:**
1. Check phone is sending data (numbers changing on interface)
2. Allow motion permissions in browser
3. Try HTTPS if motion sensors don't work (use ngrok)

### **Plots Not Updating:**
1. Install dependencies: `pip install matplotlib scipy numpy`
2. Check server is running on port 8889
3. Restart both server and playground

### **Noisy/Jerky Filtering:**
1. Increase noise parameters (more smoothing)
2. Check phone is held steadily during testing
3. Calibrate more frequently

### **Filter Too Slow:**
1. Decrease noise parameters (less smoothing)
2. Increase process noise (trust prediction less)
3. Check update rate is ~100Hz

## 🎯 Real Robot Integration

Once you've tuned the filter, use it with your robot:

```bash
cd lerobot
py -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=COM6 \
  --robot.id=follower \
  --teleop.type=phone_gyro \
  --teleop.server_url=http://localhost:8889 \
  --display_data=true \
  --fps=60
```

The enhanced server (`phone_gyro_server_with_kalman.py`) provides filtered data that the robot teleoperator will use automatically.

## 📚 Learning Resources

### **Kalman Filter Theory:**
- State estimation and sensor fusion
- Prediction and update steps
- Noise modeling and tuning
- Extended Kalman Filters for nonlinear systems

### **IMU Sensor Fusion:**
- Gyroscope vs accelerometer characteristics
- Complementary vs Kalman filtering
- Quaternion-based orientation tracking
- Calibration and bias compensation

## 🎉 Advanced Experiments

### **Try These:**
1. **Add magnetometer** for yaw estimation
2. **Implement complementary filter** for comparison
3. **Add bias estimation** for gyroscope drift
4. **Use quaternions** instead of Euler angles
5. **Add GPS** for absolute position (if available)

### **Modify Noise Models:**
1. **Adaptive noise** based on movement intensity
2. **Temperature compensation** for sensor drift
3. **Outlier rejection** for bad measurements

The playground is designed to be educational and practical - use it to understand how Kalman filters work and tune them for your specific robot control application! 🤖📱