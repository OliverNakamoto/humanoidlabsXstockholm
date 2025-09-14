# Robot Control Phone Interface - MVP Specification

## Overview
Create a horizontal (landscape) phone web interface for controlling a robot arm. The interface should be optimized for Samsung Galaxy phones held horizontally with a comfortable grip.

## Layout Requirements

### **Layout**: Horizontal Split Screen
- **Left Panel (50%)**: Large gripper button + small control buttons
- **Right Panel (50%)**: Large movement joystick

### **Components Needed**

#### **LEFT PANEL**
1. **Large Gripper Button** (takes up most of left panel)
   - Size: ~70% of left panel height
   - States: OPEN (red) / CLOSED (green)
   - Touch and hold to close, release to open
   - Large text: "GRIPPER OPEN" / "GRIPPER CLOSED"

2. **Small Control Buttons** (bottom of left panel)
   - **Connect Button**: Start/stop gyroscope sensors
   - **Calibrate Button**: Reset orientation to zero
   - Status indicator showing connection state

#### **RIGHT PANEL**
1. **Large Joystick** (centered)
   - Size: ~80% of right panel
   - Circular boundary with draggable knob
   - Crosshairs for visual reference
   - Returns to center when released
   - Range: -15cm to +15cm in both X,Y directions

## Technical Specifications

### **API Endpoints to Implement**

#### **1. GET `/status`**
Returns current robot state:
```json
{
    "valid": true,
    "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
    "joystick": {"x": 0.0, "y": 0.0},
    "gripper_closed": false,
    "age": 0.5
}
```

#### **2. POST `/update`**
Send phone sensor data:
```json
{
    "alpha": 45.5,    // Yaw (compass heading)
    "beta": 10.2,     // Pitch (forward/back tilt)
    "gamma": -5.1,    // Roll (left/right tilt)
    "accel": {
        "x": -0.3,    // Lateral acceleration
        "y": 0.1,     // Forward acceleration
        "z": 9.8      // Vertical acceleration
    }
}
```

#### **3. POST `/joystick`**
Send joystick position:
```json
{
    "x": 7.5,    // X position in cm (-15 to +15)
    "y": -3.2    // Y position in cm (-15 to +15)
}
```

#### **4. POST `/gripper`**
Send gripper state:
```json
{
    "closed": true    // true = closed, false = open
}
```

#### **5. GET `/calibrate`**
Reset orientation offsets - returns:
```json
{
    "success": true
}
```

### **Phone Sensor Integration**

#### **Device Orientation** (Rotated 90° Left for Horizontal Use)
```javascript
// Apply 90-degree left rotation:
// When phone is horizontal, natural movements map correctly
function handleOrientation(event) {
    // Rotation transformation for horizontal grip:
    const rotatedPitch = event.beta;      // Original pitch becomes roll
    const rotatedRoll = -event.gamma;     // Negative original roll becomes pitch
    const rotatedYaw = event.alpha;       // Yaw unchanged

    sendSensorData(rotatedYaw, rotatedPitch, rotatedRoll);
}
```

#### **Device Motion** (Accelerometer)
```javascript
function handleMotion(event) {
    if (event.acceleration) {
        // Also rotate acceleration 90° left:
        const accel = {
            x: event.acceleration.y,      // Y becomes X
            y: -event.acceleration.x,     // -X becomes Y
            z: event.acceleration.z       // Z unchanged
        };
        // Send with orientation data
    }
}
```

### **Visual Design Requirements**

#### **Color Scheme**
- Background: Dark blue gradient
- Gripper OPEN: Red (#e74c3c)
- Gripper CLOSED: Green (#27ae60)
- Joystick knob: Bright green with white border
- Control buttons: Blue (#3498db)

#### **Typography**
- Large gripper button: Bold, 18px+
- Control buttons: 14px
- Status text: 12px
- Font: System default (good mobile readability)

#### **Touch Targets**
- Minimum 44px touch targets
- Gripper button: Large enough for thumb
- Joystick: Easy to drag with thumb
- Control buttons: Comfortable for index finger

### **Responsive Behavior**
- Optimized for landscape orientation
- Works on screens 5"-7" (typical phone range)
- Touch-friendly for one-handed or two-handed use
- Visual feedback on all interactions

### **Connection Flow**
1. User opens page in landscape mode
2. Click "CONNECT" → Request motion sensor permissions
3. Start sending sensor data automatically
4. "CALIBRATE" → Reset orientation to current position as zero
5. Use gripper button and joystick to control robot

### **Data Flow Summary**
```
Phone Sensors → POST /update (orientation + accel)
Joystick Drag → POST /joystick (x, y position)
Gripper Button → POST /gripper (open/closed)
Status Updates ← GET /status (robot feedback)
Calibration → GET /calibrate (reset zero point)
```

### **Server Details**
- Server runs on `http://localhost:8889`
- CORS enabled for cross-origin requests
- JSON request/response format
- Real-time updates at ~20Hz (50ms intervals)

## Success Criteria
- ✅ Large, easy-to-use gripper button on left
- ✅ Responsive joystick on right
- ✅ Works smoothly in landscape orientation
- ✅ Proper 90° gyroscope rotation for horizontal grip
- ✅ All API endpoints correctly implemented
- ✅ Clean, minimal UI focused on robot control
- ✅ Touch-optimized for mobile devices