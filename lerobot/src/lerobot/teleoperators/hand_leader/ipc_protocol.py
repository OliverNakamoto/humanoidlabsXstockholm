#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
IPC Protocol for Hand Tracking Communication

Implements efficient binary protocol for Unix domain socket communication
between MediaPipe tracking process and robot control process.
"""

import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Protocol configuration
SOCKET_PATH = "/tmp/lerobot_hand_tracking_dos.sock"
MESSAGE_TIMEOUT = 0.050  # 50ms timeout for 60fps with safety margin
HEARTBEAT_INTERVAL = 1.0  # 1 second heartbeat
MAX_MESSAGE_SIZE = 128  # Maximum message size in bytes

# Message types
MSG_TYPE_HAND_DATA = 1
MSG_TYPE_HEARTBEAT = 2
MSG_TYPE_SHUTDOWN = 3
MSG_TYPE_CALIBRATION = 4
MSG_TYPE_CALIBRATION_COMPLETE = 5

# Binary message format using struct
# Format: '<BBffffffffffff' = little-endian, 2 bytes + 12 floats = 50 bytes
# B: message type (1 byte)
# B: flags (1 byte): bit 0 = hand_detected, bit 1 = calibrated, bit 2 = second_hand_detected
# f: timestamp (4 bytes)
# f: x position (4 bytes)
# f: y position (4 bytes) 
# f: z position (4 bytes)
# f: pinch percentage (4 bytes)
# f: palm_width (4 bytes)
# f: palm_height (4 bytes)
# f: quaternion x (4 bytes)
# f: quaternion y (4 bytes)
# f: quaternion z (4 bytes)
# f: quaternion w (4 bytes)
# f: second hand curl (0-100, 4 bytes)
MESSAGE_FORMAT = '<BBffffffffffff'
MESSAGE_SIZE = struct.calcsize(MESSAGE_FORMAT)


@dataclass
class HandTrackingData:
    """Hand tracking data structure."""
    timestamp: float
    hand_detected: bool
    x: float  # X position in meters
    y: float  # Y position in meters
    z: float  # Z position in meters
    pinch: float  # Pinch percentage 0-100
    palm_width: float  # Palm bounding box width in pixels
    palm_height: float  # Palm bounding box height in pixels
    qx: float  # Quaternion x component
    qy: float  # Quaternion y component
    qz: float  # Quaternion z component
    qw: float  # Quaternion w component
    second_hand_pitch: float  # Second hand curl percentage 0-100 (field kept for compatibility)
    calibrated: bool = True
    second_hand_detected: bool = False
    
    @classmethod
    def create_default(cls) -> 'HandTrackingData':
        """Create default hand tracking data when no hand detected."""
        return cls(
            timestamp=time.time(),
            hand_detected=False,
            x=0.175,  # Default safe X position (mid-range: 5-30cm)
            y=0.0,    # Default safe Y position (center: ±10cm)
            z=0.25,   # Default safe Z position (mid-range: 10-40cm)
            pinch=0.0,
            palm_width=0.0,
            palm_height=0.0,
            qx=0.0,  # Identity quaternion
            qy=0.0,
            qz=0.0,
            qw=1.0,
            second_hand_pitch=0.0,  # Default uncurled (hand open)
            calibrated=False,
            second_hand_detected=False
        )


class HandTrackingProtocol:
    """Binary protocol for efficient hand tracking communication."""
    
    @staticmethod
    def pack_hand_data(data: HandTrackingData) -> bytes:
        """Pack hand tracking data into binary message."""
        # Create flags byte
        flags = 0
        if data.hand_detected:
            flags |= 0x01
        if data.calibrated:
            flags |= 0x02
        if data.second_hand_detected:
            flags |= 0x04
        
        try:
            message = struct.pack(
                MESSAGE_FORMAT,
                MSG_TYPE_HAND_DATA,  # message type
                flags,               # flags
                data.timestamp,      # timestamp
                data.x,             # x position
                data.y,             # y position
                data.z,             # z position
                data.pinch,         # pinch percentage
                data.palm_width,    # palm width
                data.palm_height,   # palm height
                data.qx,            # quaternion x
                data.qy,            # quaternion y
                data.qz,            # quaternion z
                data.qw,            # quaternion w
                data.second_hand_pitch  # second hand curl
            )
            return message
        except struct.error as e:
            logger.error(f"Failed to pack hand data: {e}")
            return b""
    
    @staticmethod
    def unpack_hand_data(message: bytes) -> Optional[HandTrackingData]:
        """Unpack binary message into hand tracking data."""
        if len(message) != MESSAGE_SIZE:
            logger.warning(f"Invalid message size: {len(message)} (expected {MESSAGE_SIZE})")
            return None
        
        try:
            unpacked = struct.unpack(MESSAGE_FORMAT, message)
            msg_type, flags, timestamp, x, y, z, pinch, palm_width, palm_height, qx, qy, qz, qw, second_hand_pitch = unpacked
            
            if msg_type != MSG_TYPE_HAND_DATA:
                logger.warning(f"Invalid message type: {msg_type}")
                return None
            
            hand_detected = bool(flags & 0x01)
            calibrated = bool(flags & 0x02)
            second_hand_detected = bool(flags & 0x04)
            
            return HandTrackingData(
                timestamp=timestamp,
                hand_detected=hand_detected,
                x=x,
                y=y,
                z=z,
                pinch=pinch,
                palm_width=palm_width,
                palm_height=palm_height,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
                second_hand_pitch=second_hand_pitch,
                calibrated=calibrated,
                second_hand_detected=second_hand_detected
            )
        except struct.error as e:
            logger.error(f"Failed to unpack hand data: {e}")
            return None
    
    @staticmethod
    def pack_heartbeat() -> bytes:
        """Pack heartbeat message."""
        return struct.pack('<Bf', MSG_TYPE_HEARTBEAT, time.time())
    
    @staticmethod
    def pack_shutdown() -> bytes:
        """Pack shutdown message."""
        return struct.pack('<Bf', MSG_TYPE_SHUTDOWN, time.time())
    
    @staticmethod
    def pack_calibration_request() -> bytes:
        """Pack calibration request message."""
        return struct.pack('<Bf', MSG_TYPE_CALIBRATION, time.time())
    
    @staticmethod
    def pack_calibration_complete() -> bytes:
        """Pack calibration complete message."""
        return struct.pack('<Bf', MSG_TYPE_CALIBRATION_COMPLETE, time.time())
    
    @staticmethod
    def get_message_type(message: bytes) -> Optional[int]:
        """Get message type from binary message."""
        if len(message) < 1:
            return None
        return struct.unpack('<B', message[:1])[0]


class MessageValidator:
    """Validates message data for safety and sanity."""
    
    @staticmethod
    def validate_hand_data(data: HandTrackingData) -> bool:
        """Validate hand tracking data for reasonable values."""
        # Check timestamp is recent (within 5 seconds) - LOG ONLY, DON'T REJECT
        current_time = time.time()
        time_diff = current_time - data.timestamp
        if abs(time_diff) > 5.0:
            logger.debug(f"Timestamp age: {time_diff:.3f}s - continuing anyway")
        
        # Check position ranges (robot workspace bounds)
        if not (-0.5 <= data.x <= 0.5):
            logger.warning(f"X position out of range: {data.x}")
            return False
        if not (0.0 <= data.y <= 0.6):
            logger.warning(f"Y position out of range: {data.y}")
            return False
        if not (0.0 <= data.z <= 0.5):
            logger.warning(f"Z position out of range: {data.z}")
            return False
        
        # Check pinch percentage
        if not (0.0 <= data.pinch <= 100.0):
            logger.warning(f"Pinch out of range: {data.pinch}")
            return False
        
        # Check palm dimensions (reasonable pixel values)
        if data.hand_detected:
            if not (0.0 <= data.palm_width <= 1000.0):
                logger.warning(f"Palm width out of range: {data.palm_width}")
                return False
            if not (0.0 <= data.palm_height <= 1000.0):
                logger.warning(f"Palm height out of range: {data.palm_height}")
                return False
        
        # Check second hand curl percentage
        if not (0.0 <= data.second_hand_pitch <= 100.0):
            logger.warning(f"Second hand curl out of range: {data.second_hand_pitch}")
            return False
        
        return True
    
    @staticmethod
    def sanitize_hand_data(data: HandTrackingData) -> HandTrackingData:
        """Sanitize hand tracking data by clamping values to safe ranges."""
        import copy
        sanitized = copy.deepcopy(data)
        
        # Clamp positions to workspace bounds
        sanitized.x = max(-0.5, min(0.5, data.x))
        sanitized.y = max(0.0, min(0.6, data.y))
        sanitized.z = max(0.0, min(0.5, data.z))
        
        # Clamp pinch percentage
        sanitized.pinch = max(0.0, min(100.0, data.pinch))
        
        # Clamp palm dimensions
        sanitized.palm_width = max(0.0, min(1000.0, data.palm_width))
        sanitized.palm_height = max(0.0, min(1000.0, data.palm_height))
        
        # Clamp second hand curl percentage
        sanitized.second_hand_pitch = max(0.0, min(100.0, data.second_hand_pitch))
        
        return sanitized


def create_test_data() -> HandTrackingData:
    """Create test hand tracking data for debugging."""
    return HandTrackingData(
        timestamp=time.time(),
        hand_detected=True,
        x=0.1,
        y=0.3,
        z=0.25,
        pinch=50.0,
        palm_width=120.0,
        palm_height=80.0,
        qx=0.0,
        qy=0.0,
        qz=0.707,  # 90-degree rotation around Z-axis
        qw=0.707,
        second_hand_pitch=75.0,  # Test curl value
        calibrated=True,
        second_hand_detected=True
    )


if __name__ == "__main__":
    # Test the protocol
    print("Testing Hand Tracking IPC Protocol")
    print("="*40)
    
    # Create test data
    test_data = create_test_data()
    print(f"Original data: {test_data}")
    
    # Pack and unpack
    packed = HandTrackingProtocol.pack_hand_data(test_data)
    print(f"Packed size: {len(packed)} bytes")
    
    unpacked = HandTrackingProtocol.unpack_hand_data(packed)
    print(f"Unpacked data: {unpacked}")
    
    # Validate
    is_valid = MessageValidator.validate_hand_data(unpacked)
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
    
    # Test heartbeat
    heartbeat = HandTrackingProtocol.pack_heartbeat()
    print(f"Heartbeat message size: {len(heartbeat)} bytes")