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
IPC Client for Hand Tracking Communication

Client-side implementation for receiving hand tracking data
from MediaPipe process via Unix domain sockets.
"""

import socket
import time
import threading
import logging
from typing import Optional, Callable
import os

from .ipc_protocol import (
    HandTrackingData,
    HandTrackingProtocol,
    MessageValidator,
    SOCKET_PATH,
    MESSAGE_TIMEOUT,
    MSG_TYPE_HAND_DATA,
    MSG_TYPE_HEARTBEAT,
    MSG_TYPE_CALIBRATION,
    MSG_TYPE_CALIBRATION_COMPLETE
)

logger = logging.getLogger(__name__)


class HandTrackingClient:
    """Client for receiving hand tracking data via Unix domain sockets."""
    
    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.socket = None
        self.connected = False
        self.running = False
        
        # Latest received data
        self.latest_data = HandTrackingData.create_default()
        self.data_lock = threading.Lock()
        
        # Connection monitoring
        self.last_heartbeat = 0.0
        self.last_data_received = 0.0
        
        # Calibration status
        self.calibration_complete = False
        self.calibration_event = threading.Event()
        
        # Receiver thread
        self.receiver_thread = None
        
        # Callbacks
        self.data_callback: Optional[Callable[[HandTrackingData], None]] = None
        self.connection_callback: Optional[Callable[[bool], None]] = None
        
    def connect(self) -> bool:
        """Connect to the hand tracking server."""
        try:
            if self.connected:
                return True
                
            # Check if socket exists
            if not os.path.exists(self.socket_path):
                logger.warning(f"Socket does not exist: {self.socket_path}")
                return False
            
            # Create Unix domain socket
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.socket.settimeout(MESSAGE_TIMEOUT)
            
            # Test connection by sending a heartbeat request
            # We bind to a temporary socket for replies
            temp_path = f"{self.socket_path}_client_{os.getpid()}"
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            self.socket.bind(temp_path)
            
            # Send test message to server
            test_message = HandTrackingProtocol.pack_heartbeat()
            self.socket.sendto(test_message, self.socket_path)
            
            self.connected = True
            self.running = True
            
            # Start receiver thread
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()
            
            logger.info("Connected to hand tracking server")
            
            if self.connection_callback:
                self.connection_callback(True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to hand tracking server: {e}")
            self._cleanup_socket()
            return False
    
    def disconnect(self):
        """Disconnect from the hand tracking server."""
        logger.info("Disconnecting from hand tracking server")
        
        self.running = False
        self.connected = False
        
        # Wait for receiver thread to finish
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)
        
        self._cleanup_socket()
        
        if self.connection_callback:
            self.connection_callback(False)
        
        logger.info("Disconnected from hand tracking server")
    
    def _cleanup_socket(self):
        """Clean up socket resources."""
        if self.socket:
            try:
                # Get the socket path for cleanup
                sock_name = self.socket.getsockname()
                self.socket.close()
                # Remove our client socket file
                if os.path.exists(sock_name):
                    os.unlink(sock_name)
            except Exception as e:
                logger.warning(f"Error cleaning up socket: {e}")
            finally:
                self.socket = None
    
    def _receiver_loop(self):
        """Main receiver loop for handling incoming messages."""
        logger.info("Starting hand tracking data receiver loop")
        
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                # Receive message
                data, addr = self.socket.recvfrom(1024)
                
                # Reset failure counter on successful receive
                consecutive_failures = 0
                
                # Determine message type and handle accordingly
                msg_type = HandTrackingProtocol.get_message_type(data)
                
                if msg_type == MSG_TYPE_HAND_DATA:
                    self._handle_hand_data(data)
                elif msg_type == MSG_TYPE_HEARTBEAT:
                    self._handle_heartbeat(data)
                elif msg_type == MSG_TYPE_CALIBRATION_COMPLETE:
                    self._handle_calibration_complete(data)
                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                
            except socket.timeout:
                # Timeout is normal, check connection health
                self._check_connection_health()
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Receiver error ({consecutive_failures}/{max_failures}): {e}")
                
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive failures, disconnecting")
                    self.connected = False
                    if self.connection_callback:
                        self.connection_callback(False)
                    break
                
                time.sleep(0.1)  # Brief pause before retry
        
        logger.info("Hand tracking data receiver loop ended")
    
    def _handle_hand_data(self, data: bytes):
        """Handle incoming hand tracking data."""
        hand_data = HandTrackingProtocol.unpack_hand_data(data)
        if hand_data is None:
            logger.warning("Failed to unpack hand tracking data")
            return
        
        # Validate data
        if not MessageValidator.validate_hand_data(hand_data):
            logger.warning("Invalid hand tracking data received")
            hand_data = MessageValidator.sanitize_hand_data(hand_data)
        
        # Update latest data thread-safely
        with self.data_lock:
            self.latest_data = hand_data
            self.last_data_received = time.time()
        
        # Call data callback if set
        if self.data_callback:
            try:
                self.data_callback(hand_data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def _handle_heartbeat(self, data: bytes):
        """Handle incoming heartbeat message."""
        self.last_heartbeat = time.time()
        logger.debug("Received heartbeat from server")
    
    def _handle_calibration_complete(self, data: bytes):
        """Handle calibration complete message."""
        self.calibration_complete = True
        self.calibration_event.set()
        logger.info("Received calibration complete signal from server")
    
    def _check_connection_health(self):
        """Check if connection is still healthy."""
        current_time = time.time()
        
        # Check if we haven't received data or heartbeat recently
        data_timeout = current_time - self.last_data_received > 5.0  # 5 seconds
        heartbeat_timeout = current_time - self.last_heartbeat > 10.0  # 10 seconds
        
        if data_timeout and heartbeat_timeout:
            logger.warning("Connection appears to be dead (no data or heartbeat)")
            self.connected = False
            if self.connection_callback:
                self.connection_callback(False)
    
    def get_latest_data(self) -> HandTrackingData:
        """Get the latest hand tracking data."""
        with self.data_lock:
            return self.latest_data
    
    def wait_for_calibration(self, timeout: float = 30.0) -> bool:
        """Wait for calibration to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if calibration completed, False if timeout
        """
        return self.calibration_event.wait(timeout)
    
    def get_current_position(self) -> tuple[float, float, float, float]:
        """Get current hand position in format expected by existing code."""
        with self.data_lock:
            return (
                self.latest_data.x,
                self.latest_data.y,
                self.latest_data.z,
                self.latest_data.pinch
            )
    
    def get_current_position_and_orientation(self) -> tuple[float, float, float, float, float, float, float, float]:
        """Get current hand position and orientation."""
        with self.data_lock:
            return (
                self.latest_data.x,
                self.latest_data.y,
                self.latest_data.z,
                self.latest_data.pinch,
                self.latest_data.qx,
                self.latest_data.qy,
                self.latest_data.qz,
                self.latest_data.qw
            )
    
    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        return self.connected and self.running
    
    def is_hand_detected(self) -> bool:
        """Check if hand is currently detected."""
        with self.data_lock:
            return self.latest_data.hand_detected
    
    def is_calibrated(self) -> bool:
        """Check if hand tracking is calibrated."""
        with self.data_lock:
            return self.latest_data.calibrated
    
    def request_calibration(self) -> bool:
        """Request recalibration from the server."""
        if not self.connected:
            logger.error("Not connected to server")
            return False
        
        try:
            calibration_msg = HandTrackingProtocol.pack_calibration_request()
            self.socket.sendto(calibration_msg, self.socket_path)
            logger.info("Calibration request sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send calibration request: {e}")
            return False
    
    def get_data_age(self) -> float:
        """Get age of latest data in seconds."""
        with self.data_lock:
            return time.time() - self.latest_data.timestamp
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics for debugging."""
        with self.data_lock:
            current_time = time.time()
            return {
                'connected': self.connected,
                'running': self.running,
                'data_age': current_time - self.latest_data.timestamp,
                'last_heartbeat_age': current_time - self.last_heartbeat,
                'last_data_age': current_time - self.last_data_received,
                'hand_detected': self.latest_data.hand_detected,
                'calibrated': self.latest_data.calibrated
            }
    
    def set_data_callback(self, callback: Callable[[HandTrackingData], None]):
        """Set callback function for incoming data."""
        self.data_callback = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Set callback function for connection state changes."""
        self.connection_callback = callback


class HandTrackingClientManager:
    """Higher-level manager for hand tracking client with automatic reconnection."""
    
    def __init__(self, socket_path: str = SOCKET_PATH, auto_reconnect: bool = True):
        self.socket_path = socket_path
        self.auto_reconnect = auto_reconnect
        self.client = None
        self.reconnect_thread = None
        self.shutdown = False
        
    def start(self) -> bool:
        """Start the hand tracking client."""
        self.client = HandTrackingClient(self.socket_path)
        
        if self.auto_reconnect:
            self.client.set_connection_callback(self._on_connection_change)
        
        success = self.client.connect()
        
        if success and self.auto_reconnect:
            # Start reconnection monitor
            self.reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
            self.reconnect_thread.start()
        
        return success
    
    def stop(self):
        """Stop the hand tracking client."""
        self.shutdown = True
        
        if self.client:
            self.client.disconnect()
        
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=1.0)
    
    def get_client(self) -> Optional[HandTrackingClient]:
        """Get the underlying client instance."""
        return self.client
    
    def _on_connection_change(self, connected: bool):
        """Handle connection state changes."""
        if connected:
            logger.info("Hand tracking connection established")
        else:
            logger.warning("Hand tracking connection lost")
    
    def _reconnect_loop(self):
        """Automatic reconnection loop."""
        while not self.shutdown:
            try:
                if self.client and not self.client.is_connected():
                    logger.info("Attempting to reconnect to hand tracking server...")
                    if self.client.connect():
                        logger.info("Reconnection successful")
                    else:
                        logger.warning("Reconnection failed, will retry...")
                
                time.sleep(2.0)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in reconnection loop: {e}")
                time.sleep(5.0)  # Longer delay on error


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Hand Tracking IPC Client")
    print("="*40)
    
    client = HandTrackingClient()
    
    def on_data(data):
        print(f"Received: x={data.x:.3f}, y={data.y:.3f}, z={data.z:.3f}, pinch={data.pinch:.1f}")
    
    def on_connection(connected):
        print(f"Connection: {'CONNECTED' if connected else 'DISCONNECTED'}")
    
    client.set_data_callback(on_data)
    client.set_connection_callback(on_connection)
    
    if client.connect():
        print("Connected to hand tracking server")
        try:
            while True:
                time.sleep(1)
                stats = client.get_connection_stats()
                print(f"Stats: {stats}")
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        print("Failed to connect to hand tracking server")
        print("Make sure the tracking process is running")
    
    client.disconnect()