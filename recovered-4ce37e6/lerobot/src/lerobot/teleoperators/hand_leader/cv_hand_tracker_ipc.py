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

import requests
import time
import logging
from typing import Optional, Tuple, Dict
import threading

logger = logging.getLogger(__name__)


class CVHandTrackerIPC:
    """
    IPC client for hand tracking server.

    Communicates with separate MediaPipe process via HTTP API.
    Provides same interface as direct MediaPipe implementation.
    """

    def __init__(self, server_url: str = "http://localhost:8888", timeout: float = 1.0):
        """
        Initialize IPC client.

        Args:
            server_url: URL of hand tracking server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.calibrated = False
        # Reuse HTTP session to cut request overhead/latency
        self._session = requests.Session()

        # Cache for position data
        self.current_data = {
            'x': 0.0, 'y': 0.2, 'z': 0.2, 'pinch': 0.0,
            'valid': False, 'timestamp': time.time()
        }
        self.data_lock = threading.Lock()

        # Reference position for relative tracking (set on first valid reading)
        self.reference_position = None
        self.last_valid_position = None

        # Background polling
        self.polling_thread = None
        self.stop_polling = False

    def connect(self):
        """Connect to hand tracking server."""
        logger.info(f"Connecting to hand tracking server at {self.server_url}")

        try:
            # Check if server is running
            response = self._request('GET', '/status')
            if response:
                logger.info("Connected to hand tracking server")
                self.calibrated = response.get('calibrated', False)

                # Start background polling for better responsiveness
                self._start_polling()
            else:
                raise RuntimeError("Hand tracking server not responding")

        except Exception as e:
            raise RuntimeError(f"Failed to connect to hand tracking server: {e}")

    def calibrate(self):
        """Trigger calibration on server."""
        logger.info("Starting remote calibration...")

        try:
            response = self._request('GET', '/calibrate')
            if response and response.get('success'):
                self.calibrated = True
                logger.info("Remote calibration completed")
            else:
                error = response.get('error', 'Unknown error') if response else 'No response'
                raise RuntimeError(f"Calibration failed: {error}")

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise

    def start_tracking(self):
        """Start tracking (server handles this automatically)."""
        if not self._start_polling():
            logger.warning("Failed to start background polling")

    def _start_polling(self) -> bool:
        """Start background polling thread for position updates."""
        if self.polling_thread is not None:
            return True

        self.stop_polling = False
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started background position polling")
        return True

    def _polling_loop(self):
        """Background thread that polls server for position updates."""
        while not self.stop_polling:
            try:
                # Get position from server
                response = self._request('GET', '/position', timeout=0.1)  # Fast timeout

                if response:
                    with self.data_lock:
                        new_data = {
                            'x': response.get('x', 0.0),
                            'y': response.get('y', 0.2),
                            'z': response.get('z', 0.2),
                            'pinch': response.get('pinch', 0.0),
                            'valid': response.get('valid', False),
                            'timestamp': time.time()
                        }

                        # Set reference position on first valid reading
                        if new_data['valid'] and self.reference_position is None:
                            self.reference_position = {
                                'x': new_data['x'],
                                'y': new_data['y'],
                                'z': new_data['z']
                            }
                            logger.info(f"Set reference position: {self.reference_position}")

                        # Update current data and last valid position
                        self.current_data = new_data
                        if new_data['valid']:
                            self.last_valid_position = {
                                'x': new_data['x'],
                                'y': new_data['y'],
                                'z': new_data['z'],
                                'pinch': new_data['pinch']
                            }
                else:
                    # Server not responding - mark data invalid
                    with self.data_lock:
                        self.current_data['valid'] = False
                        self.current_data['timestamp'] = time.time()

            except Exception as e:
                logger.debug(f"Polling error (normal during server restart): {e}")
                with self.data_lock:
                    self.current_data['valid'] = False
                    self.current_data['timestamp'] = time.time()

            time.sleep(0.01)  # Poll at ~100Hz

    def get_current_position(self) -> Tuple[float, float, float, float]:
        """
        Get current hand position and pinch value.

        Returns:
            Tuple of (x, y, z, pinch) in robot coordinates
            x, y, z in meters, pinch in percentage (0-100)
        """
        with self.data_lock:
            data = self.current_data.copy()

        # Check if data is stale (older than 100ms)
        if time.time() - data['timestamp'] > 0.1:
            data['valid'] = False

        if not data['valid']:
            # Return last valid position instead of resetting to zero
            if self.last_valid_position is not None:
                logger.debug("No valid hand tracking data, using last valid position")
                return (
                    self.last_valid_position['x'],
                    self.last_valid_position['y'],
                    self.last_valid_position['z'],
                    self.last_valid_position['pinch']
                )
            else:
                logger.debug("No valid hand tracking data and no reference, using default")
                return (0.0, 0.2, 0.2, 0.0)

        # Return position relative to reference (if set)
        if self.reference_position is not None:
            # Calculate relative position from reference
            rel_x = data['x'] - self.reference_position['x']
            rel_y = data['y'] - self.reference_position['y']
            rel_z = data['z'] - self.reference_position['z']

            # Add relative movement to workspace center
            workspace_center_x = 0.0
            workspace_center_y = 0.2
            workspace_center_z = 0.15

            return (
                workspace_center_x + rel_x,
                workspace_center_y + rel_y,
                workspace_center_z + rel_z,
                data['pinch']
            )
        else:
            return (data['x'], data['y'], data['z'], data['pinch'])

    def reset_reference_position(self):
        """Reset the reference position to current hand position."""
        with self.data_lock:
            if self.current_data['valid']:
                self.reference_position = {
                    'x': self.current_data['x'],
                    'y': self.current_data['y'],
                    'z': self.current_data['z']
                }
                logger.info(f"Reference position reset to: {self.reference_position}")
            else:
                logger.warning("Cannot reset reference - no valid hand position available")

    def disconnect(self):
        """Disconnect from server."""
        self.stop_polling = True

        if self.polling_thread:
            self.polling_thread.join(timeout=1.0)

        logger.info("Disconnected from hand tracking server")

    def _request(self, method: str, endpoint: str, timeout: float = None) -> Optional[Dict]:
        """
        Make HTTP request to server.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., '/position')
            timeout: Request timeout (uses self.timeout if None)

        Returns:
            Response JSON data or None if request failed
        """
        if timeout is None:
            timeout = self.timeout

        try:
            url = f"{self.server_url}{endpoint}"

            if method == 'GET':
                response = self._session.get(url, timeout=timeout)
            elif method == 'POST':
                response = self._session.post(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Server returned status {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.debug(f"Request timeout to {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            logger.debug(f"Connection error to {endpoint}")
            return None
        except Exception as e:
            logger.debug(f"Request error to {endpoint}: {e}")
            return None


# Global instance for compatibility with existing code
_tracker_instance = None


def get_tracker_instance(camera_index: int = 0, server_url: str = "http://localhost:8888") -> CVHandTrackerIPC:
    """Get or create the global IPC tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CVHandTrackerIPC(server_url=server_url)
    return _tracker_instance
