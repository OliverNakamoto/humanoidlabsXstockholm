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
Process Manager for Hand Tracking IPC

Manages the lifecycle of the MediaPipe tracking process and provides
automatic startup/shutdown coordination with the teleoperation system.
"""

import subprocess
import os
import sys
import time
import signal
import logging
from pathlib import Path
from typing import Optional, List

from .ipc_protocol import SOCKET_PATH

logger = logging.getLogger(__name__)


class HandTrackingProcessManager:
    """Manages the MediaPipe hand tracking process lifecycle."""
    
    def __init__(self, 
                 camera_index: int = 0,
                 socket_path: str = SOCKET_PATH,
                 verbose: bool = False,
                 show_window: bool = False,
                 hand_env_path: Optional[str] = None):
        self.camera_index = camera_index
        self.socket_path = socket_path
        self.verbose = verbose
        self.show_window = show_window
        self.process = None
        self.hand_env_path = hand_env_path or self._find_hand_env()
        
        # Build the command to launch tracking process
        self._setup_command()
        
    def _find_hand_env(self) -> Optional[str]:
        """Find the hand tracking environment."""
        # Try the expected location in cv/.venv
        default_path = Path.home() / "Documents/AlignedRobotics/ARM/cv/.venv/bin/python"
        if default_path.exists():
            return str(default_path)
        
        # Fall back to system python
        logger.warning("Hand tracking environment not found at cv/.venv, will use system python")
        return sys.executable
    
    def _setup_command(self):
        """Set up the command to launch the tracking process."""
        script_dir = Path(__file__).parent
        tracking_script = script_dir / "tracking_process.py"
        
        if not tracking_script.exists():
            raise FileNotFoundError(f"Tracking script not found: {tracking_script}")
        
        self.command = [
            self.hand_env_path,
            str(tracking_script),
            "--camera", str(self.camera_index),
            "--socket", self.socket_path
        ]
        
        if self.verbose:
            self.command.append("--verbose")
        
        if self.show_window:
            self.command.append("--show-window")
        
        logger.info(f"Tracking process command: {' '.join(self.command)}")
    
    def start(self) -> bool:
        """Start the MediaPipe tracking process."""
        if self.is_running():
            logger.warning("Tracking process already running")
            return True
        
        try:
            logger.info("Starting MediaPipe tracking process...")
            
            # Clean up any existing socket
            self._cleanup_socket()
            
            # Start the process
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                preexec_fn=os.setsid  # Create new process group
            )
            
            logger.info(f"Tracking process started with PID: {self.process.pid}")
            
            # Wait a moment for the process to initialize
            time.sleep(1.0)
            
            # Check if process started successfully
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"Tracking process failed to start:")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
            
            # Wait for socket to be created
            max_wait = 5.0
            wait_start = time.time()
            while not os.path.exists(self.socket_path):
                if time.time() - wait_start > max_wait:
                    logger.error(f"Socket not created within {max_wait}s")
                    self.stop()
                    return False
                time.sleep(0.1)
            
            logger.info("Tracking process started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start tracking process: {e}")
            self.stop()
            return False
    
    def stop(self) -> bool:
        """Stop the MediaPipe tracking process."""
        if not self.is_running():
            logger.info("Tracking process not running")
            return True
        
        try:
            logger.info("Stopping MediaPipe tracking process...")
            
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=5.0)
                logger.info("Tracking process terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate gracefully, killing...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
            
            self.process = None
            
            # Clean up socket
            self._cleanup_socket()
            
            logger.info("Tracking process stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping tracking process: {e}")
            self.process = None
            self._cleanup_socket()
            return False
    
    def is_running(self) -> bool:
        """Check if the tracking process is running."""
        return self.process is not None and self.process.poll() is None
    
    def get_process_info(self) -> dict:
        """Get information about the tracking process."""
        return {
            'running': self.is_running(),
            'pid': self.process.pid if self.process else None,
            'socket_exists': os.path.exists(self.socket_path),
            'command': ' '.join(self.command)
        }
    
    def restart(self) -> bool:
        """Restart the tracking process."""
        logger.info("Restarting tracking process...")
        self.stop()
        time.sleep(1.0)  # Brief pause
        return self.start()
    
    def _cleanup_socket(self):
        """Clean up the Unix domain socket."""
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
                logger.debug(f"Removed socket: {self.socket_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up socket: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """Test the process manager."""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Hand Tracking Process Manager")
    print("=" * 40)
    
    with HandTrackingProcessManager(verbose=True) as manager:
        print(f"Process info: {manager.get_process_info()}")
        
        try:
            print("Process running, press Ctrl+C to stop...")
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()