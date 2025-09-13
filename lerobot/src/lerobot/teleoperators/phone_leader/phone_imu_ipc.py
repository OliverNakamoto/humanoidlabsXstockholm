#!/usr/bin/env python

import requests
import time
import threading
from typing import Optional, Dict


class PhoneIMUIPC:
    """Client to poll the phone IMU bridge HTTP server."""

    def __init__(self, server_url: str = "http://localhost:8899", timeout: float = 0.2):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._latest: Dict = {
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "gripper": 50.0, "ts": time.time()
        }
        self._poll = False
        self._thr: Optional[threading.Thread] = None

    def connect(self):
        # sanity check
        try:
            self._session.get(self.server_url + "/status", timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"Cannot reach phone IMU server at {self.server_url}: {e}")
        self._poll = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while self._poll:
            try:
                r = self._session.get(self.server_url + "/data", timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    with self._lock:
                        self._latest = data
            except Exception:
                pass
            time.sleep(0.01)

    def get(self) -> Dict:
        with self._lock:
            return dict(self._latest)

    def disconnect(self):
        self._poll = False
        if self._thr:
            self._thr.join(timeout=1.0)

