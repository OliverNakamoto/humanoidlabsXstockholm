#!/usr/bin/env python3
"""
Phone IMU HTTP Bridge

Serves a simple web UI to capture phone DeviceOrientation and control inputs,
and exposes a small HTTP API for the teleop process to poll.

Endpoints:
  - GET  /            -> serves the IMU web UI (HTML+JS)
  - GET  /status      -> status + calibration info
  - GET  /calibrate   -> set current orientation as zero (server-side)
  - GET  /data        -> latest filtered values
  - POST /update      -> accepts JSON with raw phone orientation + UI inputs

Run:
  python phone_imu_server.py --host 0.0.0.0 --port 8899

Expose to phone via LAN IP (http://<pc-ip>:8899), or use ngrok:
  ngrok http 8899

Security: this is a minimal demo server, do not expose publicly without auth/CSRF.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
import argparse
from urllib.parse import urlparse

IMU_HTML = r"""
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
  <title>Phone IMU Controller</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .row { margin: 12px 0; }
    label { display: inline-block; width: 90px; }
    input[type=range] { width: 220px; }
    button { padding: 10px 16px; margin-right: 8px; }
    .card { border: 1px solid #ccc; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .small { color: #555; font-size: 12px; }
  </style>
  <script>
    let lastSent = 0;
    let sendHz = 50;           // target send rate
    let yawOffset = 0, pitchOffset = 0, rollOffset = 0;
    let hasPermission = false;
    let latest = {yaw:0, pitch:0, roll:0};

    function wrapDeg(a){ return (a + 180) % 360 - 180; }

    function toJSON(){
      // Read UI sliders
      const vx = parseFloat(document.getElementById('vx').value);
      const vy = parseFloat(document.getElementById('vy').value);
      const vz = parseFloat(document.getElementById('vz').value);
      const grip = parseFloat(document.getElementById('gripper').value);
      // Apply offsets to orientation
      const yaw = wrapDeg(latest.yaw - yawOffset);
      const pitch = wrapDeg(latest.pitch - pitchOffset);
      const roll = wrapDeg(latest.roll - rollOffset);
      return {
        ts: Date.now()/1000,
        yaw: yaw, pitch: pitch, roll: roll,
        vx: vx, vy: vy, vz: vz,
        gripper: grip
      };
    }

    function sendUpdate(){
      const now = performance.now();
      const interval = 1000/sendHz;
      if (now - lastSent < interval) return;
      lastSent = now;
      const payload = toJSON();
      fetch('/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).catch(()=>{});
      // Update labels
      document.getElementById('yawVal').innerText   = payload.yaw.toFixed(1);
      document.getElementById('pitVal').innerText   = payload.pitch.toFixed(1);
      document.getElementById('rolVal').innerText   = payload.roll.toFixed(1);
      document.getElementById('vxVal').innerText    = payload.vx.toFixed(2);
      document.getElementById('vyVal').innerText    = payload.vy.toFixed(2);
      document.getElementById('vzVal').innerText    = payload.vz.toFixed(2);
      document.getElementById('gripVal').innerText  = payload.gripper.toFixed(0);
    }

    function requestPermission(){
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function'){
        DeviceOrientationEvent.requestPermission().then(state => {
          hasPermission = (state === 'granted');
          document.getElementById('perm').innerText = hasPermission ? 'granted' : state;
        }).catch(console.error);
      } else {
        hasPermission = true; // Android/desktop
        document.getElementById('perm').innerText = 'not-needed';
      }
    }

    function calibrate(){
      yawOffset = latest.yaw;
      pitchOffset = latest.pitch;
      rollOffset = latest.roll;
      fetch('/calibrate').catch(()=>{});
    }

    function init(){
      window.addEventListener('deviceorientation', (ev)=>{
        // alpha: z/yaw [0..360), beta: x/pitch [-180..180], gamma: y/roll [-90..90]
        latest.yaw = ev.alpha || 0;
        latest.pitch = ev.beta || 0;
        latest.roll = ev.gamma || 0;
      });
      setInterval(sendUpdate, 10);
    }
    window.addEventListener('load', init);
  </script>
</head>
<body>
  <h3>Phone IMU Controller</h3>
  <div class="card">
    <div class="row"><button onclick="requestPermission()">Enable Sensors</button> <span class="small">permission: <span id="perm">unknown</span></span></div>
    <div class="row"><button onclick="calibrate()">Calibrate Zero</button> <span class="small">zeros current yaw/pitch/roll</span></div>
  </div>
  <div class="card">
    <div class="row"><label>Yaw:</label><span id="yawVal">0.0</span>°</div>
    <div class="row"><label>Pitch:</label><span id="pitVal">0.0</span>°</div>
    <div class="row"><label>Roll:</label><span id="rolVal">0.0</span>°</div>
  </div>
  <div class="card">
    <div class="row"><label>Vx</label><input id="vx" type="range" min="-0.3" max="0.3" step="0.01" value="0"><span id="vxVal">0.00</span> m/s</div>
    <div class="row"><label>Vy</label><input id="vy" type="range" min="-0.3" max="0.3" step="0.01" value="0"><span id="vyVal">0.00</span> m/s</div>
    <div class="row"><label>Vz</label><input id="vz" type="range" min="-0.3" max="0.3" step="0.01" value="0"><span id="vzVal">0.00</span> m/s</div>
    <div class="row"><label>Grip</label><input id="gripper" type="range" min="0" max="100" step="1" value="50"><span id="gripVal">50</span>%</div>
  </div>
  <div class="small">Tip: Adjust sliders for translation velocity. Tilt for orientation. Yaw pans base.</div>
</body>
</html>
"""


class State:
    def __init__(self):
        self.zero = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        self.latest = {
            "ts": time.time(),
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "gripper": 50.0
        }

STATE = State()


class Handler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/imu.html':
            html = IMU_HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == '/status':
            payload = {
                "calibrated_zero": STATE.zero,
                "ts": time.time(),
            }
            data = json.dumps(payload).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == '/calibrate':
            # Use current latest as zero
            STATE.zero = {
                "yaw": STATE.latest["yaw"],
                "pitch": STATE.latest["pitch"],
                "roll": STATE.latest["roll"],
            }
            data = json.dumps({"ok": True, "zero": STATE.zero}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == '/data':
            # Return latest with zero removed (phone also subtracts client-side)
            def wrap(a):
                return (a + 180.0) % 360.0 - 180.0
            yaw = wrap(STATE.latest["yaw"] - STATE.zero["yaw"])
            pitch = wrap(STATE.latest["pitch"] - STATE.zero["pitch"])
            roll = wrap(STATE.latest["roll"] - STATE.zero["roll"])
            payload = {
                "ts": STATE.latest["ts"],
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "vx": STATE.latest["vx"],
                "vy": STATE.latest["vy"],
                "vz": STATE.latest["vz"],
                "gripper": STATE.latest["gripper"],
            }
            data = json.dumps(payload).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self._set_cors()
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/update':
            self.send_response(404)
            self._set_cors()
            self.end_headers()
            return

        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode('utf-8'))
            # Validate and update latest
            for k in ("yaw","pitch","roll","vx","vy","vz","gripper"):
                if k in payload:
                    STATE.latest[k] = float(payload[k])
            STATE.latest["ts"] = time.time()
            out = json.dumps({"ok": True}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(out)
        except Exception as e:
            out = json.dumps({"ok": False, "error": str(e)}).encode('utf-8')
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8899)
    args = ap.parse_args()

    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"Phone IMU server listening on http://{args.host}:{args.port}")
    print("Open this URL on your phone (use PC LAN IP instead of 0.0.0.0).")
    print("For internet access, run: ngrok http", args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

