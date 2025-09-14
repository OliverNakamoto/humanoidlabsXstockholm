#!/bin/bash
echo "Killing any old teleop and tracking processes..."
# Kill the main teleop script
pkill -f "lerobot.teleoperate"
# Kill the tracking process specifically
pkill -f "tracking_process.py"

echo "Removing stale socket file..."
# Force remove the socket file if it exists
rm -f /tmp/lerobot_hand_tracking.sock

echo "✅ System is clean. Ready to start."
