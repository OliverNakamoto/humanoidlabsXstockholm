#!/bin/bash

# Script to run hand tracking teleoperation with simulated robot

echo "=========================================="
echo "Hand Tracking Teleoperation with Mock Robot"
echo "=========================================="
echo ""
echo "This will run the teleoperation with:"
echo "- Simulated SO101 robot (no hardware needed)"
echo "- Computer vision hand tracking"
echo "- Rerun visualization"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run the teleoperation command
cd lerobot

python -m lerobot.teleoperate \
    --robot.type=mock_so101 \
    --robot.id=simulated_follower \
    --robot.smooth_movement=true \
    --robot.movement_speed=0.2 \
    --robot.add_noise=true \
    --robot.noise_level=0.3 \
    --teleop.type=hand_leader \
    --teleop.camera_index=0 \
    --teleop.urdf_path=src/lerobot/robots/so101_follower/so101.urdf \
    --teleop.id=cv_hand_tracker \
    --display_data=true \
    --fps=30