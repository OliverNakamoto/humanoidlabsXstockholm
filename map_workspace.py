#!/usr/bin/env python3

"""
Live Workspace Mapping Utility

This script allows you to map your robot's actual workspace by moving it around manually
and tracking the end effector positions using forward kinematics.

Usage:
    python map_workspace.py --robot.type=so101_follower --robot.port=/dev/ttyACM0
                           --robot.id=my_awesome_follower_arm
                           --teleop.type=hand_leader_ipc
                           --teleop.urdf_path=/path/to/robot.urdf
                           --teleop.show_window=false
                           --duration=60

Instructions:
    1. Run this script with your robot configuration
    2. Move the robot arm to explore its full workspace:
       - Reach as far forward/backward as possible
       - Reach as far left/right as possible
       - Reach as high/low as safely possible
    3. The script will track boundaries in real-time
    4. Press Ctrl+C to stop early
    5. Review the suggested workspace bounds
"""

import logging
import time
from dataclasses import dataclass

import draccus

# Import robots to register them (same as teleoperate.py)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)

# Import teleoperator
from lerobot.teleoperators.hand_leader.hand_leader_ipc import HandLeaderIPC
from lerobot.teleoperators.hand_leader.config_hand_leader import HandLeaderIPCConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceMappingConfig:
    """Configuration for workspace mapping."""
    robot: RobotConfig
    teleop: HandLeaderIPCConfig
    duration: int = 60  # Mapping duration in seconds (0 = manual stop)
    update_config: bool = False  # Whether to update workspace bounds in config


def main():
    """Main workspace mapping function."""

    @draccus.wrap()
    def run_mapping(cfg: WorkspaceMappingConfig):
        init_logging()

        logger.info("🗺️  ROBOT WORKSPACE MAPPING UTILITY")
        logger.info("="*50)

        # Create robot and teleoperator
        logger.info("Initializing robot...")
        robot = make_robot_from_config(cfg.robot)

        logger.info("Initializing teleoperator...")
        teleop = HandLeaderIPC(cfg.teleop)

        try:
            # Connect robot
            logger.info("Connecting to robot...")
            robot.connect()

            # Connect teleoperator (but we won't use hand tracking)
            logger.info("Connecting teleoperator...")
            teleop.connect()

            # Start live workspace mapping
            logger.info("Starting live workspace mapping...")
            logger.info("💡 MOVE THE ROBOT ARM MANUALLY to explore its workspace!")

            # Main mapping loop - this will track robot positions
            mapped_bounds = teleop.map_workspace_live(cfg.duration)

            if mapped_bounds:
                logger.info("✅ Workspace mapping completed successfully!")

                if cfg.update_config:
                    logger.info("💾 Updating workspace bounds in configuration...")
                    # Here you could save the bounds to a config file
                    logger.info("   (Config update not implemented yet)")
                else:
                    logger.info("💡 Copy the suggested bounds to your configuration manually")

            else:
                logger.error("❌ Workspace mapping failed")

        except KeyboardInterrupt:
            logger.info("🛑 Mapping interrupted by user")

        except Exception as e:
            logger.error(f"❌ Error during mapping: {e}")

        finally:
            # Cleanup
            try:
                teleop.disconnect()
                robot.disconnect()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")

            logger.info("👋 Workspace mapping utility finished")

    run_mapping()


if __name__ == "__main__":
    main()