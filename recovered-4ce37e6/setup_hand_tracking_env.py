#!/usr/bin/env python3
"""
Setup script for MediaPipe hand tracking environment.

Creates a clean virtual environment with MediaPipe + OpenCV.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result

def main():
    # Define paths
    base_dir = Path(__file__).parent
    venv_dir = base_dir / "hand_tracking_venv"

    print("Setting up MediaPipe Hand Tracking Environment")
    print("=" * 50)

    # Remove existing venv if it exists
    if venv_dir.exists():
        print(f"Removing existing venv at {venv_dir}")
        import shutil
        shutil.rmtree(venv_dir)

    # Create virtual environment
    print(f"Creating virtual environment at {venv_dir}")
    run_command([sys.executable, "-m", "venv", str(venv_dir)])

    # Determine Python executable in venv
    if os.name == 'nt':  # Windows
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:  # Unix-like
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    # Upgrade pip
    print("Upgrading pip...")
    run_command([str(pip_exe), "install", "--upgrade", "pip"])

    # Install MediaPipe and dependencies
    print("Installing MediaPipe and dependencies...")
    packages = [
        "mediapipe>=0.10.0,<=0.10.21",
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "requests>=2.25.0"
    ]

    for package in packages:
        print(f"Installing {package}...")
        run_command([str(pip_exe), "install", package])

    # Test installation
    print("Testing MediaPipe installation...")
    test_code = """
import mediapipe as mp
import cv2
import numpy as np
print(f'MediaPipe version: {mp.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print(f'NumPy version: {np.__version__}')
print('✓ All imports successful!')
"""

    result = run_command([str(python_exe), "-c", test_code], check=False)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("✅ HAND TRACKING ENVIRONMENT SETUP COMPLETE!")
        print("=" * 50)
        print(f"Virtual environment: {venv_dir}")
        print(f"Python executable: {python_exe}")
        print("\nTo start the hand tracking server:")
        if os.name == 'nt':
            print(f'"{python_exe}" hand_tracking_ipc_server.py --calibrate')
        else:
            print(f'"{python_exe}" hand_tracking_ipc_server.py --calibrate')
        print("=" * 50)
    else:
        print("\n❌ SETUP FAILED - Check error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()