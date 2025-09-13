@echo off
REM Windows batch script to start MediaPipe hand tracking server

echo Starting MediaPipe Hand Tracking Server...
echo ==========================================

REM Check if hand tracking environment exists
if not exist "hand_tracking_venv\Scripts\python.exe" (
    echo Error: Hand tracking environment not found!
    echo Please run: python setup_hand_tracking_env.py
    pause
    exit /b 1
)

REM Start server with calibration
echo Starting server on http://localhost:8888
echo Press Ctrl+C to stop
echo.

hand_tracking_venv\Scripts\python.exe hand_tracking_ipc_server.py --calibrate --port 8888

echo.
echo Hand tracking server stopped.
pause