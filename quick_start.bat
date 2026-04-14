@echo off
REM Quick Start Script for Eye Tracker V2
REM This script helps you get started quickly on Windows

echo ========================================
echo Eye Tracker V2 - Quick Start
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Python detected
python --version

REM Check if dependencies are installed
echo.
echo [2/4] Checking dependencies...
python -c "import cv2, numpy, mediapipe, yaml" >nul 2>&1
if errorlevel 1 (
    echo Dependencies not found. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies already installed
)

REM Create output directory
if not exist "output" mkdir output
echo [3/4] Output directory ready

REM Check if config exists
if not exist "config.yaml" (
    echo.
    echo [4/4] Generating default configuration...
    python eye_tracker_v2.py --generate-config config.yaml
    echo Configuration file created: config.yaml
) else (
    echo [4/4] Configuration file found
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Choose an option:
echo.
echo [1] Run tracker (default camera)
echo [2] Run with calibration
echo [3] Run with data export
echo [4] Run with recording
echo [5] View help
echo [6] Exit
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Starting eye tracker...
    python eye_tracker_v2.py
) else if "%choice%"=="2" (
    echo.
    echo Starting calibration mode...
    python eye_tracker_v2.py --calibrate
) else if "%choice%"=="3" (
    echo.
    echo Starting with data export...
    python eye_tracker_v2.py --export-data output/session.csv
) else if "%choice%"=="4" (
    echo.
    echo Starting with video recording...
    python eye_tracker_v2.py --record output/recording.mp4 --export-data output/session.csv
) else if "%choice%"=="5" (
    echo.
    python eye_tracker_v2.py --help
    echo.
    pause
) else (
    echo.
    echo Goodbye!
)

echo.
pause
