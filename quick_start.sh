#!/bin/bash
# Quick Start Script for Eye Tracker V2
# This script helps you get started quickly on Linux/macOS

set -e

echo "========================================"
echo "Eye Tracker V2 - Quick Start"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "[1/4] Python detected"
python3 --version

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed"
    echo "Please install pip3 from your package manager"
    exit 1
fi

# Check if dependencies are installed
echo ""
echo "[2/4] Checking dependencies..."
if ! python3 -c "import cv2, numpy, mediapipe, yaml" 2>/dev/null; then
    echo "Dependencies not found. Installing..."
    pip3 install -r requirements.txt
else
    echo "Dependencies already installed"
fi

# Create output directory
mkdir -p output
echo "[3/4] Output directory ready"

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo ""
    echo "[4/4] Generating default configuration..."
    python3 eye_tracker_v2.py --generate-config config.yaml
    echo "Configuration file created: config.yaml"
else
    echo "[4/4] Configuration file found"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Choose an option:"
echo ""
echo "[1] Run tracker (default camera)"
echo "[2] Run with calibration"
echo "[3] Run with data export"
echo "[4] Run with recording"
echo "[5] View help"
echo "[6] Exit"
echo ""

read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Starting eye tracker..."
        python3 eye_tracker_v2.py
        ;;
    2)
        echo ""
        echo "Starting calibration mode..."
        python3 eye_tracker_v2.py --calibrate
        ;;
    3)
        echo ""
        echo "Starting with data export..."
        python3 eye_tracker_v2.py --export-data output/session.csv
        ;;
    4)
        echo ""
        echo "Starting with video recording..."
        python3 eye_tracker_v2.py --record output/recording.mp4 --export-data output/session.csv
        ;;
    5)
        echo ""
        python3 eye_tracker_v2.py --help
        echo ""
        read -p "Press enter to continue..."
        ;;
    *)
        echo ""
        echo "Goodbye!"
        ;;
esac

echo ""
