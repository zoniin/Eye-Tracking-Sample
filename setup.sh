#!/usr/bin/env bash
# setup.sh — Optional: Download and extract the dlib 68-point shape predictor model.
# MediaPipe is now the default backend and requires no additional downloads!
# Only run this if you want to use the legacy dlib backend (--backend dlib).

set -euo pipefail

MODEL_BZ2="shape_predictor_68_face_landmarks.dat.bz2"
MODEL_DAT="shape_predictor_68_face_landmarks.dat"
DOWNLOAD_URL="http://dlib.net/files/${MODEL_BZ2}"

echo "==========================================="
echo " Eye Tracking Sample - Setup"
echo "==========================================="
echo ""
echo "MediaPipe is now the default backend (no downloads needed!)."
echo ""
echo "This script downloads the dlib shape predictor model (~100MB)"
echo "which is ONLY needed if you want to use the legacy dlib backend"
echo "with --backend dlib."
echo ""

if [ -f "$MODEL_DAT" ]; then
    echo "[INFO] $MODEL_DAT already exists — skipping download."
    exit 0
fi

read -p "Download dlib model for legacy backend? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "[INFO] Skipping dlib download. Use --backend mediapipe (default)."
    exit 0
fi

echo "[INFO] Downloading $MODEL_BZ2 ..."
if command -v curl &>/dev/null; then
    curl -L -o "$MODEL_BZ2" "$DOWNLOAD_URL"
elif command -v wget &>/dev/null; then
    wget -O "$MODEL_BZ2" "$DOWNLOAD_URL"
else
    echo "[ERROR] Neither curl nor wget found. Please install one and retry."
    exit 1
fi

echo "[INFO] Extracting ..."
bzip2 -d "$MODEL_BZ2"

echo "[INFO] Done. Model saved to: $MODEL_DAT"
echo "[INFO] You can now use: python eye_tracker.py --backend dlib"
