#!/usr/bin/env bash
# setup.sh — Download and extract the dlib 68-point shape predictor model.
# Run this once before using eye_tracker.py.

set -euo pipefail

MODEL_BZ2="shape_predictor_68_face_landmarks.dat.bz2"
MODEL_DAT="shape_predictor_68_face_landmarks.dat"
DOWNLOAD_URL="http://dlib.net/files/${MODEL_BZ2}"

if [ -f "$MODEL_DAT" ]; then
    echo "[INFO] $MODEL_DAT already exists — skipping download."
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
