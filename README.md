# Eye Tracking Sample

A real-time eye tracking program with **two powerful versions**:

## 🚀 Choose Your Version

### **Version 2 (V2)** - Advanced System with Full Features
**Recommended for:** Research, productivity tracking, data analysis, advanced use cases

- ⚡ **Ultra-fast** (120-180 FPS)
- 📊 **Complete data export** (CSV/JSON)
- ⚙️ **YAML configuration** system
- 🎯 **Interactive calibration** mode
- 📹 **Video recording** capability
- 🧑 **Head pose estimation** (pitch, yaw, roll)
- ⌨️ **Keyboard shortcuts** and full control
- 🧪 **Professional code quality** (type hints, tests, docs)

**[👉 See V2 Documentation (README_V2.md)](README_V2.md)**

**Quick start V2:**
```bash
pip install opencv-python numpy mediapipe pyyaml
python eye_tracker_v2.py --calibrate
```

---

### **Version 1 (V1)** - Updated Original with MediaPipe
**Recommended for:** Quick demos, learning basics, simple eye tracking

Built with **MediaPipe Face Mesh** (default) and **OpenCV**. Optional dlib backend for legacy support.

**Quick start V1:**
```bash
pip install mediapipe opencv-python numpy imutils
python eye_tracker.py
```

---

## 📊 Version Comparison

| Feature | V1 (Updated) | V2 (Advanced) |
|---------|--------------|---------------|
| **Backend** | MediaPipe or dlib | MediaPipe only |
| **Speed** | 120-180 FPS | 120-180 FPS |
| **Data Export** | ❌ None | ✅ CSV/JSON |
| **Configuration** | ❌ Hard-coded | ✅ YAML files |
| **Calibration** | ❌ No | ✅ Interactive |
| **Head Pose** | ❌ No | ✅ Yes |
| **Video Recording** | ❌ No | ✅ Yes |
| **Type Hints** | ❌ Partial | ✅ Complete |
| **Tests** | ❌ No | ✅ Comprehensive |
| **Documentation** | ✅ Basic | ✅ Extensive (7 docs) |

**Recommendation:** Use **V2** for serious work, **V1** for quick demos.

---

# Version 1 (V1) - Documentation

A real-time eye tracking program built with **MediaPipe Face Mesh** and **OpenCV** that detects where the user is looking and analyzes focus/distraction state using advanced facial landmark detection.

## Features

- **MediaPipe Face Mesh backend** (default) — 3-5x faster than dlib with better tracking robustness
- Real-time face and eye detection with 478-point facial landmarks
- Native iris tracking using MediaPipe's built-in iris landmarks (no thresholding needed!)
- Gaze direction classification: `center`, `left`, `right`, `up`, `down`, and diagonals
- Eye Aspect Ratio (EAR) blink counter with blink rate analysis
- Directional gaze arrow drawn on-screen
- **Focus tracking dashboard** — real-time focus score, state classification, and session statistics
- Focus states: `focused` (green), `semi-focused` (yellow), `distracted` (orange), `away` (red)
- Supports webcam input or pre-recorded video files
- **Dual-backend support** — MediaPipe (default) or legacy dlib backend

## Performance

| Backend | FPS (720p) | Model Size | Tracking Robustness |
|---------|-----------|------------|---------------------|
| **MediaPipe** | 120-180 | 9MB (embedded) | Excellent (±45° head rotation) |
| dlib | 30-50 | 100MB (.dat file) | Good (±30° head rotation) |

## Requirements

- Python 3.8+
- A webcam (or a video file)
- **MediaPipe** (default backend - no additional downloads needed!)
- **dlib** (optional - only for legacy `--backend dlib` mode)

## Installation

### Quick Start (MediaPipe - Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/zoniin/Eye-Tracking-Sample
cd Eye-Tracking-Sample

# 2. Install Python dependencies
pip install mediapipe>=0.10.9 opencv-python>=4.8.0 numpy>=1.24.0 imutils>=0.5.4

# 3. Run the eye tracker!
python eye_tracker.py
```

That's it! No model downloads required. MediaPipe's 9MB model is embedded in the library.

### Full Installation (with dlib backend support)

```bash
# Install all dependencies (including optional dlib)
pip install -r requirements.txt

# Download the dlib shape predictor model (~100 MB) - only if you want to use --backend dlib
bash setup.sh
```

> **Note:** Building `dlib` from source requires CMake and a C++ compiler.
> On Ubuntu/Debian: `sudo apt install build-essential cmake`
> On macOS: `brew install cmake`
> On Windows: dlib may require Visual Studio Build Tools

## Usage

### MediaPipe Backend (Default - Recommended)

```bash
# Run with the default webcam (camera index 0)
python eye_tracker.py

# Specify a different camera index
python eye_tracker.py --source 1

# Run on a pre-recorded video file
python eye_tracker.py --source path/to/video.mp4

# Explicitly specify MediaPipe backend
python eye_tracker.py --backend mediapipe

# Disable focus tracking (eye tracking only)
python eye_tracker.py --no-focus-tracking
```

### dlib Backend (Legacy)

```bash
# Use dlib backend with default model path
python eye_tracker.py --backend dlib

# Use dlib with custom model path
python eye_tracker.py --backend dlib --predictor /path/to/shape_predictor_68_face_landmarks.dat

# dlib backend on video file
python eye_tracker.py --backend dlib --source path/to/video.mp4
```

Press **`q`** in the display window to quit.

## How It Works

### MediaPipe Backend (Default)

1. **Face detection & landmark extraction** — MediaPipe Face Mesh detects faces and extracts 478 facial landmarks in a single pass using a deep learning model.
2. **Iris tracking** — MediaPipe provides 5 dedicated iris landmarks per eye (468-477) with the center landmark giving direct iris position (no thresholding needed!).
3. **Eye ROI extraction** — Eye contours are extracted from 16 specific MediaPipe landmarks per eye.
4. **Gaze classification** — Normalized iris positions (from native landmarks) are mapped to one of nine gaze directions.
5. **Blink detection** — The Eye Aspect Ratio (EAR) is calculated from 6 key eye landmarks; consecutive frames below threshold (0.21) indicate a blink.
6. **Focus analysis** — A rolling time window aggregates gaze, blink rate, and face presence into a weighted focus score (0–100), classified into a focus state with hysteresis.

### dlib Backend (Legacy)

1. **Face detection** — dlib's HOG-based frontal face detector locates faces.
2. **Landmark detection** — The 68-point shape predictor localizes key facial features.
3. **Eye ROI extraction** — Eye bounding boxes are cropped from landmarks 36-47.
4. **Iris localization** — CV2 thresholding isolates the dark iris; centroid gives iris position.
5. **Gaze & blink** — Same as MediaPipe backend.
6. **Focus analysis** — Same as MediaPipe backend.

## Visual Overlays

| Overlay | Description |
|---------|-------------|
| Cyan box | Detected face bounding box |
| Yellow dots | All facial landmarks |
| Orange box | Eye region of interest |
| Red dot | Estimated iris centre |
| Green/orange arrow | Gaze direction (green = center, orange = off-center) |
| Top-left HUD | Current gaze label and blink count |
| Focus dashboard | Focus score bar, current state, blink rate, session stats |

## Focus Dashboard

The on-screen dashboard (top-right) shows:

- **Focus Score** — composite 0–100 score weighted from gaze (50%), face presence (30%), and blink rate (20%)
- **State** — color-coded label: `FOCUSED`, `SEMI-FOCUSED`, `DISTRACTED`, or `AWAY`
- **Blink Rate** — blinks per minute with status (`very low`, `normal`, `high`, etc.)
- **Session Time** — elapsed session duration
- **Focused %** — percentage of session spent in a focused state

## Tuning

The gaze classification thresholds in `classify_gaze()` may need adjustment depending on your camera position and face geometry. Focus scoring weights and state thresholds are defined as constants at the top of the file.

---

## 🎯 Want More Features?

**Check out Version 2 (V2)** for:
- ✅ Data export for analysis
- ✅ YAML configuration
- ✅ Interactive calibration
- ✅ Head pose tracking
- ✅ Video recording
- ✅ Keyboard shortcuts
- ✅ Professional documentation

**[See V2 Documentation →](README_V2.md)**

---

## License

MIT
