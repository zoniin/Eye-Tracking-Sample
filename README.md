# Eye Tracking Sample

A real-time eye tracking program with two versions:
- **V2 (Recommended)**: Fast MediaPipe-based tracker with advanced features → [README_V2.md](README_V2.md)
- **V1 (Legacy)**: Original dlib-based tracker (this file)

## 🚀 NEW: Eye Tracker V2 is Here!

**Eye Tracker V2** offers massive improvements:
- ⚡ **3-5x faster** (120-180 FPS vs 30-50 FPS)
- 📦 **No model downloads** (MediaPipe comes ready)
- 🎯 **Data export** (CSV/JSON for analysis)
- ⚙️ **YAML configuration** (no code changes needed)
- 🎨 **Interactive calibration** (personalized accuracy)
- 📊 **Head pose tracking** (pitch, yaw, roll)
- 🎬 **Video recording** (save annotated output)
- ✅ **Easy installation** (no C++ compiler needed!)

**[👉 Click here for V2 Documentation (README_V2.md)](README_V2.md)**

**Quick start V2:**
```bash
pip install opencv-python numpy mediapipe pyyaml
python eye_tracker_v2.py
```

---

# Eye Tracker V1 (Legacy - dlib)

Original eye tracking implementation built with **dlib** and **OpenCV** using the dlib 68-point facial landmark model.

## Features

- Real-time face and eye detection via dlib's frontal face detector
- 68-point facial landmark overlay
- Iris centre detection using adaptive thresholding
- Gaze direction classification: `center`, `left`, `right`, `up`, `down`, and diagonals
- Eye Aspect Ratio (EAR) blink counter
- Directional gaze arrow drawn on-screen
- Supports webcam input or a pre-recorded video file

## Requirements

- Python 3.8+
- A webcam (or a video file)
- The dlib 68-point shape predictor model (downloaded via `setup.sh`)

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Eye-Tracking-Sample

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the dlib shape predictor model (~100 MB)
bash setup.sh
```

> **Note:** Building `dlib` from source requires CMake and a C++ compiler.
> On Ubuntu/Debian: `sudo apt install build-essential cmake`
> On macOS: `brew install cmake`

## Usage

```bash
# Run with the default webcam (camera index 0)
python eye_tracker.py

# Specify a different camera index
python eye_tracker.py --source 1

# Run on a pre-recorded video file
python eye_tracker.py --source path/to/video.mp4

# Specify a custom path to the .dat model file
python eye_tracker.py --predictor /path/to/shape_predictor_68_face_landmarks.dat
```

Press **`q`** in the display window to quit.

## How It Works

1. **Face detection** — dlib's HOG-based frontal face detector locates faces in each frame.
2. **Landmark detection** — The 68-point shape predictor localises key facial features including both eyes.
3. **Eye ROI extraction** — The bounding box of each eye's landmark points is cropped from the grayscale frame.
4. **Iris localisation** — The dark iris is isolated with Gaussian blur + binary thresholding; the centroid of the largest contour gives the iris centre.
5. **Gaze classification** — Normalised iris positions are mapped to one of nine gaze directions.
6. **Blink detection** — The Eye Aspect Ratio (EAR) drops sharply during a blink; consecutive frames below a threshold increment the blink counter.

## Visual Overlays

| Overlay | Description |
|---------|-------------|
| Cyan box | Detected face bounding box |
| Yellow dots | All 68 facial landmarks |
| Orange box | Eye region of interest |
| Red dot | Estimated iris centre |
| Green/orange arrow | Gaze direction arrow |
| Top-left HUD | Current gaze label and blink count |

## Tuning

The gaze classification thresholds in `classify_gaze()` (`eye_tracker.py` ~line 110) may need adjustment depending on your camera position and face geometry. The iris detection threshold (`30` in `detect_iris_position()`) can also be tuned for different lighting conditions.

## License

MIT
