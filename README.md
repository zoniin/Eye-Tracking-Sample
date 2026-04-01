# Eye Tracking Sample

A real-time eye tracking program built with **dlib** and **OpenCV** that detects where the user is looking and analyzes focus/distraction state using the dlib 68-point facial landmark model.

## Features

- Real-time face and eye detection via dlib's HOG-based frontal face detector
- 68-point facial landmark overlay
- Iris centre detection using adaptive thresholding
- Gaze direction classification: `center`, `left`, `right`, `up`, `down`, and diagonals
- Eye Aspect Ratio (EAR) blink counter with blink rate analysis
- Directional gaze arrow drawn on-screen
- **Focus tracking dashboard** — real-time focus score, state classification, and session statistics
- Focus states: `focused` (green), `semi-focused` (yellow), `distracted` (orange), `away` (red)
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

# Disable focus tracking (eye tracking only)
python eye_tracker.py --no-focus-tracking
```

Press **`q`** in the display window to quit.

## How It Works

1. **Face detection** — dlib's HOG-based frontal face detector locates faces in each frame.
2. **Landmark detection** — The 68-point shape predictor localises key facial features including both eyes.
3. **Eye ROI extraction** — The bounding box of each eye's landmark points is cropped from the grayscale frame.
4. **Iris localisation** — The dark iris is isolated with Gaussian blur + binary thresholding; the centroid of the largest contour gives the iris centre.
5. **Gaze classification** — Normalised iris positions are mapped to one of nine gaze directions.
6. **Blink detection** — The Eye Aspect Ratio (EAR) drops sharply during a blink; consecutive frames below a threshold increment the blink counter and track blink rate.
7. **Focus analysis** — A rolling time window aggregates gaze, blink rate, and face presence into a weighted focus score (0–100), which is classified into a focus state with hysteresis to avoid rapid state flickering.

## Visual Overlays

| Overlay | Description |
|---------|-------------|
| Cyan box | Detected face bounding box |
| Yellow dots | All 68 facial landmarks |
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

The gaze classification thresholds in `classify_gaze()` (`eye_tracker.py` ~line 110) may need adjustment depending on your camera position and face geometry. The iris detection threshold (`30` in `detect_iris_position()`) can also be tuned for different lighting conditions. Focus scoring weights and state thresholds are defined as constants at the top of the file.

## License

MIT
