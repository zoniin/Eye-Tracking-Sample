# Eye Tracker V2 - Advanced Real-Time Eye Tracking & Focus Analysis

A professional-grade eye tracking system built with **MediaPipe** for ultra-fast performance (120-180 FPS), featuring comprehensive focus analysis, head pose estimation, and advanced data analytics.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## 🚀 What's New in V2

### Major Improvements
- **MediaPipe Backend**: 3-5x faster than dlib (120-180 FPS vs 30-50 FPS)
- **No Model Downloads**: MediaPipe comes with embedded models (9MB vs 100MB)
- **YAML Configuration**: Fully customizable settings without code changes
- **Data Export**: Export session data to CSV/JSON for analysis
- **Interactive Calibration**: Personalize gaze detection thresholds
- **Head Pose Estimation**: Track pitch, yaw, and roll angles
- **Video Recording**: Save annotated video output
- **Comprehensive Logging**: Configurable logging levels and file output
- **Keyboard Shortcuts**: Full keyboard control during tracking
- **Type Hints**: Complete type annotations for better IDE support
- **Unit Tests**: Comprehensive test suite with pytest

## 📋 Features

### Eye Tracking
- Real-time gaze direction: `center`, `left`, `right`, `up`, `down`, and diagonals
- Adaptive iris position detection using MediaPipe Face Mesh (478 landmarks)
- Customizable gaze thresholds via configuration
- Gaze smoothing for stable detection

### Blink Detection
- Eye Aspect Ratio (EAR) based blink detection
- Configurable sensitivity thresholds
- Blink rate analysis (blinks per minute)
- Abnormal blink pattern detection

### Focus Analysis
- Real-time focus state classification:
  - **Focused**: High center gaze ratio (>75%)
  - **Semi-focused**: Moderate attention (50-75%)
  - **Distracted**: Low attention (<50%)
  - **Away**: Face not visible
- Sliding time window analysis (default 30 seconds)
- Session statistics and productivity metrics
- Hysteresis to prevent state flickering

### Head Pose Estimation
- 3D head orientation tracking
- Pitch (up/down), Yaw (left/right), Roll (tilt) angles
- Useful for posture monitoring and engagement detection

### Data Management
- **CSV Export**: All frame-level data for external analysis
- **JSON Export**: Structured data for programmatic access
- **Session Summaries**: Automatic statistics generation
- **Configurable Paths**: Timestamp-based file naming

### Visualization
- Comprehensive real-time dashboard
- Focus score visualization with color coding
- Session timer and productivity percentages
- FPS counter for performance monitoring
- Metric bars for gaze, blinks, and presence
- Customizable overlay elements

### Calibration
- Interactive calibration mode
- 5-point calibration (center, left, right, up, down)
- Automatic threshold calculation
- Personalized for each user and setup

## 💻 Installation

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/zoniin/Eye-Tracking-Sample.git
cd Eye-Tracking-Sample

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the tracker
python eye_tracker_v2.py
```

### Requirements
- Python 3.8+
- Webcam or video file
- No additional model downloads needed!

### Dependencies
```
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.0
pyyaml>=6.0
```

## 🎯 Usage

### Basic Usage

```bash
# Run with default webcam
python eye_tracker_v2.py

# Use specific camera
python eye_tracker_v2.py --source 1

# Process video file
python eye_tracker_v2.py --source path/to/video.mp4

# Use custom configuration
python eye_tracker_v2.py --config my_config.yaml
```

### Advanced Usage

```bash
# Start with calibration mode
python eye_tracker_v2.py --calibrate

# Export session data
python eye_tracker_v2.py --export-data session_data.csv

# Record annotated video
python eye_tracker_v2.py --record output_video.mp4

# Combine features
python eye_tracker_v2.py --source 0 --export-data data.csv --record video.mp4 --config custom.yaml
```

### Generate Default Configuration

```bash
# Create config.yaml with default settings
python eye_tracker_v2.py --generate-config config.yaml
```

## ⌨️ Keyboard Shortcuts

While tracking is running, use these keys:

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `R` | Reset session statistics |
| `S` | Save/export current data |
| `D` | Toggle dashboard visibility |
| `H` | Toggle keyboard shortcuts help |
| `P` | Pause/Resume tracking |
| `SPACE` | Capture calibration sample (during calibration) |
| `ESC` | Cancel calibration |

## ⚙️ Configuration

### Configuration File Structure

Create a `config.yaml` file to customize all settings:

```yaml
# Gaze detection settings
gaze:
  horizontal_center_min: 0.38
  horizontal_center_max: 0.62
  vertical_center_min: 0.35
  vertical_center_max: 0.65
  smoothing_window: 3

# Blink detection settings
blink:
  ear_threshold: 0.21
  consecutive_frames: 2
  normal_rate_min: 15
  normal_rate_max: 30

# Focus tracking settings
focus:
  window_size: 30.0
  gaze_threshold: 0.7
  face_loss_threshold: 0.2
  update_interval: 0.5

# Visualization settings
visualization:
  show_landmarks: false
  show_eye_regions: true
  show_gaze_arrow: true
  show_dashboard: true
  show_fps: true
  dashboard_width: 300

# Camera settings
camera:
  source: 0
  width: 1280
  height: 720
  fps: 30

# Recording settings
recording:
  enabled: false
  output_path: "output/recording_{timestamp}.mp4"
  codec: "mp4v"
  fps: 30

# Logging
log_level: "INFO"
log_file: null

# Data export
export_data: false
export_path: "output/session_{timestamp}.csv"
```

### Key Configuration Options

#### Gaze Thresholds
Adjust these based on your camera position and personal eye geometry:
- `horizontal_center_min/max`: Define what counts as "center" horizontally (0.0-1.0)
- `vertical_center_min/max`: Define what counts as "center" vertically (0.0-1.0)

**Tip**: Use calibration mode to automatically set these!

#### Blink Detection
- `ear_threshold`: Lower values = more sensitive (typical: 0.18-0.25)
- `consecutive_frames`: How many frames below threshold = blink (prevents false positives)

#### Focus Analysis
- `window_size`: Time window for analysis in seconds (larger = smoother, smaller = more responsive)
- `gaze_threshold`: Minimum center gaze ratio for "focused" state (0.0-1.0)

## 📊 Data Export Format

### CSV Output
Frame-level data with columns:
```
timestamp, gaze, blink_occurred, face_detected, focus_score, focus_state,
blink_count, fps, head_pitch, head_yaw, head_roll
```

### JSON Output
Structured JSON array with detailed frame data:
```json
[
  {
    "timestamp": 1234567890.123,
    "gaze": "center",
    "blink_occurred": false,
    "face_detected": true,
    "focus_score": 85.5,
    "focus_state": "focused",
    "blink_count": 12,
    "head_pose": {
      "pitch": 5.2,
      "yaw": -3.1,
      "roll": 0.8
    },
    "fps": 152.3
  }
]
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=eye_tracker_v2

# Run specific test file
pytest tests/test_eye_tracker.py -v
```

## 🔧 Calibration Guide

1. **Start calibration mode**:
   ```bash
   python eye_tracker_v2.py --calibrate
   ```

2. **Follow the on-screen instructions**:
   - Look at each green marker when it appears
   - Press `SPACE` to capture samples (30 samples per point)
   - Keep your head still during each point

3. **Calibration points**:
   - Center (straight ahead)
   - Left (look left)
   - Right (look right)
   - Up (look up)
   - Down (look down)

4. **Results**:
   - Thresholds are automatically calculated
   - Configuration is updated in memory
   - Save to file with the `--generate-config` option after calibration

## 📈 Performance Benchmarks

| Backend | FPS (720p) | Model Size | Installation |
|---------|-----------|------------|--------------|
| **MediaPipe (V2)** | **120-180** | **9MB** | **Easy** |
| dlib (V1) | 30-50 | 100MB | Complex (C++ compiler) |

Performance tested on:
- Intel Core i7-10700K
- 1280x720 resolution
- Single face detection

## 🎨 Use Cases

- **Productivity Tracking**: Monitor focus during work sessions
- **E-learning**: Track student engagement
- **Research**: Collect eye tracking data for studies
- **Accessibility**: Gaze-based computer control foundation
- **Gaming**: Eye tracking for enhanced gameplay
- **Health**: Monitor screen time and blink patterns
- **UX Testing**: Analyze user attention on interfaces

## 🔍 How It Works

1. **Face Detection**: MediaPipe Face Mesh detects face and 478 landmarks
2. **Eye Tracking**:
   - Extracts eye landmarks (6 points per eye)
   - Detects iris position (5 points per iris)
   - Calculates normalized iris position
3. **Blink Detection**: Computes Eye Aspect Ratio (EAR) from eye landmarks
4. **Gaze Classification**: Maps iris position to 9 gaze directions
5. **Head Pose**: Solves PnP problem to estimate 3D head orientation
6. **Focus Analysis**:
   - Maintains sliding time window of metrics
   - Computes gaze stability, blink rate, presence
   - Classifies focus state with hysteresis

## 🆚 Comparison: V1 (dlib) vs V2 (MediaPipe)

| Feature | V1 (dlib) | V2 (MediaPipe) |
|---------|-----------|----------------|
| Speed | 30-50 FPS | 120-180 FPS |
| Landmarks | 68 points | 478 points + iris |
| Installation | Complex | Simple |
| Model Size | 100MB download | 9MB embedded |
| Iris Tracking | Manual detection | Built-in |
| Platform Support | Limited (C++ needed) | Excellent |
| Configuration | Hard-coded | YAML file |
| Data Export | None | CSV/JSON |
| Calibration | None | Interactive |
| Head Pose | None | Built-in |
| Type Hints | Partial | Complete |
| Tests | None | Comprehensive |

## 🐛 Troubleshooting

### Low FPS
- Reduce resolution: Set `camera.width` and `camera.height` lower
- Disable landmarks: Set `visualization.show_landmarks: false`
- Close other applications

### Inaccurate Gaze Detection
- Run calibration mode
- Adjust lighting (face should be well-lit)
- Position camera at eye level
- Adjust thresholds in config

### Camera Not Found
- Check camera index (try 0, 1, 2...)
- Verify camera permissions
- Test camera with other applications

### Face Not Detected
- Ensure face is well-lit
- Move closer to camera
- Remove obstructions (glasses might work, but can affect accuracy)

## 📝 Legacy Version

The original dlib-based tracker is still available as `eye_tracker.py`. Use it if you:
- Already have dlib installed
- Need the exact original functionality
- Have specific dlib requirements

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional metrics and analytics
- More visualization options
- Mobile/web deployment
- Real-time alerts and notifications
- Machine learning for personalized calibration
- Multi-face tracking
- Integration with productivity tools

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **MediaPipe** by Google for excellent face tracking
- **OpenCV** for computer vision tools
- Original dlib-based implementation inspiration

## 📧 Support

- **Issues**: https://github.com/zoniin/Eye-Tracking-Sample/issues
- **Discussions**: https://github.com/zoniin/Eye-Tracking-Sample/discussions

## 🗺️ Roadmap

- [ ] Web-based interface (browser support)
- [ ] Real-time alerts (break reminders, posture warnings)
- [ ] Cloud sync for multi-device tracking
- [ ] Mobile app (iOS/Android)
- [ ] Integration with productivity apps (Toggl, RescueTime)
- [ ] Advanced analytics dashboard
- [ ] Screen region-specific attention tracking
- [ ] Multi-monitor support
- [ ] Custom calibration patterns
- [ ] Machine learning for adaptive thresholds

---

**Made with ❤️ for researchers, developers, and productivity enthusiasts**

*Star ⭐ this repo if you find it useful!*
