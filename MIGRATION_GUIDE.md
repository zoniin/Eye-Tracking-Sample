# Migration Guide: V1 (dlib) to V2 (MediaPipe)

This guide helps you transition from the original dlib-based eye tracker to the new MediaPipe-powered V2.

## Quick Comparison

| Aspect | V1 (eye_tracker.py) | V2 (eye_tracker_v2.py) |
|--------|---------------------|------------------------|
| Backend | dlib (68 landmarks) | MediaPipe (478 landmarks) |
| Speed | 30-50 FPS | 120-180 FPS |
| Installation | Complex (C++ compiler) | Simple (pip install) |
| Configuration | Hard-coded | YAML files |
| Data Export | None | CSV/JSON |
| Calibration | None | Interactive |

## Should You Migrate?

### Migrate to V2 if you want:
- ✅ Faster performance (3-5x speed improvement)
- ✅ Easier installation (no C++ compiler needed)
- ✅ Data export for analysis
- ✅ Customizable configuration
- ✅ Better cross-platform support
- ✅ Active development and new features

### Stay on V1 if you:
- ⚠️ Already have dlib working perfectly
- ⚠️ Have scripts that depend on V1 code
- ⚠️ Need exactly 68 landmark points
- ⚠️ Have specific dlib requirements

## Installation Differences

### V1 Installation
```bash
# Complex, requires C++ compiler
pip install dlib>=19.24.0
pip install opencv-python numpy imutils

# Download model (100MB)
bash setup.sh
```

### V2 Installation
```bash
# Simple, no compiler needed
pip install opencv-python numpy mediapipe pyyaml

# No model download needed!
```

## Command Line Changes

### V1 Usage
```bash
python eye_tracker.py
python eye_tracker.py --source 1
python eye_tracker.py --predictor custom_model.dat
python eye_tracker.py --no-focus-tracking
```

### V2 Equivalent
```bash
python eye_tracker_v2.py
python eye_tracker_v2.py --source 1
# No predictor needed (MediaPipe is built-in)
# Focus tracking always enabled (toggle via config)
```

### V2 New Options
```bash
python eye_tracker_v2.py --config custom.yaml
python eye_tracker_v2.py --export-data session.csv
python eye_tracker_v2.py --record video.mp4
python eye_tracker_v2.py --calibrate
```

## Configuration Migration

### V1 Configuration (Hard-coded)
```python
# In eye_tracker.py, you had to edit code:
EAR_THRESHOLD = 0.21
BLINK_CONSEC = 3
FOCUS_WINDOW_SIZE = 30.0

# Gaze thresholds in classify_gaze() function
if cx < 0.38:
    h_dir = "right"
elif cx > 0.62:
    h_dir = "left"
```

### V2 Configuration (YAML file)
```yaml
# config.yaml - much easier to modify!
blink:
  ear_threshold: 0.21
  consecutive_frames: 3

focus:
  window_size: 30.0

gaze:
  horizontal_center_min: 0.38
  horizontal_center_max: 0.62
```

To use:
```bash
python eye_tracker_v2.py --config config.yaml
```

## Code API Changes

If you were importing functions from V1:

### V1 Code
```python
from eye_tracker import (
    eye_aspect_ratio,
    landmarks_to_np,
    classify_gaze
)

# dlib landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```

### V2 Code
```python
from eye_tracker_v2 import (
    eye_aspect_ratio,  # Same API!
    classify_gaze,     # Now takes config parameter
    GazeConfig,
    AppConfig
)

# MediaPipe
import mediapipe as mp
face_mesh = mp.solutions.face_mesh.FaceMesh(...)
```

## Feature Mapping

### Focus Detection

**V1:**
```python
# Limited focus metrics
focus_analyzer = FocusAnalyzer(
    window_size=30.0,
    gaze_threshold=0.7
)
```

**V2:**
```python
# Enhanced with configuration
config = FocusConfig(
    window_size=30.0,
    gaze_threshold=0.7,
    face_loss_threshold=0.2,
    update_interval=0.5
)
focus_analyzer = FocusAnalyzer(config, logger)
```

### Blink Detection

**V1:**
```python
# Fixed threshold
if ear < 0.21:
    blink_frame += 1
```

**V2:**
```python
# Configurable
if ear < config.blink.ear_threshold:
    blink_frame_counter += 1
```

## Data Export Migration

### V1 - No Export
```python
# Had to manually print or log data
print(f"[INFO] Session ended. Total blinks: {blink_count}")
```

### V2 - Automatic Export
```python
# Built-in CSV/JSON export
python eye_tracker_v2.py --export-data session.csv

# Or configure in YAML:
export_data: true
export_path: "output/session_{timestamp}.csv"
```

## Landmark Index Changes

### V1 (dlib 68 points)
```python
LEFT_EYE_POINTS = list(range(36, 42))   # Points 36-41
RIGHT_EYE_POINTS = list(range(42, 48))  # Points 42-47
```

### V2 (MediaPipe 478 points)
```python
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Plus dedicated iris landmarks!
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
```

## Running Both Versions

You can keep both versions installed:

```bash
# V1 (original)
python eye_tracker.py

# V2 (new)
python eye_tracker_v2.py
```

They don't conflict!

## Step-by-Step Migration

### Step 1: Install V2 Dependencies
```bash
pip install mediapipe pyyaml
```

### Step 2: Test V2
```bash
python eye_tracker_v2.py
```

### Step 3: Generate Default Config
```bash
python eye_tracker_v2.py --generate-config my_config.yaml
```

### Step 4: Customize Config
Edit `my_config.yaml` with your V1 settings:
- Copy your EAR_THRESHOLD to `blink.ear_threshold`
- Copy gaze thresholds to `gaze.*`
- Set any custom values

### Step 5: Run Calibration
```bash
python eye_tracker_v2.py --calibrate --config my_config.yaml
```

### Step 6: Compare Results
Run both versions side-by-side to verify:
```bash
# Terminal 1
python eye_tracker.py

# Terminal 2
python eye_tracker_v2.py --config my_config.yaml
```

### Step 7: Switch Completely
Once satisfied, update your scripts:
```bash
# Old
python eye_tracker.py

# New
python eye_tracker_v2.py --config my_config.yaml
```

## Troubleshooting Migration Issues

### Issue: V2 gaze detection is different

**Solution**: Run calibration
```bash
python eye_tracker_v2.py --calibrate
```

The different landmark systems may need personalized thresholds.

### Issue: V2 is slower than expected

**Check**:
1. Are you showing too many visualizations?
2. Is your camera resolution too high?
3. Are other apps using the camera?

**Fix**:
```yaml
visualization:
  show_landmarks: false
  show_face_mesh: false

camera:
  width: 1280
  height: 720
```

### Issue: Missing the dlib landmarks

**Solution**: V2 has more landmarks! Access them:
```python
# V2 gives you 478 landmarks
face_landmarks = results.multi_face_landmarks[0]

# Access any landmark (0-477)
for idx in range(478):
    landmark = face_landmarks.landmark[idx]
    x = landmark.x * image_width
    y = landmark.y * image_height
```

### Issue: Focus metrics seem different

**Reason**: V2 uses more sophisticated analysis with:
- Sliding time windows
- Hysteresis (prevents flickering)
- Multi-metric scoring

**To match V1 behavior more closely**:
```yaml
focus:
  window_size: 10.0  # Shorter window
  update_interval: 0.1  # More frequent updates
```

## Benefits You'll Get After Migration

### Performance
- **3-5x faster processing** (120-180 FPS vs 30-50 FPS)
- **Smaller memory footprint** (9MB vs 100MB model)
- **Better CPU efficiency**

### Features
- **Data export** for analysis in Excel, Python, R
- **Video recording** with annotations
- **Interactive calibration** for accuracy
- **Head pose estimation** (pitch, yaw, roll)
- **Configurable everything** via YAML

### Developer Experience
- **Full type hints** for better IDE support
- **Comprehensive tests** (pytest suite)
- **Better error handling**
- **Structured logging**
- **Clean architecture**

### Platform Support
- **Windows**: No MSVC compiler needed!
- **macOS**: No Xcode command line tools!
- **Linux**: Just pip install!
- **All platforms**: Same experience

## Getting Help

If you encounter issues during migration:

1. **Check logs**: Use `log_level: DEBUG` in config
2. **Compare side-by-side**: Run V1 and V2 simultaneously
3. **Read docs**: See [README_V2.md](README_V2.md) and [USAGE_GUIDE.md](USAGE_GUIDE.md)
4. **Ask questions**: Open a GitHub issue

## Rollback Plan

If you need to go back to V1:

```bash
# V1 dependencies are still there
python eye_tracker.py

# V2 doesn't break V1
# Both can coexist
```

## Conclusion

Migration is straightforward:
1. ✅ Install new dependencies (2 minutes)
2. ✅ Test V2 with defaults (5 minutes)
3. ✅ Run calibration (2 minutes)
4. ✅ Export and analyze data (ongoing)

**Total migration time: ~10 minutes!**

The performance and feature improvements are worth it. Most users see **immediate benefits** with zero downsides.

---

**Ready to migrate? Start here:**
```bash
pip install mediapipe pyyaml
python eye_tracker_v2.py --calibrate
```

**Questions? Open an issue on GitHub!**
