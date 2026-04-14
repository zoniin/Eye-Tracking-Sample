# Eye Tracker V2 - Quick Reference Card

## Installation
```bash
pip install opencv-python numpy mediapipe pyyaml
python eye_tracker_v2.py
```

## Command Line

### Basic Usage
```bash
python eye_tracker_v2.py                          # Default camera
python eye_tracker_v2.py --source 1               # Camera 1
python eye_tracker_v2.py --source video.mp4       # Video file
```

### With Options
```bash
python eye_tracker_v2.py --config custom.yaml     # Custom config
python eye_tracker_v2.py --calibrate              # Start calibration
python eye_tracker_v2.py --export-data data.csv   # Export data
python eye_tracker_v2.py --record output.mp4      # Record video
```

### Configuration
```bash
python eye_tracker_v2.py --generate-config config.yaml  # Create config
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit and save |
| `R` | Reset session |
| `S` | Save data now |
| `D` | Toggle dashboard |
| `H` | Toggle help |
| `P` | Pause/Resume |
| `SPACE` | Calibration sample |
| `ESC` | Cancel calibration |

## Configuration Quick Reference

### Gaze Detection
```yaml
gaze:
  horizontal_center_min: 0.38  # Lower = more left
  horizontal_center_max: 0.62  # Higher = more right
  vertical_center_min: 0.35    # Lower = more up
  vertical_center_max: 0.65    # Higher = more down
```

### Blink Detection
```yaml
blink:
  ear_threshold: 0.21          # Lower = more sensitive
  consecutive_frames: 2        # Higher = less sensitive
```

### Camera Settings
```yaml
camera:
  source: 0                    # Camera index
  width: 1280                  # Resolution width
  height: 720                  # Resolution height
  fps: 30                      # Target FPS
```

### Performance Tuning
```yaml
# For speed (lower quality)
camera:
  width: 640
  height: 480
visualization:
  show_landmarks: false

# For accuracy (slower)
camera:
  width: 1920
  height: 1080
gaze:
  smoothing_window: 5
```

## Dashboard Metrics

| Metric | Meaning |
|--------|---------|
| **State** | focused / semi-focused / distracted / away |
| **Focus Score** | 0-100, higher = more focused |
| **FPS** | Processing speed |
| **Session** | Total session time |
| **Center Gaze %** | Time looking at center |
| **Blink Rate** | Blinks per minute |
| **Face Visible %** | Time face detected |
| **Head Pitch** | Up(+) / Down(-) degrees |
| **Head Yaw** | Right(+) / Left(-) degrees |
| **Head Roll** | Tilt degrees |

## Focus States

| State | Score | Description |
|-------|-------|-------------|
| **Focused** | 75-100 | High attention, center gaze |
| **Semi-focused** | 50-75 | Moderate attention |
| **Distracted** | 0-50 | Low attention, wandering gaze |
| **Away** | N/A | Face not visible |

## Blink Rate Guide

| Rate (per min) | Status | Action |
|----------------|--------|--------|
| < 10 | Very Low | Risk of dry eyes, blink more |
| 10-15 | Low | Consider blinking more |
| 15-30 | Normal | Healthy ✓ |
| 30-45 | High | Possible eye strain |
| > 45 | Very High | Take a break |

## Common Tasks

### Track Productivity Session
```bash
python eye_tracker_v2.py --export-data work_session.csv
```

### Calibrate for Accuracy
```bash
python eye_tracker_v2.py --calibrate
# Look at each marker, press SPACE to sample
```

### Record Demonstration
```bash
python eye_tracker_v2.py --record demo.mp4
```

### Analyze Exported Data
```bash
python examples/analyze_session.py session.csv --plot
```

## Data Export Format

CSV columns:
```
timestamp, gaze, blink_occurred, face_detected,
focus_score, focus_state, blink_count, fps,
head_pitch, head_yaw, head_roll
```

## Quick Python Analysis

```python
import pandas as pd

df = pd.read_csv('session.csv')

# Session duration
duration = (df['timestamp'].max() - df['timestamp'].min()) / 60
print(f"Duration: {duration:.1f} minutes")

# Focus percentage
focused = df[df['focus_state'] == 'focused']
print(f"Focused: {len(focused)/len(df)*100:.1f}%")

# Average metrics
print(f"Avg focus score: {df['focus_score'].mean():.1f}")
print(f"Avg blink rate: {df['blink_rate'].mean():.1f}/min")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Low FPS | Reduce resolution, disable landmarks |
| Inaccurate gaze | Run calibration |
| Too sensitive | Increase `window_size` in config |
| Not sensitive | Decrease `window_size` in config |
| No face detected | Improve lighting, move closer |
| High blink false positives | Lower `ear_threshold` |
| Missed blinks | Raise `ear_threshold` |

## Performance Optimization

### High Performance
```yaml
camera:
  width: 1280
  height: 720
  fps: 30
visualization:
  show_landmarks: false
  show_face_mesh: false
```

### Battery Saving
```yaml
camera:
  width: 640
  height: 480
  fps: 15
visualization:
  show_dashboard: false
```

## File Locations

```
Eye-Tracking-Sample/
├── eye_tracker_v2.py      # Main application
├── config.yaml             # Your configuration
├── output/                 # Exported data/videos
├── README_V2.md           # Full documentation
├── USAGE_GUIDE.md         # Detailed examples
└── examples/
    └── analyze_session.py  # Data analysis
```

## Getting Help

1. Press `H` during tracking for keyboard help
2. Check [USAGE_GUIDE.md](USAGE_GUIDE.md) for examples
3. Read [README_V2.md](README_V2.md) for full docs
4. Open GitHub issue for bugs

## Quick Tips

💡 **Always calibrate** when starting for best accuracy
💡 **Export data** to track productivity over time
💡 **Good lighting** is essential for accuracy
💡 **Eye-level camera** works best
💡 **Take breaks** - eye tracking is intensive
💡 **Review exports** to understand your patterns

## Version Info

| Version | File | Purpose |
|---------|------|---------|
| V2 | `eye_tracker_v2.py` | Recommended (MediaPipe) |
| V1 | `eye_tracker.py` | Legacy (dlib) |

**Use V2 for all new projects!**

---

Print this card or keep it handy for quick reference! 📄
