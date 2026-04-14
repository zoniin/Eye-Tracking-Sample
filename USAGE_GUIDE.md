# Eye Tracker V2 - Usage Guide

This guide provides practical examples and common use cases for the Eye Tracker V2.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Common Scenarios](#common-scenarios)
3. [Data Analysis Examples](#data-analysis-examples)
4. [Customization Tips](#customization-tips)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run basic tracker
python eye_tracker_v2.py

# 3. Calibrate for better accuracy (recommended)
python eye_tracker_v2.py --calibrate
```

### Understanding the Dashboard

When you run the tracker, you'll see:

```
┌─────────────────────────────┐
│   FOCUS DASHBOARD           │
├─────────────────────────────┤
│ State: FOCUSED              │ ← Your current focus state
│ Focus: 85/100               │ ← Overall focus score
│ FPS: 145.2                  │ ← Processing speed
│                             │
│ Session: 12:34              │ ← Total session time
│ Focused: 10:15 (81%)        │ ← Time in focused state
│                             │
│ Current Window:             │
│   Center gaze: 78%          │ ← % time looking center
│   Blink rate: 22/min        │ ← Blinks per minute
│   Face visible: 95%         │ ← % time face detected
│                             │
│ Head Pose:                  │
│   Pitch: 5.2°               │ ← Head tilt up/down
│   Yaw: -3.1°                │ ← Head turn left/right
│   Roll: 0.8°                │ ← Head tilt side-to-side
│                             │
│ [█████████░░] Gaze          │ ← Visual metric bars
│ [████████████] Blinks       │
│ [███████████░] Presence     │
└─────────────────────────────┘
```

## Common Scenarios

### 1. Track Your Work Session

Perfect for monitoring productivity during work or study:

```bash
# Start tracking with data export
python eye_tracker_v2.py --export-data work_session.csv

# Press 'S' to save data at any time
# Press 'Q' to quit and auto-save
```

**Tip**: Review the CSV later to see when you were most focused!

### 2. Monitor Student Engagement (E-learning)

Track attention during online lectures:

```bash
# Create custom config for classroom use
python eye_tracker_v2.py --generate-config classroom_config.yaml
```

Edit `classroom_config.yaml`:
```yaml
focus:
  window_size: 60.0  # Longer window for stable metrics
  gaze_threshold: 0.6  # More lenient threshold

export_data: true
export_path: "students/student_{timestamp}.csv"
```

Run:
```bash
python eye_tracker_v2.py --config classroom_config.yaml
```

### 3. Research Data Collection

Collect detailed eye tracking data for research:

```bash
# Export data + record video
python eye_tracker_v2.py \
  --export-data research/participant_01.csv \
  --record research/participant_01.mp4 \
  --config research_config.yaml
```

### 4. Screen Time Monitoring

Track how long you look at the screen:

```bash
# Run with visual feedback
python eye_tracker_v2.py --export-data screentime.csv
```

The `face_visible_ratio` metric tells you % time looking at screen.

### 5. Posture Monitoring

Use head pose data to monitor posture:

```bash
python eye_tracker_v2.py --export-data posture.csv
```

Later, analyze `head_pitch` values:
- Pitch > 15°: Looking up (screen too low)
- Pitch < -15°: Looking down (screen too high)
- Ideal: -5° to +10°

### 6. Blink Health Monitoring

Track blinking patterns (important for eye health):

```bash
python eye_tracker_v2.py --export-data blink_health.csv
```

Healthy blink rate: 15-30 blinks/minute
- Too low: Risk of dry eyes
- Too high: Possible eye strain

## Data Analysis Examples

### Analyze Session with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load exported data
df = pd.read_csv('work_session.csv')

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Plot focus score over time
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['focus_score'])
plt.xlabel('Time')
plt.ylabel('Focus Score')
plt.title('Focus Score Over Time')
plt.grid(True)
plt.show()

# Calculate statistics
print(f"Average focus score: {df['focus_score'].mean():.1f}")
print(f"Total blinks: {df['blink_count'].max()}")
print(f"Average blink rate: {df['blink_rate'].mean():.1f}/min")

# Find most focused period
focused_periods = df[df['focus_state'] == 'focused']
print(f"Total focused time: {len(focused_periods)} frames")
```

### Heatmap of Gaze Direction

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('session.csv')

# Count gaze directions
gaze_counts = df['gaze'].value_counts()

# Create heatmap
plt.figure(figsize=(8, 6))
sns.barplot(x=gaze_counts.index, y=gaze_counts.values)
plt.xlabel('Gaze Direction')
plt.ylabel('Count')
plt.title('Gaze Direction Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Blink Pattern Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('session.csv')

# Filter only blink events
blinks = df[df['blink_occurred'] == True]

# Calculate time between blinks
blinks['time_diff'] = blinks['timestamp'].diff()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(blinks['time_diff'], bins=50, edgecolor='black')
plt.xlabel('Time Between Blinks (seconds)')
plt.ylabel('Frequency')
plt.title('Blink Interval Distribution')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Average time between blinks: {blinks['time_diff'].mean():.2f}s")
```

## Customization Tips

### Optimize for Your Setup

#### Laptop Webcam (Close Range)
```yaml
gaze:
  horizontal_center_min: 0.35
  horizontal_center_max: 0.65
  vertical_center_min: 0.30
  vertical_center_max: 0.70
```

#### External Webcam (Far Range)
```yaml
gaze:
  horizontal_center_min: 0.40
  horizontal_center_max: 0.60
  vertical_center_min: 0.38
  vertical_center_max: 0.62
```

#### Reduce CPU Usage
```yaml
camera:
  width: 640
  height: 480
  fps: 15

visualization:
  show_landmarks: false
  show_eye_regions: false
```

#### Maximum Accuracy
```yaml
camera:
  width: 1920
  height: 1080
  fps: 30

gaze:
  smoothing_window: 5
```

### Create Presets

Create multiple config files for different scenarios:

**productivity.yaml** - For work tracking
```yaml
focus:
  window_size: 30.0
export_data: true
export_path: "productivity/session_{timestamp}.csv"
```

**research.yaml** - For detailed data collection
```yaml
camera:
  fps: 60  # High frame rate
export_data: true
recording:
  enabled: true
```

**demo.yaml** - For showing off
```yaml
visualization:
  show_landmarks: true
  show_face_mesh: true
  show_fps: true
camera:
  width: 1280
  height: 720
```

Use them:
```bash
python eye_tracker_v2.py --config productivity.yaml
python eye_tracker_v2.py --config research.yaml
python eye_tracker_v2.py --config demo.yaml
```

## Advanced Usage

### Batch Processing Videos

Process multiple video files:

```bash
# Create script: process_videos.sh
for video in videos/*.mp4; do
    echo "Processing $video"
    python eye_tracker_v2.py \
        --source "$video" \
        --export-data "output/$(basename "$video" .mp4).csv"
done
```

### Automated Session Logging

Create a wrapper script for automatic daily tracking:

```python
# daily_tracker.py
import subprocess
from datetime import datetime

# Generate filename with date
date_str = datetime.now().strftime('%Y-%m-%d')
export_file = f"daily_logs/{date_str}.csv"

# Run tracker
subprocess.run([
    'python', 'eye_tracker_v2.py',
    '--export-data', export_file,
    '--config', 'daily_config.yaml'
])
```

### Integration with Productivity Apps

Export and analyze with other tools:

```python
# export_to_toggl.py
import pandas as pd
import requests

df = pd.read_csv('session.csv')

# Calculate productive time
focused = df[df['focus_state'] == 'focused']
productive_minutes = len(focused) / (30 * 60)  # Assuming 30 FPS

# Send to Toggl (example)
# toggl_api.create_time_entry(
#     description="Focused work",
#     duration=productive_minutes * 60
# )
```

## Keyboard Shortcuts Cheat Sheet

```
┌──────────────────────────────────────────┐
│  KEYBOARD SHORTCUTS                      │
├──────────────────────────────────────────┤
│  Q       Quit and save                   │
│  R       Reset session statistics        │
│  S       Save data now                   │
│  D       Toggle dashboard                │
│  H       Toggle this help               │
│  P       Pause/Resume                    │
│  SPACE   Calibration sample              │
│  ESC     Cancel calibration              │
└──────────────────────────────────────────┘
```

## Troubleshooting

### Problem: Gaze direction is inaccurate

**Solution**: Run calibration
```bash
python eye_tracker_v2.py --calibrate
```

### Problem: Too sensitive (state changes too often)

**Solution**: Increase window size
```yaml
focus:
  window_size: 60.0  # Smoother, less reactive
```

### Problem: Not sensitive enough (slow to detect changes)

**Solution**: Decrease window size
```yaml
focus:
  window_size: 10.0  # More reactive
```

### Problem: Low FPS

**Solutions**:
1. Reduce resolution
2. Disable visual overlays
3. Close other applications
4. Update graphics drivers

### Problem: Blinks not detected

**Solution**: Adjust threshold
```yaml
blink:
  ear_threshold: 0.25  # More sensitive (higher value)
  consecutive_frames: 1  # Detect faster
```

### Problem: Too many false blink detections

**Solution**: Make less sensitive
```yaml
blink:
  ear_threshold: 0.18  # Less sensitive (lower value)
  consecutive_frames: 3  # Require longer closure
```

## Best Practices

### For Accuracy
1. **Always calibrate** when starting
2. **Maintain consistent lighting**
3. **Position camera at eye level**
4. **Keep head relatively still**
5. **Minimize background movement**

### For Performance
1. **Use appropriate resolution** (1280x720 is good balance)
2. **Disable unnecessary visualizations**
3. **Export data to SSD** not HDD
4. **Close background applications**

### For Data Quality
1. **Run calibration at start of each session**
2. **Take breaks** (eye tracking is intensive)
3. **Note environmental changes** in separate log
4. **Validate data** with known focus periods
5. **Export regularly** to avoid data loss

## Tips & Tricks

### Maximize Battery Life
```yaml
camera:
  fps: 15  # Lower FPS
  width: 640
  height: 480
visualization:
  show_dashboard: false  # Less rendering
```

### Night Mode (Low Light)
```yaml
blink:
  ear_threshold: 0.23  # More sensitive
visualization:
  show_landmarks: false  # Less distraction
```

### Privacy Mode
```yaml
recording:
  enabled: false
export_data: false
log_file: null
```

## Next Steps

- Read [README_V2.md](README_V2.md) for complete documentation
- Check [config.yaml](config.yaml) for all configuration options
- Run tests: `pytest tests/`
- Join discussions on GitHub

---

**Happy tracking! 👁️**
