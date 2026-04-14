# Examples

This directory contains example scripts and usage patterns for Eye Tracker V2.

## Files

### analyze_session.py
Comprehensive session data analysis script.

**Features:**
- Load and analyze CSV session data
- Calculate focus statistics
- Generate visualizations
- Export summary reports

**Usage:**
```bash
# Basic analysis
python analyze_session.py session.csv

# With plots
python analyze_session.py session.csv --plot

# Save plots to file
python analyze_session.py session.csv --plot-output analysis.png

# Export summary
python analyze_session.py session.csv --export-summary summary.json
```

**Requirements:**
```bash
pip install pandas matplotlib
```

## More Examples

### 1. Simple Data Loading
```python
import pandas as pd

# Load session data
df = pd.read_csv('session.csv')

# Basic stats
print(f"Session duration: {len(df) / 30 / 60:.1f} minutes")  # Assuming 30 FPS
print(f"Total blinks: {df['blink_count'].max()}")
print(f"Average focus score: {df['focus_score'].mean():.1f}")
```

### 2. Focus Time Calculation
```python
import pandas as pd

df = pd.read_csv('session.csv')

# Calculate time in each state
focused_frames = len(df[df['focus_state'] == 'focused'])
total_frames = len(df)
fps = df['fps'].mean()

focused_minutes = focused_frames / fps / 60
total_minutes = total_frames / fps / 60

print(f"Focused: {focused_minutes:.1f}/{total_minutes:.1f} minutes")
print(f"Focus percentage: {(focused_frames/total_frames)*100:.1f}%")
```

### 3. Find Peak Focus Period
```python
import pandas as pd

df = pd.read_csv('session.csv')
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Find longest continuous focus period
df['focus_change'] = df['focus_state'].ne(df['focus_state'].shift())
df['focus_group'] = df['focus_change'].cumsum()

focused_periods = df[df['focus_state'] == 'focused'].groupby('focus_group')

for group_id, group in focused_periods:
    duration = len(group) / 30  # Assuming 30 FPS
    if duration > 60:  # More than 1 minute
        start_time = group['datetime'].iloc[0]
        print(f"Focus period: {start_time.strftime('%H:%M:%S')} - {duration/60:.1f} min")
```

### 4. Blink Pattern Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('session.csv')

# Get blink events
blinks = df[df['blink_occurred'] == True]

# Calculate inter-blink intervals
blinks['time_since_last'] = blinks['timestamp'].diff()

# Plot histogram
plt.hist(blinks['time_since_last'].dropna(), bins=50)
plt.xlabel('Seconds Between Blinks')
plt.ylabel('Frequency')
plt.title('Blink Interval Distribution')
plt.show()
```

### 5. Head Posture Analysis
```python
import pandas as pd
import numpy as np

df = pd.read_csv('session.csv')

# Analyze head pose
avg_pitch = df['head_pitch'].mean()
avg_yaw = df['head_yaw'].mean()

print(f"Average head pitch: {avg_pitch:.1f}°")
print(f"Average head yaw: {avg_yaw:.1f}°")

# Check if posture is good
if abs(avg_pitch) > 15:
    print("⚠️  Screen may be too high/low")
else:
    print("✅ Good vertical posture")

if abs(avg_yaw) > 15:
    print("⚠️  Screen may be off-center")
else:
    print("✅ Good horizontal posture")
```

### 6. Export for Spreadsheet
```python
import pandas as pd

df = pd.read_csv('session.csv')

# Create summary by minute
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['minute'] = df['datetime'].dt.floor('T')

summary = df.groupby('minute').agg({
    'focus_score': 'mean',
    'blink_occurred': 'sum',
    'face_detected': lambda x: (x.sum() / len(x)) * 100
}).round(2)

summary.columns = ['Avg Focus Score', 'Blinks', 'Face Visible %']
summary.to_csv('summary_by_minute.csv')

print("Summary exported to summary_by_minute.csv")
```

### 7. Real-time Monitoring (Advanced)
```python
import pandas as pd
import time
from pathlib import Path

def monitor_session(csv_path, update_interval=5):
    """Monitor a live recording session."""
    last_size = 0

    while True:
        try:
            df = pd.read_csv(csv_path)

            if len(df) > last_size:
                # New data available
                recent = df.tail(30 * update_interval)  # Last N seconds

                focus_score = recent['focus_score'].mean()
                blink_rate = recent[recent['blink_rate'] > 0]['blink_rate'].mean()

                print(f"\rFocus: {focus_score:.0f}/100 | "
                      f"Blinks: {blink_rate:.0f}/min | "
                      f"Frames: {len(df)}", end='')

                last_size = len(df)

            time.sleep(update_interval)
        except FileNotFoundError:
            print(f"Waiting for {csv_path}...")
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break

# Usage
# monitor_session('output/session.csv')
```

## Creating Custom Analysis

Template for your own analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_custom(csv_path):
    """Your custom analysis."""
    # Load data
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Your analysis here
    # ...

    # Generate plots
    plt.figure(figsize=(12, 6))

    # Your plots here
    # ...

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyze_custom('session.csv')
```

## Tips

1. **Data Quality**: Always check for missing values and outliers
2. **Sampling Rate**: Remember FPS varies - use timestamps for accurate time
3. **Smoothing**: Consider rolling averages for noisy metrics
4. **Normalization**: Compare sessions by normalizing to duration
5. **Visualization**: Use appropriate plot types for each metric

## Next Steps

- Combine multiple sessions for longitudinal analysis
- Build a dashboard with real-time updates
- Integrate with productivity tools
- Create automated reports
- Machine learning for pattern detection

## Questions?

Check the main documentation:
- [README_V2.md](../README_V2.md) - Full feature documentation
- [USAGE_GUIDE.md](../USAGE_GUIDE.md) - Practical usage examples
- [GitHub Issues](https://github.com/zoniin/Eye-Tracking-Sample/issues) - Ask questions
