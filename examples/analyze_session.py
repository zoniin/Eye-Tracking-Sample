"""
Example script for analyzing eye tracking session data.

Usage:
    python analyze_session.py session.csv
    python analyze_session.py session.csv --plot
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def load_session_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare session data."""
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive session statistics."""
    stats = {}

    # Session duration
    duration_seconds = df['timestamp'].max() - df['timestamp'].min()
    stats['duration_minutes'] = duration_seconds / 60

    # Focus metrics
    focus_states = df['focus_state'].value_counts()
    total_frames = len(df)

    stats['focused_percent'] = (focus_states.get('focused', 0) / total_frames) * 100
    stats['semi_focused_percent'] = (focus_states.get('semi-focused', 0) / total_frames) * 100
    stats['distracted_percent'] = (focus_states.get('distracted', 0) / total_frames) * 100
    stats['away_percent'] = (focus_states.get('away', 0) / total_frames) * 100

    # Average focus score
    stats['avg_focus_score'] = df['focus_score'].mean()
    stats['max_focus_score'] = df['focus_score'].max()
    stats['min_focus_score'] = df['focus_score'].min()

    # Blink metrics
    stats['total_blinks'] = df['blink_count'].max()
    stats['avg_blink_rate'] = df[df['blink_rate'] > 0]['blink_rate'].mean()

    # Gaze distribution
    gaze_counts = df['gaze'].value_counts()
    stats['center_gaze_percent'] = (gaze_counts.get('center', 0) / total_frames) * 100

    # Head pose metrics
    stats['avg_head_pitch'] = df['head_pitch'].mean()
    stats['avg_head_yaw'] = df['head_yaw'].mean()
    stats['avg_head_roll'] = df['head_roll'].mean()

    # Performance
    stats['avg_fps'] = df['fps'].mean()

    return stats


def print_report(stats: dict, df: pd.DataFrame):
    """Print a formatted report."""
    print("\n" + "="*60)
    print("EYE TRACKING SESSION REPORT")
    print("="*60)

    print("\n📊 SESSION OVERVIEW")
    print(f"  Duration: {stats['duration_minutes']:.1f} minutes")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Total Frames: {len(df):,}")

    print("\n🎯 FOCUS ANALYSIS")
    print(f"  Average Focus Score: {stats['avg_focus_score']:.1f}/100")
    print(f"  Peak Focus Score: {stats['max_focus_score']:.1f}/100")
    print(f"  Lowest Focus Score: {stats['min_focus_score']:.1f}/100")

    print("\n  Time Distribution:")
    print(f"    Focused:      {stats['focused_percent']:.1f}%")
    print(f"    Semi-focused: {stats['semi_focused_percent']:.1f}%")
    print(f"    Distracted:   {stats['distracted_percent']:.1f}%")
    print(f"    Away:         {stats['away_percent']:.1f}%")

    print("\n👁️ GAZE ANALYSIS")
    print(f"  Center Gaze: {stats['center_gaze_percent']:.1f}%")

    gaze_distribution = df['gaze'].value_counts()
    print(f"\n  Gaze Distribution:")
    for gaze, count in gaze_distribution.head(5).items():
        percent = (count / len(df)) * 100
        print(f"    {gaze:12s}: {percent:5.1f}%")

    print("\n👀 BLINK ANALYSIS")
    print(f"  Total Blinks: {stats['total_blinks']}")
    print(f"  Average Blink Rate: {stats['avg_blink_rate']:.1f} blinks/minute")

    if stats['avg_blink_rate'] < 15:
        print(f"  ⚠️  Warning: Low blink rate (risk of dry eyes)")
    elif stats['avg_blink_rate'] > 30:
        print(f"  ⚠️  Warning: High blink rate (possible eye strain)")
    else:
        print(f"  ✅ Healthy blink rate")

    print("\n🧑 HEAD POSE ANALYSIS")
    print(f"  Average Pitch: {stats['avg_head_pitch']:+.1f}° ", end="")
    if abs(stats['avg_head_pitch']) > 15:
        print("⚠️  (check screen height)")
    else:
        print("✅")

    print(f"  Average Yaw:   {stats['avg_head_yaw']:+.1f}° ", end="")
    if abs(stats['avg_head_yaw']) > 15:
        print("⚠️  (check screen position)")
    else:
        print("✅")

    print(f"  Average Roll:  {stats['avg_head_roll']:+.1f}°")

    print("\n💡 INSIGHTS")

    # Productivity assessment
    if stats['focused_percent'] > 70:
        print("  ✅ Excellent focus session!")
    elif stats['focused_percent'] > 50:
        print("  👍 Good focus, room for improvement")
    else:
        print("  ⚠️  Low focus detected - consider taking breaks")

    # Break recommendation
    if stats['duration_minutes'] > 50 and stats['avg_blink_rate'] < 15:
        print("  💡 Consider taking a break to rest your eyes")

    # Posture check
    if abs(stats['avg_head_pitch']) > 15 or abs(stats['avg_head_yaw']) > 15:
        print("  💡 Check your screen position and posture")

    print("\n" + "="*60 + "\n")


def plot_session(df: pd.DataFrame, output_file: str = None):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Eye Tracking Session Analysis', fontsize=16, fontweight='bold')

    # 1. Focus Score Over Time
    ax = axes[0, 0]
    ax.plot(df['datetime'], df['focus_score'], linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Focus Score')
    ax.set_title('Focus Score Over Time')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 2. Focus State Distribution
    ax = axes[0, 1]
    focus_counts = df['focus_state'].value_counts()
    colors = {'focused': 'green', 'semi-focused': 'yellow',
              'distracted': 'orange', 'away': 'red', 'unknown': 'gray'}
    ax.bar(focus_counts.index, focus_counts.values,
           color=[colors.get(state, 'gray') for state in focus_counts.index])
    ax.set_xlabel('Focus State')
    ax.set_ylabel('Frame Count')
    ax.set_title('Focus State Distribution')
    ax.tick_params(axis='x', rotation=45)

    # 3. Gaze Direction Heatmap
    ax = axes[1, 0]
    gaze_counts = df['gaze'].value_counts().head(10)
    ax.barh(gaze_counts.index, gaze_counts.values)
    ax.set_xlabel('Count')
    ax.set_ylabel('Gaze Direction')
    ax.set_title('Gaze Direction Distribution')

    # 4. Blink Rate Over Time
    ax = axes[1, 1]
    # Resample to 1-minute intervals for clearer visualization
    df_resampled = df.set_index('datetime').resample('1T')['blink_rate'].mean()
    ax.plot(df_resampled.index, df_resampled.values, marker='o', markersize=3)
    ax.axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Normal range')
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Blinks per Minute')
    ax.set_title('Blink Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 5. Head Pose (Pitch/Yaw)
    ax = axes[2, 0]
    ax.scatter(df['head_yaw'], df['head_pitch'], alpha=0.3, s=1)
    ax.set_xlabel('Yaw (left/right)')
    ax.set_ylabel('Pitch (up/down)')
    ax.set_title('Head Pose Distribution')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    ax.grid(True, alpha=0.3)

    # 6. FPS Over Time
    ax = axes[2, 1]
    ax.plot(df['datetime'], df['fps'], linewidth=0.5, alpha=0.7, color='purple')
    ax.set_xlabel('Time')
    ax.set_ylabel('FPS')
    ax.set_title('Processing Performance')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def export_summary(stats: dict, output_file: str):
    """Export summary statistics to JSON."""
    import json

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Summary exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze eye tracking session data'
    )
    parser.add_argument('csv_file', help='Path to CSV session file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot-output', help='Save plot to file instead of showing')
    parser.add_argument('--export-summary', help='Export summary to JSON file')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.csv_file}...")
    df = load_session_data(args.csv_file)
    print(f"Loaded {len(df)} frames")

    # Calculate statistics
    stats = calculate_statistics(df)

    # Print report
    print_report(stats, df)

    # Export summary if requested
    if args.export_summary:
        export_summary(stats, args.export_summary)

    # Plot if requested
    if args.plot or args.plot_output:
        plot_session(df, args.plot_output)


if __name__ == '__main__':
    main()
