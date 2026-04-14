"""
Advanced Eye Tracking System with MediaPipe
Tracks gaze direction, focus state, blinks, and head pose in real-time.

Features:
- MediaPipe Face Mesh for ultra-fast tracking (120-180 FPS)
- YAML configuration support
- Data export (CSV/JSON)
- Interactive calibration mode
- Video recording with annotations
- Head pose estimation
- Comprehensive logging
- Keyboard shortcuts for control

Usage:
    python eye_tracker_v2.py
    python eye_tracker_v2.py --config custom_config.yaml
    python eye_tracker_v2.py --source 0 --calibrate
    python eye_tracker_v2.py --source video.mp4 --export-data session.csv
"""

import argparse
import sys
import time
import json
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict, field
from collections import deque
from datetime import datetime
import logging

import cv2
import numpy as np
import mediapipe as mp
import yaml


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with console and optional file output."""
    logger = logging.getLogger("EyeTracker")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Configuration Management
# ---------------------------------------------------------------------------

@dataclass
class GazeConfig:
    """Gaze detection configuration."""
    horizontal_center_min: float = 0.38
    horizontal_center_max: float = 0.62
    vertical_center_min: float = 0.35
    vertical_center_max: float = 0.65
    smoothing_window: int = 3


@dataclass
class BlinkConfig:
    """Blink detection configuration."""
    ear_threshold: float = 0.21
    consecutive_frames: int = 2
    normal_rate_min: int = 15
    normal_rate_max: int = 30


@dataclass
class FocusConfig:
    """Focus tracking configuration."""
    window_size: float = 30.0
    gaze_threshold: float = 0.7
    face_loss_threshold: float = 0.2
    update_interval: float = 0.5


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    show_landmarks: bool = True
    show_face_mesh: bool = False
    show_eye_regions: bool = True
    show_gaze_arrow: bool = True
    show_dashboard: bool = True
    show_fps: bool = True
    dashboard_width: int = 300


@dataclass
class CameraConfig:
    """Camera and video settings."""
    source: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class RecordingConfig:
    """Video recording settings."""
    enabled: bool = False
    output_path: str = "output/recording_{timestamp}.mp4"
    codec: str = "mp4v"
    fps: int = 30


@dataclass
class AppConfig:
    """Main application configuration."""
    gaze: GazeConfig = field(default_factory=GazeConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    log_level: str = "INFO"
    log_file: Optional[str] = None
    export_data: bool = False
    export_path: str = "output/session_{timestamp}.csv"

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            gaze=GazeConfig(**data.get('gaze', {})),
            blink=BlinkConfig(**data.get('blink', {})),
            focus=FocusConfig(**data.get('focus', {})),
            visualization=VisualizationConfig(**data.get('visualization', {})),
            camera=CameraConfig(**data.get('camera', {})),
            recording=RecordingConfig(**data.get('recording', {})),
            log_level=data.get('log_level', 'INFO'),
            log_file=data.get('log_file'),
            export_data=data.get('export_data', False),
            export_path=data.get('export_path', 'output/session_{timestamp}.csv')
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'gaze': asdict(self.gaze),
            'blink': asdict(self.blink),
            'focus': asdict(self.focus),
            'visualization': asdict(self.visualization),
            'camera': asdict(self.camera),
            'recording': asdict(self.recording),
            'log_level': self.log_level,
            'log_file': self.log_file,
            'export_data': self.export_data,
            'export_path': self.export_path
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HeadPose:
    """Head pose estimation results."""
    pitch: float = 0.0  # Up/down rotation
    yaw: float = 0.0    # Left/right rotation
    roll: float = 0.0   # Tilt rotation


@dataclass
class FocusMetrics:
    """Container for analyzed focus metrics."""
    center_gaze_ratio: float = 0.0
    dominant_gaze: str = "unknown"
    gaze_switches: int = 0
    blink_rate: float = 0.0
    blink_rate_status: str = "normal"
    face_visible_ratio: float = 0.0
    face_loss_count: int = 0
    focus_score: float = 0.0
    state: str = "unknown"
    head_pose: HeadPose = field(default_factory=HeadPose)


@dataclass
class FrameData:
    """Data captured from a single frame."""
    timestamp: float
    gaze: str
    blink_occurred: bool
    face_detected: bool
    focus_score: float
    focus_state: str
    blink_count: int
    head_pose: HeadPose
    fps: float = 0.0


# ---------------------------------------------------------------------------
# MediaPipe Facial Landmark Indices
# ---------------------------------------------------------------------------

# Left eye landmarks (MediaPipe Face Mesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]

# Right eye landmarks
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Face oval for head pose
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Nose tip, chin, left/right eye corners for head pose
HEAD_POSE_INDICES = {
    'nose_tip': 1,
    'chin': 152,
    'left_eye': 33,
    'right_eye': 263,
    'left_mouth': 61,
    'right_mouth': 291
}


# ---------------------------------------------------------------------------
# Eye Tracking Core Functions
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.

    Args:
        eye_landmarks: Array of 6 eye landmark points (x, y)

    Returns:
        EAR value (typically 0.2-0.3 when open, <0.2 when closed)
    """
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def get_iris_position(iris_landmarks: np.ndarray, eye_landmarks: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate normalized iris position within the eye.

    Args:
        iris_landmarks: Array of iris landmark points
        eye_landmarks: Array of eye contour landmarks

    Returns:
        (cx, cy) in [0, 1] range, or None if calculation fails
    """
    if len(iris_landmarks) == 0 or len(eye_landmarks) == 0:
        return None

    # Get iris center
    iris_center = np.mean(iris_landmarks, axis=0)

    # Get eye bounding box
    eye_min = np.min(eye_landmarks, axis=0)
    eye_max = np.max(eye_landmarks, axis=0)
    eye_size = eye_max - eye_min

    if eye_size[0] == 0 or eye_size[1] == 0:
        return None

    # Normalize position
    normalized_pos = (iris_center - eye_min) / eye_size

    return tuple(normalized_pos)


def classify_gaze(left_pos: Optional[Tuple[float, float]],
                  right_pos: Optional[Tuple[float, float]],
                  config: GazeConfig) -> str:
    """
    Classify gaze direction from normalized iris positions.

    Args:
        left_pos: Left iris position (cx, cy) in [0,1] or None
        right_pos: Right iris position (cx, cy) in [0,1] or None
        config: Gaze configuration with thresholds

    Returns:
        Gaze direction: "center", "left", "right", "up", "down", or diagonals
    """
    positions = [p for p in (left_pos, right_pos) if p is not None]
    if not positions:
        return "undetected"

    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])

    # Horizontal direction
    if cx < config.horizontal_center_min:
        h_dir = "right"
    elif cx > config.horizontal_center_max:
        h_dir = "left"
    else:
        h_dir = "center"

    # Vertical direction
    if cy < config.vertical_center_min:
        v_dir = "up"
    elif cy > config.vertical_center_max:
        v_dir = "down"
    else:
        v_dir = ""

    # Combine directions
    if h_dir == "center" and not v_dir:
        return "center"
    if v_dir and h_dir == "center":
        return v_dir
    if v_dir:
        return f"{v_dir}-{h_dir}"
    return h_dir


def estimate_head_pose(face_landmarks: np.ndarray,
                       image_shape: Tuple[int, int],
                       camera_matrix: np.ndarray) -> HeadPose:
    """
    Estimate head pose (pitch, yaw, roll) from facial landmarks.

    Args:
        face_landmarks: Facial landmark coordinates
        image_shape: (height, width) of the image
        camera_matrix: Camera intrinsic matrix

    Returns:
        HeadPose with pitch, yaw, roll in degrees
    """
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0, 170.0, -135.0),    # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)    # Right mouth corner
    ], dtype=np.float64)

    # 2D image points from landmarks
    indices = [HEAD_POSE_INDICES[key] for key in
               ['nose_tip', 'chin', 'left_eye', 'right_eye', 'left_mouth', 'right_mouth']]

    image_points = np.array([
        face_landmarks[i][:2] for i in indices
    ], dtype=np.float64)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return HeadPose()

    # Convert rotation vector to Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Calculate Euler angles
    sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = np.arctan2(-rotation_mat[2, 0], sy)
        roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
    else:
        pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        yaw = np.arctan2(-rotation_mat[2, 0], sy)
        roll = 0

    # Convert to degrees
    return HeadPose(
        pitch=np.degrees(pitch),
        yaw=np.degrees(yaw),
        roll=np.degrees(roll)
    )


# ---------------------------------------------------------------------------
# Focus Analyzer
# ---------------------------------------------------------------------------

class FocusAnalyzer:
    """
    Analyzes temporal eye tracking metrics to determine focus/distraction state.
    """

    def __init__(self, config: FocusConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Temporal data storage
        self.gaze_history: List[Tuple[float, str]] = []
        self.blink_timestamps: List[float] = []
        self.face_detected_history: List[Tuple[float, bool]] = []

        # Session statistics
        self.session_start: Optional[float] = None
        self.total_focused_time: float = 0.0
        self.total_distracted_time: float = 0.0
        self.current_state: str = "unknown"
        self.state_change_time: Optional[float] = None

        # Cache
        self.cached_metrics: Optional[FocusMetrics] = None
        self.last_analysis_time: float = 0

    def start_session(self) -> None:
        """Start a new tracking session."""
        self.session_start = time.time()
        self.logger.info("Focus tracking session started")

    def update(self, timestamp: float, gaze: str, face_detected: bool, blink_occurred: bool) -> None:
        """Update temporal buffers with new frame data."""
        self.gaze_history.append((timestamp, gaze))

        if blink_occurred:
            self.blink_timestamps.append(timestamp)

        self.face_detected_history.append((timestamp, face_detected))

        # Prune old data
        cutoff_time = timestamp - self.config.window_size
        self.gaze_history = [(t, g) for t, g in self.gaze_history if t >= cutoff_time]
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff_time]
        self.face_detected_history = [(t, f) for t, f in self.face_detected_history if t >= cutoff_time]

    def compute_metrics(self, current_time: float, head_pose: HeadPose) -> FocusMetrics:
        """Analyze time window and compute all metrics."""
        # Check cache
        if (current_time - self.last_analysis_time) < self.config.update_interval and self.cached_metrics:
            return self.cached_metrics

        metrics = FocusMetrics(head_pose=head_pose)

        # Gaze metrics
        if self.gaze_history:
            center_count = sum(1 for _, gaze in self.gaze_history if gaze == "center")
            metrics.center_gaze_ratio = center_count / len(self.gaze_history)

            gaze_counts = {}
            for _, gaze in self.gaze_history:
                gaze_counts[gaze] = gaze_counts.get(gaze, 0) + 1
            metrics.dominant_gaze = max(gaze_counts, key=gaze_counts.get)

            prev_gaze = None
            for _, gaze in self.gaze_history:
                if prev_gaze is not None and gaze != prev_gaze:
                    metrics.gaze_switches += 1
                prev_gaze = gaze

        # Blink metrics
        if self.blink_timestamps:
            window_duration_minutes = self.config.window_size / 60.0
            metrics.blink_rate = len(self.blink_timestamps) / window_duration_minutes

            if metrics.blink_rate < 10:
                metrics.blink_rate_status = "very low"
            elif metrics.blink_rate < 15:
                metrics.blink_rate_status = "low"
            elif metrics.blink_rate <= 30:
                metrics.blink_rate_status = "normal"
            elif metrics.blink_rate <= 45:
                metrics.blink_rate_status = "high"
            else:
                metrics.blink_rate_status = "very high"

        # Face detection metrics
        if self.face_detected_history:
            face_visible_count = sum(1 for _, detected in self.face_detected_history if detected)
            metrics.face_visible_ratio = face_visible_count / len(self.face_detected_history)

            prev_detected = None
            for _, detected in self.face_detected_history:
                if prev_detected is not None and prev_detected and not detected:
                    metrics.face_loss_count += 1
                prev_detected = detected

        # Compute scores
        gaze_score = metrics.center_gaze_ratio * 100.0
        blink_score = self._compute_blink_score(metrics.blink_rate)
        presence_score = self._compute_presence_score(metrics.face_visible_ratio)

        # Weighted focus score
        metrics.focus_score = 0.5 * gaze_score + 0.2 * blink_score + 0.3 * presence_score

        # State classification with hysteresis
        new_state = self._classify_state(metrics.focus_score, presence_score)

        if new_state != self.current_state:
            if self.state_change_time is None:
                self.state_change_time = current_time
            elif (current_time - self.state_change_time) >= 3.0:
                time_in_state = current_time - self.state_change_time
                if self.current_state == "focused":
                    self.total_focused_time += time_in_state
                elif self.current_state in ("distracted", "semi-focused"):
                    self.total_distracted_time += time_in_state

                self.logger.info(f"Focus state changed: {self.current_state} -> {new_state}")
                self.current_state = new_state
                self.state_change_time = current_time
        else:
            self.state_change_time = None

        metrics.state = self.current_state

        # Cache
        self.cached_metrics = metrics
        self.last_analysis_time = current_time

        return metrics

    def _compute_blink_score(self, blink_rate: float) -> float:
        """Calculate blink-rate-based score (0-100)."""
        if 15 <= blink_rate <= 30:
            return 100.0
        elif blink_rate < 10:
            return 70.0
        elif blink_rate < 15:
            return 80.0
        elif blink_rate <= 45:
            return 60.0
        else:
            return 30.0

    def _compute_presence_score(self, ratio: float) -> float:
        """Calculate face-presence score (0-100)."""
        if ratio >= 0.8:
            return ratio * 100.0
        elif ratio >= 0.5:
            return 50.0
        else:
            return 0.0

    def _classify_state(self, focus_score: float, presence_score: float) -> str:
        """Determine final focus state from scores."""
        if presence_score < 20:
            return "away"
        elif focus_score >= 75:
            return "focused"
        elif focus_score >= 50:
            return "semi-focused"
        else:
            return "distracted"

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the tracking session."""
        if not self.session_start:
            return {}

        total_time = time.time() - self.session_start
        return {
            'total_duration': total_time,
            'focused_time': self.total_focused_time,
            'distracted_time': self.total_distracted_time,
            'focused_percentage': (self.total_focused_time / total_time * 100) if total_time > 0 else 0,
            'distracted_percentage': (self.total_distracted_time / total_time * 100) if total_time > 0 else 0
        }


# ---------------------------------------------------------------------------
# Data Exporter
# ---------------------------------------------------------------------------

class DataExporter:
    """Handles exporting session data to CSV/JSON."""

    def __init__(self, export_path: str, logger: logging.Logger):
        self.export_path = export_path.replace('{timestamp}',
                                                datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.logger = logger
        self.data_buffer: List[FrameData] = []

        # Create output directory
        Path(self.export_path).parent.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame_data: FrameData) -> None:
        """Add frame data to export buffer."""
        self.data_buffer.append(frame_data)

    def export_csv(self) -> None:
        """Export data to CSV file."""
        if not self.data_buffer:
            self.logger.warning("No data to export")
            return

        try:
            with open(self.export_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'timestamp', 'gaze', 'blink_occurred', 'face_detected',
                    'focus_score', 'focus_state', 'blink_count', 'fps',
                    'head_pitch', 'head_yaw', 'head_roll'
                ])

                # Data rows
                for data in self.data_buffer:
                    writer.writerow([
                        data.timestamp,
                        data.gaze,
                        data.blink_occurred,
                        data.face_detected,
                        data.focus_score,
                        data.focus_state,
                        data.blink_count,
                        data.fps,
                        data.head_pose.pitch,
                        data.head_pose.yaw,
                        data.head_pose.roll
                    ])

            self.logger.info(f"Data exported to {self.export_path}")
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")

    def export_json(self) -> None:
        """Export data to JSON file."""
        if not self.data_buffer:
            self.logger.warning("No data to export")
            return

        json_path = self.export_path.replace('.csv', '.json')

        try:
            with open(json_path, 'w') as f:
                data = [asdict(frame) for frame in self.data_buffer]
                json.dump(data, f, indent=2)

            self.logger.info(f"Data exported to {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")


# ---------------------------------------------------------------------------
# Calibration Mode
# ---------------------------------------------------------------------------

class CalibrationManager:
    """Interactive calibration for personalized gaze thresholds."""

    def __init__(self, config: GazeConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.calibration_points = [
            ('center', 0.5, 0.5),
            ('left', 0.1, 0.5),
            ('right', 0.9, 0.5),
            ('up', 0.5, 0.1),
            ('down', 0.5, 0.9)
        ]
        self.current_point_idx = 0
        self.samples: Dict[str, List[Tuple[float, float]]] = {name: [] for name, _, _ in self.calibration_points}
        self.is_active = False
        self.sample_count = 0
        self.samples_per_point = 30

    def start(self) -> None:
        """Start calibration process."""
        self.is_active = True
        self.current_point_idx = 0
        self.sample_count = 0
        self.logger.info("Calibration started. Look at the markers and press SPACE to sample.")

    def get_current_point(self) -> Tuple[str, float, float]:
        """Get current calibration point."""
        return self.calibration_points[self.current_point_idx]

    def add_sample(self, left_pos: Optional[Tuple[float, float]],
                   right_pos: Optional[Tuple[float, float]]) -> bool:
        """
        Add calibration sample.

        Returns:
            True if calibration is complete
        """
        positions = [p for p in (left_pos, right_pos) if p is not None]
        if not positions:
            return False

        name, _, _ = self.get_current_point()
        avg_pos = (
            np.mean([p[0] for p in positions]),
            np.mean([p[1] for p in positions])
        )
        self.samples[name].append(avg_pos)
        self.sample_count += 1

        if self.sample_count >= self.samples_per_point:
            self.current_point_idx += 1
            self.sample_count = 0

            if self.current_point_idx >= len(self.calibration_points):
                self.finish()
                return True

            name, _, _ = self.get_current_point()
            self.logger.info(f"Next point: {name}")

        return False

    def finish(self) -> None:
        """Finish calibration and compute thresholds."""
        self.is_active = False

        # Calculate thresholds from samples
        if 'center' in self.samples and len(self.samples['center']) > 0:
            center_x = [p[0] for p in self.samples['center']]
            center_y = [p[1] for p in self.samples['center']]

            # Use standard deviation to set thresholds
            self.config.horizontal_center_min = np.mean(center_x) - np.std(center_x)
            self.config.horizontal_center_max = np.mean(center_x) + np.std(center_x)
            self.config.vertical_center_min = np.mean(center_y) - np.std(center_y)
            self.config.vertical_center_max = np.mean(center_y) + np.std(center_y)

        self.logger.info("Calibration complete!")
        self.logger.info(f"Horizontal center: {self.config.horizontal_center_min:.2f} - {self.config.horizontal_center_max:.2f}")
        self.logger.info(f"Vertical center: {self.config.vertical_center_min:.2f} - {self.config.vertical_center_max:.2f}")

    def draw(self, frame: np.ndarray) -> None:
        """Draw calibration UI."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Instructions
        instructions = [
            "CALIBRATION MODE",
            "",
            f"Look at the GREEN marker",
            f"Press SPACE to capture samples ({self.sample_count}/{self.samples_per_point})",
            f"Point {self.current_point_idx + 1}/{len(self.calibration_points)}: {self.get_current_point()[0].upper()}",
            "",
            "Press ESC to cancel"
        ]

        y_offset = 50
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (50, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw all calibration points
        for i, (name, x, y) in enumerate(self.calibration_points):
            px, py = int(x * w), int(y * h)
            if i == self.current_point_idx:
                cv2.circle(frame, (px, py), 20, (0, 255, 0), -1)
                cv2.circle(frame, (px, py), 25, (255, 255, 255), 2)
            else:
                cv2.circle(frame, (px, py), 10, (100, 100, 100), -1)


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def get_state_color(state: str) -> Tuple[int, int, int]:
    """Return BGR color for focus state."""
    color_map = {
        "focused": (0, 255, 0),
        "semi-focused": (0, 200, 200),
        "distracted": (0, 165, 255),
        "away": (0, 0, 255),
        "unknown": (150, 150, 150),
    }
    return color_map.get(state, (150, 150, 150))


def draw_gaze_arrow(frame: np.ndarray, gaze: str, origin: Tuple[int, int], length: int = 60) -> None:
    """Draw directional gaze arrow."""
    direction_map = {
        "center": (0, 0),
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
        "up-left": (-1, -1),
        "up-right": (1, -1),
        "down-left": (-1, 1),
        "down-right": (1, 1),
    }

    if gaze not in direction_map:
        return

    dx, dy = direction_map[gaze]
    ox, oy = origin
    tip = (int(ox + dx * length), int(oy + dy * length))
    color = (0, 255, 0) if gaze == "center" else (0, 165, 255)
    cv2.arrowedLine(frame, (ox, oy), tip, color, 3, tipLength=0.3)


def draw_metric_bar(frame: np.ndarray, x: int, y: int, width: int, height: int,
                    value: float, label: str) -> None:
    """Draw horizontal metric bar."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

    fill_width = int((value / 100.0) * width)

    if value >= 75:
        color = (0, 255, 0)
    elif value >= 50:
        color = (0, 200, 200)
    else:
        color = (0, 100, 255)

    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_dashboard(frame: np.ndarray, metrics: FocusMetrics, analyzer: FocusAnalyzer,
                   fps: float, config: VisualizationConfig) -> None:
    """Draw comprehensive metrics dashboard."""
    h, w = frame.shape[:2]
    panel_width = config.dashboard_width
    panel_x = w - panel_width - 10
    panel_y = 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_pos = panel_y + 25
    x_margin = panel_x + 15

    # Header
    cv2.putText(frame, "FOCUS DASHBOARD", (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += 25
    cv2.line(frame, (x_margin, y_pos), (w - 25, y_pos), (100, 100, 100), 1)
    y_pos += 20

    # Current state
    state_text = f"State: {metrics.state.upper()}"
    state_color = get_state_color(metrics.state)
    cv2.putText(frame, state_text, (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    y_pos += 30

    # Focus score
    cv2.putText(frame, f"Focus: {int(metrics.focus_score)}/100", (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 25

    # FPS
    if config.show_fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (x_margin, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += 20

    # Session stats
    if analyzer.session_start:
        session_duration = time.time() - analyzer.session_start
        cv2.putText(frame, f"Session: {format_time(session_duration)}", (x_margin, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += 20

        total_tracked = analyzer.total_focused_time + analyzer.total_distracted_time
        if total_tracked > 0:
            focus_pct = (analyzer.total_focused_time / total_tracked) * 100
            cv2.putText(frame, f"Focused: {format_time(analyzer.total_focused_time)} ({focus_pct:.0f}%)",
                        (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_pos += 20

    # Current window metrics
    cv2.putText(frame, "Current Window:", (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 20

    cv2.putText(frame, f"  Gaze: {metrics.center_gaze_ratio*100:.0f}% center",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 18

    cv2.putText(frame, f"  Blinks: {metrics.blink_rate:.0f}/min ({metrics.blink_rate_status})",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 18

    # Head pose
    cv2.putText(frame, "Head Pose:", (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 20

    cv2.putText(frame, f"  Pitch: {metrics.head_pose.pitch:.1f}°",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 18

    cv2.putText(frame, f"  Yaw: {metrics.head_pose.yaw:.1f}°",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 18

    cv2.putText(frame, f"  Roll: {metrics.head_pose.roll:.1f}°",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 25

    # Metric bars
    bar_width = panel_width - 40
    bar_height = 15

    draw_metric_bar(frame, x_margin, y_pos, bar_width, bar_height,
                    metrics.center_gaze_ratio * 100, "Gaze")
    y_pos += bar_height + 18

    blink_score = analyzer._compute_blink_score(metrics.blink_rate)
    draw_metric_bar(frame, x_margin, y_pos, bar_width, bar_height, blink_score, "Blinks")
    y_pos += bar_height + 18

    presence_score = analyzer._compute_presence_score(metrics.face_visible_ratio)
    draw_metric_bar(frame, x_margin, y_pos, bar_width, bar_height, presence_score, "Presence")


def draw_help_overlay(frame: np.ndarray) -> None:
    """Draw keyboard shortcuts help."""
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 180), (300, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    shortcuts = [
        "KEYBOARD SHORTCUTS:",
        "Q - Quit",
        "R - Reset session",
        "S - Save data",
        "D - Toggle dashboard",
        "H - Toggle help",
        "P - Pause/Resume",
        "SPACE - Calibration sample"
    ]

    y_pos = h - 165
    for text in shortcuts:
        cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
        y_pos += 20


# ---------------------------------------------------------------------------
# Main Eye Tracker Class
# ---------------------------------------------------------------------------

class EyeTrackerV2:
    """Advanced eye tracking system with MediaPipe."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logging(config.log_level, config.log_file)

        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Components
        self.focus_analyzer = FocusAnalyzer(config.focus, self.logger)
        self.calibration_manager = CalibrationManager(config.gaze, self.logger)
        self.data_exporter = DataExporter(config.export_path, self.logger) if config.export_data else None

        # State
        self.blink_count = 0
        self.blink_frame_counter = 0
        self.paused = False
        self.show_help = False
        self.camera_matrix: Optional[np.ndarray] = None

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Video recording
        self.video_writer: Optional[cv2.VideoWriter] = None
        if config.recording.enabled:
            self._setup_recording()

        self.logger.info("Eye Tracker V2 initialized")

    def _setup_recording(self) -> None:
        """Setup video recording."""
        output_path = self.config.recording.output_path.replace(
            '{timestamp}', datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.config.recording.codec)
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.config.recording.fps,
            (self.config.camera.width, self.config.camera.height)
        )
        self.logger.info(f"Recording to {output_path}")

    def _init_camera_matrix(self, image_shape: Tuple[int, int]) -> None:
        """Initialize camera intrinsic matrix."""
        h, w = image_shape
        focal_length = w
        center = (w / 2, h / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def _get_landmarks(self, results, indices: List[int], image_shape: Tuple[int, int]) -> np.ndarray:
        """Extract landmark coordinates."""
        h, w = image_shape
        landmarks = []
        for idx in indices:
            landmark = results.multi_face_landmarks[0].landmark[idx]
            landmarks.append([landmark.x * w, landmark.y * h, landmark.z * w])
        return np.array(landmarks, dtype=np.float32)

    def _process_frame(self, frame: np.ndarray) -> Tuple[str, bool, HeadPose, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Process single frame for eye tracking.

        Returns:
            (gaze, blink_occurred, head_pose, left_pos, right_pos)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        gaze = "no face"
        blink_occurred = False
        head_pose = HeadPose()
        left_pos = None
        right_pos = None

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]

            # Initialize camera matrix if needed
            if self.camera_matrix is None:
                self._init_camera_matrix((h, w))

            # Get eye landmarks
            left_eye = self._get_landmarks(results, LEFT_EYE_INDICES, (h, w))
            right_eye = self._get_landmarks(results, RIGHT_EYE_INDICES, (h, w))

            # Get iris landmarks
            left_iris = self._get_landmarks(results, LEFT_IRIS_INDICES, (h, w))
            right_iris = self._get_landmarks(results, RIGHT_IRIS_INDICES, (h, w))

            # Calculate EAR for blink detection
            left_ear = eye_aspect_ratio(left_eye[:, :2])
            right_ear = eye_aspect_ratio(right_eye[:, :2])
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.config.blink.ear_threshold:
                self.blink_frame_counter += 1
            else:
                if self.blink_frame_counter >= self.config.blink.consecutive_frames:
                    self.blink_count += 1
                    blink_occurred = True
                self.blink_frame_counter = 0

            # Get iris positions
            left_pos = get_iris_position(left_iris[:, :2], left_eye[:, :2])
            right_pos = get_iris_position(right_iris[:, :2], right_eye[:, :2])

            # Classify gaze
            gaze = classify_gaze(left_pos, right_pos, self.config.gaze)

            # Estimate head pose
            face_landmarks = self._get_landmarks(results, list(range(468)), (h, w))
            head_pose = estimate_head_pose(face_landmarks, (h, w), self.camera_matrix)

            # Visualization
            if self.config.visualization.show_landmarks:
                for landmark in results.multi_face_landmarks[0].landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            if self.config.visualization.show_eye_regions:
                # Draw eye contours
                for eye in [left_eye, right_eye]:
                    pts = eye[:, :2].astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 200, 255), 1)

                # Draw iris centers
                for iris in [left_iris, right_iris]:
                    center = np.mean(iris[:, :2], axis=0).astype(np.int32)
                    cv2.circle(frame, tuple(center), 3, (0, 0, 255), -1)

            if self.config.visualization.show_gaze_arrow:
                # Draw gaze arrow at nose tip
                nose_landmark = results.multi_face_landmarks[0].landmark[1]
                nose_pos = (int(nose_landmark.x * w), int(nose_landmark.y * h - 40))
                draw_gaze_arrow(frame, gaze, nose_pos)

        return gaze, blink_occurred, head_pose, left_pos, right_pos

    def _handle_keyboard(self, key: int) -> bool:
        """
        Handle keyboard input.

        Returns:
            True to continue, False to quit
        """
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.logger.info("Resetting session")
            self.focus_analyzer = FocusAnalyzer(self.config.focus, self.logger)
            self.focus_analyzer.start_session()
            self.blink_count = 0
        elif key == ord('s'):
            if self.data_exporter:
                self.data_exporter.export_csv()
                self.data_exporter.export_json()
        elif key == ord('d'):
            self.config.visualization.show_dashboard = not self.config.visualization.show_dashboard
        elif key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord('p'):
            self.paused = not self.paused
            self.logger.info(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord(' ') and self.calibration_manager.is_active:
            # Calibration sample handled in main loop
            pass

        return True

    def run(self, source: Any) -> None:
        """Run the eye tracking system."""
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video source: {source}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

        # Start session
        self.focus_analyzer.start_session()
        self.logger.info("Eye tracker running. Press 'H' for help, 'Q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_frame_time) if current_time != self.last_frame_time else 0
            self.fps_history.append(fps)
            self.last_frame_time = current_time
            avg_fps = np.mean(self.fps_history)

            if not self.paused:
                # Process frame
                gaze, blink_occurred, head_pose, left_pos, right_pos = self._process_frame(frame)

                face_detected = gaze != "no face"

                # Update focus analyzer
                self.focus_analyzer.update(current_time, gaze, face_detected, blink_occurred)
                metrics = self.focus_analyzer.compute_metrics(current_time, head_pose)

                # Export data
                if self.data_exporter:
                    frame_data = FrameData(
                        timestamp=current_time,
                        gaze=gaze,
                        blink_occurred=blink_occurred,
                        face_detected=face_detected,
                        focus_score=metrics.focus_score,
                        focus_state=metrics.state,
                        blink_count=self.blink_count,
                        head_pose=head_pose,
                        fps=avg_fps
                    )
                    self.data_exporter.add_frame(frame_data)

                # Calibration mode
                if self.calibration_manager.is_active:
                    self.calibration_manager.draw(frame)
                else:
                    # Draw HUD
                    cv2.putText(frame, f"Gaze: {gaze}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Draw dashboard
                    if self.config.visualization.show_dashboard:
                        draw_dashboard(frame, metrics, self.focus_analyzer, avg_fps, self.config.visualization)
            else:
                # Paused overlay
                h, w = frame.shape[:2]
                cv2.putText(frame, "PAUSED (Press P to resume)", (w//2 - 200, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            # Draw help
            if self.show_help:
                draw_help_overlay(frame)

            # Record frame
            if self.video_writer:
                self.video_writer.write(frame)

            # Display
            cv2.imshow("Eye Tracker V2", frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if self.calibration_manager.is_active and key == ord(' '):
                    if self.calibration_manager.add_sample(left_pos, right_pos):
                        self.logger.info("Calibration complete!")

                if not self._handle_keyboard(key):
                    break

        # Cleanup
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

        # Export final data
        if self.data_exporter:
            self.data_exporter.export_csv()
            self.data_exporter.export_json()

        # Print summary
        summary = self.focus_analyzer.get_session_summary()
        if summary:
            self.logger.info("=== SESSION SUMMARY ===")
            self.logger.info(f"Total duration: {format_time(summary['total_duration'])}")
            self.logger.info(f"Focused: {format_time(summary['focused_time'])} ({summary['focused_percentage']:.1f}%)")
            self.logger.info(f"Distracted: {format_time(summary['distracted_time'])} ({summary['distracted_percentage']:.1f}%)")
            self.logger.info(f"Total blinks: {self.blink_count}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Eye Tracking System with MediaPipe",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config', '-c',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--source', '-s',
        default=0,
        help='Video source: camera index (0, 1, ...) or video file path'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Start in calibration mode'
    )
    parser.add_argument(
        '--export-data',
        help='Export session data to CSV/JSON file'
    )
    parser.add_argument(
        '--record',
        help='Record annotated video to file'
    )
    parser.add_argument(
        '--generate-config',
        help='Generate default config file and exit'
    )

    args = parser.parse_args()

    # Generate config if requested
    if args.generate_config:
        config = AppConfig()
        config.to_yaml(args.generate_config)
        print(f"Default configuration saved to {args.generate_config}")
        return

    # Load configuration
    if args.config:
        try:
            config = AppConfig.from_yaml(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        config = AppConfig()

    # Override config with command line args
    try:
        config.camera.source = int(args.source)
    except ValueError:
        config.camera.source = args.source

    if args.export_data:
        config.export_data = True
        config.export_path = args.export_data

    if args.record:
        config.recording.enabled = True
        config.recording.output_path = args.record

    # Create and run tracker
    tracker = EyeTrackerV2(config)

    if args.calibrate:
        tracker.calibration_manager.start()

    try:
        tracker.run(config.camera.source)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logging.error(f"Error during tracking: {e}", exc_info=True)


if __name__ == "__main__":
    main()
