"""
Eye Tracking Sample with MediaPipe Face Mesh
Tracks gaze direction (left, right, center, up, down, diagonals) and analyzes
focus state using MediaPipe's 478-point facial landmark model with native iris tracking.

Features:
- 3-5x faster than dlib (120-180 FPS vs 30-50 FPS)
- Native iris landmarks (no thresholding needed)
- Better tracking robustness (handles rotation, occlusion, lighting)
- Dual backend support: MediaPipe (default) or dlib (legacy)

Usage:
    # MediaPipe backend (default - recommended)
    python eye_tracker.py
    python eye_tracker.py --source path/to/video.mp4
    python eye_tracker.py --no-focus-tracking

    # dlib backend (legacy)
    python eye_tracker.py --backend dlib
    python eye_tracker.py --backend dlib --predictor path/to/shape_predictor_68_face_landmarks.dat
"""

import argparse
import sys
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Facial landmark indices
# ---------------------------------------------------------------------------
# MediaPipe Face Mesh (478 points with refine_landmarks=True)
# Left eye:  6 key points for EAR calculation
# Right eye: 6 key points for EAR calculation
LEFT_EYE_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_POINTS = [33, 160, 158, 133, 145, 153]

# Full eye contours for visualization (16 points each)
LEFT_EYE_FULL_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_FULL_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Iris landmarks (MediaPipe native iris centers when refine_landmarks=True)
LEFT_IRIS_CENTER  = 468   # Single point - iris center
RIGHT_IRIS_CENTER = 473   # Single point - iris center

# dlib 68-point model indices (for backward compatibility with --backend dlib)
DLIB_LEFT_EYE_POINTS  = list(range(36, 42))
DLIB_RIGHT_EYE_POINTS = list(range(42, 48))

# Focus detection configuration
FOCUS_WINDOW_SIZE = 30.0          # seconds
GAZE_FOCUS_THRESHOLD = 0.7        # 70% center gaze = focused
BLINK_RATE_NORMAL = (15, 30)      # normal blinks per minute range
FACE_LOSS_THRESHOLD = 0.2         # 20% face loss = distracted
FOCUS_UPDATE_INTERVAL = 0.5       # update metrics every 0.5s


# ---------------------------------------------------------------------------
# Focus Metrics Data Structure
# ---------------------------------------------------------------------------

@dataclass
class FocusMetrics:
    """Container for analyzed focus metrics."""
    # Gaze metrics
    center_gaze_ratio: float = 0.0
    dominant_gaze: str = "unknown"
    gaze_switches: int = 0

    # Blink metrics
    blink_rate: float = 0.0
    blink_rate_status: str = "normal"

    # Face detection metrics
    face_visible_ratio: float = 0.0
    face_loss_count: int = 0

    # Composite state
    focus_score: float = 0.0
    state: str = "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Compute the Eye Aspect Ratio (EAR) to detect blinks."""
    # Vertical distances
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    # Horizontal distance
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C)


def landmarks_to_np(shape, dtype=np.float32) -> np.ndarray:
    """Convert a dlib full_object_detection to a (68, 2) NumPy array (for dlib backend)."""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_eye_region(frame_gray: np.ndarray, landmarks,
                   eye_points: list, padding: int = 5, frame_shape=None, is_mediapipe=False):
    """
    Extract the eye ROI from the grayscale frame.

    Supports both MediaPipe (normalized [0,1] coordinates) and dlib (pixel coordinates).

    Args:
        frame_gray: Grayscale frame
        landmarks: MediaPipe landmarks (list of landmark objects) or dlib landmarks (numpy array)
        eye_points: List of landmark indices for the eye
        padding: Padding around the eye region
        frame_shape: (height, width) tuple for denormalizing MediaPipe coordinates
        is_mediapipe: True if landmarks are from MediaPipe

    Returns:
        eye_roi   -- cropped grayscale eye image
        eye_rect  -- (x, y, w, h) bounding rectangle in frame coordinates
    """
    if is_mediapipe:
        # MediaPipe landmarks are normalized [0,1] - denormalize to pixel coordinates
        if frame_shape:
            h_frame, w_frame = frame_shape[:2]  # Extract only height and width
        else:
            h_frame, w_frame = frame_gray.shape[:2]
        pts = np.array([[landmarks[i].x * w_frame, landmarks[i].y * h_frame]
                        for i in eye_points], dtype=np.int32)
    else:
        # dlib landmarks are already in pixel coordinates
        pts = landmarks[eye_points].astype(np.int32)

    x, y, w, h = cv2.boundingRect(pts)
    frame_h, frame_w = frame_gray.shape[:2]
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_w - x, w + 2 * padding)
    h = min(frame_h - y, h + 2 * padding)
    return frame_gray[y:y + h, x:x + w], (x, y, w, h)


def detect_iris_position(eye_roi: np.ndarray):
    """
    Estimate iris centre within the eye ROI using thresholding + contours.

    Returns:
        (cx, cy) relative position in [0, 1] x [0, 1], or None on failure.
    """
    if eye_roi.size == 0:
        return None

    # Blur and threshold to isolate the dark iris
    blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour (likely the iris)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"] / eye_roi.shape[1]
    cy = M["m01"] / M["m00"] / eye_roi.shape[0]
    return cx, cy


def get_iris_position_mediapipe(landmarks, iris_center_idx: int,
                                  eye_points: list, frame_shape):
    """
    Get normalized iris position within eye bounding box using MediaPipe's native iris landmarks.

    MediaPipe provides direct iris center coordinates (indices 468, 473) when refine_landmarks=True.
    This function normalizes the iris position relative to the eye bounding box.

    Args:
        landmarks: MediaPipe face mesh landmarks (list of landmark objects)
        iris_center_idx: 468 (left iris) or 473 (right iris)
        eye_points: List of eye contour indices for computing bounding box
        frame_shape: (height, width, channels) of frame

    Returns:
        (cx, cy) relative position in [0, 1] x [0, 1], or None on failure.
    """
    if not landmarks or iris_center_idx >= len(landmarks):
        return None

    h_frame, w_frame = frame_shape[:2]

    # Get iris center in pixel coordinates
    iris = landmarks[iris_center_idx]
    iris_x = iris.x * w_frame
    iris_y = iris.y * h_frame

    # Get eye bounding box
    eye_pts = np.array([[landmarks[i].x * w_frame, landmarks[i].y * h_frame]
                        for i in eye_points], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(eye_pts)

    # Normalize iris position within eye bounding box
    if w == 0 or h == 0:
        return None

    cx = (iris_x - x) / w
    cy = (iris_y - y) / h

    # Clamp to [0, 1] range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))

    return cx, cy


def classify_gaze(left_pos, right_pos) -> str:
    """
    Classify gaze direction from normalised iris positions.

    left_pos / right_pos: (cx, cy) in [0,1] or None
    """
    positions = [p for p in (left_pos, right_pos) if p is not None]
    if not positions:
        return "undetected"

    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])

    # Thresholds (tune to your setup)
    if cx < 0.38:
        h_dir = "right"   # iris near the left edge → looking right
    elif cx > 0.62:
        h_dir = "left"    # iris near the right edge → looking left
    else:
        h_dir = "center"

    if cy < 0.35:
        v_dir = "up"
    elif cy > 0.65:
        v_dir = "down"
    else:
        v_dir = ""

    if h_dir == "center" and not v_dir:
        return "center"
    if v_dir and h_dir == "center":
        return v_dir
    if v_dir:
        return f"{v_dir}-{h_dir}"
    return h_dir


def draw_gaze_arrow(frame: np.ndarray, gaze: str,
                    origin, length: int = 60):
    """Draw an arrow on the frame indicating gaze direction."""
    direction_map = {
        "center":       (0,  0),
        "left":         (-1, 0),
        "right":        (1,  0),
        "up":           (0, -1),
        "down":         (0,  1),
        "up-left":      (-1, -1),
        "up-right":     (1, -1),
        "down-left":    (-1,  1),
        "down-right":   (1,  1),
    }
    if gaze not in direction_map:
        return

    dx, dy = direction_map[gaze]
    ox, oy = origin
    tip = (int(ox + dx * length), int(oy + dy * length))
    color = (0, 255, 0) if gaze == "center" else (0, 165, 255)
    cv2.arrowedLine(frame, (ox, oy), tip, color, 3, tipLength=0.3)


# ---------------------------------------------------------------------------
# Focus Analyzer Class
# ---------------------------------------------------------------------------

class FocusAnalyzer:
    """
    Analyzes temporal eye tracking metrics to determine focus/distraction state.

    Uses sliding time windows to track:
    - Gaze stability (time spent looking at center vs away)
    - Blink rate (normal vs abnormal patterns)
    - Face detection continuity (user looking away from screen)
    """

    def __init__(self,
                 window_size: float = 30.0,
                 gaze_threshold: float = 0.7,
                 blink_rate_normal: tuple = (15, 30),
                 face_loss_threshold: float = 0.2):

        # Configuration
        self.window_size = window_size
        self.gaze_threshold = gaze_threshold
        self.blink_rate_normal = blink_rate_normal
        self.face_loss_threshold = face_loss_threshold

        # Temporal data storage
        self.gaze_history = []           # [(timestamp, gaze_direction), ...]
        self.blink_timestamps = []       # [timestamp, ...]
        self.face_detected_history = []  # [(timestamp, True/False), ...]

        # Session statistics
        self.session_start = None
        self.total_focused_time = 0.0
        self.total_distracted_time = 0.0
        self.current_state = "unknown"
        self.state_change_time = None

        # Per-window analytics cache
        self.cached_metrics = None
        self.last_analysis_time = 0
        self.analysis_interval = FOCUS_UPDATE_INTERVAL

    def update(self, timestamp: float, gaze: str, face_detected: bool, blink_occurred: bool):
        """Called every frame to update temporal buffers."""
        # Add gaze data
        self.gaze_history.append((timestamp, gaze))

        # Add blink data
        if blink_occurred:
            self.blink_timestamps.append(timestamp)

        # Add face detection data
        self.face_detected_history.append((timestamp, face_detected))

        # Prune old data outside the time window
        cutoff_time = timestamp - self.window_size
        self.gaze_history = [(t, g) for t, g in self.gaze_history if t >= cutoff_time]
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff_time]
        self.face_detected_history = [(t, f) for t, f in self.face_detected_history if t >= cutoff_time]

    def compute_metrics(self, current_time: float) -> FocusMetrics:
        """Analyze time window and compute all metrics."""
        # Check cache validity
        if (current_time - self.last_analysis_time) < self.analysis_interval and self.cached_metrics:
            return self.cached_metrics

        metrics = FocusMetrics()

        # Compute gaze metrics
        if self.gaze_history:
            center_count = sum(1 for _, gaze in self.gaze_history if gaze == "center")
            metrics.center_gaze_ratio = center_count / len(self.gaze_history)

            # Find dominant gaze
            gaze_counts = {}
            for _, gaze in self.gaze_history:
                gaze_counts[gaze] = gaze_counts.get(gaze, 0) + 1
            metrics.dominant_gaze = max(gaze_counts, key=gaze_counts.get)

            # Count gaze switches
            prev_gaze = None
            for _, gaze in self.gaze_history:
                if prev_gaze is not None and gaze != prev_gaze:
                    metrics.gaze_switches += 1
                prev_gaze = gaze

        # Compute blink metrics
        if self.blink_timestamps:
            window_duration_minutes = self.window_size / 60.0
            metrics.blink_rate = len(self.blink_timestamps) / window_duration_minutes

            # Classify blink rate
            if metrics.blink_rate < self.blink_rate_normal[0]:
                if metrics.blink_rate < 10:
                    metrics.blink_rate_status = "very low"
                else:
                    metrics.blink_rate_status = "low"
            elif metrics.blink_rate <= self.blink_rate_normal[1]:
                metrics.blink_rate_status = "normal"
            elif metrics.blink_rate <= 45:
                metrics.blink_rate_status = "high"
            else:
                metrics.blink_rate_status = "very high"

        # Compute face detection metrics
        if self.face_detected_history:
            face_visible_count = sum(1 for _, detected in self.face_detected_history if detected)
            metrics.face_visible_ratio = face_visible_count / len(self.face_detected_history)

            # Count face loss events
            prev_detected = None
            for _, detected in self.face_detected_history:
                if prev_detected is not None and prev_detected and not detected:
                    metrics.face_loss_count += 1
                prev_detected = detected

        # Compute composite scores
        gaze_score = self._compute_gaze_score(metrics)
        blink_score = self._compute_blink_score(metrics)
        presence_score = self._compute_presence_score(metrics)

        # Weighted focus score
        metrics.focus_score = 0.5 * gaze_score + 0.2 * blink_score + 0.3 * presence_score

        # Classify state with hysteresis
        new_state = self._classify_state(metrics.focus_score, presence_score)

        # Apply hysteresis (3-second minimum state duration)
        if new_state != self.current_state:
            if self.state_change_time is None:
                self.state_change_time = current_time
            elif (current_time - self.state_change_time) >= 3.0:
                # Update session statistics
                time_in_state = current_time - self.state_change_time
                if self.current_state == "focused":
                    self.total_focused_time += time_in_state
                elif self.current_state in ("distracted", "semi-focused"):
                    self.total_distracted_time += time_in_state

                self.current_state = new_state
                self.state_change_time = current_time
        else:
            # Reset state change timer if we return to current state
            self.state_change_time = None

        metrics.state = self.current_state

        # Cache the result
        self.cached_metrics = metrics
        self.last_analysis_time = current_time

        return metrics

    def _compute_gaze_score(self, metrics: FocusMetrics) -> float:
        """Calculate gaze-based focus score (0-100)."""
        # Center gaze is most important for screen work
        return metrics.center_gaze_ratio * 100.0

    def _compute_blink_score(self, metrics: FocusMetrics) -> float:
        """Calculate blink-rate-based score (0-100)."""
        rate = metrics.blink_rate

        if self.blink_rate_normal[0] <= rate <= self.blink_rate_normal[1]:
            return 100.0  # Normal
        elif rate < 10:
            return 70.0   # Very low (intense focus or dry eyes)
        elif rate < self.blink_rate_normal[0]:
            return 80.0   # Low but acceptable
        elif rate <= 45:
            return 60.0   # High (possible distraction)
        else:
            return 30.0   # Very high (likely distracted)

    def _compute_presence_score(self, metrics: FocusMetrics) -> float:
        """Calculate face-presence score (0-100)."""
        ratio = metrics.face_visible_ratio

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


# ---------------------------------------------------------------------------
# Dashboard Rendering Functions
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def get_state_color(state: str) -> tuple:
    """Return BGR color for focus state."""
    color_map = {
        "focused": (0, 255, 0),        # Green
        "semi-focused": (0, 200, 200),  # Yellow-green
        "distracted": (0, 165, 255),    # Orange
        "away": (0, 0, 255),            # Red
        "unknown": (150, 150, 150),     # Gray
    }
    return color_map.get(state, (150, 150, 150))


def draw_metric_bar(frame: np.ndarray, x: int, y: int, width: int, height: int,
                    value: float, label: str):
    """Draw a single horizontal metric bar."""
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

    # Draw filled portion based on value (0-100)
    fill_width = int((value / 100.0) * width)

    # Color gradient: red (low) -> yellow (medium) -> green (high)
    if value >= 75:
        color = (0, 255, 0)  # Green
    elif value >= 50:
        color = (0, 200, 200)  # Yellow
    else:
        color = (0, 100, 255)  # Orange-red

    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    # Draw border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)

    # Draw label
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1)


def draw_focus_dashboard(frame: np.ndarray, metrics: FocusMetrics, analyzer: FocusAnalyzer):
    """Draw comprehensive focus metrics dashboard on right side of frame."""
    h, w = frame.shape[:2]

    # Dashboard dimensions
    panel_width = 280
    panel_x = w - panel_width - 10
    panel_y = 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Current Y position for drawing elements
    y_pos = panel_y + 25
    x_margin = panel_x + 15

    # --- Header ---
    cv2.putText(frame, "FOCUS DASHBOARD", (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += 25

    # Separator line
    cv2.line(frame, (x_margin, y_pos), (w - 25, y_pos), (100, 100, 100), 1)
    y_pos += 20

    # --- Current State (Large, color-coded) ---
    state_text = f"State: {metrics.state.upper()}"
    state_color = get_state_color(metrics.state)
    cv2.putText(frame, state_text, (x_margin, y_pos),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, state_color, 2)
    y_pos += 30

    # Focus score
    score_text = f"Focus Score: {int(metrics.focus_score)}/100"
    cv2.putText(frame, score_text, (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 30

    # --- Session Statistics ---
    if analyzer.session_start:
        session_duration = time.time() - analyzer.session_start

        cv2.putText(frame, f"Session: {format_time(session_duration)}",
                    (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += 20

        # Calculate percentages
        total_tracked = analyzer.total_focused_time + analyzer.total_distracted_time
        if total_tracked > 0:
            focus_pct = (analyzer.total_focused_time / total_tracked) * 100
            distract_pct = (analyzer.total_distracted_time / total_tracked) * 100

            cv2.putText(frame, f"Focused: {format_time(analyzer.total_focused_time)} ({focus_pct:.0f}%)",
                        (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_pos += 20

            cv2.putText(frame, f"Distracted: {format_time(analyzer.total_distracted_time)} ({distract_pct:.0f}%)",
                        (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            y_pos += 25

    # --- Current Window Metrics ---
    cv2.putText(frame, f"Window ({int(analyzer.window_size)}s):",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 20

    # Center gaze percentage
    gaze_pct = metrics.center_gaze_ratio * 100
    cv2.putText(frame, f"  Center gaze: {gaze_pct:.0f}%",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 18

    # Blink rate
    blink_text = f"  Blink rate: {metrics.blink_rate:.0f}/min ({metrics.blink_rate_status})"
    cv2.putText(frame, blink_text, (x_margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 18

    # Face visibility
    face_pct = metrics.face_visible_ratio * 100
    cv2.putText(frame, f"  Face visible: {face_pct:.0f}%",
                (x_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 30

    # --- Metric Bars ---
    bar_width = panel_width - 40
    bar_height = 15

    # Gaze bar
    gaze_score = metrics.center_gaze_ratio * 100
    draw_metric_bar(frame, x_margin, y_pos, bar_width, bar_height, gaze_score, "Gaze")
    y_pos += bar_height + 18

    # Blink bar
    blink_score = analyzer._compute_blink_score(metrics)
    draw_metric_bar(frame, x_margin, y_pos, bar_width, bar_height, blink_score, "Blinks")
    y_pos += bar_height + 18

    # Presence bar
    presence_score = analyzer._compute_presence_score(metrics)
    draw_metric_bar(frame, x_margin, y_pos, bar_width, bar_height, presence_score, "Presence")


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

def run(source, enable_focus_tracking: bool = True,
        use_mediapipe: bool = True, predictor_path: str = None):
    """
    Main eye tracking loop.

    Args:
        source: Video source (0 for webcam, or path to video file)
        enable_focus_tracking: Enable focus/distraction analysis
        use_mediapipe: Use MediaPipe (True) or dlib (False) backend
        predictor_path: Path to dlib .dat file (only needed if use_mediapipe=False)
    """

    # Initialize face detection backend
    detector = None
    predictor = None
    face_mesh = None

    if use_mediapipe:
        print("[INFO] Initializing MediaPipe Face Landmarker...")
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Create Face Landmarker with iris model
        model_path = "face_landmarker.task"
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        face_mesh = vision.FaceLandmarker.create_from_options(options)
        print("[INFO] MediaPipe initialized successfully.")
    else:
        # Fallback to dlib (backward compatibility)
        print("[INFO] Initializing dlib face detector...")
        import dlib
        detector = dlib.get_frontal_face_detector()
        try:
            predictor = dlib.shape_predictor(predictor_path)
            print("[INFO] dlib initialized successfully.")
        except RuntimeError as exc:
            print(f"[ERROR] Could not load shape predictor: {exc}")
            print("  Download it with:  bash setup.sh")
            sys.exit(1)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source!r}")
        sys.exit(1)

    # Blink detection state
    EAR_THRESHOLD = 0.21
    blink_count   = 0
    blink_frame   = 0
    BLINK_CONSEC  = 3          # consecutive frames below threshold = blink

    # Focus tracking state
    focus_analyzer = None
    if enable_focus_tracking:
        focus_analyzer = FocusAnalyzer(
            window_size=FOCUS_WINDOW_SIZE,
            gaze_threshold=GAZE_FOCUS_THRESHOLD,
            blink_rate_normal=BLINK_RATE_NORMAL,
            face_loss_threshold=FACE_LOSS_THRESHOLD
        )
        focus_analyzer.session_start = time.time()

    print(f"[INFO] Eye tracker running with {'MediaPipe' if use_mediapipe else 'dlib'} backend — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_timestamp = time.time()
        face_detected = False
        gaze = "no face"
        h, w = frame.shape[:2]

        if use_mediapipe:
            # MediaPipe processing pipeline
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect face landmarks (use timestamp in milliseconds)
            timestamp_ms = int(current_timestamp * 1000)
            results = face_mesh.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                face_detected = True
                landmarks = results.face_landmarks[0]

                # --- Compute face bounding box from landmarks ---
                # Note: landmarks is a list of NormalizedLandmark objects
                face_x_coords = [lm.x * w for lm in landmarks[:468]]
                face_y_coords = [lm.y * h for lm in landmarks[:468]]
                x1, x2 = int(min(face_x_coords)), int(max(face_x_coords))
                y1, y2 = int(min(face_y_coords)), int(max(face_y_coords))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                # --- Draw eye landmarks ---
                for idx in LEFT_EYE_FULL_CONTOUR + RIGHT_EYE_FULL_CONTOUR:
                    x_pt = int(landmarks[idx].x * w)
                    y_pt = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x_pt, y_pt), 1, (0, 255, 255), -1)

                # --- Extract eye ROIs (for visualization) ---
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                left_roi, left_rect = get_eye_region(
                    frame_gray, landmarks, LEFT_EYE_FULL_CONTOUR,
                    padding=5, frame_shape=frame.shape, is_mediapipe=True
                )
                right_roi, right_rect = get_eye_region(
                    frame_gray, landmarks, RIGHT_EYE_FULL_CONTOUR,
                    padding=5, frame_shape=frame.shape, is_mediapipe=True
                )

                # Draw eye rectangles
                for rect in (left_rect, right_rect):
                    rx, ry, rw, rh = rect
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 1)

                # --- Iris detection (native MediaPipe landmarks) ---
                left_pos = get_iris_position_mediapipe(
                    landmarks, LEFT_IRIS_CENTER, LEFT_EYE_FULL_CONTOUR, frame.shape
                )
                right_pos = get_iris_position_mediapipe(
                    landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_FULL_CONTOUR, frame.shape
                )

                # Draw iris centers
                for iris_idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
                    iris_x = int(landmarks[iris_idx].x * w)
                    iris_y = int(landmarks[iris_idx].y * h)
                    cv2.circle(frame, (iris_x, iris_y), 4, (0, 0, 255), -1)

                # --- EAR / blink detection ---
                # Extract 6-point eye coordinates for EAR
                left_eye_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                                         for i in LEFT_EYE_POINTS], dtype=np.float32)
                right_eye_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                                          for i in RIGHT_EYE_POINTS], dtype=np.float32)

                ear = (eye_aspect_ratio(left_eye_pts) +
                       eye_aspect_ratio(right_eye_pts)) / 2.0

                if ear < EAR_THRESHOLD:
                    blink_frame += 1
                else:
                    if blink_frame >= BLINK_CONSEC:
                        blink_count += 1
                        if focus_analyzer:
                            focus_analyzer.update(current_timestamp, gaze,
                                                face_detected, blink_occurred=True)
                    blink_frame = 0

                # --- Gaze classification ---
                gaze = classify_gaze(left_pos, right_pos)

                # --- Gaze arrow (centered above face) ---
                arrow_origin = (int((x1 + x2) / 2), max(0, y1 - 30))
                draw_gaze_arrow(frame, gaze, arrow_origin)

                # --- Per-face info overlay ---
                cv2.putText(frame, f"EAR: {ear:.2f}", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        else:
            # dlib processing pipeline (backward compatibility)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)

            faces = detector(frame_gray, 0)
            face_detected = len(faces) > 0

            for face in faces:
                shape     = predictor(frame_gray, face)
                landmarks = landmarks_to_np(shape)

                # --- Draw face bounding box ---
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                # --- Draw all 68 landmarks ---
                for (lx, ly) in landmarks.astype(np.int32):
                    cv2.circle(frame, (lx, ly), 1, (0, 255, 255), -1)

                # --- Extract eye ROIs ---
                left_roi,  left_rect  = get_eye_region(frame_gray, landmarks,
                                                        DLIB_LEFT_EYE_POINTS,
                                                        is_mediapipe=False)
                right_roi, right_rect = get_eye_region(frame_gray, landmarks,
                                                        DLIB_RIGHT_EYE_POINTS,
                                                        is_mediapipe=False)

                # Draw eye rectangles
                for rect in (left_rect, right_rect):
                    rx, ry, rw, rh = rect
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                                  (0, 200, 255), 1)

                # --- Iris detection ---
                left_pos  = detect_iris_position(left_roi)
                right_pos = detect_iris_position(right_roi)

                # Draw iris centres in frame coordinates
                for pos, rect in ((left_pos, left_rect), (right_pos, right_rect)):
                    if pos is None:
                        continue
                    rx, ry, rw, rh = rect
                    ix = int(rx + pos[0] * rw)
                    iy = int(ry + pos[1] * rh)
                    cv2.circle(frame, (ix, iy), 4, (0, 0, 255), -1)

                # --- EAR / blink detection ---
                left_eye_pts  = landmarks[DLIB_LEFT_EYE_POINTS]
                right_eye_pts = landmarks[DLIB_RIGHT_EYE_POINTS]
                ear = (eye_aspect_ratio(left_eye_pts) +
                       eye_aspect_ratio(right_eye_pts)) / 2.0

                if ear < EAR_THRESHOLD:
                    blink_frame += 1
                else:
                    if blink_frame >= BLINK_CONSEC:
                        blink_count += 1
                        if focus_analyzer:
                            focus_analyzer.update(current_timestamp, gaze, face_detected, blink_occurred=True)
                    blink_frame = 0

                # --- Gaze classification ---
                gaze = classify_gaze(left_pos, right_pos)

                # --- Gaze arrow (centred above the face) ---
                arrow_origin = (int((x1 + x2) / 2), max(0, y1 - 30))
                draw_gaze_arrow(frame, gaze, arrow_origin)

                # --- Per-face info overlay ---
                cv2.putText(frame, f"EAR: {ear:.2f}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 200, 200), 1)

                # Only process the first face for clarity
                break

        # Update focus analyzer for non-blink frames
        if focus_analyzer and blink_frame == 0:
            focus_analyzer.update(current_timestamp, gaze, face_detected, blink_occurred=False)

        # --- Global HUD ---
        cv2.putText(frame, f"Gaze: {gaze}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # Backend indicator
        backend_text = "MediaPipe" if use_mediapipe else "dlib"
        cv2.putText(frame, f"Backend: {backend_text}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 200, 255), 1)

        cv2.putText(frame, "Press 'q' to quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1)

        # Draw focus dashboard
        if focus_analyzer:
            metrics = focus_analyzer.compute_metrics(current_timestamp)
            draw_focus_dashboard(frame, metrics, focus_analyzer)

        cv2.imshow("Eye Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Cleanup backend resources
    if use_mediapipe and face_mesh:
        face_mesh.close()

    # Print focus tracking summary
    if focus_analyzer:
        total_time = time.time() - focus_analyzer.session_start
        print(f"[INFO] Session duration: {format_time(total_time)}")
        print(f"[INFO] Time focused: {format_time(focus_analyzer.total_focused_time)} "
              f"({focus_analyzer.total_focused_time/total_time*100:.1f}%)")
        print(f"[INFO] Time distracted: {format_time(focus_analyzer.total_distracted_time)} "
              f"({focus_analyzer.total_distracted_time/total_time*100:.1f}%)")

    print(f"[INFO] Session ended. Total blinks detected: {blink_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Eye tracker with MediaPipe Face Mesh or dlib backend"
    )
    p.add_argument(
        "--backend",
        choices=["mediapipe", "dlib"],
        default="mediapipe",
        help="Face detection backend: 'mediapipe' (default, faster) or 'dlib' (legacy)",
    )
    p.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-point shape predictor .dat file "
             "(only needed for --backend dlib; default: shape_predictor_68_face_landmarks.dat)",
    )
    p.add_argument(
        "--source",
        default=0,
        help="Video source: 0 for webcam (default), or path to a video file",
    )
    p.add_argument(
        "--no-focus-tracking",
        action="store_true",
        help="Disable focus/distraction detection (default: enabled)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Allow passing an integer camera index via --source
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    use_mediapipe = (args.backend == "mediapipe")

    run(
        source=source,
        enable_focus_tracking=not args.no_focus_tracking,
        use_mediapipe=use_mediapipe,
        predictor_path=args.predictor if not use_mediapipe else None
    )
