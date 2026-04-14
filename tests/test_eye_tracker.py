"""
Unit tests for Eye Tracker V2
"""

import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eye_tracker_v2 import (
    eye_aspect_ratio,
    get_iris_position,
    classify_gaze,
    estimate_head_pose,
    GazeConfig,
    BlinkConfig,
    FocusConfig,
    AppConfig,
    FocusMetrics,
    HeadPose,
    FrameData,
    FocusAnalyzer,
    CalibrationManager,
    DataExporter,
    format_time,
    get_state_color,
)


class TestEyeAspectRatio:
    """Test eye aspect ratio calculation."""

    def test_normal_open_eye(self):
        """Test EAR for normal open eye."""
        # Simulate open eye landmarks (horizontal)
        eye_pts = np.array([
            [0, 10],   # left corner
            [5, 8],    # top left
            [10, 6],   # top right
            [15, 10],  # right corner
            [10, 14],  # bottom right
            [5, 12],   # bottom left
        ], dtype=np.float32)

        ear = eye_aspect_ratio(eye_pts)
        assert 0.2 < ear < 0.4  # Typical range for open eye

    def test_closed_eye(self):
        """Test EAR for closed eye."""
        # Simulate closed eye (minimal vertical distance)
        eye_pts = np.array([
            [0, 10],
            [5, 10],
            [10, 10],
            [15, 10],
            [10, 10],
            [5, 10],
        ], dtype=np.float32)

        ear = eye_aspect_ratio(eye_pts)
        assert ear < 0.15  # Very low for closed eye


class TestIrisPosition:
    """Test iris position calculation."""

    def test_center_iris(self):
        """Test iris at center of eye."""
        eye_landmarks = np.array([
            [0, 0], [20, 0], [20, 20], [0, 20]
        ], dtype=np.float32)

        iris_landmarks = np.array([
            [9, 9], [11, 9], [11, 11], [9, 11], [10, 10]
        ], dtype=np.float32)

        pos = get_iris_position(iris_landmarks, eye_landmarks)
        assert pos is not None
        cx, cy = pos
        assert 0.4 < cx < 0.6  # Should be centered
        assert 0.4 < cy < 0.6

    def test_left_iris(self):
        """Test iris looking left."""
        eye_landmarks = np.array([
            [0, 0], [20, 0], [20, 20], [0, 20]
        ], dtype=np.float32)

        iris_landmarks = np.array([
            [14, 9], [16, 9], [16, 11], [14, 11], [15, 10]
        ], dtype=np.float32)

        pos = get_iris_position(iris_landmarks, eye_landmarks)
        assert pos is not None
        cx, cy = pos
        assert cx > 0.6  # Iris on right side = looking left

    def test_empty_landmarks(self):
        """Test with empty landmarks."""
        pos = get_iris_position(np.array([]), np.array([]))
        assert pos is None


class TestGazeClassification:
    """Test gaze direction classification."""

    def setup_method(self):
        """Setup test configuration."""
        self.config = GazeConfig()

    def test_center_gaze(self):
        """Test center gaze classification."""
        left_pos = (0.5, 0.5)
        right_pos = (0.5, 0.5)

        gaze = classify_gaze(left_pos, right_pos, self.config)
        assert gaze == "center"

    def test_left_gaze(self):
        """Test left gaze classification."""
        left_pos = (0.7, 0.5)
        right_pos = (0.7, 0.5)

        gaze = classify_gaze(left_pos, right_pos, self.config)
        assert gaze == "left"

    def test_right_gaze(self):
        """Test right gaze classification."""
        left_pos = (0.3, 0.5)
        right_pos = (0.3, 0.5)

        gaze = classify_gaze(left_pos, right_pos, self.config)
        assert gaze == "right"

    def test_up_gaze(self):
        """Test up gaze classification."""
        left_pos = (0.5, 0.2)
        right_pos = (0.5, 0.2)

        gaze = classify_gaze(left_pos, right_pos, self.config)
        assert gaze == "up"

    def test_down_gaze(self):
        """Test down gaze classification."""
        left_pos = (0.5, 0.8)
        right_pos = (0.5, 0.8)

        gaze = classify_gaze(left_pos, right_pos, self.config)
        assert gaze == "down"

    def test_diagonal_gaze(self):
        """Test diagonal gaze classification."""
        left_pos = (0.7, 0.2)
        right_pos = (0.7, 0.2)

        gaze = classify_gaze(left_pos, right_pos, self.config)
        assert gaze == "up-left"

    def test_no_detection(self):
        """Test when no iris detected."""
        gaze = classify_gaze(None, None, self.config)
        assert gaze == "undetected"


class TestHeadPose:
    """Test head pose estimation."""

    def test_head_pose_structure(self):
        """Test HeadPose data structure."""
        pose = HeadPose(pitch=10.0, yaw=-5.0, roll=2.0)
        assert pose.pitch == 10.0
        assert pose.yaw == -5.0
        assert pose.roll == 2.0


class TestConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = AppConfig()
        assert config.gaze.horizontal_center_min == 0.38
        assert config.blink.ear_threshold == 0.21
        assert config.focus.window_size == 30.0

    def test_yaml_roundtrip(self):
        """Test saving and loading YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            # Create and save config
            config1 = AppConfig()
            config1.gaze.horizontal_center_min = 0.4
            config1.blink.ear_threshold = 0.25
            config1.to_yaml(config_path)

            # Load config
            config2 = AppConfig.from_yaml(config_path)

            # Verify
            assert config2.gaze.horizontal_center_min == 0.4
            assert config2.blink.ear_threshold == 0.25
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestFocusAnalyzer:
    """Test focus analysis."""

    def setup_method(self):
        """Setup test analyzer."""
        from logging import getLogger
        self.config = FocusConfig(window_size=10.0)
        self.logger = getLogger("test")
        self.analyzer = FocusAnalyzer(self.config, self.logger)
        self.analyzer.start_session()

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.current_state == "unknown"
        assert self.analyzer.session_start is not None
        assert len(self.analyzer.gaze_history) == 0

    def test_update_adds_data(self):
        """Test that update adds data to buffers."""
        self.analyzer.update(1.0, "center", True, False)
        assert len(self.analyzer.gaze_history) == 1
        assert len(self.analyzer.face_detected_history) == 1

    def test_update_adds_blink(self):
        """Test that blinks are recorded."""
        self.analyzer.update(1.0, "center", True, True)
        assert len(self.analyzer.blink_timestamps) == 1

    def test_data_pruning(self):
        """Test that old data is pruned."""
        # Add data at time 0
        self.analyzer.update(0.0, "center", True, False)
        assert len(self.analyzer.gaze_history) == 1

        # Add data at time > window_size
        self.analyzer.update(15.0, "left", True, False)

        # Old data should be pruned
        assert len(self.analyzer.gaze_history) == 1
        assert self.analyzer.gaze_history[0][1] == "left"

    def test_focused_state(self):
        """Test focused state detection."""
        # Simulate focused behavior: mostly center gaze
        for i in range(100):
            gaze = "center" if i < 80 else "left"
            self.analyzer.update(float(i) * 0.1, gaze, True, False)

        metrics = self.analyzer.compute_metrics(10.0, HeadPose())
        assert metrics.center_gaze_ratio > 0.7

    def test_distracted_state(self):
        """Test distracted state detection."""
        # Simulate distracted behavior: mostly non-center gaze
        for i in range(100):
            gaze = "center" if i < 20 else "left"
            self.analyzer.update(float(i) * 0.1, gaze, True, False)

        metrics = self.analyzer.compute_metrics(10.0, HeadPose())
        assert metrics.center_gaze_ratio < 0.3

    def test_session_summary(self):
        """Test session summary generation."""
        summary = self.analyzer.get_session_summary()
        assert 'total_duration' in summary
        assert 'focused_time' in summary
        assert 'distracted_time' in summary


class TestCalibrationManager:
    """Test calibration functionality."""

    def setup_method(self):
        """Setup calibration manager."""
        from logging import getLogger
        self.config = GazeConfig()
        self.logger = getLogger("test")
        self.manager = CalibrationManager(self.config, self.logger)

    def test_initialization(self):
        """Test manager initialization."""
        assert not self.manager.is_active
        assert len(self.manager.calibration_points) == 5

    def test_start_calibration(self):
        """Test starting calibration."""
        self.manager.start()
        assert self.manager.is_active
        assert self.manager.current_point_idx == 0

    def test_add_samples(self):
        """Test adding calibration samples."""
        self.manager.start()

        # Add samples for first point
        for _ in range(self.manager.samples_per_point):
            complete = self.manager.add_sample((0.5, 0.5), (0.5, 0.5))

        # Should move to next point
        assert self.manager.current_point_idx == 1

    def test_calibration_completion(self):
        """Test full calibration process."""
        self.manager.start()

        # Complete all calibration points
        for point_idx in range(len(self.manager.calibration_points)):
            for _ in range(self.manager.samples_per_point):
                complete = self.manager.add_sample((0.5, 0.5), (0.5, 0.5))

        assert not self.manager.is_active  # Should finish


class TestDataExporter:
    """Test data export functionality."""

    def setup_method(self):
        """Setup exporter."""
        from logging import getLogger
        self.temp_dir = tempfile.mkdtemp()
        self.export_path = str(Path(self.temp_dir) / "test_export.csv")
        self.logger = getLogger("test")
        self.exporter = DataExporter(self.export_path, self.logger)

    def teardown_method(self):
        """Cleanup temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_frame(self):
        """Test adding frame data."""
        frame_data = FrameData(
            timestamp=1.0,
            gaze="center",
            blink_occurred=False,
            face_detected=True,
            focus_score=85.0,
            focus_state="focused",
            blink_count=10,
            head_pose=HeadPose(),
            fps=30.0
        )

        self.exporter.add_frame(frame_data)
        assert len(self.exporter.data_buffer) == 1

    def test_csv_export(self):
        """Test CSV export."""
        # Add some data
        for i in range(10):
            frame_data = FrameData(
                timestamp=float(i),
                gaze="center",
                blink_occurred=False,
                face_detected=True,
                focus_score=85.0,
                focus_state="focused",
                blink_count=i,
                head_pose=HeadPose(),
                fps=30.0
            )
            self.exporter.add_frame(frame_data)

        # Export
        self.exporter.export_csv()

        # Verify file exists and has content
        export_file = Path(self.exporter.export_path)
        assert export_file.exists()
        assert export_file.stat().st_size > 0

        # Verify CSV content
        import csv
        with open(export_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 10
            assert rows[0]['gaze'] == 'center'

    def test_json_export(self):
        """Test JSON export."""
        # Add some data
        for i in range(5):
            frame_data = FrameData(
                timestamp=float(i),
                gaze="left",
                blink_occurred=i % 2 == 0,
                face_detected=True,
                focus_score=75.0,
                focus_state="semi-focused",
                blink_count=i,
                head_pose=HeadPose(),
                fps=25.0
            )
            self.exporter.add_frame(frame_data)

        # Export
        self.exporter.export_json()

        # Verify file exists
        json_path = Path(self.exporter.export_path.replace('.csv', '.json'))
        assert json_path.exists()

        # Verify JSON content
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert len(data) == 5
            assert data[0]['gaze'] == 'left'


class TestUtilityFunctions:
    """Test utility functions."""

    def test_format_time(self):
        """Test time formatting."""
        assert format_time(0) == "0:00"
        assert format_time(30) == "0:30"
        assert format_time(60) == "1:00"
        assert format_time(125) == "2:05"
        assert format_time(3661) == "61:01"

    def test_get_state_color(self):
        """Test state color mapping."""
        assert get_state_color("focused") == (0, 255, 0)
        assert get_state_color("semi-focused") == (0, 200, 200)
        assert get_state_color("distracted") == (0, 165, 255)
        assert get_state_color("away") == (0, 0, 255)
        assert get_state_color("unknown") == (150, 150, 150)
        assert get_state_color("invalid") == (150, 150, 150)


class TestDataStructures:
    """Test data structure classes."""

    def test_focus_metrics(self):
        """Test FocusMetrics dataclass."""
        metrics = FocusMetrics(
            center_gaze_ratio=0.8,
            dominant_gaze="center",
            gaze_switches=5,
            blink_rate=20.0,
            blink_rate_status="normal",
            face_visible_ratio=0.95,
            face_loss_count=2,
            focus_score=85.0,
            state="focused",
            head_pose=HeadPose(pitch=5.0, yaw=-3.0, roll=1.0)
        )

        assert metrics.center_gaze_ratio == 0.8
        assert metrics.focus_score == 85.0
        assert metrics.head_pose.pitch == 5.0

    def test_frame_data(self):
        """Test FrameData dataclass."""
        data = FrameData(
            timestamp=123.45,
            gaze="center",
            blink_occurred=True,
            face_detected=True,
            focus_score=90.0,
            focus_state="focused",
            blink_count=15,
            head_pose=HeadPose(),
            fps=29.8
        )

        assert data.timestamp == 123.45
        assert data.blink_occurred is True
        assert data.fps == 29.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
