"""
End-to-end pipeline (poster Figure 1):

    webcam frame -> face & eye detection -> gaze / blink signals
      -> eye events (fixations, saccades, blinks) -> windowed features
      -> behaviour classification -> educator alerts + researcher logs

``EyeTrackingPipeline.process_frame`` runs the whole thing on a video frame.
``process_sample`` starts from an already-extracted ``FrameSample`` so the
exact same event/feature/behaviour code runs offline (re-analysing a
``frames.csv``) and in simulation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from .aoi import AngularScreenModel, AOILayout
from .behavior import (AlertManager, BehaviorEstimate, RuleBasedClassifier,
                       StateSmoother)
from .calibration import ScreenCalibration
from .events import EventConfig, EventStream
from .features import compute_window_features, feature_names
from .filters import OneEuroFilter
from .sample import FrameSample

NAN = math.nan


@dataclass
class PipelineConfig:
    window_s: float = 30.0              # feature window length
    update_interval_s: float = 1.0      # how often features/state are recomputed
    screen_width_cm: float = 34.0       # physical screen size, for gaze in degrees
    screen_height_cm: float = 19.0
    default_distance_cm: float = 60.0
    smoothing_min_cutoff: float = 1.5   # 1-Euro filter on gaze
    smoothing_beta: float = 0.02
    events: EventConfig = field(default_factory=EventConfig)
    alert_after_s: float = 60.0
    alert_states: tuple = ("distracted", "fatigued", "confused", "away")
    state_tau_s: float = 4.0
    state_min_dwell_s: float = 3.0
    camera_focal_scale: float = 1.0


@dataclass
class FrameResult:
    sample: FrameSample
    events: list
    estimate: Optional[BehaviorEstimate]
    features: Optional[Dict[str, float]]
    alert: object = None
    observation: object = None
    pose: object = None
    eyes: object = None
    gaze: object = None


class EyeTrackingPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None, tracker=None,
                 gaze_estimator=None, calibration: Optional[ScreenCalibration] = None,
                 aoi_layout: Optional[AOILayout] = None, classifier=None,
                 writer=None, camera_matrix: Optional[np.ndarray] = None):
        self.cfg = cfg or PipelineConfig()
        self.tracker = tracker
        self.gaze_estimator = gaze_estimator
        self.calibration = calibration
        self.aoi_layout = aoi_layout or AOILayout()
        self.angular_screen = AngularScreenModel()
        self.classifier = classifier or RuleBasedClassifier()
        self.writer = writer
        self.camera_matrix = camera_matrix
        self.head_pose = None
        self.events = EventStream(self.cfg.events)
        self.smoother = StateSmoother(self.cfg.state_tau_s, self.cfg.state_min_dwell_s)
        self.alerts = AlertManager(self.cfg.alert_states, self.cfg.alert_after_s)
        self._buf: Deque[FrameSample] = deque()
        self._last_update = -math.inf
        self._gaze_filter = OneEuroFilter(self.cfg.smoothing_min_cutoff, self.cfg.smoothing_beta)
        self._screen_filter = OneEuroFilter(self.cfg.smoothing_min_cutoff, self.cfg.smoothing_beta)
        self.last_estimate: Optional[BehaviorEstimate] = None
        self.last_features: Optional[Dict[str, float]] = None
        self.feature_names = feature_names(self.aoi_layout.names)

    # ------------------------------------------------------------------ frames
    def _ensure_geometry(self, width: int, height: int) -> None:
        from .geometry import HeadPoseEstimator, default_camera_matrix
        if self.camera_matrix is None:
            self.camera_matrix = default_camera_matrix(width, height, self.cfg.camera_focal_scale)
        if self.head_pose is None:
            self.head_pose = HeadPoseEstimator(self.camera_matrix)
        if self.gaze_estimator is None:
            from .gaze import GeometricGazeEstimator
            self.gaze_estimator = GeometricGazeEstimator()

    def measure(self, frame_bgr: np.ndarray, t: float):
        """Perception only: frame -> (FrameSample, obs, pose, eyes, gaze)."""
        from .gaze import to_gray
        from .landmarks import measure_eyes
        h, w = frame_bgr.shape[:2]
        self._ensure_geometry(w, h)
        obs = self.tracker.process(frame_bgr, t)
        if obs is None:
            self.head_pose.reset()
            self._gaze_filter.reset()       # don't smooth across a tracking loss
            self._screen_filter.reset()
            if hasattr(self.gaze_estimator, "reset"):
                self.gaze_estimator.reset()
            return FrameSample(t=t, face=False), None, None, None, None
        pose = self.head_pose.estimate(obs.landmarks)
        if pose is None:
            return FrameSample(t=t, face=False), obs, None, None, None
        eyes = measure_eyes(obs)
        gaze = self.gaze_estimator.estimate(to_gray(frame_bgr), pose, eyes)
        bs = obs.blendshapes
        s = FrameSample(
            t=t, face=True, ear=eyes.ear, closure=eyes.blink_score,
            head_pitch=pose.pitch, head_yaw=pose.yaw, head_roll=pose.roll,
            head_distance=pose.distance_cm,
            brow_down=0.5 * (bs.get("browDownLeft", NAN) + bs.get("browDownRight", NAN)),
            brow_inner_up=bs.get("browInnerUp", NAN),
            eye_squint=0.5 * (bs.get("eyeSquintLeft", NAN) + bs.get("eyeSquintRight", NAN)),
            eye_wide=0.5 * (bs.get("eyeWideLeft", NAN) + bs.get("eyeWideRight", NAN)),
            jaw_open=bs.get("jawOpen", NAN),
            iris_right_h=eyes.iris_right[0], iris_right_v=eyes.iris_right[1],
            iris_left_h=eyes.iris_left[0], iris_left_v=eyes.iris_left[1],
        )
        self._fill_gaze(s, gaze)
        return s, obs, pose, eyes, gaze

    def _fill_gaze(self, s: FrameSample, gaze) -> None:
        if gaze is None or gaze.pitchyaw is None:
            return
        if s.closure >= self.cfg.events.blink_close:
            return      # closed eyes: no gaze, screen point or AOI for this frame
        pitch, yaw = self._gaze_filter(np.array(np.degrees(gaze.pitchyaw)), s.t)
        s.gaze_pitch, s.gaze_yaw = float(pitch), float(yaw)
        dist = s.head_distance if np.isfinite(s.head_distance) else self.cfg.default_distance_cm
        if self.calibration is not None and gaze.source == self.calibration.source:
            sx, sy = self.calibration.predict(gaze.features)[0]
            sx, sy = self._screen_filter(np.array([sx, sy]), s.t)
            s.screen_x, s.screen_y = float(sx), float(sy)
            # gaze position in visual degrees (for I-DT), screen centre = 0
            s.gaze_x_deg = math.degrees(math.atan2((sx - 0.5) * self.cfg.screen_width_cm, dist))
            s.gaze_y_deg = math.degrees(math.atan2((sy - 0.5) * self.cfg.screen_height_cm, dist))
            s.aoi = self.aoi_layout.classify(s.screen_x, s.screen_y)
        else:
            s.gaze_x_deg, s.gaze_y_deg = s.gaze_yaw, -s.gaze_pitch
            s.aoi = self.angular_screen.classify(s.gaze_pitch, s.gaze_yaw)

    def process_frame(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        s, obs, pose, eyes, gaze = self.measure(frame_bgr, t)
        res = self.process_sample(s)
        res.observation, res.pose, res.eyes, res.gaze = obs, pose, eyes, gaze
        return res

    # ---------------------------------------------------------------- samples
    def process_sample(self, s: FrameSample) -> FrameResult:
        new_events = self.events.update(s)
        self._buf.append(s)
        while self._buf and self._buf[0].t < s.t - self.cfg.window_s:
            self._buf.popleft()
        if self.writer:
            self.writer.frame(s)
            for ev in new_events:
                self.writer.event(ev)
        est, feats, alert = None, None, None
        if s.t - self._last_update >= self.cfg.update_interval_s:
            self._last_update = s.t
            feats, est, alert = self._update_state(s.t)
        return FrameResult(s, new_events, est, feats, alert)

    def _update_state(self, t: float):
        t0 = max(t - self.cfg.window_s, self._buf[0].t if self._buf else t)
        feats = compute_window_features(list(self._buf), self.events.fixations,
                                        self.events.saccades, self.events.blinks,
                                        t0, t, self.aoi_layout.names)
        self.events.prune(t - self.cfg.window_s - 5.0)
        raw = self.classifier.predict(feats)
        est = self.smoother.update(t, raw)
        alert = self.alerts.update(t, est.state)
        self.last_estimate, self.last_features = est, feats
        if self.writer:
            row = {"t": round(t, 3), "window_s": round(t - t0, 2), "state": est.state,
                   "raw_state": raw.state, "confidence": round(est.confidence, 4),
                   "reasons": "; ".join(est.reasons)}
            row.update({f"p_{k}": round(v, 4) for k, v in est.probs.items()})
            row.update({k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else "")
                        for k, v in feats.items()})
            self.writer.window(row)
            if alert:
                self.writer.alert(alert)
        return feats, est, alert

    def flush(self) -> List:
        evs = self.events.flush()
        if self.writer:
            for ev in evs:
                self.writer.event(ev)
        return evs
