"""
Face / eye perception on top of the MediaPipe Tasks FaceLandmarker.

The legacy ``mp.solutions.face_mesh`` API used by ``eye_tracker_v2.py`` was
removed from recent MediaPipe releases; this module uses the Tasks API with the
``face_landmarker.task`` bundle that ships in the repository and also exposes
the 52 ARKit-style blendshapes (eyeBlink, browDown, jawOpen, ...), which are
useful behavioural features (brow lowering ~ confusion, jaw opening ~ yawning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# 6-point eye contours in EAR order: outer corner, top x2, inner corner, bottom x2
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (263, 387, 385, 362, 380, 373)
RIGHT_IRIS = (468, 469, 470, 471, 472)   # centre first
LEFT_IRIS = (473, 474, 475, 476, 477)
# Upper / lower eyelid mid points (used for vertical iris position)
RIGHT_LID = (159, 145)
LEFT_LID = (386, 374)


def default_model_path() -> Optional[Path]:
    """Locate ``face_landmarker.task`` (package assets, repo root or CWD)."""
    here = Path(__file__).resolve().parent
    for cand in (here / "assets" / "face_landmarker.task",
                 here.parent / "face_landmarker.task",
                 Path.cwd() / "face_landmarker.task"):
        if cand.is_file():
            return cand
    return None


@dataclass
class FaceObservation:
    """Everything the perception layer extracts from a single frame."""
    landmarks: np.ndarray                 # (478, 3) pixels (z scaled like x)
    image_size: Tuple[int, int]           # (width, height)
    blendshapes: Dict[str, float] = field(default_factory=dict)

    def points(self, ids) -> np.ndarray:
        return self.landmarks[list(ids), :2]


class FaceLandmarkTracker:
    """Thin wrapper around ``mediapipe.tasks.vision.FaceLandmarker``.

    ``mode='video'`` uses temporal tracking (fast, needs increasing
    timestamps); ``mode='image'`` processes frames independently.
    """

    def __init__(self, model_path: Optional[str] = None, mode: str = "video",
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 blendshapes: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = Path(model_path) if model_path else default_model_path()
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(
                "face_landmarker.task not found. Download it from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task")
        self._mp = mp
        self.mode = mode
        running_mode = (vision.RunningMode.VIDEO if mode == "video"
                        else vision.RunningMode.IMAGE)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=blendshapes,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray,
                timestamp_s: Optional[float] = None) -> Optional[FaceObservation]:
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            ts = int(round((timestamp_s or 0.0) * 1000.0))
            ts = max(ts, self._last_ts_ms + 1)   # must be strictly increasing
            self._last_ts_ms = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        arr = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float64)
        shapes: Dict[str, float] = {}
        if result.face_blendshapes:
            shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        return FaceObservation(landmarks=arr, image_size=(w, h), blendshapes=shapes)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover - mediapipe teardown quirks
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Eye measurements
# ---------------------------------------------------------------------------

def eye_aspect_ratio(pts: np.ndarray) -> float:
    """Soukupova & Cech EAR from 6 points (outer, top1, top2, inner, bot1, bot2)."""
    pts = np.asarray(pts, dtype=np.float64)
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return 0.0 if c < 1e-9 else float((a + b) / (2.0 * c))


def iris_offset(iris_center: np.ndarray, outer: np.ndarray, inner: np.ndarray,
                upper_lid: np.ndarray, lower_lid: np.ndarray) -> Tuple[float, float]:
    """Iris position in an eye-aligned frame, robust to head roll and scale.

    Returns ``(h, v)``: ``h`` is the projection of the iris centre onto the
    outer->inner corner axis (0 = outer corner, 1 = inner corner);
    ``v`` is the signed perpendicular offset from the corner line divided by
    the eye width (positive = downwards in the image), corrected by the
    eyelid mid-point so that it is ~0 for a neutral gaze.
    """
    axis = inner - outer
    width = np.linalg.norm(axis)
    if width < 1e-9:
        return 0.5, 0.0
    u = axis / width
    n = np.array([-u[1], u[0]])
    if n[1] < 0:      # make the normal point down in the image
        n = -n
    h = float(np.dot(iris_center - outer, u) / width)
    lid_mid = 0.5 * (upper_lid + lower_lid)
    v = float(np.dot(iris_center - lid_mid, n) / width)
    return h, v


@dataclass
class EyeMeasurements:
    ear_right: float
    ear_left: float
    iris_right: Tuple[float, float]    # (h, v) see iris_offset
    iris_left: Tuple[float, float]
    blink_score: float                 # 0 (open) .. 1 (closed), blendshape based if available

    @property
    def ear(self) -> float:
        return 0.5 * (self.ear_right + self.ear_left)

    def gaze_features(self) -> np.ndarray:
        """Features for screen calibration.  Horizontal offsets are expressed
        so that a larger value = looking towards the image right for both
        eyes (the right eye's inner corner is on the image right, the left
        eye's inner corner on the image left)."""
        return np.array([
            self.iris_right[0], self.iris_right[1],
            1.0 - self.iris_left[0], self.iris_left[1],
        ], dtype=np.float64)


def measure_eyes(obs: FaceObservation) -> EyeMeasurements:
    lm = obs.landmarks[:, :2]
    ear_r = eye_aspect_ratio(lm[list(RIGHT_EYE_EAR)])
    ear_l = eye_aspect_ratio(lm[list(LEFT_EYE_EAR)])
    iris_r = iris_offset(lm[RIGHT_IRIS[0]], lm[RIGHT_EYE_EAR[0]], lm[RIGHT_EYE_EAR[3]],
                         lm[RIGHT_LID[0]], lm[RIGHT_LID[1]])
    iris_l = iris_offset(lm[LEFT_IRIS[0]], lm[LEFT_EYE_EAR[0]], lm[LEFT_EYE_EAR[3]],
                         lm[LEFT_LID[0]], lm[LEFT_LID[1]])
    bs = obs.blendshapes
    if "eyeBlinkLeft" in bs and "eyeBlinkRight" in bs:
        blink = 0.5 * (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"])
    else:
        # Map EAR ~0.30 (open) .. ~0.12 (closed) to 0..1
        blink = float(np.clip((0.30 - 0.5 * (ear_r + ear_l)) / 0.18, 0.0, 1.0))
    return EyeMeasurements(ear_r, ear_l, iris_r, iris_l, blink)
