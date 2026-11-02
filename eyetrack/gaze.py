"""
Gaze estimation back-ends.

``GeometricGazeEstimator``
    Landmark-based: iris position inside the eye + head pose.  Works out of
    the box, needs no weights, and is the default.  Its absolute angles are
    rough; per-person screen calibration (``calibration.py``) turns it into a
    usable point-of-regard estimate.

``AppearanceGazeEstimator``
    Runs a CNN trained with ``training/train_mpiigaze.py`` (ResNet-8 / LeNet
    from Model_MPIIGaze) or ``training/train_eve.py`` (EyeNet-GRU from
    Model_EVE) exported to ONNX.  Eye patches are produced with the MPIIGaze
    normalisation so that the webcam input matches the training data.  The
    EyeNet-GRU is run in *streaming* mode: its hidden state is carried from
    frame to frame, which gives the temporal refinement of the EVE work at
    ~1 frame of latency.

Both return camera-space (pitch, yaw) plus a feature vector that the screen
calibration regresses onto screen coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .geometry import HeadPose, eye_centers_3d, vector_to_pitchyaw
from .landmarks import EyeMeasurements
from .normalization import EyeNormalizer

# Rough eye geometry for the uncalibrated geometric estimate: iris travel per
# unit of eye width relative to the eyeball radius (~30 mm width / 12 mm radius)
_H_GAIN = 2.5
_V_GAIN = 3.5


@dataclass
class GazeResult:
    pitchyaw: Optional[np.ndarray]   # camera-space [pitch, yaw] radians
    features: np.ndarray             # input vector for screen calibration
    source: str


def _head_features(pose: HeadPose) -> np.ndarray:
    t = pose.translation
    return np.array([np.radians(pose.pitch), np.radians(pose.yaw),
                     t[0] / t[2], t[1] / t[2]], dtype=np.float64)


class GeometricGazeEstimator:
    source = "geometric"

    def estimate(self, frame_gray: Optional[np.ndarray], pose: HeadPose,
                 eyes: EyeMeasurements) -> GazeResult:
        f = eyes.gaze_features()
        h = 0.5 * (f[0] + f[2]) - 0.5          # >0: iris towards image right
        v = 0.5 * (f[1] + f[3])                # >0: iris lower in the eye
        eye_yaw = -np.arcsin(np.clip(h * _H_GAIN, -1, 1))
        eye_pitch = -np.arcsin(np.clip(v * _V_GAIN, -1, 1))
        pitch = np.radians(pose.pitch) + eye_pitch
        yaw = np.radians(pose.yaw) + eye_yaw
        feats = np.concatenate([f, _head_features(pose)])
        return GazeResult(np.array([pitch, yaw]), feats, self.source)

    def reset(self) -> None:
        pass


class AppearanceGazeEstimator:
    """ONNX gaze CNN on normalised eye patches (needs ``onnxruntime``)."""

    source = "appearance"

    def __init__(self, model_path: str, camera_matrix: np.ndarray):
        import onnxruntime as ort

        model_path = Path(model_path)
        # calibrations are only valid for the exact model they were made with
        self.source = f"appearance:{model_path.name}"
        meta_path = model_path.with_suffix(".json")
        self.meta: Dict = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        self.arch = self.meta.get("arch", "resnet8")
        w, h = self.meta.get("input_size", [60, 36])
        norm = self.meta.get("normalization", {})
        self.normalizer = EyeNormalizer(
            camera_matrix, patch_size=(w, h),
            focal=norm.get("focal", 960.0),
            distance_cm=norm.get("distance_cm", 60.0),
            equalize=norm.get("equalize", True))
        self.flip_right = bool(self.meta.get("flip_right_eye", True))
        self.uses_head_pose = bool(self.meta.get("uses_head_pose", self.arch != "eyenet_gru"))
        self.session = ort.InferenceSession(str(model_path),
                                            providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.hidden_size = int(self.meta.get("hidden_size", 256))
        self._hidden: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._hidden = None

    def estimate(self, frame_gray: np.ndarray, pose: HeadPose,
                 eyes: EyeMeasurements) -> GazeResult:
        right_c, left_c = eye_centers_3d(pose)
        n_right = self.normalizer.normalize(frame_gray, pose, right_c, flip=self.flip_right)
        n_left = self.normalizer.normalize(frame_gray, pose, left_c, flip=False)
        normed = [n_right, n_left]
        imgs = np.stack([n.patch for n in normed]).astype(np.float32)[:, None] / 255.0
        heads = np.stack([n.head_pitchyaw for n in normed]).astype(np.float32)

        if self.arch == "eyenet_gru":
            if self._hidden is None:
                self._hidden = np.zeros((1, 2, self.hidden_size), np.float32)
            feeds = {"image": imgs[:, None], "h0": self._hidden}
            out = self.session.run(None, feeds)
            pred, self._hidden = out[0][:, -1], out[-1]
        else:
            feeds = {"image": imgs}
            if "head_pose" in self.input_names:
                feeds["head_pose"] = heads
            pred = self.session.run(None, feeds)[0]

        vecs = [n.denormalize_gaze(p) for n, p in zip(normed, pred)]
        # Skip an eye that is closed (its patch carries no gaze information)
        weights = np.array([1.0 - min(1.0, max(0.0, eyes.blink_score))] * 2) + 1e-3
        g = (vecs[0] * weights[0] + vecs[1] * weights[1]) / weights.sum()
        pitchyaw = vector_to_pitchyaw(g[None])[0]
        center = 0.5 * (right_c + left_c)
        feats = np.concatenate([
            [g[0] / -g[2], g[1] / -g[2]],
            [center[0] / center[2], center[1] / center[2]],
            _head_features(pose),
        ])
        return GazeResult(pitchyaw, feats, self.source)


def build_gaze_estimator(model_path: Optional[str], camera_matrix: np.ndarray):
    if not model_path:
        return GeometricGazeEstimator()
    return AppearanceGazeEstimator(model_path, camera_matrix)


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
