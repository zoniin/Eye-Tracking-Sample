"""
Geometry helpers shared by the whole pipeline.

* Gaze-angle <-> 3D vector conversions for both conventions used by the
  reference repos (MPIIGaze and EVE), plus angular error.
* MediaPipe canonical face model (468 vertices, centimetres) and
  canonical-model based head-pose estimation with solvePnP.

Coordinate conventions
----------------------
Camera space is the OpenCV convention: +x right, +y down, +z forward (out of
the lens, towards the user).  A gaze vector points from the eye towards what
is being looked at, so somebody looking straight into the camera has a gaze
vector of roughly (0, 0, -1).

``pitch``/``yaw`` follow the MPIIGaze convention (Zhang et al. 2015/2017):

    g = (-cos(pitch) * sin(yaw), -sin(pitch), -cos(pitch) * cos(yaw))

so positive pitch = looking up, positive yaw = looking towards the camera's
left (the subject's right).  EVE labels are ``[pitch, yaw]`` with the
sign-flipped vector ``(cos p sin y, sin p, cos p cos y)``, i.e. the same angles;
Model_EVE read them as ``[yaw, pitch]``, see ``training/train_eve.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"


# ---------------------------------------------------------------------------
# Gaze angle conventions
# ---------------------------------------------------------------------------

def pitchyaw_to_vector(pitchyaw: np.ndarray) -> np.ndarray:
    """(N, 2) [pitch, yaw] radians -> (N, 3) unit gaze vectors (MPIIGaze)."""
    py = np.atleast_2d(np.asarray(pitchyaw, dtype=np.float64))
    pitch, yaw = py[:, 0], py[:, 1]
    v = np.stack([
        -np.cos(pitch) * np.sin(yaw),
        -np.sin(pitch),
        -np.cos(pitch) * np.cos(yaw),
    ], axis=1)
    return v


def vector_to_pitchyaw(vectors: np.ndarray) -> np.ndarray:
    """(N, 3) gaze vectors -> (N, 2) [pitch, yaw] radians (MPIIGaze)."""
    v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    pitch = np.arcsin(np.clip(-v[:, 1], -1.0, 1.0))
    yaw = np.arctan2(-v[:, 0], -v[:, 2])
    return np.stack([pitch, yaw], axis=1)


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample angular error (degrees) between two sets of gaze.

    Accepts either (N, 2) pitch/yaw arrays or (N, 3) vectors.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a.shape[1] == 2:
        a = pitchyaw_to_vector(a)
    if b.shape[1] == 2:
        b = pitchyaw_to_vector(b)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def rotation_to_pitchyaw(rotation: np.ndarray) -> Tuple[float, float]:
    """Head rotation matrix -> 2-angle head pose as used by MPIIGaze models.

    Follows Zhang et al.: take the head's z-axis expressed in camera space and
    convert it to (pitch, yaw).  For the normalised MPIIGaze data this is the
    2-D ``headpose`` input of GazeNet / the ResNet8 model.
    """
    z = rotation[:, 2]
    pitch = float(np.arcsin(np.clip(z[1], -1.0, 1.0)))
    yaw = float(np.arctan2(z[0], z[2]))
    return pitch, yaw


# ---------------------------------------------------------------------------
# Canonical face model and head pose
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def canonical_face_model() -> np.ndarray:
    """MediaPipe canonical face mesh in the OpenCV camera convention.

    Returns (468, 3) float64 in **centimetres**, origin near the head centre,
    +x towards the image right of a frontal face (subject's left), +y down,
    +z pointing away from the camera (into the head).  Extracted from the
    ``geometry_pipeline_metadata_landmarks.binarypb`` inside
    ``face_landmarker.task``.
    """
    raw = np.load(ASSET_DIR / "canonical_face_model.npy").astype(np.float64)
    # MediaPipe: y up, z towards the viewer.  OpenCV: y down, z forward.
    return raw * np.array([1.0, -1.0, -1.0])


# Landmarks that barely move with expressions (forehead, nose, eye corners,
# cheekbones).  Mouth, jaw, eyelids and eyebrows are deliberately excluded.
RIGID_LANDMARKS: Tuple[int, ...] = (
    1, 4, 5, 6, 8, 9, 10, 151, 168, 195, 197,      # midline: nose & forehead
    33, 133, 263, 362,                             # eye corners
    234, 454, 127, 356, 93, 323,                   # face sides
    116, 345, 50, 280,                             # cheekbones
)

# Eye corners (MediaPipe indices).  "right"/"left" are the subject's eyes, so
# the subject's right eye appears on the left side of a non-mirrored image.
RIGHT_EYE_CORNERS = (33, 133)    # outer, inner
LEFT_EYE_CORNERS = (263, 362)    # outer, inner


def default_camera_matrix(width: int, height: int,
                          focal_scale: float = 1.0) -> np.ndarray:
    """Pinhole intrinsics guess for an uncalibrated webcam (~53 deg HFOV)."""
    f = float(width) * focal_scale
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class HeadPose:
    """Head pose of the subject in camera space.

    ``pitch``/``yaw``/``roll`` are intuitive Euler angles in degrees:
    pitch > 0 = head tilted up, yaw > 0 = head turned towards the camera's
    left (the subject's right), roll > 0 = head tilted clockwise in the image.
    """
    rotation: np.ndarray            # (3, 3) model -> camera
    translation: np.ndarray         # (3,) centimetres
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    reprojection_error: float = 0.0  # mean pixels

    def model_to_camera(self, points: np.ndarray) -> np.ndarray:
        return (self.rotation @ np.asarray(points, dtype=np.float64).T).T + self.translation

    @property
    def distance_cm(self) -> float:
        return float(np.linalg.norm(self.translation))


def _euler_from_rotation(R: np.ndarray) -> Tuple[float, float, float]:
    # The face's forward direction (towards the camera) is -z in model space.
    fwd = R @ np.array([0.0, 0.0, -1.0])
    yaw = np.degrees(np.arctan2(-fwd[0], -fwd[2]))
    pitch = np.degrees(np.arcsin(np.clip(-fwd[1], -1.0, 1.0)))
    x_axis = R[:, 0]
    roll = np.degrees(np.arctan2(x_axis[1], x_axis[0]))
    return float(pitch), float(yaw), float(roll)


class HeadPoseEstimator:
    """solvePnP between 2-D MediaPipe landmarks and the canonical 3-D face."""

    def __init__(self, camera_matrix: np.ndarray,
                 dist_coeffs: Optional[np.ndarray] = None,
                 landmark_ids: Sequence[int] = RIGID_LANDMARKS):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = (np.zeros((4, 1)) if dist_coeffs is None
                            else np.asarray(dist_coeffs, dtype=np.float64))
        self.landmark_ids = np.asarray(landmark_ids, dtype=int)
        self.model_points = canonical_face_model()[self.landmark_ids]
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._rvec = self._tvec = None

    def estimate(self, landmarks_px: np.ndarray) -> Optional[HeadPose]:
        """``landmarks_px``: (>=468, 2+) pixel coordinates of all landmarks."""
        image_points = np.asarray(landmarks_px, dtype=np.float64)[self.landmark_ids, :2]
        use_guess = self._rvec is not None
        if use_guess:
            rvec, tvec = self._rvec.copy(), self._tvec.copy()
        else:
            rvec = np.zeros((3, 1))
            tvec = np.array([[0.0], [0.0], [60.0]])
        ok, rvec, tvec = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2, 0] <= 0:
            self.reset()
            return None
        self._rvec, self._tvec = rvec, tvec
        R, _ = cv2.Rodrigues(rvec)
        proj, _ = cv2.projectPoints(self.model_points, rvec, tvec,
                                    self.camera_matrix, self.dist_coeffs)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - image_points, axis=1)))
        pitch, yaw, roll = _euler_from_rotation(R)
        return HeadPose(rotation=R, translation=tvec.ravel(), pitch=pitch,
                        yaw=yaw, roll=roll, reprojection_error=err)


def eye_centers_3d(pose: HeadPose) -> Tuple[np.ndarray, np.ndarray]:
    """3-D eye centres (camera space, cm) of the subject's (right, left) eye.

    Uses the mid-point of the two eye corners of the canonical model, which is
    the definition used by the MPIIGaze data-normalisation procedure.
    """
    model = canonical_face_model()
    right = model[list(RIGHT_EYE_CORNERS)].mean(axis=0)
    left = model[list(LEFT_EYE_CORNERS)].mean(axis=0)
    return pose.model_to_camera(right[None])[0], pose.model_to_camera(left[None])[0]
