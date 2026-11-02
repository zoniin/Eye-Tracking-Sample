"""
MPIIGaze-style data normalisation (Sugano et al. 2014; Zhang et al. 2018,
"Revisiting Data Normalization for Appearance-Based Gaze Estimation").

Appearance-based gaze CNNs such as the ones trained in ``Model_MPIIGaze``
expect *normalised* eye patches: a virtual camera is rotated to look straight
at the eye centre from a fixed distance, cancelling head roll and distance.
Without this step a model trained on MPIIGaze's ``Data/Normalized`` patches
does not transfer to raw webcam crops - which is why the Model_MPIIGaze
scripts could only be evaluated on the dataset itself.  This module builds the
same patches from a live webcam frame using MediaPipe landmarks + head pose,
and maps the model output back to camera space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .geometry import HeadPose, rotation_to_pitchyaw

# Parameters of the MPIIGaze "Normalized" release.
MPIIGAZE_FOCAL = 960.0
MPIIGAZE_DISTANCE_CM = 60.0
MPIIGAZE_PATCH = (60, 36)   # (width, height)


@dataclass
class NormalizedEye:
    patch: np.ndarray              # (H, W) uint8, histogram equalised
    rotation: np.ndarray           # (3, 3) camera -> normalised camera
    head_pitchyaw: np.ndarray      # (2,) radians, normalised head pose
    center: np.ndarray             # (3,) eye centre in camera space (cm)
    flipped: bool = False          # True if patch was mirrored (right eye)

    def denormalize_gaze(self, pitchyaw_norm: np.ndarray) -> np.ndarray:
        """Normalised-space (pitch, yaw) -> camera-space unit gaze vector."""
        from .geometry import pitchyaw_to_vector
        py = np.asarray(pitchyaw_norm, dtype=np.float64).reshape(2).copy()
        if self.flipped:
            py[1] = -py[1]
        g_norm = pitchyaw_to_vector(py[None])[0]
        g_cam = self.rotation.T @ g_norm
        return g_cam / np.linalg.norm(g_cam)


class EyeNormalizer:
    def __init__(self, camera_matrix: np.ndarray,
                 patch_size: Tuple[int, int] = MPIIGAZE_PATCH,
                 focal: float = MPIIGAZE_FOCAL,
                 distance_cm: float = MPIIGAZE_DISTANCE_CM,
                 equalize: bool = True):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.distance = float(distance_cm)
        self.equalize = equalize
        w, h = self.patch_size
        self.norm_camera = np.array([[focal, 0.0, w / 2.0],
                                     [0.0, focal, h / 2.0],
                                     [0.0, 0.0, 1.0]], dtype=np.float64)
        self._inv_camera = np.linalg.inv(self.camera_matrix)

    def rotation_for(self, pose: HeadPose, center: np.ndarray) -> np.ndarray:
        center = np.asarray(center, dtype=np.float64)
        z = center / np.linalg.norm(center)
        head_x = pose.rotation[:, 0]
        y = np.cross(z, head_x)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        return np.stack([x, y, z], axis=0)

    def normalize(self, gray: np.ndarray, pose: HeadPose, center: np.ndarray,
                  flip: bool = False) -> NormalizedEye:
        """Warp ``gray`` (full frame, uint8) into a normalised eye patch."""
        center = np.asarray(center, dtype=np.float64)
        R_n = self.rotation_for(pose, center)
        scale = np.diag([1.0, 1.0, self.distance / np.linalg.norm(center)])
        W = self.norm_camera @ scale @ R_n @ self._inv_camera
        patch = cv2.warpPerspective(gray, W, self.patch_size)
        if self.equalize:
            patch = cv2.equalizeHist(patch)
        head_rot_n = R_n @ pose.rotation
        head_py = np.array(rotation_to_pitchyaw(head_rot_n), dtype=np.float64)
        if flip:
            patch = patch[:, ::-1].copy()
            head_py[1] = -head_py[1]
        return NormalizedEye(patch=patch, rotation=R_n, head_pitchyaw=head_py,
                             center=center, flipped=flip)
