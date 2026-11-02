"""
Person-specific screen calibration (poster: "the pipeline performs
webcam-based calibration where users fixate on several targets on the screen,
allowing the system to learn person specific mapping of the eyes and screen
coordinates").

The previous calibration only re-centred the left/centre/right thresholds.
This one learns a regression from the gaze feature vector (iris offsets +
head pose for the geometric estimator, or CNN gaze + eye position for the
appearance estimator) to normalised screen coordinates:

* features are standardised, expanded to a 1st or 2nd order polynomial and fit
  with ridge regression (closed form, numpy only);
* polynomial degree and ridge strength are chosen by **leave-one-point-out**
  cross-validation, and that held-out error is reported - an honest estimate
  of on-screen accuracy for this person/session.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def grid_points(n: int = 9, margin: float = 0.08) -> List[Tuple[float, float]]:
    """Calibration targets in normalised screen coordinates (5, 9 or 13)."""
    lo, mid, hi = margin, 0.5, 1.0 - margin
    nine = [(x, y) for y in (lo, mid, hi) for x in (lo, mid, hi)]
    if n == 5:
        return [(mid, mid), (lo, lo), (hi, lo), (lo, hi), (hi, hi)]
    if n == 9:
        return nine
    if n == 13:
        q1, q3 = 0.29, 0.71
        return nine + [(q1, q1), (q3, q1), (q1, q3), (q3, q3)]
    raise ValueError("n must be 5, 9 or 13")


def _poly(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones(len(x))]
    cols += [x[:, i] for i in range(x.shape[1])]
    if degree >= 2:
        cols += [x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2)]
    return np.stack(cols, axis=1)


def _ridge(A: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0.0                      # do not penalise the intercept
    return np.linalg.solve(A.T @ A + reg, A.T @ Y)


@dataclass
class ScreenCalibration:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray                     # (n_terms, 2)
    degree: int
    alpha: float
    loo_error: float                     # mean Euclidean error, normalised screen units
    source: str = "geometric"
    n_points: int = 0
    created: float = field(default_factory=time.time)

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(features) - self.mean) / self.scale
        return _poly(x, self.degree) @ self.coef

    def error_degrees(self, screen_width_cm: float, distance_cm: float) -> float:
        return float(np.degrees(np.arctan2(self.loo_error * screen_width_cm, distance_cm)))

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "degree": self.degree, "alpha": self.alpha,
                "loo_error": self.loo_error, "source": self.source,
                "n_points": self.n_points, "created": self.created}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "ScreenCalibration":
        d = json.loads(Path(path).read_text())
        return cls(mean=np.array(d["mean"]), scale=np.array(d["scale"]),
                   coef=np.array(d["coef"]), degree=int(d["degree"]),
                   alpha=float(d["alpha"]), loo_error=float(d["loo_error"]),
                   source=d.get("source", "geometric"), n_points=int(d.get("n_points", 0)),
                   created=float(d.get("created", 0.0)))


def fit_calibration(features: np.ndarray, targets: np.ndarray, groups: np.ndarray,
                    source: str = "geometric",
                    degrees: Sequence[int] = (1, 2),
                    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)) -> ScreenCalibration:
    """Fit features (N, D) -> targets (N, 2); ``groups`` = calibration point id."""
    X = np.asarray(features, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(groups)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y, groups = X[ok], Y[ok], groups[ok]
    uniq = np.unique(groups)
    if len(uniq) < 4:
        raise ValueError("need samples from at least 4 calibration points")
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Xs = (X - mean) / scale

    best = None
    for deg in degrees:
        A = _poly(Xs, deg)
        for alpha in alphas:
            errs = []
            for g in uniq:
                tr, te = groups != g, groups == g
                coef = _ridge(A[tr], Y[tr], alpha)
                pred = A[te] @ coef
                errs.append(np.linalg.norm(pred.mean(axis=0) - Y[te].mean(axis=0)))
            err = float(np.mean(errs))
            if best is None or err < best[0]:
                best = (err, deg, alpha)
    err, deg, alpha = best
    coef = _ridge(_poly(Xs, deg), Y, alpha)
    return ScreenCalibration(mean=mean, scale=scale, coef=coef, degree=deg, alpha=alpha,
                             loo_error=err, source=source, n_points=len(uniq))


class CalibrationSession:
    """Drives a fixation-target calibration and collects samples.

    Call ``update(t, features, valid)`` every frame; read ``current_target``
    to draw the dot.  Each target is shown for ``settle_s + collect_s``
    seconds; only the ``collect_s`` part is used (the first part lets the eyes
    land on the target).  Frames with closed eyes should pass ``valid=False``.
    """

    def __init__(self, n_points: int = 9, settle_s: float = 0.8, collect_s: float = 1.2,
                 shuffle: bool = True, seed: Optional[int] = None):
        pts = grid_points(n_points)
        if shuffle:
            rng = np.random.default_rng(seed)
            centre = pts.index((0.5, 0.5))
            rest = [p for i, p in enumerate(pts) if i != centre]
            rng.shuffle(rest)
            pts = [pts[centre]] + rest
        self.points = pts
        self.settle_s, self.collect_s = settle_s, collect_s
        self.idx = 0
        self._t0: Optional[float] = None
        self.features: List[np.ndarray] = []
        self.targets: List[Tuple[float, float]] = []
        self.groups: List[int] = []

    @property
    def done(self) -> bool:
        return self.idx >= len(self.points)

    @property
    def current_target(self) -> Optional[Tuple[float, float]]:
        return None if self.done else self.points[self.idx]

    def progress(self, t: float) -> float:
        if self._t0 is None or self.done:
            return 0.0
        return min(1.0, (t - self._t0) / (self.settle_s + self.collect_s))

    def update(self, t: float, features: Optional[np.ndarray], valid: bool = True) -> None:
        if self.done:
            return
        if self._t0 is None:
            self._t0 = t
        el = t - self._t0
        if el >= self.settle_s and valid and features is not None and np.isfinite(features).all():
            self.features.append(np.asarray(features, dtype=np.float64))
            self.targets.append(self.points[self.idx])
            self.groups.append(self.idx)
        if el >= self.settle_s + self.collect_s:
            self.idx += 1
            self._t0 = t

    def fit(self, source: str = "geometric") -> ScreenCalibration:
        return fit_calibration(np.array(self.features), np.array(self.targets),
                               np.array(self.groups), source=source)
