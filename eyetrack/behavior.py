"""
Behaviour classification (pipeline stage 5).

Two interchangeable classifiers map a window's features to learner states:

``RuleBasedClassifier``
    Transparent, literature-informed baseline that needs no training data.
    Every state score is a weighted average of "evidence" terms (each 0..1)
    and the estimate carries the human-readable reasons, so educators see
    *why* an alert fired.  Thresholds are starting points, not validated
    cut-offs; tune them in YAML or replace with the learned model.

``LearnedBehaviorClassifier``
    A scikit-learn model trained with ``training/train_behavior.py`` on
    windows labelled by self-reports (thought probes) and/or task performance,
    as proposed in poster section 5.  Its label set is whatever the study
    collected (e.g. engaged / distracted / confused / frustrated / tired).

``StateSmoother`` adds temporal smoothing and hysteresis so labels do not
flicker, and ``AlertManager`` turns sustained states into educator alerts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

STATES = ("engaged", "distracted", "fatigued", "confused", "away")

STATE_DESCRIPTIONS = {
    "engaged": "attending to the task: on screen, steady and longer fixations",
    "distracted": "attention away from the task: off-screen gaze, head movement, scattered gaze",
    "fatigued": "drowsy or tired: long/slow blinks, high PERCLOS, yawning",
    "confused": "searching/struggling: rapid AOI switching, short fixations, brow lowering",
    "away": "no face in view",
}


@dataclass
class Evidence:
    feature: str
    lo: float
    hi: float
    weight: float
    label: str
    invert: bool = False     # True: low values are the evidence
    support_only: bool = False  # absence is not counter-evidence (e.g. yawning)

    def score(self, value: float) -> Optional[float]:
        if value is None or not np.isfinite(value):
            return None
        lo, hi = self.lo, self.hi
        x = (value - lo) / (hi - lo) if hi != lo else float(value >= hi)
        x = float(np.clip(x, 0.0, 1.0))
        return 1.0 - x if self.invert else x


def default_rules() -> Dict[str, List[Evidence]]:
    """Starting-point rules.  ``lo``/``hi`` give the ramp over which a feature
    goes from "no evidence" (0) to "full evidence" (1)."""
    return {
        "engaged": [
            Evidence("on_screen_ratio", 0.70, 0.90, 2.0, "gaze mostly on screen"),
            Evidence("fixation_dur_mean", 200.0, 350.0, 1.0, "long fixations (deep processing)"),
            Evidence("fixation_time_ratio", 0.30, 0.60, 1.0, "most time spent fixating"),
            Evidence("aoi_switch_ratio", 0.20, 0.45, 1.0, "stays within an AOI", invert=True),
            Evidence("perclos", 0.05, 0.15, 1.0, "eyes open", invert=True),
            Evidence("blink_dur_mean", 200.0, 320.0, 1.0, "normal blink speed", invert=True),
            Evidence("head_motion", 5.0, 20.0, 0.5, "head is still", invert=True),
            Evidence("brow_down_mean", 0.15, 0.35, 0.5, "relaxed brow", invert=True),
        ],
        "distracted": [
            Evidence("on_screen_ratio", 0.50, 0.85, 2.0, "gaze often off screen", invert=True),
            Evidence("head_yaw_sd", 5.0, 12.0, 1.0, "head turning away"),
            Evidence("head_motion", 15.0, 45.0, 0.5, "frequent head movement", support_only=True),
            Evidence("gaze_dispersion", 8.0, 15.0, 0.5, "scattered gaze"),
            Evidence("track_loss_count", 0.5, 3.0, 0.5, "face repeatedly lost", support_only=True),
        ],
        "fatigued": [
            Evidence("perclos", 0.08, 0.20, 2.0, "eyes closed often (PERCLOS)"),
            Evidence("blink_dur_mean", 220.0, 380.0, 1.5, "slow blinks"),
            Evidence("long_closure_count", 0.5, 2.0, 1.5, "long eye closures", support_only=True),
            Evidence("yawn_count", 0.5, 1.5, 1.0, "yawning", support_only=True),
            Evidence("blink_rate", 30.0, 45.0, 0.5, "very high blink rate", support_only=True),
        ],
        "confused": [
            Evidence("aoi_switch_ratio", 0.30, 0.55, 1.5, "rapid switching between AOIs"),
            Evidence("fixation_dur_mean", 160.0, 240.0, 1.0, "short fixations (scanning)", invert=True),
            Evidence("aoi_transition_entropy", 0.6, 0.9, 0.5, "unsystematic gaze transitions"),
            Evidence("brow_down_mean", 0.15, 0.35, 1.5, "brow lowering", support_only=True),
        ],
    }


@dataclass
class BehaviorEstimate:
    state: str
    probs: Dict[str, float]
    reasons: List[str] = field(default_factory=list)
    source: str = "rules"

    @property
    def confidence(self) -> float:
        return float(self.probs.get(self.state, 0.0))


class RuleBasedClassifier:
    source = "rules"

    def __init__(self, rules: Optional[Dict[str, List[Evidence]]] = None,
                 temperature: float = 0.12, away_presence: float = 0.5,
                 overrides: Optional[Dict] = None):
        self.rules = rules or default_rules()
        if overrides:
            self.apply_overrides(overrides)
        self.temperature = temperature
        self.away_presence = away_presence

    @property
    def labels(self) -> List[str]:
        return list(STATES)

    def apply_overrides(self, overrides: Dict) -> None:
        """``{state: {feature: {lo:.., hi:.., weight:..}}}`` from YAML."""
        for state, feats in overrides.items():
            for ev in self.rules.get(state, []):
                if ev.feature in feats:
                    for k, v in feats[ev.feature].items():
                        setattr(ev, k, v)

    def predict(self, feats: Dict[str, float]) -> BehaviorEstimate:
        presence = feats.get("face_present_ratio", 0.0)
        if presence is None or not np.isfinite(presence) or presence < self.away_presence:
            probs = {s: 0.0 for s in STATES}
            probs["away"] = 1.0
            ok = presence is not None and np.isfinite(presence)
            return BehaviorEstimate("away", probs, [f"face visible {presence * 100:.0f}% of the window"
                                                    if ok else "no face"])
        scores, reasons = {}, {}
        for state, evs in self.rules.items():
            num = den = 0.0
            rs: List[Tuple[float, str]] = []
            for ev in evs:
                e = ev.score(feats.get(ev.feature, math.nan))
                if e is None or (ev.support_only and e <= 0.0):
                    continue
                num += ev.weight * e
                den += ev.weight
                if e >= 0.5:
                    rs.append((ev.weight * e, ev.label))
            scores[state] = num / den if den > 0 else 0.0
            reasons[state] = [r for _, r in sorted(rs, reverse=True)]
        z = np.array([scores[s] for s in self.rules]) / self.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        probs = {s: float(v) for s, v in zip(self.rules, p)}
        probs["away"] = 0.0
        state = max(probs, key=probs.get)
        return BehaviorEstimate(state, probs, reasons.get(state, []), self.source)


class LearnedBehaviorClassifier:
    """Wraps a model bundle written by ``training/train_behavior.py``."""

    source = "learned"

    def __init__(self, path: str):
        import joblib
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.feature_names: List[str] = list(bundle["feature_names"])
        self.labels: List[str] = list(bundle["labels"])
        self.window_s: float = float(bundle.get("window_s", 30.0))
        self.meta: Dict = bundle.get("meta", {})
        self.baseline = None
        self.normalize_participant = bool(bundle.get("participant_normalized", False))
        self.warmup_windows = int(bundle.get("warmup_windows", 60))
        self._history: List[np.ndarray] = []

    def _vector(self, feats: Dict[str, float]) -> Optional[np.ndarray]:
        x = np.array([feats.get(n, math.nan) for n in self.feature_names], dtype=float)
        if self.normalize_participant:
            # Running session-baseline z-scoring, mirroring training (session
            # statistics over all windows).  Needs a warm-up before it is
            # meaningful; until then no prediction is made.
            self._history.append(x)
            self._history = self._history[-3600:]
            if len(self._history) < self.warmup_windows:
                return None
            hist = np.array(self._history)
            with np.errstate(all="ignore"):
                mu = np.nanmean(hist, axis=0)
                sd = np.nanstd(hist, axis=0)
            sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
            x = (x - mu) / sd
        return x

    def predict(self, feats: Dict[str, float]) -> BehaviorEstimate:
        x = self._vector(feats)
        if x is None:
            n = len(self.labels)
            return BehaviorEstimate("unknown", {c: 1.0 / n for c in self.labels},
                                    [f"collecting baseline ({len(self._history)}/"
                                     f"{self.warmup_windows} windows)"], self.source)
        x = x[None]
        proba = self.pipeline.predict_proba(x)[0]
        classes = [str(c) for c in self.pipeline.classes_]
        probs = {c: float(p) for c, p in zip(classes, proba)}
        state = max(probs, key=probs.get)
        return BehaviorEstimate(state, probs, [], self.source)


class StateSmoother:
    """Exponential smoothing of class probabilities + switching hysteresis."""

    def __init__(self, tau_s: float = 4.0, min_dwell_s: float = 3.0, margin: float = 0.05):
        self.tau_s, self.min_dwell_s, self.margin = tau_s, min_dwell_s, margin
        self._p: Optional[Dict[str, float]] = None
        self._t: Optional[float] = None
        self.state = "unknown"
        self._candidate: Optional[str] = None
        self._candidate_since = 0.0

    def update(self, t: float, est: BehaviorEstimate) -> BehaviorEstimate:
        if est.state == "unknown":        # classifier not ready: keep current state
            return BehaviorEstimate(self.state, dict(self._p or est.probs), est.reasons, est.source)
        if self._p is None or self._t is None:
            self._p = dict(est.probs)
            self.state = est.state
        else:
            a = 1.0 - math.exp(-max(t - self._t, 0.0) / self.tau_s)
            keys = set(self._p) | set(est.probs)
            self._p = {k: (1 - a) * self._p.get(k, 0.0) + a * est.probs.get(k, 0.0) for k in keys}
        self._t = t
        best = max(self._p, key=self._p.get)
        if est.state == "away":           # react to the face disappearing immediately
            self.state, self._candidate = "away", None
        elif best != self.state and self._p[best] > self._p.get(self.state, 0.0) + self.margin:
            if self._candidate != best:
                self._candidate, self._candidate_since = best, t
            elif t - self._candidate_since >= self.min_dwell_s:
                self.state, self._candidate = best, None
        else:
            self._candidate = None
        return BehaviorEstimate(self.state, dict(self._p), est.reasons, est.source)


@dataclass
class Alert:
    t: float
    state: str
    duration_s: float
    message: str


class AlertManager:
    """Raise an alert once a non-engaged state has lasted ``after_s`` seconds."""

    def __init__(self, states: Sequence[str] = ("distracted", "fatigued", "confused", "away"),
                 after_s: float = 60.0, cooldown_s: float = 120.0,
                 callback: Optional[Callable[[Alert], None]] = None):
        self.states, self.after_s, self.cooldown_s = set(states), after_s, cooldown_s
        self.callback = callback
        self._state: Optional[str] = None
        self._since = 0.0
        self._fired = False
        self._last_alert = -math.inf

    def update(self, t: float, state: str) -> Optional[Alert]:
        if state != self._state:
            self._state, self._since, self._fired = state, t, False
            return None
        if (state in self.states and not self._fired and t - self._since >= self.after_s
                and t - self._last_alert >= self.cooldown_s):
            self._fired, self._last_alert = True, t
            alert = Alert(t, state, t - self._since,
                          f"Learner {state} for {t - self._since:.0f}s - {STATE_DESCRIPTIONS.get(state, '')}")
            if self.callback:
                self.callback(alert)
            return alert
        return None


def load_classifier(model_path: Optional[str], rule_overrides: Optional[Dict] = None):
    if model_path:
        return LearnedBehaviorClassifier(model_path)
    return RuleBasedClassifier(overrides=rule_overrides)
