"""
Synthetic gaze-stream generator.

**For testing and demos only - not evidence about real learners.**  It
produces ``FrameSample`` streams whose statistics follow the qualitative
patterns described in the literature for each state (e.g. long blinks and high
PERCLOS when fatigued, rapid AOI switching when confused) so that the event,
feature, behaviour and training code can be exercised end to end without a
webcam or a study dataset.  Any classifier trained on this data only shows
that the code runs; real accuracy must come from labelled study sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aoi import AOILayout
from .sample import FrameSample

SIM_STATES = ("engaged", "distracted", "fatigued", "confused")


@dataclass
class StateParams:
    fix_ms: float            # mean fixation duration
    p_stay: float            # P(next fixation stays in the same AOI)
    p_off: float             # P(fixation off screen)
    blink_per_min: float
    blink_ms: float
    p_long_closure: float
    head_sd: float           # deg, slow head wander
    p_head_turn: float       # per-second probability of a large head turn
    brow: float
    yawn_per_min: float
    p_face_loss: float       # per-second probability of losing the face briefly


PARAMS: Dict[str, StateParams] = {
    "engaged":    StateParams(330, 0.80, 0.03, 13, 120, 0.00, 1.5, 0.01, 0.08, 0.0, 0.002),
    "distracted": StateParams(260, 0.50, 0.40, 17, 130, 0.01, 4.0, 0.12, 0.08, 0.0, 0.03),
    "fatigued":   StateParams(380, 0.75, 0.10, 22, 300, 0.15, 2.0, 0.02, 0.06, 0.8, 0.005),
    "confused":   StateParams(175, 0.25, 0.04, 15, 125, 0.00, 2.0, 0.02, 0.32, 0.0, 0.002),
}

def default_layout() -> AOILayout:
    from .aoi import AOI
    return AOILayout(aois=[AOI("instructions", 0.0, 0.05, 0.28, 0.95),
                           AOI("code", 0.28, 0.05, 0.72, 0.60),
                           AOI("output", 0.28, 0.65, 0.47, 0.35),
                           AOI("feedback", 0.75, 0.65, 0.25, 0.35)])


@dataclass
class Participant:
    blink_mult: float = 1.0
    fix_mult: float = 1.0
    brow_base: float = 0.0
    noise_deg: float = 0.6

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Participant":
        return cls(blink_mult=float(rng.uniform(0.6, 1.6)), fix_mult=float(rng.uniform(0.8, 1.2)),
                   brow_base=float(rng.uniform(-0.04, 0.08)), noise_deg=float(rng.uniform(0.4, 1.0)))


def simulate_segment(state: str, t0: float, duration_s: float, rng: np.random.Generator,
                     who: Participant, layout: Optional[AOILayout] = None, fps: float = 30.0,
                     screen_cm: Tuple[float, float] = (34.0, 19.0), distance_cm: float = 60.0
                     ) -> List[FrameSample]:
    layout = layout or default_layout()
    dt = 1.0 / fps
    n = int(duration_s * fps)
    ts = t0 + np.arange(n) * dt
    if state == "away":
        return [FrameSample(t=float(t), face=False) for t in ts]
    P = PARAMS[state]

    # --- fixation sequence (screen coordinates) ---------------------------
    sx = np.zeros(n)
    sy = np.zeros(n)
    aoi_idx = int(rng.integers(len(layout.aois)))
    sac_frames = max(1, int(round(0.04 * fps)))     # ~40 ms saccades
    i, prev = 0, None
    while i < n:
        dur = rng.gamma(4.0, P.fix_ms * who.fix_mult / 4.0) / 1000.0
        k = max(1, int(dur * fps))
        if rng.random() < P.p_off:
            x, y = float(rng.choice([-0.4, 1.4])), float(rng.uniform(-0.2, 1.2))
        else:
            if rng.random() > P.p_stay:
                aoi_idx = int(rng.integers(len(layout.aois)))
            a = layout.aois[aoi_idx]
            x, y = a.x + rng.uniform(0.1, 0.9) * a.w, a.y + rng.uniform(0.1, 0.9) * a.h
        if prev is not None:
            for m in range(sac_frames):
                if i >= n:
                    break
                f = (m + 1) / (sac_frames + 1)
                sx[i], sy[i] = prev[0] + (x - prev[0]) * f, prev[1] + (y - prev[1]) * f
                i += 1
        sx[i:i + k], sy[i:i + k] = x, y
        i += k
        prev = (x, y)
    noise = who.noise_deg / math.degrees(math.atan2(screen_cm[0], distance_cm))
    sx = sx + rng.normal(0, noise, n)
    sy = sy + rng.normal(0, noise, n)

    # --- blinks / closures ----------------------------------------------------
    closure = np.zeros(n)
    rate = P.blink_per_min * who.blink_mult / 60.0
    t = rng.exponential(1.0 / rate)
    while t < duration_s:
        d = P.blink_ms / 1000.0 * rng.uniform(0.7, 1.4)
        if rng.random() < P.p_long_closure:
            d = rng.uniform(0.6, 1.5)
        a, b = int(t * fps), int((t + d) * fps) + 1
        closure[a:b] = 1.0
        t += d + rng.exponential(1.0 / rate)
    closure = np.clip(closure + rng.normal(0, 0.03, n), 0, 1)

    # --- head, face loss, yawns, brows -------------------------------------
    yaw = np.cumsum(rng.normal(0, P.head_sd * 0.05, n))
    yaw -= np.linspace(0, yaw[-1], n)                 # keep it bounded
    pitch = np.cumsum(rng.normal(0, P.head_sd * 0.03, n)) - 8.0
    face = np.ones(n, bool)
    jaw = np.clip(rng.normal(0.05, 0.02, n), 0, 1)
    for sec in range(int(duration_s)):
        a = sec * int(fps)
        if rng.random() < P.p_head_turn:
            m = int(rng.uniform(1.0, 3.0) * fps)
            yaw[a:a + m] += rng.choice([-1, 1]) * rng.uniform(20, 40)
            sx[a:a + m] = rng.choice([-0.5, 1.5])
        if rng.random() < P.p_face_loss:
            face[a:a + int(rng.uniform(1, 3) * fps)] = False
        if rng.random() < P.yawn_per_min / 60.0:
            jaw[a:a + int(rng.uniform(2.0, 4.0) * fps)] = rng.uniform(0.6, 0.9)
    brow = np.clip(P.brow + who.brow_base + rng.normal(0, 0.05, n), 0, 1)

    out = []
    for j in range(n):
        if not face[j]:
            out.append(FrameSample(t=float(ts[j]), face=False))
            continue
        x, y = float(sx[j]), float(sy[j])
        gx = math.degrees(math.atan2((x - 0.5) * screen_cm[0], distance_cm))
        gy = math.degrees(math.atan2((y - 0.5) * screen_cm[1], distance_cm))
        closed = closure[j] > 0.5
        out.append(FrameSample(
            t=float(ts[j]), face=True, ear=0.30 - 0.18 * closure[j], closure=float(closure[j]),
            gaze_pitch=-gy - 15.0, gaze_yaw=gx,
            gaze_x_deg=math.nan if closed else gx, gaze_y_deg=math.nan if closed else gy,
            screen_x=math.nan if closed else x, screen_y=math.nan if closed else y,
            aoi="" if closed else layout.classify(x, y),
            head_pitch=float(pitch[j]), head_yaw=float(yaw[j]), head_roll=0.0,
            head_distance=distance_cm, brow_down=float(brow[j]), brow_inner_up=0.1,
            eye_squint=0.2, eye_wide=0.05, jaw_open=float(jaw[j])))
    return out


def simulate_session(states: Sequence[Tuple[str, float]], seed: int = 0, fps: float = 30.0,
                     participant: Optional[Participant] = None,
                     layout: Optional[AOILayout] = None) -> Tuple[List[FrameSample], List[Tuple[float, str]]]:
    """Concatenate ``(state, seconds)`` segments.

    Returns the samples and the ground-truth ``(segment_end_time, state)``
    pairs, i.e. where a thought probe would have been answered.
    """
    rng = np.random.default_rng(seed)
    who = participant or Participant.random(rng)
    samples: List[FrameSample] = []
    truth = []
    t = 0.0
    for state, dur in states:
        seg = simulate_segment(state, t, dur, rng, who, layout, fps)
        samples.extend(seg)
        t += dur
        truth.append((t, state))
    return samples, truth
