"""
Windowed feature calculation (pipeline stage 4, "Feature calculation").

For every time window the four poster metrics (Table 1) - fixation count,
fixation duration, blink count, blink rate - are computed together with
additional measures that the literature links to the target states:

=====================  ==========================================  ==========================
feature                meaning                                     relevant for
=====================  ==========================================  ==========================
perclos                % time eyes >= ~80 % closed (Wierwille)     fatigue
long_closure_count     closures > 500 ms                           fatigue / micro-sleep
blink_dur_mean         mean blink duration                         fatigue
yawn_count             sustained jaw opening (blendshape)          fatigue
on_screen_ratio        share of gaze samples on the screen         distraction / mind-wandering
gaze_dispersion        spread of gaze (deg)                        distraction
head_motion            mean head angular speed (deg/s)             distraction / restlessness
aoi_transition_rate    AOI switches per minute                     searching / confusion / anxiety
aoi_switch_ratio       share of fixations that change AOI          (rate-independent version)
aoi_transition_entropy normalised gaze transition entropy          searching (Krejtz et al. 2015)
fixation_dur_mean      long = deep processing, short = scanning    engagement vs searching
brow_down_mean         brow lowering (FACS AU4)                    confusion / effort
=====================  ==========================================  ==========================
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .aoi import OFF_SCREEN, OTHER
from .events import Blink, Fixation, Saccade
from .sample import FrameSample

BASE_FEATURES: List[str] = [
    # presence / tracking
    "face_present_ratio", "track_loss_count",
    # fixations (Table 1)
    "fixation_count", "fixation_rate", "fixation_dur_mean", "fixation_dur_median",
    "fixation_dur_sd", "fixation_time_ratio", "long_fixation_ratio",
    # saccades
    "saccade_count", "saccade_rate", "saccade_amp_mean",
    # blinks (Table 1) + fatigue markers
    "blink_count", "blink_rate", "blink_dur_mean", "long_closure_count", "perclos",
    # gaze spread / screen
    "gaze_dispersion", "on_screen_ratio",
    # AOI dynamics
    "aoi_transition_count", "aoi_transition_rate", "aoi_switch_ratio", "aoi_transition_entropy",
    "aoi_unique_count", "aoi_max_dwell_ratio",
    # head
    "head_yaw_sd", "head_pitch_sd", "head_motion", "head_distance_mean",
    # facial expression (blendshapes)
    "brow_down_mean", "brow_inner_up_mean", "eye_squint_mean", "eye_wide_mean",
    "jaw_open_mean", "yawn_count",
]


def feature_names(aoi_names: Sequence[str] = ()) -> List[str]:
    return BASE_FEATURES + [f"dwell_{n}" for n in aoi_names]


def _nanmean(vals) -> float:
    a = np.asarray(list(vals), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else math.nan


def _nanstd(vals) -> float:
    a = np.asarray(list(vals), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.std()) if a.size > 1 else math.nan


def transition_entropy(seq: Sequence[str]) -> float:
    """Normalised stationary gaze transition entropy (Krejtz et al. 2015).

    0 = always the same transitions (systematic reading), 1 = transitions are
    uniformly random among the visited AOIs (scanning/searching).
    """
    states = sorted(set(seq))
    k = len(states)
    if k < 2 or len(seq) < 3:
        return math.nan
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((k, k))
    for a, b in zip(seq[:-1], seq[1:]):
        if a != b:
            counts[idx[a], idx[b]] += 1
    rows = counts.sum(axis=1)
    if rows.sum() == 0:
        return 0.0
    pi = rows / rows.sum()
    h = 0.0
    for i in range(k):
        if rows[i] == 0:
            continue
        p = counts[i] / rows[i]
        p = p[p > 0]
        h -= pi[i] * float(np.sum(p * np.log2(p)))
    return float(h / math.log2(k - 1)) if k > 2 else float(h)


def _count_yawns(samples: List[FrameSample], thresh: float = 0.5, min_s: float = 1.2) -> int:
    count, start = 0, None
    for s in samples:
        open_ = s.face and not math.isnan(s.jaw_open) and s.jaw_open >= thresh
        if open_ and start is None:
            start = s.t
        elif not open_ and start is not None:
            if s.t - start >= min_s:
                count += 1
            start = None
    if start is not None and samples and samples[-1].t - start >= min_s:
        count += 1
    return count


def compute_window_features(samples: List[FrameSample], fixations: Iterable[Fixation],
                            saccades: Iterable[Saccade], blinks: Iterable[Blink],
                            t_start: float, t_end: float,
                            aoi_names: Sequence[str] = (),
                            long_fixation_s: float = 0.5,
                            perclos_threshold: float = 0.7) -> Dict[str, float]:
    """Features over ``[t_start, t_end]``.  Events are clipped to the window."""
    win = [s for s in samples if t_start <= s.t <= t_end]
    fix = [f for f in fixations if f.end >= t_start and f.start <= t_end]
    sac = [s for s in saccades if s.end >= t_start and s.start <= t_end]
    bl = [b for b in blinks if b.end >= t_start and b.start <= t_end]
    dur = max(t_end - t_start, 1e-6)
    feats: Dict[str, float] = {k: math.nan for k in feature_names(aoi_names)}

    n = len(win)
    if n == 0:
        feats["face_present_ratio"] = 0.0
        return feats
    face = np.array([s.face for s in win])
    feats["face_present_ratio"] = float(face.mean())
    feats["track_loss_count"] = float(np.sum(face[:-1] & ~face[1:])) if n > 1 else 0.0
    present_min = max(feats["face_present_ratio"] * dur / 60.0, 1e-6)

    # --- fixations -------------------------------------------------------
    durs = np.array([min(f.end, t_end) - max(f.start, t_start) for f in fix])
    feats["fixation_count"] = float(len(fix))
    feats["fixation_rate"] = len(fix) / present_min
    if len(fix):
        full = np.array([f.duration for f in fix]) * 1000.0
        feats["fixation_dur_mean"] = float(full.mean())
        feats["fixation_dur_median"] = float(np.median(full))
        feats["fixation_dur_sd"] = float(full.std()) if len(full) > 1 else 0.0
        feats["fixation_time_ratio"] = float(durs.sum() / dur)
        feats["long_fixation_ratio"] = float(np.mean(full >= long_fixation_s * 1000.0))
    else:
        feats["fixation_time_ratio"] = 0.0

    # --- saccades --------------------------------------------------------
    feats["saccade_count"] = float(len(sac))
    feats["saccade_rate"] = len(sac) / present_min
    feats["saccade_amp_mean"] = _nanmean(s.amplitude_deg for s in sac)

    # --- blinks ----------------------------------------------------------
    normal = [b for b in bl if b.kind == "blink"]
    feats["blink_count"] = float(len(normal))
    feats["blink_rate"] = len(normal) / present_min
    feats["blink_dur_mean"] = _nanmean(b.duration * 1000.0 for b in normal)
    feats["long_closure_count"] = float(sum(b.kind == "long_closure" for b in bl))
    closure = np.array([s.closure for s in win if s.face], dtype=float)
    closure = closure[np.isfinite(closure)]
    feats["perclos"] = float(np.mean(closure >= perclos_threshold)) if closure.size else math.nan

    # --- gaze spread / screen ---------------------------------------------
    gx = np.array([s.gaze_x_deg for s in win if s.has_gaze])
    gy = np.array([s.gaze_y_deg for s in win if s.has_gaze])
    if gx.size > 1:
        feats["gaze_dispersion"] = float(math.sqrt(gx.var() + gy.var()))
    aois = [s.aoi for s in win if s.face and s.aoi]
    if aois:
        feats["on_screen_ratio"] = float(np.mean([a != OFF_SCREEN for a in aois]))

    # --- AOI dynamics (fixation level) --------------------------------------
    seq = [f.aoi for f in fix if f.aoi and f.aoi not in (OFF_SCREEN,)]
    named = [a for a in seq if a != OTHER] if aoi_names else seq
    if aoi_names:
        trans = sum(1 for a, b in zip(named[:-1], named[1:]) if a != b)
        feats["aoi_transition_count"] = float(trans)
        feats["aoi_transition_rate"] = trans / present_min
        if len(named) > 1:
            feats["aoi_switch_ratio"] = trans / (len(named) - 1)
        feats["aoi_transition_entropy"] = transition_entropy(named)
        feats["aoi_unique_count"] = float(len(set(named)))
        total = durs.sum() if len(fix) else 0.0
        for name in aoi_names:
            d = sum(dd for f, dd in zip(fix, durs) if f.aoi == name)
            feats[f"dwell_{name}"] = float(d / total) if total > 0 else 0.0
        dwell_vals = [feats[f"dwell_{a}"] for a in aoi_names]
        feats["aoi_max_dwell_ratio"] = float(max(dwell_vals)) if dwell_vals else math.nan

    # --- head ---------------------------------------------------------------
    present = [s for s in win if s.face and not math.isnan(s.head_yaw)]
    feats["head_yaw_sd"] = _nanstd(s.head_yaw for s in present)
    feats["head_pitch_sd"] = _nanstd(s.head_pitch for s in present)
    feats["head_distance_mean"] = _nanmean(s.head_distance for s in present)
    if len(present) > 1:
        speeds = []
        for a, b in zip(present[:-1], present[1:]):
            dt = b.t - a.t
            if 0 < dt < 0.5:
                speeds.append(math.hypot(b.head_yaw - a.head_yaw, b.head_pitch - a.head_pitch) / dt)
        feats["head_motion"] = float(np.median(speeds)) if speeds else math.nan

    # --- expression -------------------------------------------------------------
    for key in ("brow_down", "brow_inner_up", "eye_squint", "eye_wide", "jaw_open"):
        feats[f"{key}_mean"] = _nanmean(getattr(s, key) for s in win if s.face)
    feats["yawn_count"] = float(_count_yawns(win))
    return feats


def windows_from_session(samples: List[FrameSample], stream, window_s: float = 30.0,
                         hop_s: float = 5.0, aoi_names: Sequence[str] = (),
                         t_ends: Optional[Sequence[float]] = None) -> List[Dict[str, float]]:
    """Offline: features for sliding windows (or windows ending at ``t_ends``)."""
    if not samples:
        return []
    if t_ends is None:
        t0, t1 = samples[0].t, samples[-1].t
        t_ends = np.arange(t0 + window_s, t1 + 1e-9, hop_s)
    ts = np.array([s.t for s in samples])
    rows = []
    for te in t_ends:
        lo = int(np.searchsorted(ts, te - window_s, side="left"))
        hi = int(np.searchsorted(ts, te, side="right"))
        f = compute_window_features(samples[lo:hi], stream.fixations, stream.saccades,
                                    stream.blinks, te - window_s, te, aoi_names)
        f["t"] = float(te)
        rows.append(f)
    return rows
