"""
Eye-event detection: fixations, saccades and blinks (pipeline stage 3,
"Eye event detection" in the poster).

* Fixations use the dispersion-threshold algorithm I-DT (Salvucci & Goldberg,
  ETRA 2000 - reference [3] on the poster), run online so events are emitted
  as soon as they end.  Thresholds default to values suited to webcam noise
  (2.5 deg dispersion, 100 ms minimum).
* Saccades are the transitions between consecutive fixations.
* Blinks use the eye-closure signal with hysteresis and are timed in
  milliseconds (frame-rate independent, unlike the frame counters used
  before).  Closures longer than ``long_closure_ms`` are reported as
  ``long_closure`` events - a drowsiness marker rather than a normal blink.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from .sample import FrameSample


@dataclass
class Fixation:
    start: float
    end: float
    x_deg: float
    y_deg: float
    screen_x: float = math.nan
    screen_y: float = math.nan
    aoi: str = ""
    n_samples: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Saccade:
    start: float
    end: float
    amplitude_deg: float
    from_aoi: str = ""
    to_aoi: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Blink:
    start: float
    end: float
    kind: str = "blink"          # "blink" or "long_closure"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class EventConfig:
    dispersion_deg: float = 2.5
    min_fixation_ms: float = 100.0
    max_gap_ms: float = 150.0          # tracking gap that still continues a fixation
    median_window: int = 5             # causal median pre-filter (samples); webcam gaze is noisy
    max_saccade_ms: float = 250.0      # longer gaps between fixations are not saccades
    blink_close: float = 0.5           # closure above this = eyes closed
    blink_open: float = 0.35           # closure below this = eyes open again
    min_blink_ms: float = 50.0
    long_closure_ms: float = 500.0
    max_closure_ms: float = 3000.0     # longer = eyes closed / looking down, not a blink


class FixationDetector:
    """Online I-DT."""

    def __init__(self, cfg: EventConfig):
        self.cfg = cfg
        self._win: List[FrameSample] = []
        self._in_fix = False
        self._last_t: Optional[float] = None
        self._raw: List[Tuple[float, float]] = []

    def _prefilter(self, s: FrameSample) -> FrameSample:
        """Causal moving median: removes sample noise but keeps saccade edges."""
        k = max(1, int(self.cfg.median_window))
        self._raw.append((s.gaze_x_deg, s.gaze_y_deg))
        self._raw = self._raw[-k:]
        if k == 1:
            return s
        xs, ys = zip(*self._raw)
        return replace(s, gaze_x_deg=float(np.median(xs)), gaze_y_deg=float(np.median(ys)))

    @staticmethod
    def _dispersion(win: List[FrameSample]) -> float:
        xs = [s.gaze_x_deg for s in win]
        ys = [s.gaze_y_deg for s in win]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def _make(self, win: List[FrameSample]) -> Fixation:
        xs = np.array([s.gaze_x_deg for s in win])
        ys = np.array([s.gaze_y_deg for s in win])
        sx = np.array([s.screen_x for s in win], dtype=float)
        sy = np.array([s.screen_y for s in win], dtype=float)
        aois = [s.aoi for s in win if s.aoi]
        aoi = max(set(aois), key=aois.count) if aois else ""
        return Fixation(start=win[0].t, end=win[-1].t, x_deg=float(np.median(xs)),
                        y_deg=float(np.median(ys)),
                        screen_x=float(np.nanmedian(sx)) if np.isfinite(sx).any() else math.nan,
                        screen_y=float(np.nanmedian(sy)) if np.isfinite(sy).any() else math.nan,
                        aoi=aoi, n_samples=len(win))

    def _close(self) -> Optional[Fixation]:
        fix = self._make(self._win) if self._in_fix else None
        if fix is not None and fix.duration * 1000.0 < self.cfg.min_fixation_ms:
            fix = None
        self._win, self._in_fix = [], False
        return fix

    def update(self, s: FrameSample) -> Optional[Fixation]:
        """Feed one sample; returns a Fixation when one has just ended."""
        out = None
        if not s.has_gaze:
            if self._last_t is not None and (s.t - self._last_t) * 1000.0 > self.cfg.max_gap_ms:
                out = self._close()
                self._raw = []
            return out
        if self._last_t is not None and (s.t - self._last_t) * 1000.0 > self.cfg.max_gap_ms:
            out = self._close()
            self._raw = []
        self._last_t = s.t
        s = self._prefilter(s)
        self._win.append(s)
        min_dur = self.cfg.min_fixation_ms / 1000.0
        if self._in_fix:
            if self._dispersion(self._win) > self.cfg.dispersion_deg:
                self._win.pop()
                out = self._close()
                self._win = [s]
            return out
        # Not in a fixation: slide the window until it is long enough and compact.
        while len(self._win) > 1 and self._win[-1].t - self._win[0].t >= min_dur:
            if self._dispersion(self._win) <= self.cfg.dispersion_deg:
                self._in_fix = True
                break
            self._win.pop(0)
        return out

    def flush(self) -> Optional[Fixation]:
        return self._close()


class BlinkDetector:
    def __init__(self, cfg: EventConfig):
        self.cfg = cfg
        self._closed_since: Optional[float] = None
        self._last_t: Optional[float] = None

    def update(self, s: FrameSample) -> Optional[Blink]:
        out = None
        if not s.face or math.isnan(s.closure):
            # Losing the face mid-blink makes the closure unmeasurable: drop it.
            self._closed_since = None
            return None
        if self._closed_since is None:
            if s.closure >= self.cfg.blink_close:
                self._closed_since = self._last_t if self._last_t is not None else s.t
                # start between the last open frame and this one
                self._closed_since = 0.5 * (self._closed_since + s.t)
        elif s.closure <= self.cfg.blink_open:
            end = 0.5 * ((self._last_t or s.t) + s.t)
            dur_ms = (end - self._closed_since) * 1000.0
            if self.cfg.min_blink_ms <= dur_ms <= self.cfg.max_closure_ms:
                kind = "long_closure" if dur_ms > self.cfg.long_closure_ms else "blink"
                out = Blink(self._closed_since, end, kind)
            self._closed_since = None
        self._last_t = s.t
        return out

    @property
    def eyes_closed(self) -> bool:
        return self._closed_since is not None


@dataclass
class EventStream:
    """Runs all detectors over a sample stream and keeps the events."""
    cfg: EventConfig = field(default_factory=EventConfig)
    fixations: List[Fixation] = field(default_factory=list)
    saccades: List[Saccade] = field(default_factory=list)
    blinks: List[Blink] = field(default_factory=list)

    def __post_init__(self):
        self._fix = FixationDetector(self.cfg)
        self._blink = BlinkDetector(self.cfg)

    def update(self, s: FrameSample) -> list:
        """Process one sample; returns the list of events that just ended."""
        new = []
        b = self._blink.update(s)
        if b is not None:
            self.blinks.append(b)
            new.append(b)
        # Samples with closed eyes carry no gaze information.
        gaze_sample = s
        if self._blink.eyes_closed and s.has_gaze:
            gaze_sample = FrameSample(t=s.t, face=s.face)
        f = self._fix.update(gaze_sample)
        if f is not None:
            new.extend(self._add_fixation(f))
        return new

    def _add_fixation(self, f: Fixation) -> list:
        out = []
        if self.fixations:
            prev = self.fixations[-1]
            gap = f.start - prev.end
            if 0 <= gap * 1000.0 <= self.cfg.max_saccade_ms:
                amp = math.hypot(f.x_deg - prev.x_deg, f.y_deg - prev.y_deg)
                sac = Saccade(prev.end, f.start, amp, prev.aoi, f.aoi)
                self.saccades.append(sac)
                out.append(sac)
        self.fixations.append(f)
        out.append(f)
        return out

    def flush(self) -> list:
        f = self._fix.flush()
        return self._add_fixation(f) if f is not None else []

    def prune(self, before: float) -> None:
        """Drop events that ended before ``before`` (keeps memory bounded)."""
        self.fixations = [e for e in self.fixations if e.end >= before]
        self.saccades = [e for e in self.saccades if e.end >= before]
        self.blinks = [e for e in self.blinks if e.end >= before]


def detect_events(samples: List[FrameSample], cfg: Optional[EventConfig] = None) -> EventStream:
    """Offline helper: run the detectors over a whole recording."""
    es = EventStream(cfg or EventConfig())
    for s in samples:
        es.update(s)
    es.flush()
    return es
