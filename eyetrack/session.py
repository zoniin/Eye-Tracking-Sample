"""
Streaming session logger - the "Researchers (behavioural data)" output.

Each session is a folder::

    output/<participant>_<YYYYmmdd_HHMMSS>/
        meta.json      config, calibration accuracy, software versions
        frames.csv     one row per video frame (FrameSample)
        events.csv     fixations, saccades, blinks
        windows.csv    windowed features + behaviour state + class probabilities
        probes.csv     self-report probe responses (labels)
        alerts.csv     educator alerts

Rows are written as they are produced (nothing is buffered for the whole
session as in V2), so a crash loses at most a few seconds of data.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .sample import FRAME_FIELDS, FrameSample, sample_to_row

EVENT_FIELDS = ["type", "start", "end", "duration_ms", "x_deg", "y_deg", "screen_x",
                "screen_y", "aoi", "amplitude_deg", "from_aoi", "to_aoi", "kind"]
PROBE_FIELDS = ["t_shown", "t_answered", "latency_s", "label", "key"]
ALERT_FIELDS = ["t", "state", "duration_s", "message"]


def event_row(ev) -> Dict:
    from .events import Blink, Fixation, Saccade
    row = {k: "" for k in EVENT_FIELDS}
    row.update(start=f"{ev.start:.4f}", end=f"{ev.end:.4f}",
               duration_ms=f"{ev.duration * 1000.0:.1f}")
    if isinstance(ev, Fixation):
        row.update(type="fixation", x_deg=f"{ev.x_deg:.3f}", y_deg=f"{ev.y_deg:.3f}",
                   screen_x=f"{ev.screen_x:.4f}", screen_y=f"{ev.screen_y:.4f}", aoi=ev.aoi)
    elif isinstance(ev, Saccade):
        row.update(type="saccade", amplitude_deg=f"{ev.amplitude_deg:.3f}",
                   from_aoi=ev.from_aoi, to_aoi=ev.to_aoi)
    elif isinstance(ev, Blink):
        row.update(type="blink", kind=ev.kind)
    return row


class _CsvStream:
    def __init__(self, path: Path, fields: List[str]):
        self.path, self.fields = path, fields
        self._f = None
        self._w = None

    def write(self, row: Dict) -> None:
        if self._w is None:
            self._f = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self.fields, extrasaction="ignore")
            self._w.writeheader()
        self._w.writerow(row)

    def flush(self) -> None:
        if self._f:
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


class SessionWriter:
    def __init__(self, root: str = "output", participant: str = "anon",
                 meta: Optional[Dict] = None, window_fields: Optional[List[str]] = None,
                 log_frames: bool = True):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / f"{participant}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"participant": participant, "started": stamp,
                     "python": platform.python_version(), **(meta or {})}
        self.log_frames = log_frames
        self.frames = _CsvStream(self.dir / "frames.csv", FRAME_FIELDS)
        self.events = _CsvStream(self.dir / "events.csv", EVENT_FIELDS)
        self.windows: Optional[_CsvStream] = None
        self._window_fields = window_fields
        self.probes = _CsvStream(self.dir / "probes.csv", PROBE_FIELDS)
        self.alerts = _CsvStream(self.dir / "alerts.csv", ALERT_FIELDS)
        self._last_flush = time.time()
        self.write_meta()

    def write_meta(self, **extra) -> None:
        self.meta.update(extra)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def frame(self, s: FrameSample) -> None:
        if self.log_frames:
            self.frames.write(sample_to_row(s))
        self._maybe_flush()

    def event(self, ev) -> None:
        self.events.write(event_row(ev))

    def window(self, row: Dict) -> None:
        if self.windows is None:
            fields = self._window_fields or list(row.keys())
            self.windows = _CsvStream(self.dir / "windows.csv", fields)
        self.windows.write(row)

    def probe(self, r) -> None:
        self.probes.write({"t_shown": f"{r.t_shown:.3f}", "t_answered": f"{r.t_answered:.3f}",
                           "latency_s": f"{r.latency_s:.3f}", "label": r.label, "key": r.key})
        self.probes.flush()

    def alert(self, a) -> None:
        self.alerts.write({"t": f"{a.t:.3f}", "state": a.state,
                           "duration_s": f"{a.duration_s:.1f}", "message": a.message})
        self.alerts.flush()

    def _maybe_flush(self) -> None:
        if time.time() - self._last_flush > 2.0:
            for s in (self.frames, self.events, self.windows, self.probes, self.alerts):
                if s:
                    s.flush()
            self._last_flush = time.time()

    def close(self, **summary) -> None:
        for s in (self.frames, self.events, self.windows, self.probes, self.alerts):
            if s:
                s.close()
        if summary:
            self.write_meta(summary=summary)
