"""
Turn recorded sessions into a labelled behaviour dataset.

For each label (a thought-probe answer in ``probes.csv`` or a row of an
external ``labels.csv`` with columns ``t,label`` - e.g. observer coding or
task-performance events) the features of the window that ends right *before*
the label time are computed from ``frames.csv``.  Because features are
recomputed from the raw per-frame log, the window length and event thresholds
can be changed after data collection.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .events import EventConfig, detect_events
from .features import compute_window_features, windows_from_session
from .sample import read_frames_csv


def read_labels(session_dir: Path, source: str = "probes") -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    if source in ("probes", "both") and (session_dir / "probes.csv").is_file():
        with open(session_dir / "probes.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("label") and r["label"] != "no_response":
                    out.append((float(r["t_shown"]), r["label"]))
    if source in ("labels", "both") and (session_dir / "labels.csv").is_file():
        with open(session_dir / "labels.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("label"):
                    out.append((float(r["t"]), r["label"]))
    return sorted(out)


def session_meta(session_dir: Path) -> Dict:
    p = session_dir / "meta.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def baseline_stats(rows: List[Dict]) -> Dict[str, List[float]]:
    """Per-feature [mean, sd] over a session's sliding windows (label-independent)."""
    import numpy as np
    stats: Dict[str, List[float]] = {}
    keys = {k for r in rows for k in r if k != "t"}
    for k in sorted(keys):
        v = np.array([r.get(k, math.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            stats[k] = [float(v.mean()), float(v.std()) if v.size > 1 else 0.0]
    return stats


def build_examples(session_dir: str, window_s: float = 30.0, label_source: str = "probes",
                   aoi_names: Optional[Sequence[str]] = None, min_presence: float = 0.0,
                   event_cfg: Optional[EventConfig] = None,
                   stats_out: Optional[Dict] = None, stats_hop_s: float = 10.0) -> List[Dict]:
    """Labelled windows for one session.

    If ``stats_out`` is a dict, the session's baseline statistics (computed from
    *all* sliding windows, labelled or not) are stored in it under the session
    name - used for per-session feature normalisation without label leakage.
    """
    d = Path(session_dir)
    labels = read_labels(d, label_source)
    if not labels or not (d / "frames.csv").is_file():
        return []
    meta = session_meta(d)
    if aoi_names is None:
        aoi_names = meta.get("aoi_names", [])
    samples = read_frames_csv(d / "frames.csv")
    stream = detect_events(samples, event_cfg)
    if stats_out is not None:
        stats_out[d.name] = baseline_stats(
            windows_from_session(samples, stream, window_s, stats_hop_s, aoi_names))
    rows = []
    for t, label in labels:
        t_end = t - 0.05                      # exclude the probe display itself
        feats = compute_window_features(samples, stream.fixations, stream.saccades,
                                        stream.blinks, t_end - window_s, t_end, aoi_names)
        if feats.get("face_present_ratio", 0.0) < min_presence:
            continue
        row = {"participant": meta.get("participant", d.name.split("_")[0]),
               "session": d.name, "t": round(t, 3), "label": label, "window_s": window_s}
        row.update({k: (v if isinstance(v, float) and math.isfinite(v) else "")
                    for k, v in feats.items()})
        rows.append(row)
    return rows


def write_dataset(rows: List[Dict], path: str) -> None:
    if not rows:
        raise ValueError("no labelled windows found")
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
