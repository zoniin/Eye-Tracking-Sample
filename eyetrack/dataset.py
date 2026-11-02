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
from .features import compute_window_features
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


def build_examples(session_dir: str, window_s: float = 30.0, label_source: str = "probes",
                   aoi_names: Optional[Sequence[str]] = None, min_presence: float = 0.0,
                   event_cfg: Optional[EventConfig] = None) -> List[Dict]:
    d = Path(session_dir)
    labels = read_labels(d, label_source)
    if not labels or not (d / "frames.csv").is_file():
        return []
    meta = session_meta(d)
    if aoi_names is None:
        aoi_names = meta.get("aoi_names", [])
    samples = read_frames_csv(d / "frames.csv")
    stream = detect_events(samples, event_cfg)
    rows = []
    for t, label in labels:
        t_end = t - 0.05                      # exclude the probe display itself
        feats = compute_window_features(samples, stream.fixations, stream.saccades,
                                        stream.blinks, t_end - window_s, t_end, aoi_names)
        if feats.get("face_present_ratio", 0.0) < min_presence:
            continue
        row = {"participant": meta.get("participant", d.name.split("_")[0]),
               "session": d.name, "t": round(t, 3), "label": label}
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
