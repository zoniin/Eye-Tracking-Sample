"""Per-frame record shared by the live pipeline, CSV logs and offline tools."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, List

NAN = float("nan")


@dataclass
class FrameSample:
    t: float                         # seconds since session start
    face: bool = False
    ear: float = NAN                 # mean eye aspect ratio
    closure: float = NAN             # 0 = open .. 1 = closed (blendshape/EAR)
    gaze_pitch: float = NAN          # camera-space gaze, degrees
    gaze_yaw: float = NAN
    gaze_x_deg: float = NAN          # 2-D gaze in degrees used for fixations
    gaze_y_deg: float = NAN
    screen_x: float = NAN            # normalised screen coords (0..1) if calibrated
    screen_y: float = NAN
    aoi: str = ""                    # AOI name, "off_screen", "other", or "" if unknown
    head_pitch: float = NAN          # degrees
    head_yaw: float = NAN
    head_roll: float = NAN
    head_distance: float = NAN       # cm
    brow_down: float = NAN           # blendshapes (0..1)
    brow_inner_up: float = NAN
    eye_squint: float = NAN
    eye_wide: float = NAN
    jaw_open: float = NAN
    iris_right_h: float = NAN        # raw iris offsets (landmarks.iris_offset), kept for
    iris_right_v: float = NAN        # offline re-calibration / model training
    iris_left_h: float = NAN
    iris_left_v: float = NAN

    @property
    def has_gaze(self) -> bool:
        return self.face and not math.isnan(self.gaze_x_deg)


FRAME_FIELDS: List[str] = [f.name for f in fields(FrameSample)]
_TYPES = {f.name: f.type for f in fields(FrameSample)}


def sample_to_row(s: FrameSample) -> dict:
    d = asdict(s)
    d["face"] = int(s.face)
    return d


def row_to_sample(row: dict) -> FrameSample:
    kw = {}
    for name in FRAME_FIELDS:
        v = row.get(name, "")
        if name == "face":
            kw[name] = str(v).strip().lower() in ("1", "true")
        elif name == "aoi":
            kw[name] = v or ""
        else:
            kw[name] = float(v) if v not in ("", None) else NAN
    return FrameSample(**kw)


def read_frames_csv(path: Path) -> List[FrameSample]:
    with open(path, newline="") as f:
        return [row_to_sample(r) for r in csv.DictReader(f)]


def write_frames_csv(path: Path, samples: Iterable[FrameSample]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
        w.writeheader()
        for s in samples:
            w.writerow(sample_to_row(s))


def iter_valid(samples: Iterable[FrameSample]) -> Iterator[FrameSample]:
    return (s for s in samples if s.has_gaze)
