"""
Areas of interest (AOIs) on the screen, e.g. code / instructions / compiler
output / feedback panes of a programming environment (poster section 5).

AOIs are rectangles in *normalised* screen coordinates (0..1, origin top-left)
so a layout works at any resolution.  Example ``configs/aoi_programming.yaml``::

    aois:
      - {name: instructions, x: 0.00, y: 0.05, w: 0.28, h: 0.95}
      - {name: code,         x: 0.28, y: 0.05, w: 0.72, h: 0.65}
      - {name: output,       x: 0.28, y: 0.70, w: 0.72, h: 0.30}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

OFF_SCREEN = "off_screen"
OTHER = "other"


@dataclass
class AOI:
    name: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, sx: float, sy: float) -> bool:
        return self.x <= sx < self.x + self.w and self.y <= sy < self.y + self.h


@dataclass
class AOILayout:
    aois: List[AOI] = field(default_factory=list)
    margin: float = 0.05     # tolerance outside the screen edge before "off_screen"

    @property
    def names(self) -> List[str]:
        return [a.name for a in self.aois]

    def classify(self, sx: float, sy: float) -> str:
        if math.isnan(sx) or math.isnan(sy):
            return ""
        m = self.margin
        if sx < -m or sx > 1 + m or sy < -m or sy > 1 + m:
            return OFF_SCREEN
        for a in self.aois:
            if a.contains(min(max(sx, 0.0), 0.999), min(max(sy, 0.0), 0.999)):
                return a.name
        return OTHER

    @classmethod
    def from_yaml(cls, path: str) -> "AOILayout":
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
        aois = [AOI(**a) for a in data.get("aois", [])]
        return cls(aois=aois, margin=float(data.get("margin", 0.05)))


@dataclass
class AngularScreenModel:
    """Fallback on/off-screen test when no screen calibration is available.

    With a laptop webcam above the display, looking at the screen means a
    downward camera-space gaze pitch and a moderate yaw.  The limits are
    deliberately generous; calibrate for anything finer than on/off screen.
    """
    yaw_limit_deg: float = 25.0
    pitch_min_deg: float = -40.0
    pitch_max_deg: float = 8.0

    def classify(self, pitch_deg: float, yaw_deg: float) -> str:
        if math.isnan(pitch_deg) or math.isnan(yaw_deg):
            return ""
        if abs(yaw_deg) > self.yaw_limit_deg or not (self.pitch_min_deg <= pitch_deg <= self.pitch_max_deg):
            return OFF_SCREEN
        return OTHER
