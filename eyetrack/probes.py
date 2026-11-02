"""
Experience-sampling "thought probes" for collecting ground-truth labels.

The poster plans to validate the behaviour model "against student task
performance and self-reports".  Thought probes are the standard way to do this
in mind-wandering / engagement research (e.g. Hutt et al., gaze-based
mind-wandering detection in classrooms): at random intervals the learner is
asked what state they were just in.  The window of eye-tracking features that
*precedes* each probe becomes one labelled training example.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

DEFAULT_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("1", "engaged"),       # "focused on the task"
    ("2", "distracted"),    # "thinking about something else / mind-wandering"
    ("3", "confused"),      # "confused / stuck"
    ("4", "fatigued"),      # "tired / sleepy"
    ("5", "frustrated"),
)

PROMPT = "What were you doing just before this appeared?"


@dataclass
class ProbeResponse:
    t_shown: float
    t_answered: float
    label: str               # option label or "no_response"
    key: str = ""

    @property
    def latency_s(self) -> float:
        return self.t_answered - self.t_shown


@dataclass
class ProbeScheduler:
    min_interval_s: float = 90.0
    max_interval_s: float = 180.0
    timeout_s: float = 20.0
    options: Sequence[Tuple[str, str]] = DEFAULT_OPTIONS
    seed: Optional[int] = None
    responses: List[ProbeResponse] = field(default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._next: Optional[float] = None
        self.active_since: Optional[float] = None

    def start(self, t: float) -> None:
        self._next = t + self._rng.uniform(self.min_interval_s, self.max_interval_s)

    @property
    def active(self) -> bool:
        return self.active_since is not None

    def update(self, t: float) -> Optional[ProbeResponse]:
        """Call every frame.  Opens a probe when due; closes it on timeout."""
        if self._next is None:
            self.start(t)
        if self.active:
            if t - self.active_since >= self.timeout_s:
                return self._close(t, "no_response", "")
            return None
        if t >= self._next:
            self.active_since = t
        return None

    def trigger(self, t: float) -> None:
        """Show a probe now (e.g. researcher hotkey)."""
        self.active_since = t

    def key(self, t: float, key: str) -> Optional[ProbeResponse]:
        if not self.active:
            return None
        for k, label in self.options:
            if key == k:
                return self._close(t, label, k)
        return None

    def _close(self, t: float, label: str, key: str) -> ProbeResponse:
        r = ProbeResponse(self.active_since, t, label, key)
        self.responses.append(r)
        self.active_since = None
        self.start(t)
        return r
