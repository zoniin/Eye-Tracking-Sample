"""Time-based measurements for behavior research; no learned behavior labels."""
from collections import deque
from dataclasses import dataclass
import math


@dataclass
class EyeObservation:
    ear: float | None = None
    closure_seconds: float = 0.0
    blink_duration_seconds: float | None = None
    closed_time_ratio: float | None = None
    observed_seconds: float = 0.0
    blink: bool = False


class TemporalEyes:
    def __init__(self, threshold=0.21, window=30.0, min_blink=0.06,
                 max_blink=0.8, max_gap=0.25):
        self.threshold = threshold
        self.window = window
        self.min_blink = min_blink
        self.max_blink = max_blink
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.previous = None
        self.closed_since = None
        self.intervals = deque()

    def update(self, timestamp, ear):
        if not math.isfinite(timestamp):
            raise ValueError('Timestamp must be finite')
        valid = ear is not None and math.isfinite(ear) and ear >= 0
        closed = valid and ear < self.threshold
        result = EyeObservation(ear=float(ear) if valid else None)
        if self.previous is not None:
            prev_time, prev_valid, prev_closed = self.previous
            dt = timestamp - prev_time
            if dt <= 0:
                self.reset()
            elif dt <= self.max_gap and prev_valid and valid:
                self.intervals.append((prev_time, timestamp, prev_closed))
                if not closed and self.closed_since is not None:
                    duration = timestamp - self.closed_since
                    result.blink_duration_seconds = duration
                    result.blink = self.min_blink <= duration <= self.max_blink
                    self.closed_since = None
            else:
                self.closed_since = None
        if not valid:
            self.closed_since = None
        elif closed and self.closed_since is None:
            self.closed_since = timestamp
        if self.closed_since is not None:
            result.closure_seconds = timestamp - self.closed_since
        self.previous = (timestamp, valid, closed)
        cutoff = timestamp - self.window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        observed = closed_time = 0.0
        for start, end, was_closed in self.intervals:
            duration = end - max(start, cutoff)
            observed += duration
            if was_closed:
                closed_time += duration
        result.observed_seconds = observed
        if observed > 0:
            result.closed_time_ratio = closed_time / observed
        return result


class PositionSmoother:
    """Smooth each eye independently; discard history on loss or closure."""
    def __init__(self, window=3):
        self.history = [deque(maxlen=max(1, window)), deque(maxlen=max(1, window))]

    def reset(self):
        for values in self.history:
            values.clear()

    def update(self, left, right):
        output = []
        for history, position in zip(self.history, (left, right)):
            if position is None or not all(math.isfinite(v) for v in position):
                history.clear()
                output.append(None)
            else:
                history.append(position)
                output.append(tuple(sum(p[i] for p in history) / len(history) for i in (0, 1)))
        return tuple(output)


def canonical_gaze_angles(values, source):
    """Return named yaw/pitch radians. Does NOT transform camera coordinates.

    MPIIGaze scripts emit pitch,yaw; Taha's EVE scripts declare yaw,pitch.
    Dataset coordinate conventions must still be verified for a checkpoint.
    """
    a, b = map(float, values)
    if not all(math.isfinite(v) for v in (a, b)):
        raise ValueError('Gaze angles must be finite')
    if source == 'mpiigaze':
        return {'yaw_rad': b, 'pitch_rad': a}
    if source == 'eve':
        return {'yaw_rad': a, 'pitch_rad': b}
    raise ValueError('Expected mpiigaze or eve')
