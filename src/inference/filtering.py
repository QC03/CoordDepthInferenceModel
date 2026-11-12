import math
from dataclasses import dataclass
from typing import Dict, Hashable, Optional


@dataclass
class _State:
    value: float
    derivative: float
    timestamp: float


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 derivative_cutoff: float = 1.0, max_dt: float = 1.0) -> None:
        self.min_cutoff, self.beta = min_cutoff, beta
        self.derivative_cutoff, self.max_dt = derivative_cutoff, max_dt
        self._states: Dict[Hashable, _State] = {}

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self, track_id: Optional[Hashable] = None) -> None:
        if track_id is None:
            self._states.clear()
        else:
            self._states.pop(track_id, None)

    def filter(self, value: float, timestamp: float, track_id: Hashable = 0) -> float:
        previous = self._states.get(track_id)
        if previous is None:
            self._states[track_id] = _State(value, 0.0, timestamp)
            return value
        dt = timestamp - previous.timestamp
        if dt <= 0 or dt > self.max_dt:
            self.reset(track_id)
            self._states[track_id] = _State(value, 0.0, timestamp)
            return value
        derivative = (value - previous.value) / dt
        a_d = self._alpha(self.derivative_cutoff, dt)
        derivative = a_d * derivative + (1 - a_d) * previous.derivative
        cutoff = self.min_cutoff + self.beta * abs(derivative)
        a = self._alpha(cutoff, dt)
        filtered = a * value + (1 - a) * previous.value
        self._states[track_id] = _State(filtered, derivative, timestamp)
        return filtered
