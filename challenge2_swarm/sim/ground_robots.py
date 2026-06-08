"""
Simulated RoboMaster convoy for Challenge 2 dry-run.

Robots loiter (slowly drift) within the arena, like the real convoy that
"loiters for a period of time" per the briefing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GroundRobot:
    robot_id: int
    n0: float
    e0: float
    drift_radius: float = 0.05
    drift_speed: float = 0.2  # radians per sim second
    _t: float = 0.0

    def step(self, dt: float) -> None:
        self._t += dt

    def position(self) -> tuple[float, float]:
        # Small circular loiter around the spawn point
        n = self.n0 + self.drift_radius * math.sin(self.drift_speed * self._t)
        e = self.e0 + self.drift_radius * math.cos(self.drift_speed * self._t)
        return n, e


def default_convoy() -> list[GroundRobot]:
    """5 ground robots spread across a 1x1 m arena (the convoy)."""
    return [
        GroundRobot(0, 0.25, 0.20),
        GroundRobot(1, 0.50, 0.45),
        GroundRobot(2, 0.75, 0.30),
        GroundRobot(3, 0.40, 0.75),
        GroundRobot(4, 0.80, 0.80),
    ]
