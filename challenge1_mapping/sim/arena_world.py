"""
Simulated arena layout for dry-run.

Landing pads are placed at fixed world (N, E) coordinates. The fake camera
renders whichever pads fall inside its field of view at each waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimLandingPad:
    marker_id: int
    n: float
    e: float
    valid: bool


@dataclass(frozen=True)
class SimObstacle:
    """Axis-aligned box on the ground (N/E corners), height in meters."""
    n0: float
    e0: float
    n1: float
    e1: float
    height_m: float = 0.5


# Pads near each survey corner (N, E) so the down-camera sees them in FOV
DEFAULT_PADS = [
    SimLandingPad(0, 0.08, 0.08, True),   # waypoint (0, 0)
    SimLandingPad(1, 0.92, 0.08, True),   # waypoint (1, 0)
    SimLandingPad(2, 0.92, 0.92, True),   # waypoint (1, 1)
    SimLandingPad(3, 0.08, 0.92, True),   # waypoint (0, 1)
    SimLandingPad(10, 0.50, 0.08, False),
    SimLandingPad(11, 0.50, 0.50, False),
]

DEFAULT_OBSTACLES = [
    SimObstacle(0.45, 0.1, 0.55, 0.25, height_m=0.4),
    SimObstacle(0.15, 0.6, 0.35, 0.85, height_m=0.6),
]
