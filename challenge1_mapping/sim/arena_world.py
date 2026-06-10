"""
Simulated arena layout for dry-run.

Landing pads are placed at fixed world (N, E) coordinates. The fake camera
renders whichever pads fall inside its field of view at each waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


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


# Current onsite practice layout. User-provided coordinates are x/y in meters
# inside the UWB anchor bounds, where x maps to East and y maps to North.
DEFAULT_PADS = [
    SimLandingPad(11, 4.40, 1.35, True),
    SimLandingPad(45, 7.85, 1.30, True),
    SimLandingPad(51, 4.40, 4.40, True),
    SimLandingPad(67, 8.70, 1.95, True),
    SimLandingPad(101, 7.85, 4.40, True),
    SimLandingPad(201, 2.30, 2.35, False),
    SimLandingPad(202, 6.10, 3.10, False),
    SimLandingPad(203, 9.10, 3.85, False),
]

DEFAULT_OBSTACLES = [
    SimObstacle(1.00, 0.85, 1.60, 1.45, height_m=0.45),
    SimObstacle(2.70, 3.15, 3.35, 3.85, height_m=0.65),
    SimObstacle(5.00, 1.90, 5.75, 2.55, height_m=0.50),
    SimObstacle(6.60, 0.65, 7.20, 1.20, height_m=0.75),
    SimObstacle(8.45, 3.00, 9.10, 3.70, height_m=0.40),
]


def random_obstacles(
    count: int,
    *,
    seed: int = 7,
    n_min: float = 0.5,
    n_max: float = 9.5,
    e_min: float = 0.5,
    e_max: float = 4.5,
    max_height_m: float = 1.1,
) -> list[SimObstacle]:
    """Generate deterministic random obstacle boxes inside the sim arena."""
    rng = random.Random(seed)
    obstacles: list[SimObstacle] = []
    for i in range(count):
        size_n = rng.uniform(0.35, 0.85)
        size_e = rng.uniform(0.35, 0.85)
        n0 = rng.uniform(n_min, n_max - size_n)
        e0 = rng.uniform(e_min, e_max - size_e)
        # Spread heights across the allowed range, then add small deterministic jitter.
        base = 0.25 + (max_height_m - 0.25) * ((i + 1) / max(1, count))
        height = min(max_height_m, max(0.20, base + rng.uniform(-0.08, 0.08)))
        obstacles.append(
            SimObstacle(
                round(n0, 2),
                round(e0, 2),
                round(n0 + size_n, 2),
                round(e0 + size_e, 2),
                height_m=round(height, 2),
            )
        )
    return obstacles
