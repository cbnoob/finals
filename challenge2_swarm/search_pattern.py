"""
Search coverage patterns for the HULA swarm (Challenge 2).

Goal: systematically cover an arena region so the drone's downward camera passes
over every ground robot. A boustrophedon ("lawnmower") path gives full coverage
with minimal turns. Pure functions — unit-testable without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    n_min: float
    n_max: float
    e_min: float
    e_max: float


def split_region(area: Region, num_drones: int, index: int) -> Region:
    """Divide the search area into vertical (East) strips, one per drone."""
    if num_drones <= 1:
        return area
    width = (area.e_max - area.e_min) / num_drones
    e0 = area.e_min + index * width
    e1 = e0 + width
    return Region(area.n_min, area.n_max, e0, e1)


def lawnmower_waypoints(region: Region, spacing: float) -> list[tuple[float, float]]:
    """
    Boustrophedon coverage of a region.

    Sweeps along North, stepping East by `spacing` each leg, alternating
    direction so the path is continuous. Returns [(n, e), ...].
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    waypoints: list[tuple[float, float]] = []
    e = region.e_min
    going_up = True
    # Guard against pathological huge loops
    max_legs = 1000
    legs = 0
    while e <= region.e_max + 1e-9 and legs < max_legs:
        if going_up:
            waypoints.append((region.n_min, e))
            waypoints.append((region.n_max, e))
        else:
            waypoints.append((region.n_max, e))
            waypoints.append((region.n_min, e))
        going_up = not going_up
        e += spacing
        legs += 1
    return waypoints


def densify_path(
    waypoints: list[tuple[float, float]], step: float = 0.05
) -> list[tuple[float, float]]:
    """Sample points along the straight legs between consecutive waypoints."""
    import math

    if len(waypoints) < 2:
        return list(waypoints)
    pts: list[tuple[float, float]] = []
    for (n0, e0), (n1, e1) in zip(waypoints, waypoints[1:]):
        seg = math.hypot(n1 - n0, e1 - e0)
        count = max(1, int(seg / step))
        for k in range(count + 1):
            t = k / count
            pts.append((n0 + (n1 - n0) * t, e0 + (e1 - e0) * t))
    return pts


def coverage_fraction(
    region: Region,
    visited: list[tuple[float, float]],
    sensor_radius: float,
    grid_step: float = 0.1,
) -> float:
    """
    Coverage estimate: fraction of region grid cells within sensor_radius of the
    flown path (waypoints are densified into a continuous path first).
    """
    if not visited:
        return 0.0
    import math

    path = densify_path(visited, step=min(grid_step, sensor_radius))
    covered = 0
    total = 0
    n = region.n_min
    while n <= region.n_max + 1e-9:
        e = region.e_min
        while e <= region.e_max + 1e-9:
            total += 1
            for vn, ve in path:
                if math.hypot(vn - n, ve - e) <= sensor_radius:
                    covered += 1
                    break
            e += grid_step
        n += grid_step
    return covered / total if total else 0.0
