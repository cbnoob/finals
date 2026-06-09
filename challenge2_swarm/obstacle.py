"""
Obstacle sensing + avoidance for the HULA swarm.

BRIEF RULE: "Strictly no flying over obstacles" — violation invalidates the score.
The HULA flies at a fixed low height (~1.1 m), so obstacles must be avoided
*horizontally* (go around), never by climbing over them.

Design: the nav layer asks an ObstacleSensor "how far is the nearest obstacle if I
move in direction (dn, de)?". Two implementations:

  MapObstacleSensor  -> known obstacle boxes in arena N/E (dry-run, or from the
                        Challenge 1 map). Pure + unit-testable.
  HulaObstacleSensor -> reads the HULA's onboard obstacle sensing (lidar) via
                        pyhulax. The exact SDK call is confirmed on the unit; until
                        wired it FAILS SAFE (reports blocked) so we never pretend the
                        path is clear and break the no-fly-over rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

try:
    from pyhulax.core import Direction
except ImportError:

    class Direction:  # type: ignore[no-redef]
        FORWARD = "FORWARD"
        BACKWARD = "BACKWARD"
        LEFT = "LEFT"
        RIGHT = "RIGHT"


# Direction -> unit (dNorth, dEast)
DIR_DELTA = {
    Direction.FORWARD: (1.0, 0.0),
    Direction.BACKWARD: (-1.0, 0.0),
    Direction.RIGHT: (0.0, 1.0),
    Direction.LEFT: (0.0, -1.0),
}


class ObstacleSensor(Protocol):
    def distance_ahead(self, n: float, e: float, dn: float, de: float) -> float:
        """Meters to the nearest obstacle from (n,e) along unit dir (dn,de).
        Return math.inf if clear."""
        ...


@dataclass(frozen=True)
class ObstacleBox:
    """Axis-aligned obstacle footprint in arena N/E (meters)."""

    n0: float
    e0: float
    n1: float
    e1: float

    def inflated(self, margin: float) -> "ObstacleBox":
        return ObstacleBox(
            self.n0 - margin, self.e0 - margin, self.n1 + margin, self.e1 + margin
        )


def _ray_box_distance(
    n: float, e: float, dn: float, de: float, box: ObstacleBox
) -> float:
    """Distance from (n,e) to an axis-aligned box along an axis-aligned ray.

    Movement is quantized to N/E, so exactly one of dn/de is non-zero.
    """
    if dn > 0:  # heading +North
        if box.e0 <= e <= box.e1 and box.n1 >= n:
            return max(0.0, box.n0 - n)
    elif dn < 0:  # heading -North
        if box.e0 <= e <= box.e1 and box.n0 <= n:
            return max(0.0, n - box.n1)
    elif de > 0:  # heading +East
        if box.n0 <= n <= box.n1 and box.e1 >= e:
            return max(0.0, box.e0 - e)
    elif de < 0:  # heading -East
        if box.n0 <= n <= box.n1 and box.e0 <= e:
            return max(0.0, e - box.e1)
    return math.inf


class MapObstacleSensor:
    """Obstacle sensing from known boxes (sim, or Challenge 1 map)."""

    def __init__(self, boxes: list[ObstacleBox], clearance_m: float = 0.3) -> None:
        self.boxes = [b.inflated(clearance_m) for b in boxes]

    def distance_ahead(self, n: float, e: float, dn: float, de: float) -> float:
        best = math.inf
        for box in self.boxes:
            best = min(best, _ray_box_distance(n, e, dn, de, box))
        return best


class HulaObstacleSensor:
    """HULA onboard obstacle sensing (lidar) via pyhulax.

    The pyhulax SDK exposes "obstacle sensing" but the exact accessor varies by
    build. Set `reader` to a callable returning a dict {Direction: distance_m}
    once you confirm the call on the test drone (see _default_reader).

    FAIL SAFE: if no reader is wired, distance_ahead returns 0.0 (blocked) so the
    drone refuses to advance into unknown space rather than risk flying over an
    obstacle.
    """

    def __init__(self, api, reader=None) -> None:
        self.api = api
        self._wired = reader is not None
        self.reader = reader or self._default_reader
        self._warned = False

    def _default_reader(self):
        # TODO: replace with the confirmed pyhulax obstacle-sensing call, e.g.
        #   return self.api.get_obstacle_distances()  # {Direction: meters}
        # Until then we have no data -> treat every direction as blocked.
        return None

    def is_wired(self) -> bool:
        return self._wired

    def distance_ahead(self, n: float, e: float, dn: float, de: float) -> float:
        try:
            data = self.reader()
        except Exception as exc:  # never let a sensor read crash the loop
            if not self._warned:
                print(f"Obstacle sensor read failed ({exc}) — treating as blocked")
                self._warned = True
            return 0.0
        if not data:
            if not self._warned:
                print("Obstacle sensor not wired — failing safe (blocked)")
                self._warned = True
            return 0.0
        # Map (dn,de) back to a Direction key
        for direction, (ddn, dde) in DIR_DELTA.items():
            if (ddn, dde) == (dn, de):
                return float(data.get(direction, math.inf))
        return math.inf
