"""
Software geofence for UWB anchor coverage.

UWB position is only trustworthy inside the region bounded by the anchors.
Outside that zone readings degrade and velocity-control loops chase bad data,
which causes erratic flight.

This module:
  - validates waypoints/targets before flight (safe zone = bounds minus margin)
  - monitors live UWB each nav tick (hard anchor bounds)
  - stops movement and raises GeofenceViolation if the drone leaves the anchors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GeofenceViolation(Exception):
    """Raised when UWB reports a position outside the anchor zone."""

    def __init__(self, n: float, e: float, message: str) -> None:
        self.n = n
        self.e = e
        super().__init__(message)


@dataclass(frozen=True)
class ArenaBounds:
    """UWB anchor coverage in arena North/East (meters)."""

    n_min: float
    n_max: float
    e_min: float
    e_max: float
    safety_margin_m: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ArenaBounds | None:
        arena = cfg.get("arena", {})
        if not arena.get("geofence_enabled", True):
            return None
        raw = arena.get("uwb_bounds")
        if not raw:
            return None
        return cls(
            n_min=float(raw["n_min"]),
            n_max=float(raw["n_max"]),
            e_min=float(raw["e_min"]),
            e_max=float(raw["e_max"]),
            safety_margin_m=float(arena.get("safety_margin_m", 0.5)),
        )

    def _safe_limits(self) -> tuple[float, float, float, float]:
        m = self.safety_margin_m
        return (
            self.n_min + m,
            self.n_max - m,
            self.e_min + m,
            self.e_max - m,
        )

    def in_anchor_zone(self, n: float, e: float) -> bool:
        """Hard UWB coverage — full anchor bounds."""
        return (
            self.n_min <= n <= self.n_max and self.e_min <= e <= self.e_max
        )

    def in_safe_zone(self, n: float, e: float) -> bool:
        """Inset safe region for planning waypoints (margin from edges)."""
        sn, sx, se, ex = self._safe_limits()
        if sn > sx or se > ex:
            return False
        return sn <= n <= sx and se <= e <= ex

    def clamp_to_safe_zone(self, n: float, e: float) -> tuple[float, float]:
        sn, sx, se, ex = self._safe_limits()
        return (
            max(sn, min(sx, n)),
            max(se, min(ex, e)),
        )

    def check_position(self, n: float, e: float) -> None:
        """Raise if live UWB is outside anchor coverage."""
        if not self.in_anchor_zone(n, e):
            raise GeofenceViolation(
                n,
                e,
                f"UWB position N={n:.2f} E={e:.2f} is outside anchor zone "
                f"(N=[{self.n_min:.2f},{self.n_max:.2f}], "
                f"E=[{self.e_min:.2f},{self.e_max:.2f}])",
            )

    def validate_point(self, n: float, e: float, label: str = "point") -> None:
        """Raise if a planned target is outside the safe zone."""
        if not self.in_safe_zone(n, e):
            sn, sx, se, ex = self._safe_limits()
            raise GeofenceViolation(
                n,
                e,
                f"{label} N={n:.2f} E={e:.2f} is outside safe zone "
                f"(margin={self.safety_margin_m:.2f} m, "
                f"safe N=[{sn:.2f},{sx:.2f}], E=[{se:.2f},{ex:.2f}])",
            )

    def validate_waypoints(self, waypoints: list[dict]) -> None:
        for i, wp in enumerate(waypoints):
            self.validate_point(float(wp["n"]), float(wp["e"]), f"waypoint {i}")

    def validate_ne_points(
        self, points: list[tuple[float, float, str]]
    ) -> None:
        for n, e, label in points:
            self.validate_point(n, e, label)

    def validate_region(self, n_min: float, n_max: float, e_min: float, e_max: float, label: str = "region") -> None:
        """Ensure a search/arena rectangle fits inside the safe zone."""
        for n, e in (
            (n_min, e_min),
            (n_min, e_max),
            (n_max, e_min),
            (n_max, e_max),
        ):
            self.validate_point(n, e, f"{label} corner")
