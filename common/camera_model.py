"""
Camera geometry for a down-facing RealSense (D430 / D450 modules).

At the 3.5 m minimum flight height the camera sees a large ground patch, and
small markers / ground robots occupy few pixels. These helpers turn the height
+ field-of-view into a ground footprint so search-leg spacing and the minimum
detectable object size are grounded in real numbers, not guesses.

FOV figures are the published Intel defaults; resolution is configurable on the
drone, which changes pixel density but NOT the field of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraFOV:
    name: str
    # horizontal / vertical field of view in degrees
    color_hfov: float
    color_vfov: float
    depth_hfov: float
    depth_vfov: float


# Intel published defaults. Confirm exact values per the unit's datasheet.
D430 = CameraFOV("D430", color_hfov=69.0, color_vfov=42.0, depth_hfov=87.0, depth_vfov=58.0)
D450 = CameraFOV("D450", color_hfov=90.0, color_vfov=65.0, depth_hfov=87.0, depth_vfov=58.0)


def ground_footprint(height_m: float, hfov_deg: float, vfov_deg: float) -> tuple[float, float]:
    """Ground patch (width_m, length_m) a down-facing camera sees from height_m."""
    width = 2.0 * height_m * math.tan(math.radians(hfov_deg) / 2.0)
    length = 2.0 * height_m * math.tan(math.radians(vfov_deg) / 2.0)
    return width, length


def recommended_leg_spacing(
    height_m: float, fov: CameraFOV, use_color: bool = True, overlap: float = 0.2
) -> float:
    """
    Spacing between lawnmower legs that guarantees no gaps.

    Uses the *narrow* footprint dimension (so coverage holds regardless of how
    the image axes map to N/E), shrunk by `overlap` for safety.
    """
    if use_color:
        w, ln = ground_footprint(height_m, fov.color_hfov, fov.color_vfov)
    else:
        w, ln = ground_footprint(height_m, fov.depth_hfov, fov.depth_vfov)
    return min(w, ln) * (1.0 - overlap)


def meters_per_pixel(height_m: float, hfov_deg: float, image_width_px: int) -> float:
    """Ground resolution at the given height — how many meters one pixel covers."""
    width_m = 2.0 * height_m * math.tan(math.radians(hfov_deg) / 2.0)
    return width_m / image_width_px
