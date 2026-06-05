"""
Top-down occupancy grid from RealSense depth — the Challenge 1 mapping deliverable.

Adapted from the organizer's generateTopDown.py. The camera faces down:
    RealSense frame: Z forward, X right, Y down
    Grid is the X-Z plane: forward (North) = up, right (East) = +x
The pure build_occupancy_grid() runs on any laptop (no camera needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GridConfig:
    min_depth_m: float = 0.2
    max_depth_m: float = 5.0
    resolution_m: float = 0.05  # meters per cell
    width_cells: int = 200      # 10 m wide
    height_cells: int = 200     # 10 m forward
    denoise: bool = True


def build_occupancy_grid(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: GridConfig | None = None,
) -> np.ndarray:
    """
    depth_m: HxW depth in METERS (already scaled by depth_scale).
    Returns: uint8 occupancy grid (height_cells x width_cells), 255 = occupied,
             128 = camera position marker, 0 = free/unknown.
    """
    cfg = config or GridConfig()
    h, w = depth_m.shape[:2]

    u_coords, v_coords = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )

    valid = (depth_m > cfg.min_depth_m) & (depth_m < cfg.max_depth_m)

    z = depth_m
    x = (u_coords - cx) * z / fx

    X = x[valid]
    Z = z[valid]

    occupancy = np.zeros((cfg.height_cells, cfg.width_cells), dtype=np.uint8)
    grid_center_x = cfg.width_cells // 2

    gx = (X / cfg.resolution_m).astype(np.int32) + grid_center_x
    gz = (Z / cfg.resolution_m).astype(np.int32)

    in_grid = (
        (gx >= 0) & (gx < cfg.width_cells) & (gz >= 0) & (gz < cfg.height_cells)
    )
    gx = gx[in_grid]
    gz = gz[in_grid]

    occupancy[cfg.height_cells - 1 - gz, gx] = 255

    if cfg.denoise:
        kernel = np.ones((3, 3), np.uint8)
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_OPEN, kernel)

    cv2.circle(occupancy, (grid_center_x, cfg.height_cells - 1), 5, 128, -1)
    return occupancy
