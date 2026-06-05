"""Top-down occupancy grid built from synthetic depth maps (no camera)."""

import numpy as np

from detection.occupancy_grid import GridConfig, build_occupancy_grid

# Typical RealSense-ish intrinsics for 640x480
FX = FY = 600.0
CX, CY = 320.0, 240.0


def test_grid_shape_and_camera_marker():
    depth = np.zeros((480, 640), dtype=np.float32)  # all invalid
    cfg = GridConfig()
    grid = build_occupancy_grid(depth, FX, FY, CX, CY, cfg)
    assert grid.shape == (cfg.height_cells, cfg.width_cells)
    # Camera marker (128) at bottom center
    assert grid[cfg.height_cells - 1, cfg.width_cells // 2] == 128


def test_empty_when_depth_out_of_range():
    depth = np.full((480, 640), 100.0, dtype=np.float32)  # beyond max_depth
    grid = build_occupancy_grid(depth, FX, FY, CX, CY)
    # Only the camera marker should be set (128), no occupied (255)
    assert np.count_nonzero(grid == 255) == 0


def test_obstacle_registers_in_grid():
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[200:280, 280:360] = 2.0  # a patch 2 m away near image center
    cfg = GridConfig(denoise=False)
    grid = build_occupancy_grid(depth, FX, FY, CX, CY, cfg)
    assert np.count_nonzero(grid == 255) > 0
    # 2 m forward at 0.05 m/cell -> row ~ 40 cells from bottom
    occupied_rows = np.where((grid == 255).any(axis=1))[0]
    expected_row = cfg.height_cells - 1 - int(2.0 / cfg.resolution_m)
    assert occupied_rows.min() <= expected_row + 3
    assert occupied_rows.max() >= expected_row - 3


def test_centered_obstacle_maps_near_center_column():
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[230:250, 310:330] = 1.5  # near principal point -> X ~ 0
    cfg = GridConfig(denoise=False)
    grid = build_occupancy_grid(depth, FX, FY, CX, CY, cfg)
    cols = np.where((grid == 255).any(axis=0))[0]
    assert abs(int(cols.mean()) - cfg.width_cells // 2) < 5
