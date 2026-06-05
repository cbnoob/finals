"""
Visual occupancy-grid demo — no camera needed.

Builds a synthetic depth map with a few "obstacles" at known distances, runs the
same build_occupancy_grid() used on the mapping drone, and saves the result.

Run:
    python scripts/occupancy_demo.py
Output:
    output/occupancy_demo.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.occupancy_grid import GridConfig, build_occupancy_grid  # noqa: E402

FX = FY = 600.0
CX, CY = 320.0, 240.0


def _fill_with_spread(depth, rows, cols, base, spread=0.4):
    """Fill a patch with a depth gradient so it occupies several grid rows."""
    r0, r1 = rows
    c0, c1 = cols
    grad = np.linspace(base - spread / 2, base + spread / 2, r1 - r0, dtype=np.float32)
    depth[r0:r1, c0:c1] = grad[:, None]


def build_fake_depth() -> np.ndarray:
    depth = np.zeros((480, 640), dtype=np.float32)
    # Wall straight ahead around 3 m
    _fill_with_spread(depth, (150, 200), (100, 540), base=3.0, spread=0.5)
    # Obstacle to the left around 1.5 m
    _fill_with_spread(depth, (250, 320), (150, 230), base=1.5, spread=0.4)
    # Obstacle to the right around 2.2 m
    _fill_with_spread(depth, (230, 300), (420, 500), base=2.2, spread=0.4)
    return depth


def main() -> None:
    depth = build_fake_depth()
    cfg = GridConfig()
    grid = build_occupancy_grid(depth, FX, FY, CX, CY, cfg)

    occupied = int(np.count_nonzero(grid == 255))
    print(f"Grid {grid.shape}, occupied cells: {occupied}")

    vis = cv2.resize(grid, (600, 600), interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        vis, "Top-Down Occupancy (camera at bottom center)", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
    )
    out = ROOT / "output" / "occupancy_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
