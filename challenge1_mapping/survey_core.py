"""Shared survey logic used by real mission and laptop dry-run."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from challenge1_mapping.arena_map import ArenaMap, marker_world_position
from detection.aruco_depth import ArucoDepthDetector
from detection.occupancy_grid import GridConfig, build_occupancy_grid
from detection.realsense_capture import FramePair


def process_waypoint(
    waypoint_index: int,
    drone_n: float,
    drone_e: float,
    frames: FramePair,
    aruco: ArucoDepthDetector,
    arena: ArenaMap,
    observations: list[dict],
    output_dir: Path,
    grid_cfg: GridConfig | None = None,
) -> None:
    """Detect pads, build occupancy grid, update arena map and observations."""
    grid_cfg = grid_cfg or GridConfig()
    arena.add_path_point(drone_n, drone_e)

    depth_m = frames.depth_mm.astype("float32") / 1000.0
    grid = build_occupancy_grid(
        depth_m,
        frames.intrinsics.fx,
        frames.intrinsics.fy,
        frames.intrinsics.cx,
        frames.intrinsics.cy,
        grid_cfg,
    )
    cv2.imwrite(str(output_dir / f"occupancy_wp{waypoint_index:02d}.png"), grid)

    markers = aruco.detect(frames.color_bgr, frames.depth_mm)
    for obs in markers:
        world_n, world_e = marker_world_position(
            drone_n, drone_e, obs.x_m, obs.y_m, arena.cfg
        )
        arena.add_landing_pad(obs.marker_id, obs.valid_landing, world_n, world_e)
        observations.append(
            {
                "waypoint_index": waypoint_index,
                "drone_n": drone_n,
                "drone_e": drone_e,
                "world_n": world_n,
                "world_e": world_e,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **asdict(obs),
            }
        )
        status = "VALID" if obs.valid_landing else "INVALID"
        print(
            f"  ArUco id={obs.marker_id} {status} "
            f"world N={world_n:.2f} E={world_e:.2f} (z={obs.z_m:.2f}m)"
        )


def save_mission_report(
    arena: ArenaMap,
    observations: list[dict],
    output_dir: Path,
    *,
    simulated: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "arena_map.png"), arena.render_bgr())

    valid_pads = [
        {"marker_id": p.marker_id, "n": p.n, "e": p.e}
        for p in arena.pads
        if p.valid
    ]
    report = {
        "challenge": 1,
        "simulated": simulated,
        "observations": observations,
        "valid_landing_ids": sorted({p.marker_id for p in arena.pads if p.valid}),
        "valid_landing_zones": valid_pads,
    }
    out_path = output_dir / "landing_pad_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {out_path}")
    print(f"Arena map saved: {output_dir / 'arena_map.png'}")
    print(f"Valid landing zones: {len(valid_pads)}")
