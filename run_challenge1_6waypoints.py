"""Entry: fast 6-waypoint FOV mapper for Challenge 1.

This mode is a faster alternative to the full lawnmower survey. It uses the
D430 depth/IR footprint at 2 m to cover the 10 m x 5 m arena with a 3 x 2
waypoint pattern. It still runs the normal Challenge 1 camera/map pipeline.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import yaml

from challenge1_mapping.mission import run_mission
from common.config_loader import load_config


SIX_WAYPOINTS = [
    {"n": 2.0, "e": 1.4},
    {"n": 5.0, "e": 1.4},
    {"n": 8.3, "e": 1.7},
    {"n": 8.3, "e": 3.6},
    {"n": 5.0, "e": 3.6},
    {"n": 2.0, "e": 3.6},
]


def _write_temp_config() -> str:
    cfg = load_config()
    mapping = cfg["mapping_drone"]
    mapping["auto_survey"] = False
    mapping["start_nearest_waypoint"] = True
    mapping["survey_waypoints"] = SIX_WAYPOINTS
    mapping["hover_at_waypoint_s"] = max(float(mapping.get("hover_at_waypoint_s", 2.0)), 2.0)

    tmp_dir = tempfile.TemporaryDirectory(prefix="challenge1_6wp_")
    path = Path(tmp_dir.name) / "challenge_6waypoints.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Keep the directory alive for the full mission run.
    _write_temp_config._tmp_dir = tmp_dir  # type: ignore[attr-defined]
    return str(path)


def main() -> None:
    print("=== Challenge 1: 6-waypoint FOV mapper ===")
    print("Waypoints: (2.0,1.4), (5.0,1.4), (8.3,1.7), (8.3,3.6), (5.0,3.6), (2.0,3.6)")
    print("This is faster than lawnmower but less exhaustive.")
    asyncio.run(run_mission(_write_temp_config()))


if __name__ == "__main__":
    main()
