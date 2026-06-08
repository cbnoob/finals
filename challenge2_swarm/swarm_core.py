"""Shared swarm state machine — real hardware and dry-run.

Primary objective (Challenge 2): cover the arena, find the ground-robot convoy,
and snapshot each robot. Flow per drone:

  TAKEOFF -> GO_TO_REGION -> SEARCH (lawnmower coverage + sensing)
          -> SNAPSHOT (on new target) -> back to SEARCH -> RETURN -> DONE
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from challenge2_swarm.search_pattern import Region, lawnmower_waypoints, split_region
from challenge2_swarm.uwb_nav import apply_nav_tick, uwb_nav_tick
from common.uwb_c2 import UWBSource
from common.velocity_nav import NavGains

try:
    from pyhulax.core import Direction
except ImportError:
    Direction = None  # type: ignore


class DroneState(IntEnum):
    IDLE = 0
    TAKEOFF = 1
    GO_TO_REGION = 2
    SEARCH = 3
    SNAPSHOT = 4
    RETURN = 5
    DONE = 6


@dataclass
class DroneContext:
    ip: str
    api: object
    tag_id: int
    stream: object | None = None
    state: DroneState = DroneState.IDLE
    state_entered: float = field(default_factory=time.time)
    # landing zone (ambush position from Challenge 1)
    target_n: float = 0.0
    target_e: float = 0.0
    # search coverage
    search_waypoints: list[tuple[float, float]] = field(default_factory=list)
    search_idx: int = 0
    # results
    snapshots_taken: int = 0
    found_target_ids: set = field(default_factory=set)
    last_snapshot_t: float = 0.0
    _pending: list = field(default_factory=list)


def load_landing_zones(report_path: Path | None = None) -> list[dict]:
    report = report_path or (
        Path(__file__).resolve().parents[1] / "output" / "challenge1" / "landing_pad_report.json"
    )
    if not report.exists():
        return []
    data = json.loads(report.read_text(encoding="utf-8"))
    zones = data.get("valid_landing_zones", [])
    if not zones:
        zones = [
            {"n": o.get("world_n", 0), "e": o.get("world_e", 0), "marker_id": o.get("marker_id")}
            for o in data.get("observations", [])
            if o.get("valid_landing")
        ]
    return zones[:3]


def load_arena_bounds(report_path: Path | None = None) -> dict | None:
    """Read arena_bounds from the Challenge 1 map so the swarm searches the
    exact area the mapping drone surveyed."""
    report = report_path or (
        Path(__file__).resolve().parents[1] / "output" / "challenge1" / "landing_pad_report.json"
    )
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    return data.get("arena_bounds")


def _elapsed(ctx: DroneContext) -> float:
    return time.time() - ctx.state_entered


def _set_state(ctx: DroneContext, state: DroneState) -> None:
    ctx.state = state
    ctx.state_entered = time.time()


def _search_area(swarm_cfg: dict, use_map_bounds: bool = True) -> Region:
    # Prefer the bounds the mapping drone actually surveyed (from its map);
    # fall back to the config search_area.
    a = (load_arena_bounds() if use_map_bounds else None) or swarm_cfg.get("search_area", {})
    return Region(
        n_min=float(a.get("n_min", 0.0)),
        n_max=float(a.get("n_max", 1.0)),
        e_min=float(a.get("e_min", 0.0)),
        e_max=float(a.get("e_max", 1.0)),
    )


def assign_search_regions(
    contexts: dict[str, DroneContext], swarm_cfg: dict
) -> None:
    """Split the search area into per-drone strips and build lawnmower paths."""
    area = _search_area(swarm_cfg, use_map_bounds=bool(swarm_cfg.get("use_map_bounds", True)))
    spacing = float(swarm_cfg.get("search_spacing_m", 0.3))
    ips = list(contexts.keys())
    num = len(ips)
    for i, ip in enumerate(ips):
        region = split_region(area, num, i)
        contexts[ip].search_waypoints = lawnmower_waypoints(region, spacing)


def run_swarm_loop(
    contexts: dict[str, DroneContext],
    uwb: UWBSource,
    cfg: dict,
    sensor,
    *,
    simulated: bool = False,
    on_tick=None,
) -> None:
    swarm_cfg = cfg["swarm"]
    nav_cfg = cfg["navigation"]
    gains = NavGains(**{k: nav_cfg[k] for k in NavGains.__dataclass_fields__})
    arrive_th = float(swarm_cfg.get("waypoint_threshold_m", 0.12))
    gains.n_threshold = arrive_th
    gains.e_threshold = arrive_th

    landing_zones = load_landing_zones()
    assign_search_regions(contexts, swarm_cfg)

    ips = list(contexts.keys())
    for i, ip in enumerate(ips):
        ctx = contexts[ip]
        _set_state(ctx, DroneState.TAKEOFF)
        if i < len(landing_zones):
            ctx.target_n = float(landing_zones[i].get("n", 0))
            ctx.target_e = float(landing_zones[i].get("e", 0))

    takeoff_wait = float(swarm_cfg.get("takeoff_wait_s", 5))
    move_speed = float(swarm_cfg.get("move_speed", 0.5))
    nav_timeout = float(swarm_cfg.get("uwb_nav_timeout_s", 120))
    min_move_speed = float(swarm_cfg.get("min_move_speed", 0.05))
    snapshot_cooldown = float(swarm_cfg.get("snapshot_cooldown_s", 1.0))
    dedup_dist = float(swarm_cfg.get("target_dedup_m", 0.25))
    return_home = bool(swarm_cfg.get("return_after_search", True))

    snapshot_dir = Path(swarm_cfg.get("snapshot_dir", "output/snapshots"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    mode = "SIM" if simulated else "LIVE"
    print(f"Swarm SEARCH mission ({mode}) — Ctrl+C to stop")

    def _nav_to(ctx, tn, te):
        tick = uwb_nav_tick(uwb, ctx.tag_id, tn, te, gains, move_speed)
        apply_nav_tick(ctx.api, tick, min_speed=min_move_speed)
        return tick

    def _new_targets(ctx, sensed):
        """Filter out robots already snapshotted (by id or proximity + cooldown)."""
        out = []
        for t in sensed:
            if t.target_id is not None:
                if t.target_id in ctx.found_target_ids:
                    continue
            else:
                if (time.time() - ctx.last_snapshot_t) < snapshot_cooldown:
                    continue
            out.append(t)
        return out

    try:
        while any(c.state != DroneState.DONE for c in contexts.values()):
            for ip, ctx in contexts.items():
                api = ctx.api

                if ctx.state == DroneState.TAKEOFF:
                    api.takeoff()
                    if _elapsed(ctx) >= (0.5 if simulated else takeoff_wait):
                        _set_state(ctx, DroneState.GO_TO_REGION)

                elif ctx.state == DroneState.GO_TO_REGION:
                    if not ctx.search_waypoints:
                        _set_state(ctx, DroneState.SEARCH)
                        continue
                    first_n, first_e = ctx.search_waypoints[0]
                    tick = _nav_to(ctx, first_n, first_e)
                    if tick.at_goal or _elapsed(ctx) > nav_timeout:
                        ctx.search_idx = 0
                        _set_state(ctx, DroneState.SEARCH)

                elif ctx.state == DroneState.SEARCH:
                    # 1) sense every tick (primary goal)
                    sensed = sensor.sense(ctx)
                    new = _new_targets(ctx, sensed)
                    if new:
                        ctx._pending = new
                        _set_state(ctx, DroneState.SNAPSHOT)
                        continue

                    # 2) advance along lawnmower coverage path
                    if ctx.search_idx >= len(ctx.search_waypoints):
                        _set_state(ctx, DroneState.RETURN if return_home else DroneState.DONE)
                        continue
                    wn, we = ctx.search_waypoints[ctx.search_idx]
                    tick = _nav_to(ctx, wn, we)
                    if tick.at_goal:
                        ctx.search_idx += 1

                elif ctx.state == DroneState.SNAPSHOT:
                    api.hover()
                    out = snapshot_dir / f"drone{ctx.tag_id}_snap{ctx.snapshots_taken:02d}.jpg"
                    count = sensor.save_snapshot(ctx, ctx._pending, out)
                    ctx.snapshots_taken += 1
                    ctx.last_snapshot_t = time.time()
                    for t in ctx._pending:
                        if t.target_id is not None:
                            ctx.found_target_ids.add(t.target_id)
                    ids = [t.target_id for t in ctx._pending]
                    print(f"{ip}: SNAPSHOT {out.name} targets={ids} ({count} boxes)")
                    ctx._pending = []
                    _set_state(ctx, DroneState.SEARCH)

                elif ctx.state == DroneState.RETURN:
                    tick = _nav_to(ctx, ctx.target_n, ctx.target_e)
                    if tick.at_goal or _elapsed(ctx) > nav_timeout:
                        api.hover()
                        print(f"{ip}: returned to zone, found {len(ctx.found_target_ids)} robots")
                        _set_state(ctx, DroneState.DONE)

            if on_tick is not None:
                on_tick(contexts)
            time.sleep(0.02 if simulated else 0.1)

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        for ctx in contexts.values():
            try:
                ctx.api.land()
            except Exception:
                pass
