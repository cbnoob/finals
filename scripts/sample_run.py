"""
Fallback movement-only sample run for the mapping drone.

Use this if the full Challenge 1 mapping run is not behaving onsite and you
want a simpler organiser-style movement script for minimum movement credit.

Run from the project root on the drone:
    python3 scripts/sample_run.py

Safety:
    Press "e" in the terminal to land immediately.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_loader import load_config
from common.emergency import emergency_land_mavsdk, fly_with_emergency_land
from common.geofence import ArenaBounds
from common.uwb_listener import get_uwb_position, shutdown_uwb, start_uwb_thread, wait_for_uwb
from challenge1_mapping.survey_core import order_waypoints_from_start


DEFAULT_SPEED_MPS = 0.3
DEFAULT_HEIGHT_M = 2.0
DEFAULT_POSITION_STEP_M = 0.4
DEFAULT_WAYPOINTS = [
    (0.7, 0.7),
    (2.5, 0.7),
    (4.5, 0.7),
    (6.5, 0.7),
    (8.5, 0.7),
    (8.5, 2.5),
    (6.5, 2.5),
    (4.5, 2.5),
    (2.5, 2.5),
    (0.7, 2.5),
    (0.7, 4.3),
    (2.5, 4.3),
    (4.5, 4.3),
    (6.5, 4.3),
    (8.5, 4.3),
]


def _parse_waypoint(value: str) -> tuple[float, float]:
    try:
        n_text, e_text = value.split(",", 1)
        return float(n_text), float(e_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("waypoint must be N,E, for example 2.0,1.0") from exc


def _velocity_towards(
    target_n: float,
    target_e: float,
    current_n: float,
    current_e: float,
    max_speed: float,
) -> tuple[float, float, float]:
    err_n = target_n - current_n
    err_e = target_e - current_e
    distance = math.hypot(err_n, err_e)
    if distance < 1e-6:
        return 0.0, 0.0, 0.0
    speed = min(max_speed, max(0.05, 0.15 * distance))
    return speed * err_n / distance, speed * err_e / distance, distance


async def _wait_connected(drone) -> None:
    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected.")
            return


async def _wait_local_position(drone) -> None:
    print("Waiting for local position estimate...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("Local position estimate OK.")
            return


async def _wait_height(drone, height_m: float, timeout_s: float = 15.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    async for position in drone.telemetry.position():
        height = position.relative_altitude_m
        print(f"Height: {height:.2f}m / {height_m:.2f}m")
        if abs(height - height_m) <= 0.15:
            return
        if asyncio.get_running_loop().time() > deadline:
            print("Height wait timed out; continuing with 2m setpoint held.")
            return
        await asyncio.sleep(0.1)


async def _setpoint(drone, n: float, e: float, d: float, vn: float, ve: float, vd: float) -> None:
    from mavsdk.offboard import PositionNedYaw, VelocityNedYaw

    await drone.offboard.set_position_velocity_ned(
        PositionNedYaw(n, e, d, 0.0),
        VelocityNedYaw(vn, ve, vd, 0.0),
    )


async def _fly_to_waypoint(
    drone,
    target_n: float,
    target_e: float,
    height_m: float,
    speed_mps: float,
    start_n: float,
    start_e: float,
    timeout_s: float,
    position_step_m: float,
) -> None:
    target_d = -abs(height_m)
    deadline = asyncio.get_running_loop().time() + timeout_s

    print(f"Waypoint target N={target_n:.2f} E={target_e:.2f} D={target_d:.2f}")
    while True:
        current_n, current_e, ok = get_uwb_position()
        if not ok:
            print("Waiting for UWB during waypoint...")
            await asyncio.sleep(0.2)
            continue

        vn, ve, distance_to_target = _velocity_towards(
            target_n,
            target_e,
            current_n,
            current_e,
            speed_mps,
        )
        distance_from_start = math.hypot(current_n - start_n, current_e - start_e)
        distance_from_origin = math.hypot(current_n, current_e)
        print(
            f"UWB N={current_n:.2f} E={current_e:.2f} "
            f"target_dist={distance_to_target:.2f}m "
            f"from_start={distance_from_start:.2f}m "
            f"from_origin={distance_from_origin:.2f}m"
        )

        if distance_to_target <= 0.20:
            await _setpoint(drone, target_n, target_e, target_d, 0.0, 0.0, 0.0)
            print("Waypoint reached.")
            return

        if distance_to_target > 1e-6:
            step = min(position_step_m, distance_to_target)
            command_n = current_n + step * (target_n - current_n) / distance_to_target
            command_e = current_e + step * (target_e - current_e) / distance_to_target
        else:
            command_n, command_e = target_n, target_e
        await _setpoint(drone, command_n, command_e, target_d, vn, ve, 0.0)

        if asyncio.get_running_loop().time() > deadline:
            print("Waypoint timeout; holding current position.")
            await _setpoint(drone, current_n, current_e, target_d, 0.0, 0.0, 0.0)
            return

        await asyncio.sleep(0.2)


async def _sample_flight(drone, args, waypoints: list[tuple[float, float]]) -> None:
    from mavsdk.offboard import OffboardError

    await _wait_connected(drone)
    await _wait_local_position(drone)

    start_n, start_e = await wait_for_uwb(timeout_s=30.0)
    print(f"Initial UWB N={start_n:.2f} E={start_e:.2f}")
    if not args.keep_order and len(waypoints) > 1:
        ordered = order_waypoints_from_start(
            [{"n": n, "e": e} for n, e in waypoints],
            start_n,
            start_e,
        )
        waypoints = [(float(wp["n"]), float(wp["e"])) for wp in ordered]
        print(f"Random-start ordering: first sample waypoint N={waypoints[0][0]:.2f} E={waypoints[0][1]:.2f}")

    print("Arming...")
    await drone.action.arm()

    # Organiser-style direct UWB PositionNedYaw: N/E are not offset from home.
    await _setpoint(drone, start_n, start_e, 0.0, 0.0, 0.0, 0.0)
    try:
        await drone.offboard.start()
    except OffboardError as exc:
        print(f"Offboard start failed: {exc}")
        await emergency_land_mavsdk(drone)
        return

    print(f"Taking off to {args.height:.2f}m.")
    await _setpoint(drone, start_n, start_e, -abs(args.height), 0.0, 0.0, -args.speed)
    await _wait_height(drone, args.height, timeout_s=15.0)
    await _setpoint(drone, start_n, start_e, -abs(args.height), 0.0, 0.0, 0.0)
    await asyncio.sleep(args.settle)

    for idx, (target_n, target_e) in enumerate(waypoints, start=1):
        print(f"\nSample waypoint {idx}/{len(waypoints)}")
        await _fly_to_waypoint(
            drone,
            target_n,
            target_e,
            args.height,
            args.speed,
            start_n,
            start_e,
            args.timeout,
            args.position_step,
        )
        await asyncio.sleep(args.hover)

    print("Sample run complete; landing.")
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    await drone.action.land()


def _load_default_waypoints(config_path: str | None) -> list[tuple[float, float]]:
    cfg = load_config(config_path)
    arena = cfg.get("arena", {})
    mapping = cfg.get("mapping_drone", {})
    bounds = arena.get("uwb_bounds", {})
    margin = float(arena.get("safety_margin_m", 0.5))

    if not mapping.get("auto_survey", True):
        return [
            (float(wp["n"]), float(wp["e"]))
            for wp in mapping.get("survey_waypoints", [])
        ]

    n_min = float(bounds.get("n_min", 0.0)) + margin
    n_max = float(bounds.get("n_max", 10.0)) - margin
    e_min = float(bounds.get("e_min", 0.0)) + margin
    e_max = float(bounds.get("e_max", 5.0)) - margin
    if n_min >= n_max or e_min >= e_max:
        return DEFAULT_WAYPOINTS

    rows = [e_min, (e_min + e_max) / 2.0, e_max]
    cols = [n_min, n_min + 2.0, n_min + 4.0, n_min + 6.0, n_max]
    waypoints: list[tuple[float, float]] = []
    for row_idx, e in enumerate(rows):
        ns = cols if row_idx % 2 == 0 else list(reversed(cols))
        waypoints.extend((n, e) for n in ns)
    return waypoints


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fallback movement-only sample run.")
    parser.add_argument("--config", default=None, help="Path to challenge.yaml.")
    parser.add_argument("--serial", default=None, help="MAVSDK serial address.")
    parser.add_argument("--uwb-topic", default=None, help="ROS2 UWB topic name.")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_M, help="Flight height in metres.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED_MPS, help="Max XY speed in m/s.")
    parser.add_argument("--timeout", type=float, default=35.0, help="Seconds per waypoint before holding.")
    parser.add_argument("--hover", type=float, default=0.5, help="Seconds to pause at each waypoint.")
    parser.add_argument("--settle", type=float, default=1.0, help="Seconds to hold after takeoff.")
    parser.add_argument("--position-step", type=float, default=DEFAULT_POSITION_STEP_M, help="Rolling PositionNED target step in metres.")
    parser.add_argument("--keep-order", action="store_true", help="Do not reorder waypoints from current UWB start.")
    parser.add_argument(
        "--waypoint",
        action="append",
        type=_parse_waypoint,
        help="Override path with waypoint N,E. Can be used multiple times.",
    )
    return parser


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    mapping = cfg.get("mapping_drone", {})
    nav = cfg.get("navigation", {})

    serial = args.serial or mapping.get("serial_address", "serial:///dev/ttyS6:921600")
    topic = args.uwb_topic or mapping.get("uwb_topic", "uwb_tag")
    args.height = float(args.height or mapping.get("takeoff_height_m", DEFAULT_HEIGHT_M))
    args.speed = min(float(args.speed), float(nav.get("max_vel_xy", DEFAULT_SPEED_MPS)), DEFAULT_SPEED_MPS)
    args.position_step = min(
        float(args.position_step),
        float(nav.get("max_position_step_m", DEFAULT_POSITION_STEP_M)),
    )
    waypoints = args.waypoint or _load_default_waypoints(args.config)

    arena = cfg.get("arena", {})
    if arena.get("geofence_enabled", True):
        bounds = ArenaBounds.from_config(cfg)
        if bounds is not None:
            waypoints = [(n, e) for n, e in waypoints if bounds.in_safe_zone(n, e)]

    print("=== SAMPLE RUN FALLBACK ===")
    print(f"Serial: {serial}")
    print(f"UWB topic: {topic}")
    print(f"Height: {args.height:.2f}m, max speed: {args.speed:.2f}m/s")
    print(f"Rolling position step: {args.position_step:.2f}m")
    print(f"Waypoints: {len(waypoints)}")
    print("Emergency key: press 'e' to land.")

    from mavsdk import System

    start_uwb_thread(topic)
    drone = System()
    await drone.connect(system_address=serial)

    try:
        await fly_with_emergency_land(
            _sample_flight(drone, args, waypoints),
            drone,
            navigator=None,
            emergency_key="e",
        )
    finally:
        shutdown_uwb()


if __name__ == "__main__":
    asyncio.run(main())
