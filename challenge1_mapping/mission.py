"""
Challenge 1 — Mapping drone (University teams).

Cooked from the organizer reference code into one mission:
  - kolomee.py        -> UWB + velocity offboard navigation (common/velocity_nav)
  - ArUco sample      -> landing-pad detection + depth (detection/aruco_depth)
  - getSyncDepthColor -> aligned color+depth (detection/realsense_capture)
  - generateTopDown   -> per-waypoint top-down occupancy (detection/occupancy_grid)

Laptop dry-run (no hardware): python scripts/dry_run_challenge1.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from challenge1_mapping.arena_map import ArenaMap, ArenaMapConfig
from challenge1_mapping.survey_core import (
    build_survey_waypoints,
    order_waypoints_from_start,
    process_waypoint,
    save_mission_report,
)
from common.config_loader import load_config
from common.emergency import emergency_land_mavsdk, fly_with_emergency_land
from common.geofence import ArenaBounds, GeofenceViolation
from common.uwb_listener import (
    get_uwb_position,
    shutdown_uwb,
    start_uwb_thread,
    wait_for_uwb,
)
from common.position_nav import PositionNedNavigator
from common.velocity_nav import NavGains, run_telemetry_tasks
from detection.aruco_depth import ArucoDepthDetector
from detection.occupancy_grid import GridConfig
from detection.realsense_capture import RealSenseCapture

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "challenge1"


async def run_mission(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    m = cfg["mapping_drone"]
    nav_cfg = cfg["navigation"]

    start_uwb_thread(m.get("uwb_topic", "uwb_tag"))
    await wait_for_uwb()

    from mavsdk import System

    drone = System()
    print("Connecting mapping drone...")
    await drone.connect(system_address=m["serial_address"])

    state = await run_telemetry_tasks(drone)
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("Local position estimate OK")
            break

    geofence = ArenaBounds.from_config(cfg)
    gains = NavGains(**{k: nav_cfg[k] for k in NavGains.__dataclass_fields__})
    home_n, home_e = await wait_for_uwb()
    navigator = PositionNedNavigator(
        drone,
        gains,
        home_n=home_n,
        home_e=home_e,
        get_yaw=lambda: state["yaw"],
        get_down=lambda: state["down_m"],
        origin_mode=m.get("position_ned_origin", "uwb"),
        geofence=geofence,
    )

    rs = RealSenseCapture(
        width=int(m.get("camera_width", 640)),
        height=int(m.get("camera_height", 480)),
        fps=int(m.get("camera_fps", 30)),
        image_source=m.get("camera_image_source", "auto"),
        disable_emitter_for_ir=bool(m.get("disable_ir_emitter", True)),
        auto_exposure=m.get("camera_auto_exposure"),
        exposure_us=m.get("camera_exposure"),
        gain=m.get("camera_gain"),
    )
    aruco = ArucoDepthDetector(
        fx=rs.intrinsics.fx,
        fy=rs.intrinsics.fy,
        cx=rs.intrinsics.cx,
        cy=rs.intrinsics.cy,
        dictionary_name=m.get("aruco_dictionary", "DICT_7X7_1000"),
        valid_ids=m.get("valid_marker_ids", []),
        invalid_ids=m.get("invalid_marker_ids", []),
        marker_size_m=m.get("marker_size_m"),
        preprocess=m.get("aruco_preprocess", "auto"),
    )

    bounds_raw = cfg.get("arena", {}).get("uwb_bounds", {})
    arena = ArenaMap(
        ArenaMapConfig(
            n_min=float(bounds_raw.get("n_min", -5.0)),
            n_max=float(bounds_raw.get("n_max", 5.0)),
            e_min=float(bounds_raw.get("e_min", -5.0)),
            e_max=float(bounds_raw.get("e_max", 5.0)),
        )
    )
    grid_cfg = GridConfig()
    observations: list[dict] = []
    takeoff_d = -float(m["takeoff_height_m"])
    mission_complete = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        navigator.takeoff_yaw = state["yaw"]
        print(
            f"Home UWB N={home_n:.2f} E={home_e:.2f}, "
            f"yaw={navigator.takeoff_yaw:.1f}, "
            f"position_ned_origin={navigator.origin_mode}"
        )
        print(f"Battery: {state['battery']:.0f}%")

        waypoints = build_survey_waypoints(cfg)
        if m.get("start_nearest_waypoint", True):
            waypoints = order_waypoints_from_start(waypoints, home_n, home_e)
            if waypoints:
                first = waypoints[0]
                print(
                    f"Random-start ordering: first waypoint is nearest to start "
                    f"N={float(first['n']):.2f} E={float(first['e']):.2f}"
                )
        print(f"Survey plan: {len(waypoints)} waypoints "
              f"({'auto full-area' if m.get('auto_survey') else 'manual'})")
        if geofence is not None:
            geofence.check_position(home_n, home_e)
            geofence.validate_waypoints(waypoints)
            print(
                f"Geofence OK — UWB anchors "
                f"N=[{geofence.n_min:.1f},{geofence.n_max:.1f}] "
                f"E=[{geofence.e_min:.1f},{geofence.e_max:.1f}] "
                f"(margin {geofence.safety_margin_m:.1f} m)"
            )

        await drone.action.set_takeoff_altitude(float(m["takeoff_height_m"]))
        await asyncio.sleep(1.0)

        choice = await asyncio.get_running_loop().run_in_executor(
            None, input, "Arm and start mission? (y/n): "
        )
        if choice.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

        hover_s = float(m.get("hover_at_waypoint_s", 2.0))
        continuous_capture_enabled = bool(m.get("continuous_capture_enabled", True))
        continuous_capture_interval_s = max(
            0.2,
            float(m.get("continuous_capture_interval_s", 1.0)),
        )
        capture_index = 0

        def _capture_sync(index: int, label: str) -> None:
            drone_n, drone_e, _ = get_uwb_position()
            frames = rs.get_frames()
            print(
                f"--- Capture {index:02d} ({label}) "
                f"at N={drone_n:.2f} E={drone_e:.2f} ---"
            )
            process_waypoint(
                index, drone_n, drone_e, frames, aruco, arena, observations,
                OUTPUT_DIR, grid_cfg,
            )
            save_mission_report(
                arena,
                observations,
                OUTPUT_DIR,
                simulated=False,
                mission_status="partial",
            )

        async def _capture_current(label: str, *, required: bool) -> None:
            nonlocal capture_index
            index = capture_index
            capture_index += 1
            try:
                await asyncio.to_thread(_capture_sync, index, label)
            except Exception as exc:
                print(f"Capture {index:02d} ({label}) failed: {exc}")
                if required:
                    raise

        async def _fly_to_with_captures(
            target_n: float,
            target_e: float,
            target_d: float,
            label: str,
        ) -> None:
            flight = asyncio.create_task(
                navigator.fly_to(target_n, target_e, target_d, ignore_height=False)
            )
            if not continuous_capture_enabled:
                await flight
                return

            loop = asyncio.get_running_loop()
            next_capture = loop.time()
            while not flight.done():
                now = loop.time()
                if now >= next_capture:
                    await _capture_current(f"{label} travel", required=False)
                    next_capture = loop.time() + continuous_capture_interval_s
                await asyncio.sleep(0.1)
            await flight

        async def _flight() -> None:
            """Arm → survey → normal land. Any error here (including a geofence
            breach / dangerous location) propagates to the emergency lander."""
            nonlocal mission_complete
            await drone.action.arm()
            await navigator.start_offboard()
            await navigator.fly_to(
                home_n,
                home_e,
                takeoff_d,
                ignore_height=False,
                validate_target=False,
            )
            await navigator.hover(float(m.get("takeoff_settle_s", 1.0)), ignore_height=False)
            await _capture_current("takeoff settle", required=False)

            for i, wp in enumerate(waypoints):
                tn, te = float(wp["n"]), float(wp["e"])
                print(f"--- Waypoint {i + 1}/{len(waypoints)} -> N={tn:.2f} E={te:.2f} ---")
                await _fly_to_with_captures(tn, te, takeoff_d, f"waypoint {i + 1}")
                await navigator.hover(hover_s, ignore_height=False)
                await _capture_current(f"waypoint {i + 1} hover", required=True)

            await navigator.send_velocity(0.0, 0.0, 0.0)
            await drone.offboard.stop()
            await drone.action.land()
            async for in_air in drone.telemetry.in_air():
                if not in_air:
                    break
                await asyncio.sleep(0.3)
            try:
                await drone.action.disarm()
            except Exception:
                pass
            mission_complete = True

        # Ctrl+C, kill signal, crash, or geofence breach -> land before exiting.
        try:
            await fly_with_emergency_land(_flight(), drone, navigator)
        except GeofenceViolation as exc:
            print(f"GEOFENCE (dangerous location): {exc}")

    except Exception as exc:
        print(f"Mission setup error: {exc}")
        await emergency_land_mavsdk(drone, navigator)
        raise
    finally:
        try:
            save_mission_report(
                arena,
                observations,
                OUTPUT_DIR,
                simulated=False,
                mission_status="complete" if mission_complete else "partial",
            )
        except Exception as exc:
            print(f"Could not save latest map/report during cleanup: {exc}")
        rs.stop()
        shutdown_uwb()


def main() -> None:
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_mission(cfg_arg))


if __name__ == "__main__":
    main()
