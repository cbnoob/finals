"""
MAVSDK position-NED navigation for the mapping drone.

Based on the organiser's move_it.py sample: enter offboard with an initial
PositionNedYaw + VelocityNedYaw setpoint, then fly by set_position_velocity_ned.
UWB is still used for arena/geofence checks and for deciding when a surveyed
world waypoint has been reached.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from common.uwb_listener import get_uwb_position
from common.velocity_nav import NavGains, compute_hover_velocity

if TYPE_CHECKING:
    from common.geofence import ArenaBounds
    from mavsdk import System


@dataclass(frozen=True)
class LocalNedTarget:
    north_m: float
    east_m: float
    down_m: float


def world_to_local_ned(
    target_n: float,
    target_e: float,
    target_d: float,
    home_n: float,
    home_e: float,
    origin_mode: str = "home",
) -> LocalNedTarget:
    """Convert UWB world N/E to the PositionNedYaw frame."""
    if origin_mode == "uwb":
        return LocalNedTarget(
            north_m=target_n,
            east_m=target_e,
            down_m=target_d,
        )
    if origin_mode != "home":
        raise ValueError("origin_mode must be 'home' or 'uwb'")
    return LocalNedTarget(
        north_m=target_n - home_n,
        east_m=target_e - home_e,
        down_m=target_d,
    )


class PositionNedNavigator:
    def __init__(
        self,
        drone: "System",
        gains: NavGains,
        home_n: float,
        home_e: float,
        get_yaw: Callable[[], float],
        get_down: Callable[[], float],
        origin_mode: str = "home",
        geofence: "ArenaBounds | None" = None,
    ) -> None:
        self.drone = drone
        self.gains = gains
        self.home_n = home_n
        self.home_e = home_e
        self._get_yaw = get_yaw
        self._get_down = get_down
        if origin_mode not in ("home", "uwb"):
            raise ValueError("origin_mode must be 'home' or 'uwb'")
        self.origin_mode = origin_mode
        self._geofence = geofence
        self.takeoff_yaw = 0.0
        self._current_target = LocalNedTarget(0.0, 0.0, 0.0)

    def _target_for_world(self, n: float, e: float, d: float) -> LocalNedTarget:
        return world_to_local_ned(
            n,
            e,
            d,
            self.home_n,
            self.home_e,
            origin_mode=self.origin_mode,
        )

    async def _set_position_velocity(
        self,
        target: LocalNedTarget,
        vn: float,
        ve: float,
        vd: float,
    ) -> None:
        from mavsdk.offboard import PositionNedYaw, VelocityNedYaw

        pos = PositionNedYaw(
            target.north_m,
            target.east_m,
            target.down_m,
            self.takeoff_yaw,
        )
        vel = VelocityNedYaw(vn, ve, vd, self.takeoff_yaw)
        self._current_target = target
        await self.drone.offboard.set_position_velocity_ned(pos, vel)

    async def send_velocity(self, vn: float, ve: float, vd: float) -> None:
        """Emergency-compatible zeroing hook."""
        await self._set_position_velocity(self._current_target, vn, ve, vd)

    async def start_offboard(self) -> None:
        from mavsdk.offboard import OffboardError

        self.takeoff_yaw = self._get_yaw()
        current_n, current_e, uwb_ok = get_uwb_position()
        if not uwb_ok:
            current_n, current_e = self.home_n, self.home_e
        initial = self._target_for_world(current_n, current_e, self._get_down())
        await self._set_position_velocity(initial, 0.0, 0.0, 0.0)
        try:
            await self.drone.offboard.start()
        except OffboardError:
            raise

    async def fly_to(
        self,
        target_n: float,
        target_e: float,
        target_d: float,
        *,
        ignore_height: bool = True,
        timeout_s: float = 90.0,
        validate_target: bool = True,
    ) -> None:
        if self._geofence is not None and validate_target:
            self._geofence.validate_point(target_n, target_e, "position target")

        target = self._target_for_world(target_n, target_e, target_d)

        print(f"Position-NED fly to N={target_n:.2f} E={target_e:.2f} D={target_d:.2f}")
        start_t = asyncio.get_running_loop().time()
        last_sent = 0.0

        while True:
            current_n, current_e, uwb_ok = get_uwb_position()
            if not uwb_ok:
                await asyncio.sleep(0.1)
                continue

            if self._geofence is not None:
                self._geofence.check_position(current_n, current_e)

            err_n = target_n - current_n
            err_e = target_e - current_e
            current_d = self._get_down()
            err_d = target_d - current_d
            at_xy = (
                abs(err_n) < self.gains.n_threshold and
                abs(err_e) < self.gains.e_threshold
            )
            at_height = ignore_height or abs(err_d) < self.gains.d_threshold
            if at_xy and at_height:
                await self._set_position_velocity(target, 0.0, 0.0, 0.0)
                print(f"Waypoint reached (D={current_d:.2f})")
                return

            now = asyncio.get_running_loop().time()
            if now - last_sent >= 0.2:
                # PositionNedYaw is the real position target; PX4 may move fast
                # toward a far setpoint. Send a rolling nearby target so the
                # position controller cannot chase a distant point aggressively.
                dist = math.hypot(err_n, err_e)
                if dist > 1e-6:
                    speed = min(self.gains.max_vel_xy, max(0.05, self.gains.kp_xy * dist))
                    vn = speed * err_n / dist
                    ve = speed * err_e / dist
                    step = min(float(self.gains.max_position_step_m), dist)
                    command_n = current_n + step * err_n / dist
                    command_e = current_e + step * err_e / dist
                else:
                    vn = ve = 0.0
                    command_n = target_n
                    command_e = target_e
                if ignore_height:
                    vd = 0.0
                else:
                    vd = max(
                        -self.gains.max_vel_z,
                        min(self.gains.max_vel_z, self.gains.kp_z * err_d),
                    )
                command_target = self._target_for_world(command_n, command_e, target_d)
                await self._set_position_velocity(command_target, vn, ve, vd)
                last_sent = now

            if now - start_t > timeout_s:
                print("Waypoint timeout; holding current position")
                hold_target = self._target_for_world(current_n, current_e, target_d)
                await self._set_position_velocity(hold_target, 0.0, 0.0, 0.0)
                return

            await asyncio.sleep(0.1)

    async def hover(self, seconds: float, *, ignore_height: bool = True) -> None:
        hover_n, hover_e, ok = get_uwb_position()
        if not ok:
            raise RuntimeError("UWB not ready for hover")
        if self._geofence is not None:
            self._geofence.check_position(hover_n, hover_e)

        # Height is not part of the geofence, but the mapping mission should
        # keep holding the last commanded flight altitude during waypoint hovers.
        target_d = self._current_target.down_m
        target = self._target_for_world(hover_n, hover_e, target_d)
        print(f"Position-NED hover lock N={hover_n:.2f} E={hover_e:.2f} D={target_d:.2f}")
        end = asyncio.get_running_loop().time() + seconds

        while asyncio.get_running_loop().time() < end:
            current_n, current_e, uwb_ok = get_uwb_position()
            if not uwb_ok:
                await asyncio.sleep(0.1)
                continue

            if self._geofence is not None:
                self._geofence.check_position(current_n, current_e)

            vn, ve, vd = compute_hover_velocity(
                hover_n - current_n,
                hover_e - current_e,
                target_d - self._get_down(),
                self.gains,
                ignore_height=ignore_height,
            )
            await self._set_position_velocity(target, vn, ve, vd)
            await asyncio.sleep(0.1)

        await self._set_position_velocity(target, 0.0, 0.0, 0.0)
