"""Position-NED helpers for organiser move_it.py style mapping flight."""

import asyncio
import sys
import types

import pytest

from common.position_nav import LocalNedTarget, PositionNedNavigator
from common.position_nav import world_to_local_ned
from common.uwb_listener import set_simulated_position
from common.velocity_nav import NavGains


def test_world_to_local_ned_uses_home_as_origin():
    target = world_to_local_ned(
        target_n=4.0,
        target_e=1.5,
        target_d=-2.0,
        home_n=1.0,
        home_e=-0.5,
    )
    assert target.north_m == 3.0
    assert target.east_m == 2.0
    assert target.down_m == -2.0


class _PositionNedYaw:
    def __init__(self, north_m, east_m, down_m, yaw_deg):
        self.north_m = north_m
        self.east_m = east_m
        self.down_m = down_m
        self.yaw_deg = yaw_deg


class _VelocityNedYaw:
    def __init__(self, north_m_s, east_m_s, down_m_s, yaw_deg):
        self.north_m_s = north_m_s
        self.east_m_s = east_m_s
        self.down_m_s = down_m_s
        self.yaw_deg = yaw_deg


class _FakeOffboard:
    def __init__(self):
        self.calls = []

    async def set_position_velocity_ned(self, pos, vel):
        self.calls.append((pos, vel))

    async def start(self):
        pass


class _FakeDrone:
    def __init__(self):
        self.offboard = _FakeOffboard()


@pytest.fixture(autouse=True)
def _fake_mavsdk_offboard(monkeypatch):
    offboard_mod = types.SimpleNamespace(
        PositionNedYaw=_PositionNedYaw,
        VelocityNedYaw=_VelocityNedYaw,
        OffboardError=RuntimeError,
    )
    mavsdk_mod = types.SimpleNamespace(offboard=offboard_mod)
    monkeypatch.setitem(sys.modules, "mavsdk", mavsdk_mod)
    monkeypatch.setitem(sys.modules, "mavsdk.offboard", offboard_mod)


def test_position_ned_takeoff_waits_for_height():
    set_simulated_position(0.0, 0.0)
    drone = _FakeDrone()
    down_values = iter([0.0, -0.6, -1.2, -1.85, -1.95])
    current_down = {"value": 0.0}

    def _get_down():
        try:
            current_down["value"] = next(down_values)
        except StopIteration:
            pass
        return current_down["value"]

    nav = PositionNedNavigator(
        drone,
        NavGains(d_threshold=0.1, max_vel_z=0.3),
        home_n=0.0,
        home_e=0.0,
        get_yaw=lambda: 0.0,
        get_down=_get_down,
    )

    asyncio.run(nav.fly_to(0.0, 0.0, -2.0, ignore_height=False, timeout_s=2.0))

    assert len(drone.offboard.calls) > 1
    assert drone.offboard.calls[-1][0].down_m == -2.0


def test_hover_keeps_commanded_flight_height():
    set_simulated_position(1.0, 1.0)
    drone = _FakeDrone()
    nav = PositionNedNavigator(
        drone,
        NavGains(),
        home_n=0.0,
        home_e=0.0,
        get_yaw=lambda: 0.0,
        get_down=lambda: -1.8,
    )
    nav._current_target = LocalNedTarget(1.0, 1.0, -2.0)

    asyncio.run(nav.hover(0.01, ignore_height=False))

    assert drone.offboard.calls
    assert drone.offboard.calls[-1][0].down_m == -2.0
    assert drone.offboard.calls[0][1].down_m_s < 0.0
