"""Emergency landing for both challenges."""

import asyncio

from common.emergency import (
    SwarmEmergencyGuard,
    emergency_land_mavsdk,
    fly_with_emergency_land,
    land_all_hulas,
)


class _FakeHula:
    def __init__(self):
        self.landed = False
        self.hovered = False

    def hover(self):
        self.hovered = True

    def land(self):
        self.landed = True


class _Ctx:
    def __init__(self, ip):
        self.ip = ip
        self.api = _FakeHula()
        self.landed = False


def test_land_all_hulas_lands_every_drone():
    contexts = {"a": _Ctx("a"), "b": _Ctx("b")}
    land_all_hulas(contexts)
    for ctx in contexts.values():
        assert ctx.api.landed is True
        assert ctx.landed is True


def test_land_all_hulas_survives_api_error():
    class _Broken(_Ctx):
        def __init__(self, ip):
            super().__init__(ip)
            self.api.land = self._boom

        def _boom(self):
            raise RuntimeError("link lost")

    contexts = {"a": _Broken("a"), "b": _Ctx("b")}
    land_all_hulas(contexts)  # must not raise
    assert contexts["b"].api.landed is True


def test_swarm_guard_installs_and_restores():
    import signal

    contexts = {"a": _Ctx("a")}
    before = signal.getsignal(signal.SIGINT)
    with SwarmEmergencyGuard(contexts):
        assert signal.getsignal(signal.SIGINT) is not before
    assert signal.getsignal(signal.SIGINT) is before


# --- MAVSDK async side (fakes) ---
class _FakeAction:
    def __init__(self):
        self.landed = False

    async def land(self):
        self.landed = True


class _FakeOffboard:
    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _FakeDrone:
    def __init__(self):
        self.action = _FakeAction()
        self.offboard = _FakeOffboard()


class _FakeNav:
    def __init__(self):
        self.zeroed = False

    async def send_velocity(self, vn, ve, vd):
        self.zeroed = (vn, ve, vd) == (0.0, 0.0, 0.0)


def test_emergency_land_mavsdk_lands():
    drone = _FakeDrone()
    nav = _FakeNav()
    asyncio.run(emergency_land_mavsdk(drone, nav))
    assert drone.action.landed is True
    assert drone.offboard.stopped is True
    assert nav.zeroed is True


def test_fly_with_emergency_land_lands_on_crash():
    drone = _FakeDrone()
    nav = _FakeNav()

    async def _flight():
        raise RuntimeError("boom")

    async def _run():
        try:
            await fly_with_emergency_land(_flight(), drone, nav)
        except RuntimeError:
            return "raised"
        return "no-raise"

    result = asyncio.run(_run())
    assert result == "raised"
    assert drone.action.landed is True  # landed before re-raising


def test_fly_with_emergency_land_normal_completion():
    drone = _FakeDrone()
    nav = _FakeNav()

    async def _flight():
        return None

    asyncio.run(fly_with_emergency_land(_flight(), drone, nav))
    # Normal completion must NOT trigger an emergency land
    assert drone.action.landed is False
