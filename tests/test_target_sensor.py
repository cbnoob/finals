"""SimTargetSensor proximity sensing."""

from challenge2_swarm.sim.ground_robots import GroundRobot
from challenge2_swarm.target_sensor import SimTargetSensor
from common.uwb_c2 import SimulatedUWBC2


class _Ctx:
    def __init__(self, tag_id):
        self.tag_id = tag_id
        self.stream = None


def test_sees_robot_within_footprint():
    uwb = SimulatedUWBC2({0: (0.5, 0.5)})
    robot = GroundRobot(7, 0.5, 0.5, drift_radius=0.0)
    sensor = SimTargetSensor(uwb, [robot], camera_footprint_m=0.35)
    seen = sensor.sense(_Ctx(0))
    assert len(seen) == 1
    assert seen[0].target_id == 7
    assert seen[0].confidence > 0.5


def test_ignores_robot_outside_footprint():
    uwb = SimulatedUWBC2({0: (0.0, 0.0)})
    robot = GroundRobot(1, 0.9, 0.9, drift_radius=0.0)
    sensor = SimTargetSensor(uwb, [robot], camera_footprint_m=0.35)
    assert sensor.sense(_Ctx(0)) == []


def test_save_snapshot_writes_file(tmp_path):
    uwb = SimulatedUWBC2({0: (0.5, 0.5)})
    robot = GroundRobot(2, 0.5, 0.5, drift_radius=0.0)
    sensor = SimTargetSensor(uwb, [robot], camera_footprint_m=0.35)
    seen = sensor.sense(_Ctx(0))
    out = tmp_path / "snap.jpg"
    n = sensor.save_snapshot(_Ctx(0), seen, out)
    assert n == 1
    assert out.exists()
