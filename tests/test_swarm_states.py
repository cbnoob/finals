"""Swarm FSM state ordering for the new search -> zone -> land flow."""

import pytest

from challenge2_swarm.obstacle import MapObstacleSensor, ObstacleBox
from challenge2_swarm.swarm_core import DroneContext, DroneState, setup_obstacle_sensors


def test_states_present():
    names = {s.name for s in DroneState}
    assert {"TAKEOFF", "SEARCH", "GO_TO_ZONE", "SNAPSHOT", "LAND", "DONE"} <= names


def test_search_precedes_zone_and_land():
    assert DroneState.SEARCH < DroneState.GO_TO_ZONE < DroneState.LAND < DroneState.DONE


def _ctx() -> DroneContext:
    return DroneContext(ip="1.1.1.1", api=object(), tag_id=0)


def test_lidar_avoidance_refuses_to_fly_when_unwired():
    # brief: flying over obstacles invalidates the score -> must fail safe.
    contexts = {"a": _ctx()}
    cfg = {"obstacle_avoidance_enabled": True, "obstacle_source": "lidar"}
    with pytest.raises(RuntimeError):
        setup_obstacle_sensors(contexts, cfg)


def test_avoidance_disabled_skips_sensor_setup():
    contexts = {"a": _ctx()}
    setup_obstacle_sensors(contexts, {"obstacle_avoidance_enabled": False})
    assert contexts["a"].obstacle_sensor is None


def test_preset_sensor_is_kept():
    ctx = _ctx()
    ctx.obstacle_sensor = MapObstacleSensor([ObstacleBox(0, 0, 1, 1)])
    contexts = {"a": ctx}
    setup_obstacle_sensors(contexts, {"obstacle_avoidance_enabled": True, "obstacle_source": "lidar"})
    assert isinstance(ctx.obstacle_sensor, MapObstacleSensor)
