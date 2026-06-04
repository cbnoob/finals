"""Config loads and has the keys the missions rely on."""

from common.config_loader import load_config


def test_config_loads():
    cfg = load_config()
    assert "mapping_drone" in cfg
    assert "navigation" in cfg
    assert "swarm" in cfg


def test_mapping_drone_keys():
    m = load_config()["mapping_drone"]
    for key in ("serial_address", "takeoff_height_m", "aruco_dictionary", "survey_waypoints"):
        assert key in m, f"missing mapping_drone.{key}"
    assert isinstance(m["survey_waypoints"], list)
    for wp in m["survey_waypoints"]:
        assert "n" in wp and "e" in wp


def test_navigation_gains_present():
    nav = load_config()["navigation"]
    for key in ("kp_xy", "kp_z", "max_vel_xy", "max_vel_z", "n_threshold"):
        assert key in nav
