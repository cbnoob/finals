"""UWB anchor geofence."""

import pytest

from common.geofence import ArenaBounds, GeofenceViolation


def _bounds() -> ArenaBounds:
    return ArenaBounds(
        n_min=0.0, n_max=10.0, e_min=0.0, e_max=10.0, safety_margin_m=0.5
    )


def test_in_anchor_zone():
    b = _bounds()
    assert b.in_anchor_zone(5.0, 5.0)
    assert not b.in_anchor_zone(11.0, 5.0)


def test_safe_zone_is_inset():
    b = _bounds()
    assert b.in_safe_zone(0.6, 0.6)
    assert not b.in_safe_zone(0.2, 0.2)


def test_validate_waypoint_inside_passes():
    _bounds().validate_waypoints([{"n": 5.0, "e": 5.0}])


def test_validate_waypoint_outside_raises():
    b = _bounds()
    with pytest.raises(GeofenceViolation):
        b.validate_waypoints([{"n": 0.1, "e": 0.1}])


def test_check_position_outside_raises():
    b = _bounds()
    with pytest.raises(GeofenceViolation):
        b.check_position(-1.0, 5.0)


def test_clamp_to_safe_zone():
    b = _bounds()
    n, e = b.clamp_to_safe_zone(0.0, 9.9)
    assert n == 0.5
    assert e == 9.5
