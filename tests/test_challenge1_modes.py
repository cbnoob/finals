"""Challenge 1 run-mode entry points."""

from run_challenge1_6waypoints import SIX_WAYPOINTS


def test_six_waypoint_mode_stays_inside_current_safe_zone():
    assert len(SIX_WAYPOINTS) == 6
    for wp in SIX_WAYPOINTS:
        assert 0.5 <= wp["n"] <= 9.5
        assert 0.5 <= wp["e"] <= 4.5
