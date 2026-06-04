"""Pure controller math for the mapping drone velocity navigation."""

import math

from common.velocity_nav import NavGains, compute_hover_velocity, compute_nav_velocity


def test_at_goal_returns_zero():
    g = NavGains()
    vn, ve, vd, at_goal = compute_nav_velocity(0.02, -0.03, 0.0, g, ignore_height=True)
    assert at_goal is True
    assert (vn, ve, vd) == (0.0, 0.0, 0.0)


def test_velocity_proportional_to_error():
    g = NavGains(kp_xy=0.1, n_threshold=0.1, e_threshold=0.1)
    vn, ve, vd, at_goal = compute_nav_velocity(2.0, 0.0, 0.0, g, ignore_height=True)
    assert at_goal is False
    # 0.1 * 2.0 = 0.2, under max_vel_xy=0.5 so unscaled
    assert math.isclose(vn, 0.2, rel_tol=1e-6)
    assert ve == 0.0
    assert vd == 0.0  # ignore_height


def test_horizontal_speed_clamped():
    g = NavGains(kp_xy=0.1, max_vel_xy=0.5)
    # error 10,10 -> raw v (1.0,1.0) -> hypot ~1.414 -> scaled to 0.5 total
    vn, ve, _, _ = compute_nav_velocity(10.0, 10.0, 0.0, g, ignore_height=True)
    assert math.isclose(math.hypot(vn, ve), 0.5, rel_tol=1e-6)


def test_vertical_clamped_when_not_ignoring_height():
    g = NavGains(kp_z=0.1, max_vel_z=0.3, d_threshold=0.1)
    vn, ve, vd, _ = compute_nav_velocity(0.0, 0.0, 100.0, g, ignore_height=False)
    assert math.isclose(vd, 0.3, rel_tol=1e-6)


def test_ignore_height_zeroes_vd():
    g = NavGains()
    _, _, vd, _ = compute_nav_velocity(0.0, 0.0, 100.0, g, ignore_height=True)
    assert vd == 0.0


def test_hover_deadband_zeroes_small_error():
    g = NavGains(hover_deadband=0.03)
    vn, ve, vd = compute_hover_velocity(0.01, 0.01, 0.01, g, ignore_height=False)
    assert (vn, ve, vd) == (0.0, 0.0, 0.0)


def test_hover_corrects_drift():
    g = NavGains(kp_xy=0.1, hover_deadband=0.03)
    vn, ve, _ = compute_hover_velocity(0.5, 0.0, 0.0, g)
    assert vn > 0.0
    assert ve == 0.0
