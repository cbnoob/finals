"""Lawnmower search coverage + region split."""

from challenge2_swarm.search_pattern import (
    Region,
    coverage_fraction,
    lawnmower_waypoints,
    split_region,
)


def test_split_region_thirds():
    area = Region(0, 1, 0, 3)
    r0 = split_region(area, 3, 0)
    r1 = split_region(area, 3, 1)
    r2 = split_region(area, 3, 2)
    assert r0.e_min == 0 and abs(r0.e_max - 1.0) < 1e-9
    assert abs(r1.e_min - 1.0) < 1e-9 and abs(r1.e_max - 2.0) < 1e-9
    assert abs(r2.e_max - 3.0) < 1e-9


def test_split_single_drone_returns_full():
    area = Region(0, 1, 0, 1)
    assert split_region(area, 1, 0) == area


def test_lawnmower_alternates_direction():
    region = Region(0, 1, 0, 1)
    wps = lawnmower_waypoints(region, spacing=0.5)
    # First leg goes up (n_min -> n_max), second leg comes down
    assert wps[0] == (0.0, 0.0)
    assert wps[1] == (1.0, 0.0)
    assert wps[2][0] == 1.0  # next column starts at top (coming down)


def test_lawnmower_covers_region():
    region = Region(0, 1, 0, 1)
    wps = lawnmower_waypoints(region, spacing=0.25)
    # Densify path by sampling endpoints; coverage with 0.3 m sensor should be high
    cov = coverage_fraction(region, wps, sensor_radius=0.3, grid_step=0.1)
    assert cov > 0.9


def test_lawnmower_rejects_bad_spacing():
    import pytest

    with pytest.raises(ValueError):
        lawnmower_waypoints(Region(0, 1, 0, 1), spacing=0)
