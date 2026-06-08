"""Down-facing camera footprint geometry."""

from common.camera_model import (
    D430,
    D450,
    ground_footprint,
    meters_per_pixel,
    recommended_leg_spacing,
)


def test_footprint_scales_with_height():
    w1, l1 = ground_footprint(3.5, D430.color_hfov, D430.color_vfov)
    w2, l2 = ground_footprint(7.0, D430.color_hfov, D430.color_vfov)
    assert abs(w2 - 2 * w1) < 1e-6
    assert abs(l2 - 2 * l1) < 1e-6


def test_d430_footprint_at_3_5m():
    w, ln = ground_footprint(3.5, D430.color_hfov, D430.color_vfov)
    assert 4.5 < w < 5.1   # ~4.8 m wide
    assert 2.5 < ln < 2.9  # ~2.7 m


def test_d450_is_wider_than_d430():
    w430, _ = ground_footprint(3.5, D430.color_hfov, D430.color_vfov)
    w450, _ = ground_footprint(3.5, D450.color_hfov, D450.color_vfov)
    assert w450 > w430


def test_leg_spacing_has_overlap():
    _, ln = ground_footprint(3.5, D430.color_hfov, D430.color_vfov)
    spacing = recommended_leg_spacing(3.5, D430, overlap=0.2)
    assert spacing < ln  # narrower than footprint -> guarantees overlap
    assert spacing > 0


def test_meters_per_pixel_drops_with_resolution():
    mpp_low = meters_per_pixel(3.5, D430.color_hfov, 640)
    mpp_high = meters_per_pixel(3.5, D430.color_hfov, 1280)
    assert mpp_high < mpp_low
