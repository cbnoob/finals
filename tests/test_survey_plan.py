"""Auto full-area survey waypoint generation."""

from challenge1_mapping.survey_core import build_survey_waypoints


def _cfg(auto: bool) -> dict:
    return {
        "arena": {
            "uwb_bounds": {"n_min": 0.0, "n_max": 10.0, "e_min": 0.0, "e_max": 10.0},
            "safety_margin_m": 0.5,
        },
        "mapping_drone": {
            "auto_survey": auto,
            "survey_spacing_m": 2.0,
            "survey_waypoints": [{"n": 0.0, "e": 0.0}, {"n": 1.0, "e": 1.0}],
        },
    }


def test_manual_waypoints_when_auto_off():
    wps = build_survey_waypoints(_cfg(False))
    assert wps == [{"n": 0.0, "e": 0.0}, {"n": 1.0, "e": 1.0}]


def test_auto_survey_covers_safe_zone():
    wps = build_survey_waypoints(_cfg(True))
    assert len(wps) > 2
    # All waypoints must sit inside the safe zone (bounds inset by margin)
    for wp in wps:
        assert 0.5 <= wp["n"] <= 9.5
        assert 0.5 <= wp["e"] <= 9.5
    # Coverage should span most of the North/East range
    ns = [wp["n"] for wp in wps]
    es = [wp["e"] for wp in wps]
    assert max(ns) - min(ns) > 8.0
    assert max(es) - min(es) > 6.0
