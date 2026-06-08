"""Swarm FSM state ordering for the new search -> zone -> land flow."""

from challenge2_swarm.swarm_core import DroneState


def test_states_present():
    names = {s.name for s in DroneState}
    assert {"TAKEOFF", "SEARCH", "GO_TO_ZONE", "SNAPSHOT", "LAND", "DONE"} <= names


def test_search_precedes_zone_and_land():
    assert DroneState.SEARCH < DroneState.GO_TO_ZONE < DroneState.LAND < DroneState.DONE
