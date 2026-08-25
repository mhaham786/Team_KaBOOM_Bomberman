"""Regression tests for the 31-feature representation and escape planner."""

import unittest

import numpy as np

from ..features import *
from ..features import (
    _blast_tiles, _build_danger_map, _build_time_hazard_model,
    _is_safe_at_time, _time_safe_escape_directions,
)
from .fixtures import bordered_field

def _make_test_state():
    """Create a small deterministic synthetic state for executable self-tests."""
    field = bordered_field(7)
    field[2, 3] = 1
    field[3, 2] = 1
    field[5, 2] = 1

    explosion_map = np.zeros_like(field, dtype=np.float32)
    explosion_map[4, 3] = 1.0

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("uttam", 0, True, (3, 3)),
        "others": [("enemy", 0, True, (5, 3))],
        "bombs": [((3, 5), 1)],
        "coins": [(4, 3)],
        "user_input": None,
        "explosion_map": explosion_map,
    }


def _make_escape_state(trapped=False):
    """Create a synthetic state for safe-bomb-escape tests."""
    field = bordered_field(7)
    if trapped:
        field[:, :] = -1
        field[1, 1] = 0
        field[2, 1] = 0
        field[3, 1] = 0
        field[4, 1] = 0
        field[5, 1] = 1
        position = (1, 1)
    else:
        position = (3, 3)
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("uttam", 0, True, position),
        "others": [],
        "bombs": [],
        "coins": [],
        "user_input": None,
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


def _make_open_state(size=9, position=(4, 4), bomb_available=True):
    """Create an open square arena with stone walls at the border."""
    field = bordered_field(size)
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("uttam", 0, bomb_available, position),
        "others": [],
        "bombs": [],
        "coins": [],
        "user_input": None,
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


def _run_self_test() -> None:
    state = _make_test_state()
    original_field = state["field"].copy()
    original_explosion_map = state["explosion_map"].copy()

    features = state_to_features(state)
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()

    mask = valid_action_mask(state)
    assert mask.shape == (len(ACTIONS),)
    assert mask.dtype == np.bool_
    assert mask.any()

    assert features[3] == 0.0
    assert mask[3] == np.bool_(False)

    assert features[11] == 1.0
    assert 0.0 < features[14] < 1.0

    assert features[4] == 1.0
    assert mask[5] == np.bool_(True)
    assert set(np.unique(features[25:29])).issubset({0.0, 1.0})

    assert features[5] > 0.0
    danger_map = _build_danger_map(state["field"], state["bombs"], state["explosion_map"])
    assert danger_map[3, 2] > 0.0
    assert danger_map[3, 1] == 0.0
    assert features[7] == 1.0

    blast_state = _make_open_state()
    blast_field = blast_state["field"].copy()
    blast_field[5, 4] = 1
    blast_state["field"] = blast_field
    blast_tiles = set(_blast_tiles((4, 4), blast_field))
    assert (5, 4) in blast_tiles
    assert (6, 4) in blast_tiles
    assert (7, 4) in blast_tiles
    blast_danger = _build_danger_map(blast_field, [((4, 4), 1)], np.zeros_like(blast_field))
    assert blast_danger[6, 4] > 0.0

    wall_field = blast_field.copy()
    wall_field[6, 4] = -1
    wall_tiles = set(_blast_tiles((4, 4), wall_field))
    assert (5, 4) in wall_tiles
    assert (6, 4) not in wall_tiles
    assert (7, 4) not in wall_tiles

    six_crate_state = _make_open_state()
    six_field = six_crate_state["field"].copy()
    for tile in ((5, 4), (6, 4), (7, 4), (3, 4), (2, 4), (1, 4)):
        six_field[tile] = 1
    six_crate_state["field"] = six_field
    assert state_to_features(six_crate_state)[29] == np.float32(0.5)

    max_crate_state = _make_open_state()
    max_field = max_crate_state["field"].copy()
    for tile in _blast_tiles((4, 4), max_field):
        if tile != (4, 4):
            max_field[tile] = 1
    max_crate_state["field"] = max_field
    assert state_to_features(max_crate_state)[29] == np.float32(1.0)

    opponent_blast_state = _make_open_state()
    opponent_field = opponent_blast_state["field"].copy()
    opponent_field[5, 4] = 1
    opponent_blast_state["field"] = opponent_field
    opponent_blast_state["others"] = [("enemy", 0, True, (6, 4))]
    assert state_to_features(opponent_blast_state)[30] == 1.0

    hazard_field = _make_open_state()["field"]
    timer_hazard = _build_time_hazard_model(hazard_field, [((4, 6), 1)], np.zeros_like(hazard_field))
    assert _is_safe_at_time((4, 4), 1, timer_hazard)
    assert not _is_safe_at_time((4, 4), 2, timer_hazard)
    assert not _is_safe_at_time((4, 4), 3, timer_hazard)
    assert _is_safe_at_time((4, 4), 4, timer_hazard)

    hypothetical_hazard = _build_time_hazard_model(
        hazard_field,
        [],
        np.zeros_like(hazard_field),
        hypothetical_origin=(4, 4),
    )
    assert _is_safe_at_time((4, 4), 4, hypothetical_hazard)
    assert not _is_safe_at_time((4, 4), 5, hypothetical_hazard)
    assert not _is_safe_at_time((4, 4), 6, hypothetical_hazard)
    assert _is_safe_at_time((4, 4), 7, hypothetical_hazard)

    map_one = np.zeros_like(hazard_field, dtype=np.float32)
    map_one[5, 4] = 1.0
    map_one_hazard = _build_time_hazard_model(hazard_field, [], map_one)
    assert not _is_safe_at_time((5, 4), 1, map_one_hazard)

    map_zero = np.zeros_like(hazard_field, dtype=np.float32)
    map_zero_hazard = _build_time_hazard_model(hazard_field, [], map_zero, min_horizon=1)
    assert _is_safe_at_time((5, 4), 1, map_zero_hazard)

    timing_state = _make_open_state(position=(4, 4), bomb_available=False)
    timing_state["explosion_map"][5, 4] = 1.0
    actual_escape = _time_safe_escape_directions(timing_state)
    assert actual_escape[1] == 0.0

    timing_state["explosion_map"][5, 4] = 0.0
    actual_escape_after_last_tick = _time_safe_escape_directions(timing_state)
    assert actual_escape_after_last_tick[1] == 1.0

    timing_state["self"] = ("uttam", 0, True, (4, 4))
    hypothetical_escape = _time_safe_escape_directions(timing_state)
    assert hypothetical_escape[1] == 1.0

    origin_danger_state = _make_open_state(position=(4, 4), bomb_available=True)
    origin_danger_state["explosion_map"][4, 4] = 1.0
    assert not _time_safe_escape_directions(origin_danger_state).any()

    actual_origin_bomb_state = _make_open_state(position=(4, 4), bomb_available=True)
    actual_origin_bomb_state["bombs"] = [((4, 4), 3)]
    actual_origin_bomb_state["explosion_map"][5, 4] = 1.0
    assert valid_action_mask(actual_origin_bomb_state)[5] == np.bool_(False)
    assert _time_safe_escape_directions(actual_origin_bomb_state)[1] == 0.0

    moving_escape_state = _make_open_state(position=(4, 4), bomb_available=True)
    moving_escape = _time_safe_escape_directions(moving_escape_state)
    moving_hazard = _build_time_hazard_model(
        moving_escape_state["field"],
        [],
        moving_escape_state["explosion_map"],
        hypothetical_origin=(4, 4),
    )
    assert not _is_safe_at_time((5, 4), 5, moving_hazard)
    assert moving_escape[1] == 1.0

    four_crate_state = dict(state)
    four_crate_field = bordered_field(7)
    four_crate_field[3, 2] = 1
    four_crate_field[4, 3] = 1
    four_crate_field[3, 4] = 1
    four_crate_field[2, 3] = 1
    four_crate_state["field"] = four_crate_field
    four_crate_state["others"] = []
    four_crate_state["bombs"] = []
    four_crate_state["coins"] = []
    four_crate_state["explosion_map"] = np.zeros_like(four_crate_field, dtype=np.float32)
    assert state_to_features(four_crate_state)[29] == np.float32(4.0 / 12.0)

    unreachable_opponent_state = dict(state)
    opponent_field = bordered_field(7)
    opponent_field[2, 1:6] = 1
    unreachable_opponent_state["field"] = opponent_field
    unreachable_opponent_state["self"] = ("uttam", 0, True, (1, 1))
    unreachable_opponent_state["others"] = [
        ("near_blocked", 0, True, (3, 1)),
        ("far_reachable", 0, True, (1, 5)),
    ]
    unreachable_opponent_state["bombs"] = []
    unreachable_opponent_state["coins"] = []
    unreachable_opponent_state["explosion_map"] = np.zeros_like(opponent_field, dtype=np.float32)
    opponent_features = state_to_features(unreachable_opponent_state)
    assert opponent_features[22] == 1.0
    assert 0.0 < opponent_features[24] < 1.0

    assert mask[1] == np.bool_(True)

    bomb_blocked_state = dict(state)
    bomb_blocked_state["bombs"] = [((4, 3), 3)]
    assert valid_action_mask(bomb_blocked_state)[1] == np.bool_(False)

    no_bomb_state = dict(state)
    no_bomb_state["self"] = ("uttam", 0, False, (3, 3))
    assert state_to_features(no_bomb_state)[4] == 0.0
    assert valid_action_mask(no_bomb_state)[5] == np.bool_(False)

    assert state_to_features(None) is None

    escape_state = _make_escape_state(trapped=False)
    escape_field_before = escape_state["field"].copy()
    escape_explosion_before = escape_state["explosion_map"].copy()
    assert has_safe_bomb_escape(escape_state, 4) is True
    np.testing.assert_array_equal(escape_state["field"], escape_field_before)
    np.testing.assert_array_equal(escape_state["explosion_map"], escape_explosion_before)

    trapped_state = _make_escape_state(trapped=True)
    assert has_safe_bomb_escape(trapped_state, 4) is False

    for invalid_steps in (0, -1, 1.5, True):
        try:
            has_safe_bomb_escape(escape_state, invalid_steps)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid max_steps")

    np.testing.assert_array_equal(state["field"], original_field)
    np.testing.assert_array_equal(state["explosion_map"], original_explosion_map)

    print("features.py V2 timing regression passed")
    print("features.py self-test passed")

class FeatureRegressionTests(unittest.TestCase):
    def test_existing_regressions(self):
        _run_self_test()


if __name__ == "__main__":
    unittest.main()
