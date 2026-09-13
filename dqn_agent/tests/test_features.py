import unittest

import numpy as np

from ...common.features import classic_peace_improved_oc33
from ...common.helpers import (
    action_mask,
    blast_tiles,
    build_danger_map,
    build_time_hazard_model,
    has_safe_bomb_escape,
    is_safe_at_time,
    time_safe_escape_directions,
)
from .. import config
from .fixtures import bordered_field


def make_state(position=(3, 3), bomb_available=True):
    field = bordered_field(9)
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("dqn", 0, bomb_available, position),
        "others": [],
        "bombs": [],
        "coins": [],
        "user_input": None,
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


class FeatureTests(unittest.TestCase):
    def test_shape_dtype_finiteness_and_none(self):
        features = classic_peace_improved_oc33(make_state())
        self.assertEqual(features.shape, (config.OBSERVATION_COUNT,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertIsNone(classic_peace_improved_oc33(None))

    def test_walkability_coin_direction_and_action_mask(self):
        state = make_state()
        state["field"][3, 2] = 1
        state["coins"] = [(4, 3)]
        features = classic_peace_improved_oc33(state)
        mask = action_mask(state)
        self.assertEqual(features[0], 0.0)
        self.assertEqual(features[11], 1.0)
        self.assertGreater(features[14], 0.0)
        self.assertFalse(mask[0])
        self.assertTrue(mask[4])
        self.assertTrue(mask[5])

    def test_bombs_and_opponents_block_movement(self):
        state = make_state()
        state["bombs"] = [((4, 3), 3)]
        state["others"] = [("enemy", 0, True, (2, 3))]
        mask = action_mask(state)
        features = classic_peace_improved_oc33(state)
        self.assertFalse(mask[1])
        self.assertFalse(mask[3])
        self.assertEqual(features[1], 0.0)
        self.assertEqual(features[3], 1.0)

    def test_crates_do_not_stop_blasts_and_walls_do(self):
        field = bordered_field(9)
        field[5, 4] = 1
        reached = set(blast_tiles((4, 4), field))
        self.assertIn((5, 4), reached)
        self.assertIn((6, 4), reached)
        field[6, 4] = -1
        blocked = set(blast_tiles((4, 4), field))
        self.assertIn((5, 4), blocked)
        self.assertNotIn((6, 4), blocked)
        self.assertNotIn((7, 4), blocked)

    def test_danger_and_engine_timing(self):
        state = make_state(position=(4, 4), bomb_available=False)
        danger = build_danger_map(state["field"], [((4, 6), 1)], state["explosion_map"])
        self.assertGreater(danger[4, 4], 0.0)
        hazard = build_time_hazard_model(state["field"], [((4, 6), 1)], state["explosion_map"])
        self.assertTrue(is_safe_at_time((4, 4), 1, hazard))
        self.assertFalse(is_safe_at_time((4, 4), 2, hazard))
        self.assertFalse(is_safe_at_time((4, 4), 3, hazard))
        self.assertTrue(is_safe_at_time((4, 4), 4, hazard))

        hypothetical = build_time_hazard_model(
            state["field"], [], state["explosion_map"], hypothetical_origin=(4, 4)
        )
        self.assertTrue(is_safe_at_time((4, 4), 4, hypothetical))
        self.assertFalse(is_safe_at_time((4, 4), 5, hypothetical))
        self.assertFalse(is_safe_at_time((4, 4), 6, hypothetical))
        self.assertTrue(is_safe_at_time((4, 4), 7, hypothetical))

        state["explosion_map"][5, 4] = 1.0
        active = build_time_hazard_model(state["field"], [], state["explosion_map"])
        self.assertFalse(is_safe_at_time((5, 4), 1, active))
        state["explosion_map"][5, 4] = 0.0
        expired = build_time_hazard_model(
            state["field"], [], state["explosion_map"], min_horizon=1
        )
        self.assertTrue(is_safe_at_time((5, 4), 1, expired))

    def test_actual_and_hypothetical_first_move_timing(self):
        state = make_state(position=(4, 4), bomb_available=False)
        state["explosion_map"][5, 4] = 1.0
        actual = time_safe_escape_directions(
            state["field"], (4, 4), False, [], set(), state["explosion_map"]
        )
        self.assertEqual(actual[1], 0.0)
        state["explosion_map"][5, 4] = 0.0
        actual = time_safe_escape_directions(
            state["field"], (4, 4), False, [], set(), state["explosion_map"]
        )
        self.assertEqual(actual[1], 1.0)
        hypothetical = time_safe_escape_directions(
            state["field"], (4, 4), True, [], set(), state["explosion_map"]
        )
        self.assertEqual(hypothetical[1], 1.0)

    def test_origin_hazards_and_full_horizon_movement(self):
        state = make_state(position=(4, 4))
        state["explosion_map"][4, 4] = 1.0
        directions = time_safe_escape_directions(
            state["field"], (4, 4), True, [], set(), state["explosion_map"]
        )
        self.assertFalse(directions.any())
        state["explosion_map"][:] = 0
        state["bombs"] = [((4, 4), 3)]
        state["explosion_map"][5, 4] = 1.0
        directions = time_safe_escape_directions(
            state["field"], (4, 4), True, state["bombs"], set(), state["explosion_map"]
        )
        self.assertEqual(directions[1], 0.0)
        moving = make_state(position=(4, 4))
        directions = time_safe_escape_directions(
            moving["field"], (4, 4), True, [], set(), moving["explosion_map"]
        )
        hazard = build_time_hazard_model(
            moving["field"], [], moving["explosion_map"], hypothetical_origin=(4, 4)
        )
        self.assertFalse(is_safe_at_time((5, 4), 5, hazard))
        self.assertEqual(directions[1], 1.0)

    def test_safe_bomb_escape_and_trapped_state(self):
        state = make_state(position=(4, 4))
        field_before = state["field"].copy()
        explosion_before = state["explosion_map"].copy()
        self.assertTrue(has_safe_bomb_escape(
            state["field"], (4, 4), [], set(), state["explosion_map"], 4
        ))
        np.testing.assert_array_equal(state["field"], field_before)
        np.testing.assert_array_equal(state["explosion_map"], explosion_before)
        field = np.full((7, 7), -1, dtype=np.int64)
        field[1:5, 1] = 0
        explosion_map = np.zeros_like(field, dtype=np.float32)
        self.assertFalse(has_safe_bomb_escape(field, (1, 1), [], set(), explosion_map, 4))
        with self.assertRaises(ValueError):
            has_safe_bomb_escape(field, (1, 1), [], set(), explosion_map, 0)
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                has_safe_bomb_escape(field, (1, 1), [], set(), explosion_map, value)


if __name__ == "__main__":
    unittest.main()
