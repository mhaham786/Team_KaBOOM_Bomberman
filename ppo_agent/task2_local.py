import numpy as np

from ..common.helpers import (
    MOVEMENTS, BOMB_TIMER, add_position, is_walkable, bomb_positions,
    opponent_positions, bfs_first_step, direction_and_distance_features,
    crate_adjacent_targets, bomb_effects_from, can_place_bomb, has_safe_bomb_escape,
    build_time_hazard_model, has_time_safe_path_after_first_move,
)


def loot_crate_bfs_oc9(game_state):
    """Four walkable moves, four target directions, one normalized distance."""
    if game_state is None:
        return None
    field = game_state['field']
    origin = tuple(game_state['self'][3])
    bombs = bomb_positions(game_state['bombs'])
    others = opponent_positions(game_state['others'])
    moves = [is_walkable(add_position(origin, delta), field, bombs, others)
             for delta in MOVEMENTS]
    direction, distance = bfs_first_step(field, origin, game_state['coins'], bombs, others)
    if distance is None:
        targets = crate_adjacent_targets(field, bombs, others)
        direction, distance = bfs_first_step(field, origin, targets, bombs, others)
    route = direction_and_distance_features(direction, distance, field.shape)
    return np.concatenate((np.asarray(moves, dtype=np.float32), route))


def task2_action_mask(game_state):
    """Keep moves with a timed escape; bomb only for safely reachable crate damage."""
    field = game_state['field']
    origin = tuple(game_state['self'][3])
    bombs = game_state['bombs']
    others = opponent_positions(game_state['others'])
    occupied = bomb_positions(bombs)
    explosions = game_state['explosion_map']
    hazard = build_time_hazard_model(field, bombs, explosions, min_horizon=1)
    mask = np.zeros(6, dtype=np.bool_)
    for index, delta in enumerate(MOVEMENTS):
        mask[index] = is_walkable(add_position(origin, delta), field, occupied, others)
        if mask[index]:
            mask[index] = has_time_safe_path_after_first_move(
                origin, delta, 1, field, others, hazard)
    if can_place_bomb(origin, game_state['self'][2], occupied):
        crates, _ = bomb_effects_from(origin, field, others)
        mask[5] = crates > 0 and has_safe_bomb_escape(
            field, origin, bombs, others, explosions, BOMB_TIMER)
    # WAIT is the physical fallback when no useful safe action remains.
    mask[4] = not mask.any()
    return mask
