import numpy as np

from .helpers import *


def coin_heaven_minimal_oc4(game_state):
    """Encode player and nearest reachable coin as [x, y, coin_x, coin_y]"""
    if game_state is None:
        return None

    _, _, _, player_position = game_state["self"]
    player_x, player_y = player_position
    coin_position = nearest_coin_bfs(game_state["field"], player_position, game_state["coins"])

    coin_x, coin_y = coin_position if coin_position is not None else (-1, -1)
    return [float(player_x), float(player_y), float(coin_x), float(coin_y)]


def advanced_features_oc31(game_state):
    """Convert a game state into the advanced 31-value feature vector.

    Feature layout:
        0-3: walkability of UP, RIGHT, DOWN, LEFT neighbour tiles.
        4: whether this agent can currently place a bomb.
        5-9: danger at current, UP, RIGHT, DOWN, LEFT tiles.
        10-13: first BFS direction toward nearest visible coin.
        14: normalized BFS distance to nearest visible coin, or 1.0.
        15-18: first BFS direction toward nearest reachable free tile next to a crate.
        19: normalized BFS distance to that crate-adjacent target, or 1.0.
        20-23: first BFS direction toward a free tile next to the nearest opponent.
        24: normalized BFS distance to that opponent-adjacent target, or 1.0.
        25-28: time-safe escape indicators for first movement UP, RIGHT,
            DOWN, LEFT. When BOMB is currently legal, these are evaluated after
            adding a hypothetical bomb at the agent's current tile; the agent
            stays at the origin through t=1 and the first escape movement is at
            t=2. Otherwise they are evaluated using actual hazards only and the
            first movement is at t=1.
        29: normalized count of crates destroyable by a bomb at the current tile.
        30: whether an opponent is in the current tile's potential bomb blast line.

    Bomb blasts use range three, stop only at stone walls, include crates, and
    continue through crates.
    """
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = game_state.get("bombs", ())
    occupied_by_bombs = bomb_positions(bombs)
    opponents = opponent_positions(game_state.get("others", ()))
    explosion_map = game_state.get("explosion_map")

    features = np.zeros(31, dtype=np.float32)

    neighbours = [add_position(position, delta) for delta in MOVEMENTS]
    for index, tile in enumerate(neighbours):
        features[index] = is_walkable(tile, field, occupied_by_bombs, opponents)

    features[4] = can_place_bomb(position, bombs_left, occupied_by_bombs)

    danger_map = build_danger_map(field, bombs, explosion_map)
    for index, tile in enumerate((position, *neighbours), start=5):
        features[index] = danger_at(danger_map, tile)

    direction, distance = bfs_first_step(
        field,
        position,
        game_state.get("coins", ()),
        occupied_by_bombs,
        opponents,
    )
    write_direction_and_distance(
        features, 10, 14, direction, distance, field.shape
    )

    targets = crate_adjacent_targets(field, occupied_by_bombs, opponents)
    direction, distance = bfs_first_step(
        field, position, targets, occupied_by_bombs, opponents
    )
    write_direction_and_distance(
        features, 15, 19, direction, distance, field.shape
    )

    targets = opponent_adjacent_targets(field, opponents, occupied_by_bombs)
    direction, distance = bfs_first_step(
        field, position, targets, occupied_by_bombs, opponents
    )
    write_direction_and_distance(
        features, 20, 24, direction, distance, field.shape
    )

    features[25:29] = time_safe_escape_directions(
        field,
        position,
        bombs_left,
        bombs,
        opponents,
        explosion_map,
    )

    destroyed_crates, opponent_in_blast = bomb_effects_from(
        position, field, opponents
    )
    features[29] = np.clip(
        destroyed_crates / (BOMB_POWER * len(MOVEMENTS)), 0.0, 1.0
    )
    features[30] = opponent_in_blast

    if not np.isfinite(features).all():
        raise RuntimeError("feature vector contains non-finite values")
    return features
