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


def coin_heaven_bfs_oc9(game_state):
    """Encode walkable moves and the BFS route to the nearest reachable coin.

    Feature layout:
        0-3: walkability of UP, RIGHT, DOWN, LEFT neighbour tiles.
        4-7: first BFS direction toward the nearest reachable coin.
        8: normalized BFS distance to that coin, or 1.0 when none is reachable.
    """
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    position = tuple(game_state["self"][3])
    walkability = np.array(
        [
            is_walkable(add_position(position, delta), field)
            for delta in MOVEMENTS
        ],
        dtype=np.float32,
    )

    direction, distance = bfs_first_step(
        field,
        position,
        game_state.get("coins", ()),
    )
    route = direction_and_distance_features(direction, distance, field.shape)
    return np.concatenate((walkability, route))

def classic_peace_minimal_oc8(game_state):
    """Encode minimal navigation, crate bombing.

    Feature layout:
        0-1: normalized dx and dy to the nearest visible coin.
        2-3: normalized dx and dy to the nearest crate.
        4: whether the agent can place a bomb.
        5-6: normalized dx and dy to the nearest active bomb.
        7: normalized timer of the nearest active bomb.
    """
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = game_state.get("bombs", ())
    occupied_by_bombs = bomb_positions(bombs)

    nearest_coin = nearest_coin_bfs(
        field,
        position,
        game_state.get("coins", ()),
    )
    nearest_crate = nearest_position(position, zip(*np.where(field == 1)))

    closest_bomb = nearest_bomb(position, bombs)
    bomb_relative = relative_position(
        position,
        tuple(closest_bomb[0]) if closest_bomb is not None else None,
        field.shape,
    )
    bomb_timer = np.array(
        [
            np.clip((closest_bomb[1] + 1) / (BOMB_TIMER + 1), 0.0, 1.0)
            if closest_bomb is not None
            else 0.0
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        (
            relative_position(position, nearest_coin, field.shape),
            relative_position(position, nearest_crate, field.shape),
            np.array(
                [can_place_bomb(position, bombs_left, occupied_by_bombs)],
                dtype=np.float32,
            ),
            bomb_relative,
            bomb_timer,
        )
    )


def classic_peace_improved_oc28(game_state):
    """Encode navigation, crate bombing, and safe bomb escapes without opponents.

    Feature layout:
        0-3: walkability of UP, RIGHT, DOWN, LEFT neighbour tiles.
        4: whether this agent can currently place a bomb.
        5-9: danger at current, UP, RIGHT, DOWN, LEFT tiles.
        10-13: first BFS direction toward nearest visible coin.
        14: normalized BFS distance to nearest visible coin, or 1.0.
        15-18: first BFS direction toward nearest reachable free tile next to a crate.
        19: normalized BFS distance to that crate-adjacent target, or 1.0.
        20-23: time-safe escape indicators for UP, RIGHT, DOWN, LEFT.
        24: normalized count of crates destroyable by a bomb at the current tile.
        25-26: normalized dx and dy to the nearest active bomb.
        27: normalized timer of the nearest active bomb, or 0.0 when absent.
    """
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = game_state.get("bombs", ())
    occupied_by_bombs = bomb_positions(bombs)
    explosion_map = game_state.get("explosion_map")

    features = np.zeros(28, dtype=np.float32)

    neighbours = [add_position(position, delta) for delta in MOVEMENTS]
    for index, tile in enumerate(neighbours):
        features[index] = is_walkable(tile, field, occupied_by_bombs)

    features[4] = can_place_bomb(position, bombs_left, occupied_by_bombs)

    danger_map = build_danger_map(field, bombs, explosion_map)
    for index, tile in enumerate((position, *neighbours), start=5):
        features[index] = danger_at(danger_map, tile)

    direction, distance = bfs_first_step(
        field,
        position,
        game_state.get("coins", ()),
        occupied_by_bombs,
    )
    features[10:15] = direction_and_distance_features(
        direction, distance, field.shape
    )

    targets = crate_adjacent_targets(field, occupied_by_bombs, ())
    direction, distance = bfs_first_step(
        field, position, targets, occupied_by_bombs
    )
    features[15:20] = direction_and_distance_features(
        direction, distance, field.shape
    )

    features[20:24] = time_safe_escape_directions(
        field,
        position,
        bombs_left,
        bombs,
        (),
        explosion_map,
    )

    destroyed_crates = sum(
        field[tile] == 1 for tile in blast_tiles(position, field)
    )
    features[24] = np.clip(
        destroyed_crates / (BOMB_POWER * len(MOVEMENTS)), 0.0, 1.0
    )

    closest_bomb = nearest_bomb(position, bombs)
    bomb_position = (
        tuple(closest_bomb[0]) if closest_bomb is not None else None
    )
    features[25:27] = relative_position(
        position, bomb_position, field.shape
    )
    features[27] = (
        np.clip((closest_bomb[1] + 1) / (BOMB_TIMER + 1), 0.0, 1.0)
        if closest_bomb is not None
        else 0.0
    )

    if not np.isfinite(features).all():
        raise RuntimeError("feature vector contains non-finite values")
    return features


def classic_peace_improved_oc33(game_state):
    """Encode compact Task 2 navigation, bombing, and escape information.

    Feature layout:
        0-3: walkability of UP, RIGHT, DOWN, LEFT neighbour tiles.
        4: whether this agent can currently place a bomb.
        5-9: danger at current, UP, RIGHT, DOWN, LEFT tiles.
        10-13: first BFS direction toward the nearest visible coin.
        14: normalized BFS distance to the nearest visible coin, or 1.0.
        15-18: first BFS direction toward the nearest crate-adjacent tile.
        19: normalized BFS distance to that crate-adjacent tile, or 1.0.
        20-23: time-safe escape indicators for UP, RIGHT, DOWN, LEFT.
        24: normalized count of crates destroyable from the current tile.
        25-26: normalized dx and dy to the nearest active bomb.
        27: normalized timer of the nearest active bomb, or 0.0 when absent.
        28-31: first BFS direction toward the most efficient reachable bombing tile.
        32: normalized BFS distance to that bombing tile, or 1.0.
    """
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = game_state.get("bombs", ())
    occupied_by_bombs = bomb_positions(bombs)
    explosion_map = game_state.get("explosion_map")
    features = np.zeros(33, dtype=np.float32)

    neighbours = [add_position(position, delta) for delta in MOVEMENTS]
    for index, tile in enumerate(neighbours):
        features[index] = is_walkable(tile, field, occupied_by_bombs)

    features[4] = can_place_bomb(position, bombs_left, occupied_by_bombs)

    danger_map = build_danger_map(field, bombs, explosion_map)
    for index, tile in enumerate((position, *neighbours), start=5):
        features[index] = danger_at(danger_map, tile)

    direction, distance = bfs_first_step(
        field,
        position,
        game_state.get("coins", ()),
        occupied_by_bombs,
    )
    features[10:15] = direction_and_distance_features(
        direction,
        distance,
        field.shape,
    )

    targets = crate_adjacent_targets(field, occupied_by_bombs, ())
    direction, distance = bfs_first_step(
        field,
        position,
        targets,
        occupied_by_bombs,
    )
    features[15:20] = direction_and_distance_features(
        direction,
        distance,
        field.shape,
    )

    features[20:24] = time_safe_escape_directions(
        field,
        position,
        bombs_left,
        bombs,
        (),
        explosion_map,
    )

    destroyed_crates, _ = bomb_effects_from(position, field, ())
    features[24] = np.clip(
        destroyed_crates / (BOMB_POWER * len(MOVEMENTS)),
        0.0,
        1.0,
    )

    closest_bomb = nearest_bomb(position, bombs)
    bomb_position = (
        tuple(closest_bomb[0]) if closest_bomb is not None else None
    )
    features[25:27] = relative_position(
        position,
        bomb_position,
        field.shape,
    )
    features[27] = (
        np.clip((closest_bomb[1] + 1) / (BOMB_TIMER + 1), 0.0, 1.0)
        if closest_bomb is not None
        else 0.0
    )

    direction, distance = efficient_crate_bombing_target_bfs(
        field,
        position,
        occupied_by_bombs,
    )
    features[28:33] = direction_and_distance_features(
        direction,
        distance,
        field.shape,
    )

    if not np.isfinite(features).all():
        raise RuntimeError("feature vector contains non-finite values")
    return features


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
    features[10:15] = direction_and_distance_features(
        direction, distance, field.shape
    )

    targets = crate_adjacent_targets(field, occupied_by_bombs, opponents)
    direction, distance = bfs_first_step(
        field, position, targets, occupied_by_bombs, opponents
    )
    features[15:20] = direction_and_distance_features(
        direction, distance, field.shape
    )

    targets = opponent_adjacent_targets(field, opponents, occupied_by_bombs)
    direction, distance = bfs_first_step(
        field, position, targets, occupied_by_bombs, opponents
    )
    features[20:25] = direction_and_distance_features(
        direction, distance, field.shape
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
