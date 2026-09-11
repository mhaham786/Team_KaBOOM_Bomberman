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


def sarsa_task2_features(game_state):
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = game_state.get("bombs", ())
    explosion_map = game_state.get("explosion_map")
    occupied_by_bombs = bomb_positions(bombs)
    opponents = []
    
    neighbours = [(position[0], position[1]-1), (position[0]+1, position[1]), 
                  (position[0], position[1]+1), (position[0]-1, position[1])]

    danger_map = build_danger_map(field, bombs, explosion_map)
    current_danger = 1 if danger_at(danger_map, position) > 0 else 0

    escapes = [0, 0, 0, 0]
    valid_moves = [0, 0, 0, 0]
    target_route = [0, 0, 0, 0]
    can_bomb = 0
    good_bomb_spot = 0

    if current_danger:
        escapes = [int(e) for e in time_safe_escape_directions(field, position, bombs_left, bombs, opponents, explosion_map)]
    else:
        for i, tile in enumerate(neighbours):
            is_walk = is_walkable(tile, field, occupied_by_bombs, opponents)
            is_safe = danger_at(danger_map, tile) == 0
            valid_moves[i] = 1 if (is_walk and is_safe) else 0

        coin_dir, _ = bfs_first_step(field, position, game_state.get("coins", ()), occupied_by_bombs, opponents)
        if coin_dir is not None:
            target_route = list(direction_and_distance_features(coin_dir, 0, field.shape)[:4])
        else:
            crate_targets = crate_adjacent_targets(field, occupied_by_bombs, opponents)
            crate_dir, _ = bfs_first_step(field, position, crate_targets, occupied_by_bombs, opponents)
            if crate_dir is not None:
                target_route = list(direction_and_distance_features(crate_dir, 0, field.shape)[:4])

        can_bomb = 1 if can_place_bomb(position, bombs_left, occupied_by_bombs) else 0
        destroyed_crates, _ = bomb_effects_from(position, field, opponents)
        good_bomb_spot = 1 if destroyed_crates > 0 else 0

    state_list = [current_danger] + escapes + valid_moves + target_route + [can_bomb, good_bomb_spot]
    return tuple(int(x) for x in state_list)


def sarsa_task3_features_oc18(game_state):
    """Extend the 15-value SARSA state with compact hunting information.

    Feature layout:
        0-14: sarsa_task2_features.
        15: BFS opponent direction as 0=none or 1-4=UP, RIGHT, DOWN, LEFT.
        16: whether the nearest reachable opponent is within six steps.
        17: whether a bomb placed here would hit an opponent.
    """
    if game_state is None:
        return None

    features = list(sarsa_task2_features(game_state))
    field = np.asarray(game_state["field"])
    position = tuple(game_state["self"][3])
    bombs = bomb_positions(game_state.get("bombs", ()))
    opponents = opponent_positions(game_state.get("others", ()))

    targets = opponent_adjacent_targets(field, opponents, bombs)
    direction, distance = bfs_first_step(
        field, position, targets, bombs, opponents
    )
    opponent_direction = 0 if direction is None else direction + 1
    opponent_distance = None if distance is None else distance + 1
    opponent_nearby = (
        opponent_distance is not None
        and opponent_distance <= 2 * BOMB_POWER
    )
    opponent_in_blast = bool(opponents & set(blast_tiles(position, field)))

    features.extend(
        (
            opponent_direction,
            int(opponent_nearby),
            int(opponent_in_blast),
        )
    )
    return tuple(features)


def sarsa_task3_features_oc10(game_state):
    """Encode compact navigation, bomb safety, and nearby combat guidance.

    Feature layout:
        0: whether the current position is dangerous.
        1-4: safe movement in UP, RIGHT, DOWN, LEFT.
        5-8: route toward a nearby opponent, coin, or crate.
        9: whether placing a bomb here is useful and leaves a safe escape.
    """
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = game_state.get("bombs", ())
    occupied_by_bombs = bomb_positions(bombs)
    opponents = opponent_positions(game_state.get("others", ()))

    danger_map = build_danger_map(
        field,
        bombs,
        game_state.get("explosion_map"),
    )
    current_danger = int(danger_at(danger_map, position) > 0)

    if current_danger:
        safe_moves = time_safe_escape_directions(
            field,
            position,
            bombs_left,
            bombs,
            opponents,
            game_state.get("explosion_map"),
        )
        return tuple([current_danger, *map(int, safe_moves), 0, 0, 0, 0, 0])

    neighbours = [add_position(position, movement) for movement in MOVEMENTS]
    safe_moves = [
        int(
            is_walkable(tile, field, occupied_by_bombs, opponents)
            and danger_at(danger_map, tile) == 0
        )
        for tile in neighbours
    ]

    opponent_targets = opponent_adjacent_targets(
        field, opponents, occupied_by_bombs
    )
    opponent_direction, opponent_distance = bfs_first_step(
        field, position, opponent_targets, occupied_by_bombs, opponents
    )
    coin_direction, coin_distance = bfs_first_step(
        field,
        position,
        game_state.get("coins", ()),
        occupied_by_bombs,
        opponents,
    )
    crate_targets = crate_adjacent_targets(
        field, occupied_by_bombs, opponents
    )
    crate_direction, crate_distance = bfs_first_step(
        field, position, crate_targets, occupied_by_bombs, opponents
    )

    if opponent_distance is not None and opponent_distance <= 3 * BOMB_POWER:
        target_direction = opponent_direction
    elif coin_distance is not None:
        target_direction = coin_direction
    elif crate_distance is not None:
        target_direction = crate_direction
    else:
        target_direction = opponent_direction
    target_route = direction_and_distance_features(
        target_direction, 0, field.shape
    )[:4]

    destroyed_crates, opponent_hit = bomb_effects_from(
        position, field, opponents
    )
    useful_bomb = bool(destroyed_crates or opponent_hit)
    good_bomb_spot = (
        can_place_bomb(position, bombs_left, occupied_by_bombs)
        and useful_bomb
        and has_safe_bomb_escape(
            field,
            position,
            bombs,
            opponents,
            game_state.get("explosion_map"),
            BOMB_TIMER,
        )
    )
    return tuple(
        [
            current_danger,
            *safe_moves,
            *map(int, target_route),
            int(good_bomb_spot),
        ]
    )


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
