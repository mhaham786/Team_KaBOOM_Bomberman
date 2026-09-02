from collections import deque

import numpy as np


MOVEMENTS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
BOMB_POWER = 3
BOMB_TIMER = 4
WAIT_DELTA = (0, 0)


def nearest_coin_bfs(field, start, coins):
    """Return the nearest coin reachable through free tiles, or None."""
    coin_positions = {tuple(coin) for coin in coins}
    if not coin_positions:
        return None

    queue = deque([start])
    visited = {start}

    while queue:
        position = queue.popleft()
        if position in coin_positions:
            return position

        for delta in MOVEMENTS:
            neighbour = add_position(position, delta)
            if (
                in_bounds(neighbour, field.shape)
                and neighbour not in visited
                and field[neighbour] == 0
            ):
                visited.add(neighbour)
                queue.append(neighbour)

    return None


def valid_action_mask(field, position, bombs_left, bombs, opponents):
    """Return legal actions in UP, RIGHT, DOWN, LEFT, WAIT, BOMB order."""
    mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    for index, delta in enumerate(MOVEMENTS):
        mask[index] = is_walkable(add_position(position, delta), field, bombs, opponents)
    mask[4] = True
    mask[5] = can_place_bomb(position, bombs_left, bombs)
    return mask


def has_safe_bomb_escape(field, position, bombs, opponents, explosion_map, max_steps):
    """Return whether placing a bomb leaves a time-safe escape path."""
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    directions = time_safe_escape_directions(
        field,
        position,
        True,
        bombs,
        opponents,
        explosion_map,
        force_hypothetical=True,
        min_horizon=max_steps,
    )
    return bool(directions.any())


def bomb_positions(bombs):
    """Return the occupied coordinates from bomb tuples."""
    return {tuple(position) for position, _ in bombs}


def exploding_opponent_bombs(bombs, own_bomb_positions):
    """Return opponent bomb positions that are about to explode."""
    return [
        tuple(position)
        for position, timer in bombs
        if tuple(position) not in own_bomb_positions and timer <= 0
    ]


def opponent_positions(others):
    """Return the occupied coordinates from opponent tuples."""
    return {tuple(position) for _, _, _, position in others}


def nearest_opponent_distance(position, opponents):
    """Return the Manhattan distance to the nearest opponent, or None."""
    if not opponents:
        return None
    return min(
        abs(position[0] - opponent[0]) + abs(position[1] - opponent[1])
        for opponent in opponents
    )


def add_position(position, delta):
    """Add a movement delta to a board position."""
    return position[0] + delta[0], position[1] + delta[1]


def in_bounds(position, shape):
    """Return whether a position is inside the board."""
    x, y = position
    return 0 <= x < shape[0] and 0 <= y < shape[1]


def is_walkable(position, field, bombs=(), opponents=()):
    """Return whether movement can enter a position."""
    return (
        in_bounds(position, field.shape)
        and field[position] == 0
        and position not in bombs
        and position not in opponents
    )


def can_place_bomb(position, bombs_left, bombs):
    """Return whether a bomb can be placed at a position."""
    return bool(bombs_left) and position not in bombs


def build_danger_map(field, bombs, explosion_map):
    """Return danger values from active explosions and predicted bomb blasts."""
    danger = np.zeros(field.shape, dtype=np.float32)
    if explosion_map is not None:
        danger[np.asarray(explosion_map) > 0] = 1.0

    for position, timer in bombs:
        try:
            score = 1.0 / (float(timer) + 1.0)
        except (TypeError, ValueError):
            score = 1.0
        score = np.float32(np.clip(score, 0.0, 1.0))
        for tile in blast_tiles(tuple(position), field):
            danger[tile] = max(danger[tile], score)
    return danger


def blast_tiles(origin, field):
    """Return the tiles reached by a bomb blast."""
    tiles = []
    if in_bounds(origin, field.shape):
        tiles.append(origin)

    for delta in MOVEMENTS:
        current = origin
        for _ in range(BOMB_POWER):
            current = add_position(current, delta)
            if not in_bounds(current, field.shape) or field[current] == -1:
                break
            tiles.append(current)
    return tiles


def danger_at(danger_map, position):
    """Return the danger at a position or one outside the board."""
    if not in_bounds(position, danger_map.shape):
        return np.float32(1.0)
    return np.float32(danger_map[position])


def time_safe_escape_directions(
    field,
    origin,
    bombs_left,
    bombs,
    opponents,
    explosion_map,
    force_hypothetical=False,
    min_horizon=0,
):
    """Return safe escape indicators for UP, RIGHT, DOWN, and LEFT."""
    occupied_by_bombs = bomb_positions(bombs)
    use_hypothetical = force_hypothetical or can_place_bomb(
        origin, bombs_left, occupied_by_bombs
    )
    if origin in occupied_by_bombs:
        use_hypothetical = False

    first_move_time = 2 if use_hypothetical else 1
    hazard = build_time_hazard_model(
        field,
        bombs,
        explosion_map,
        origin if use_hypothetical else None,
        max(min_horizon, first_move_time),
    )

    if use_hypothetical and (
        not is_safe_at_time(origin, 0, hazard)
        or not is_safe_at_time(origin, 1, hazard)
    ):
        return np.zeros(len(MOVEMENTS), dtype=np.float32)

    result = np.zeros(len(MOVEMENTS), dtype=np.float32)
    for index, delta in enumerate(MOVEMENTS):
        if has_time_safe_path_after_first_move(
            origin, delta, first_move_time, field, opponents, hazard
        ):
            result[index] = 1.0
    return result


def build_time_hazard_model(
    field, bombs, explosion_map, hypothetical_origin=None, min_horizon=0
):
    """Return timed explosion, bomb, and crate hazards."""
    danger_by_time = {}
    bomb_block_until = {}
    crate_destroy_time = {}
    horizon = max(0, int(min_horizon))

    if explosion_map is not None:
        explosion_map = np.asarray(explosion_map)
        for position in zip(*np.where(explosion_map > 0)):
            position = tuple(position)
            remaining = max(1, int(np.ceil(float(explosion_map[position]))))
            for time in range(remaining + 1):
                danger_by_time.setdefault(time, set()).add(position)
            horizon = max(horizon, remaining)

    timed_bombs = []
    for position, timer in bombs:
        try:
            explosion_time = int(timer) + 1
        except (TypeError, ValueError):
            explosion_time = 1
        timed_bombs.append((tuple(position), max(1, explosion_time)))

    if hypothetical_origin is not None:
        timed_bombs.append((tuple(hypothetical_origin), BOMB_TIMER + 1))

    for position, explosion_time in timed_bombs:
        bomb_block_until[position] = max(
            bomb_block_until.get(position, 0), explosion_time
        )
        tiles = blast_tiles(position, field)
        for time in (explosion_time, explosion_time + 1):
            danger_by_time.setdefault(time, set()).update(tiles)
        for tile in tiles:
            if field[tile] == 1:
                previous = crate_destroy_time.get(tile)
                if previous is None or explosion_time < previous:
                    crate_destroy_time[tile] = explosion_time
        horizon = max(horizon, explosion_time + 1)

    return {
        "danger_by_time": danger_by_time,
        "bomb_block_until": bomb_block_until,
        "crate_destroy_time": crate_destroy_time,
        "horizon": horizon,
    }


def has_time_safe_path_after_first_move(
    origin, delta, first_move_time, field, opponents, hazard
):
    """Return whether one forced first move leads to a safe path."""
    first_position = add_position(origin, delta)
    if not is_legal_and_safe_at_time(
        first_position, first_move_time, field, opponents, hazard
    ):
        return False

    horizon = max(first_move_time, int(hazard["horizon"]))
    if first_move_time >= horizon:
        return True

    frontier = deque([(first_position, first_move_time)])
    visited = {(first_position, first_move_time)}
    moves = MOVEMENTS + (WAIT_DELTA,)

    while frontier:
        position, time = frontier.popleft()
        next_time = time + 1
        if next_time > horizon:
            continue
        for next_delta in moves:
            next_position = add_position(position, next_delta)
            state = next_position, next_time
            if state in visited:
                continue
            if not is_legal_and_safe_at_time(
                next_position, next_time, field, opponents, hazard
            ):
                continue
            if next_time >= horizon:
                return True
            visited.add(state)
            frontier.append(state)
    return False


def is_legal_and_safe_at_time(position, time, field, opponents, hazard):
    """Return whether a position can be occupied at a given time."""
    if not in_bounds(position, field.shape):
        return False
    if field[position] == -1 or position in opponents:
        return False
    if field[position] == 1 and time <= hazard["crate_destroy_time"].get(
        position, float("inf")
    ):
        return False
    if time <= hazard["bomb_block_until"].get(position, -1):
        return False
    return is_safe_at_time(position, time, hazard)


def is_safe_at_time(position, time, hazard):
    """Return whether a position is safe from explosions at a given time."""
    return position not in hazard["danger_by_time"].get(time, set())


def bfs_first_step(field, start, targets, bombs=(), opponents=()):
    """Return the first direction and distance to the nearest reachable target."""
    targets = {
        tuple(target)
        for target in targets
        if tuple(target) == start
        or is_walkable(tuple(target), field, bombs, opponents)
    }
    if not targets or not in_bounds(start, field.shape):
        return None, None
    if start in targets:
        return None, 0

    queue = deque([(start, None, 0)])
    visited = {start}
    while queue:
        position, first_direction, distance = queue.popleft()
        for index, delta in enumerate(MOVEMENTS):
            neighbour = add_position(position, delta)
            if neighbour in visited or not is_walkable(
                neighbour, field, bombs, opponents
            ):
                continue
            direction = index if first_direction is None else first_direction
            if neighbour in targets:
                return direction, distance + 1
            visited.add(neighbour)
            queue.append((neighbour, direction, distance + 1))
    return None, None


def direction_and_distance_features(direction, distance, board_shape):
    """Encode a direction and normalized distance as five values."""
    features = np.zeros(5, dtype=np.float32)
    if direction is not None:
        features[direction] = 1.0
    features[4] = normalize_distance(distance, board_shape)
    return features


def normalize_distance(distance, board_shape):
    """Return a board-normalized distance between zero and one."""
    if distance is None:
        return np.float32(1.0)
    scale = max(1, board_shape[0] + board_shape[1] - 2)
    return np.float32(np.clip(distance / scale, 0.0, 1.0))


def crate_adjacent_targets(field, bombs, opponents):
    """Return free tiles adjacent to crates."""
    targets = set()
    for crate in zip(*np.where(field == 1)):
        for delta in MOVEMENTS:
            neighbour = add_position(tuple(crate), delta)
            if is_walkable(neighbour, field, bombs, opponents):
                targets.add(neighbour)
    return targets


def opponent_adjacent_targets(field, opponents, bombs):
    """Return free tiles adjacent to opponents."""
    targets = set()
    for opponent in opponents:
        for delta in MOVEMENTS:
            neighbour = add_position(opponent, delta)
            if is_walkable(neighbour, field, bombs, opponents):
                targets.add(neighbour)
    return targets


def bomb_effects_from(origin, field, opponents):
    """Return the crates and opponents hit by a bomb at an origin."""
    destroyed_crates = 0
    opponent_hit = origin in opponents
    for tile in blast_tiles(origin, field):
        if tile == origin:
            continue
        opponent_hit = opponent_hit or tile in opponents
        if field[tile] == 1:
            destroyed_crates += 1
    return destroyed_crates, opponent_hit


def safe_adjacent_tile_count(field, opponent, blast, bombs, other_opponents):
    """Return adjacent free tiles outside a newly placed bomb's blast."""
    return sum(
        neighbour not in blast
        and is_walkable(neighbour, field, bombs, other_opponents)
        for neighbour in (
            add_position(opponent, movement) for movement in MOVEMENTS
        )
    )
