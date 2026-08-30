from collections import deque
from typing import Iterable, Optional

import numpy as np


FEATURE_DIM = 31
FEATURE_SCHEMA = "uttam_dqn_v2_time_escape_31"
ARTIFACT_VERSION = 2

MOVEMENTS = (
    ("UP", (0, -1)),
    ("RIGHT", (1, 0)),
    ("DOWN", (0, 1)),
    ("LEFT", (-1, 0)),
)
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2
WAIT_DELTA = (0, 0)


def state_to_features(game_state):
    """Convert a BombeRLe game state into the fixed 31-feature baseline.

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

    V2 engine-corrected blast convention:
        Bomb blasts use range three, stop only at stone walls, include crates,
        and continue through crates. This affects danger features 5-9, escape
        features 25-28, destroyable-crate feature 29 and opponent-in-blast
        feature 30.

    Returns None for the terminal sentinel state. Otherwise returns a finite
    np.float32 array with shape (31,). The input game_state arrays are read-only
    from this function's point of view and are not modified.
    """
    if game_state is None:
        return None

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = _bomb_positions(game_state)
    opponents = _opponent_positions(game_state)

    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    neighbour_positions = [_add(position, delta) for _, delta in MOVEMENTS]
    for i, tile in enumerate(neighbour_positions):
        features[i] = float(_is_walkable(tile, field, bombs, opponents))

    features[4] = float(_can_place_bomb(position, bombs_left, bombs))

    danger_map = _build_danger_map(field, game_state.get("bombs", ()), game_state.get("explosion_map"))
    danger_tiles = (position, *neighbour_positions)
    for offset, tile in enumerate(danger_tiles, start=5):
        features[offset] = _danger_at(danger_map, tile)

    coin_direction, coin_distance = _bfs_first_step(
        field,
        position,
        game_state.get("coins", ()),
        bombs,
        opponents,
    )
    _write_direction_and_distance(features, 10, 14, coin_direction, coin_distance, field.shape)

    crate_targets = _crate_adjacent_targets(field, bombs, opponents)
    crate_direction, crate_distance = _bfs_first_step(
        field,
        position,
        crate_targets,
        bombs,
        opponents,
    )
    _write_direction_and_distance(features, 15, 19, crate_direction, crate_distance, field.shape)

    opponent_targets = _opponent_adjacent_targets(field, opponents, bombs)
    opponent_direction, opponent_distance = _bfs_first_step(
        field,
        position,
        opponent_targets,
        bombs,
        opponents,
    )
    _write_direction_and_distance(features, 20, 24, opponent_direction, opponent_distance, field.shape)

    features[25:29] = _time_safe_escape_directions(game_state)

    destroyed_crates, opponent_in_blast = _bomb_effects_from(position, field, opponents)
    features[29] = np.float32(np.clip(destroyed_crates / float(BOMB_POWER * len(MOVEMENTS)), 0.0, 1.0))
    features[30] = float(opponent_in_blast)

    if features.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature vector must have shape ({FEATURE_DIM},)")
    if not np.isfinite(features).all():
        raise RuntimeError("feature vector contains non-finite values")
    return features.astype(np.float32, copy=False)


def valid_action_mask(game_state):
    """Return legal actions as a Boolean mask in UP, RIGHT, DOWN, LEFT, WAIT, BOMB order.

    Movement actions require a walkable destination: field value 0 and no bomb or
    opponent on the destination tile. WAIT is always valid. BOMB is valid only
    when the self tuple reports an available bomb and the current tile does not
    already contain a bomb. Danger is deliberately ignored because danger is a
    learned feature, not a hard legality constraint.
    """
    if game_state is None:
        return np.zeros(len(ACTIONS), dtype=np.bool_)

    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    position = tuple(position)
    bombs = _bomb_positions(game_state)
    opponents = _opponent_positions(game_state)

    mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    for i, (_, delta) in enumerate(MOVEMENTS):
        mask[i] = _is_walkable(_add(position, delta), field, bombs, opponents)
    mask[4] = True
    mask[5] = _can_place_bomb(position, bombs_left, bombs)

    if not mask.any():
        raise RuntimeError("non-terminal game state must have at least one valid action")
    return mask


def has_safe_bomb_escape(game_state, max_steps):
    """Return whether a bomb placed now leaves a time-safe escape path.

    This public helper keeps the V1 training API but now uses the V2
    time-expanded planner. It models a successful BOMB action at t=1, validates
    that the origin remains safe through t=1, starts candidate escape movements
    at t=2, and requires a legal safe path through the full hazard horizon.
    Opponents are treated as static blockers throughout the horizon, a
    conservative approximation because their future movement is unknown.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if game_state is None:
        raise ValueError("game_state must not be None")
    return bool(_time_safe_escape_directions(game_state, force_hypothetical=True, min_horizon=max_steps).any())


def _bomb_positions(game_state) -> set[tuple[int, int]]:
    """Extract occupied bomb coordinates from the game state's bomb tuples."""
    return {tuple(position) for position, _ in game_state.get("bombs", ())}


def _opponent_positions(game_state) -> set[tuple[int, int]]:
    """Extract occupied opponent coordinates from the game state's others tuples."""
    return {tuple(position) for _, _, _, position in game_state.get("others", ())}


def _add(position: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    """Add a movement delta to an (x, y) board position."""
    return position[0] + delta[0], position[1] + delta[1]


def _in_bounds(position: tuple[int, int], shape: tuple[int, int]) -> bool:
    """Check whether a board position is inside an array with shape (cols, rows)."""
    x, y = position
    return 0 <= x < shape[0] and 0 <= y < shape[1]


def _is_walkable(
    position: tuple[int, int],
    field: np.ndarray,
    bombs: set[tuple[int, int]],
    opponents: set[tuple[int, int]],
) -> bool:
    """Return whether movement can enter position, ignoring danger scores."""
    return _in_bounds(position, field.shape) and field[position] == 0 and position not in bombs and position not in opponents


def _can_place_bomb(position: tuple[int, int], bombs_left, bombs: set[tuple[int, int]]) -> bool:
    """Return whether BOMB is legal at the current position."""
    return bool(bombs_left) and position not in bombs


def _build_danger_map(field: np.ndarray, bombs: Iterable, explosion_map: Optional[np.ndarray]) -> np.ndarray:
    """Build per-tile danger from active explosions and predicted bomb blasts.

    Active explosion tiles receive danger 1.0. Each bomb contributes
    1 / (timer + 1), clipped to [0, 1], along the bomb tile and four blast rays.
    The V2 blast helper matches the engine: stone walls stop the ray, while
    crates are included and do not block propagation.
    """
    danger = np.zeros(field.shape, dtype=np.float32)

    if explosion_map is not None:
        explosion_array = np.asarray(explosion_map)
        danger[explosion_array > 0] = 1.0

    for bomb_position, timer in bombs:
        bomb_position = tuple(bomb_position)
        try:
            score = 1.0 / (float(timer) + 1.0)
        except (TypeError, ValueError):
            score = 1.0
        score = np.float32(np.clip(score, 0.0, 1.0))
        for tile in _blast_tiles(bomb_position, field):
            danger[tile] = max(danger[tile], score)

    return danger


def _blast_tiles(origin: tuple[int, int], field: np.ndarray) -> list[tuple[int, int]]:
    """Return engine-faithful bomb blast tiles.

    The engine's Bomb.get_blast_coords() uses range three, includes the origin,
    stops only at stone walls, includes crates, and continues through crates.
    There are no bomb chain reactions in the engine, so this helper does not
    model any timer acceleration.
    """
    tiles = []
    if _in_bounds(origin, field.shape):
        tiles.append(origin)
    for _, delta in MOVEMENTS:
        current = origin
        for _ in range(BOMB_POWER):
            current = _add(current, delta)
            if not _in_bounds(current, field.shape):
                break
            if field[current] == -1:
                break
            tiles.append(current)
    return tiles


def _danger_at(danger_map: np.ndarray, position: tuple[int, int]) -> np.float32:
    """Return danger at position, or 1.0 outside the board as a conservative value."""
    if not _in_bounds(position, danger_map.shape):
        return np.float32(1.0)
    return np.float32(danger_map[position])


def _time_safe_escape_directions(
    game_state,
    *,
    force_hypothetical: bool = False,
    min_horizon: int = 0,
) -> np.ndarray:
    """Return V2 safe-escape indicators for first moves UP, RIGHT, DOWN, LEFT.

    If BOMB is legal, the default feature semantics add a hypothetical bomb at
    the agent's current tile. The BOMB action happens during t=1, the agent is
    still at the origin through t=1, candidate first escape moves occur at t=2,
    and that hypothetical bomb explodes at t=5. If BOMB is not legal, actual
    hazards only are modeled and candidate first movements happen at t=1.

    A direction is true only when the first move is physically legal and BFS can
    keep moving or waiting through the full hazard horizon. The path may move at
    every step; no single tile has to stay safe for the whole horizon.
    """
    field = np.asarray(game_state["field"])
    _, _, bombs_left, position = game_state["self"]
    origin = tuple(position)
    bomb_positions = _bomb_positions(game_state)
    opponents = _opponent_positions(game_state)
    use_hypothetical = force_hypothetical or _can_place_bomb(origin, bombs_left, bomb_positions)
    if origin in bomb_positions:
        use_hypothetical = False

    first_move_time = 2 if use_hypothetical else 1
    hazard = _build_time_hazard_model(
        field,
        game_state.get("bombs", ()),
        game_state.get("explosion_map"),
        hypothetical_origin=origin if use_hypothetical else None,
        min_horizon=max(min_horizon, first_move_time),
    )

    if use_hypothetical and (not _is_safe_at_time(origin, 0, hazard) or not _is_safe_at_time(origin, 1, hazard)):
        return np.zeros(len(MOVEMENTS), dtype=np.float32)

    result = np.zeros(len(MOVEMENTS), dtype=np.float32)
    for direction_index, (_, delta) in enumerate(MOVEMENTS):
        if _has_time_safe_path_after_first_move(
            origin,
            direction_index,
            delta,
            first_move_time,
            field,
            opponents,
            hazard,
        ):
            result[direction_index] = 1.0
    return result


def _build_time_hazard_model(
    field: np.ndarray,
    bombs: Iterable,
    explosion_map: Optional[np.ndarray],
    *,
    hypothetical_origin: Optional[tuple[int, int]] = None,
    min_horizon: int = 0,
) -> dict:
    """Build timed bomb blockers, explosion danger, and crate destruction times.

    State features are computed at t=0 before this agent acts. For an actual
    bomb observed with timer T, the engine checks timer <= 0 before decrementing,
    so it explodes at planner time T + 1. A hypothetical bomb placed by choosing
    BOMB is created during t=1 with timer four, is decremented to three in that
    same step, and explodes at t=5. Existing explosion_map values model future
    danger after the next agent movement and update_explosions() call, not the
    engine's pre-update internal Explosion.stage. Therefore map value 1 remains
    dangerous for the next evaluation, while map value 0 has no future dangerous
    tick because it becomes smoke before evaluate_explosions(). Crates are
    cleared during a bomb's explosion update, so they become walkable for
    subsequent movement steps only.
    """
    danger_by_time: dict[int, set[tuple[int, int]]] = {}
    bomb_block_until: dict[tuple[int, int], int] = {}
    crate_destroy_time: dict[tuple[int, int], int] = {}
    horizon = max(0, int(min_horizon))

    if explosion_map is not None:
        explosion_array = np.asarray(explosion_map)
        for position in zip(*np.where(explosion_array > 0)):
            position = tuple(position)
            remaining = max(1, int(np.ceil(float(explosion_array[position]))))
            for time in range(0, remaining + 1):
                danger_by_time.setdefault(time, set()).add(position)
            horizon = max(horizon, remaining)

    timed_bombs = []
    for bomb_position, timer in bombs:
        position = tuple(bomb_position)
        try:
            explosion_time = int(timer) + 1
        except (TypeError, ValueError):
            explosion_time = 1
        explosion_time = max(1, explosion_time)
        timed_bombs.append((position, explosion_time))

    if hypothetical_origin is not None:
        timed_bombs.append((tuple(hypothetical_origin), BOMB_TIMER + 1))

    for position, explosion_time in timed_bombs:
        bomb_block_until[position] = max(bomb_block_until.get(position, 0), explosion_time)
        blast_tiles = _blast_tiles(position, field)
        for time in (explosion_time, explosion_time + 1):
            danger_by_time.setdefault(time, set()).update(blast_tiles)
        for tile in blast_tiles:
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


def _has_time_safe_path_after_first_move(
    origin: tuple[int, int],
    direction_index: int,
    delta: tuple[int, int],
    first_move_time: int,
    field: np.ndarray,
    opponents: set[tuple[int, int]],
    hazard: dict,
) -> bool:
    """Run deterministic time-expanded BFS after forcing one first movement."""
    del direction_index
    first_position = _add(origin, delta)
    if not _is_legal_and_safe_at_time(first_position, first_move_time, field, opponents, hazard):
        return False

    horizon = max(first_move_time, int(hazard["horizon"]))
    frontier = deque([(first_position, first_move_time)])
    visited = {(first_position, first_move_time)}
    if first_move_time >= horizon:
        return True

    timed_moves = tuple(delta for _, delta in MOVEMENTS) + (WAIT_DELTA,)
    while frontier:
        position, time = frontier.popleft()
        next_time = time + 1
        if next_time > horizon:
            continue
        for next_delta in timed_moves:
            next_position = _add(position, next_delta)
            state = (next_position, next_time)
            if state in visited:
                continue
            if not _is_legal_and_safe_at_time(next_position, next_time, field, opponents, hazard):
                continue
            if next_time >= horizon:
                return True
            visited.add(state)
            frontier.append(state)

    return False


def _is_legal_and_safe_at_time(
    position: tuple[int, int],
    time: int,
    field: np.ndarray,
    opponents: set[tuple[int, int]],
    hazard: dict,
) -> bool:
    """Return whether the position can be occupied at a planner time."""
    if not _in_bounds(position, field.shape):
        return False
    if field[position] == -1:
        return False
    if position in opponents:
        return False
    if field[position] == 1 and time <= hazard["crate_destroy_time"].get(position, float("inf")):
        return False
    if time <= hazard["bomb_block_until"].get(position, -1):
        return False
    return _is_safe_at_time(position, time, hazard)


def _is_safe_at_time(position: tuple[int, int], time: int, hazard: dict) -> bool:
    """Return whether no modeled explosion is dangerous at position and time."""
    return position not in hazard["danger_by_time"].get(time, set())


def _bfs_first_step(
    field: np.ndarray,
    start: tuple[int, int],
    targets: Iterable[tuple[int, int]],
    bombs: set[tuple[int, int]],
    opponents: set[tuple[int, int]],
) -> tuple[Optional[int], Optional[int]]:
    """Find the first deterministic movement index and distance to the nearest target.

    BFS expands neighbours in fixed UP, RIGHT, DOWN, LEFT order. Walls, crates,
    bombs and opponents block movement. Returns (None, None) when no target is
    reachable. The first component is an index into MOVEMENTS.
    """
    target_set = {tuple(target) for target in targets}
    target_set = {target for target in target_set if _is_walkable(target, field, bombs, opponents) or target == start}
    if not target_set or not _in_bounds(start, field.shape):
        return None, None
    if start in target_set:
        return None, 0

    frontier = deque([(start, None, 0)])
    visited = {start}

    while frontier:
        position, first_direction, distance = frontier.popleft()
        for direction_index, (_, delta) in enumerate(MOVEMENTS):
            next_position = _add(position, delta)
            if next_position in visited:
                continue
            if not _is_walkable(next_position, field, bombs, opponents):
                continue
            next_first_direction = direction_index if first_direction is None else first_direction
            next_distance = distance + 1
            if next_position in target_set:
                return next_first_direction, next_distance
            visited.add(next_position)
            frontier.append((next_position, next_first_direction, next_distance))

    return None, None


def _write_direction_and_distance(
    features: np.ndarray,
    direction_start: int,
    distance_index: int,
    direction: Optional[int],
    distance: Optional[int],
    board_shape: tuple[int, int],
) -> None:
    """Write a four-way one-hot direction and normalized clipped BFS distance."""
    if direction is not None:
        features[direction_start + direction] = 1.0
    features[distance_index] = _normalize_distance(distance, board_shape)


def _normalize_distance(distance: Optional[int], board_shape: tuple[int, int]) -> np.float32:
    """Normalize BFS distance by board dimensions and clip to [0, 1]."""
    if distance is None:
        return np.float32(1.0)
    scale = max(1, int(board_shape[0]) + int(board_shape[1]) - 2)
    return np.float32(np.clip(float(distance) / float(scale), 0.0, 1.0))


def _crate_adjacent_targets(
    field: np.ndarray,
    bombs: set[tuple[int, int]],
    opponents: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return reachable-candidate free tiles adjacent to any crate."""
    targets = set()
    crate_positions = zip(*np.where(field == 1))
    for crate in crate_positions:
        crate = tuple(crate)
        for _, delta in MOVEMENTS:
            neighbour = _add(crate, delta)
            if _is_walkable(neighbour, field, bombs, opponents):
                targets.add(neighbour)
    return targets


def _opponent_adjacent_targets(
    field: np.ndarray,
    opponents: set[tuple[int, int]],
    bombs: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return free tiles adjacent to all opponents.

    The deterministic BFS later chooses the nearest reachable target. This keeps
    an unreachable nearby opponent from hiding a farther reachable opponent.
    """
    targets = set()
    for opponent in opponents:
        for _, delta in MOVEMENTS:
            neighbour = _add(opponent, delta)
            if _is_walkable(neighbour, field, bombs, opponents):
                targets.add(neighbour)
    return targets


def _bomb_effects_from(
    origin: tuple[int, int],
    field: np.ndarray,
    opponents: set[tuple[int, int]],
) -> tuple[int, bool]:
    """Count crates and detect opponents hit by a hypothetical bomb at origin."""
    destroyed_crates = 0
    opponent_hit = origin in opponents
    for tile in _blast_tiles(origin, field):
        if tile == origin:
            continue
        if tile in opponents:
            opponent_hit = True
        if field[tile] == 1:
            destroyed_crates += 1
    return destroyed_crates, opponent_hit
