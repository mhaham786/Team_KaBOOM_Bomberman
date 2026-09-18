import numpy as np

import settings as s
from ..common.features import classic_peace_improved_oc33
from ..common.helpers import (
    MOVEMENTS,
    WAIT_DELTA,
    add_position,
    bomb_effects_from,
    build_time_hazard_model,
    has_safe_bomb_escape,
    is_legal_and_safe_at_time,
    is_safe_at_time,
    opponent_positions,
)


DELTAS = MOVEMENTS + (WAIT_DELTA, WAIT_DELTA)


def task3_features_oc33(game_state):
    return classic_peace_improved_oc33(game_state)


def safe_bomb_escape(game_state):
    return has_safe_bomb_escape(
        np.asarray(game_state["field"]),
        tuple(game_state["self"][3]),
        game_state.get("bombs", ()),
        opponent_positions(game_state.get("others", ())),
        game_state.get("explosion_map"),
        s.BOMB_TIMER,
    )


def prefer_safe_offensive_bomb(game_state, legal, ranked):
    if not legal[5]: 
        return False
    _, hit = bomb_effects_from(
        tuple(game_state["self"][3]),
        np.asarray(game_state["field"]),
        opponent_positions(game_state.get("others", ())),
    )
    return bool(hit and safe_bomb_escape(game_state))


def action_is_threatened(game_state, action):
    field = np.asarray(game_state["field"])
    origin = tuple(game_state["self"][3])
    hazard = build_time_hazard_model(
        field,
        game_state.get("bombs", ()),
        game_state.get("explosion_map"),
        origin if action == 5 else None,
        1,
    )
    successor = add_position(origin, DELTAS[action])
    return any(
        origin in hazard["danger_by_time"].get(time, set())
        or successor in hazard["danger_by_time"].get(time, set())
        for time in range(1, int(hazard["horizon"]) + 1)
    )


def safe_routes(game_state, legal):
    field = np.asarray(game_state["field"])
    origin = tuple(game_state["self"][3])
    opponents = opponent_positions(game_state.get("others", ()))
    routes = {}
    for action in np.flatnonzero(legal):
        action = int(action)
        hazard = build_time_hazard_model(
            field,
            game_state.get("bombs", ()),
            game_state.get("explosion_map"),
            origin if action == 5 else None,
            1,
        )
        horizon = max(1, int(hazard["horizon"]))
        first = add_position(origin, DELTAS[action])
        first_ok = (
            field[first] == 0
            and first not in opponents
            and is_safe_at_time(first, 1, hazard)
            if action == 5
            else is_legal_and_safe_at_time(first, 1, field, opponents, hazard)
        )
        if not first_ok:
            routes[action] = False
            continue
        layers = {1: {first}}
        edges = {}
        for time in range(1, horizon):
            next_time = time + 1
            next_layer = set()
            for position in layers.get(time, ()):
                choices = set()
                for delta in MOVEMENTS + (WAIT_DELTA,):
                    target = add_position(position, delta)
                    if is_legal_and_safe_at_time(
                        target, next_time, field, opponents, hazard
                    ):
                        choices.add(target)
                        next_layer.add(target)
                edges[position, time] = choices
            layers[next_time] = next_layer
        viable = {horizon: set(layers.get(horizon, ()))}
        for time in range(horizon - 1, 0, -1):
            viable[time] = {
                position
                for position in layers.get(time, ())
                if edges.get((position, time), set()) & viable[time + 1]
            }
        routes[action] = first in viable.get(1, set())
    return routes


def choose_safe_action(game_state, legal, ranked, selected):
    if not action_is_threatened(game_state, selected):
        return selected
    routes = safe_routes(game_state, legal)
    if routes.get(selected, False):
        return selected
    safe = {action for action, survives in routes.items() if survives}
    return next((int(action) for action in ranked if int(action) in safe), selected)
