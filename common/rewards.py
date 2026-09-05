import events as e
import numpy as np
import settings as s

from .helpers import (
    bfs_first_step,
    bomb_effects_from,
    bomb_positions,
    build_danger_map,
    danger_at,
    efficient_crate_bombing_target_bfs,
    has_safe_bomb_escape,
    normalize_distance,
    opponent_adjacent_targets,
    opponent_positions,
)


def coin_heaven_rewards_sarsa(events):
    rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_SELF: -500,
        e.WAITED: -1,
        e.INVALID_ACTION: -5,
    }
    return sum(rewards.get(event, 0) for event in events)


def coin_heaven_rewards_dqn(events):
    reward = -0.01
    if e.COIN_COLLECTED in events:
        reward += 10.0
    if e.CRATE_DESTROYED in events:
        reward += 2.0
    if e.COIN_FOUND in events:
        reward += 1.0
    if e.KILLED_OPPONENT in events:
        reward += 50.0
    if e.SURVIVED_ROUND in events:
        reward += 5.0
    if e.INVALID_ACTION in events:
        reward += -5.0
    if e.KILLED_SELF in events:
        reward += -50.0
    if e.GOT_KILLED in events:
        reward += -40.0
    if e.WAITED in events:
        reward += -0.1
    if e.BOMB_DROPPED in events:
        reward += -0.1

    return reward

def classic_peace_rewards_dqn_baseline(events):
    reward = -0.01
    if e.COIN_COLLECTED in events:
        reward += 5.0
    if e.CRATE_DESTROYED in events:
        reward += 2.0
    if e.COIN_FOUND in events:
        reward += 1.0
    if e.INVALID_ACTION in events:
        reward += -0.5
    if e.WAITED in events:
        reward += -0.05
    if e.BOMB_DROPPED in events:
        reward += -0.1
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward += -5.0
    if e.SURVIVED_ROUND in events:
        reward += 1.0

    return reward

def classic_peace_rewards_dqn_baseline(events):
    reward = -0.01
    reward += 5.0 * events.count(e.COIN_COLLECTED)
    reward += 1.0 * events.count(e.CRATE_DESTROYED)
    reward += 1.0 * events.count(e.COIN_FOUND)
    reward -= 0.5 * events.count(e.INVALID_ACTION)
    reward -= 0.05 * events.count(e.WAITED)
    killed = e.KILLED_SELF in events or e.GOT_KILLED in events
    if killed:
        reward -= 5.0
    return reward

def classic_peace_rewards_dqn_improved(events):
    reward = -0.05
    reward += 10.0 * events.count(e.COIN_COLLECTED)
    reward += 1.0 * events.count(e.CRATE_DESTROYED)
    reward += 2.0 * events.count(e.COIN_FOUND)
    reward -= 0.5 * events.count(e.INVALID_ACTION)
    reward -= 0.05 * events.count(e.WAITED)
    killed = e.KILLED_SELF in events or e.GOT_KILLED in events
    if killed:
        reward -= 20.0
    return reward


def coin_heaven_rewards_ppo_baseline(events):
    reward = -0.001
    if e.COIN_COLLECTED in events:
        reward += 2.0
    if e.INVALID_ACTION in events:
        reward += -0.02
    if e.WAITED in events:
        reward += -0.01

    return reward


def coin_heaven_rewards_ppo_improved(events):
    reward = -0.5
    if e.COIN_COLLECTED in events:
        reward += 2.0
    if e.INVALID_ACTION in events:
        reward += -0.5
    if e.WAITED in events:
        reward += -0.01

    return reward


def advanced_reward_shaping_dqn(
    action,
    old_game_state,
    new_game_state,
    events,
    gamma,
):
    """Reward safer progress toward targets and penalize useless or unsafe bombs."""

    def state_potential(game_state):
        if game_state is None:
            return 0.0

        field = np.asarray(game_state["field"])
        position = tuple(game_state["self"][3])
        bombs = game_state.get("bombs", ())
        occupied_by_bombs = bomb_positions(bombs)
        opponents = opponent_positions(game_state.get("others", ()))

        danger_map = build_danger_map(
            field,
            bombs,
            game_state.get("explosion_map"),
        )
        potential = -6.0 * float(danger_at(danger_map, position))

        _, coin_distance = bfs_first_step(
            field,
            position,
            game_state.get("coins", ()),
            occupied_by_bombs,
            opponents,
        )
        _, crate_distance = efficient_crate_bombing_target_bfs(
            field,
            position,
            occupied_by_bombs,
        )
        opponent_targets = opponent_adjacent_targets(
            field,
            opponents,
            occupied_by_bombs,
        )
        _, opponent_distance = bfs_first_step(
            field,
            position,
            opponent_targets,
            occupied_by_bombs,
            opponents,
        )

        if coin_distance is not None:
            potential -= 8.0 * float(
                normalize_distance(coin_distance, field.shape)
            )
        elif crate_distance is not None:
            potential -= 3.0 * float(
                normalize_distance(crate_distance, field.shape)
            )

        if opponent_distance is not None:
            potential -= float(
                normalize_distance(opponent_distance, field.shape)
            )
        return potential

    shaping = gamma * state_potential(new_game_state)
    shaping -= state_potential(old_game_state)
    shaping = float(np.clip(shaping, -2.0, 2.0))

    old_score = old_game_state["self"][1]
    new_score = (
        new_game_state["self"][1]
        if new_game_state is not None
        else old_score + events.count(e.COIN_COLLECTED)
    )
    ninth_coin_collected = (
        e.COIN_COLLECTED in events
        and old_score < 9 <= new_score
        and not old_game_state.get("others")
    )
    if ninth_coin_collected:
        step = old_game_state["step"]
        speed_bonus = 20.0 * max(0, s.MAX_STEPS - step) / s.MAX_STEPS
        shaping += speed_bonus

    if action != "BOMB" or e.BOMB_DROPPED not in events:
        return shaping

    field = np.asarray(old_game_state["field"])
    position = tuple(old_game_state["self"][3])
    opponents = opponent_positions(old_game_state.get("others", ()))
    destroyed_crates, opponent_in_blast = bomb_effects_from(
        position,
        field,
        opponents,
    )
    if not destroyed_crates and not opponent_in_blast:
        shaping -= 1.0

    if not has_safe_bomb_escape(
        field,
        position,
        old_game_state.get("bombs", ()),
        opponents,
        old_game_state.get("explosion_map"),
        s.BOMB_TIMER,
    ):
        shaping -= 5.0
    return shaping


def classic_peace_shaping(
    action,
    old_game_state,
    new_game_state,
    events,
    gamma,
):
    """Reward Task 2 target progress and actual movement out of danger."""

    def target_potential(game_state):
        if game_state is None:
            return 0.0

        field = np.asarray(game_state["field"])
        position = tuple(game_state["self"][3])
        bombs = game_state.get("bombs", ())
        occupied_by_bombs = bomb_positions(bombs)

        _, coin_distance = bfs_first_step(
            field,
            position,
            game_state.get("coins", ()),
            occupied_by_bombs,
        )
        _, crate_distance = efficient_crate_bombing_target_bfs(
            field,
            position,
            occupied_by_bombs,
        )

        if coin_distance is not None:
            return -8.0 * float(
                normalize_distance(coin_distance, field.shape)
            )
        if crate_distance is not None:
            return -3.0 * float(
                normalize_distance(crate_distance, field.shape)
            )
        return 0.0

    def danger_potential(game_state):
        if game_state is None:
            return 0.0

        field = np.asarray(game_state["field"])
        position = tuple(game_state["self"][3])
        danger_map = build_danger_map(
            field,
            game_state.get("bombs", ()),
            game_state.get("explosion_map"),
        )
        return -2.0 * float(danger_at(danger_map, position))

    if action == "BOMB" and e.BOMB_DROPPED in events:
        field = np.asarray(old_game_state["field"])
        position = tuple(old_game_state["self"][3])
        destroyed_crates, _ = bomb_effects_from(position, field, ())
        safe_escape = has_safe_bomb_escape(
            field,
            position,
            old_game_state.get("bombs", ()),
            (),
            old_game_state.get("explosion_map"),
            s.BOMB_TIMER,
        )

        shaping = -1.0 if not destroyed_crates else 0.0
        if not safe_escape:
            shaping -= 5.0
        return shaping

    killed = e.KILLED_SELF in events or e.GOT_KILLED in events
    if new_game_state is None and killed:
        return 0.0

    target_progress = gamma * target_potential(new_game_state)
    target_progress -= target_potential(old_game_state)
    danger_progress = gamma * danger_potential(new_game_state)
    danger_progress -= danger_potential(old_game_state)
    return float(np.clip(target_progress + danger_progress, -2.0, 2.0))
