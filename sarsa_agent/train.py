import json
import pickle
from pathlib import Path

import numpy as np

from ..common.metrics import Task1Metrics
from ..common.rewards import coin_heaven_rewards_sarsa
from . import config
from .callbacks import ACTIONS, MODEL_PATH, action_values, select_action, state_to_features


METRICS_PATH = Path(__file__).with_name("train.jsonl")


def setup_training(self):
    self.alpha = config.ALPHA
    self.gamma = config.GAMMA
    self.epsilon = config.EPSILON
    self.epsilon_min = config.EPSILON_MIN
    self.epsilon_decay = config.EPSILON_DECAY
    self.next_action = None
    self.pending_transition = None
    self.episode = 0
    self.current_steps = 0
    self.metrics = Task1Metrics()
    METRICS_PATH.write_text("")


def _reward(old_state, action, events):
    reward = coin_heaven_rewards_sarsa(events)
    route = state_to_features(old_state)[4:8]
    if any(route):
        reward += 2 if action == ACTIONS[int(np.argmax(route))] else -1
    return reward


def _record(self, old_game_state, action, new_game_state, next_action, events):
    state = state_to_features(old_game_state)
    reward = _reward(old_game_state, action, events)
    values = action_values(self, state)
    target = reward
    if new_game_state is not None:
        next_state = state_to_features(new_game_state)
        target += self.gamma * action_values(self, next_state)[next_action]
    values[action] += self.alpha * (target - values[action])
    self.current_steps += 1
    self.metrics.record_events(events, reward, old_game_state, new_game_state)


def _flush_pending(self, terminal=False, events=None):
    pending = self.pending_transition
    final_events = list(events) if events is not None else pending[4]
    new_state = None if terminal else pending[2]
    next_action = None if terminal else pending[3]
    _record(self, pending[0], pending[1], new_state, next_action, final_events)
    self.pending_transition = None


def game_events_occurred(
    self, old_game_state, self_action, new_game_state, events
):
    if self.pending_transition is not None:
        _flush_pending(self)
    next_action = select_action(self, new_game_state, explore=True)
    self.next_action = next_action
    self.pending_transition = (
        old_game_state, self_action, new_game_state, next_action, list(events)
    )


def _pending_matches(self, game_state, action):
    pending_state, pending_action = self.pending_transition[:2]
    return (
        pending_state["round"] == game_state["round"]
        and pending_state["step"] == game_state["step"]
        and pending_action == action
    )


def end_of_round(self, last_game_state, last_action, events):
    if self.pending_transition is not None and _pending_matches(
        self, last_game_state, last_action
    ):
        _flush_pending(self, terminal=True, events=events)
    else:
        if self.pending_transition is not None:
            _flush_pending(self)
        _record(self, last_game_state, last_action, None, None, list(events))

    self.episode += 1
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    metric = self.metrics.to_dict(self.episode, self.current_steps)
    with METRICS_PATH.open("a") as file:
        file.write(json.dumps(metric) + "\n")
    with MODEL_PATH.open("wb") as file:
        pickle.dump(self.q_table, file)

    self.next_action = None
    self.current_steps = 0
    self.metrics.reset()
