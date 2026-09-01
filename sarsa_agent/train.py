import json
import pickle
from pathlib import Path

import numpy as np

from ..common.metrics import Task1Metrics
from ..common.rewards import coin_heaven_rewards_sarsa
from . import config
from .callbacks import ACTIONS, state_to_features


METRICS_PATH = Path(__file__).with_name("train.jsonl")


def setup_training(self):
    self.alpha = config.ALPHA
    self.gamma = config.GAMMA
    self.epsilon = config.EPSILON
    self.epsilon_min = config.EPSILON_MIN
    self.epsilon_decay = config.EPSILON_DECAY

    self.next_action = None
    self.episode = 0
    self.current_steps = 0
    self.metrics = Task1Metrics()
    METRICS_PATH.write_text("")


def get_action_for_state(self, state):
    if state not in self.q_table:
        self.q_table[state] = dict.fromkeys(ACTIONS, 0.0)

    if np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS)

    actions = list(self.q_table[state])
    q_values = list(self.q_table[state].values())
    return actions[np.argmax(q_values)]


def game_events_occurred(
    self, old_game_state, self_action, new_game_state, events
):
    state = state_to_features(old_game_state)
    action = self_action
    reward = coin_heaven_rewards_sarsa(events)

    route = state[4:8]
    if any(route):
        path_action = ACTIONS[int(np.argmax(route))]
        reward += 2 if action == path_action else -1

    next_state = state_to_features(new_game_state)
    next_action = get_action_for_state(self, next_state)
    self.next_action = next_action

    if state not in self.q_table:
        self.q_table[state] = dict.fromkeys(ACTIONS, 0.0)

    old_q_value = self.q_table[state][action]
    next_q_value = self.q_table[next_state][next_action]
    self.q_table[state][action] = old_q_value + self.alpha * (
        reward + self.gamma * next_q_value - old_q_value
    )

    self.current_steps += 1
    self.metrics.record_events(events, reward, old_game_state)


def end_of_round(self, last_game_state, last_action, events):
    state = state_to_features(last_game_state)
    reward = coin_heaven_rewards_sarsa(events)

    if state not in self.q_table:
        self.q_table[state] = dict.fromkeys(ACTIONS, 0.0)

    old_q_value = self.q_table[state][last_action]
    self.q_table[state][last_action] = old_q_value + self.alpha * (
        reward - old_q_value
    )

    self.current_steps += 1
    self.metrics.record_events(events, reward, last_game_state)
    self.episode += 1
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    metric = self.metrics.to_dict(self.episode, self.current_steps)
    with METRICS_PATH.open("a") as file:
        file.write(json.dumps(metric) + "\n")

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

    self.next_action = None
    self.current_steps = 0
    self.metrics.reset()
