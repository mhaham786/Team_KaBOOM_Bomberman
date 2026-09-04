import atexit
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from ..common.features import coin_heaven_bfs_oc9
from . import config


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MODEL_FILENAME = "sarsa_model_task1.pt"
MODEL_PATH = Path(__file__).with_name(MODEL_FILENAME)
TIMING_PATH_ENV = "SARSA_DECISION_TIMING_PATH"


def setup(self):
    self.rng = np.random.default_rng(config.RANDOM_SEED)
    self.decision_count = 0
    self.decision_time_total = 0.0
    self.decision_time_max = 0.0
    self.timing_path = os.environ.get(TIMING_PATH_ENV)
    if self.timing_path:
        atexit.register(_write_decision_summary, self)

    if self.train:
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {}
        return
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"SARSA model not found: {MODEL_PATH}")

    self.logger.info("Loading Q-table from saved state.")
    try:
        with MODEL_PATH.open("rb") as file:
            self.q_table = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise RuntimeError(f"Failed to load SARSA model: {MODEL_PATH}") from error
    if not isinstance(self.q_table, dict):
        raise RuntimeError(f"Invalid SARSA model data: {MODEL_PATH}")


def state_to_features(game_state):
    features = coin_heaven_bfs_oc9(game_state)
    return None if features is None else tuple(features)

def available_actions(game_state, state=None):
    if game_state.get("others") or np.any(np.asarray(game_state["field"]) == 1):
        return ACTIONS
    state = state_to_features(game_state) if state is None else state
    movements = [action for action, free in zip(ACTIONS[:4], state[:4]) if free]
    return movements or ["WAIT"]


def select_action(self, game_state, explore=False):
    state = state_to_features(game_state)
    actions = available_actions(game_state, state)
    unseen = state not in self.q_table
    values = action_values(self, state) if explore else self.q_table.get(state)
    exploring = explore and self.rng.random() < self.epsilon
    if unseen or exploring:
        return str(self.rng.choice(actions))
    best = max(values[action] for action in actions)
    tied = [action for action in actions if values[action] == best]
    return str(self.rng.choice(tied))


def act(self, game_state):
    start = perf_counter()
    if self.train and self.next_action is not None:
        action = self.next_action
        self.next_action = None
    else:
        action = select_action(self, game_state)

    elapsed = perf_counter() - start
    self.decision_count += 1
    self.decision_time_total += elapsed
    self.decision_time_max = max(self.decision_time_max, elapsed)
    if hasattr(self, "metrics"):
        self.metrics.record_decision_time(elapsed)
    return action


def action_values(self, state):
    return self.q_table.setdefault(state, dict.fromkeys(ACTIONS, 0.0))


def _write_decision_summary(self):
    path = Path(self.timing_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = self.decision_time_total / self.decision_count if self.decision_count else 0.0
    data = {"count": self.decision_count, "mean": mean, "max": self.decision_time_max}
    path.write_text(json.dumps(data, indent=2) + "\n")
