import atexit
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from ..common.features import sarsa_task3_features_oc10
from . import config

# Initializing random seeds, performance timers, and load the Q-table from disk (or create a new one)
def setup(self):
    self.rng = np.random.default_rng(config.RANDOM_SEED)
    self.decision_count = 0
    self.decision_time_total = 0.0
    self.decision_time_max = 0.0
    self.timing_path = os.environ.get(config.TIMING_PATH_ENV)
    
    # Ensuring timing summaries are written when the program terminates
    if self.timing_path:
        atexit.register(_write_decision_summary, self)
        
    # Starting with a fresh Q-table if requested via config
    if self.train and config.REINITIALIZE_Q_TABLE:
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {}
        return
    if not config.LOAD_PATH.is_file():
        raise FileNotFoundError(f"SARSA model not found: {config.LOAD_PATH}")
        
    # Loading the pre-trained Q-table model weights from the pickle file
    self.logger.info("Loading Q-table from saved state.")
    try:
        with config.LOAD_PATH.open("rb") as file:
            self.q_table = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise RuntimeError(
            f"Failed to load SARSA model: {config.LOAD_PATH}"
        ) from error
    if not isinstance(self.q_table, dict):
        raise RuntimeError(f"Invalid SARSA model data: {config.LOAD_PATH}")


# Extracting the 10-dimensional feature vector from the raw game state dictionary
def state_to_features(game_state):
    features = sarsa_task3_features_oc10(game_state)
    return None if features is None else tuple(features)

# Determining which actions are currently valid, restricting illogical moves when no opponents/crates exist
def available_actions(game_state, state=None):
    if game_state.get("others") or np.any(np.asarray(game_state["field"]) == 1):
        return config.ACTIONS
    state = state_to_features(game_state) if state is None else state
    movements = [
        action
        for action, free in zip(config.ACTIONS[:4], state[1:5])
        if free
    ]
    return movements or ["WAIT"]

# Choosing an action using an epsilon-greedy policy for exploration during training
def select_action(self, game_state, explore=False):
    state = state_to_features(game_state)
    actions = available_actions(game_state, state)
    unseen = state not in self.q_table
    values = action_values(self, state) if explore else self.q_table.get(state)
    exploring = explore and self.rng.random() < self.epsilon

    # Taking a random action if exploring or if the current state has never been visited before
    if unseen or exploring:
        return str(self.rng.choice(actions))
        
    # Otherwise, exploiting the best known action (handling ties randomly)
    best = max(values[action] for action in actions)
    tied = [action for action in actions if values[action] == best]
    return str(self.rng.choice(tied))

# Main game engine: measuring decision time and return the selected action for the current step
def act(self, game_state):
    start = perf_counter()

    # During training, using the action that was pre-calculated during the previous step's SARSA update
    if self.train and self.next_action is not None:
        action = self.next_action
        self.next_action = None
    else:
        action = select_action(self, game_state)

    # Tracking decision profiling metrics (total time, max time)
    elapsed = perf_counter() - start
    self.decision_count += 1
    self.decision_time_total += elapsed
    self.decision_time_max = max(self.decision_time_max, elapsed)
    if hasattr(self, "metrics"):
        self.metrics.record_decision_time(elapsed)
    return action

# Fetching the Q-values for a specific state, initializing them to 0.0 if the state is unvisited
def action_values(self, state):
    return self.q_table.setdefault(state, dict.fromkeys(config.ACTIONS, 0.0))

# Dumping the performance timing metrics (mean/max decision time) to a JSON file when the program exits
def _write_decision_summary(self):
    path = Path(self.timing_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = self.decision_time_total / self.decision_count if self.decision_count else 0.0
    data = {"count": self.decision_count, "mean": mean, "max": self.decision_time_max}
    path.write_text(json.dumps(data, indent=2) + "\n")
