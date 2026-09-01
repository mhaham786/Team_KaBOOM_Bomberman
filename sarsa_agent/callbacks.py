import os
import pickle

import numpy as np

from ..common.features import coin_heaven_bfs_oc9


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {}
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)


def state_to_features(game_state):
    """Return the shared Coin Heaven features as a hashable state."""
    features = coin_heaven_bfs_oc9(game_state)
    if features is None:
        return None
    return tuple(features)


def act(self, game_state):
    if self.train and self.next_action is not None:
        action = self.next_action
        self.next_action = None
        return action

    state = state_to_features(game_state)
    if state not in self.q_table:
        return np.random.choice(ACTIONS)

    actions = list(self.q_table[state])
    q_values = list(self.q_table[state].values())
    return actions[np.argmax(q_values)]
