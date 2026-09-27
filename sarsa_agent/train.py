import json
import pickle
from pathlib import Path

import numpy as np

from ..common.metrics import Task3Metrics
from ..common.features import sarsa_task3_features_oc10
from ..common.rewards import task3_rewards_sarsa
from . import config
from .callbacks import action_values, select_action, state_to_features


# Initializing SARSA hyperparameters, training state variables, and reset the metrics log file
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
    self.metrics = Task3Metrics()
    config.METRICS_PATH.write_text("")
    
# Calculating a custom shaped reward based on the agent's survival (avoiding danger) and progress toward targets
def _reward(old_state, action, events):
    reward = task3_rewards_sarsa(events)
    if old_state is None:
        return reward

    features = sarsa_task3_features_oc10(old_state)
    
    current_danger = features[0]
    safe_moves = features[1:5]
    target_route = features[5:9]
    good_bomb_spot = features[9]
    
# Penalizing the agent for staying in danger or reward it for moving to safety
    if current_danger:
        if action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
            action_idx = ['UP', 'RIGHT', 'DOWN', 'LEFT'].index(action)
            if safe_moves[action_idx]:
                reward += 2
            else:
                reward -= 5
        else:
            reward -= 10
    else:
        # Applying a slight time penalty to encourage movement, and penalize moving into unsafe tiles
        reward -= 0.1 
        
        if action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
            action_idx = ['UP', 'RIGHT', 'DOWN', 'LEFT'].index(action)
            if not safe_moves[action_idx]:
                reward -= 5
        # Rewarding placing a bomb in a useful spot, or reward stepping toward the nearest target
        if action == 'BOMB':
            if good_bomb_spot:
                reward += 5
            else:
                reward -= 10
        elif action != 'WAIT':
            if any(target_route):
                best_action = config.ACTIONS[int(np.argmax(target_route))]
                reward += 1 if action == best_action else -1

    return reward

# Performing the core SARSA Q-table update using the temporal difference error and update step metrics
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

# Processing the cached transition from the previous step now that the next action is known
def _flush_pending(self, terminal=False, events=None):
    pending = self.pending_transition
    final_events = list(events) if events is not None else pending[4]
    new_state = None if terminal else pending[2]
    next_action = None if terminal else pending[3]
    _record(self, pending[0], pending[1], new_state, next_action, final_events)
    self.pending_transition = None

# Handling mid-game steps: flush the previous transition, pick the next action, and cache the current state
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

# Verifying if the currently cached transition matches the final state and action of the round
def _pending_matches(self, game_state, action):
    pending_state, pending_action = self.pending_transition[:2]
    return (
        pending_state["round"] == game_state["round"]
        and pending_state["step"] == game_state["step"]
        and pending_action == action
    )

# Finalizing the episode, process the last transition, decay the exploration rate, and save the model
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
    with config.METRICS_PATH.open("a") as file:
        file.write(json.dumps(metric) + "\n")
    with config.MODEL_PATH.open("wb") as file:
        pickle.dump(self.q_table, file)

    self.next_action = None
    self.current_steps = 0
    self.metrics.reset()
