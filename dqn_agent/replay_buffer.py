from collections import deque, namedtuple
import random

import numpy as np
import torch

from . import config


ReplayTransition = namedtuple(
    "ReplayTransition",
    ("state", "action", "reward", "next_state", "done", "next_valid_mask"),
)


class ReplayBuffer:
    """Store a fixed number of DQN transitions."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, reward, next_state, done, next_valid_mask):
        if state is None:
            raise ValueError("state must not be None")
        if action < 0 or action >= len(config.ACTIONS):
            raise ValueError("action index is outside the configured actions")
        if next_state is None and not done:
            raise ValueError("non-terminal transition requires a next state")
        mask = valid_mask(next_valid_mask, done)
        self.memory.append(
            ReplayTransition(
                state,
                int(action),
                float(reward),
                next_state,
                bool(done),
                mask,
            )
        )

    def sample(self, batch_size):
        if batch_size <= 0 or batch_size > len(self.memory):
            raise ValueError("invalid replay sample size")
        return random.sample(self.memory, batch_size)

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size


def valid_mask(mask, done):
    """Return a valid action mask for a replay transition."""
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.shape != (len(config.ACTIONS),):
        raise ValueError("action mask has the wrong shape")
    if done:
        return np.zeros(len(config.ACTIONS), dtype=np.bool_)
    if not mask.any():
        raise ValueError("non-terminal action mask must allow an action")
    return mask.copy()


def transitions_to_tensors(transitions, device):
    """Convert replay transitions into tensors for one DQN update."""
    transitions = list(transitions)
    if not transitions:
        raise ValueError("transitions must not be empty")

    state_shape = np.asarray(transitions[0].state, dtype=np.float32).shape
    empty_state = np.zeros(state_shape, dtype=np.float32)

    states = np.asarray([item.state for item in transitions], dtype=np.float32)
    actions = np.asarray([[item.action] for item in transitions], dtype=np.int64)
    rewards = np.asarray([[item.reward] for item in transitions], dtype=np.float32)
    next_states = np.asarray(
        [
            empty_state if item.done or item.next_state is None else item.next_state
            for item in transitions
        ],
        dtype=np.float32,
    )
    dones = np.asarray([[item.done] for item in transitions], dtype=np.bool_)
    masks = np.asarray(
        [valid_mask(item.next_valid_mask, item.done) for item in transitions],
        dtype=np.bool_,
    )

    return (
        torch.as_tensor(states, dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.long, device=device),
        torch.as_tensor(rewards, dtype=torch.float32, device=device),
        torch.as_tensor(next_states, dtype=torch.float32, device=device),
        torch.as_tensor(dones, dtype=torch.bool, device=device),
        torch.as_tensor(masks, dtype=torch.bool, device=device),
    )
