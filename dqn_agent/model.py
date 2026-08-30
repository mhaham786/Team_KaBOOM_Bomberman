from collections import deque, namedtuple
import random
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
N_ACTIONS = 6

if len(ACTIONS) != N_ACTIONS:
    raise RuntimeError("ACTIONS must contain exactly six fixed actions")

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "next_valid_mask"),
)


def _coerce_valid_mask(next_valid_mask, *, done: bool) -> np.ndarray:
    """Return a validated Boolean mask with one entry per fixed action.

    REVIEW: terminal transitions intentionally store an all-false mask. Training
    code must still use the done tensor to skip bootstrapping from terminal rows.
    Non-terminal all-false masks are rejected because max over no legal actions is
    ill-defined and can lead to unstable target computation.
    """
    mask = np.asarray(next_valid_mask, dtype=np.bool_)
    if mask.shape != (N_ACTIONS,):
        raise ValueError(f"next_valid_mask must have shape ({N_ACTIONS},)")
    if done:
        return np.zeros(N_ACTIONS, dtype=np.bool_)
    if not mask.any():
        raise ValueError("non-terminal next_valid_mask must allow at least one action")
    return mask.copy()


class DQN(nn.Module):
    """
    Configurable multilayer perceptron for estimating action values.

    Input shape:
        (batch_size, input_dim)

    Output shape:
        (batch_size, 6)

    Each output is the estimated Q-value for one action in fixed ACTIONS order.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if n_actions != N_ACTIONS:
            raise ValueError(f"DQN baseline has a fixed output size of {N_ACTIONS}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden layer sizes must be positive")
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, N_ACTIONS))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Return Q-values for a batch of feature vectors.

        REVIEW: advanced_features_oc31 should produce flat float vectors whose
        length matches input_dim. If you later switch to image-like channels,
        replace this MLP with a CNN or flatten before calling the model.
        """
        if states.ndim == 1:
            states = states.unsqueeze(0)
        return self.network(states.float())


class ReplayBuffer:
    """
    Fixed-size FIFO replay memory for off-policy DQN updates.

    Stores transitions as:
        state: np.ndarray or torch.Tensor, shape (input_dim,)
        action: int, scalar index into ACTIONS
        reward: float, scalar immediate reward
        next_state: same shape as state, or None for terminal transitions
        done: bool, True when the transition ends an episode
        next_valid_mask: bool array, shape (6,), valid actions in next_state
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        next_valid_mask,
    ) -> None:
        if state is None:
            raise ValueError("state must not be None")
        if action < 0 or action >= N_ACTIONS:
            raise ValueError(f"action must be in [0, {N_ACTIONS - 1}]")
        done = bool(done)
        if next_state is None and not done:
            raise ValueError("next_state must not be None for non-terminal transitions")
        mask = _coerce_valid_mask(next_valid_mask, done=done)
        self.memory.append(
            Transition(state, int(action), float(reward), next_state, done, mask)
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.memory):
            raise ValueError("batch_size cannot exceed replay buffer length")
        return random.sample(self.memory, batch_size)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size


def action_to_index(action: str) -> int:
    """Convert an environment action string to the model output index."""
    return ACTIONS.index(action)


def index_to_action(index: int) -> str:
    """Convert a model output index back to an environment action string."""
    if index < 0 or index >= N_ACTIONS:
        raise ValueError(f"index must be in [0, {N_ACTIONS - 1}]")
    return ACTIONS[index]


def build_dqn(
    input_dim: int,
    hidden_dims: Sequence[int] = (128, 128),
) -> DQN:
    """
    Factory for the baseline MLP DQN.

    REVIEW: The default hidden_dims=(128, 128) is the requested baseline.
    Tune this only after feature extraction and reward shaping are stable.
    """
    return DQN(input_dim=input_dim, hidden_dims=hidden_dims, n_actions=N_ACTIONS)


def transitions_to_tensors(
    transitions: Iterable[Transition],
    device: Optional[torch.device] = None,
):
    """
    Convert sampled replay transitions into tensors for a DQN training step.

    Returns:
        states: float tensor, shape (batch_size, input_dim)
        actions: long tensor, shape (batch_size, 1)
        rewards: float tensor, shape (batch_size, 1)
        next_states: float tensor, shape (batch_size, input_dim)
        dones: bool tensor, shape (batch_size, 1)
        next_valid_masks: bool tensor, shape (batch_size, 6)

    REVIEW: Terminal transitions get a zero next_state and an all-false valid
    mask. The done tensor must be applied before future-Q bootstrapping, so
    terminal rows never require max over masked invalid Q-values.
    """
    transitions = list(transitions)
    if not transitions:
        raise ValueError("transitions must not be empty")

    first_state = transitions[0].state
    if first_state is None:
        raise ValueError("transition state must not be None")

    state_shape = np.asarray(first_state, dtype=np.float32).shape
    zero_next_state = np.zeros(state_shape, dtype=np.float32)

    states = np.asarray([t.state for t in transitions], dtype=np.float32)
    actions = np.asarray([[t.action] for t in transitions], dtype=np.int64)
    rewards = np.asarray([[t.reward] for t in transitions], dtype=np.float32)
    next_states = np.asarray(
        [zero_next_state if t.done or t.next_state is None else t.next_state for t in transitions],
        dtype=np.float32,
    )
    dones = np.asarray([[t.done] for t in transitions], dtype=np.bool_)
    next_valid_masks = np.asarray(
        [_coerce_valid_mask(t.next_valid_mask, done=t.done) for t in transitions],
        dtype=np.bool_,
    )

    return (
        torch.as_tensor(states, dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.long, device=device),
        torch.as_tensor(rewards, dtype=torch.float32, device=device),
        torch.as_tensor(next_states, dtype=torch.float32, device=device),
        torch.as_tensor(dones, dtype=torch.bool, device=device),
        torch.as_tensor(next_valid_masks, dtype=torch.bool, device=device),
    )
