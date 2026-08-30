import csv
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import events as e
import settings as s
from ..common.metrics import EpisodeMetrics
from ..common.features import advanced_features_oc31
from ..common.helpers import (
    bfs_first_step,
    bomb_positions,
    has_safe_bomb_escape,
    opponent_positions,
    valid_action_mask,
)
from ..common.plots.general_plot import create_figure
from ..common.plots.task1 import create_figure_task1
from .model import (
    N_ACTIONS,
    ReplayBuffer,
    action_to_index,
    build_dqn,
    transitions_to_tensors,
)


BATCH_SIZE = 64
FEATURE_DIM = 31
FEATURE_SCHEMA = "uttam_dqn_v2_time_escape_31"
ARTIFACT_VERSION = 2
GAMMA = 0.99
LEARNING_RATE = 1e-4
REPLAY_CAPACITY = 50_000
REPLAY_FORMAT_VERSION = 1
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_INTERVAL = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 30_000
GRADIENT_CLIP_NORM = 10.0
RANDOM_SEED = 42
SAVE_EVERY_ROUNDS = 25
STEP_PENALTY = -0.01
USELESS_BOMB_PENALTY = -1.0
UNSAFE_BOMB_PENALTY = -5.0

COLLECT_ONLY_ENV = "UTTAM_DQN_COLLECT_ONLY"
TRAIN_EVERY_ENV = "UTTAM_DQN_TRAIN_EVERY"
SAVE_EVERY_ENV = "UTTAM_DQN_SAVE_EVERY_ROUNDS"
LEARNING_RATE_ENV = "UTTAM_DQN_LEARNING_RATE"

MODEL_FILENAME = "dqn_model_v2.pt"
CHECKPOINT_FILENAME = "dqn_training_checkpoint_v2.pt"
METRICS_FILENAME = "training_metrics_v2.csv"

EVENT_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.CRATE_DESTROYED: 2.0,
    e.COIN_FOUND: 1.0,
    e.KILLED_OPPONENT: 50.0,
    e.SURVIVED_ROUND: 5.0,
    e.INVALID_ACTION: -5.0,
    e.KILLED_SELF: -50.0,
    e.GOT_KILLED: -40.0,
    e.WAITED: -0.1,
    e.BOMB_DROPPED: -0.1,
}

METRICS_COLUMNS = (
    "round",
    "score",
    "steps",
    "round_reward",
    "event_reward",
    "shaping_reward",
    "epsilon",
    "replay_size",
    "mean_loss",
    "coins_collected",
    "crates_destroyed",
    "opponents_killed",
    "self_kills",
    "got_killed",
    "invalid_actions",
    "survived",
)


class CheckpointError(RuntimeError):
    """Raised when a resumable DQN training checkpoint cannot be restored."""


def setup_training(self):
    """Initialize baseline CPU DQN training state.

    The policy network is created by callbacks.setup() and reused here. A
    separate target network is initialized from the policy network and updated
    periodically. New checkpoints include a compact tensor replay snapshot and
    RNG states so resumed training can continue sampling from prior experience.
    Legacy checkpoints without replay data still load with an empty replay
    buffer for backward compatibility.
    """
    _require_callback_state(self)
    self.device = torch.device("cuda")
    self.metrics = EpisodeMetrics()

    torch.manual_seed(RANDOM_SEED)
    self.rng.seed(RANDOM_SEED)
    self.replay_rng = random.Random(RANDOM_SEED)

    base_dir = Path(__file__).resolve().parent
    self.model_path = _require_v2_model_path(getattr(self, "model_path", base_dir / MODEL_FILENAME))
    self.checkpoint_path = getattr(self, "checkpoint_path", base_dir / CHECKPOINT_FILENAME)
    self.metrics_path = getattr(self, "metrics_path", base_dir / METRICS_FILENAME)

    checkpoint_exists = self.checkpoint_path.exists()
    model_exists = _compatible_v2_model_exists(self.model_path)

    self.policy_net.to(self.device)
    if not checkpoint_exists and not model_exists:
        _reset_network_parameters(self.policy_net)

    self.target_net = build_dqn(FEATURE_DIM).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.learning_rate, learning_rate_overridden = _learning_rate_from_env()
    self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
    self.collect_only, self.train_every, self.save_every_rounds = _training_mode_from_env()

    self.pending_transition = None
    self.total_transitions = 0
    self.optimization_steps = 0
    self.completed_rounds = 0
    self.epsilon = _epsilon_by_transition_count(self.total_transitions)

    _reset_round_stats(self)
    if checkpoint_exists:
        _load_training_checkpoint(self, self.checkpoint_path)
        if learning_rate_overridden:
            _set_optimizer_learning_rate(self.optimizer, self.learning_rate)
        else:
            self.learning_rate = _optimizer_learning_rate(self.optimizer)
        _validate_metrics_history(self.metrics_path, self.completed_rounds)
    elif self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
        raise CheckpointError(
            f"metrics file exists without a training checkpoint: {self.metrics_path}"
        )

    self.policy_net.eval()
    self.target_net.eval()


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    if getattr(self, "needs_new_target_distance", True):
        field = np.asarray(old_game_state["field"])
        position = tuple(old_game_state["self"][3])
        coins = old_game_state.get("coins", ())
        bombs = bomb_positions(old_game_state.get("bombs", ()))
        opponents = opponent_positions(old_game_state.get("others", ()))

        _, distance = bfs_first_step(field, position, coins, bombs, opponents)
        if distance is not None:
            self.current_target_distance = distance 
        self.needs_new_target_distance = False

    if e.COIN_COLLECTED in events:
        if hasattr(self, "current_target_distance"):
            self.cumulative_optimal_distance += self.current_target_distance
        self.needs_new_target_distance = True
    """Record the newest transition pending and flush the previous one.

    BombeRLe can report a surviving final action here and then call
    end_of_round() with the same old state and action. Keeping the newest
    transition pending lets end_of_round() store that final action exactly once
    as terminal with the final event list.
    """
    if self.pending_transition is not None:
        _insert_pending_transition(self, self.pending_transition, done=False)
        self.pending_transition = None

    self.pending_transition = _make_pending_transition(
        old_game_state,
        self_action,
        new_game_state,
        events,
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Finalize pending/final transitions, append metrics, and save periodically."""
    completed_score = _terminal_score_from_state_and_events(last_game_state, events)

    if self.pending_transition is not None:
        if _pending_matches_final(self.pending_transition, last_game_state, last_action):
            completed_score = self.pending_transition["next_score"]
            _insert_pending_transition(
                self,
                self.pending_transition,
                done=True,
                replacement_events=events,
            )
        else:
            _insert_pending_transition(self, self.pending_transition, done=False)
            _insert_terminal_transition(self, last_game_state, last_action, events)
        self.pending_transition = None
    else:
        _insert_terminal_transition(self, last_game_state, last_action, events)

    self.completed_rounds += 1

    if self.round_event_counts[e.COIN_COLLECTED] > 0:
        efficiency = self.cumulative_optimal_distance / max(1, self.round_steps)
        self.metrics.record_path_efficiency(efficiency)

    if hasattr(self, 'metrics'):
        metric_dict = self.metrics.to_dict(self.completed_rounds, self.round_steps)
        _append_metrics_row(self, metric_dict)

    if self.completed_rounds % self.save_every_rounds == 0:
        _save_policy_model(self, self.model_path)
        _save_training_checkpoint(self, self.checkpoint_path)

    _reset_round_stats(self)
    self.policy_net.eval()

    target_rounds = getattr(self, 'n_rounds',10000)

    if self.completed_rounds == target_rounds:

        history_df = pd.read_csv(self.metrics_path)
        history_list = history_df.to_dict(orient='records')


        # 1. General Plot
        fig = create_figure(history_list)
        fig.savefig("final_training_plot_general_GAMMA0.99.png")
        plt.close(fig)

        # 2. Task 1 Plot
        fig_task1 = create_figure_task1(history_list)
        fig_task1.savefig("task1_training_plot_GAMMA0.99.png")
        plt.close(fig_task1)


def reward_from_events(events: Iterable[str]) -> float:
    """Return baseline event reward without rewarding ordinary movement."""
    return float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))



def potential_from_features(features: Optional[np.ndarray]) -> float:
    """Calculate state potential from current danger and target distances."""
    if features is None:
        return 0.0
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != (FEATURE_DIM,):
        raise ValueError(f"features must have shape ({FEATURE_DIM},)")

    potential = -6.0 * float(feature_array[5])

    coin_exists = bool(feature_array[10:14].any() or feature_array[14] == 0.0)
    crate_exists = bool(feature_array[15:19].any() or feature_array[19] == 0.0)
    opponent_exists = bool(feature_array[20:24].any() or feature_array[24] == 0.0)

    if coin_exists:
        potential -= 8.0 * float(feature_array[14])
    elif crate_exists:
        potential -= 3.0 * float(feature_array[19])

    if opponent_exists:
        potential -= 1.0 * float(feature_array[24])

    return float(potential)


def potential_shaping(old_features: np.ndarray, next_features: Optional[np.ndarray]) -> float:
    """Return clipped potential-based shaping reward."""
    shaping = GAMMA * potential_from_features(next_features) - potential_from_features(old_features)
    return float(np.clip(shaping, -2.0, 2.0))


def action_shaping_reward(
    action: str,
    old_features: np.ndarray,
    events: Iterable[str],
    old_game_state: Optional[dict] = None,
) -> float:
    """Return immediate action shaping for successful bomb placement.

    Useless bombs and unsafe bombs are independent interpretable penalties. An
    invalid attempted bomb receives neither because BOMB_DROPPED is absent and
    INVALID_ACTION already has its own event penalty.
    """
    feature_array = np.asarray(old_features, dtype=np.float32)
    if feature_array.shape != (FEATURE_DIM,):
        raise ValueError(f"features must have shape ({FEATURE_DIM},)")
    event_set = set(events)
    if action != "BOMB" or e.BOMB_DROPPED not in event_set:
        return 0.0

    reward = 0.0
    if feature_array[29] == 0.0 and feature_array[30] == 0.0:
        reward += USELESS_BOMB_PENALTY
    if old_game_state is not None:
        field = np.asarray(old_game_state["field"])
        position = tuple(old_game_state["self"][3])
        if not has_safe_bomb_escape(
            field,
            position,
            old_game_state.get("bombs", ()),
            opponent_positions(old_game_state.get("others", ())),
            old_game_state.get("explosion_map"),
            s.BOMB_TIMER,
        ):
            reward += UNSAFE_BOMB_PENALTY
    return float(reward)


def optimize_model(self) -> Optional[float]:
    """Run one Huber-loss DQN update when enough replay data exists."""
    if not self.replay_buffer.can_sample(BATCH_SIZE) or len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return None
    if getattr(self, "collect_only", False):
        return None
    train_every = int(getattr(self, "train_every", 1))
    if train_every <= 0:
        raise RuntimeError("train_every must be a positive integer")
    if self.total_transitions % train_every != 0:
        return None

    transitions = _sample_replay_batch(self)
    states, actions, rewards, next_states, dones, next_valid_masks = transitions_to_tensors(
        transitions,
        device=self.device,
    )

    self.policy_net.train()
    q_values = self.policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = self.target_net(next_states)
        masked_next_q_values = next_q_values.masked_fill(
            ~next_valid_masks,
            torch.finfo(next_q_values.dtype).min,
        )
        future_values = masked_next_q_values.max(dim=1, keepdim=True).values
        future_values = torch.where(dones, torch.zeros_like(future_values), future_values)
        targets = rewards + GAMMA * future_values

    loss = F.smooth_l1_loss(q_values, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("DQN optimization produced a non-finite loss")

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_NORM)
    self.optimizer.step()

    self.optimization_steps += 1
    if self.optimization_steps % TARGET_UPDATE_INTERVAL == 0:
        _sync_target_network(self)

    self.policy_net.eval()
    return float(loss.detach().cpu().item())


def _make_pending_transition(
    old_game_state: dict,
    action: str,
    new_game_state: dict,
    events: Iterable[str],
) -> dict:
    """Build a pending transition from the framework's step callback."""
    old_features = advanced_features_oc31(old_game_state)
    new_features = advanced_features_oc31(new_game_state)
    if old_features is None or new_features is None:
        raise ValueError("non-terminal game event states must not be None")
    event_list = list(events)
    return {
        "round": old_game_state["round"],
        "step": old_game_state["step"],
        "action": action,
        "state": old_features,
        "old_game_state": old_game_state,
        "action_index": action_to_index(action),
        "next_state": new_features,
        "next_valid_mask": valid_action_mask(
            np.asarray(new_game_state["field"]),
            tuple(new_game_state["self"][3]),
            new_game_state["self"][2],
            bomb_positions(new_game_state.get("bombs", ())),
            opponent_positions(new_game_state.get("others", ())),
        ),
        "events": event_list,
        "event_reward": reward_from_events(event_list),
        "shaping_reward": potential_shaping(old_features, new_features)
        + action_shaping_reward(action, old_features, event_list, old_game_state),
        "next_score": _score_from_state(new_game_state),
    }


def _pending_matches_final(pending: dict, last_game_state: dict, last_action: str) -> bool:
    """Return whether pending and end_of_round refer to the same action."""
    return (
        pending["round"] == last_game_state.get("round")
        and pending["step"] == last_game_state.get("step")
        and pending["action"] == last_action
    )


def _insert_pending_transition(
    self,
    pending: dict,
    *,
    done: bool,
    replacement_events: Optional[Iterable[str]] = None,
) -> Optional[float]:
    """Insert a pending transition as non-terminal or terminal exactly once."""
    if done:
        events = list(replacement_events if replacement_events is not None else pending["events"])
        event_reward = reward_from_events(events)
        shaping_reward = potential_shaping(pending["state"], None) + action_shaping_reward(
            pending["action"], pending["state"], events, pending["old_game_state"]
        )
        next_state = None
        next_valid_mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    else:
        events = pending["events"]
        event_reward = pending["event_reward"]
        shaping_reward = pending["shaping_reward"]
        next_state = pending["next_state"]
        next_valid_mask = pending["next_valid_mask"]

    reward = event_reward + shaping_reward + STEP_PENALTY
    self.replay_buffer.push(
        pending["state"],
        pending["action_index"],
        reward,
        next_state,
        done,
        next_valid_mask,
    )
    _after_transition(self, reward, event_reward, shaping_reward, events)
    loss = optimize_model(self)
    _record_loss(self, loss)
    return loss


def _insert_terminal_transition(
    self,
    last_game_state: dict,
    last_action: str,
    events: Iterable[str],
) -> Optional[float]:
    """Insert a terminal transition for a final action without a post-state."""
    last_features = advanced_features_oc31(last_game_state)
    if last_features is None:
        raise ValueError("last_game_state must not be None for terminal transition")
    event_list = list(events)
    event_reward = reward_from_events(event_list)
    shaping_reward = potential_shaping(last_features, None) + action_shaping_reward(
        last_action, last_features, event_list, last_game_state
    )
    reward = event_reward + shaping_reward + STEP_PENALTY
    terminal_mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    self.replay_buffer.push(
        last_features,
        action_to_index(last_action),
        reward,
        None,
        True,
        terminal_mask,
    )
    _after_transition(self, reward, event_reward, shaping_reward, event_list)
    loss = optimize_model(self)
    _record_loss(self, loss)
    return loss


def _require_callback_state(self) -> None:
    """Ensure callbacks.setup() prepared the objects training depends on."""
    missing = [name for name in ("policy_net", "model_path", "device", "rng") if not hasattr(self, name)]
    if missing:
        raise RuntimeError(f"callbacks.setup() must run before setup_training(); missing {missing}")


def _reset_network_parameters(policy_net: nn.Module) -> None:
    """Reinitialize fresh networks reproducibly after torch.manual_seed()."""
    for module in policy_net.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a strictly positive integer environment setting."""
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return parsed


def _learning_rate_from_env() -> tuple[float, bool]:
    """Return a positive finite learning rate and whether it was overridden."""
    value = os.environ.get(LEARNING_RATE_ENV)
    if value is None:
        return LEARNING_RATE, False
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{LEARNING_RATE_ENV} must be a positive finite float; got {value!r}"
        ) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(
            f"{LEARNING_RATE_ENV} must be a positive finite float; got {value!r}"
        )
    return parsed, True


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    """Apply an explicit experiment learning rate to every optimizer group."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Validate and return a checkpoint-restored optimizer learning rate."""
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not learning_rates or any(
        not np.isfinite(value) or value <= 0.0 for value in learning_rates
    ):
        raise ValueError("checkpoint optimizer learning rates must be positive and finite")
    if any(value != learning_rates[0] for value in learning_rates[1:]):
        raise ValueError("checkpoint optimizer parameter groups use inconsistent learning rates")
    return learning_rates[0]


def _training_mode_from_env() -> tuple[bool, int, int]:
    """Read optional replay, update-frequency and save-frequency controls."""
    collect_value = os.environ.get(COLLECT_ONLY_ENV, "0")
    if collect_value not in {"0", "1"}:
        raise ValueError(f"{COLLECT_ONLY_ENV} must be '0' or '1'; got {collect_value!r}")

    train_every = _positive_int_from_env(TRAIN_EVERY_ENV, 1)
    save_every_rounds = _positive_int_from_env(SAVE_EVERY_ENV, SAVE_EVERY_ROUNDS)
    return collect_value == "1", train_every, save_every_rounds


def _epsilon_by_transition_count(total_transitions: int) -> float:
    """Linear epsilon schedule from EPSILON_START to EPSILON_END."""
    fraction = min(max(total_transitions, 0) / float(EPSILON_DECAY_STEPS), 1.0)
    return float(EPSILON_START + fraction * (EPSILON_END - EPSILON_START))


def _sample_replay_batch(self):
    """Sample replay transitions with the agent-local deterministic RNG."""
    return self.replay_rng.sample(list(self.replay_buffer.memory), BATCH_SIZE)


def _after_transition(
    self,
    reward: float,
    event_reward: float,
    shaping_reward: float,
    events: Iterable[str],
) -> None:
    """Update counters only after a transition is inserted into replay."""
    self.total_transitions += 1
    self.epsilon = _epsilon_by_transition_count(self.total_transitions)
    self.round_steps += 1
    self.round_reward += float(reward)
    self.round_event_reward += float(event_reward)
    self.round_shaping_reward += float(shaping_reward)
    self.round_event_counts.update(events)
    self.metrics.record_events(list(events), float(event_reward))


def _record_loss(self, loss: Optional[float]) -> None:
    """Accumulate finite optimization losses for per-round metrics."""
    if loss is None:
        return
    if not np.isfinite(loss):
        raise RuntimeError("loss metric is not finite")
    self.round_losses.append(float(loss))


def _reset_round_stats(self) -> None:
    """Reset per-round reward, event and loss counters."""
    self.round_steps = 0
    self.round_reward = 0.0
    self.round_event_reward = 0.0
    self.round_shaping_reward = 0.0
    self.round_event_counts = Counter()
    self.round_losses = []
    self.cumulative_optimal_distance = 0
    self.needs_new_target_distance = True
    if hasattr(self, 'metrics'):
        self.metrics.reset()


def _sync_target_network(self) -> None:
    """Hard-copy policy weights to the target network."""
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()


def _score_from_state(game_state: dict) -> int:
    """Read the official score from a game-state self tuple."""
    return int(game_state["self"][1])


def _terminal_score_from_state_and_events(last_game_state: dict, events: Iterable[str]) -> int:
    """Estimate official final score when the framework provides no post-state."""
    event_counts = Counter(events)
    return (
        _score_from_state(last_game_state)
        + event_counts[e.COIN_COLLECTED] * s.REWARD_COIN
        + event_counts[e.KILLED_OPPONENT] * s.REWARD_KILL
    )


def _metrics_row(self, completed_score: int) -> dict:
    """Build one CSV metrics row for the completed round."""
    mean_loss = float(np.mean(self.round_losses)) if self.round_losses else 0.0
    row = {
        "round": self.completed_rounds,
        "score": int(completed_score),
        "steps": self.round_steps,
        "round_reward": self.round_reward,
        "event_reward": self.round_event_reward,
        "shaping_reward": self.round_shaping_reward,
        "epsilon": self.epsilon,
        "replay_size": len(self.replay_buffer),
        "mean_loss": mean_loss,
        "coins_collected": self.round_event_counts[e.COIN_COLLECTED],
        "crates_destroyed": self.round_event_counts[e.CRATE_DESTROYED],
        "opponents_killed": self.round_event_counts[e.KILLED_OPPONENT],
        "self_kills": self.round_event_counts[e.KILLED_SELF],
        "got_killed": self.round_event_counts[e.GOT_KILLED],
        "invalid_actions": self.round_event_counts[e.INVALID_ACTION],
        "survived": int(self.round_event_counts[e.SURVIVED_ROUND] > 0),
    }
    numeric_values = [float(value) for value in row.values()]
    if not np.isfinite(numeric_values).all():
        raise RuntimeError("metrics row contains non-finite values")
    return row


def _append_metrics_row(self, metric_dict: dict) -> None:
    """Append one round of metrics using the dictionary from metrics.py."""
    self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
    
    with self.metrics_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metric_dict)

def _require_v2_model_path(model_path) -> Path:
    """Return model_path only when it names the isolated V2 model artifact."""
    path = Path(model_path)
    if path.name != MODEL_FILENAME:
        raise CheckpointError(
            f"V2 training model_path must be named {MODEL_FILENAME!r}; got {path.name!r}"
        )
    return path


def _validate_finite_state_dict(state_dict, label: str) -> None:
    """Reject model state dictionaries containing NaN or infinity."""
    if not isinstance(state_dict, dict):
        raise TypeError(f"{label} is not a state_dict dictionary")
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{label} entry {name!r} is not a tensor")
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"{label} entry {name!r} contains NaN or infinity")


def _validate_metrics_history(metrics_path: Path, completed_rounds: int) -> None:
    """Require metrics rounds to be unique, increasing and checkpoint-aligned."""
    metrics_path = Path(metrics_path)
    if completed_rounds < 0:
        raise ValueError("checkpoint completed_rounds must be non-negative")
    if not metrics_path.is_file():
        if completed_rounds == 0:
            return
        raise CheckpointError(
            f"metrics file is missing for checkpoint round {completed_rounds}: {metrics_path}"
        )

    with metrics_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(METRICS_COLUMNS):
            raise CheckpointError(f"metrics CSV has incompatible columns: {metrics_path}")
        rounds = [int(row["round"]) for row in reader]
    if any(current <= previous for previous, current in zip(rounds, rounds[1:])):
        raise CheckpointError(
            f"metrics rounds must be strictly increasing and unique: {metrics_path}"
        )
    if not rounds:
        if completed_rounds != 0:
            raise CheckpointError(
                f"metrics CSV has no row for checkpoint round {completed_rounds}: {metrics_path}"
            )
        return
    if rounds[-1] != completed_rounds:
        raise CheckpointError(
            f"checkpoint completed_rounds={completed_rounds} does not match "
            f"metrics final round={rounds[-1]} in {metrics_path}"
        )


def _compatible_v2_model_exists(model_path: Path) -> bool:
    """Return whether an existing V2 model artifact is schema-compatible."""
    model_path = _require_v2_model_path(model_path)
    if not model_path.exists():
        return False
    try:
        payload = torch.load(model_path, map_location="cpu")
        _validate_artifact_metadata(payload, model_path)
        if "state_dict" not in payload or not isinstance(payload["state_dict"], dict):
            raise TypeError("missing state_dict dictionary")
        _validate_finite_state_dict(payload["state_dict"], "model state_dict")
    except Exception as exc:
        raise CheckpointError(f"Failed to validate V2 model artifact at {model_path}: {exc}") from exc
    return True

def _artifact_metadata() -> dict:
    """Return metadata that prevents silent V1/V2 artifact mixing."""
    return {
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "artifact_version": ARTIFACT_VERSION,
    }


def _validate_artifact_metadata(payload: dict, artifact_path: Path) -> None:
    """Validate the V2 feature schema metadata stored in model/checkpoint files."""
    if not isinstance(payload, dict):
        raise TypeError("artifact payload is not a dictionary")
    for key, expected_value in _artifact_metadata().items():
        actual_value = payload.get(key)
        if actual_value != expected_value:
            raise CheckpointError(
                f"{artifact_path} has incompatible {key}={actual_value!r}; "
                f"expected {expected_value!r}"
            )


def _model_artifact_dict(self) -> dict:
    """Build the versioned V2 policy model artifact used by callbacks.py."""
    return {**_artifact_metadata(), "state_dict": self.policy_net.state_dict()}


def _save_policy_model(self, model_path: Path) -> None:
    """Atomically save the versioned V2 policy model artifact."""
    _atomic_torch_save(_model_artifact_dict(self), _require_v2_model_path(model_path))


def _checkpoint_dict(self) -> dict:
    """Build the resumable V2 training checkpoint with compact replay tensors."""
    return {
        **_artifact_metadata(),
        "policy_state_dict": self.policy_net.state_dict(),
        "target_state_dict": self.target_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "total_transitions": self.total_transitions,
        "optimization_steps": self.optimization_steps,
        "completed_rounds": self.completed_rounds,
        "epsilon": self.epsilon,
        "replay_buffer": _serialize_replay_buffer(self.replay_buffer),
        "replay_rng_state": self.replay_rng.getstate(),
        "action_rng_state": self.rng.getstate(),
    }


def _save_training_checkpoint(self, checkpoint_path: Path) -> None:
    """Atomically save the resumable V2 training checkpoint."""
    _atomic_torch_save(_checkpoint_dict(self), checkpoint_path)


def _serialize_replay_buffer(replay_buffer: ReplayBuffer) -> dict:
    """Pack replay memory into compact CPU tensors for checkpointing."""
    transitions = list(replay_buffer.memory)
    length = len(transitions)
    states = torch.empty((length, FEATURE_DIM), dtype=torch.float32)
    actions = torch.empty((length,), dtype=torch.long)
    rewards = torch.empty((length,), dtype=torch.float32)
    next_states = torch.zeros((length, FEATURE_DIM), dtype=torch.float32)
    next_state_present = torch.empty((length,), dtype=torch.bool)
    dones = torch.empty((length,), dtype=torch.bool)
    next_valid_masks = torch.empty((length, N_ACTIONS), dtype=torch.bool)

    for index, transition in enumerate(transitions):
        state = torch.as_tensor(transition.state, dtype=torch.float32).reshape(-1)
        if state.shape != (FEATURE_DIM,):
            raise ValueError(f"replay state at index {index} has shape {tuple(state.shape)}")
        if not torch.isfinite(state).all():
            raise ValueError(f"replay state at index {index} contains NaN or infinity")
        action = int(transition.action)
        if action < 0 or action >= N_ACTIONS:
            raise ValueError(f"replay action at index {index} is outside [0, {N_ACTIONS - 1}]")
        reward = float(transition.reward)
        if not np.isfinite(reward):
            raise ValueError(f"replay reward at index {index} is not finite")
        done = bool(transition.done)
        has_next_state = transition.next_state is not None and not done
        if has_next_state:
            next_state = torch.as_tensor(transition.next_state, dtype=torch.float32).reshape(-1)
            if next_state.shape != (FEATURE_DIM,):
                raise ValueError(f"replay next_state at index {index} has shape {tuple(next_state.shape)}")
            if not torch.isfinite(next_state).all():
                raise ValueError(f"replay next_state at index {index} contains NaN or infinity")
            next_states[index] = next_state
        mask = torch.as_tensor(transition.next_valid_mask, dtype=torch.bool).reshape(-1)
        if mask.shape != (N_ACTIONS,):
            raise ValueError(f"replay next_valid_mask at index {index} has shape {tuple(mask.shape)}")
        if done:
            mask = torch.zeros(N_ACTIONS, dtype=torch.bool)
        elif not has_next_state:
            raise ValueError(f"non-terminal replay transition {index} is missing next_state")
        elif not bool(mask.any()):
            raise ValueError(f"non-terminal replay transition {index} has no valid next actions")

        states[index] = state
        actions[index] = action
        rewards[index] = reward
        next_state_present[index] = has_next_state
        dones[index] = done
        next_valid_masks[index] = mask

    return {
        "format_version": REPLAY_FORMAT_VERSION,
        "capacity": int(replay_buffer.memory.maxlen),
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_state_present": next_state_present,
        "dones": dones,
        "next_valid_masks": next_valid_masks,
    }


def _restore_replay_buffer(payload: dict) -> ReplayBuffer:
    """Validate and restore a ReplayBuffer from compact checkpoint tensors."""
    if not isinstance(payload, dict):
        raise TypeError("replay_buffer payload is not a dictionary")
    version = int(payload.get("format_version", -1))
    if version != REPLAY_FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported replay format version {version}; expected {REPLAY_FORMAT_VERSION}"
        )
    capacity = int(payload.get("capacity", REPLAY_CAPACITY))
    if capacity != REPLAY_CAPACITY:
        raise ValueError(f"replay capacity {capacity} does not match {REPLAY_CAPACITY}")

    required = {
        "states",
        "actions",
        "rewards",
        "next_states",
        "next_state_present",
        "dones",
        "next_valid_masks",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"replay payload missing keys: {sorted(missing)}")

    raw_actions = torch.as_tensor(payload["actions"], device="cpu")
    raw_next_state_present = torch.as_tensor(payload["next_state_present"], device="cpu")
    raw_dones = torch.as_tensor(payload["dones"], device="cpu")
    raw_next_valid_masks = torch.as_tensor(payload["next_valid_masks"], device="cpu")
    if torch.is_floating_point(raw_actions) or raw_actions.dtype == torch.bool:
        raise ValueError("replay actions must use an integer tensor dtype")
    if raw_next_state_present.dtype != torch.bool:
        raise ValueError("replay next_state_present must use bool dtype")
    if raw_dones.dtype != torch.bool:
        raise ValueError("replay dones must use bool dtype")
    if raw_next_valid_masks.dtype != torch.bool:
        raise ValueError("replay next_valid_masks must use bool dtype")

    states = torch.as_tensor(payload["states"], dtype=torch.float32, device="cpu")
    actions = raw_actions.to(dtype=torch.long)
    rewards = torch.as_tensor(payload["rewards"], dtype=torch.float32, device="cpu")
    next_states = torch.as_tensor(payload["next_states"], dtype=torch.float32, device="cpu")
    next_state_present = raw_next_state_present
    dones = raw_dones
    next_valid_masks = raw_next_valid_masks

    if states.ndim != 2 or states.shape[1] != FEATURE_DIM:
        raise ValueError(f"replay states must have shape (n, {FEATURE_DIM}); got {tuple(states.shape)}")
    length = states.shape[0]
    if length > capacity or length > REPLAY_CAPACITY:
        raise ValueError(f"replay length {length} exceeds capacity {capacity}")
    expected_vector_shape = (length,)
    expected_state_shape = (length, FEATURE_DIM)
    expected_mask_shape = (length, N_ACTIONS)
    shape_checks = {
        "actions": (actions.shape, expected_vector_shape),
        "rewards": (rewards.shape, expected_vector_shape),
        "next_states": (next_states.shape, expected_state_shape),
        "next_state_present": (next_state_present.shape, expected_vector_shape),
        "dones": (dones.shape, expected_vector_shape),
        "next_valid_masks": (next_valid_masks.shape, expected_mask_shape),
    }
    for name, (actual, expected) in shape_checks.items():
        if actual != expected:
            raise ValueError(f"replay {name} must have shape {expected}; got {tuple(actual)}")
    if not torch.isfinite(states).all():
        raise ValueError("replay states contain NaN or infinity")
    if not torch.isfinite(rewards).all():
        raise ValueError("replay rewards contain NaN or infinity")
    if not torch.isfinite(next_states).all():
        raise ValueError("replay next_states contain NaN or infinity")
    if bool(((actions < 0) | (actions >= N_ACTIONS)).any()):
        raise ValueError(f"replay actions must be in [0, {N_ACTIONS - 1}]")
    if bool((~dones & ~next_state_present).any()):
        raise ValueError("non-terminal replay rows must have next_state_present=True")
    if bool((dones & next_state_present).any()):
        raise ValueError("terminal replay rows must have next_state_present=False")
    if bool((~dones & ~next_valid_masks.any(dim=1)).any()):
        raise ValueError("non-terminal replay rows must have at least one valid next action")
    if bool((dones & next_valid_masks.any(dim=1)).any()):
        raise ValueError("terminal replay rows must have all-false next_valid_masks")
        
    replay_buffer = ReplayBuffer(capacity)
    for index in range(length):
        done = bool(dones[index].item())
        replay_buffer.push(
            states[index].numpy().copy(),
            int(actions[index].item()),
            float(rewards[index].item()),
            None if done else next_states[index].numpy().copy(),
            done,
            next_valid_masks[index].numpy().copy(),
        )
    return replay_buffer

def _load_training_checkpoint(self, checkpoint_path: Path) -> None:
    """Restore compatible training state, failing clearly on incompatible files."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        required = {
            "feature_schema",
            "feature_dim",
            "artifact_version",
            "policy_state_dict",
            "target_state_dict",
            "optimizer_state_dict",
            "total_transitions",
            "optimization_steps",
            "completed_rounds",
            "epsilon",
        }
        missing = required - set(checkpoint)
        if missing:
            raise KeyError(f"missing checkpoint keys: {sorted(missing)}")
        _validate_artifact_metadata(checkpoint, checkpoint_path)
        _validate_finite_state_dict(checkpoint["policy_state_dict"], "checkpoint policy_state_dict")
        _validate_finite_state_dict(checkpoint["target_state_dict"], "checkpoint target_state_dict")
        total_transitions = int(checkpoint["total_transitions"])
        optimization_steps = int(checkpoint["optimization_steps"])
        completed_rounds = int(checkpoint["completed_rounds"])
        epsilon = float(checkpoint["epsilon"])
        if min(total_transitions, optimization_steps, completed_rounds) < 0:
            raise ValueError("checkpoint counters must be non-negative")
        if not np.isfinite(epsilon):
            raise ValueError("checkpoint epsilon must be finite")

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_transitions = total_transitions
        self.optimization_steps = optimization_steps
        self.completed_rounds = completed_rounds
        self.epsilon = epsilon
        if "replay_buffer" in checkpoint:
            self.replay_buffer = _restore_replay_buffer(checkpoint["replay_buffer"])
        if "replay_rng_state" in checkpoint:
            self.replay_rng.setstate(checkpoint["replay_rng_state"])
        if "action_rng_state" in checkpoint:
            self.rng.setstate(checkpoint["action_rng_state"])
    except Exception as exc:
        raise CheckpointError(f"Failed to load training checkpoint from {checkpoint_path}: {exc}") from exc
    self.policy_net.eval()
    self.target_net.eval()
    self.logger.info("Loaded DQN training checkpoint from %s", checkpoint_path)

def _atomic_torch_save(payload, destination: Path) -> None:
    """Write a torch file atomically by replacing from a sibling temporary file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as file:
        temp_path = Path(file.name)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
