import random

import numpy as np
import torch

from ..common.metrics import Task2Metrics
from . import config
from .experiment import (
    create_experiment,
    load_checkpoint,
    load_initial_policy,
    save_checkpoint,
)
from .model import DQN
from .replay_buffer import ReplayBuffer
from .trainer import DQNTrainer
from .transition import Transition


def setup_training(self):
    """Initialize the configured DQN training run."""
    if config.RESUME_TRAINING and config.RESTART_EXPERIMENT:
        raise ValueError(
            "RESUME_TRAINING and RESTART_EXPERIMENT cannot both be True."
        )
    if config.RESUME_TRAINING and config.INITIAL_WEIGHTS_EXPERIMENT is not None:
        raise ValueError(
            "RESUME_TRAINING and INITIAL_WEIGHTS_EXPERIMENT cannot both be set."
        )
    if config.INITIAL_WEIGHTS_EXPERIMENT == config.EXPERIMENT_NAME:
        raise ValueError(
            "INITIAL_WEIGHTS_EXPERIMENT must differ from EXPERIMENT_NAME."
        )

    if config.RESTART_EXPERIMENT:
        self.run.restart()
        checkpoint = None
    elif config.RESUME_TRAINING:
        checkpoint = self.run.load_latest()
        if checkpoint is None:
            raise FileNotFoundError(
                f"Cannot resume {config.EXPERIMENT_NAME}: latest.pt does not exist."
            )
    elif self.run.path.exists():
        raise FileExistsError(
            f"Experiment {config.EXPERIMENT_NAME} already exists. "
            "Use a new EXPERIMENT_NAME, set RESUME_TRAINING = True, or "
            "set RESTART_EXPERIMENT = True."
        )
    else:
        checkpoint = None

    torch.manual_seed(config.RANDOM_SEED)
    self.rng.seed(config.RANDOM_SEED)
    self.replay_rng = random.Random(config.RANDOM_SEED)

    if checkpoint is None:
        reset_network_parameters(self.policy_net)
        if config.INITIAL_WEIGHTS_EXPERIMENT is not None:
            load_initial_policy(
                self.policy_net,
                config.INITIAL_WEIGHTS_EXPERIMENT,
            )
            self.logger.info(
                "Initialized policy weights from experiment %s.",
                config.INITIAL_WEIGHTS_EXPERIMENT,
            )

    self.target_net = DQN(
        config.OBSERVATION_COUNT,
        len(config.ACTIONS),
    ).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(
        self.policy_net.parameters(),
        lr=config.LEARNING_RATE,
    )
    self.replay_buffer = ReplayBuffer(config.REPLAY_CAPACITY)
    self.trainer = DQNTrainer(
        self.policy_net,
        self.target_net,
        self.optimizer,
        self.device,
        self.replay_rng,
    )
    self.metrics = Task2Metrics()
    self.transition = Transition(self)
    self.total_transitions = 0
    self.episode = 0
    self.epsilon = self.transition.epsilon_for(self.total_transitions)
    reset_round(self)

    create_experiment(self)
    if checkpoint is not None:
        load_checkpoint(self, checkpoint)
        if self.run.get_progress() != self.episode:
            raise ValueError("checkpoint episode does not match train.jsonl")

    self.policy_net.eval()


def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events,
):
    self.transition.record(
        old_game_state,
        self_action,
        new_game_state,
        events,
    )


def end_of_round(self, last_game_state, last_action, events):
    self.transition.finish(
        last_game_state,
        last_action,
        events,
    )
    self.episode += 1

    metric = self.metrics.to_dict(self.episode, self.round_steps)
    metric.update(
        {
            "total_reward": self.round_reward,
            "event_reward": self.round_event_reward,
            "shaping_reward": self.round_shaping_reward,
            "epsilon": self.epsilon,
            "replay_size": len(self.replay_buffer),
            "mean_loss": float(np.mean(self.round_losses))
            if self.round_losses
            else 0.0,
        }
    )
    self.run.append_train_metric(metric)

    if self.episode % config.SAVE_EVERY_ROUNDS == 0:
        save_checkpoint(self)

    reset_round(self)
    self.policy_net.eval()


def reset_round(self):
    """Reset metrics accumulated for one round."""
    self.round_steps = 0
    self.round_reward = 0.0
    self.round_event_reward = 0.0
    self.round_shaping_reward = 0.0
    self.round_losses = []
    self.metrics.reset()


def reset_network_parameters(policy_net):
    """Reset every trainable layer in a policy network."""
    for module in policy_net.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
