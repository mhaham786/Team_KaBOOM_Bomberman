import random
from time import perf_counter

import numpy as np
import torch

from ..common.experiment import ExperimentRun
from ..common.features import *
from ..common.helpers import action_mask
from . import config
from .experiment import load_policy
from .model import DQN


def setup(self):
    self.device = torch.device(
        "cuda" if self.train and torch.cuda.is_available() else "cpu"
    )
    self.policy_net = DQN(
        config.OBSERVATION_COUNT,
        len(config.ACTIONS),
    ).to(self.device)
    self.run = ExperimentRun(config.EXPERIMENTS_DIR, config.EXPERIMENT_NAME)
    self.rng = random.Random()
    self.epsilon = config.EPSILON_START if self.train else 0.0

    if not self.train:
        checkpoint = self.run.load_latest()
        if checkpoint is None:
            self.logger.warning(
                "No latest.pt found for experiment %s.",
                config.EXPERIMENT_NAME,
            )
        else:
            load_policy(self.policy_net, checkpoint)
            self.logger.info(
                "Loaded latest.pt from experiment %s.",
                config.EXPERIMENT_NAME,
            )

    self.policy_net.eval()


def act(self, game_state):
    start_time = perf_counter() if self.train else None
    features = advanced_features_oc31(game_state)
    mask = action_mask(game_state)

    if self.train and self.rng.random() < self.epsilon:
        action_index = sample_legal_action(mask, self.rng)
    else:
        action_index = select_greedy_action(
            self.policy_net,
            features,
            mask,
            self.device,
        )

    if self.train:
        self.metrics.record_decision_time(perf_counter() - start_time)
    return config.ACTIONS[action_index]


def sample_legal_action(mask, rng):
    legal_actions = np.flatnonzero(mask).tolist()
    return int(rng.choice(legal_actions))


def select_greedy_action(policy_net, features, mask, device):
    state = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)

    with torch.no_grad():
        q_values = policy_net(state).squeeze(0)
        q_values = q_values.masked_fill(
            ~mask,
            torch.finfo(q_values.dtype).min,
        )
    return int(torch.argmax(q_values).item())
