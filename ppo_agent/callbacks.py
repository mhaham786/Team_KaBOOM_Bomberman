from time import perf_counter

import torch
from torch.distributions import Categorical

from . import config
from .experiment import ExperimentRun
from .model import ActorCritic

from ..common.features import *


def setup(self):
    self.model = ActorCritic(config.OBSERVATION_COUNT, len(config.ACTIONS))
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
    self.run = ExperimentRun(config.EXPERIMENT_NAME)

    if self.train:
        if config.RESUME_TRAINING and config.RESTART_EXPERIMENT:
            raise ValueError(
                "RESUME_TRAINING and RESTART_EXPERIMENT cannot both be True."
            )
        if config.RESTART_EXPERIMENT:
            self.run.restart()
            loaded = False
        elif config.RESUME_TRAINING:
            loaded = self.run.load_latest(self.model, self.optimizer)
            if not loaded:
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
            loaded = False
        self.model.train()
        checkpoint_name = "latest.pt"
    else:
        loaded = self.run.load_latest(self.model, self.optimizer)
        self.model.eval()
        checkpoint_name = "latest.pt"

    if loaded:
        self.logger.info("Loaded %s from experiment %s.", checkpoint_name, config.EXPERIMENT_NAME)
    else:
        self.logger.info("No %s found for experiment %s.", checkpoint_name, config.EXPERIMENT_NAME)


def act(self, game_state):
    start_time = perf_counter() if self.train else None
    state = torch.tensor(coin_heaven_minimal_oc4(game_state), dtype=torch.float32)

    with torch.no_grad():
        logits, value = self.model(state)
        if self.train:
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        else:
            action = torch.argmax(logits)

    action_index = int(action.item())
    if self.train:
        self.metrics.record_decision_time(perf_counter() - start_time)
        self.buffer.add(state, action_index, log_prob.item(), value.item())

    return config.ACTIONS[action_index]
