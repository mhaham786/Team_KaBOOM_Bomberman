from time import perf_counter

import torch

from ..common.experiment import ExperimentRun
from ..common.features import competitive_features_oc54
from ..common.helpers import action_mask
from . import config
from .model import ActorCritic, action_distribution, load_weights

EVALUATION_SEED = 71042
EVALUATION_TEMPERATURE = 0.75


def setup(self):
    torch.set_num_threads(1)
    torch.manual_seed(config.RANDOM_SEED)
    self.model = ActorCritic(config.OBSERVATION_COUNT, len(config.ACTIONS))
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
    self.run = ExperimentRun(config.EXPERIMENTS_DIR, config.EXPERIMENT_NAME)

    metrics_only = self.train and config.EVALUATION_ONLY

    if self.train:
        if config.RESUME_TRAINING and config.INITIAL_WEIGHTS_EXPERIMENT is not None:
            raise ValueError(
                "RESUME_TRAINING and INITIAL_WEIGHTS_EXPERIMENT cannot both be set."
            )
        if config.INITIAL_WEIGHTS_EXPERIMENT == config.EXPERIMENT_NAME:
            raise ValueError(
                "INITIAL_WEIGHTS_EXPERIMENT must differ from EXPERIMENT_NAME."
            )
        if config.RESUME_TRAINING and config.RESTART_EXPERIMENT:
            raise ValueError(
                "RESUME_TRAINING and RESTART_EXPERIMENT cannot both be True."
            )
        if config.RESTART_EXPERIMENT:
            self.run.restart()
        if config.RESUME_TRAINING:
            checkpoint = self.run.load_latest()
            if checkpoint is None:
                raise FileNotFoundError(
                    f"Cannot resume {config.EXPERIMENT_NAME}: latest.pt does not exist."
                )
        elif self.run.path.exists():
            raise FileExistsError(
                f"Experiment {config.EXPERIMENT_NAME} already exists. "
                "Use a new EXPERIMENT_NAME or set RESUME_TRAINING = True."
            )
        else:
            checkpoint = None
        if metrics_only:
            self.model.eval()
        else:
            self.model.train()
    else:
        checkpoint = self.run.load_latest()
        if checkpoint is None:
            raise FileNotFoundError(f"Missing checkpoint: {self.run.latest_path}")
        self.model.eval()

    if checkpoint is not None:
        load_weights(self.model, checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.logger.info("Loaded latest.pt from experiment %s.", config.EXPERIMENT_NAME)
    elif self.train and config.INITIAL_WEIGHTS_EXPERIMENT is not None:
        source_run = ExperimentRun(
            config.EXPERIMENTS_DIR,
            config.INITIAL_WEIGHTS_EXPERIMENT,
        )
        initial_checkpoint = source_run.load_latest()
        if initial_checkpoint is None:
            raise FileNotFoundError(
                f"Missing initial checkpoint: {source_run.latest_path}"
            )
        load_weights(self.model, initial_checkpoint["model_state"])
        if config.LOAD_INITIAL_OPTIMIZER:
            self.optimizer.load_state_dict(initial_checkpoint["optimizer_state"])
        self.logger.info(
            "Initialized policy%s from experiment %s.",
            " and optimizer" if config.LOAD_INITIAL_OPTIMIZER else "",
            config.INITIAL_WEIGHTS_EXPERIMENT,
        )
    else:
        self.logger.info(
            "Initialized a fresh fixed-feature network for experiment %s.",
            config.EXPERIMENT_NAME,
        )
    if not self.train or metrics_only:
        self.eval_generator = torch.Generator().manual_seed(EVALUATION_SEED)


def act(self, game_state):
    start_time = perf_counter() if self.train else None
    learning = self.train and not config.EVALUATION_ONLY
    state = torch.tensor(competitive_features_oc54(game_state), dtype=torch.float32)
    mask = torch.from_numpy(action_mask(game_state))

    with torch.no_grad():
        logits, value = self.model(state)
        policy_logits = logits if learning else logits / EVALUATION_TEMPERATURE
        distribution = action_distribution(policy_logits, mask)
        if learning:
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        else:
            action = torch.multinomial(
                distribution.probs, 1, generator=self.eval_generator
            ).squeeze(0)

    action_index = int(action.item())
    if self.train:
        self.metrics.record_decision_time(perf_counter() - start_time)
        stored_log_prob = distribution.log_prob(action).item()
        self.buffer.add(state, action_index, stored_log_prob, value.item(), mask)

    return config.ACTIONS[action_index]
