from time import perf_counter

import torch

from ..common.experiment import ExperimentRun
from . import config
from .model import ActorCritic, action_distribution, load_weights
from .task2 import loot_crate_bfs_oc9, task2_action_mask

EVALUATION_SEED = 71042
EVALUATION_TEMPERATURE = 0.75


def setup(self):
    torch.set_num_threads(1)
    torch.manual_seed(config.RANDOM_SEED)
    self.model = ActorCritic(config.OBSERVATION_COUNT, len(config.ACTIONS))
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
    self.run = ExperimentRun(config.EXPERIMENTS_DIR, config.EXPERIMENT_NAME)

    if self.train:
        if config.RESUME_TRAINING and config.RESTART_EXPERIMENT:
            raise ValueError(
                "RESUME_TRAINING and RESTART_EXPERIMENT cannot both be True."
            )
        if config.RESTART_EXPERIMENT:
            raise ValueError("Use a new Task 2 experiment name; existing runs are preserved.")
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
        self.model.train()
    else:
        checkpoint = self.run.load_latest()
        if checkpoint is None:
            raise FileNotFoundError(f"Missing Task 2 checkpoint: {self.run.latest_path}")
        self.model.eval()

    if checkpoint is not None:
        load_weights(self.model, checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.logger.info("Loaded latest.pt from experiment %s.", config.EXPERIMENT_NAME)
    else:
        checkpoint = torch.load(config.TASK1_CHECKPOINT, map_location="cpu", weights_only=False)
        load_weights(self.model, checkpoint["model_state"], task1=True)
        self.logger.info("Transferred Task 1 weights; initialized only BOMB output and fresh optimizer.")
    if not self.train:
        self.eval_generator = torch.Generator().manual_seed(EVALUATION_SEED)


def act(self, game_state):
    start_time = perf_counter() if self.train else None
    state = torch.tensor(loot_crate_bfs_oc9(game_state), dtype=torch.float32)
    mask = torch.from_numpy(task2_action_mask(game_state))

    with torch.no_grad():
        logits, value = self.model(state)
        policy_logits = logits if self.train else logits / EVALUATION_TEMPERATURE
        distribution = action_distribution(policy_logits, mask)
        if self.train:
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        else:
            action = torch.multinomial(
                distribution.probs, 1, generator=self.eval_generator
            ).squeeze(0)

    action_index = int(action.item())
    if self.train:
        self.metrics.record_decision_time(perf_counter() - start_time)
        self.buffer.add(state, action_index, log_prob.item(), value.item(), mask)

    return config.ACTIONS[action_index]
