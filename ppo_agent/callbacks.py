import torch
from torch.distributions import Categorical
from time import perf_counter

from . import config
from .experiment import ExperimentRun
from .model import ActorCritic


def setup(self):
    self.model = ActorCritic(config.OBSERVATION_COUNT, len(config.ACTIONS))
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
    self.run = ExperimentRun(config.EXPERIMENT_NAME)
    self.evaluating = False

    if self.train:
        if config.RESUME_TRAINING:
            loaded = self.run.load_latest(self.model, self.optimizer)
            if not loaded:
                raise FileNotFoundError(
                    f"Cannot resume {config.EXPERIMENT_NAME}: latest.pt does not exist."
                )
        elif self.run.path.exists():
            raise FileExistsError(
                f"Experiment {config.EXPERIMENT_NAME} already exists. "
                "Use a new EXPERIMENT_NAME or set RESUME_TRAINING = True."
            )
        else:
            loaded = False
        self.model.train()
        checkpoint_name = "latest.pt"
    else:
        loaded = self.run.load_best(self.model)
        self.model.eval()
        checkpoint_name = "best.pt"

    if loaded:
        self.logger.info("Loaded %s from experiment %s.", checkpoint_name, config.EXPERIMENT_NAME)
    else:
        self.logger.info("No %s found for experiment %s.", checkpoint_name, config.EXPERIMENT_NAME)


def act(self, game_state):
    if self.evaluating:
        start_time = perf_counter()

    state = torch.tensor(config.FEATURES.encode(game_state), dtype=torch.float32)

    with torch.no_grad():
        logits, value = self.model(state)
        if self.train and not self.evaluating:
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        else:
            action = torch.argmax(logits)

    action_index = int(action.item())
    if self.train and not self.evaluating:
        self.pending_transition = {
            "state": state,
            "action": action_index,
            "log_prob": log_prob.item(),
            "value": value.item(),
        }
    elif self.evaluating:
        self.metrics.record_decision_time(perf_counter() - start_time)

    return config.ACTIONS[action_index]
