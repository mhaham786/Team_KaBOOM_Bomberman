"""Create and restore DQN-specific experiment data."""

from ..common.experiment import ExperimentRun
from . import config
from .replay_buffer import ReplayBuffer


def create_experiment(self):
    """Create the configured DQN experiment."""
    self.run.create(
        {
            "description": config.DESCRIPTION,
            "actions": list(config.ACTIONS),
            "config": {
                name.lower(): value
                for name, value in vars(config).items()
                if name.isupper() and isinstance(value, int | float | bool | str)
            },
            "model_structure": str(self.policy_net),
        }
    )


def save_checkpoint(self):
    """Save all state needed to continue DQN training."""
    self.run.save_latest(
        {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "replay_memory": list(self.replay_buffer.memory),
            "total_transitions": self.total_transitions,
            "optimization_steps": self.trainer.optimization_steps,
            "episode": self.episode,
            "epsilon": self.epsilon,
            "action_random_state": self.rng.getstate(),
            "replay_random_state": self.replay_rng.getstate(),
        }
    )


def load_policy(policy_net, checkpoint):
    """Load the policy network from a DQN checkpoint."""
    policy_net.load_state_dict(checkpoint["policy_state"])


def load_initial_policy(policy_net, experiment_name):
    """Initialize a policy from another experiment's latest checkpoint."""
    source_run = ExperimentRun(config.EXPERIMENTS_DIR, experiment_name)
    checkpoint = source_run.load_latest()
    if checkpoint is None:
        raise FileNotFoundError(
            f"Cannot initialize from {experiment_name}: latest.pt does not exist."
        )

    try:
        load_policy(policy_net, checkpoint)
    except RuntimeError as error:
        raise ValueError(
            f"Initial weights from {experiment_name} are incompatible with "
            "the configured DQN model."
        ) from error


def load_checkpoint(self, checkpoint):
    """Restore all state needed to continue DQN training."""
    self.policy_net.load_state_dict(checkpoint["policy_state"])
    self.target_net.load_state_dict(checkpoint["target_state"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    self.replay_buffer = ReplayBuffer(config.REPLAY_CAPACITY)
    self.replay_buffer.memory.extend(checkpoint["replay_memory"])
    self.total_transitions = checkpoint["total_transitions"]
    self.trainer.optimization_steps = checkpoint["optimization_steps"]
    self.episode = checkpoint["episode"]
    self.epsilon = checkpoint["epsilon"]
    self.rng.setstate(checkpoint["action_random_state"])
    self.replay_rng.setstate(checkpoint["replay_random_state"])
