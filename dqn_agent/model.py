from torch import nn


class DQN(nn.Module):
    """Estimate one Q-value for each available action."""

    def __init__(self, observation_count, action_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_count, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, states):
        if states.ndim == 1:
            states = states.unsqueeze(0)
        return self.network(states.float())
