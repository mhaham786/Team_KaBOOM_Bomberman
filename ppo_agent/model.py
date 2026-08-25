from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, observation_count, action_count):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(observation_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observations):
        return self.actor(observations), self.critic(observations)
