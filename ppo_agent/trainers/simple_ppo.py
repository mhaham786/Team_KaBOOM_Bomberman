import torch

from .base import TrainerBase


class PPOTrainer(TrainerBase):
    def compute_advantages(self, rewards, dones, values):
        from .. import config

        returns = []
        discounted_return = 0.0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_return = 0.0
            discounted_return = reward + config.GAMMA * discounted_return
            returns.append(discounted_return)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        return returns - values, returns
