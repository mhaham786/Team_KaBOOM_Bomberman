import torch

from .base import TrainerBase


class GAEPPOTrainer(TrainerBase):
    def compute_advantages(self, rewards, dones, values):
        from .. import config

        rewards = torch.tensor(rewards, dtype=torch.float32)
        advantages = torch.zeros_like(values)
        gae = 0.0
        next_value = 0.0

        for index in reversed(range(len(rewards))):
            if dones[index]:
                next_value = 0.0
                gae = 0.0

            delta = rewards[index] + config.GAMMA * next_value - values[index]
            gae = delta + config.GAMMA * config.GAE_LAMBDA * gae
            advantages[index] = gae
            next_value = values[index]

        return advantages, advantages + values
