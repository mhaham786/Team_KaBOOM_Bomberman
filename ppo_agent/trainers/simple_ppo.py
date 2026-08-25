import torch
from torch.distributions import Categorical

from .. import config
from .base import TrainerBase


def compute_returns(rewards, dones, gamma):
    returns = []
    discounted_return = 0.0

    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_return = 0.0
        discounted_return = reward + gamma * discounted_return
        returns.append(discounted_return)

    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


class PPOTrainer(TrainerBase):
    def update(self, transitions):
        states = torch.stack([transition["state"] for transition in transitions])
        actions = torch.tensor([transition["action"] for transition in transitions], dtype=torch.long)
        old_log_probs = torch.tensor(
            [transition["log_prob"] for transition in transitions], dtype=torch.float32
        )
        rewards = [transition["reward"] for transition in transitions]
        dones = [transition["done"] for transition in transitions]
        old_values = torch.tensor([transition["value"] for transition in transitions], dtype=torch.float32)

        returns = compute_returns(rewards, dones, config.GAMMA)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(config.UPDATE_EPOCHS):
            logits, values = self.model(states)
            values = values.squeeze(-1)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio,
                1.0 - config.CLIP_EPSILON,
                1.0 + config.CLIP_EPSILON,
            ) * advantages

            actor_loss = -torch.min(unclipped, clipped).mean()
            critic_loss = ((values - returns) ** 2).mean()
            loss = (
                actor_loss
                + config.VALUE_COEF * critic_loss
                - config.ENTROPY_COEF * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
