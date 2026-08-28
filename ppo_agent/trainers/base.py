import torch
from torch.distributions import Categorical


class TrainerBase:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def update(self, buffer):
        from .. import config

        states = torch.stack(buffer.states)
        actions = torch.tensor(buffer.actions, dtype=torch.long)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
        old_values = torch.tensor(buffer.values, dtype=torch.float32)

        advantages, returns = self.compute_advantages(
            buffer.rewards, buffer.dones, old_values
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(config.UPDATE_EPOCHS):
            indices = torch.randperm(len(buffer))

            for start in range(0, len(buffer), config.MINIBATCH_SIZE):
                batch = indices[start : start + config.MINIBATCH_SIZE]
                self.update_minibatch(
                    states[batch],
                    actions[batch],
                    old_log_probs[batch],
                    advantages[batch],
                    returns[batch],
                )

    def update_minibatch(self, states, actions, old_log_probs, advantages, returns):
        from .. import config

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

    def compute_advantages(self, rewards, dones, values):
        raise NotImplementedError
