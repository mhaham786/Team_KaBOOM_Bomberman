import torch
from torch import nn
from torch.distributions import Categorical


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


def action_distribution(logits, masks=None):
    if masks is not None:
        masks = torch.as_tensor(masks, dtype=torch.bool, device=logits.device)
        if masks.shape != logits.shape or not masks.any(dim=-1).all():
            raise ValueError("Action mask must match logits with a valid action per row")
        logits = logits.masked_fill(~masks, -torch.inf)
    return Categorical(logits=logits)


def load_weights(model, state, task1=False):
    """Validate every tensor before loading; retain only the new BOMB row on transfer."""
    current = model.state_dict()
    if set(state) != set(current):
        raise ValueError("Checkpoint parameter names do not match the PPO model")
    output_keys = {'actor.4.weight', 'actor.4.bias'}
    for key, target in current.items():
        expected = (5, *target.shape[1:]) if task1 and key in output_keys else target.shape
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != expected:
            raise ValueError(f"Incompatible PPO checkpoint shape: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite PPO checkpoint parameter: {key}")
    for key in current:
        if task1 and key in output_keys:
            current[key][:5].copy_(state[key])
        else:
            current[key] = state[key]
    model.load_state_dict(current)
