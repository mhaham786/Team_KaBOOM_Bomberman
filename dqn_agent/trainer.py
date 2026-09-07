import torch
import torch.nn.functional as F

from . import config
from .replay_buffer import transitions_to_tensors


class DQNTrainer:
    def __init__(self, policy_net, target_net, optimizer, device, replay_rng):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.device = device
        self.replay_rng = replay_rng
        self.optimization_steps = 0

    def update(self, replay_buffer, total_transitions):
        """Run one DQN update when enough replay data is available."""
        if len(replay_buffer) < max(config.BATCH_SIZE, config.MIN_REPLAY_SIZE):
            return None
        if config.COLLECT_ONLY:
            return None
        if total_transitions % config.TRAIN_EVERY != 0:
            return None

        transitions = self.replay_rng.sample(
            list(replay_buffer.memory),
            config.BATCH_SIZE,
        )
        states, actions, rewards, next_states, dones, masks = (
            transitions_to_tensors(transitions, self.device)
        )

        self.policy_net.train()
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.masked_fill(
                ~masks,
                torch.finfo(next_q_values.dtype).min,
            )
            future_values = next_q_values.max(dim=1, keepdim=True).values
            future_values = torch.where(
                dones,
                torch.zeros_like(future_values),
                future_values,
            )
            targets = rewards + config.GAMMA * future_values

        loss = F.smooth_l1_loss(q_values, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("DQN optimization produced a non-finite loss")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            config.GRADIENT_CLIP_NORM,
        )
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % config.TARGET_UPDATE_INTERVAL == 0:
            self.sync_target_network()

        self.policy_net.eval()
        return float(loss.detach().cpu().item())

    def sync_target_network(self):
        """Copy policy parameters into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
