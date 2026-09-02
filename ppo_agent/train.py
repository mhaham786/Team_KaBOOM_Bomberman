from . import config
from .episode_buffer import EpisodeBuffer
from ..common.metrics import Task1Metrics
from .trainers import GAEPPOTrainer, PPOTrainer

from ..common.rewards import *


def setup_training(self):
    self.trainer = PPOTrainer(self.model, self.optimizer)
    self.run.create(self.model)
    self.buffer = EpisodeBuffer()
    self.metrics = Task1Metrics()
    self.episode, _ = self.run.get_progress()


def game_events_occurred(
    self, old_game_state, self_action, new_game_state, events
):
    if not self.buffer.pending:
        return

    reward = coin_heaven_rewards_ppo_improved(events)
    self.metrics.record_events(events, reward, old_game_state, new_game_state)
    self.buffer.finish(reward, False)


def end_of_round(self, last_game_state, last_action, events):
    if self.buffer.pending:
        reward = coin_heaven_rewards_ppo_improved(events)
        self.metrics.record_events(events, reward, last_game_state)
        self.buffer.finish(reward, True)
    elif self.buffer.states:
        self.buffer.dones[-1] = True
    else:
        return

    self.trainer.update(self.buffer)
    self.episode += 1
    metric = self.metrics.to_dict(self.episode, len(self.buffer))

    self.run.save_latest(self.model, self.optimizer)
    self.run.append_train_metric(metric)

    self.buffer.reset()
    self.metrics.reset()
