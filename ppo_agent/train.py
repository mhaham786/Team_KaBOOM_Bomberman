from collections import Counter

import events as e

from . import config
from .episode_buffer import EpisodeBuffer
from ..common.metrics import Task3Metrics
from .trainers import GAEPPOTrainer, PPOTrainer

from ..common.helpers import bomb_effects_from


REWARDS = {e.COIN_COLLECTED: 10, e.CRATE_DESTROYED: 2, e.COIN_FOUND: 1,
           e.SURVIVED_ROUND: 5, e.INVALID_ACTION: -5,
           e.KILLED_SELF: -50, e.GOT_KILLED: -40,
           e.KILLED_OPPONENT: 50}


def task3_reward(events, state):
    reward = sum(REWARDS.get(event, 0) for event in events)
    if e.BOMB_DROPPED in events:
        others_positions = [other[3] for other in state['others']]
        

        crates, opponents_hit = bomb_effects_from(tuple(state['self'][3]), state['field'], others_positions)
        

        if not crates and not opponents_hit:
            reward -= 1
            
    return float(reward)


def setup_training(self):
    self.trainer = GAEPPOTrainer(self.model, self.optimizer)
    self.run.create(
        {
            "description": getattr(config, "DESCRIPTION", ""),
            "actions": list(config.ACTIONS),
            "config": {
                name.lower(): value
                for name, value in vars(config).items()
                if name.isupper() and isinstance(value, int | float)
            },
            "model_structure": str(self.model),
        }
    )
    self.buffer = EpisodeBuffer()
    self.metrics = Task3Metrics()
    self.last_events = []
    self.episode = self.run.get_progress()


def game_events_occurred(
    self, old_game_state, self_action, new_game_state, events
):
    if not self.buffer.pending:
        return

    reward = task3_reward(events, old_game_state)
    self.metrics.record_events(events, reward, old_game_state, new_game_state)
    self.buffer.finish(reward, False)
    self.last_events = list(events)


def end_of_round(self, last_game_state, last_action, events):
    if self.buffer.pending:
        reward = task3_reward(events, last_game_state)
        self.metrics.record_events(events, reward, last_game_state)
        self.buffer.finish(reward, True)
    elif self.buffer.states:
        # Survivors already reported their final action; credit only new events.
        final_events = list((Counter(events) - Counter(self.last_events)).elements())
        reward = task3_reward(final_events, last_game_state)
        self.metrics.record_events(final_events, reward, last_game_state)
        self.buffer.rewards[-1] += reward
        self.buffer.dones[-1] = True
    else:
        return

    self.trainer.update(self.buffer)
    self.episode += 1
    metric = self.metrics.to_dict(self.episode, len(self.buffer))

    self.run.save_latest(
        {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
    )
    self.run.append_train_metric(metric)

    self.buffer.reset()
    self.metrics.reset()
    self.last_events = []
