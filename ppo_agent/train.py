from collections import Counter

from . import config
from .episode_buffer import EpisodeBuffer
from ..common.metrics import Task4Metrics
from ..common.rewards import task4_rewards_ppo
from .trainers import GAEPPOTrainer


def setup_training(self):
    self.trainer = (
        None
        if config.EVALUATION_ONLY
        else GAEPPOTrainer(self.model, self.optimizer)
    )
    self.run.create(
        {
            "description": getattr(config, "DESCRIPTION", ""),
            "evaluation_only": config.EVALUATION_ONLY,
            "initial_weights_experiment": config.INITIAL_WEIGHTS_EXPERIMENT,
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
    self.metrics = Task4Metrics()
    self.last_events = []
    self.episode = self.run.get_progress()
    self.round_start = 0


def game_events_occurred(
    self, old_game_state, self_action, new_game_state, events
):
    if not self.buffer.pending:
        return

    reward = task4_rewards_ppo(events)
    self.metrics.record_events(events, reward, old_game_state, new_game_state)
    self.buffer.finish(reward, False)
    self.last_events = list(events)


def end_of_round(self, last_game_state, last_action, events):
    if self.buffer.pending:
        reward = final_reward(self, events, last_game_state)
        self.buffer.finish(reward, True)
    elif len(self.buffer) > self.round_start:
        # Survivors already reported their final action; credit only new events.
        final_events = list((Counter(events) - Counter(self.last_events)).elements())
        reward = final_reward(self, final_events, last_game_state)
        self.buffer.rewards[-1] += reward
        self.buffer.dones[-1] = True
    else:
        return

    round_steps = len(self.buffer) - self.round_start
    update_model = (
        not config.EVALUATION_ONLY
        and len(self.buffer) >= config.ROLLOUT_STEPS
    )
    if update_model:
        self.trainer.update(self.buffer)

    self.episode += 1
    metric = self.metrics.to_dict(self.episode, round_steps)

    if not config.EVALUATION_ONLY:
        self.run.save_latest(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }
        )
    self.run.append_train_metric(metric)

    if update_model or config.EVALUATION_ONLY:
        self.buffer.reset()
    self.round_start = len(self.buffer)
    self.metrics.reset()
    self.last_events = []


def final_reward(self, events, game_state):
    """Return event and round-outcome rewards for the final transition."""
    event_reward = task4_rewards_ppo(events)
    self.metrics.record_events(events, event_reward, game_state)

    scores = self.metrics.scores
    best_score = max(scores.values())
    leaders = [name for name, score in scores.items() if score == best_score]
    if self.metrics.killed:
        outcome = False
    elif leaders == [self.metrics.agent_name]:
        outcome = True
    elif self.metrics.agent_name not in leaders:
        outcome = False
    else:
        outcome = None
    outcome_reward = task4_rewards_ppo((), outcome) if outcome is not None else 0.0
    self.metrics.reward += outcome_reward
    return event_reward + outcome_reward
