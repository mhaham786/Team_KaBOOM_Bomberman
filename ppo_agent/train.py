from . import config
from .metrics import EpisodeMetrics
from .trainers.simple_ppo import PPOTrainer


def setup_training(self):
    self.trainer = PPOTrainer(self.model, self.optimizer)
    self.run.create(self.model, self.trainer)
    self.transitions = []
    self.pending_transition = None
    self.metrics = EpisodeMetrics()
    self.episode, _ = self.run.get_progress()
    self.eval_episode, self.best_score = self.run.get_eval_progress()


def game_events_occurred(
    self, old_game_state, self_action, new_game_state, events
):
    if self.evaluating:
        self.metrics.record_events(events, config.REWARDS.reward(events))
        return

    if self.pending_transition is None:
        return

    reward = config.REWARDS.reward(events)
    self.metrics.record_events(events, reward)
    self.pending_transition["reward"] = reward
    self.pending_transition["done"] = False
    self.transitions.append(self.pending_transition)
    self.pending_transition = None


def end_of_round(self, last_game_state, last_action, events):
    if self.evaluating:
        self.metrics.record_events(events, config.REWARDS.reward(events))
        self.eval_episode += 1
        metric = self.metrics.to_dict(self.eval_episode, last_game_state["step"])
        metric["training_episode"] = self.episode
        self.run.append_eval_metric(metric)
        if metric["score"] > self.best_score:
            self.run.save_best(self.model)
            self.best_score = metric["score"]
        self.metrics.reset()
        self.model.train()
        self.evaluating = False
        return

    if self.pending_transition is not None:
        reward = config.REWARDS.reward(events)
        self.metrics.record_events(events, reward)
        self.pending_transition["reward"] = reward
        self.pending_transition["done"] = True
        self.transitions.append(self.pending_transition)
    elif self.transitions:
        self.transitions[-1]["done"] = True
    else:
        return

    self.trainer.update(self.transitions)
    self.episode += 1
    metric = self.metrics.to_dict(self.episode, len(self.transitions))

    self.run.save_latest(self.model, self.optimizer)
    self.run.append_train_metric(metric)

    self.transitions.clear()
    self.pending_transition = None
    self.metrics.reset()

    if config.EVAL_EVERY and self.episode % config.EVAL_EVERY == 0:
        self.model.eval()
        self.evaluating = True
