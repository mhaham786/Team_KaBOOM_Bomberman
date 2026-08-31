import events as e


class GeneralMetrics:

    def __init__(self):
        self.reset()

    def record_events(self, events, reward, game_state=None):
        self.reward += reward
        self.coins += events.count(e.COIN_COLLECTED)
        self.invalid_moves += events.count(e.INVALID_ACTION)
        self.kills += events.count(e.KILLED_OPPONENT)
        self.killed = self.killed or e.GOT_KILLED in events
        self.suicide = self.suicide or e.KILLED_SELF in events

    def to_dict(self, episode, steps):
        metric = {
            "episode": episode,
            "steps": steps,
            "score": self.coins + 5 * self.kills,
            "coins": self.coins,
            "kills": self.kills,
            "invalid_moves": self.invalid_moves,
            "killed": self.killed,
            "suicide": self.suicide,
            "episode_reward": self.reward,
        }
        if self.decision_count:
            metric.update(
                {
                    "decision_time_mean": self.decision_time_total / self.decision_count,
                    "decision_time_max": self.decision_time_max,
                }
            )
        return metric

    def record_decision_time(self, duration):
        self.decision_count += 1
        self.decision_time_total += duration
        self.decision_time_max = max(self.decision_time_max, duration)

    def reset(self):
        self.reward = 0.0
        self.coins = 0
        self.invalid_moves = 0
        self.kills = 0
        self.killed = False
        self.suicide = False
        self.decision_count = 0
        self.decision_time_total = 0.0
        self.decision_time_max = 0.0
