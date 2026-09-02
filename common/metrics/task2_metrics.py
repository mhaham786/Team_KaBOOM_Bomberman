import events as e

from .general_metrics import GeneralMetrics


class Task2Metrics(GeneralMetrics):

    def record_events(
        self,
        events,
        reward,
        old_game_state=None,
        new_game_state=None,
    ):
        super().record_events(events, reward, old_game_state, new_game_state)
        self.crates_destroyed += events.count(e.CRATE_DESTROYED)
        self.bombs_dropped += events.count(e.BOMB_DROPPED)

        exploded = events.count(e.BOMB_EXPLODED)
        if not exploded:
            return

        self.bombs_exploded += exploded
        useful = min(exploded, int(e.CRATE_DESTROYED in events))
        self.useful_bombs += useful
        self.useless_bombs += exploded - useful
        if e.GOT_KILLED not in events and e.KILLED_SELF not in events:
            self.successful_escapes += exploded

    def to_dict(self, episode, steps):
        metric = super().to_dict(episode, steps)
        metric.update(
            {
                "crates_destroyed": self.crates_destroyed,
                "bombs_dropped": self.bombs_dropped,
                "bombs_exploded": self.bombs_exploded,
                "successful_escapes": self.successful_escapes,
                "crates_per_bomb": self.crates_destroyed / max(1, self.bombs_dropped),
                "escape_success_rate": self.successful_escapes / max(1, self.bombs_exploded),
                "useful_bombs": self.useful_bombs,
                "useless_bombs": self.useless_bombs,
            }
        )
        return metric

    def reset(self):
        super().reset()
        self.crates_destroyed = 0
        self.bombs_dropped = 0
        self.bombs_exploded = 0
        self.useful_bombs = 0
        self.useless_bombs = 0
        self.successful_escapes = 0
