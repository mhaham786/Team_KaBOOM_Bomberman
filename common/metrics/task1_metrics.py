import events as e

from ..helpers import bfs_first_step, bomb_positions, opponent_positions
from .general_metrics import GeneralMetrics


class Task1Metrics(GeneralMetrics):

    def record_events(self, events, reward, game_state=None):
        super().record_events(events, reward, game_state)
        if game_state is not None:
            self.record_path_progress(game_state, events)

    def record_path_progress(self, game_state, events):
        self.path_tracking = True
        if self.needs_target_distance:
            field = game_state["field"]
            position = tuple(game_state["self"][3])
            _, distance = bfs_first_step(
                field,
                position,
                game_state.get("coins", ()),
                bomb_positions(game_state.get("bombs", ())),
                opponent_positions(game_state.get("others", ())),
            )
            self.target_distance = distance
            self.needs_target_distance = False

        if e.COIN_COLLECTED in events:
            if self.target_distance is not None:
                self.optimal_distance += self.target_distance
            self.needs_target_distance = True

    def to_dict(self, episode, steps):
        metric = super().to_dict(episode, steps)
        if self.path_tracking and self.coins:
            metric["path_efficiency"] = self.optimal_distance / max(1, steps)
        return metric

    def reset(self):
        super().reset()
        self.path_tracking = False
        self.optimal_distance = 0
        self.target_distance = None
        self.needs_target_distance = True
