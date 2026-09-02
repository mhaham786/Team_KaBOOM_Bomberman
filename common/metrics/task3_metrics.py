import events as e

from ..helpers import (
    blast_tiles,
    bomb_positions,
    exploding_opponent_bombs,
    nearest_opponent_distance,
    opponent_positions,
    safe_adjacent_tile_count,
)
from .general_metrics import GeneralMetrics


class Task3Metrics(GeneralMetrics):

    def record_events(
        self,
        events,
        reward,
        old_game_state=None,
        new_game_state=None,
    ):
        super().record_events(events, reward, old_game_state, new_game_state)
        if old_game_state is None:
            return

        self.record_distance(old_game_state)
        self.record_opponent_coins(old_game_state, new_game_state)
        self.record_opponent_bomb_outcomes(old_game_state, events)
        self.record_own_bomb_outcomes(events)

        if e.BOMB_DROPPED in events:
            self.record_bomb_placement(old_game_state, new_game_state)

        if e.KILLED_OPPONENT in events and self.steps_to_kill is None:
            self.steps_to_kill = old_game_state.get("step")

    def record_bomb_placement(self, old_game_state, new_game_state):
        field = old_game_state["field"]
        origin = tuple(old_game_state["self"][3])
        others = old_game_state.get("others", ())
        opponents = opponent_positions(others)
        blast = set(blast_tiles(origin, field))
        threatened = opponents & blast

        self.bombs_dropped += 1
        self.own_bomb_positions.add(origin)
        self.pending_offensive_bombs.append(bool(threatened))

        if not threatened:
            return

        self.offensive_bombs += 1

        bombs = bomb_positions(new_game_state.get("bombs", ()))
        safe_tiles = [
            safe_adjacent_tile_count(
                field,
                opponent,
                blast,
                bombs,
                opponents - {opponent},
            )
            for opponent in threatened
        ]
        self.safe_adjacent_tiles_total += min(safe_tiles)

    def record_own_bomb_outcomes(self, events):
        exploded = events.count(e.BOMB_EXPLODED)
        for _ in range(exploded):
            offensive = (
                self.pending_offensive_bombs.pop(0)
                if self.pending_offensive_bombs
                else False
            )
            if offensive:
                self.offensive_bomb_kills += int(e.KILLED_OPPONENT in events)
                self.offensive_bomb_suicides += int(e.KILLED_SELF in events)

        if exploded:
            self.own_bomb_positions.clear()

    def record_distance(self, game_state):
        position = tuple(game_state["self"][3])
        opponents = opponent_positions(game_state.get("others", ()))
        distance = nearest_opponent_distance(position, opponents)
        if distance is None:
            return

        self.opponent_distance_total += distance
        self.opponent_distance_samples += 1

    def record_opponent_bomb_outcomes(self, game_state, events):
        field = game_state["field"]
        position = tuple(game_state["self"][3])
        exploding_bombs = exploding_opponent_bombs(
            game_state.get("bombs", ()),
            self.own_bomb_positions,
        )
        threats = sum(
            position in blast_tiles(bomb, field)
            for bomb in exploding_bombs
        )

        killed_by_opponent = e.GOT_KILLED in events and e.KILLED_SELF not in events
        if killed_by_opponent:
            self.killed_by_opponent_bomb += 1
        else:
            self.successful_opponent_bomb_escapes += threats

    def record_opponent_coins(self, old_game_state, new_game_state):
        if new_game_state is None:
            return

        old_scores = {
            name: score for name, score, _, _ in old_game_state.get("others", ())
        }
        for name, score, _, _ in new_game_state.get("others", ()):
            score_increase = max(0, score - old_scores.get(name, score))
            self.opponent_coins += score_increase % 5

    def to_dict(self, episode, steps):
        metric = super().to_dict(episode, steps)
        metric.update(
            {
                "bombs_dropped": self.bombs_dropped,
                "offensive_bombs": self.offensive_bombs,
                "offensive_bomb_ratio": self.offensive_bombs
                / max(1, self.bombs_dropped),
                "offensive_bomb_kills": self.offensive_bomb_kills,
                "offensive_bomb_suicides": self.offensive_bomb_suicides,
                "offensive_bomb_kill_ratio": self.offensive_bomb_kills
                / max(1, self.offensive_bombs),
                "offensive_bomb_suicide_ratio": self.offensive_bomb_suicides
                / max(1, self.offensive_bombs),
                "average_safe_adjacent_tiles": self.safe_adjacent_tiles_total
                / max(1, self.offensive_bombs),
                "average_opponent_distance": self.opponent_distance_total
                / max(1, self.opponent_distance_samples),
                "killed_by_opponent_bomb": self.killed_by_opponent_bomb,
                "successful_opponent_bomb_escapes": self.successful_opponent_bomb_escapes,
                "opponent_bomb_escape_rate": self.successful_opponent_bomb_escapes
                / max(
                    1,
                    self.successful_opponent_bomb_escapes
                    + self.killed_by_opponent_bomb,
                ),
                "opponent_coins": self.opponent_coins,
            }
        )
        if self.steps_to_kill is not None:
            metric["steps_to_kill"] = self.steps_to_kill
        return metric

    def reset(self):
        super().reset()
        self.bombs_dropped = 0
        self.offensive_bombs = 0
        self.offensive_bomb_kills = 0
        self.offensive_bomb_suicides = 0
        self.safe_adjacent_tiles_total = 0
        self.opponent_distance_total = 0
        self.opponent_distance_samples = 0
        self.killed_by_opponent_bomb = 0
        self.successful_opponent_bomb_escapes = 0
        self.opponent_coins = 0
        self.own_bomb_positions = set()
        self.pending_offensive_bombs = []
        self.steps_to_kill = None
