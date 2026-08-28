from collections import deque

import numpy as np

from .base import Features


class CoinHeavenSpacialFeatures(Features):
    description = "Three 17x17 maps: board tiles, agent position, and coin positions."

    def encode(self, game_state):
        if game_state is None:
            return None

        field = game_state["field"].astype(np.float32)

        self_map = np.zeros_like(field, dtype=np.float32)
        _, _, _, (x, y) = game_state["self"]
        self_map[x, y] = 1.0

        coin_map = np.zeros_like(field, dtype=np.float32)
        for x, y in game_state["coins"]:
            coin_map[x, y] = 1.0

        return np.concatenate([field.ravel(), self_map.ravel(), coin_map.ravel()])

    def observation_count(self):
        return 3 * 17 * 17

class CoinHeavenMinimalFeatures(Features):
    description = (
        "Agent position, nearest reachable coin direction and BFS distance, "
        "four free neighbour flags, and remaining coin fraction."
    )

    def __init__(self):
        self.round = None
        self.total_coins = 0

    def encode(self, game_state):
        if game_state is None:
            return None

        field = game_state["field"]
        coins = game_state["coins"]
        _, _, _, (agent_x, agent_y) = game_state["self"]
        width, height = field.shape

        if game_state["round"] != self.round:
            self.round = game_state["round"]
            self.total_coins = len(coins)

        coin_x, coin_y, distance = self._nearest_coin(field, coins, agent_x, agent_y)
        max_distance = width + height - 2

        features = [
            agent_x / (width - 1),
            agent_y / (height - 1),
            (coin_x - agent_x) / (width - 1),
            (coin_y - agent_y) / (height - 1),
            distance / max_distance,
            self._is_free(field, agent_x, agent_y - 1),
            self._is_free(field, agent_x + 1, agent_y),
            self._is_free(field, agent_x, agent_y + 1),
            self._is_free(field, agent_x - 1, agent_y),
            len(coins) / self.total_coins if self.total_coins else 0.0,
        ]
        return np.array(features, dtype=np.float32)

    def observation_count(self):
        return 10

    def _nearest_coin(self, field, coins, start_x, start_y):
        if not coins:
            return start_x, start_y, 0

        coin_positions = set(coins)
        queue = deque([(start_x, start_y, 0)])
        visited = {(start_x, start_y)}

        while queue:
            x, y, distance = queue.popleft()
            if (x, y) in coin_positions:
                return x, y, distance

            for next_x, next_y in (
                (x, y - 1),
                (x + 1, y),
                (x, y + 1),
                (x - 1, y),
            ):
                if (next_x, next_y) not in visited and self._is_free(field, next_x, next_y):
                    visited.add((next_x, next_y))
                    queue.append((next_x, next_y, distance + 1))

        return start_x, start_y, 0

    def _is_free(self, field, x, y):
        width, height = field.shape
        return 0 <= x < width and 0 <= y < height and field[x, y] == 0
