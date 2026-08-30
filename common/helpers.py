from collections import deque
from collections.abc import Iterable


def nearest_coin_bfs(field, start, coins):
    """Return the nearest coin reachable through free tiles, or None"""
    coin_positions = {tuple(coin) for coin in coins}
    if not coin_positions:
        return None

    width, height = field.shape
    queue = deque([start])
    visited = {start}
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

    while queue:
        position = queue.popleft()
        if position in coin_positions:
            return position

        for dx, dy in directions:
            neighbour = (position[0] + dx, position[1] + dy)
            x, y = neighbour
            if (
                0 <= x < width
                and 0 <= y < height
                and neighbour not in visited
                and field[x, y] == 0
            ):
                visited.add(neighbour)
                queue.append(neighbour)

    return None
