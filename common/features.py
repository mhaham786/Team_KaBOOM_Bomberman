from .helpers import *


def coin_heaven_minimal(game_state):
    """Encode player and nearest reachable coin as [x, y, coin_x, coin_y]"""
    if game_state is None:
        return None

    _, _, _, player_position = game_state["self"]
    player_x, player_y = player_position
    coin_position = nearest_coin_bfs(game_state["field"], player_position, game_state["coins"])

    coin_x, coin_y = coin_position if coin_position is not None else (-1, -1)
    return [float(player_x), float(player_y), float(coin_x), float(coin_y)]
