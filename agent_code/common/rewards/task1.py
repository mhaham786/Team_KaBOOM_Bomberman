from .base import RewardFunction


class CoinHeavenRewards(RewardFunction):
    description = "Collect coins and avoid invalid moves."

    def __init__(self):
        rewards = {
            "step_reward": -0.001,
            "coin_reward": 2.0,
            "invalid_action_reward": -0.02,
            "wait_reward": -0.01,
            "death_reward": -1.0,
        }
        super().__init__(**rewards)
