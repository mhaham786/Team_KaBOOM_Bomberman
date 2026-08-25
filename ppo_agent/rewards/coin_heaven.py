import events as e

from .base import RewardFunction


class CoinHeavenRewards(RewardFunction):
    def __init__(
        self,
        step_reward=-0.001,
        coin_reward=2.0,
        invalid_action_reward=-0.02,
        wait_reward=-0.01,
        death_reward=-1.0,
    ):
        self.step_reward = step_reward
        self.coin_reward = coin_reward
        self.invalid_action_reward = invalid_action_reward
        self.wait_reward = wait_reward
        self.death_reward = death_reward

    def reward(self, events):
        reward = self.step_reward

        if e.COIN_COLLECTED in events:
            reward += self.coin_reward
        if e.INVALID_ACTION in events:
            reward += self.invalid_action_reward
        if e.WAITED in events:
            reward += self.wait_reward
        if e.GOT_KILLED in events:
            reward += self.death_reward

        return reward
