import events as e


class RewardFunction:
    description = "Common event rewards with optional task shaping."

    def __init__(
        self,
        step_reward=0.0,
        coin_reward=0.0,
        invalid_action_reward=0.0,
        wait_reward=0.0,
        crate_destroyed_reward=0.0,
        opponent_killed_reward=0.0,
        death_reward=0.0,
        suicide_reward=0.0,
        survived_round_reward=0.0,
    ):
        self.step_reward = step_reward
        self.coin_reward = coin_reward
        self.invalid_action_reward = invalid_action_reward
        self.wait_reward = wait_reward
        self.crate_destroyed_reward = crate_destroyed_reward
        self.opponent_killed_reward = opponent_killed_reward
        self.death_reward = death_reward
        self.suicide_reward = suicide_reward
        self.survived_round_reward = survived_round_reward

    def reward(self, events, old_game_state=None, new_game_state=None):
        return self.event_reward(events) + self.shaping_reward(
            events, old_game_state, new_game_state
        )

    def event_reward(self, events):
        reward = self.step_reward
        if e.COIN_COLLECTED in events:
            reward += self.coin_reward
        if e.INVALID_ACTION in events:
            reward += self.invalid_action_reward
        if e.WAITED in events:
            reward += self.wait_reward
        if e.CRATE_DESTROYED in events:
            reward += self.crate_destroyed_reward
        if e.KILLED_OPPONENT in events:
            reward += self.opponent_killed_reward
        if e.GOT_KILLED in events:
            reward += self.death_reward
        if e.KILLED_SELF in events:
            reward += self.suicide_reward
        if e.SURVIVED_ROUND in events:
            reward += self.survived_round_reward
        return reward

    def shaping_reward(self, events, old_game_state, new_game_state):
        return 0.0

    def shaping_metadata(self):
        return {}

    def metadata(self):
        return {
            "description": self.description,
            "common": {
                "step_reward": self.step_reward,
                "coin_reward": self.coin_reward,
                "invalid_action_reward": self.invalid_action_reward,
                "wait_reward": self.wait_reward,
                "crate_destroyed_reward": self.crate_destroyed_reward,
                "opponent_killed_reward": self.opponent_killed_reward,
                "death_reward": self.death_reward,
                "suicide_reward": self.suicide_reward,
                "survived_round_reward": self.survived_round_reward,
            },
            "shaping": self.shaping_metadata(),
        }
