import events as e


def coin_heaven_rewards_sarsa(events):
    rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_SELF: -500,
        e.WAITED: -1,
        e.INVALID_ACTION: -5,
    }
    return sum(rewards.get(event, 0) for event in events)


def coin_heaven_rewards_ppo_baseline(events):
    reward = -0.001
    if e.COIN_COLLECTED in events:
        reward += 2.0
    if e.INVALID_ACTION in events:
        reward += -0.02
    if e.WAITED in events:
        reward += -0.01

    return reward

def coin_heaven_rewards_ppo_improved(events):
    reward = -0.5
    if e.COIN_COLLECTED in events:
        reward += 2.0
    if e.INVALID_ACTION in events:
        reward += -0.5
    if e.WAITED in events:
        reward += -0.01

    return reward
