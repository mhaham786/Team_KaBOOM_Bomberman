import events as e

def coin_heaven_rewards_ppo(events):
    reward = -0.001
    if e.COIN_COLLECTED in events:
        reward += 2.0
    if e.INVALID_ACTION in events:
        reward += -0.02
    if e.WAITED in events:
        reward += -0.01
    if e.KILLED_SELF in events:
        reward +=  -1.0
    if e.GOT_KILLED in events:
        reward += -1.0

    return reward