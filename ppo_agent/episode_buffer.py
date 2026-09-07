class EpisodeBuffer:
    def __init__(self):
        self.reset()

    def add(self, state, action, log_prob, value, mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(None if mask is None else mask.clone())
        self.rewards.append(0.0)
        self.dones.append(False)
        self.pending = True

    def finish(self, reward, done):
        if not self.pending:
            return False

        self.rewards[-1] = reward
        self.dones[-1] = done
        self.pending = False
        return True

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.masks = []
        self.rewards = []
        self.dones = []
        self.pending = False

    def __len__(self):
        return len(self.states)
