class RewardFunction:
    def metadata(self):
        return dict(self.__dict__)

    def reward(self, events):
        raise NotImplementedError
