class Features:
    description = None

    def metadata(self):
        if self.description is None:
            raise NotImplementedError("Every feature set needs a description.")
        return {"description": self.description}

    def encode(self, game_state):
        raise NotImplementedError

    def observation_count(self):
        raise NotImplementedError
