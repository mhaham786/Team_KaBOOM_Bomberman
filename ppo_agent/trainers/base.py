class TrainerBase:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def update(self, transitions):
        raise NotImplementedError
