import json

import torch

from . import config


class ExperimentRun:
    def __init__(self, name):
        self.name = name
        self.path = config.EXPERIMENTS_DIR / name

    def create(self, model):
        checkpoints = self.path / "checkpoints"
        metrics = self.path / "metrics"
        plots = self.path / "plots"
        metadata_path = self.path / "metadata.json"

        checkpoints.mkdir(parents=True, exist_ok=True)
        metrics.mkdir(exist_ok=True)
        plots.mkdir(exist_ok=True)
        (metrics / "train.jsonl").touch()

        if not metadata_path.is_file():
            metadata = {
                "run_id": self.name,
                "plot": config.PLOT,
                "actions": list(config.ACTIONS),
                "config": {
                    name.lower(): value
                    for name, value in vars(config).items()
                    if name.isupper() and isinstance(value, int | float)
                },
                "trainer": config.TRAINER.__name__,
                "model_structure": str(model),
                "rewards": {
                    "class": config.REWARDS.__class__.__name__,
                    **config.REWARDS.metadata(),
                },
                "features": {
                    "class": config.FEATURES.__class__.__name__,
                    "description": config.FEATURES.metadata()["description"],
                },
            }
            with metadata_path.open("w") as file:
                json.dump(metadata, file, indent=2)

    def load_latest(self, model, optimizer):
        latest_path = self.path / "checkpoints" / "latest.pt"
        if not latest_path.is_file():
            return False

        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        return True

    def save_latest(self, model, optimizer):
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            self.path / "checkpoints" / "latest.pt",
        )

    def append_train_metric(self, metric):
        with (self.path / "metrics" / "train.jsonl").open("a") as file:
            file.write(json.dumps(metric) + "\n")

    def get_progress(self):
        episode = 0
        best_score = -1
        metrics_path = self.path / "metrics" / "train.jsonl"
        if not metrics_path.is_file():
            return episode, best_score

        with metrics_path.open() as file:
            for line in file:
                if line.strip():
                    metric = json.loads(line)
                    episode = max(episode, metric["episode"])
                    best_score = max(best_score, metric["score"])

        return episode, best_score
