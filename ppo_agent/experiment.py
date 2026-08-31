import json
import shutil

import torch

from . import config


class ExperimentRun:
    def __init__(self, name):
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("EXPERIMENT_NAME must be one directory name")
        self.name = name
        self.path = config.EXPERIMENTS_DIR / name
        self.latest_path = self.path / "latest.pt"
        self.metadata_path = self.path / "metadata.json"
        self.train_metrics_path = self.path / "train.jsonl"
        self.plots_path = self.path / "plots"

    def restart(self):
        if self.path.exists():
            shutil.rmtree(self.path)

    def create(self, model):
        self.path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(exist_ok=True)
        self.train_metrics_path.touch(exist_ok=True)

        if not self.metadata_path.is_file():
            metadata = {
                "run_name": self.name,
                "description": getattr(config, "DESCRIPTION", ""),
                "actions": list(config.ACTIONS),
                "config": {
                    name.lower(): value
                    for name, value in vars(config).items()
                    if name.isupper() and isinstance(value, int | float)
                },
                "model_structure": str(model),
            }
            with self.metadata_path.open("w") as file:
                json.dump(metadata, file, indent=2)

    def load_latest(self, model, optimizer):
        if not self.latest_path.is_file():
            return False

        saved = torch.load(self.latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        return True

    def save_latest(self, model, optimizer):
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            self.latest_path,
        )

    def append_train_metric(self, metric):
        with self.train_metrics_path.open("a") as file:
            file.write(json.dumps(metric) + "\n")

    def get_progress(self):
        episode = 0
        best_score = -1
        if not self.train_metrics_path.is_file():
            return episode, best_score

        with self.train_metrics_path.open() as file:
            for line in file:
                if line.strip():
                    metric = json.loads(line)
                    episode = max(episode, metric["episode"])
                    best_score = max(best_score, metric["score"])

        return episode, best_score
