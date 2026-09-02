import json
import shutil
from pathlib import Path

import torch


class ExperimentRun:
    def __init__(self, experiments_dir, name):
        self.name = name
        self.path = Path(experiments_dir) / name
        self.latest_path = self.path / "latest.pt"
        self.metadata_path = self.path / "metadata.json"
        self.train_metrics_path = self.path / "train.jsonl"
        self.plots_path = self.path / "plots"

    def restart(self):
        if self.path.exists():
            shutil.rmtree(self.path)

    def create(self, metadata):
        self.path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(exist_ok=True)
        self.train_metrics_path.touch(exist_ok=True)

        if self.metadata_path.is_file():
            return

        with self.metadata_path.open("w") as file:
            json.dump({"run_name": self.name, **metadata}, file, indent=2)

    def save_latest(self, checkpoint):
        torch.save(checkpoint, self.latest_path)

    def load_latest(self):
        if not self.latest_path.is_file():
            return None
        return torch.load(
            self.latest_path,
            map_location="cpu",
            weights_only=False,
        )

    def append_train_metric(self, metric):
        with self.train_metrics_path.open("a") as file:
            file.write(json.dumps(metric) + "\n")

    def get_progress(self):
        episode = 0
        if not self.train_metrics_path.is_file():
            return episode

        with self.train_metrics_path.open() as file:
            for line in file:
                if not line.strip():
                    continue
                metric = json.loads(line)
                episode = max(episode, metric["episode"])

        return episode
