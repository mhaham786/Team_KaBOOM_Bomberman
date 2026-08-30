import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from . import general_plot, task1
from .helpers import load_metrics


# ----------
# Settings
# ----------

SHOW = True
SAVE = True


def load_plot(metrics_path):
    metadata_path = metrics_path.parent.parent / "metadata.json"
    if not metadata_path.is_file():
        return None

    with metadata_path.open() as file:
        return json.load(file).get("plot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_path", type=Path)
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_path)
    output_dir = args.metrics_path.parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    general_figure = general_plot.create_figure(metrics)
    if SAVE:
        general_figure.savefig(output_dir / "general_metrics.png")

    if load_plot(args.metrics_path) == "task1":
        task_figure = task1.create_figure(metrics)
        if SAVE:
            task_figure.savefig(output_dir / "task1_metrics.png")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
