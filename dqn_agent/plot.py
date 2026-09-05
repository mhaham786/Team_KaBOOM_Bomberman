"""Plot metrics for the configured DQN experiment."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..common.plots import general_plot, task1, task2
from ..common.plots.helpers import load_metrics
from . import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        help="Relative path to a train.jsonl file; overrides EXPERIMENT_NAME.",
    )
    parser.add_argument(
        "--task",
        choices=(1, 2, 3),
        type=int,
        help="Task-specific plot to create.",
    )
    return parser.parse_args()


def metrics_path_from_args(args):
    if args.path is not None:
        return args.path
    return config.EXPERIMENTS_DIR / config.EXPERIMENT_NAME / "train.jsonl"


def create_task_figure(metrics, task):
    if task == 1:
        return task1.create_figure_task1(metrics), "task1_metrics.png"
    if task == 2:
        return task2.create_figure_task2(metrics), "task2_metrics.png"
    return None


def main():
    args = parse_args()
    metrics_path = metrics_path_from_args(args)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = load_metrics(metrics_path)
    output_dir = metrics_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    general_figure = general_plot.create_figure(metrics)
    general_figure.savefig(output_dir / "general_metrics.png")

    task_figure = create_task_figure(metrics, args.task)
    if task_figure is not None:
        figure, filename = task_figure
        figure.savefig(output_dir / filename)

    plt.show()


if __name__ == "__main__":
    main()
