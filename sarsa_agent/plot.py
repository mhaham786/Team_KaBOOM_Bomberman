"""Plot metrics from SARSA training."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.plots import general_plot, task1, task2, task3
from common.plots.helpers import load_metrics
import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        help="Relative path to a train.jsonl file; overrides the configured path.",
    )
    parser.add_argument(
        "--task",
        choices=(1, 2, 3),
        type=int,
        help="Additional task-specific plot to create.",
    )
    return parser.parse_args()


def metrics_path_from_args(args):
    if args.path is not None:
        return args.path
    return config.METRICS_PATH


def create_task_figure(metrics, task):
    if task == 1:
        return task1.create_figure_task1(metrics), "task1_metrics.png"
    if task == 2:
        return task2.create_figure_task2(metrics), "task2_metrics.png"
    if task == 3:
        return task3.create_figure_task3(metrics), "task3_metrics.png"
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
