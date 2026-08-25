import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# ----------
# Settings
# ----------

RUNNING_AVERAGE_WINDOW = 100
SHOW_RAW_VALUES = True
SHOW = True
SAVE = True


# ----------
# Helpers
# ----------

def load_metrics(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def running_average(values, window):
    averages = []
    total = 0.0

    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        averages.append(total / min(index + 1, window))

    return averages


# ----------
# Plots
# ----------

def plot_coins(metrics, output_path):
    episodes = [metric["episode"] for metric in metrics]
    coins = [metric["coins"] for metric in metrics]

    plt.figure(figsize=(10, 5))
    if SHOW_RAW_VALUES:
        plt.scatter(episodes, coins, s=10, alpha=0.2, label="Episode coins")
    plt.plot(
        episodes,
        running_average(coins, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Coins collected")
    plt.title("Coins collected per episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if SAVE:
        plt.savefig(output_path)
    if SHOW:
        plt.show()
    plt.close()


def plot_invalid_moves(metrics, output_path):
    episodes = [metric["episode"] for metric in metrics]
    invalid_moves = [metric["invalid_moves"] for metric in metrics]

    plt.figure(figsize=(10, 5))
    if SHOW_RAW_VALUES:
        plt.scatter(episodes, invalid_moves, s=10, alpha=0.2, label="Episode invalid moves")
    plt.plot(
        episodes,
        running_average(invalid_moves, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Invalid moves")
    plt.title("Invalid moves per episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if SAVE:
        plt.savefig(output_path)
    if SHOW:
        plt.show()
    plt.close()


def plot_eval_coins(metrics, output_path):
    episodes = [metric.get("training_episode", metric["episode"]) for metric in metrics]
    coins = [metric["coins"] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, coins, marker="o")
    plt.xlabel("Training episode")
    plt.ylabel("Coins collected")
    plt.title("Evaluation coins")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE:
        plt.savefig(output_path)
    if SHOW:
        plt.show()
    plt.close()


def plot_eval_invalid_moves(metrics, output_path):
    episodes = [metric.get("training_episode", metric["episode"]) for metric in metrics]
    invalid_moves = [metric["invalid_moves"] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, invalid_moves, marker="o")
    plt.xlabel("Training episode")
    plt.ylabel("Invalid moves")
    plt.title("Evaluation invalid moves")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE:
        plt.savefig(output_path)
    if SHOW:
        plt.show()
    plt.close()


def plot_eval_time(metrics, output_path):
    metrics = [metric for metric in metrics if "decision_time_mean" in metric]
    if not metrics:
        return

    episodes = [metric.get("training_episode", metric["episode"]) for metric in metrics]
    mean_times = [metric["decision_time_mean"] for metric in metrics]
    max_times = [metric["decision_time_max"] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mean_times, marker="o", label="Mean time")
    plt.plot(episodes, max_times, marker="o", label="Max time")
    plt.xlabel("Training episode")
    plt.ylabel("Decision time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Evaluation decision time")
    plt.tight_layout()
    if SAVE:
        plt.savefig(output_path)
    if SHOW:
        plt.show()
    plt.close()


PLOTS = {
    "coins": ("train", plot_coins, "coins_per_episode.png"),
    "invalid_moves": ("train", plot_invalid_moves, "invalid_moves_per_episode.png"),
    "eval_coins": ("eval", plot_eval_coins, "eval_coins.png"),
    "eval_invalid_moves": ("eval", plot_eval_invalid_moves, "eval_invalid_moves.png"),
    "eval_time": ("eval", plot_eval_time, "eval_time.png"),
}


# ----------
# Main
# ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", help="Name configured in config.py")
    parser.add_argument("plots", nargs="*", choices=PLOTS)
    args = parser.parse_args()

    experiment_path = Path(__file__).parent / "experiments" / args.experiment_name
    plots_path = experiment_path / "plots"

    plots_path.mkdir(exist_ok=True)
    for plot_name in args.plots or PLOTS:
        metric_type, plot, filename = PLOTS[plot_name]
        metrics_path = experiment_path / "metrics" / f"{metric_type}.jsonl"

        metrics = load_metrics(metrics_path) if metrics_path.is_file() else []
        if not metrics:
            if args.plots:
                raise ValueError(f"No {metric_type} metrics found at {metrics_path}")
            continue

        plot(metrics, plots_path / filename)

    if SAVE:
        print(f"Saved plots to {plots_path}")


if __name__ == "__main__":
    main()
