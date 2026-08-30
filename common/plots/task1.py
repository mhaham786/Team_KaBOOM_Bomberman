import matplotlib.pyplot as plt

try:
    import scienceplots
except ModuleNotFoundError:
    scienceplots = None

from .helpers import running_average


# ----------
# Settings
# ----------

RUNNING_AVERAGE_WINDOW = 100
SHOW_RAW_VALUES = True
TOTAL_COINS = 50


# ----------
# Plots
# ----------

def plot_coin_distribution(metrics, ax):
    completion_steps = [metric["steps"] for metric in metrics if metric["coins"] == TOTAL_COINS]
    if not completion_steps:
        return

    if SHOW_RAW_VALUES:
        ax.hist(completion_steps, bins=20, color="blanchedalmond", edgecolor="black", alpha=0.7)

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Frequency")
    ax.set_title("Steps to collect all coins")
    ax.grid(True, alpha=0.3)


def plot_path_efficiency(metrics, ax):
    metrics = [metric for metric in metrics if "path_efficiency" in metric]
    if not metrics:
        return

    episodes = [metric["episode"] for metric in metrics]
    path_efficiency = [metric["path_efficiency"] for metric in metrics]

    if SHOW_RAW_VALUES:
        ax.scatter(episodes, path_efficiency, s=10, alpha=0.2, color="black")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Path efficiency")
    ax.set_title("Path efficiency per episode")
    ax.grid(True, alpha=0.3)
    ax.legend()


def create_figure_task1(metric):
    if scienceplots:
        plt.style.use(["science", "no-latex"])
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plot_coin_distribution(metric, ax1)
    plot_path_efficiency(metric, ax2)

    figure.tight_layout()
    return figure
