import matplotlib.pyplot as plt
import numpy as np

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


# ----------
# Plots
# ----------

def plot_total_score(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]
    total_score = [metric["score"] for metric in metrics]

    if SHOW_RAW_VALUES:
        ax.scatter(episodes, total_score, s=10, alpha=0.2, color="black")
    ax.plot(
        episodes,
        running_average(total_score, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="red",
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total_score")
    ax.set_title("Total_score per episode")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_invalid_moves(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]
    invalid_moves = [metric["invalid_moves"] for metric in metrics]

    if SHOW_RAW_VALUES:
        ax.scatter(episodes, invalid_moves, s=10, alpha=0.2, color="black")
    ax.plot(
        episodes,
        running_average(invalid_moves, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Invalid moves")
    ax.set_title("Invalid moves per episode")
    ax.grid(True, alpha=0.3)
    ax.legend()



def plot_decision_time(metrics, ax):
    metrics = [metric for metric in metrics if "decision_time_mean" in metric]
    if not metrics:
        return

    episodes = [metric["episode"] for metric in metrics]
    mean_times = [metric["decision_time_mean"] for metric in metrics]
    max_times = [metric["decision_time_max"] for metric in metrics]

    ax.plot(episodes, mean_times, marker="o",ls="none", label="Mean time", ms=2, color="Blue", alpha=0.3)
    ax.plot(episodes, max_times, marker="o", ls="none", label="Max time", ms=2, color="Red", alpha=0.3)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Decision time (seconds)")
    ax.set_title("Decision time per episode")
    upper_limit = np.percentile(max_times, 99)
    ax.set_ylim(0, upper_limit)
    ax.grid(True, alpha=0.3)
    ax.legend()
    

def plot_survival(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]
    survival_steps = [metric["steps"] for metric in metrics]

    ax.plot(episodes, survival_steps, marker="o", ls="none", color="orange", ms=2, alpha=0.2, label="Steps")
    ax.plot(
        episodes,
        running_average(survival_steps, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Steps survived")
    ax.set_title("Survival")
    ax.grid(True, alpha=0.3)
    ax.legend()

"""
dummy_metrics = []
for i in range(50):
    dummy_metrics.append({
        "episode": i,
        "score": max(0, int(random.gauss(10 + i, 5))),               
        "invalid_moves": max(0, int(random.gauss(15 - (i * 0.3), 3))), 
        "decision_time_mean": random.uniform(0.01, 0.05),
        "decision_time_max": random.uniform(0.08, 0.2),
        "steps": min(400, int(random.gauss(50 + (i * 7), 20)))         
    })
"""

def create_figure(metric):
    if scienceplots:
        plt.style.use(["science", "no-latex"])
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    plot_total_score(metric, ax=ax1)
    plot_invalid_moves(metric, ax=ax2)
    plot_decision_time(metric, ax=ax3)
    plot_survival(metric, ax=ax4)

    figure.tight_layout()
    return figure
