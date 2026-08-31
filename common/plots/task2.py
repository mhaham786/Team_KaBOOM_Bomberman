import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots
except ModuleNotFoundError:
    scienceplots = None

from helpers import running_average


# ----------
# Settings
# ----------

RUNNING_AVERAGE_WINDOW = 100
SHOW_RAW_VALUES = True
TOTAL_COINS = 9


# ----------
# Plots
# ----------

def plot_crates_bombs(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]
    crates_destroyed = [metric["crates_destroyed"] for metric in metrics]
    useful_bombs = [metric["useful_bombs"] for metric in metrics]
    useless_bombs = [metric["useless_bombs"] for metric in metrics]
    
    if not episodes:
        return
        
    #if SHOW_RAW_VALUES:
        #ax.scatter(episodes, crates_destroyed, s=10, alpha=0.2, color='black', label="Total Crates destroyed")
        #ax.scatter(episodes, useful_bombs, s=10, alpha=0.2, color="blue", label="Useful bombs")
        #ax.scatter(episodes, useless_bombs, s=10, alpha=0.2, color="red", label="Useless bombs")

    ax.plot(
        episodes,
        running_average(crates_destroyed, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="black",
        label=f"crates destroyed - {RUNNING_AVERAGE_WINDOW}-episode average",
    )
    ax.plot(
        episodes,
        running_average(useful_bombs, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="blue",
        label=f"useful bombs - {RUNNING_AVERAGE_WINDOW}-episode average",
    )
    ax.plot(
        episodes,
        running_average(useless_bombs, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="red",
        label=f"useless bombs - {RUNNING_AVERAGE_WINDOW}-episode average",
    )

    ax.set_xlabel("Training episodes")
    ax.set_ylabel("amount")
    ax.set_title("Crates destroyed and bombs used per episode")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_escape(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]  # FIXED: Added missing variable
    bombs_exploded = [metric["bombs_exploded"] for metric in metrics]
    successful_escapes = [metric["successful_escapes"] for metric in metrics]
    
    if not bombs_exploded:
        return
        
    if SHOW_RAW_VALUES: 
        ax.scatter(episodes, bombs_exploded, s=10, alpha=0.2, color="red", label="Bombs exploded")
        ax.scatter(episodes, successful_escapes, s=10, alpha=0.2, color="green", label="Successful escapes")

    valid_episodes = episodes[RUNNING_AVERAGE_WINDOW - 1:] 
    
    ax.plot(
        episodes,
        running_average(bombs_exploded, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="red",
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    ax.plot(
        episodes,
        running_average(successful_escapes, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="green",
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    
    ax.set_xlabel("Training episode")
    ax.set_ylabel("amount")
    ax.set_title("Bombs exploded and successful escapes per episode")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_coins_found(metrics, ax): 
    coins = [metric["coins"] for metric in metrics]

    if not coins:
        return

    if SHOW_RAW_VALUES:
        ax.hist(coins, bins=np.arange(-0.5, TOTAL_COINS + 1.5, 1), color="blanchedalmond", edgecolor="black", alpha=0.7)

    ax.set_xlabel("Number of coins found")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of coins found")
    ax.set_xticks(range(TOTAL_COINS + 1))
    ax.grid(True, alpha=0.3)


def create_figure_task2(metric):
    if scienceplots:
        plt.style.use(["science", "no-latex"])
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    plot_crates_bombs(metric, ax1)
    plot_coins_found(metric, ax2)
    plot_escape(metric, ax3)

    figure.tight_layout()
    return figure
