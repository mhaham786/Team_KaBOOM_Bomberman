import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import scienceplots
import random


# ----------
# Settings
# ----------

RUNNING_AVERAGE_WINDOW = 100
SHOW_RAW_VALUES = True
SHOW = True
SAVE = True
TOTAL_COINS = 50

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

def plot_coin_distribution(metrics, ax):
    completion_steps = [m["steps"] for m in metrics if m["coins"] == 50]


    if not completion_steps:
        return

    if SHOW_RAW_VALUES:
        ax.hist(completion_steps, bins=20, color="blanchedalmond", edgecolor="black", alpha=0.7)    

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Frequency")
    ax.set_title("Steps Taken to collect all the coins (Distribution)")
    ax.grid(True, alpha=0.3)
    ax.legend()



def plot_path_efficiency(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]
    path_efficiency = [metric["path_efficiency"] for metric in metrics]

    if SHOW_RAW_VALUES:
        ax.scatter(episodes, path_efficiency, s=10, alpha=0.2, color = 'black')
    ax.plot(
        episodes,
        running_average(path_efficiency, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="black",
        label=f"{RUNNING_AVERAGE_WINDOW}-episode average",
    )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Path Efficiency")
    ax.set_title("Path Efficiency per episode")
    ax.grid(True, alpha=0.3)
    ax.legend()



dummy_metrics = []
for i in range(1000):
    # Simulate learning progression:
    # Early on, agent rarely gets all 50 coins. Later, it almost always does.
    success_probability = min(0.95, i / 300) 
    
    if random.random() < success_probability:
        coins_collected = 50
        # Steps decrease over time, eventually centering heavily around 130 steps
        base_steps = max(130, 300 - (i * 0.4))
        steps_taken = int(random.gauss(base_steps, 15))
    else:
        coins_collected = int(random.uniform(10, 49))
        steps_taken = 400 # Hit step limit
        
    # Path efficiency starts around 0.2 and asymptotes to ~0.95
    base_eff = 0.95 - (0.75 * (0.995 ** i))
    eff = min(1.0, max(0.0, random.gauss(base_eff, 0.04)))
    
    dummy_metrics.append({
        "episode": i,
        "coins": coins_collected,
        "steps": steps_taken,
        "path_efficiency": eff
    })

plt.style.use(['science', 'no-latex'])
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 6))

plot_coin_distribution(dummy_metrics, ax=ax1)
plot_path_efficiency(dummy_metrics, ax=ax2)

plt.tight_layout()
#plt.savefig("genera_metric.png")
plt.show()