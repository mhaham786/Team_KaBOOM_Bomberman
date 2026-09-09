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

def plot_offensive_bombs_stats(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]
    
    if not episodes:
        return 
        
    offensive_bomb_ratio = running_average([metric["offensive_bomb_ratio"] for metric in metrics], RUNNING_AVERAGE_WINDOW)
    offensive_bomb_kill_ratio = running_average([metric["offensive_bomb_kill_ratio"] for metric in metrics], RUNNING_AVERAGE_WINDOW)
    offensive_bomb_suicide_ratio = running_average([metric["offensive_bomb_suicide_ratio"] for metric in metrics], RUNNING_AVERAGE_WINDOW)


    ax.stackplot(episodes, offensive_bomb_kill_ratio, offensive_bomb_suicide_ratio,
                 labels=['kill ratio of offensive bombs', 'suicide ratio of offensive bombs'],
                 colors=['red', 'orange'], alpha=0.7)
    
    ax.plot(episodes, offensive_bomb_ratio, color='grey', linewidth=2, label='offensive bomb ratio')
    
    ax.set_title('Agent bomb outcomes')
    ax.set_ylabel('ratio')
    ax.set_xlabel("Training Episode")
    ax.legend(loc="upper left")  



def plot_opponent_bombs_stats(metrics, ax): 
    episodes = [metric["episode"] for metric in metrics]
    
    if not episodes:
        return
        
    killed_by_opponent_bomb = [metric["killed_by_opponent_bomb"] for metric in metrics]
    successful_opponent_bomb_escapes  = [metric["successful_opponent_bomb_escapes"] for metric in metrics]

    ax.scatter(episodes, killed_by_opponent_bomb, s=10, alpha=0.4, color='red', label="killed by opponent")
    ax.scatter(episodes, successful_opponent_bomb_escapes, s=10, alpha=0.4, color="blue", label="successful escapes")

    ax.plot(
        episodes,
        running_average(killed_by_opponent_bomb, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="red",
        label="Avg kills by opponent",
    )
    ax.plot(
        episodes,
        running_average(successful_opponent_bomb_escapes, RUNNING_AVERAGE_WINDOW),
        linewidth=2,
        color="blue",
        label="Avg successful escapes from opponent",
    )

    ax.set_xlabel('Training Episode')   
    ax.set_ylabel('amount')
    ax.set_title('Opponent bomb outcomes')
    ax.legend(loc="upper left")    


def plot_hunting_behavior(metrics, ax): 
    episodes = [metric["episode"] for metric in metrics]
    
    average_opponent_distance = running_average([metric["average_opponent_distance"] for metric in metrics], RUNNING_AVERAGE_WINDOW)
    average_safe_adjacent_tiles = running_average([metric["average_safe_adjacent_tiles"] for metric in metrics], RUNNING_AVERAGE_WINDOW)
    
    line1, = ax.plot(episodes, average_opponent_distance, color='red', label='Avg Opponent Distance')
    line2, = ax.plot(episodes, average_safe_adjacent_tiles, color='green', linestyle='--', label='Opponent\'s Avg Safe Escape Tiles')
    ax.set_ylabel('Spatial Distance / Tile Count')
    ax.set_xlabel('Training Episode')
    
    ax2 = ax.twinx()
    kill_line = None
    valid_episodes = []
    valid_kills = []
    for metric in metrics:
        steps = metric.get("steps_to_kill")
        if steps is not None:
            valid_episodes.append(metric["episode"])
            valid_kills.append(steps)
            
<<<<<<< HEAD
    plot = None
    if valid_kills: 
        plot = ax2.plot(valid_episodes, running_average(valid_kills, RUNNING_AVERAGE_WINDOW), linewidth=2, color="blue", label=f"steps to kill - {RUNNING_AVERAGE_WINDOW}-episode average")
=======
    if valid_kills:
        kill_line = ax2.plot(
            valid_episodes,
            running_average(valid_kills, RUNNING_AVERAGE_WINDOW),
            linewidth=2,
            color="blue",
            label=f"steps to kill - {RUNNING_AVERAGE_WINDOW}-episode average",
        )[0]
>>>>>>> 8bf78fdd26b7f044223513bd4980a487896ffe90
        
    ax2.set_ylabel('Steps to Kill')
    ax.set_title('Hunting Behavior & Trapping Efficiency')

    handles = [line1, line2]
    if kill_line is not None:
        handles.append(kill_line)
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right")



def plot_coin_stats(metrics, ax):
    episodes = [metric["episode"] for metric in metrics]
    if not episodes:
        return

    agent_coins = np.array(running_average([metric.get("coins", 0) for metric in metrics], RUNNING_AVERAGE_WINDOW))
    opponent_coins = np.array(running_average([metric.get("opponent_coins", 0) for metric in metrics], RUNNING_AVERAGE_WINDOW))
    
    ax.plot(episodes, agent_coins, color='gold', label='Agent Coins', linewidth=2)
    ax.plot(episodes, opponent_coins, color='grey', label='Opponent Coins', linewidth=2)

    ax.fill_between(episodes, agent_coins, opponent_coins, 
                    where=(agent_coins >= opponent_coins), 
                    interpolate=True, color='gold', alpha=0.3, label='Agent Lead')
    
    ax.fill_between(episodes, agent_coins, opponent_coins, 
                    where=(agent_coins < opponent_coins), 
                    interpolate=True, color='grey', alpha=0.3, label='Opponent Lead')
    
    ax.set_title('Agent vs Opponent Coins')
    ax.set_ylabel('Total Coins Gathered')
    ax.set_xlabel('Training Episode')
    ax.legend(loc="upper left")


def create_figure_task3(metric):
    if scienceplots:
        plt.style.use(["science", "no-latex"])
    
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,8))

    plot_offensive_bombs_stats(metric, ax1)
    plot_opponent_bombs_stats(metric, ax2)
    plot_hunting_behavior(metric, ax3)
    plot_coin_stats(metric, ax4)

    figure.tight_layout()
<<<<<<< HEAD
    return figure
=======
    return figure
>>>>>>> 8bf78fdd26b7f044223513bd4980a487896ffe90
