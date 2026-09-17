import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots
except ImportError:
    pass

from .helpers import running_average

def get_agent_names(metrics):
    names = set()
    for m in metrics:
        if "agent_metrics" in m:
            names.update(m["agent_metrics"].keys())
    return sorted(list(names))

def plot_win_rate(metrics, ax, color_map):
    agents = get_agent_names(metrics)
    if not agents:
        return

    wins = {name: 0 for name in agents}
    for m in metrics:
        for name, data in m.get("agent_metrics", {}).items():
            if data.get("won"):
                wins[name] += 1

    labels = [k for k, v in wins.items() if v > 0]
    counts = [v for v in wins.values() if v > 0]
    colors = [color_map[l] for l in labels]

    ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel("Total Wins")
    ax.set_title("Overall Win Count")
    ax.tick_params(axis='x', rotation=15)

def plot_running_score(metrics, ax, color_map):
    episodes = [m["episode"] for m in metrics]
    agents = get_agent_names(metrics)
    
    if not episodes or not agents:
        return

    for name in agents:
        scores = [m.get("agent_metrics", {}).get(name, {}).get("score", 0) for m in metrics]
        smoothed = running_average(scores, 100)
        ax.plot(episodes, smoothed, linewidth=1.5, color=color_map[name], label=name)

    ax.set_title("Average Score (100-ep rolling)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left")

def plot_survival_distribution(metrics, ax, color_map):
    agents = get_agent_names(metrics)
    if not agents:
        return

    data, colors = [], []
    for name in agents:
        steps = [m.get("agent_metrics", {}).get(name, {}).get("steps_survived") for m in metrics]
        valid_steps = [s for s in steps if s is not None]
        if valid_steps:
            data.append(valid_steps)
            colors.append(color_map[name])

    if not data:
        return

    parts = ax.violinplot(data, showmeans=True, showmedians=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
        
    for part in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        parts[part].set_edgecolor('black')
        parts[part].set_linewidth(1)

    ax.set_xticks(np.arange(1, len(agents) + 1))
    ax.set_xticklabels(agents, rotation=15)
    ax.set_title("Survival Distribution Density")
    ax.set_ylabel("Steps")

def plot_pacing_and_ratios(metrics, ax):
    episodes = [m["episode"] for m in metrics]
    if not episodes:
        return

    c_eps = [m["episode"] for m in metrics if m.get("steps_to_five_coins") is not None]
    c_vals = [m["steps_to_five_coins"] for m in metrics if m.get("steps_to_five_coins") is not None]
    
    e_eps = [m["episode"] for m in metrics if m.get("steps_to_first_elimination") is not None]
    e_vals = [m["steps_to_first_elimination"] for m in metrics if m.get("steps_to_first_elimination") is not None]

    kd = [m.get("kill_death_ratio", 0) for m in metrics]
    ks = [m.get("kill_suicide_ratio", 0) for m in metrics]

    if c_eps:
        ax.plot(c_eps, running_average(c_vals, 100), color='tab:orange', label="Steps to 5 Coins")
    if e_eps:
        ax.plot(e_eps, running_average(e_vals, 100), color='tab:red', label="Steps to 1st Elim")
        
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Steps Taken")
    
    ax2 = ax.twinx()
    ax2.plot(episodes, running_average(kd, 100), color='tab:blue', linestyle='--', label="K/D Ratio")
    ax2.plot(episodes, running_average(ks, 100), color='tab:purple', linestyle=':', label="K/S Ratio")
    ax2.set_ylabel("Ratio")
    
    ax.set_title("Primary Agent Pacing & Combat")
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

def plot_early_survival_rates(metrics, ax, color_map):
    agents = get_agent_names(metrics)
    total = len(metrics)
    if not agents or total == 0:
        return

    s50, s100 = [], []
    for name in agents:
        c50 = sum(1 for m in metrics if m.get("agent_metrics", {}).get(name, {}).get("survived_first_50_steps"))
        c100 = sum(1 for m in metrics if m.get("agent_metrics", {}).get(name, {}).get("survived_first_100_steps"))
        s50.append((c50 / total) * 100)
        s100.append((c100 / total) * 100)

    x = np.arange(len(agents))
    width = 0.35

    ax.bar(x - width/2, s50, width, label='> 50 Steps', color='lightgray', edgecolor='black')
    bars = ax.bar(x + width/2, s100, width, label='> 100 Steps', edgecolor='black')
    
    for i, bar in enumerate(bars):
        bar.set_color(color_map[agents[i]])
        bar.set_edgecolor('black')

    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Early Game Survival')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15)

def plot_first_eliminated_freq(metrics, ax, color_map):
    agents = get_agent_names(metrics)
    total = len(metrics)
    if not agents or total == 0:
        return

    counts = {name: sum(1 for m in metrics if m.get("agent_metrics", {}).get(name, {}).get("first_eliminated")) for name in agents}
    percentages = [(counts[name] / total) * 100 for name in agents]
    colors = [color_map[name] for name in agents]

    ax.bar(agents, percentages, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('% First Eliminated')
    ax.set_title('First Blood Victim Frequency')
    ax.tick_params(axis='x', rotation=15)

def create_figure_task4(metrics):
    try:
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass
    
    agents = get_agent_names(metrics)
    cmap = plt.get_cmap('tab10')
    color_map = {name: cmap(i % 10) for i, name in enumerate(agents)}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    plot_win_rate(metrics, ax1, color_map)
    plot_running_score(metrics, ax2, color_map)
    plot_survival_distribution(metrics, ax3, color_map)
    plot_pacing_and_ratios(metrics, ax4)
    plot_early_survival_rates(metrics, ax5, color_map)
    plot_first_eliminated_freq(metrics, ax6, color_map)

    fig.tight_layout()
    return fig
