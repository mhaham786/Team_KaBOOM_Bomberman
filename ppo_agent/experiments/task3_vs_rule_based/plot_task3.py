import sys
import os


agent_code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(agent_code_path)

from common.plots.general_plot import create_figure
from common.plots.task3 import create_figure_task3
import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_metrics(filepath):
    """Reads the JSON Lines file generated during training."""
    metrics = []
    path = Path(filepath)
    if not path.is_file():
        print(f"No metrics file found at {filepath}")
        return metrics
        
    with path.open("r") as file:
        for line in file:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics

if __name__ == "__main__":

    output_dir = Path("training_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_data = load_metrics("train.jsonl")
    

    metrics_data = metrics_data[0:]
    
    if metrics_data:
        fig_general = create_figure(metrics_data)
        general_save_path = output_dir / "general_metrics.png"
        fig_general.savefig(general_save_path, dpi=300)
        
        fig_task3 = create_figure_task3(metrics_data)
        task3_save_path = output_dir / "task3_metrics.png"
        fig_task3.savefig(task3_save_path, dpi=300)
        
        print(f"Plots successfully saved to: {output_dir.resolve()}")
        
        plt.show()