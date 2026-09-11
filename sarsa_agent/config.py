"""Configure SARSA training parameters."""

from pathlib import Path


REINITIALIZE_Q_TABLE = True
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
LOAD_MODEL_FILENAME = "sarsa_model_task_3.pt"
SAVE_MODEL_FILENAME = "sarsa_model_task_3.pt"
LOAD_PATH = Path(__file__).with_name(LOAD_MODEL_FILENAME)
MODEL_PATH = Path(__file__).with_name(SAVE_MODEL_FILENAME)
METRICS_PATH = Path(__file__).with_name("train.jsonl")
TIMING_PATH_ENV = "SARSA_DECISION_TIMING_PATH"

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998
RANDOM_SEED = 42
