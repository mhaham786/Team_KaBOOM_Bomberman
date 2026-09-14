"""Configure all training and experiment settings."""

from pathlib import Path

AGENT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = AGENT_DIR / "experiments"

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

EXPERIMENT_NAME = "task3_vs_rule_based"
DESCRIPTION = "Task3- Aggressive hunting and anti-suicide mask"
TASK1_CHECKPOINT = EXPERIMENTS_DIR / "task3_combat_run_v2" / "latest.pt"
RANDOM_SEED = 42

RESUME_TRAINING = False
RESTART_EXPERIMENT = False

OBSERVATION_COUNT = 9

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2 # changed from 0.1 
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01 # changed form 0.0025
UPDATE_EPOCHS = 3
MINIBATCH_SIZE = 64
