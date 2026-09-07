"""Configure all training and experiment settings."""

from pathlib import Path

AGENT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = AGENT_DIR / "experiments"

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

EXPERIMENT_NAME = "task2_classic_gae_entropy0025_1000_seed42"
DESCRIPTION = "Task 2 PPO with nine BFS observations and safe action masks."
TASK1_CHECKPOINT = EXPERIMENTS_DIR / "features_rewards_ppo" / "latest.pt"
RANDOM_SEED = 42

RESUME_TRAINING = False
RESTART_EXPERIMENT = False

OBSERVATION_COUNT = 9

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.1
VALUE_COEF = 0.5
ENTROPY_COEF = 0.0025
UPDATE_EPOCHS = 3
MINIBATCH_SIZE = 64
