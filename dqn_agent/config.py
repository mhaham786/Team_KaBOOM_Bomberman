"""Configure DQN training and experiment settings."""

from pathlib import Path


AGENT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = AGENT_DIR / "experiments"

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

EXPERIMENT_NAME = "pretrained_dqn_t2"
DESCRIPTION = "Improve Rewards, shaping v2, improved features v2, improved hyperparameters."

RESUME_TRAINING = False
RESTART_EXPERIMENT = True
INITIAL_WEIGHTS_EXPERIMENT = None

OBSERVATION_COUNT = 33

# DQN
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_UPDATE_INTERVAL = 1_000
EPSILON_START = 0.5
EPSILON_END = 0.001
EPSILON_DECAY_STEPS = 20_000
GRADIENT_CLIP_NORM = 10.0
RANDOM_SEED = 42

# Replay buffer
BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
MIN_REPLAY_SIZE = 1_000

COLLECT_ONLY = False
TRAIN_EVERY = 1
SAVE_EVERY_ROUNDS = 25
