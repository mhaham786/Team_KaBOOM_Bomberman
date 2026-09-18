"""Configure DQN training and experiment settings."""

from pathlib import Path


AGENT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = AGENT_DIR / "experiments"

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

EXPERIMENT_NAME = "task4_mixed"
DESCRIPTION = "Task 4 against 3 rule based agents"

RESUME_TRAINING = False
RESTART_EXPERIMENT = False
INITIAL_WEIGHTS_EXPERIMENT = "task4_3_rule_based_v1"

OBSERVATION_COUNT = 33

# DQN
GAMMA = 0.99
LEARNING_RATE = 5e-5 #reduced from 1e-4 to 5e-5
TARGET_UPDATE_INTERVAL = 1_000
EPSILON_START = 0.1  #reduced from 0.5 to 0.1
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
