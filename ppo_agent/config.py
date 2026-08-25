"""Configure all training and experiment settings."""

from pathlib import Path

from . import features, rewards


EXPERIMENT_NAME = "coin_heaven_small_net_less_features"
RESUME_TRAINING = False
EVAL_EVERY = 100
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")
AGENT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = AGENT_DIR / "experiments"

FEATURES = features.CoinHeavenMinimalFeatures()
REWARDS = rewards.CoinHeavenRewards()
OBSERVATION_COUNT = FEATURES.observation_count()

LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.1
VALUE_COEF = 0.5
ENTROPY_COEF = 0.005
UPDATE_EPOCHS = 3
