"""Configure all training and experiment settings."""

from pathlib import Path

AGENT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = AGENT_DIR / "experiments"

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

EXPERIMENT_NAME = "task4_eval_snapshot9500_vs_sarsa_dqn_ruehl"
DESCRIPTION = "Frozen snapshot 9500 evaluated against SARSA Task 4, DQN Task 4, and RUEHL."
RANDOM_SEED = 42

RESUME_TRAINING = False
RESTART_EXPERIMENT = True
INITIAL_WEIGHTS_EXPERIMENT = "task4_oc54_rotating_league_snapshot9500"
LOAD_INITIAL_OPTIMIZER = False
EVALUATION_ONLY = True

OBSERVATION_COUNT = 54

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2 # changed from 0.1 
VALUE_COEF = 0.5
ENTROPY_COEF = 0.005
UPDATE_EPOCHS = 3
MINIBATCH_SIZE = 64
ROLLOUT_STEPS = 2048
GRADIENT_CLIP_NORM = 0.5
