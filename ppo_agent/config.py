"""Configure all training and experiment settings."""

from pathlib import Path

AGENT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = AGENT_DIR / "experiments"

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

EXPERIMENT_NAME = "task4_oc54_team_models_entropy005_v24"
DESCRIPTION = "Continue v22 against the team models with entropy reduced to 0.005."
RANDOM_SEED = 42

RESUME_TRAINING = False
RESTART_EXPERIMENT = True
INITIAL_WEIGHTS_EXPERIMENT = "task4_oc54_team_models_continue_v22"
LOAD_INITIAL_OPTIMIZER = True

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
