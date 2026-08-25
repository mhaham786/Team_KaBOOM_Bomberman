# Uttam DQN V2 - Refactored Verified Round 10620

This clean package contains the refactored Uttam DQN V2 runtime and regression
tests at verified training round 10620. The feature schema is
uttam_dqn_v2_time_escape_31, the feature dimension is 31, and the artifact
version is 2.

## Install

Copy the included uttam_dqn directory into another Bomberman repository:

    <bomberman-repository>/agent_code/uttam_dqn/

The gameplay runtime consists of callbacks.py, features.py, model.py, and
dqn_model_v2.pt. train.py is included for framework training compatibility.
This clean package intentionally excludes resumable checkpoints, metrics,
experiments, logs, backups, and caches.

## Test

From the Bomberman repository root, with its environment activated:

    python -m unittest discover -s agent_code/uttam_dqn/tests -t . -v

The tests use temporary directories and do not alter active artifacts.

## Play

From the Bomberman repository root:

    python main.py play --agents uttam_dqn rule_based_agent --n-rounds 10 --no-gui
