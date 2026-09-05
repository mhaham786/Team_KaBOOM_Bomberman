# DQN Agent

The agent uses the shared feature representation, metrics, plots, and
experiment directory structure.

Configure a run in `config.py`, then train from the repository root:

    python main.py play --agents Team_KaBOOM_Bomberman.dqn_agent --train 1 --no-gui --n-rounds 10000

Each run is stored as:

    experiments/<name>/latest.pt
    experiments/<name>/metadata.json
    experiments/<name>/train.jsonl
    experiments/<name>/plots/

Plot the configured run with:

    python -m agent_code.Team_KaBOOM_Bomberman.dqn_agent.plot

Add the Task 1 plots with:

    python -m agent_code.Team_KaBOOM_Bomberman.dqn_agent.plot --task 1
