PYTHON ?= python
AGENT ?= Team_KaBOOM_Bomberman.ppo_agent
ROUNDS ?= 10000
SEED ?= 42

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/../../..)
RUN = cd $(PROJECT_ROOT) && $(PYTHON) main.py play --no-gui --n-rounds $(ROUNDS) --seed $(SEED)

.PHONY: train-scene-1 train-scene-2 train-scene-3 train-scene-4 plot

train-scene-1:
	$(RUN) --agents $(AGENT) --train 1 --scenario coin-heaven

train-scene-2:
	$(RUN) --agents $(AGENT) --train 1 --scenario classic

train-scene-3:
	$(RUN) --agents $(AGENT) peaceful_agent coin_collector_agent --train 1 --scenario classic

train-scene-4:
	$(RUN) --agents $(AGENT) rule_based_agent --train 1 --scenario classic

plot:
	@test -n "$(EXPERIMENT)" || (echo "Usage: make plot EXPERIMENT=<name>"; exit 1)
	cd $(PROJECT_ROOT) && $(PYTHON) -m agent_code.Team_KaBOOM_Bomberman.common.plots.plot $(MAKEFILE_DIR)/experiments/$(EXPERIMENT)/metrics/train.jsonl
