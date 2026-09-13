import logging
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from ...common.experiment import ExperimentRun
from ...common.features import classic_peace_improved_oc33
from ...common.helpers import action_mask
from .. import config
from ..callbacks import act, sample_legal_action, select_greedy_action, setup
from ..experiment import load_policy
from ..model import DQN
from .fixtures import bordered_field


class FixedPolicy(nn.Module):
    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("q_values", torch.tensor(q_values, dtype=torch.float32))

    def forward(self, states):
        return self.q_values.unsqueeze(0).repeat(states.shape[0], 1)


def make_state(blocked=False):
    field = bordered_field(7)
    others = []
    bombs = []
    coins = [(4, 3)]
    if blocked:
        field[3, 2] = 1
        bombs = [((4, 3), 3)]
        others = [("enemy", 0, True, (2, 3))]
        coins = []
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("dqn", 0, True, (3, 3)),
        "others": others,
        "bombs": bombs,
        "coins": coins,
        "user_input": None,
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


class CallbackTests(unittest.TestCase):
    def test_network_and_masked_greedy_selection(self):
        features = classic_peace_improved_oc33(make_state())
        policy = DQN(config.OBSERVATION_COUNT, len(config.ACTIONS))
        self.assertEqual(tuple(policy(torch.tensor(features)).shape), (1, 6))
        mask = np.array([False, True, False, False, True, False])
        fixed = FixedPolicy([100, 1, 50, 25, 0.5, 75])
        selected = select_greedy_action(fixed, features, mask, torch.device("cpu"))
        self.assertEqual(selected, 1)
        self.assertTrue(mask[selected])

    def test_sampling_and_act_respect_current_mask(self):
        state = make_state(True)
        mask = action_mask(state)
        rng = random.Random(0)
        sampled = {sample_legal_action(mask, rng) for _ in range(100)}
        self.assertTrue(sampled)
        self.assertTrue(all(mask[index] for index in sampled))
        agent = SimpleNamespace(
            train=True,
            epsilon=1.0,
            rng=random.Random(1),
            policy_net=FixedPolicy([100, 1, 50, 25, 0.5, 75]),
            device=torch.device("cpu"),
            metrics=SimpleNamespace(record_decision_time=lambda duration: None),
        )
        for _ in range(100):
            self.assertTrue(mask[config.ACTIONS.index(act(agent, state))])

    def test_policy_loading_and_incompatible_shapes(self):
        source = DQN(33, 6)
        target = DQN(33, 6)
        load_policy(target, {"policy_state": source.state_dict()})
        for left, right in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(left, right))
        with self.assertRaises(RuntimeError):
            load_policy(target, {"policy_state": DQN(9, 6).state_dict()})

    def test_setup_loads_current_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = ExperimentRun(root, "test")
            run.path.mkdir()
            source = DQN(33, 6)
            torch.save({"policy_state": source.state_dict()}, run.latest_path)
            agent = SimpleNamespace(train=False, logger=logging.getLogger("callbacks-test"))
            with patch.multiple(config, EXPERIMENTS_DIR=root, EXPERIMENT_NAME="test"):
                setup(agent)
            for left, right in zip(source.parameters(), agent.policy_net.parameters()):
                self.assertTrue(torch.equal(left, right))
            self.assertEqual(agent.run.latest_path, run.latest_path)
            self.assertEqual(agent.epsilon, 0.0)
            self.assertFalse(agent.policy_net.training)


if __name__ == "__main__":
    unittest.main()
