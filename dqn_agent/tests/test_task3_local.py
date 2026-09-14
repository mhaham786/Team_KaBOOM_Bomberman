"""Focused local tests for the final Task 3 runtime."""

import copy
import hashlib
import logging
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import events as e
from agent_code.common.features import classic_peace_improved_oc33
from agent_code.common.helpers import action_mask
from agent_code.common.metrics import Task3Metrics
from agent_code.common.rewards import (
    advanced_reward_shaping_dqn, classic_peace_rewards_dqn_improved,
)
from agent_code.dqn_agent import callbacks, config, task3, train
from agent_code.dqn_agent.model import DQN
from agent_code.dqn_agent.task3 import task3_features_oc33


ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / "agent_code/dqn_agent/experiments/dqn_task3_op/latest.pt"


def game_state(opponents=((6, 4),)):
    field = np.zeros((9, 9), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("dqn_agent", 0, True, (4, 4)),
        "others": [(f"enemy{i}", 0, True, p) for i, p in enumerate(opponents)],
        "coins": [(6, 6)], "bombs": [], "explosion_map": np.zeros_like(field),
    }


def corridor():
    state = game_state(((4, 2),))
    state["field"][:] = -1
    for tile in ((4, 4), (4, 3), (4, 2)):
        state["field"][tile] = 0
    return state


def fixed_policy(q_values):
    policy = DQN(33, 6)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.network[-1].bias.copy_(torch.tensor(q_values, dtype=torch.float32))
    return policy


def action_agent(q_values, training=False):
    return SimpleNamespace(
        train=training, policy_net=fixed_policy(q_values), device=torch.device("cpu"),
        rng=random.Random(42), epsilon=0.0,
        metrics=SimpleNamespace(record_decision_time=lambda _: None),
    )


class Task3LocalTests(unittest.TestCase):
    def training_agent(self, directory):
        source = Path(directory) / "source" / "latest.pt"
        source.parent.mkdir(parents=True)
        shutil.copy2(POLICY, source)
        self.enterContext(patch.object(config, "EXPERIMENTS_DIR", Path(directory)))
        self.enterContext(patch.object(config, "EXPERIMENT_NAME", "test_task3"))
        self.enterContext(patch.object(config, "INITIAL_WEIGHTS_EXPERIMENT", "source"))
        self.enterContext(patch("torch.cuda.is_available", return_value=False))
        agent = SimpleNamespace(train=True, logger=logging.getLogger("task3_test"))
        callbacks.setup(agent)
        train.setup_training(agent)
        return agent

    def test_all_features_exactly_match_task2(self):
        for opponents in (((6, 4),), ((4, 2),), ((2, 4), (7, 4))):
            state = game_state(opponents)
            state["field"][5, 4] = 1
            state["bombs"] = [((2, 2), 2)]
            features = task3_features_oc33(state)
            self.assertEqual(features.shape, (33,))
            self.assertEqual(features.dtype, np.float32)
            self.assertTrue(np.isfinite(features).all())
            np.testing.assert_array_equal(features, classic_peace_improved_oc33(state))
        self.assertIsNone(task3_features_oc33(None))

    def test_safe_offensive_bomb_preference_and_rank(self):
        agent = action_agent([0, 0, 0, 0, 3, 2])
        self.assertEqual(callbacks.act(agent, game_state()), "BOMB")
        agent = action_agent([0, 2, 0, 0, 3, 1])
        self.assertEqual(callbacks.act(agent, game_state()), "WAIT")

    def test_original_action_mask_and_blocked_movement(self):
        state = game_state()
        state["field"][4, 3] = 1
        state["bombs"] = [((5, 4), 2)]
        mask = action_mask(state)
        np.testing.assert_array_equal(mask, [False, False, True, True, True, True])
        agent = action_agent([10, 9, 8, 7, 6, 5])
        self.assertEqual(callbacks.act(agent, state), "DOWN")

    def test_unsafe_unavailable_and_non_offensive_bombs_are_not_preferred(self):
        agent = action_agent([0, 0, 0, 0, 3, 2])
        self.assertEqual(callbacks.act(agent, corridor()), "WAIT")
        unavailable = game_state()
        unavailable["self"] = ("dqn_agent", 0, False, (4, 4))
        self.assertEqual(callbacks.act(agent, unavailable), "WAIT")
        self.assertEqual(callbacks.act(agent, game_state(())), "WAIT")

    def test_unchanged_greedy_fallback(self):
        agent = action_agent([0, 4, 1, 0, 3, 2])
        self.assertEqual(callbacks.act(agent, game_state(())), "RIGHT")

    def test_training_action_selection_is_unchanged(self):
        agent = action_agent([0, 0, 0, 0, 3, 2], training=True)
        self.assertEqual(callbacks.act(agent, game_state()), "WAIT")

    def test_deterministic_ties_match_stable_ranking(self):
        tied = action_agent([0, 0, 0, 0, 3, 3])
        self.assertEqual([callbacks.act(tied, game_state()) for _ in range(5)], ["BOMB"] * 5)
        self.assertEqual(callbacks.act(tied, game_state(())), "WAIT")

    def test_safe_selected_action_remains(self):
        ranked = torch.tensor([4, 2, 3, 0, 1, 5])
        with patch.object(task3, "action_is_threatened", return_value=True), patch.object(
            task3, "safe_routes", return_value={4: True, 2: True}
        ):
            self.assertEqual(
                task3.choose_safe_action(game_state(), np.ones(6, bool), ranked, 4),
                4,
            )

    def test_unsafe_action_uses_safe_alternative_or_original_fallback(self):
        ranked = torch.tensor([4, 2, 3, 0, 1, 5])
        with patch.object(task3, "action_is_threatened", return_value=True), patch.object(
            task3, "safe_routes", return_value={4: False, 2: True, 3: True}
        ):
            self.assertEqual(
                task3.choose_safe_action(game_state(), np.ones(6, bool), ranked, 4),
                2,
            )
        with patch.object(task3, "action_is_threatened", return_value=True), patch.object(
            task3, "safe_routes", return_value={4: False, 2: False}
        ):
            self.assertEqual(
                task3.choose_safe_action(game_state(), np.ones(6, bool), ranked, 4),
                4,
            )

    def test_task3_metrics_remain_selected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "audit") as directory:
            agent = self.training_agent(directory)
            self.assertIsInstance(agent.metrics, Task3Metrics)
            self.assertEqual(len(agent.replay_buffer), 0)
            self.assertEqual(agent.optimizer.state_dict()["state"], {})

    def test_policy_checkpoint_load_and_architecture(self):
        agent = SimpleNamespace(train=False, logger=logging.getLogger("task3_eval"))
        callbacks.setup(agent)
        expected = torch.load(POLICY, map_location="cpu", weights_only=True)["policy_state"]
        for key, value in expected.items():
            self.assertTrue(torch.equal(agent.policy_net.state_dict()[key], value))
        layers = [m for m in agent.policy_net.modules() if isinstance(m, torch.nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in layers], [(33, 128), (128, 128), (128, 6)])
        self.assertEqual(config.ACTIONS, ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"))

    def test_transition_uses_original_mask_and_rewards(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "audit") as directory:
            agent = self.training_agent(directory)
            old = game_state()
            new = copy.deepcopy(old)
            new["step"] = 2
            events = [e.KILLED_OPPONENT, e.BOMB_DROPPED]
            agent.transition.store(old, "BOMB", new, events)
            replay = agent.replay_buffer.memory[-1]
            np.testing.assert_array_equal(replay.state, classic_peace_improved_oc33(old))
            np.testing.assert_array_equal(replay.next_valid_mask, action_mask(new))
            expected = classic_peace_rewards_dqn_improved(events)
            expected += advanced_reward_shaping_dqn("BOMB", old, new, events, config.GAMMA)
            self.assertAlmostEqual(replay.reward, expected)

    def test_config_protects_policy_checkpoint_from_training(self):
        before = hashlib.sha256(POLICY.read_bytes()).hexdigest()
        agent = SimpleNamespace(train=True, logger=logging.getLogger("protected_training"))
        callbacks.setup(agent)
        with self.assertRaises(FileExistsError):
            train.setup_training(agent)
        self.assertEqual(hashlib.sha256(POLICY.read_bytes()).hexdigest(), before)

    def test_runtime_contains_no_rejected_hooks(self):
        paths = [
            ROOT / "agent_code/dqn_agent/callbacks.py",
            ROOT / "agent_code/dqn_agent/transition.py",
            ROOT / "agent_code/dqn_agent/config.py",
        ]
        text = "\n".join(path.read_text() for path in paths)
        for rejected in (
            "task3_action_mask", "task3_event_reward", "task3_shaping_reward",
            "dqn_task3_mask_reward_oc33_smoke_seed42",
        ):
            self.assertNotIn(rejected, text)
        runtime = "\n".join(
            (ROOT / "agent_code/dqn_agent" / name).read_text()
            for name in ("callbacks.py", "config.py", "task3.py")
        ).lower()
        self.assertNotIn("top2", runtime)


if __name__ == "__main__":
    unittest.main()
