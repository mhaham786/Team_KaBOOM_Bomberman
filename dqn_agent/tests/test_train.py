import hashlib
import logging
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import events as e
from ...common.experiment import ExperimentRun
from ...common.metrics import Task3Metrics
from .. import config
from ..experiment import save_checkpoint
from ..model import DQN
from ..replay_buffer import ReplayBuffer
from ..train import setup_training
from ..trainer import DQNTrainer
from ..transition import Transition
from .fixtures import bordered_field

def make_state(step=1, position=(3, 3), others=()):
    field = bordered_field(7)
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("dqn", 0, True, position),
        "others": list(others),
        "bombs": [],
        "coins": [(4, 3)],
        "user_input": None,
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }

def make_agent(root, name):
    return SimpleNamespace(
        train=True,
        logger=logging.getLogger(f"train-test-{name}"),
        device=torch.device("cpu"),
        rng=random.Random(),
        epsilon=config.EPSILON_START,
        policy_net=DQN(config.OBSERVATION_COUNT, len(config.ACTIONS)),
        run=ExperimentRun(root, name),
    )

def configured(root, name, resume=False, initial=None):
    return patch.multiple(
        config,
        EXPERIMENTS_DIR=Path(root),
        EXPERIMENT_NAME=name,
        RESUME_TRAINING=resume,
        RESTART_EXPERIMENT=False,
        INITIAL_WEIGHTS_EXPERIMENT=initial,
    )

def make_trainer(capacity=2):
    policy = DQN(33, 6)
    target = DQN(33, 6)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.LEARNING_RATE)
    trainer = DQNTrainer(policy, target, optimizer, torch.device("cpu"), random.Random(42))
    return policy, target, optimizer, trainer, ReplayBuffer(capacity)

class TrainingTests(unittest.TestCase):
    def test_epsilon_schedule(self):
        self.assertEqual(Transition.epsilon_for(0), config.EPSILON_START)
        middle = Transition.epsilon_for(config.EPSILON_DECAY_STEPS // 2)
        self.assertGreater(middle, config.EPSILON_END)
        self.assertLess(middle, config.EPSILON_START)
        self.assertAlmostEqual(Transition.epsilon_for(config.EPSILON_DECAY_STEPS), config.EPSILON_END)
    def test_existing_experiment_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protected").mkdir()
            agent = make_agent(root, "protected")
            with configured(root, "protected"):
                with self.assertRaises(FileExistsError):
                    setup_training(agent)
    def test_policy_only_warm_start_is_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run = ExperimentRun(root, "source")
            source_run.path.mkdir()
            torch.manual_seed(7)
            source = DQN(33, 6)
            torch.save({"policy_state": source.state_dict()}, source_run.latest_path)
            before = hashlib.sha256(source_run.latest_path.read_bytes()).hexdigest()
            agent = make_agent(root, "new")
            with configured(root, "new", initial="source"):
                setup_training(agent)
            for expected, policy, target in zip(
                source.parameters(), agent.policy_net.parameters(), agent.target_net.parameters()
            ):
                self.assertTrue(torch.equal(expected, policy))
                self.assertTrue(torch.equal(policy, target))
            self.assertEqual(agent.optimizer.state, {})
            self.assertEqual(len(agent.replay_buffer), 0)
            self.assertEqual(agent.episode, 0)
            self.assertIsInstance(agent.metrics, Task3Metrics)
            self.assertEqual(before, hashlib.sha256(source_run.latest_path.read_bytes()).hexdigest())
    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = make_agent(root, "resume")
            with configured(root, "resume"):
                setup_training(agent)
                state = np.zeros(33, dtype=np.float32)
                mask = np.ones(6, dtype=np.bool_)
                agent.replay_buffer.push(state, 1, 2.0, state, False, mask)
                agent.total_transitions = 1
                agent.rng.random()
                agent.replay_rng.random()
                action_state = agent.rng.getstate()
                replay_state = agent.replay_rng.getstate()
                save_checkpoint(agent)
            restored = make_agent(root, "resume")
            with configured(root, "resume", resume=True):
                setup_training(restored)
            self.assertEqual(len(restored.replay_buffer), 1)
            self.assertEqual(restored.total_transitions, 1)
            self.assertEqual(restored.episode, 0)
            self.assertTrue(np.array_equal(restored.replay_buffer.memory[0].state, state))
            self.assertEqual(restored.rng.getstate(), action_state)
            self.assertEqual(restored.replay_rng.getstate(), replay_state)
    def test_pending_terminal_transition_is_stored_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = make_agent(root, "pending")
            with configured(root, "pending"):
                setup_training(agent)
                old = make_state(step=1)
                new = make_state(step=2, position=(4, 3))
                agent.transition.record(old, "RIGHT", new, [e.MOVED_RIGHT])
                self.assertEqual(len(agent.replay_buffer), 0)
                agent.transition.finish(old, "RIGHT", [e.SURVIVED_ROUND])
            self.assertEqual(len(agent.replay_buffer), 1)
            self.assertEqual(agent.total_transitions, 1)
            self.assertTrue(agent.replay_buffer.memory[0].done)
            self.assertTrue(np.isfinite(agent.replay_buffer.memory[0].reward))
    def test_nonmatching_final_stores_nonterminal_and_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = make_agent(root, "death")
            with configured(root, "death"):
                setup_training(agent)
                old = make_state(step=1)
                new = make_state(step=2, position=(4, 3))
                agent.transition.record(old, "RIGHT", new, [e.MOVED_RIGHT])
                agent.transition.finish(new, "WAIT", [e.GOT_KILLED])
            self.assertEqual(len(agent.replay_buffer), 2)
            self.assertFalse(agent.replay_buffer.memory[0].done)
            self.assertTrue(agent.replay_buffer.memory[1].done)
            self.assertEqual(agent.total_transitions, 2)
    def test_finite_optimization_and_target_sync(self):
        torch.manual_seed(3)
        policy, target, _, trainer, replay = make_trainer(4)
        state = np.zeros(33, dtype=np.float32)
        mask = np.ones(6, dtype=np.bool_)
        replay.push(state, 0, 1.0, state + 1, False, mask)
        replay.push(state + 1, 5, -1.0, None, True, mask)
        before = [value.detach().clone() for value in policy.parameters()]
        with patch.multiple(config, BATCH_SIZE=2, MIN_REPLAY_SIZE=2,
                            TARGET_UPDATE_INTERVAL=1, TRAIN_EVERY=1, COLLECT_ONLY=False):
            loss = trainer.update(replay, 2)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(left, right) for left, right in zip(before, policy.parameters())))
        for left, right in zip(policy.parameters(), target.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(policy.training)
    def test_train_every_gates_updates(self):
        _, _, _, trainer, replay = make_trainer()
        state = np.zeros(33, dtype=np.float32)
        replay.push(state, 0, 1.0, state + 1, False, np.ones(6, dtype=np.bool_))
        with patch.multiple(config, BATCH_SIZE=1, MIN_REPLAY_SIZE=1,
                            TRAIN_EVERY=4, COLLECT_ONLY=False):
            self.assertIsNone(trainer.update(replay, 1))
            self.assertIsNone(trainer.update(replay, 2))
            self.assertIsNone(trainer.update(replay, 3))
            self.assertTrue(np.isfinite(trainer.update(replay, 4)))
    def test_collect_only_preserves_parameters(self):
        policy, target, optimizer, trainer, replay = make_trainer(1)
        state = np.zeros(33, dtype=np.float32)
        replay.push(state, 0, 0.0, state, False, np.ones(6, dtype=np.bool_))
        before = [value.detach().clone() for value in policy.parameters()]
        target_before = [value.detach().clone() for value in target.parameters()]
        optimizer_before = optimizer.state_dict()
        with patch.multiple(config, BATCH_SIZE=1, MIN_REPLAY_SIZE=1, COLLECT_ONLY=True):
            self.assertIsNone(trainer.update(replay, 1))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, policy.parameters())))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(target_before, target.parameters())))
        self.assertEqual(optimizer.state_dict(), optimizer_before)


if __name__ == "__main__":
    unittest.main()
