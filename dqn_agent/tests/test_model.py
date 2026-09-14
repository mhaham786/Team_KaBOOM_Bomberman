import unittest

import numpy as np
import torch

from .. import config
from ..model import DQN
from ..replay_buffer import ReplayBuffer, ReplayTransition, transitions_to_tensors


class ModelTests(unittest.TestCase):
    def test_architecture_and_output_shapes(self):
        model = DQN(config.OBSERVATION_COUNT, len(config.ACTIONS))
        self.assertEqual(tuple(model(torch.zeros(4, 33)).shape), (4, 6))
        self.assertEqual(tuple(model(torch.zeros(33)).shape), (1, 6))
        layers = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                         [(33, 128), (128, 128), (128, 6)])

    def test_replay_validation_and_capacity(self):
        replay = ReplayBuffer(2)
        state = np.arange(33, dtype=np.float32)
        mask = np.ones(6, dtype=np.bool_)
        with self.assertRaises(ValueError):
            replay.push(None, 0, 1.0, state, False, mask)
        with self.assertRaises(ValueError):
            replay.push(state, 0, 1.0, None, False, mask)
        with self.assertRaises(ValueError):
            replay.push(state, 6, 1.0, state, False, mask)
        with self.assertRaises(ValueError):
            replay.push(state, 0, 1.0, state, False, mask[:5])
        with self.assertRaises(ValueError):
            replay.push(state, 0, 1.0, state, False, np.zeros(6, dtype=np.bool_))
        replay.push(state, 0, 1.0, state + 1, False, mask)
        replay.push(state + 1, 5, -1.0, state + 2, False, mask)
        replay.push(state + 2, 4, 0.5, None, True, mask)
        self.assertEqual(len(replay), 2)
        self.assertTrue(replay.can_sample(2))
        self.assertIsInstance(replay.memory[-1], ReplayTransition)
        self.assertTrue(replay.memory[-1].done)
        self.assertIsNone(replay.memory[-1].next_state)

    def test_tensor_conversion_and_terminal_safety(self):
        state = np.arange(33, dtype=np.float32)
        mask = np.array([True, False, True, False, True, False])
        replay = ReplayBuffer(2)
        replay.push(state, 2, 1.0, state + 1, False, mask)
        replay.push(state + 1, 4, -1.0, None, True, np.ones(6, dtype=np.bool_))
        tensors = transitions_to_tensors(list(replay.memory), torch.device("cpu"))
        states, actions, rewards, next_states, dones, next_masks = tensors
        self.assertEqual(tuple(states.shape), (2, 33))
        self.assertEqual(tuple(actions.shape), (2, 1))
        self.assertEqual(tuple(rewards.shape), (2, 1))
        self.assertEqual(tuple(next_states.shape), (2, 33))
        self.assertEqual(tuple(dones.shape), (2, 1))
        self.assertEqual(tuple(next_masks.shape), (2, 6))
        self.assertEqual(next_masks.dtype, torch.bool)
        self.assertTrue(torch.equal(next_states[1], torch.zeros(33)))
        self.assertFalse(next_masks[1].any())
        next_q = torch.randn(2, 6)
        masked = next_q.masked_fill(~next_masks, torch.finfo(next_q.dtype).min)
        future = torch.where(dones, torch.zeros((2, 1)), masked.max(1, keepdim=True).values)
        self.assertTrue(torch.isfinite(future).all())

    def test_action_order_is_fixed(self):
        self.assertEqual(config.ACTIONS, ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"))


if __name__ == "__main__":
    unittest.main()
