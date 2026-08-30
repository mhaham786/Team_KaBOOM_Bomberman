"""Regression tests for DQN callbacks."""

import logging
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .. import callbacks as callbacks_module
from ..callbacks import (
    ACTIONS, ARTIFACT_VERSION, CPU_DEVICE, FEATURE_DIM, FEATURE_SCHEMA,
    MODEL_FILENAME, N_ACTIONS, act, build_dqn, index_to_action,
    state_to_features, valid_action_mask, _default_model_path,
    _load_policy_state, _sample_legal_action, _select_greedy_action,
)
from .fixtures import bordered_field

class _FixedQPolicy(nn.Module):
    """Tiny deterministic policy used only by executable self-tests."""

    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("q_values", torch.as_tensor(q_values, dtype=torch.float32))

    def forward(self, states):
        batch_size = states.shape[0]
        return self.q_values.unsqueeze(0).repeat(batch_size, 1)


def _make_test_state():
    field = bordered_field(7)
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("uttam", 0, True, (3, 3)),
        "others": [],
        "bombs": [],
        "coins": [(4, 3)],
        "user_input": None,
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


def _make_blocked_test_state():
    state = _make_test_state()
    field = state["field"].copy()
    field[3, 2] = 1
    state["field"] = field
    state["bombs"] = [((4, 3), 3)]
    state["others"] = [("enemy", 0, True, (2, 3))]
    state["coins"] = []
    state["explosion_map"] = np.zeros_like(field, dtype=np.float32)
    return state


def _run_self_test() -> None:
    state = _make_test_state()
    features = state_to_features(state)
    assert features.shape == (FEATURE_DIM,)

    policy = build_dqn(FEATURE_DIM)
    with torch.no_grad():
        output = policy(torch.as_tensor(features, dtype=torch.float32))
    assert output.shape == (1, N_ACTIONS)

    legal_mask = np.array([False, True, False, False, True, False], dtype=np.bool_)
    fixed_policy = _FixedQPolicy([100.0, 1.0, 50.0, 25.0, 0.5, 75.0])
    action_index, q_values = _select_greedy_action(fixed_policy, features, legal_mask)
    assert action_index == 1
    assert legal_mask[action_index]
    assert q_values[0] == np.float32(100.0)
    assert index_to_action(action_index) in ACTIONS

    rng = random.Random(0)
    sampled = {_sample_legal_action(legal_mask, rng) for _ in range(100)}
    assert sampled <= {1, 4}
    assert sampled == {1, 4}

    blocked_state = _make_blocked_test_state()
    blocked_mask = valid_action_mask(blocked_state)
    assert not blocked_mask[0]
    assert not blocked_mask[1]
    assert not blocked_mask[3]

    fake_self = SimpleNamespace(
        train=True,
        epsilon=1.0,
        rng=random.Random(1),
        policy_net=fixed_policy,
        device=CPU_DEVICE,
        logger=logging.getLogger("callbacks_self_test"),
    )
    for _ in range(100):
        action = act(fake_self, blocked_state)
        assert action in ACTIONS
        assert blocked_mask[ACTIONS.index(action)]

    eval_self = SimpleNamespace(
        train=False,
        epsilon=0.0,
        rng=random.Random(2),
        policy_net=fixed_policy,
        device=CPU_DEVICE,
        logger=logging.getLogger("callbacks_self_test"),
    )
    assert act(eval_self, state) in ACTIONS

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        logger = logging.getLogger("callbacks_self_test")

        source_policy = build_dqn(FEATURE_DIM)
        good_path = tmpdir_path / MODEL_FILENAME
        torch.save(
            {
                "feature_schema": FEATURE_SCHEMA,
                "feature_dim": FEATURE_DIM,
                "artifact_version": ARTIFACT_VERSION,
                "state_dict": source_policy.state_dict(),
            },
            good_path,
        )
        loaded_policy = build_dqn(FEATURE_DIM)
        _load_policy_state(loaded_policy, good_path, logger)
        for source, loaded in zip(source_policy.parameters(), loaded_policy.parameters()):
            assert torch.equal(source, loaded)

        nonfinite_path = tmpdir_path / "nonfinite_model.pt"
        nonfinite_state = {
            name: value.detach().clone() for name, value in source_policy.state_dict().items()
        }
        first_name = next(iter(nonfinite_state))
        nonfinite_state[first_name].reshape(-1)[0] = float("nan")
        torch.save(
            {
                "feature_schema": FEATURE_SCHEMA,
                "feature_dim": FEATURE_DIM,
                "artifact_version": ARTIFACT_VERSION,
                "state_dict": nonfinite_state,
            },
            nonfinite_path,
        )
        try:
            _load_policy_state(build_dqn(FEATURE_DIM), nonfinite_path, logger)
        except RuntimeError as exc:
            assert "NaN or infinity" in str(exc)
        else:
            raise AssertionError("non-finite V2 model parameters were accepted")

        bad_schema_path = tmpdir_path / "bad_schema.pt"
        torch.save(
            {
                "feature_schema": "uttam_dqn_v1_31",
                "feature_dim": FEATURE_DIM,
                "artifact_version": ARTIFACT_VERSION,
                "state_dict": source_policy.state_dict(),
            },
            bad_schema_path,
        )
        try:
            _load_policy_state(build_dqn(FEATURE_DIM), bad_schema_path, logger)
        except RuntimeError as exc:
            assert "incompatible feature_schema" in str(exc)
        else:
            raise AssertionError("mismatched V2 model schema did not raise RuntimeError")

        v1_raw_path = tmpdir_path / "dqn_model.pt"
        torch.save(source_policy.state_dict(), v1_raw_path)
        try:
            _load_policy_state(build_dqn(FEATURE_DIM), v1_raw_path, logger)
        except RuntimeError as exc:
            assert "feature_schema" in str(exc)
        else:
            raise AssertionError("unversioned V1 raw state_dict was accepted")

    resolved_model_path = _default_model_path()
    assert resolved_model_path.name == MODEL_FILENAME
    assert resolved_model_path.name == "dqn_model_v2.pt"
    assert resolved_model_path.name != "dqn_model.pt"
    assert resolved_model_path.parent == Path(callbacks_module.__file__).resolve().parent
    print("callbacks.py V2 schema regression passed")
    print("callbacks.py self-test passed")

class CallbacksRegressionTests(unittest.TestCase):
    def test_existing_regressions(self):
        _run_self_test()


if __name__ == "__main__":
    unittest.main()
