"""Regression tests for the DQN and replay buffer."""

import unittest

from ..model import *

def _run_self_test() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    input_dim = 10
    model = build_dqn(input_dim)
    batch = torch.zeros(4, input_dim)
    output = model(batch)
    assert output.shape == (4, N_ACTIONS), output.shape
    assert model(torch.zeros(input_dim)).shape == (1, N_ACTIONS)

    custom_model = build_dqn(input_dim, hidden_dims=(64, 32))
    assert custom_model(torch.zeros(2, input_dim)).shape == (2, N_ACTIONS)

    try:
        DQN(input_dim, n_actions=5)
    except ValueError:
        pass
    else:
        raise AssertionError("DQN accepted non-six output size")

    buffer = ReplayBuffer(capacity=2)
    state0 = np.arange(input_dim, dtype=np.float32)
    state1 = state0 + 1
    state2 = state0 + 2
    valid_all = np.ones(N_ACTIONS, dtype=bool)
    valid_some = np.array([True, False, True, False, True, False])

    try:
        buffer.push(None, action_to_index("UP"), 1.0, state1, False, valid_all)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted transition with state=None")

    try:
        buffer.push(state0, action_to_index("UP"), 1.0, None, False, valid_all)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted non-terminal transition with next_state=None")

    buffer.push(state0, action_to_index("UP"), 1.0, state1, False, valid_all)
    buffer.push(state1, action_to_index("BOMB"), -1.0, state2, False, valid_some)
    buffer.push(state2, action_to_index("WAIT"), 0.5, None, True, valid_all)
    assert buffer.memory[-1].next_state is None
    assert buffer.memory[-1].done is True
    assert len(buffer) == 2
    assert buffer.can_sample(2)

    sample = buffer.sample(2)
    assert len(sample) == 2
    assert all(isinstance(t, Transition) for t in sample)

    tensors = transitions_to_tensors(sample)
    states, actions, rewards, next_states, dones, next_valid_masks = tensors
    assert states.shape == (2, input_dim)
    assert actions.shape == (2, 1)
    assert rewards.shape == (2, 1)
    assert next_states.shape == (2, input_dim)
    assert dones.shape == (2, 1)
    assert next_valid_masks.shape == (2, N_ACTIONS)
    assert next_valid_masks.dtype == torch.bool

    terminal_rows = dones.squeeze(1)
    assert terminal_rows.any()
    assert torch.equal(next_valid_masks[terminal_rows], torch.zeros((1, N_ACTIONS), dtype=torch.bool))
    assert torch.equal(next_states[terminal_rows], torch.zeros((1, input_dim)))

    next_q_values = torch.randn(2, N_ACTIONS)
    masked_next_q_values = next_q_values.masked_fill(~next_valid_masks, torch.finfo(next_q_values.dtype).min)
    safe_next_values = torch.where(
        dones.squeeze(1),
        torch.zeros(2, dtype=next_q_values.dtype),
        masked_next_q_values.max(dim=1).values,
    )
    assert torch.isfinite(safe_next_values).all()

    try:
        buffer.push(state0, 0, 0.0, state1, False, [True] * 5)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted mask with wrong length")

    try:
        buffer.push(state0, 0, 0.0, state1, False, [False] * N_ACTIONS)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted all-false non-terminal mask")

    assert index_to_action(0) == "UP"
    assert index_to_action(5) == "BOMB"
    for bad_index in (-1, N_ACTIONS):
        try:
            index_to_action(bad_index)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action index {bad_index}")

    print("model.py self-test passed")

class ModelRegressionTests(unittest.TestCase):
    def test_existing_regressions(self):
        _run_self_test()


if __name__ == "__main__":
    unittest.main()
