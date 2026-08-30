"""Regression tests for DQN rewards, optimization, and persistence."""

import logging
import unittest
from types import SimpleNamespace

from ..train import *
from ..train import (
    _checkpoint_dict, _epsilon_by_transition_count, _learning_rate_from_env,
    _reset_network_parameters, _sample_replay_batch, _save_policy_model,
    _save_training_checkpoint, _validate_artifact_metadata,
    _validate_metrics_history,
)
from .fixtures import bordered_field

def _snapshot_value(value):
    """Recursively clone tensor values for self-test state comparisons."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    return value


def _values_equal(left, right) -> bool:
    """Recursively compare values that may contain tensors."""
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(_values_equal(l_item, r_item) for l_item, r_item in zip(left, right))
        )
    if isinstance(left, tuple):
        return (
            isinstance(right, tuple)
            and len(left) == len(right)
            and all(_values_equal(l_item, r_item) for l_item, r_item in zip(left, right))
        )
    return left == right


def _optimizer_state_snapshot(optimizer: torch.optim.Optimizer):
    """Return a detached copy of optimizer state for self-test comparisons."""
    return _snapshot_value(optimizer.state_dict())


def _optimizer_states_equal(left, right) -> bool:
    """Compare optimizer state snapshots containing tensors and scalars."""
    return _values_equal(left, right)


def _make_test_state(
    position=(3, 3),
    coin=(4, 3),
    bomb_available=True,
    score=0,
    round_id=1,
    step=1,
):
    """Create a compact synthetic game state for executable self-tests."""
    field = bordered_field(7)
    field[3, 2] = 1
    return {
        "round": round_id,
        "step": step,
        "field": field,
        "self": ("uttam", score, bomb_available, position),
        "others": [("enemy", 0, True, (5, 3))],
        "bombs": [((3, 5), 2)],
        "coins": [coin] if coin is not None else [],
        "user_input": None,
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


def _make_useless_bomb_state(**kwargs):
    """Create a state where a placed bomb has no immediate crate/opponent target."""
    state = _make_test_state(coin=None, **kwargs)
    field = state["field"].copy()
    field[3, 2] = 0
    state["field"] = field
    state["others"] = []
    state["bombs"] = []
    state["explosion_map"] = np.zeros_like(field, dtype=np.float32)
    return state


def _make_unsafe_bomb_state(*, useful=False, **kwargs):
    """Create a trapped corridor state with no safe bomb escape."""
    state = _make_useless_bomb_state(**kwargs)
    field = np.full((7, 7), -1, dtype=np.int64)
    field[1, 1] = 0
    field[2, 1] = 0
    field[3, 1] = 0
    if useful:
        field[4, 1] = 1
    else:
        field[4, 1] = 0
    state["field"] = field
    state["self"] = (state["self"][0], state["self"][1], state["self"][2], (1, 1))
    state["others"] = []
    state["bombs"] = []
    state["coins"] = []
    state["explosion_map"] = np.zeros_like(field, dtype=np.float32)
    return state


def _make_training_self(tmpdir: Path, suffix: str = ""):
    """Create a fake callback self object for isolated train.py self-tests."""
    logger = logging.getLogger("train_self_test" + suffix)
    logger.addHandler(logging.NullHandler())
    agent_dir = tmpdir / (suffix.strip("_") or "agent")
    agent_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        train=True,
        logger=logger,
        device=torch.device("cpu"),
        rng=random.Random(),
        epsilon=EPSILON_START,
        policy_net=build_dqn(FEATURE_DIM),
        model_path=agent_dir / MODEL_FILENAME,
        checkpoint_path=agent_dir / CHECKPOINT_FILENAME,
        metrics_path=agent_dir / METRICS_FILENAME,
    )


def _fill_replay(agent, old_features, next_features, next_mask) -> None:
    """Fill replay to MIN_REPLAY_SIZE for optimization self-tests."""
    while len(agent.replay_buffer) < MIN_REPLAY_SIZE:
        agent.replay_buffer.push(
            old_features,
            action_to_index("RIGHT"),
            1.0,
            next_features,
            False,
            next_mask,
        )


def _assert_finite_module(module: nn.Module) -> None:
    """Assert all parameters are finite."""
    for parameter in module.parameters():
        assert torch.isfinite(parameter).all()


def _run_self_test() -> None:
    assert reward_from_events([e.COIN_COLLECTED, e.CRATE_DESTROYED, e.MOVED_UP]) == 12.0
    assert reward_from_events([e.INVALID_ACTION, e.KILLED_SELF, e.WAITED]) == -55.1

    old_state = _make_test_state(step=1)
    next_state = _make_test_state(position=(4, 3), coin=(4, 3), score=s.REWARD_COIN, step=2)
    old_features = advanced_features_oc31(old_state)
    next_features = advanced_features_oc31(next_state)
    next_mask = valid_action_mask(
        np.asarray(next_state["field"]),
        tuple(next_state["self"][3]),
        next_state["self"][2],
        bomb_positions(next_state.get("bombs", ())),
        opponent_positions(next_state.get("others", ())),
    )
    assert np.isfinite(potential_from_features(old_features))
    shaping = potential_shaping(old_features, next_features)
    assert -2.0 <= shaping <= 2.0
    terminal_shaping = potential_shaping(old_features, None)
    assert -2.0 <= terminal_shaping <= 2.0

    useless_bomb_state = _make_useless_bomb_state(step=1)
    useless_bomb_features = advanced_features_oc31(useless_bomb_state)
    crate_bomb_state = _make_test_state(step=1)
    crate_bomb_features = advanced_features_oc31(crate_bomb_state)
    opponent_bomb_state = _make_useless_bomb_state(step=1)
    opponent_bomb_state["others"] = [("enemy", 0, True, (5, 3))]
    opponent_bomb_features = advanced_features_oc31(opponent_bomb_state)
    unsafe_useless_state = _make_unsafe_bomb_state(useful=False, step=1)
    unsafe_useless_features = advanced_features_oc31(unsafe_useless_state)
    unsafe_useful_state = _make_unsafe_bomb_state(useful=True, step=1)
    unsafe_useful_features = advanced_features_oc31(unsafe_useful_state)
    assert useless_bomb_features[29] == 0.0
    assert useless_bomb_features[30] == 0.0
    assert action_shaping_reward("BOMB", useless_bomb_features, [e.BOMB_DROPPED], useless_bomb_state) == USELESS_BOMB_PENALTY
    assert action_shaping_reward("BOMB", crate_bomb_features, [e.BOMB_DROPPED], crate_bomb_state) == 0.0
    assert action_shaping_reward("BOMB", opponent_bomb_features, [e.BOMB_DROPPED], opponent_bomb_state) == 0.0
    assert action_shaping_reward("BOMB", unsafe_useful_features, [e.BOMB_DROPPED], unsafe_useful_state) == UNSAFE_BOMB_PENALTY
    assert action_shaping_reward("BOMB", unsafe_useless_features, [e.BOMB_DROPPED], unsafe_useless_state) == USELESS_BOMB_PENALTY + UNSAFE_BOMB_PENALTY
    assert action_shaping_reward("BOMB", unsafe_useless_features, [e.INVALID_ACTION], unsafe_useless_state) == 0.0

    assert _epsilon_by_transition_count(0) == EPSILON_START
    middle_epsilon = _epsilon_by_transition_count(EPSILON_DECAY_STEPS // 2)
    assert EPSILON_END < middle_epsilon < EPSILON_START
    assert np.isclose(_epsilon_by_transition_count(EPSILON_DECAY_STEPS), EPSILON_END)
    assert np.isclose(_epsilon_by_transition_count(EPSILON_DECAY_STEPS * 2), EPSILON_END)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        bad_path_agent = _make_training_self(tmpdir, "_bad_path")
        bad_path_agent.model_path = bad_path_agent.model_path.with_name("dqn_model.pt")
        try:
            setup_training(bad_path_agent)
        except CheckpointError as exc:
            assert "dqn_model_v2.pt" in str(exc)
        else:
            raise AssertionError("setup_training accepted a non-V2 model filename")

        fresh_a = _make_training_self(tmpdir, "_fresh_a")
        fresh_b = _make_training_self(tmpdir, "_fresh_b")
        torch.manual_seed(999)
        _reset_network_parameters(fresh_a.policy_net)
        torch.manual_seed(123)
        _reset_network_parameters(fresh_b.policy_net)
        setup_training(fresh_a)
        setup_training(fresh_b)
        for left, right in zip(fresh_a.policy_net.parameters(), fresh_b.policy_net.parameters()):
            assert torch.equal(left, right)

        tmpdir = Path(tmp)

        survivor = _make_training_self(tmpdir, "_survivor")
        setup_training(survivor)
        game_events_occurred(survivor, old_state, "RIGHT", next_state, [e.COIN_COLLECTED])
        assert len(survivor.replay_buffer) == 0
        assert survivor.total_transitions == 0
        end_of_round(survivor, old_state, "RIGHT", [e.COIN_COLLECTED, e.SURVIVED_ROUND])
        assert len(survivor.replay_buffer) == 1
        survivor_transition = survivor.replay_buffer.memory[-1]
        assert survivor_transition.done is True
        assert survivor_transition.next_state is None
        assert survivor.round_steps == 0
        survivor_metrics = survivor.metrics_path.read_text(encoding="utf-8").strip().splitlines()
        survivor_row = dict(zip(METRICS_COLUMNS, survivor_metrics[1].split(",")))
        assert int(survivor_row["steps"]) == 1
        assert int(survivor_row["coins_collected"]) == 1
        assert int(survivor_row["survived"]) == 1
        assert int(survivor_row["score"]) == s.REWARD_COIN
        assert float(survivor_row["event_reward"]) == EVENT_REWARDS[e.COIN_COLLECTED] + EVENT_REWARDS[e.SURVIVED_ROUND]

        survivor_bomb = _make_training_self(tmpdir, "_survivor_bomb")
        setup_training(survivor_bomb)
        useless_old = _make_unsafe_bomb_state(useful=False, step=1, score=0)
        useless_new = _make_unsafe_bomb_state(useful=False, step=2, score=0)
        game_events_occurred(survivor_bomb, useless_old, "BOMB", useless_new, [e.BOMB_DROPPED])
        end_of_round(survivor_bomb, useless_old, "BOMB", [e.BOMB_DROPPED, e.SURVIVED_ROUND])
        assert len(survivor_bomb.replay_buffer) == 1
        survivor_bomb_metrics = survivor_bomb.metrics_path.read_text(encoding="utf-8").strip().splitlines()
        survivor_bomb_row = dict(zip(METRICS_COLUMNS, survivor_bomb_metrics[1].split(",")))
        assert int(survivor_bomb_row["steps"]) == 1
        assert float(survivor_bomb_row["shaping_reward"]) == USELESS_BOMB_PENALTY + UNSAFE_BOMB_PENALTY

        death_agent = _make_training_self(tmpdir, "_death")
        setup_training(death_agent)
        previous_old = _make_test_state(step=1, score=0)
        previous_new = _make_test_state(position=(4, 3), coin=None, score=0, step=2)
        final_old = _make_test_state(position=(4, 3), coin=None, score=s.REWARD_COIN, step=2)
        game_events_occurred(death_agent, previous_old, "RIGHT", previous_new, [e.MOVED_RIGHT])
        end_of_round(death_agent, final_old, "WAIT", [e.COIN_COLLECTED, e.KILLED_OPPONENT, e.GOT_KILLED])
        assert len(death_agent.replay_buffer) == 2
        assert death_agent.replay_buffer.memory[0].done is False
        assert death_agent.replay_buffer.memory[1].done is True
        death_metrics = death_agent.metrics_path.read_text(encoding="utf-8").strip().splitlines()
        death_row = dict(zip(METRICS_COLUMNS, death_metrics[1].split(",")))
        expected_death_score = s.REWARD_COIN + s.REWARD_COIN + s.REWARD_KILL
        assert int(death_row["score"]) == expected_death_score
        assert int(death_row["steps"]) == 2
        assert int(death_row["coins_collected"]) == 1
        assert int(death_row["opponents_killed"]) == 1
        assert int(death_row["got_killed"]) == 1

        death_bomb = _make_training_self(tmpdir, "_death_bomb")
        setup_training(death_bomb)
        prior_old = _make_useless_bomb_state(step=1, score=0)
        prior_new = _make_useless_bomb_state(position=(4, 3), step=2, score=0)
        final_bomb_state = _make_unsafe_bomb_state(useful=False, step=2, score=0)
        game_events_occurred(death_bomb, prior_old, "RIGHT", prior_new, [e.MOVED_RIGHT])
        end_of_round(death_bomb, final_bomb_state, "BOMB", [e.BOMB_DROPPED, e.GOT_KILLED])
        assert len(death_bomb.replay_buffer) == 2
        death_bomb_metrics = death_bomb.metrics_path.read_text(encoding="utf-8").strip().splitlines()
        death_bomb_row = dict(zip(METRICS_COLUMNS, death_bomb_metrics[1].split(",")))
        assert int(death_bomb_row["steps"]) == 2
        assert float(death_bomb_row["shaping_reward"]) == USELESS_BOMB_PENALTY + UNSAFE_BOMB_PENALTY

        rng_agent_a = _make_training_self(tmpdir, "_rng_a")
        rng_agent_b = _make_training_self(tmpdir, "_rng_b")
        setup_training(rng_agent_a)
        setup_training(rng_agent_b)
        for reward in range(BATCH_SIZE + 8):
            for agent in (rng_agent_a, rng_agent_b):
                agent.replay_buffer.push(
                    old_features,
                    action_to_index("RIGHT"),
                    float(reward),
                    next_features,
                    False,
                    next_mask,
                )
        sample_a = [transition.reward for transition in _sample_replay_batch(rng_agent_a)]
        sample_b = [transition.reward for transition in _sample_replay_batch(rng_agent_b)]
        assert sample_a == sample_b

        saved_env = {
            COLLECT_ONLY_ENV: os.environ.get(COLLECT_ONLY_ENV),
            TRAIN_EVERY_ENV: os.environ.get(TRAIN_EVERY_ENV),
            SAVE_EVERY_ENV: os.environ.get(SAVE_EVERY_ENV),
            LEARNING_RATE_ENV: os.environ.get(LEARNING_RATE_ENV),
        }
        try:
            os.environ.pop(LEARNING_RATE_ENV, None)
            os.environ[COLLECT_ONLY_ENV] = "2"
            invalid_collect = _make_training_self(tmpdir, "_invalid_collect")
            try:
                setup_training(invalid_collect)
            except ValueError as exc:
                assert COLLECT_ONLY_ENV in str(exc)
            else:
                raise AssertionError("invalid collect-only value was accepted")

            os.environ[COLLECT_ONLY_ENV] = "0"
            os.environ[TRAIN_EVERY_ENV] = "0"
            invalid_train_every = _make_training_self(tmpdir, "_invalid_train_every")
            try:
                setup_training(invalid_train_every)
            except ValueError as exc:
                assert TRAIN_EVERY_ENV in str(exc)
            else:
                raise AssertionError("non-positive train-every value was accepted")

            os.environ[TRAIN_EVERY_ENV] = "not-an-int"
            invalid_train_every_text = _make_training_self(tmpdir, "_invalid_train_every_text")
            try:
                setup_training(invalid_train_every_text)
            except ValueError as exc:
                assert TRAIN_EVERY_ENV in str(exc)
            else:
                raise AssertionError("non-integer train-every value was accepted")

            os.environ[TRAIN_EVERY_ENV] = "1"
            for invalid_save_value in ("0", "-1", "not-an-int"):
                os.environ[SAVE_EVERY_ENV] = invalid_save_value
                invalid_save = _make_training_self(tmpdir, f"_invalid_save_{invalid_save_value}")
                try:
                    setup_training(invalid_save)
                except ValueError as exc:
                    assert SAVE_EVERY_ENV in str(exc)
                else:
                    raise AssertionError("invalid save-every-rounds value was accepted")

            os.environ[SAVE_EVERY_ENV] = "5"
            save_five_agent = _make_training_self(tmpdir, "_save_five")
            setup_training(save_five_agent)
            assert save_five_agent.save_every_rounds == 5
            os.environ[SAVE_EVERY_ENV] = "10"
            save_ten_agent = _make_training_self(tmpdir, "_save_ten")
            setup_training(save_ten_agent)
            assert save_ten_agent.save_every_rounds == 10
            os.environ.pop(SAVE_EVERY_ENV)
            default_save_agent = _make_training_self(tmpdir, "_save_default")
            setup_training(default_save_agent)
            assert default_save_agent.save_every_rounds == SAVE_EVERY_ROUNDS

            default_learning_rate, was_overridden = _learning_rate_from_env()
            assert default_learning_rate == LEARNING_RATE
            assert was_overridden is False
            for invalid_learning_rate in ("0", "-1", "nan", "inf", "-inf", "not-a-float"):
                os.environ[LEARNING_RATE_ENV] = invalid_learning_rate
                invalid_lr_agent = _make_training_self(
                    tmpdir, f"_invalid_lr_{invalid_learning_rate}"
                )
                try:
                    setup_training(invalid_lr_agent)
                except ValueError as exc:
                    assert LEARNING_RATE_ENV in str(exc)
                else:
                    raise AssertionError("invalid learning-rate value was accepted")

            os.environ[LEARNING_RATE_ENV] = "3e-5"
            configured_lr_agent = _make_training_self(tmpdir, "_configured_lr")
            setup_training(configured_lr_agent)
            assert configured_lr_agent.learning_rate == 3e-5
            assert all(
                group["lr"] == 3e-5
                for group in configured_lr_agent.optimizer.param_groups
            )
            os.environ.pop(LEARNING_RATE_ENV)

            os.environ[COLLECT_ONLY_ENV] = "1"
            os.environ[TRAIN_EVERY_ENV] = "1"
            collect_only_agent = _make_training_self(tmpdir, "_collect_only")
            setup_training(collect_only_agent)
            _fill_replay(collect_only_agent, old_features, next_features, next_mask)
            collect_only_agent.total_transitions = MIN_REPLAY_SIZE
            policy_before = [parameter.detach().clone() for parameter in collect_only_agent.policy_net.parameters()]
            target_before = [parameter.detach().clone() for parameter in collect_only_agent.target_net.parameters()]
            optimizer_before = _optimizer_state_snapshot(collect_only_agent.optimizer)
            loss = optimize_model(collect_only_agent)
            assert loss is None
            assert len(collect_only_agent.replay_buffer) == MIN_REPLAY_SIZE
            assert all(torch.equal(before, after) for before, after in zip(policy_before, collect_only_agent.policy_net.parameters()))
            assert all(torch.equal(before, after) for before, after in zip(target_before, collect_only_agent.target_net.parameters()))
            assert _optimizer_states_equal(optimizer_before, _optimizer_state_snapshot(collect_only_agent.optimizer))

            os.environ[COLLECT_ONLY_ENV] = "0"
            os.environ[TRAIN_EVERY_ENV] = "4"
            train_every_agent = _make_training_self(tmpdir, "_train_every")
            setup_training(train_every_agent)
            _fill_replay(train_every_agent, old_features, next_features, next_mask)
            policy_before = [parameter.detach().clone() for parameter in train_every_agent.policy_net.parameters()]
            for transition_count in (1, 2, 3):
                train_every_agent.total_transitions = transition_count
                assert optimize_model(train_every_agent) is None
                assert all(torch.equal(before, after) for before, after in zip(policy_before, train_every_agent.policy_net.parameters()))
            train_every_agent.total_transitions = 4
            gated_loss = optimize_model(train_every_agent)
            assert gated_loss is not None
            assert np.isfinite(gated_loss)
            assert any(not torch.equal(before, after) for before, after in zip(policy_before, train_every_agent.policy_net.parameters()))
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        agent = _make_training_self(tmpdir, "_train")
        setup_training(agent)
        game_events_occurred(agent, old_state, "RIGHT", next_state, [e.COIN_COLLECTED])
        end_of_round(agent, old_state, "RIGHT", [e.COIN_COLLECTED, e.SURVIVED_ROUND])
        assert len(agent.replay_buffer) == 1
        assert agent.replay_buffer.memory[-1].done is True

        metrics_lines = agent.metrics_path.read_text(encoding="utf-8").strip().splitlines()
        assert metrics_lines[0].split(",") == list(METRICS_COLUMNS)
        assert len(metrics_lines) == 2

        _fill_replay(agent, old_features, next_features, next_mask)
        before_parameters = [parameter.detach().clone() for parameter in agent.policy_net.parameters()]
        loss = optimize_model(agent)
        assert loss is not None
        assert np.isfinite(loss)
        after_parameters = [parameter.detach().clone() for parameter in agent.policy_net.parameters()]
        assert any(not torch.equal(before, after) for before, after in zip(before_parameters, after_parameters))
        _assert_finite_module(agent.policy_net)

        agent.optimization_steps = TARGET_UPDATE_INTERVAL - 1
        before_target = [parameter.detach().clone() for parameter in agent.target_net.parameters()]
        sync_loss = optimize_model(agent)
        assert sync_loss is not None
        after_target = [parameter.detach().clone() for parameter in agent.target_net.parameters()]
        assert any(not torch.equal(before, after) for before, after in zip(before_target, after_target))
        for policy_parameter, target_parameter in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(policy_parameter, target_parameter)

        _save_policy_model(agent, agent.model_path)
        model_payload = torch.load(agent.model_path, map_location="cpu")
        _validate_artifact_metadata(model_payload, agent.model_path)
        loaded_policy = build_dqn(FEATURE_DIM)
        loaded_policy.load_state_dict(model_payload["state_dict"])

        bad_model_path = tmpdir / "bad_model_schema.pt"
        torch.save({**model_payload, "feature_schema": "uttam_dqn_v1_31"}, bad_model_path)
        try:
            _validate_artifact_metadata(torch.load(bad_model_path, map_location="cpu"), bad_model_path)
        except CheckpointError:
            pass
        else:
            raise AssertionError("mismatched V2 model schema was accepted")

        v1_raw_path = tmpdir / "dqn_model.pt"
        torch.save(agent.policy_net.state_dict(), v1_raw_path)
        try:
            _validate_artifact_metadata(torch.load(v1_raw_path, map_location="cpu"), v1_raw_path)
        except CheckpointError:
            pass
        else:
            raise AssertionError("unversioned V1 model artifact was accepted")

        _save_training_checkpoint(agent, agent.checkpoint_path)
        checkpoint_lr_path = tmpdir / "checkpoint_lr_override.pt"
        checkpoint_with_lr = _checkpoint_dict(agent)
        for parameter_group in checkpoint_with_lr["optimizer_state_dict"]["param_groups"]:
            parameter_group["lr"] = 7e-4
        torch.save(checkpoint_with_lr, checkpoint_lr_path)
        saved_learning_rate_env = os.environ.get(LEARNING_RATE_ENV)
        try:
            os.environ[LEARNING_RATE_ENV] = "1e-5"
            lr_restored = _make_training_self(tmpdir, "_lr_restored")
            lr_restored.checkpoint_path = checkpoint_lr_path
            lr_restored.metrics_path = agent.metrics_path
            setup_training(lr_restored)
            assert lr_restored.learning_rate == 1e-5
            assert all(
                group["lr"] == 1e-5
                for group in lr_restored.optimizer.param_groups
            )
        finally:
            if saved_learning_rate_env is None:
                os.environ.pop(LEARNING_RATE_ENV, None)
            else:
                os.environ[LEARNING_RATE_ENV] = saved_learning_rate_env

        bad_checkpoint_path = tmpdir / "bad_checkpoint_schema.pt"
        bad_checkpoint = _checkpoint_dict(agent)
        bad_checkpoint["feature_schema"] = "uttam_dqn_v1_31"
        torch.save(bad_checkpoint, bad_checkpoint_path)
        schema_agent = _make_training_self(tmpdir, "_schema")
        schema_agent.checkpoint_path = bad_checkpoint_path
        try:
            setup_training(schema_agent)
        except CheckpointError:
            pass
        else:
            raise AssertionError("mismatched V2 training checkpoint schema was accepted")

        nonfinite_checkpoint_path = tmpdir / "nonfinite_checkpoint.pt"
        nonfinite_checkpoint = _checkpoint_dict(agent)
        nonfinite_policy = {
            name: value.detach().clone()
            for name, value in nonfinite_checkpoint["policy_state_dict"].items()
        }
        first_parameter = next(iter(nonfinite_policy))
        nonfinite_policy[first_parameter].reshape(-1)[0] = float("inf")
        nonfinite_checkpoint["policy_state_dict"] = nonfinite_policy
        torch.save(nonfinite_checkpoint, nonfinite_checkpoint_path)
        nonfinite_agent = _make_training_self(tmpdir, "_nonfinite_checkpoint")
        nonfinite_agent.checkpoint_path = nonfinite_checkpoint_path
        try:
            setup_training(nonfinite_agent)
        except CheckpointError as exc:
            assert "NaN or infinity" in str(exc)
        else:
            raise AssertionError("non-finite checkpoint parameters were accepted")

        restored = _make_training_self(tmpdir, "_restored")
        restored.checkpoint_path = agent.checkpoint_path
        restored.metrics_path = agent.metrics_path
        setup_training(restored)
        assert restored.total_transitions == agent.total_transitions
        assert restored.optimization_steps == agent.optimization_steps
        assert restored.completed_rounds == agent.completed_rounds
        assert restored.epsilon == agent.epsilon
        assert len(restored.replay_buffer) == len(agent.replay_buffer)
        for original, loaded in zip(agent.replay_buffer.memory, restored.replay_buffer.memory):
            assert np.array_equal(original.state, loaded.state)
            assert original.action == loaded.action
            assert np.isclose(original.reward, loaded.reward)
            assert original.done == loaded.done
            if original.next_state is None:
                assert loaded.next_state is None
            else:
                assert np.array_equal(original.next_state, loaded.next_state)
            assert np.array_equal(original.next_valid_mask, loaded.next_valid_mask)
        assert restored.replay_rng.getstate() == agent.replay_rng.getstate()
        assert restored.rng.getstate() == agent.rng.getstate()

        legacy_checkpoint_path = tmpdir / "legacy_checkpoint.pt"
        legacy_checkpoint = _checkpoint_dict(agent)
        for key in ("replay_buffer", "replay_rng_state", "action_rng_state"):
            legacy_checkpoint.pop(key)
        torch.save(legacy_checkpoint, legacy_checkpoint_path)
        legacy_agent = _make_training_self(tmpdir, "_legacy")
        legacy_agent.checkpoint_path = legacy_checkpoint_path
        legacy_agent.metrics_path = agent.metrics_path
        setup_training(legacy_agent)
        assert legacy_agent.completed_rounds == agent.completed_rounds
        assert len(legacy_agent.replay_buffer) == 0

        def write_metrics_history(path, rounds):
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=METRICS_COLUMNS)
                writer.writeheader()
                for round_number in rounds:
                    row = {column: 0 for column in METRICS_COLUMNS}
                    row["round"] = round_number
                    writer.writerow(row)

        valid_metrics_path = tmpdir / "valid_metrics.csv"
        write_metrics_history(valid_metrics_path, [1, 2, 3])
        _validate_metrics_history(valid_metrics_path, 3)

        duplicate_metrics_path = tmpdir / "duplicate_metrics.csv"
        write_metrics_history(duplicate_metrics_path, [1, 2, 2])
        try:
            _validate_metrics_history(duplicate_metrics_path, 2)
        except CheckpointError as exc:
            assert "strictly increasing and unique" in str(exc)
        else:
            raise AssertionError("duplicate metrics rounds were accepted")

        decreasing_metrics_path = tmpdir / "decreasing_metrics.csv"
        write_metrics_history(decreasing_metrics_path, [1, 3, 2])
        try:
            _validate_metrics_history(decreasing_metrics_path, 2)
        except CheckpointError as exc:
            assert "strictly increasing and unique" in str(exc)
        else:
            raise AssertionError("decreasing metrics rounds were accepted")

        mismatched_metrics_path = tmpdir / "mismatched_metrics.csv"
        write_metrics_history(mismatched_metrics_path, [1, 2])
        mismatch_agent = _make_training_self(tmpdir, "_metrics_mismatch")
        mismatch_agent.checkpoint_path = agent.checkpoint_path
        mismatch_agent.metrics_path = mismatched_metrics_path
        try:
            setup_training(mismatch_agent)
        except CheckpointError as exc:
            assert "does not match metrics final round" in str(exc)
        else:
            raise AssertionError("checkpoint/metrics round mismatch was accepted")

        malformed_checkpoint_path = tmpdir / "malformed_replay_checkpoint.pt"
        malformed_checkpoint = _checkpoint_dict(agent)
        malformed_checkpoint["replay_buffer"] = dict(malformed_checkpoint["replay_buffer"])
        malformed_checkpoint["replay_buffer"]["states"] = torch.zeros((1, FEATURE_DIM + 1), dtype=torch.float32)
        torch.save(malformed_checkpoint, malformed_checkpoint_path)
        malformed_agent = _make_training_self(tmpdir, "_malformed_replay")
        malformed_agent.checkpoint_path = malformed_checkpoint_path
        try:
            setup_training(malformed_agent)
        except CheckpointError as exc:
            assert "replay states" in str(exc)
        else:
            raise AssertionError("malformed replay checkpoint was accepted")

        bad_replay_version_path = tmpdir / "bad_replay_version_checkpoint.pt"
        bad_replay_version = _checkpoint_dict(agent)
        bad_replay_version["replay_buffer"] = dict(bad_replay_version["replay_buffer"])
        bad_replay_version["replay_buffer"]["format_version"] = REPLAY_FORMAT_VERSION + 1
        torch.save(bad_replay_version, bad_replay_version_path)
        bad_replay_agent = _make_training_self(tmpdir, "_bad_replay_version")
        bad_replay_agent.checkpoint_path = bad_replay_version_path
        try:
            setup_training(bad_replay_agent)
        except CheckpointError as exc:
            assert "replay format version" in str(exc)
        else:
            raise AssertionError("incompatible replay format was accepted")

        for value in (agent.epsilon, loss, sync_loss):
            assert np.isfinite(value)

    print("V2 artifact schema regression passed")
    print("unsafe bomb regression passed")
    print("useless bomb regression passed")
    print("survivor regression passed")
    print("death regression passed")
    print("train.py self-test passed")

class TrainingRegressionTests(unittest.TestCase):
    def test_existing_regressions(self):
        _run_self_test()


if __name__ == "__main__":
    unittest.main()
