import logging
import random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn

from .features import ARTIFACT_VERSION, FEATURE_DIM, FEATURE_SCHEMA, state_to_features, valid_action_mask
from .model import ACTIONS, N_ACTIONS, build_dqn, index_to_action


MODEL_FILENAME = "dqn_model_v2.pt"
CPU_DEVICE = torch.device("cpu")


def _default_model_path() -> Path:
    """Return dqn_model_v2.pt beside this callbacks.py file."""
    return Path(__file__).resolve().with_name(MODEL_FILENAME)


def setup(self):
    """Initialize CPU-only DQN inference state for training or evaluation.

    The saved V2 policy artifact is expected at dqn_model_v2.pt beside this
    file. The path is derived from __file__ so loading is independent of the
    process working directory. V1 artifacts are never used as a fallback.
    """
    self.device = CPU_DEVICE
    self.model_path = _default_model_path()
    self.policy_net = build_dqn(FEATURE_DIM).to(self.device)
    self.rng = random.Random()
    self.epsilon = 1.0 if self.train else 0.0

    if self.model_path.exists():
        _load_policy_state(self.policy_net, self.model_path, self.logger)
    else:
        self.logger.warning(
            "No saved DQN model found at %s; using a newly initialized policy network.",
            self.model_path,
        )

    self.policy_net.eval()


def act(self, game_state: dict) -> str:
    """Return one legal action selected by epsilon-greedy masked DQN inference."""
    start_time = perf_counter() if self.train else None
    features = state_to_features(game_state)
    if features is None:
        raise ValueError("act() received terminal game_state=None")

    legal_mask = valid_action_mask(game_state)
    _validate_legal_mask(legal_mask)

    if self.train and self.epsilon > 0.0 and self.rng.random() < self.epsilon:
        self.metrics.record_decision_time(perf_counter() - start_time)
        action_index = _sample_legal_action(legal_mask, self.rng)
        action = index_to_action(action_index)
        self.logger.debug("Exploration selected legal action %s", action)
        return action

    action_index, q_values = _select_greedy_action(
        self.policy_net,
        features,
        legal_mask,
        self.device,
    )
    action = index_to_action(action_index)
    self.logger.debug(
        "Greedy selected legal action %s with q=%.6f and mask=%s",
        action,
        float(q_values[action_index]),
        legal_mask.astype(int).tolist(),
    )
    if self.train:
        self.metrics.record_decision_time(perf_counter() - start_time)
    return action


def _load_policy_state(policy_net: nn.Module, model_path: Path, logger: logging.Logger) -> None:
    """Load a versioned V2 policy artifact, failing clearly on incompatibility."""
    try:
        payload = torch.load(model_path, map_location="cpu")
        state_dict = _state_dict_from_model_artifact(payload, model_path)
        policy_net.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(f"Failed to load V2 DQN model artifact from {model_path}: {exc}") from exc
    logger.info("Loaded V2 DQN policy artifact from %s", model_path)


def _state_dict_from_model_artifact(payload, model_path: Path) -> dict:
    """Validate V2 model metadata and return the contained policy state_dict."""
    if not isinstance(payload, dict):
        raise TypeError("saved model artifact is not a dictionary")
    _validate_artifact_metadata(payload, model_path)
    if "state_dict" not in payload:
        raise KeyError("missing state_dict")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict is not a dictionary")
    _validate_finite_state_dict(state_dict)
    return state_dict


def _validate_finite_state_dict(state_dict: dict) -> None:
    """Reject model parameters containing NaN or infinity before loading."""
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"state_dict entry {name!r} contains NaN or infinity")


def _validate_artifact_metadata(payload: dict, model_path: Path) -> None:
    """Validate feature schema metadata shared by V2 model and checkpoint files."""
    expected = {
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "artifact_version": ARTIFACT_VERSION,
    }
    for key, expected_value in expected.items():
        actual_value = payload.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"{model_path} has incompatible {key}={actual_value!r}; "
                f"expected {expected_value!r}"
            )


def _validate_legal_mask(legal_mask: np.ndarray) -> np.ndarray:
    """Validate and return a Boolean legal-action mask with the fixed action size."""
    mask = np.asarray(legal_mask, dtype=np.bool_)
    if mask.shape != (N_ACTIONS,):
        raise ValueError(f"legal action mask must have shape ({N_ACTIONS},)")
    if not mask.any():
        raise ValueError("legal action mask must contain at least one valid action")
    return mask


def _sample_legal_action(legal_mask: np.ndarray, rng: random.Random) -> int:
    """Sample uniformly from currently legal action indices."""
    mask = _validate_legal_mask(legal_mask)
    legal_indices = np.flatnonzero(mask).tolist()
    return int(rng.choice(legal_indices))


def _select_greedy_action(
    policy_net: nn.Module,
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device = CPU_DEVICE,
) -> tuple[int, np.ndarray]:
    """Select the highest-Q legal action after masking impossible actions."""
    mask = _validate_legal_mask(legal_mask)
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != (FEATURE_DIM,):
        raise ValueError(f"features must have shape ({FEATURE_DIM},)")
    if not np.isfinite(feature_array).all():
        raise ValueError("features must be finite")

    state_tensor = torch.as_tensor(feature_array, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)

    with torch.no_grad():
        q_tensor = policy_net(state_tensor).squeeze(0)
        if q_tensor.shape != (N_ACTIONS,):
            raise RuntimeError(f"policy network must return shape ({N_ACTIONS},)")
        masked_q_tensor = q_tensor.masked_fill(~mask_tensor, torch.finfo(q_tensor.dtype).min)
        action_index = int(torch.argmax(masked_q_tensor).item())

    if not mask[action_index]:
        raise RuntimeError("masked greedy selection produced an invalid action")
    return action_index, q_tensor.detach().cpu().numpy()
