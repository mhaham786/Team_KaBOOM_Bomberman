"""Shared synthetic-state helpers for DQN tests."""

import numpy as np


def bordered_field(size: int) -> np.ndarray:
    """Return an empty square field enclosed by stone walls."""
    field = np.zeros((size, size), dtype=np.int64)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field
