"""Simple detector helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def intensity_from_field(U: NDArray[np.complexfloating]) -> NDArray[np.float64]:
    """Return detector intensity `|U|^2`."""

    return np.abs(U) ** 2


def normalize_to_bit_depth(
    intensity: NDArray[np.floating],
    bit_depth: int = 12,
) -> NDArray[np.uint16]:
    """Normalize intensity to an unsigned integer detector range."""

    max_value = 2**bit_depth - 1
    arr = np.asarray(intensity, dtype=np.float64)
    peak = arr.max()
    if peak <= 0:
        return np.zeros_like(arr, dtype=np.uint16)
    return np.rint(np.clip(arr / peak, 0.0, 1.0) * max_value).astype(np.uint16)
