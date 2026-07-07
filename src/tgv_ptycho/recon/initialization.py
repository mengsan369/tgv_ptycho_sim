"""Reconstruction initialization helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def initialize_unit_object(shape: tuple[int, int]) -> NDArray[np.complex128]:
    """Return a unit complex object estimate."""

    return np.ones(shape, dtype=np.complex128)


def initialize_probe_from_mean_intensity(
    I_stack: NDArray[np.floating],
) -> NDArray[np.complex128]:
    """Initialize a probe amplitude from mean detector intensity."""

    intensities = np.asarray(I_stack, dtype=np.float64)
    return np.sqrt(np.maximum(intensities.mean(axis=0), 0.0)).astype(np.complex128)
