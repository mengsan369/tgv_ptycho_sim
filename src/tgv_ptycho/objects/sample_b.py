"""Synthetic scanning sample B generators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_random_phase_object(
    shape: tuple[int, int],
    phase_range: float = np.pi,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Create a random phase-only transmission object."""

    rng = np.random.default_rng(seed)
    phase = rng.uniform(-phase_range, phase_range, size=shape)
    return np.exp(1j * phase).astype(np.complex128)


def make_random_amp_phase_object(
    shape: tuple[int, int],
    amp_range: tuple[float, float] = (0.5, 1.0),
    phase_range: float = np.pi,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Create a random amplitude-phase transmission object."""

    amp_min, amp_max = amp_range
    if amp_min < 0 or amp_max < amp_min:
        msg = "amp_range must satisfy 0 <= min <= max."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(amp_min, amp_max, size=shape)
    phase = rng.uniform(-phase_range, phase_range, size=shape)
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)
