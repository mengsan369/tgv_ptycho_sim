"""Synthetic scanning sample B generators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _make_random_feature_map(
    shape: tuple[int, int],
    feature_size_px: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Create a uniform random map with square, piecewise-constant features."""

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        msg = "shape must be a positive (ny, nx) tuple."
        raise ValueError(msg)
    if feature_size_px <= 0:
        msg = "feature_size_px must be a positive integer."
        raise ValueError(msg)

    ny, nx = int(shape[0]), int(shape[1])
    coarse_shape = (
        int(np.ceil(ny / feature_size_px)),
        int(np.ceil(nx / feature_size_px)),
    )
    coarse = rng.random(coarse_shape)
    expanded = np.repeat(
        np.repeat(coarse, feature_size_px, axis=0), feature_size_px, axis=1
    )
    return expanded[:ny, :nx].astype(np.float64, copy=False)


def make_random_phase_object(
    shape: tuple[int, int],
    phase_range: float = np.pi,
    seed: int | None = None,
    feature_size_px: int = 1,
) -> NDArray[np.complex128]:
    """Create a random phase-only transmission object.

    `feature_size_px=1` produces independent pixels. Larger values produce
    square random features, which are useful for controlled synthetic masks.
    """

    if phase_range < 0:
        msg = "phase_range must be non-negative."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    phase = (2.0 * _make_random_feature_map(shape, feature_size_px, rng) - 1.0) * (
        phase_range
    )
    return np.exp(1j * phase).astype(np.complex128)


def make_random_amp_phase_object(
    shape: tuple[int, int],
    amp_range: tuple[float, float] = (0.5, 1.0),
    phase_range: float = np.pi,
    seed: int | None = None,
    feature_size_px: int = 1,
) -> NDArray[np.complex128]:
    """Create a random amplitude-phase transmission object.

    Amplitude and phase share the requested feature size but are generated
    from independent random maps.
    """

    amp_min, amp_max = amp_range
    if amp_min < 0 or amp_max < amp_min:
        msg = "amp_range must satisfy 0 <= min <= max."
        raise ValueError(msg)
    if phase_range < 0:
        msg = "phase_range must be non-negative."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    amplitude = amp_min + (amp_max - amp_min) * _make_random_feature_map(
        shape, feature_size_px, rng
    )
    phase = (
        2.0 * _make_random_feature_map(shape, feature_size_px, rng) - 1.0
    ) * phase_range
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)
