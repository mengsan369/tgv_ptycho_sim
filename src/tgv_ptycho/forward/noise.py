"""Noise models for simulated detector intensities."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def apply_noise(
    intensity: NDArray[np.floating],
    noise_config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Apply simple Poisson and Gaussian noise to an intensity array."""

    clean = np.asarray(intensity, dtype=np.float64)
    if noise_config is None:
        return clean

    rng = np.random.default_rng(seed)
    noisy = clean.copy()

    photon_scale = noise_config.get("photon_scale")
    if photon_scale is not None:
        scaled = np.clip(noisy * float(photon_scale), 0.0, None)
        noisy = rng.poisson(scaled).astype(np.float64) / float(photon_scale)

    gaussian_sigma = float(noise_config.get("gaussian_sigma", 0.0))
    if gaussian_sigma > 0:
        noisy += rng.normal(0.0, gaussian_sigma, size=noisy.shape)

    return np.clip(noisy, 0.0, None)
