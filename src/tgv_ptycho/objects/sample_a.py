"""Synthetic two-dimensional thin-phase sample A generators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from tgv_ptycho.optics.angular_spectrum import _normalize_dx
from tgv_ptycho.optics.fields import coordinate_grid


def make_smooth_random_thin_phase(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    radius: float,
    phase_rms: float,
    correlation_length: float,
    seed: int | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.bool_]]:
    """Create a smooth random pure-phase transmission within a circular support.

    Arrays use ``(ny, nx)`` order. A scalar ``dx`` means square sampling; a
    tuple is ``(dy, dx)`` in meters. Outside ``radius`` the transmission is
    exactly one, providing a known blank reference region for removing the
    blind-reconstruction phase-plane ambiguity.
    """

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        msg = "shape must be a positive (ny, nx) tuple."
        raise ValueError(msg)
    if radius <= 0:
        msg = "radius must be positive."
        raise ValueError(msg)
    if phase_rms < 0:
        msg = "phase_rms must be non-negative."
        raise ValueError(msg)
    if correlation_length <= 0:
        msg = "correlation_length must be positive."
        raise ValueError(msg)

    x_grid, y_grid = coordinate_grid(shape, dx)
    support = x_grid**2 + y_grid**2 <= radius**2
    if not np.any(support) or np.all(support):
        msg = "radius must leave both active and blank-reference pixels."
        raise ValueError(msg)

    dy_m, dx_m = _normalize_dx(dx)
    rng = np.random.default_rng(seed)
    white = rng.normal(size=shape)
    smooth = gaussian_filter(
        white,
        sigma=(correlation_length / dy_m, correlation_length / dx_m),
        mode="wrap",
    )
    smooth -= float(np.mean(smooth[support]))
    current_rms = float(np.sqrt(np.mean(smooth[support] ** 2)))
    if current_rms <= np.finfo(float).eps:
        msg = "Generated phase map has zero variance; adjust the configuration."
        raise ValueError(msg)

    phase = np.zeros(shape, dtype=np.float64)
    phase[support] = smooth[support] * (phase_rms / current_rms)
    transmission = np.exp(1j * phase).astype(np.complex128)
    return transmission, phase, support
