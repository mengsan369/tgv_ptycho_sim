"""Reconstruction initialization helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate


def initialize_unit_object(shape: tuple[int, int]) -> NDArray[np.complex128]:
    """Return a unit complex object estimate."""

    return np.ones(shape, dtype=np.complex128)


def initialize_probe_from_mean_intensity(
    I_stack: NDArray[np.floating],
) -> NDArray[np.complex128]:
    """Initialize a probe amplitude from mean detector intensity."""

    intensities = np.asarray(I_stack, dtype=np.float64)
    return np.sqrt(np.maximum(intensities.mean(axis=0), 0.0)).astype(np.complex128)


def initialize_probe_by_detector_backpropagation(
    I_stack: NDArray[np.floating],
    dx: float | tuple[float, float],
    wavelength: float,
    z_BC: float,
) -> NDArray[np.complex128]:
    """Backpropagate the mean measured detector amplitude with zero phase.

    The estimate is scaled so its L2 norm equals the square root of the mean
    measured frame energy. This is a truth-free initialization and is exact as
    an energy constraint when propagation is unitary and sample B is phase-only.
    """

    intensities = np.asarray(I_stack, dtype=np.float64)
    if intensities.ndim != 3 or intensities.shape[0] == 0:
        msg = "I_stack must have shape (num_positions, ny, nx)."
        raise ValueError(msg)
    if not np.all(np.isfinite(intensities)) or np.any(intensities < 0):
        msg = "I_stack must contain finite, non-negative intensities."
        raise ValueError(msg)

    mean_amplitude = np.sqrt(np.maximum(np.mean(intensities, axis=0), 0.0))
    probe = angular_spectrum_propagate(
        mean_amplitude.astype(np.complex128), dx, wavelength, -z_BC
    )
    target_norm = float(np.sqrt(np.mean(np.sum(intensities, axis=(1, 2)))))
    current_norm = float(np.sqrt(np.sum(np.abs(probe) ** 2)))
    if current_norm <= np.finfo(float).eps:
        msg = "Cannot initialize a probe from zero-energy measurements."
        raise ValueError(msg)
    return (probe * (target_norm / current_norm)).astype(np.complex128)
