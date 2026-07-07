"""Error metrics for complex fields and phase maps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def complex_relative_error(
    U_rec: NDArray[np.complexfloating],
    U_true: NDArray[np.complexfloating],
) -> float:
    """Return `||U_rec - U_true||_2 / ||U_true||_2`."""

    rec = np.asarray(U_rec, dtype=np.complex128)
    true = np.asarray(U_true, dtype=np.complex128)
    return float(np.linalg.norm(rec - true) / (np.linalg.norm(true) + np.finfo(float).eps))


def amplitude_rmse(
    U_rec: NDArray[np.complexfloating],
    U_true: NDArray[np.complexfloating],
) -> float:
    """Return RMSE between amplitudes."""

    return float(np.sqrt(np.mean((np.abs(U_rec) - np.abs(U_true)) ** 2)))


def phase_rmse(
    phi_rec: NDArray[np.floating],
    phi_true: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return wrapped phase RMSE in radians."""

    wrapped = np.angle(np.exp(1j * (np.asarray(phi_rec) - np.asarray(phi_true))))
    if mask is not None:
        wrapped = wrapped[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(wrapped**2)))
