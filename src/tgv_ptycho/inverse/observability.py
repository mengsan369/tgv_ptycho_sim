"""Observability helpers for comparing probe signatures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compare_probe_sensitivity(
    P1: NDArray[np.complexfloating],
    P2: NDArray[np.complexfloating],
) -> dict[str, float]:
    """Compare two probe fields with relative complex and amplitude metrics."""

    probe_1 = np.asarray(P1, dtype=np.complex128)
    probe_2 = np.asarray(P2, dtype=np.complex128)
    if probe_1.shape != probe_2.shape:
        msg = "P1 and P2 must have the same shape."
        raise ValueError(msg)
    eps = np.finfo(float).eps
    relative_l2 = np.linalg.norm(probe_1 - probe_2) / (np.linalg.norm(probe_1) + eps)
    amplitude_relative_l2 = np.linalg.norm(np.abs(probe_1) - np.abs(probe_2)) / (
        np.linalg.norm(np.abs(probe_1)) + eps
    )
    return {
        "relative_l2": float(relative_l2),
        "amplitude_relative_l2": float(amplitude_relative_l2),
    }
