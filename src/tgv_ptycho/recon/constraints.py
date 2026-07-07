"""Simple reconstruction constraints."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def enforce_measured_amplitude(
    predicted_field: NDArray[np.complexfloating],
    measured_intensity: NDArray[np.floating],
) -> NDArray[np.complex128]:
    """Replace field amplitude by the measured amplitude."""

    amplitude = np.sqrt(np.maximum(np.asarray(measured_intensity), 0.0))
    return (amplitude * np.exp(1j * np.angle(predicted_field))).astype(np.complex128)
