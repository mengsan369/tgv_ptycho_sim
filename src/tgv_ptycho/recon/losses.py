"""Loss functions for reconstruction diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def relative_amplitude_loss(
    predicted_field: NDArray[np.complexfloating],
    measured_intensity: NDArray[np.floating],
) -> float:
    """Return relative detector amplitude mismatch."""

    pred_amp = np.abs(predicted_field)
    meas_amp = np.sqrt(np.maximum(np.asarray(measured_intensity), 0.0))
    return float(np.linalg.norm(pred_amp - meas_amp) / (np.linalg.norm(meas_amp) + 1e-15))
