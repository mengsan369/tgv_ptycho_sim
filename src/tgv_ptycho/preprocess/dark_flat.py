"""Dark-frame and flat-field preprocessing interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def apply_dark_flat_correction(
    I_stack: NDArray[np.floating],
    dark_frame: NDArray[np.floating] | None = None,
    flat_field: NDArray[np.floating] | None = None,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Apply dark subtraction and flat-field correction to raw intensities.

    TODO: Implement validated correction rules for real CMOS data, including
    units, saturation handling, and per-frame dark models.
    """

    raise NotImplementedError("Dark/flat preprocessing is a reserved interface.")


def estimate_dark_frame(
    dark_stack: NDArray[np.floating],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Estimate a master dark frame from repeated dark exposures.

    TODO: Add robust statistics, exposure grouping, and hot-pixel tracking.
    """

    raise NotImplementedError("Dark-frame estimation is not implemented yet.")


def estimate_flat_field(
    flat_stack: NDArray[np.floating],
    dark_frame: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Estimate a normalized flat field from calibration frames.

    TODO: Add dark correction, illumination smoothing, and bad-pixel exclusion.
    """

    raise NotImplementedError("Flat-field estimation is not implemented yet.")
