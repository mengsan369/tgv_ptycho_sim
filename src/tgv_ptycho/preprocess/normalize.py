"""Intensity normalization interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def normalize_stack(
    I_stack: NDArray[np.floating],
    method: str = "median",
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Normalize an intensity stack frame-by-frame.

    TODO: Support monitor counts, exposure time, background masks, and robust
    percentile normalization for real experimental data.
    """

    raise NotImplementedError("Intensity normalization is a reserved interface.")


def normalize_by_exposure(
    I_stack: NDArray[np.floating],
    exposure_s: float | NDArray[np.floating],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Normalize intensities by scalar or per-frame exposure time in seconds.

    TODO: Validate exposure metadata and camera-count units.
    """

    raise NotImplementedError("Exposure normalization is not implemented yet.")
