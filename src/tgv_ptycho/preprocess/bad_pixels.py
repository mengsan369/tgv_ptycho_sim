"""Bad-pixel detection and correction interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def detect_bad_pixels(
    calibration_stack: NDArray[np.floating],
    method: str = "robust_zscore",
    threshold: float = 8.0,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Detect bad pixels from calibration frames.

    TODO: Support hot pixels, dead pixels, saturated pixels, and temporal noise
    outliers with camera-specific thresholds.
    """

    raise NotImplementedError("Bad-pixel detection is a reserved interface.")


def correct_bad_pixels(
    I_stack: NDArray[np.floating],
    bad_pixel_mask: NDArray[np.bool_],
    method: str = "median_neighbors",
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Correct bad pixels in an intensity stack.

    TODO: Implement neighborhood interpolation and track all corrected pixels
    in the preprocessing metadata.
    """

    raise NotImplementedError("Bad-pixel correction is not implemented yet.")
