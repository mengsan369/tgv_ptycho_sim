"""Region-of-interest preprocessing interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def crop_roi(
    image: NDArray[np.floating],
    roi: tuple[int, int, int, int],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Crop a 2D image by `(y0, y1, x0, x1)` pixel bounds.

    TODO: Decide final ROI convention and preserve coordinate metadata.
    """

    raise NotImplementedError("ROI cropping is a reserved interface.")


def crop_stack_roi(
    I_stack: NDArray[np.floating],
    roi: tuple[int, int, int, int],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Crop every frame in an intensity stack by `(y0, y1, x0, x1)`.

    TODO: Update detector center and downstream geometry metadata.
    """

    raise NotImplementedError("Stack ROI cropping is not implemented yet.")


def suggest_center_roi(
    image_shape: tuple[int, int],
    roi_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Suggest a centered ROI for a detector frame.

    TODO: Add automatic diffraction-pattern centering.
    """

    raise NotImplementedError("Automatic ROI suggestion is not implemented yet.")
