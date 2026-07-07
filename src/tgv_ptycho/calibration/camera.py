"""Camera calibration interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_camera_calibration(path: str | Path) -> dict[str, Any]:
    """Load camera calibration metadata from a file.

    TODO: Define accepted YAML/HDF5 calibration formats, including pixel size,
    gain, offset, dark frame, flat field, saturation, and bad-pixel mask paths.
    """

    raise NotImplementedError("Camera calibration loading is not implemented yet.")


def estimate_camera_response(
    flat_stacks: list[NDArray[np.floating]],
    exposure_s: list[float],
) -> dict[str, Any]:
    """Estimate camera response from flat-field exposure series.

    TODO: Estimate linearity, gain, offset, read noise, and saturation range.
    """

    raise NotImplementedError("Camera response estimation is not implemented yet.")
