"""Sampling and scan-position conversion helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import _normalize_dx


def position_to_pixel_shift(
    position_xy: tuple[float, float] | NDArray[np.floating],
    dx: float | tuple[float, float],
) -> tuple[int, int]:
    """Convert an `(x, y)` position in meters to integer `(dy, dx)` pixels.

    This is intentionally integer-only for the first simulation stage. Subpixel
    interpolation should be added before serious reconstruction studies.
    """

    dy_m, dx_m = _normalize_dx(dx)
    x_m = float(position_xy[0])
    y_m = float(position_xy[1])
    shift_x = int(np.rint(x_m / dx_m))
    shift_y = int(np.rint(y_m / dy_m))
    return shift_y, shift_x
