"""Mask helpers for future structured sample B models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_binary_checkerboard(
    shape: tuple[int, int],
    period_pixels: int,
    low: float = 0.0,
    high: float = 1.0,
) -> NDArray[np.float64]:
    """Create a simple checkerboard mask for debugging scan shifts."""

    if period_pixels <= 0:
        msg = "period_pixels must be positive."
        raise ValueError(msg)
    yy, xx = np.indices(shape)
    cells = (xx // period_pixels + yy // period_pixels) % 2
    return np.where(cells == 0, low, high).astype(np.float64)
