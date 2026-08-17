"""Scan-position generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _normalize_step(step: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(step, tuple):
        if len(step) != 2:
            msg = "step tuple must be (step_x, step_y) in meters."
            raise ValueError(msg)
        return float(step[0]), float(step[1])
    value = float(step)
    return value, value


def make_grid_scan(
    num_x: int,
    num_y: int,
    step: float | tuple[float, float],
    center: bool = True,
) -> NDArray[np.float64]:
    """Create a Cartesian scan grid in meters.

    Returns an array of shape `(num_x * num_y, 2)` with columns `(x, y)`.
    """

    if num_x <= 0 or num_y <= 0:
        msg = "num_x and num_y must be positive."
        raise ValueError(msg)
    step_x, step_y = _normalize_step(step)
    xs = np.arange(num_x, dtype=np.float64) * step_x
    ys = np.arange(num_y, dtype=np.float64) * step_y
    if center:
        xs -= xs.mean()
        ys -= ys.mean()
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)


def add_position_jitter(
    positions: NDArray[np.floating],
    sigma: float,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add Gaussian stage-position jitter in meters."""

    if sigma < 0:
        msg = "sigma must be non-negative."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    return np.asarray(positions, dtype=np.float64) + rng.normal(
        loc=0.0, scale=sigma, size=np.asarray(positions).shape
    )


def add_integer_pixel_jitter(
    positions: NDArray[np.floating],
    dx: float | tuple[float, float],
    max_jitter_px: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add reproducible integer-pixel jitter to ``(x, y)`` scan positions.

    ``dx`` follows the field convention: a tuple is ``(dy, dx)`` in meters.
    Jitter is sampled independently and uniformly from the inclusive interval
    ``[-max_jitter_px, max_jitter_px]`` on each axis.
    """

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        msg = "positions must have shape (num_positions, 2)."
        raise ValueError(msg)
    if max_jitter_px < 0:
        msg = "max_jitter_px must be non-negative."
        raise ValueError(msg)
    if isinstance(dx, tuple):
        if len(dx) != 2:
            msg = "dx tuple must be (dy, dx) in meters."
            raise ValueError(msg)
        dy_m, dx_m = float(dx[0]), float(dx[1])
    else:
        dy_m = dx_m = float(dx)
    if dy_m <= 0 or dx_m <= 0:
        msg = "dx values must be positive."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    offsets_px = rng.integers(
        -max_jitter_px,
        max_jitter_px + 1,
        size=values.shape,
    )
    offsets_m = np.empty_like(values)
    offsets_m[:, 0] = offsets_px[:, 0] * dx_m
    offsets_m[:, 1] = offsets_px[:, 1] * dy_m
    return values + offsets_m
