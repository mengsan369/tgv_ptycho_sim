"""Simplified 2D effective TGV-like objects."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.fields import coordinate_grid


def make_thin_phase_disk(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    diameter: float,
    phase_shift: float,
) -> NDArray[np.complex128]:
    """Create a centered thin phase disk transmission function.

    The returned transmission is one outside the disk and
    `exp(i * phase_shift)` inside the disk. This is a 2D test object only.
    """

    if diameter <= 0:
        msg = "diameter must be positive."
        raise ValueError(msg)
    x_grid, y_grid = coordinate_grid(shape, dx)
    radius = diameter / 2.0
    inside = (x_grid**2 + y_grid**2) <= radius**2
    transmission = np.ones(shape, dtype=np.complex128)
    transmission[inside] = np.exp(1j * phase_shift)
    return transmission


def make_tgv_effective_phase_2d(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    diameter_top: float,
    diameter_bottom: float | None = None,
    phase_shift: float = 1.0,
) -> NDArray[np.complex128]:
    """Create a crude 2D effective phase proxy for a TGV aperture.

    This helper is only for early probe-recovery and sensitivity tests. It is
    not equivalent to a real 3D waist model. If `diameter_bottom` is provided,
    the effective diameter is the arithmetic mean of top and bottom diameters.
    """

    if diameter_bottom is None:
        effective_diameter = diameter_top
    else:
        effective_diameter = 0.5 * (diameter_top + diameter_bottom)
    return make_thin_phase_disk(shape, dx, effective_diameter, phase_shift)
