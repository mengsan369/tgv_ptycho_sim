"""Field and aperture generators."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import _normalize_dx

ComplexArray: TypeAlias = NDArray[np.complexfloating]
FloatArray: TypeAlias = NDArray[np.floating]


def coordinate_grid(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """Return centered `(x, y)` coordinate grids in meters."""

    if len(shape) != 2:
        msg = "shape must be (ny, nx)."
        raise ValueError(msg)
    ny, nx = int(shape[0]), int(shape[1])
    dy, dx_x = _normalize_dx(dx)
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx_x
    x_grid, y_grid = np.meshgrid(x, y)
    return x_grid, y_grid


def make_plane_wave(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    wavelength: float,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    amplitude: float = 1.0,
) -> ComplexArray:
    """Create a tilted plane wave sampled on a 2D grid.

    Angles are in radians, sampling is in meters, and wavelength is the vacuum
    wavelength in meters.
    """

    x_grid, y_grid = coordinate_grid(shape, dx)
    k0 = 2.0 * np.pi / wavelength
    kx = k0 * np.sin(theta_x)
    ky = k0 * np.sin(theta_y)
    return (amplitude * np.exp(1j * (kx * x_grid + ky * y_grid))).astype(
        np.complex128
    )


def make_gaussian_field(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    waist: float,
    amplitude: float = 1.0,
) -> ComplexArray:
    """Create a real Gaussian field with 1/e amplitude radius `waist` in meters."""

    if waist <= 0:
        msg = "waist must be positive."
        raise ValueError(msg)
    x_grid, y_grid = coordinate_grid(shape, dx)
    r_squared = x_grid**2 + y_grid**2
    return (amplitude * np.exp(-r_squared / waist**2)).astype(np.complex128)


def make_circular_aperture(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    radius: float,
) -> FloatArray:
    """Create a centered binary circular aperture."""

    if radius <= 0:
        msg = "radius must be positive."
        raise ValueError(msg)
    x_grid, y_grid = coordinate_grid(shape, dx)
    aperture = (x_grid**2 + y_grid**2) <= radius**2
    return aperture.astype(np.float64)
