"""Fresnel propagation utilities."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import _normalize_dx

ComplexArray: TypeAlias = NDArray[np.complexfloating]


def fresnel_propagate(
    U: ComplexArray,
    dx: float | tuple[float, float],
    wavelength: float,
    z: float,
    n: float = 1.0,
) -> ComplexArray:
    r"""Propagate a 2D field with the paraxial Fresnel transfer function.

    This is a near-axis approximation to angular spectrum propagation:

    ```text
    kz = sqrt(k^2 - kx^2 - ky^2)
       ~= k - (kx^2 + ky^2) / (2k)

    H_fresnel = exp(i*k*z) * exp(-i*z*(kx^2 + ky^2)/(2k))
    ```

    The project will prefer `angular_spectrum_propagate` for early TGV
    simulations, but Fresnel propagation is useful for checks and approximate
    models.
    """

    field = np.asarray(U, dtype=np.complex128)
    if field.ndim != 2:
        msg = "U must be a 2D complex field."
        raise ValueError(msg)
    if wavelength <= 0:
        msg = "wavelength must be positive."
        raise ValueError(msg)
    if n <= 0:
        msg = "n must be positive."
        raise ValueError(msg)

    dy, dx_x = _normalize_dx(dx)
    ny, nx = field.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx_x)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_medium = 2.0 * np.pi * n / wavelength

    transfer = np.exp(1j * k_medium * z) * np.exp(
        -1j * z * (kx_grid**2 + ky_grid**2) / (2.0 * k_medium)
    )
    return np.fft.ifft2(np.fft.fft2(field) * transfer).astype(
        np.complex128, copy=False
    )
