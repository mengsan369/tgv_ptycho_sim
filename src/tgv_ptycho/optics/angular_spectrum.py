"""Angular spectrum wave propagation."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ComplexArray: TypeAlias = NDArray[np.complexfloating]


def _normalize_dx(dx: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(dx, tuple):
        if len(dx) != 2:
            msg = "dx tuple must be (dy, dx) in meters."
            raise ValueError(msg)
        return float(dx[0]), float(dx[1])
    value = float(dx)
    return value, value


def make_angular_spectrum_transfer(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    wavelength: float,
    z: float,
    n: float = 1.0,
    bandlimit: bool = True,
) -> NDArray[np.complex128]:
    """Build a reusable angular-spectrum transfer array.

    The array maps a source spectrum to the propagated spectrum.  Its complex
    conjugate is the Euclidean adjoint transfer.  With ``bandlimit=True`` the
    adjoint is not an inverse because the evanescent-frequency mask is a
    projection.
    """

    if len(shape) != 2 or min(shape) <= 0:
        msg = "shape must be a positive (ny, nx) tuple."
        raise ValueError(msg)
    if wavelength <= 0:
        msg = "wavelength must be positive."
        raise ValueError(msg)
    if n <= 0:
        msg = "n must be positive."
        raise ValueError(msg)
    dy, dx_x = _normalize_dx(dx)
    if dy <= 0 or dx_x <= 0:
        msg = "dx entries must be positive."
        raise ValueError(msg)

    ny, nx = int(shape[0]), int(shape[1])
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx_x)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_medium = 2.0 * np.pi * n / wavelength
    kz_squared = k_medium**2 - kx_grid**2 - ky_grid**2

    if bandlimit:
        propagating = kz_squared >= 0.0
        kz = np.zeros_like(kz_squared, dtype=np.float64)
        kz[propagating] = np.sqrt(kz_squared[propagating])
        transfer = np.zeros(shape, dtype=np.complex128)
        transfer[propagating] = np.exp(1j * kz[propagating] * z)
        return transfer

    kz = np.sqrt(kz_squared.astype(np.complex128))
    return np.exp(1j * kz * z).astype(np.complex128, copy=False)


def apply_angular_spectrum_transfer(
    field: ComplexArray,
    transfer: ComplexArray,
) -> NDArray[np.complex128]:
    """Apply a precomputed angular-spectrum transfer to a 2D field."""

    values = np.asarray(field, dtype=np.complex128)
    kernel = np.asarray(transfer, dtype=np.complex128)
    if values.ndim != 2:
        msg = "field must be a 2D complex array."
        raise ValueError(msg)
    if kernel.shape != values.shape:
        msg = "transfer must match field shape."
        raise ValueError(msg)
    return np.fft.ifft2(np.fft.fft2(values) * kernel).astype(
        np.complex128, copy=False
    )


def angular_spectrum_propagate(
    U: ComplexArray,
    dx: float | tuple[float, float],
    wavelength: float,
    z: float,
    n: float = 1.0,
    bandlimit: bool = True,
) -> ComplexArray:
    r"""Propagate a 2D scalar field with the angular spectrum method.

    Parameters
    ----------
    U:
        Complex field sampled on a regular `(ny, nx)` grid.
    dx:
        Sampling interval in meters. A scalar means square pixels; a tuple is
        interpreted as `(dy, dx)`.
    wavelength:
        Vacuum wavelength in meters.
    z:
        Propagation distance in meters. Positive and negative distances are
        both allowed.
    n:
        Refractive index of the homogeneous propagation medium.
    bandlimit:
        If true, remove evanescent components where
        `kx**2 + ky**2 > (2*pi*n/wavelength)**2`.

    Returns
    -------
    U_z:
        Propagated complex field with the same shape as `U`.

    Notes
    -----
    The angular spectrum propagation formula is

    ```text
    U(x, y, z) = F^-1{ F{U(x, y, 0)} * exp(i * kz * z) }
    kz = sqrt(k^2 - kx^2 - ky^2)
    k = 2*pi*n / wavelength
    ```

    The current bandlimit is deliberately simple: non-propagating evanescent
    frequencies are set to zero. A stricter Matsushima-style bandlimit can be
    added later if needed.
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

    transfer = make_angular_spectrum_transfer(
        field.shape, dx, wavelength, z, n=n, bandlimit=bandlimit
    )
    return apply_angular_spectrum_transfer(field, transfer)
