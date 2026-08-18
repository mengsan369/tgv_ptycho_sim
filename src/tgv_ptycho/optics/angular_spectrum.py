"""Angular spectrum wave propagation."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ComplexArray: TypeAlias = NDArray[np.complexfloating]


def _normalize_dx(dx: float | tuple[float, float]) -> tuple[float, float]:
    """Normalize scalar ``dx`` or tuple ``(dy, dx)``.

    The first tuple value is the y sampling and the second is the x sampling.
    """
    if isinstance(dx, tuple):
        if len(dx) != 2:
            msg = "dx tuple must be (dy, dx) in meters."
            raise ValueError(msg)
        return float(dx[0]), float(dx[1])
    value = float(dx)
    return value, value


def make_transfer_sampling_alias_mask(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    wavelength: float,
    z: float,
    n: float = 1.0,
) -> NDArray[np.bool_]:
    r"""Return the Matsushima exact common-ellipse Nyquist mask.

    The mask controls aliasing of the sampled angular-spectrum transfer
    function on the *actual same-grid FFT frequency sampling*.  It applies
    the two-dimensional local-phase-frequency conditions from Matsushima and
    Shimobaba (Optics Express 17, 19662-19673, 2009):

    ``du^-1 >= 2 |f_u|`` and ``dv^-1 >= 2 |f_v|``.

    This mask does not linearize the circular FFT convolution and does not
    create an open boundary.  ``wavelength`` is the vacuum wavelength and
    ``n`` sets the wavelength in the homogeneous medium.
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
    if not np.isfinite(z):
        msg = "z must be finite."
        raise ValueError(msg)
    dy, dx_x = _normalize_dx(dx)
    if dy <= 0 or dx_x <= 0:
        msg = "dx entries must be positive."
        raise ValueError(msg)

    ny, nx = int(shape[0]), int(shape[1])
    v = np.fft.fftfreq(ny, d=dy)
    u = np.fft.fftfreq(nx, d=dx_x)
    u_grid, v_grid = np.meshgrid(u, v)
    inverse_medium_wavelength = float(n / wavelength)
    du = 1.0 / (nx * dx_x)
    dv = 1.0 / (ny * dy)
    distance = abs(float(z))
    u_limit = inverse_medium_wavelength / np.sqrt(
        1.0 + (2.0 * du * distance) ** 2
    )
    v_limit = inverse_medium_wavelength / np.sqrt(
        1.0 + (2.0 * dv * distance) ** 2
    )
    inverse_lambda_squared = inverse_medium_wavelength**2
    tolerance = 32.0 * np.finfo(np.float64).eps
    propagating = u_grid**2 + v_grid**2 <= (
        inverse_lambda_squared * (1.0 + tolerance)
    )
    u_nyquist = (
        u_grid**2 / u_limit**2
        + v_grid**2 / inverse_lambda_squared
        <= 1.0 + tolerance
    )
    v_nyquist = (
        u_grid**2 / inverse_lambda_squared
        + v_grid**2 / v_limit**2
        <= 1.0 + tolerance
    )
    return np.asarray(
        propagating & u_nyquist & v_nyquist,
        dtype=np.bool_,
    )


def make_angular_spectrum_transfer(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    wavelength: float,
    z: float,
    n: float = 1.0,
    bandlimit: bool = True,
    alias_control: bool = False,
) -> NDArray[np.complex128]:
    """Build a reusable angular-spectrum transfer array.

    The array maps a source spectrum to the propagated spectrum.  Its complex
    conjugate is the Euclidean adjoint transfer.  With ``bandlimit=True`` the
    adjoint is not an inverse because the evanescent-frequency mask is a
    projection.  ``alias_control=True`` additionally applies the exact
    common-ellipse transfer-sampling mask on the same periodic FFT grid.
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
    if not isinstance(alias_control, bool):
        msg = "alias_control must be a bool."
        raise TypeError(msg)
    if alias_control and not bandlimit:
        msg = "alias_control requires bandlimit=True."
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
        if alias_control:
            propagating &= make_transfer_sampling_alias_mask(
                shape,
                dx,
                wavelength,
                z,
                n=n,
            )
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
    alias_control: bool = False,
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
    alias_control:
        If true, also apply the Matsushima exact common-ellipse Nyquist mask
        for transfer-function sampling on this same periodic FFT grid.

    Returns
    -------
    U_z:
        Propagated complex field with the same shape as `U`.
        Input and output share the x-y coordinate system; the z coordinate is
        shifted by `z`.

    Notes
    -----
    The angular spectrum propagation formula is

    ```text
    U(x, y, z) = F^-1{ F{U(x, y, 0)} * exp(i * kz * z) }
    kz = sqrt(k^2 - kx^2 - ky^2)
    k = 2*pi*n / wavelength
    ```

    ``alias_control=False`` preserves the historical implementation, whose
    bandlimit only removes non-propagating evanescent frequencies.
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
        field.shape,
        dx,
        wavelength,
        z,
        n=n,
        bandlimit=bandlimit,
        alias_control=alias_control,
    )
    return apply_angular_spectrum_transfer(field, transfer)
