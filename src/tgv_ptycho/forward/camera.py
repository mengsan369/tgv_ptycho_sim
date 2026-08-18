"""Simple detector helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_square_pixel_mtf(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    pixel_size_m: float,
) -> NDArray[np.float64]:
    r"""Return the ideal square-pixel intensity MTF on an FFT grid.

    ``dx`` follows the project convention ``(dy, dx)``.  The returned MTF is
    ``sinc(pixel_size_m * fx) * sinc(pixel_size_m * fy)``, where NumPy's
    normalized ``sinc`` is used.  This is a periodic, band-limited diagnostic
    operator; it does not model a measured detector PSF or pixel gaps.
    """

    if len(shape) != 2 or min(shape) <= 0:
        msg = "shape must be a positive (ny, nx) tuple."
        raise ValueError(msg)
    if isinstance(dx, tuple):
        if len(dx) != 2:
            msg = "dx tuple must be (dy, dx) in meters."
            raise ValueError(msg)
        dy_m, dx_m = float(dx[0]), float(dx[1])
    else:
        dy_m = dx_m = float(dx)
    pixel_size = float(pixel_size_m)
    if (
        not np.isfinite(dy_m)
        or not np.isfinite(dx_m)
        or dy_m <= 0.0
        or dx_m <= 0.0
    ):
        msg = "dx entries must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(pixel_size) or pixel_size <= 0.0:
        msg = "pixel_size_m must be finite and positive."
        raise ValueError(msg)

    fy = np.fft.fftfreq(int(shape[0]), d=dy_m)
    fx = np.fft.fftfreq(int(shape[1]), d=dx_m)
    mtf = np.sinc(pixel_size * fy)[:, None] * np.sinc(pixel_size * fx)[None, :]
    return np.asarray(mtf, dtype=np.float64)


def periodic_square_pixel_average(
    intensity: NDArray[np.floating],
    dx: float | tuple[float, float],
    pixel_size_m: float,
) -> NDArray[np.complex128]:
    r"""Apply an ideal square-pixel area average to periodic intensity data.

    The complex result is returned deliberately so callers can measure the
    numerical imaginary leakage before taking the real component.  Exact
    finite-pixel integration of an under-resolved non-periodic field is not
    implied by this diagnostic operator.
    """

    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim < 2:
        msg = "intensity must have at least two dimensions."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "intensity must contain only finite values."
        raise ValueError(msg)
    mtf = make_square_pixel_mtf(values.shape[-2:], dx, pixel_size_m)
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    return np.asarray(
        np.fft.ifft2(spectrum * mtf, axes=(-2, -1)),
        dtype=np.complex128,
    )


def positive_midpoint_pixel_average(
    intensity_nodes: NDArray[np.floating], factor: int
) -> NDArray[np.float64]:
    """Average staggered midpoint intensities with positive uniform weights.

    The last two axes contain ``factor x factor`` midpoint nodes per physical
    detector pixel.  Their sizes must be divisible by ``factor``.  The output
    preserves all leading axes and is nonnegative whenever the input is.
    """

    values = np.asarray(intensity_nodes, dtype=np.float64)
    if values.ndim < 2:
        msg = "intensity_nodes must have at least two dimensions."
        raise ValueError(msg)
    if not isinstance(factor, (int, np.integer)) or int(factor) <= 0:
        msg = "factor must be a positive integer."
        raise ValueError(msg)
    q = int(factor)
    ny, nx = values.shape[-2:]
    if ny % q or nx % q:
        msg = "node-grid shape must be divisible by factor."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "intensity_nodes must contain only finite values."
        raise ValueError(msg)
    reshaped = values.reshape(*values.shape[:-2], ny // q, q, nx // q, q)
    return np.asarray(reshaped.mean(axis=(-3, -1)), dtype=np.float64)


def intensity_from_field(U: NDArray[np.complexfloating]) -> NDArray[np.float64]:
    """Return detector intensity `|U|^2`."""

    return np.abs(U) ** 2


def normalize_to_bit_depth(
    intensity: NDArray[np.floating],
    bit_depth: int = 12,
) -> NDArray[np.uint16]:
    """Normalize intensity to an unsigned integer detector range."""

    max_value = 2**bit_depth - 1
    arr = np.asarray(intensity, dtype=np.float64)
    peak = arr.max()
    if peak <= 0:
        return np.zeros_like(arr, dtype=np.uint16)
    return np.rint(np.clip(arr / peak, 0.0, 1.0) * max_value).astype(np.uint16)
