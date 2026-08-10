"""Shared integer-pixel field shifts for forward and reconstruction models."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.sampling import position_to_pixel_shift

ComplexArray: TypeAlias = NDArray[np.complexfloating]
ShiftBoundary: TypeAlias = Literal["periodic", "constant"]


def _overlap_slices(
    shape: tuple[int, int], shift_y: int, shift_x: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    """Return source/destination slices for a non-periodic integer shift."""

    ny, nx = shape
    if abs(shift_y) >= ny or abs(shift_x) >= nx:
        return None
    if shift_y >= 0:
        source_y, destination_y = slice(0, ny - shift_y), slice(shift_y, ny)
    else:
        source_y, destination_y = slice(-shift_y, ny), slice(0, ny + shift_y)
    if shift_x >= 0:
        source_x, destination_x = slice(0, nx - shift_x), slice(shift_x, nx)
    else:
        source_x, destination_x = slice(-shift_x, nx), slice(0, nx + shift_x)
    return (source_y, source_x), (destination_y, destination_x)


def shift_field_integer_pixels(
    field: ComplexArray,
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
    *,
    boundary: ShiftBoundary = "periodic",
    fill_value: complex = 0.0j,
) -> NDArray[np.complex128]:
    """Shift a 2D field using periodic wrap or a constant exterior value."""

    values = np.asarray(field, dtype=np.complex128)
    if values.ndim != 2:
        msg = "field must be a 2D complex array."
        raise ValueError(msg)
    shift_y, shift_x = position_to_pixel_shift(position_xy, dx)
    if boundary == "periodic":
        return np.roll(values, shift=(shift_y, shift_x), axis=(0, 1)).astype(
            np.complex128, copy=False
        )
    if boundary != "constant":
        msg = "boundary must be 'periodic' or 'constant'."
        raise ValueError(msg)

    shifted = np.full(values.shape, complex(fill_value), dtype=np.complex128)
    overlap = _overlap_slices(values.shape, shift_y, shift_x)
    if overlap is not None:
        source, destination = overlap
        shifted[destination] = values[source]
    return shifted


def unshift_field_delta_integer_pixels(
    shifted_delta: ComplexArray,
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
    *,
    boundary: ShiftBoundary = "periodic",
) -> NDArray[np.complex128]:
    """Apply the adjoint of the linear part of ``shift_field_integer_pixels``."""

    values = np.asarray(shifted_delta, dtype=np.complex128)
    if values.ndim != 2:
        msg = "shifted_delta must be a 2D complex array."
        raise ValueError(msg)
    shift_y, shift_x = position_to_pixel_shift(position_xy, dx)
    if boundary == "periodic":
        return np.roll(values, shift=(-shift_y, -shift_x), axis=(0, 1)).astype(
            np.complex128, copy=False
        )
    if boundary != "constant":
        msg = "boundary must be 'periodic' or 'constant'."
        raise ValueError(msg)

    unshifted = np.zeros(values.shape, dtype=np.complex128)
    overlap = _overlap_slices(values.shape, shift_y, shift_x)
    if overlap is not None:
        source, destination = overlap
        unshifted[source] = values[destination]
    return unshifted
