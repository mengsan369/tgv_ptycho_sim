"""Multi-slice propagation through sample A."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import (
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)

SliceWidths = float | NDArray[np.floating] | list[float] | tuple[float, ...]


def _validated_volume_and_widths(
    n_volume: NDArray[np.floating],
    dz: SliceWidths,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_vol = np.asarray(n_volume, dtype=np.float64)
    if n_vol.ndim != 3:
        msg = "n_volume must have shape (nz, ny, nx)."
        raise ValueError(msg)
    if min(n_vol.shape) <= 0:
        msg = "n_volume dimensions must be positive."
        raise ValueError(msg)
    if not np.all(np.isfinite(n_vol)) or np.any(n_vol <= 0.0):
        msg = "n_volume must contain finite, positive refractive indices."
        raise ValueError(msg)

    width_values = np.asarray(dz, dtype=np.float64)
    if width_values.ndim == 0:
        widths = np.full(n_vol.shape[0], float(width_values), dtype=np.float64)
    elif width_values.ndim == 1 and len(width_values) == n_vol.shape[0]:
        widths = width_values.astype(np.float64, copy=True)
    else:
        msg = "dz must be a scalar or a length-nz array of slice widths."
        raise ValueError(msg)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        msg = "slice widths must be finite and positive."
        raise ValueError(msg)
    return n_vol, widths


def multislice_phase_screen_product(
    incident_field: NDArray[np.complexfloating],
    n_volume: NDArray[np.floating],
    slice_thickness_m: SliceWidths,
    wavelength: float,
    n_ref: float = 1.5,
) -> NDArray[np.complex128]:
    r"""Apply the no-propagation relative phase-screen product.

    The result is ``prod_j exp(i*k0*(n_j-n_ref)*width_j)`` and therefore omits
    the homogeneous reference carrier ``exp(i*k0*n_ref*sum(width_j))``.
    """

    n_vol, widths = _validated_volume_and_widths(n_volume, slice_thickness_m)
    field = np.asarray(incident_field, dtype=np.complex128).copy()
    if field.shape != n_vol.shape[1:]:
        msg = "incident_field shape must match each n_volume slice."
        raise ValueError(msg)
    if not np.all(np.isfinite(field)):
        msg = "incident_field must contain only finite values."
        raise ValueError(msg)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        msg = "wavelength must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(n_ref) or n_ref <= 0.0:
        msg = "n_ref must be finite and positive."
        raise ValueError(msg)

    k0 = 2.0 * np.pi / wavelength
    for index, width in enumerate(widths):
        field *= np.exp(1j * k0 * (n_vol[index] - n_ref) * width)
    return field


def multislice_propagate_A(
    incident_field: NDArray[np.complexfloating],
    n_volume: NDArray[np.floating],
    dx: float | tuple[float, float],
    dz: SliceWidths,
    wavelength: float,
    n_ref: float = 1.5,
    *,
    bandlimit: bool = True,
    alias_control: bool = False,
) -> NDArray[np.complex128]:
    r"""Propagate to sample A's exit with a centered symmetric split-step.

    Slice ``j`` is sampled at its physical center and has its own width.  The
    operator sequence is an entrance half-step, the first centered phase
    screen, center-to-center propagations and phase screens, then an exit
    half-step.  A single slice is therefore ``P(w/2) T P(w/2)``.  The returned
    raw full field lies at ``z=sum(widths)`` and retains the reference-medium
    carrier phase produced by the angular-spectrum propagator.

    ``dz`` accepts the historical scalar spacing or a length-``nz`` vector of
    exact physical slice widths; the latter is required for a shortened final
    slice. ``alias_control=True`` applies the Matsushima transfer-sampling
    mask to every homogeneous propagation step and requires ``bandlimit``.
    """

    field = np.asarray(incident_field, dtype=np.complex128).copy()
    n_vol, widths = _validated_volume_and_widths(n_volume, dz)
    if field.shape != n_vol.shape[1:]:
        msg = "incident_field shape must match each n_volume slice."
        raise ValueError(msg)
    if not np.all(np.isfinite(field)):
        msg = "incident_field must contain only finite values."
        raise ValueError(msg)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        msg = "wavelength must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(n_ref) or n_ref <= 0.0:
        msg = "n_ref must be finite and positive."
        raise ValueError(msg)

    k0 = 2.0 * np.pi / wavelength
    transfer_cache: dict[float, NDArray[np.complex128]] = {}

    def propagate(distance: float) -> NDArray[np.complex128]:
        key = float(distance)
        transfer = transfer_cache.get(key)
        if transfer is None:
            transfer = make_angular_spectrum_transfer(
                field.shape,
                dx,
                wavelength,
                key,
                n=n_ref,
                bandlimit=bandlimit,
                alias_control=alias_control,
            )
            transfer_cache[key] = transfer
        return apply_angular_spectrum_transfer(field, transfer)

    field = propagate(0.5 * widths[0])
    for iz, width in enumerate(widths):
        transmission = np.exp(1j * k0 * (n_vol[iz] - n_ref) * width)
        field *= transmission
        if iz < n_vol.shape[0] - 1:
            center_distance = 0.5 * (width + widths[iz + 1])
            field = propagate(center_distance)
    field = propagate(0.5 * widths[-1])
    return field.astype(np.complex128, copy=False)


def multislice_propagate_streamed_A(
    incident_field: NDArray[np.complexfloating],
    n_slices: Iterable[NDArray[np.floating]],
    dx: float | tuple[float, float],
    slice_thickness_m: SliceWidths,
    wavelength: float,
    n_ref: float = 1.5,
    *,
    bandlimit: bool = True,
    alias_control: bool = False,
) -> NDArray[np.complex128]:
    r"""Stream centered index slices through the symmetric split-step.

    This is algebraically identical to :func:`multislice_propagate_A` but
    consumes one two-dimensional refractive-index slice at a time.  It is
    intended for deterministic procedural volumes that should not be retained
    as a full ``(nz, ny, nx)`` array. ``alias_control=True`` applies the
    Matsushima transfer-sampling mask to every propagation step.
    """

    field = np.asarray(incident_field, dtype=np.complex128).copy()
    if field.ndim != 2 or min(field.shape) <= 0:
        msg = "incident_field must be a nonempty 2D field."
        raise ValueError(msg)
    if not np.all(np.isfinite(field)):
        msg = "incident_field must contain only finite values."
        raise ValueError(msg)
    widths_value = np.asarray(slice_thickness_m, dtype=np.float64)
    if widths_value.ndim == 0:
        msg = "streamed propagation requires an explicit slice-width array."
        raise ValueError(msg)
    if (
        widths_value.ndim != 1
        or len(widths_value) == 0
        or not np.all(np.isfinite(widths_value))
        or np.any(widths_value <= 0.0)
    ):
        msg = "slice_thickness_m must be a nonempty positive 1D array."
        raise ValueError(msg)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        msg = "wavelength must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(n_ref) or n_ref <= 0.0:
        msg = "n_ref must be finite and positive."
        raise ValueError(msg)

    iterator = iter(n_slices)
    k0 = 2.0 * np.pi / wavelength
    transfer_cache: dict[float, NDArray[np.complex128]] = {}

    def propagate(
        values: NDArray[np.complex128], distance: float
    ) -> NDArray[np.complex128]:
        key = float(distance)
        transfer = transfer_cache.get(key)
        if transfer is None:
            transfer = make_angular_spectrum_transfer(
                field.shape,
                dx,
                wavelength,
                key,
                n=n_ref,
                bandlimit=bandlimit,
                alias_control=alias_control,
            )
            transfer_cache[key] = transfer
        return apply_angular_spectrum_transfer(values, transfer)

    field = propagate(field, 0.5 * widths_value[0])
    for index, width in enumerate(widths_value):
        try:
            n_slice = np.asarray(next(iterator), dtype=np.float64)
        except StopIteration as error:
            msg = "n_slices ended before the registered slice-width array."
            raise ValueError(msg) from error
        if (
            n_slice.shape != field.shape
            or not np.all(np.isfinite(n_slice))
            or np.any(n_slice <= 0.0)
        ):
            msg = "each streamed index slice must match the field and be positive."
            raise ValueError(msg)
        field *= np.exp(1j * k0 * (n_slice - n_ref) * width)
        if index < len(widths_value) - 1:
            distance = 0.5 * (width + widths_value[index + 1])
            field = propagate(field, float(distance))
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        msg = "n_slices contains more slices than the registered widths."
        raise ValueError(msg)
    field = propagate(field, 0.5 * widths_value[-1])
    return field.astype(np.complex128, copy=False)
