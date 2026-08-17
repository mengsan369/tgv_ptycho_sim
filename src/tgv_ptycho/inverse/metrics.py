"""Error metrics for complex fields and phase maps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def complex_relative_error(
    U_rec: NDArray[np.complexfloating],
    U_true: NDArray[np.complexfloating],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return `||U_rec - U_true||_2 / ||U_true||_2`."""

    rec = np.asarray(U_rec, dtype=np.complex128)
    true = np.asarray(U_true, dtype=np.complex128)
    if rec.shape != true.shape:
        msg = "U_rec and U_true must have the same shape."
        raise ValueError(msg)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != rec.shape or not np.any(selected):
            msg = "mask must match the field shape and select at least one pixel."
            raise ValueError(msg)
        rec = rec[selected]
        true = true[selected]
    numerator = float(np.sum(np.abs(rec - true) ** 2))
    denominator = float(np.sum(np.abs(true) ** 2))
    return float(np.sqrt(numerator / (denominator + np.finfo(float).eps)))


def amplitude_rmse(
    U_rec: NDArray[np.complexfloating],
    U_true: NDArray[np.complexfloating],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return RMSE between amplitudes."""

    rec = np.asarray(U_rec)
    true = np.asarray(U_true)
    if rec.shape != true.shape:
        msg = "U_rec and U_true must have the same shape."
        raise ValueError(msg)
    difference = np.abs(rec) - np.abs(true)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != rec.shape or not np.any(selected):
            msg = "mask must match the field shape and select at least one pixel."
            raise ValueError(msg)
        difference = difference[selected]
    return float(np.sqrt(np.mean(difference**2)))


def phase_rmse(
    phi_rec: NDArray[np.floating],
    phi_true: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return wrapped phase RMSE in radians."""

    wrapped = np.angle(np.exp(1j * (np.asarray(phi_rec) - np.asarray(phi_true))))
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != wrapped.shape or not np.any(selected):
            msg = "mask must match the phase shape and select at least one pixel."
            raise ValueError(msg)
        wrapped = wrapped[selected]
    return float(np.sqrt(np.mean(wrapped**2)))


def align_global_phase(
    U_rec: NDArray[np.complexfloating],
    U_true: NDArray[np.complexfloating],
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.complex128], float]:
    """Align a reconstructed field to truth by one global phase factor.

    Intensity-only ptychography cannot determine a constant global phase. The
    returned phase in radians is multiplied onto `U_rec` before simulation
    truth metrics or comparison figures are computed.
    """

    rec = np.asarray(U_rec, dtype=np.complex128)
    true = np.asarray(U_true, dtype=np.complex128)
    if rec.shape != true.shape:
        msg = "U_rec and U_true must have the same shape."
        raise ValueError(msg)
    if mask is None:
        selected_rec = rec.ravel()
        selected_true = true.ravel()
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != rec.shape or not np.any(selected):
            msg = "mask must match the field shape and select at least one pixel."
            raise ValueError(msg)
        selected_rec = rec[selected]
        selected_true = true[selected]
    correlation = np.sum(np.conj(selected_rec) * selected_true)
    phase_offset = float(np.angle(correlation)) if correlation != 0 else 0.0
    return (rec * np.exp(1j * phase_offset)).astype(np.complex128), phase_offset


def align_affine_phase_and_complex_gain(
    U_rec: NDArray[np.complexfloating],
    U_true: NDArray[np.complexfloating],
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.complex128], complex, tuple[float, float]]:
    """Align blind-reconstruction affine phase and complex scale to truth.

    This helper is for simulation evaluation only. It estimates a periodic
    linear phase ramp on the discrete FFT grid, then applies the least-squares
    complex gain. Returned ramp components are ``(ky, kx)`` in rad/pixel.
    """

    rec = np.asarray(U_rec, dtype=np.complex128)
    true = np.asarray(U_true, dtype=np.complex128)
    if rec.shape != true.shape or rec.ndim != 2:
        msg = "U_rec and U_true must be same-shaped 2D arrays."
        raise ValueError(msg)
    if mask is None:
        selected = np.ones(rec.shape, dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != rec.shape or not np.any(selected):
            msg = "mask must match the field shape and select at least one pixel."
            raise ValueError(msg)

    cross = np.where(selected, np.conj(rec) * true, 0.0)
    peak_y, peak_x = np.unravel_index(np.argmax(np.abs(np.fft.fft2(cross))), rec.shape)
    ky = float(2.0 * np.pi * np.fft.fftfreq(rec.shape[0])[peak_y])
    kx = float(2.0 * np.pi * np.fft.fftfreq(rec.shape[1])[peak_x])
    yy, xx = np.indices(rec.shape, dtype=np.float64)
    ramped = rec * np.exp(1j * (ky * yy + kx * xx))
    selected_rec = ramped[selected]
    selected_true = true[selected]
    denominator = np.sum(np.conj(selected_rec) * selected_rec)
    gain = (
        complex(np.sum(np.conj(selected_rec) * selected_true) / denominator)
        if abs(denominator) > np.finfo(float).eps
        else 1.0 + 0.0j
    )
    return (
        (gain * ramped).astype(np.complex128),
        gain,
        (ky, kx),
    )
