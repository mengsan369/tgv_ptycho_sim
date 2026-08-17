"""Backpropagation from the B plane toward sample A."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate


def backpropagate_probe_to_A(
    P_B_rec: NDArray[np.complexfloating],
    dx: float | tuple[float, float],
    wavelength: float,
    z_AB: float,
) -> NDArray[np.complex128]:
    """Backpropagate recovered B-plane probe by `-z_AB`."""

    return angular_spectrum_propagate(P_B_rec, dx, wavelength, -z_AB)


def remove_reference_phase_plane(
    field: NDArray[np.complexfloating],
    reference_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], dict[str, float]]:
    """Normalize a field using a known constant blank reference region.

    A least-squares plane is fitted to the unwrapped reference phase and the
    median reference amplitude sets the scale. No simulation truth is used.
    Phase slopes are returned in rad/pixel in ``(y, x)`` order.
    """

    values = np.asarray(field, dtype=np.complex128)
    selected = np.asarray(reference_mask, dtype=bool)
    if values.ndim != 2:
        msg = "field must be a 2D complex array."
        raise ValueError(msg)
    if selected.shape != values.shape or np.count_nonzero(selected) < 3:
        msg = "reference_mask must match field and select at least three pixels."
        raise ValueError(msg)

    amplitude_scale = float(np.median(np.abs(values[selected])))
    if not np.isfinite(amplitude_scale) or amplitude_scale <= np.finfo(float).eps:
        msg = "Reference region must have finite, non-zero amplitude."
        raise ValueError(msg)

    phase = np.unwrap(np.unwrap(np.angle(values), axis=0), axis=1)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    y_values = yy[selected]
    x_values = xx[selected]
    phase_values = phase[selected]
    y_mean = float(np.mean(y_values))
    x_mean = float(np.mean(x_values))
    phase_mean = float(np.mean(phase_values))
    y_centered = y_values - y_mean
    x_centered = x_values - x_mean
    phase_centered = phase_values - phase_mean
    sum_yy = float(np.sum(y_centered * y_centered))
    sum_xx = float(np.sum(x_centered * x_centered))
    sum_yx = float(np.sum(y_centered * x_centered))
    sum_yp = float(np.sum(y_centered * phase_centered))
    sum_xp = float(np.sum(x_centered * phase_centered))
    determinant = sum_yy * sum_xx - sum_yx**2
    if abs(determinant) <= np.finfo(float).eps:
        msg = "reference_mask geometry cannot determine a 2D phase plane."
        raise ValueError(msg)
    slope_y = (sum_yp * sum_xx - sum_xp * sum_yx) / determinant
    slope_x = (sum_xp * sum_yy - sum_yp * sum_yx) / determinant
    offset = phase_mean - slope_y * y_mean - slope_x * x_mean
    coefficients = (offset, slope_y, slope_x)
    phase_plane = coefficients[0] + coefficients[1] * yy + coefficients[2] * xx
    corrected = values * np.exp(-1j * phase_plane) / amplitude_scale
    metadata = {
        "reference_amplitude_scale": amplitude_scale,
        "reference_phase_offset_rad": coefficients[0],
        "reference_phase_slope_y_rad_per_px": coefficients[1],
        "reference_phase_slope_x_rad_per_px": coefficients[2],
    }
    return corrected.astype(np.complex128), metadata


def recover_thin_phase_A(
    P_B_rec: NDArray[np.complexfloating],
    incident_field: NDArray[np.complexfloating],
    reference_mask: NDArray[np.bool_],
    dx: float | tuple[float, float],
    wavelength: float,
    z_AB: float,
) -> dict[str, object]:
    """Backpropagate a recovered probe and infer a pure-phase sample A.

    The returned raw transmission is truth-free. The reference-corrected field
    uses only the supplied known blank region, and ``A_phase_only`` applies the
    experiment's pure-phase prior by setting amplitude to one.
    """

    incident = np.asarray(incident_field, dtype=np.complex128)
    probe = np.asarray(P_B_rec, dtype=np.complex128)
    if incident.shape != probe.shape or incident.ndim != 2:
        msg = "incident_field and P_B_rec must be same-shaped 2D arrays."
        raise ValueError(msg)
    if np.any(np.abs(incident) <= np.finfo(float).eps):
        msg = "incident_field must be non-zero everywhere."
        raise ValueError(msg)

    field_after_A = backpropagate_probe_to_A(probe, dx, wavelength, z_AB)
    raw = field_after_A / incident
    corrected, reference_metadata = remove_reference_phase_plane(raw, reference_mask)
    phase_only = np.exp(1j * np.angle(corrected)).astype(np.complex128)
    phase_only[np.asarray(reference_mask, dtype=bool)] = 1.0 + 0.0j
    return {
        "field_after_A_rec": field_after_A,
        "A_rec_raw": raw.astype(np.complex128),
        "A_rec_reference_corrected": corrected,
        "A_rec_phase_only": phase_only,
        "reference_correction": reference_metadata,
    }
