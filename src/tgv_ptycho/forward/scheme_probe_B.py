"""Forward model for sample A generated probe and scanning sample B."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.integer_shift import shift_field_integer_pixels
from tgv_ptycho.forward.noise import apply_noise
from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.fields import make_plane_wave


def _roll_object_integer_pixels(
    B_object: NDArray[np.complexfloating],
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
) -> NDArray[np.complex128]:
    return shift_field_integer_pixels(
        B_object, position_xy, dx, boundary="periodic"
    )


def simulate_probe_B_forward(
    A_transmission_or_field: NDArray[np.complexfloating],
    B_object: NDArray[np.complexfloating],
    scan_positions: NDArray[np.floating],
    dx: float | tuple[float, float],
    wavelength: float,
    z_AB: float,
    z_BC: float,
    incident_field: NDArray[np.complexfloating] | None = None,
    noise_config: dict[str, Any] | None = None,
    object_boundary: str = "periodic",
    object_boundary_value: complex = 1.0 + 0.0j,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
    dict[str, Any],
]:
    """Simulate scheme 1: A creates a probe on B, then B is scanned.

    ``object_boundary="periodic"`` preserves the historical ``np.roll``
    model.  ``object_boundary="constant"`` is a paired finite-FOV control;
    the default exterior transmission is one.  Scan shifts remain integer
    pixels and subpixel interpolation remains a future extension.
    """

    A = np.asarray(A_transmission_or_field, dtype=np.complex128)
    B = np.asarray(B_object, dtype=np.complex128)
    positions = np.asarray(scan_positions, dtype=np.float64)
    if A.shape != B.shape:
        msg = "A_transmission_or_field and B_object must have the same shape."
        raise ValueError(msg)
    if positions.ndim != 2 or positions.shape[1] != 2:
        msg = "scan_positions must have shape (num_positions, 2)."
        raise ValueError(msg)

    if incident_field is None:
        U0 = make_plane_wave(A.shape, dx, wavelength)
    else:
        U0 = np.asarray(incident_field, dtype=np.complex128)
        if U0.shape != A.shape:
            msg = "incident_field must match A shape."
            raise ValueError(msg)

    P_B = angular_spectrum_propagate(A * U0, dx, wavelength, z_AB)

    intensity_stack = np.empty((len(positions), *A.shape), dtype=np.float64)
    for idx, position_xy in enumerate(positions):
        shifted_B = shift_field_integer_pixels(
            B,
            position_xy,
            dx,
            boundary=object_boundary,
            fill_value=object_boundary_value,
        )
        exit_wave = P_B * shifted_B
        U_det = angular_spectrum_propagate(exit_wave, dx, wavelength, z_BC)
        intensity = np.abs(U_det) ** 2
        intensity_stack[idx] = apply_noise(
            intensity, noise_config=noise_config, seed=None
        )

    metadata: dict[str, Any] = {
        "model": "scheme_probe_B",
        "z_AB_m": z_AB,
        "z_BC_m": z_BC,
        "wavelength_m": wavelength,
        "dx_m": dx,
        "num_scan_positions": int(len(positions)),
        "integer_pixel_shifts_only": True,
        "object_boundary": object_boundary,
        "object_boundary_value": object_boundary_value,
        "todo": "Add subpixel interpolation and position refinement.",
    }
    return intensity_stack, P_B, B, metadata
