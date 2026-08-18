"""Forward model for sample A generated probe and scanning sample B."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.integer_shift import shift_field_integer_pixels
from tgv_ptycho.forward.noise import apply_noise
from tgv_ptycho.optics.angular_spectrum import (
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)
from tgv_ptycho.optics.fields import make_plane_wave


def _roll_object_integer_pixels(
    B_object: NDArray[np.complexfloating],
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
) -> NDArray[np.complex128]:
    return shift_field_integer_pixels(
        B_object, position_xy, dx, boundary="periodic"
    )


def _validate_exit_field_inputs(
    U_A_exit: NDArray[np.complexfloating],
    B_object: NDArray[np.complexfloating],
    scan_positions: NDArray[np.floating],
    dx: float | tuple[float, float],
    wavelength: float,
    z_AB: float,
    z_BC: float,
    external_medium_index: float,
    bandlimit: bool,
    alias_control: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]:
    """Validate and normalize the exit-field-to-detector inputs."""

    exit_field = np.asarray(U_A_exit, dtype=np.complex128)
    sample_b = np.asarray(B_object, dtype=np.complex128)
    positions = np.asarray(scan_positions, dtype=np.float64)
    if exit_field.ndim != 2:
        msg = "U_A_exit must be a 2D complex field."
        raise ValueError(msg)
    if sample_b.ndim != 2 or sample_b.shape != exit_field.shape:
        msg = "B_object must be 2D and match U_A_exit shape."
        raise ValueError(msg)
    if not np.all(np.isfinite(exit_field)):
        msg = "U_A_exit must contain only finite values."
        raise ValueError(msg)
    if not np.all(np.isfinite(sample_b)):
        msg = "B_object must contain only finite values."
        raise ValueError(msg)
    if positions.ndim != 2 or positions.shape[1] != 2:
        msg = "scan_positions must have shape (num_positions, 2)."
        raise ValueError(msg)
    if not np.all(np.isfinite(positions)):
        msg = "scan_positions must contain only finite (x, y) values."
        raise ValueError(msg)
    if isinstance(dx, tuple):
        if len(dx) != 2:
            msg = "dx tuple must be (dy, dx) in meters."
            raise ValueError(msg)
        sampling = np.asarray(dx, dtype=np.float64)
    else:
        sampling = np.asarray([dx, dx], dtype=np.float64)
    if not np.all(np.isfinite(sampling)) or np.any(sampling <= 0.0):
        msg = "dx entries must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        msg = "wavelength must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(z_AB) or not np.isfinite(z_BC):
        msg = "z_AB and z_BC must be finite."
        raise ValueError(msg)
    if not np.isfinite(external_medium_index) or external_medium_index <= 0.0:
        msg = "external_medium_index must be finite and positive."
        raise ValueError(msg)
    if not isinstance(bandlimit, bool):
        msg = "bandlimit must be a bool."
        raise TypeError(msg)
    if not isinstance(alias_control, bool):
        msg = "alias_control must be a bool."
        raise TypeError(msg)
    if alias_control and not bandlimit:
        msg = "alias_control requires bandlimit=True."
        raise ValueError(msg)
    return exit_field, sample_b, positions


def simulate_exit_field_B_forward(
    U_A_exit: NDArray[np.complexfloating],
    B_object: NDArray[np.complexfloating],
    scan_positions: NDArray[np.floating],
    dx: float | tuple[float, float],
    wavelength: float,
    z_AB: float,
    z_BC: float,
    noise_config: dict[str, Any] | None = None,
    object_boundary: str = "periodic",
    object_boundary_value: complex = 1.0 + 0.0j,
    external_medium_index: float = 1.0,
    bandlimit: bool = True,
    alias_control: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
    dict[str, Any],
]:
    """Propagate an A-exit field to scanned B and detector intensity.

    ``U_A_exit`` is already the full complex field at the physical exit plane
    of sample A.  This function never multiplies it by an incident field.
    ``z_AB`` therefore starts at that exit plane.  A-to-B and B-to-detector
    angular-spectrum transfers are each built once and reused for every scan
    position.

    Scan positions have columns ``(x, y)`` in meters.  The default periodic
    object boundary preserves the historical integer-shift ``np.roll`` model.
    """

    exit_field, sample_b, positions = _validate_exit_field_inputs(
        U_A_exit,
        B_object,
        scan_positions,
        dx,
        wavelength,
        z_AB,
        z_BC,
        external_medium_index,
        bandlimit,
        alias_control,
    )
    transfer_AB = make_angular_spectrum_transfer(
        exit_field.shape,
        dx,
        wavelength,
        z_AB,
        n=external_medium_index,
        bandlimit=bandlimit,
        alias_control=alias_control,
    )
    transfer_BC = make_angular_spectrum_transfer(
        exit_field.shape,
        dx,
        wavelength,
        z_BC,
        n=external_medium_index,
        bandlimit=bandlimit,
        alias_control=alias_control,
    )
    P_B = apply_angular_spectrum_transfer(exit_field, transfer_AB)

    intensity_stack = np.empty(
        (len(positions), *exit_field.shape), dtype=np.float64
    )
    for idx, position_xy in enumerate(positions):
        shifted_B = shift_field_integer_pixels(
            sample_b,
            position_xy,
            dx,
            boundary=object_boundary,
            fill_value=object_boundary_value,
        )
        detector_field = apply_angular_spectrum_transfer(
            P_B * shifted_B, transfer_BC
        )
        intensity = np.abs(detector_field) ** 2
        intensity_stack[idx] = apply_noise(
            intensity, noise_config=noise_config, seed=None
        )

    if not np.all(np.isfinite(P_B)) or not np.all(np.isfinite(intensity_stack)):
        msg = "Forward result contains non-finite values."
        raise FloatingPointError(msg)

    metadata: dict[str, Any] = {
        "model": "scheme_exit_field_B",
        "input_plane": "sample_A_exit",
        "z_AB_reference_plane": "sample_A_exit",
        "z_AB_m": z_AB,
        "z_BC_m": z_BC,
        "wavelength_m": wavelength,
        "dx_m": dx,
        "external_medium_index": external_medium_index,
        "bandlimit": bandlimit,
        "alias_control": alias_control,
        "num_scan_positions": int(len(positions)),
        "integer_pixel_shifts_only": True,
        "object_boundary": object_boundary,
        "object_boundary_value": object_boundary_value,
        "propagation_transfers_cached": True,
        "todo": "Add subpixel interpolation and position refinement.",
    }
    return intensity_stack, P_B, sample_b, metadata


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
    external_medium_index: float = 1.0,
    bandlimit: bool = True,
    alias_control: bool = False,
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
    if A.ndim != 2 or B.shape != A.shape:
        msg = "A_transmission_or_field and B_object must have the same shape."
        raise ValueError(msg)

    if incident_field is None:
        U0 = make_plane_wave(A.shape, dx, wavelength)
    else:
        U0 = np.asarray(incident_field, dtype=np.complex128)
        if U0.shape != A.shape:
            msg = "incident_field must match A shape."
            raise ValueError(msg)

    result = simulate_exit_field_B_forward(
        A * U0,
        B,
        scan_positions,
        dx,
        wavelength,
        z_AB,
        z_BC,
        noise_config=noise_config,
        object_boundary=object_boundary,
        object_boundary_value=object_boundary_value,
        external_medium_index=external_medium_index,
        bandlimit=bandlimit,
        alias_control=alias_control,
    )
    intensity_stack, P_B, returned_B, metadata = result
    metadata = {
        **metadata,
        "model": "scheme_probe_B",
        "input_semantics": "A_transmission_or_field_times_incident_field",
    }
    return intensity_stack, P_B, returned_B, metadata
