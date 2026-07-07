"""Minimal ePIE reconstruction scaffold."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.sampling import position_to_pixel_shift


def _shift_object(
    obj: NDArray[np.complexfloating],
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
) -> NDArray[np.complex128]:
    shift_y, shift_x = position_to_pixel_shift(position_xy, dx)
    return np.roll(obj, shift=(shift_y, shift_x), axis=(0, 1)).astype(
        np.complex128, copy=False
    )


def _unshift_delta(
    delta_shifted: NDArray[np.complexfloating],
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
) -> NDArray[np.complex128]:
    shift_y, shift_x = position_to_pixel_shift(position_xy, dx)
    return np.roll(delta_shifted, shift=(-shift_y, -shift_x), axis=(0, 1)).astype(
        np.complex128, copy=False
    )


def epie_reconstruct(
    I_stack: NDArray[np.floating],
    scan_positions: NDArray[np.floating],
    dx: float | tuple[float, float],
    wavelength: float,
    z_BC: float,
    num_iters: int = 300,
    beta_probe: float = 0.2,
    beta_object: float = 0.2,
    init_probe: NDArray[np.complexfloating] | None = None,
    init_object: NDArray[np.complexfloating] | None = None,
) -> dict[str, Any]:
    """Run a compact ePIE-style reconstruction loop.

    Algorithm sketch:
    1. For each scan position, shift the current object estimate.
    2. Form the exit wave `psi = probe * object_j`.
    3. Forward propagate to the detector.
    4. Replace predicted amplitude with measured amplitude.
    5. Backpropagate to the object/probe plane.
    6. Update probe and object using ePIE-like normalized gradients.
    7. Record a relative amplitude loss.

    This implementation is intentionally minimal and assumes full-field
    periodic integer shifts. It is useful for shape checks and early synthetic
    experiments, but it is not yet a production ptychography engine.
    """

    intensities = np.asarray(I_stack, dtype=np.float64)
    positions = np.asarray(scan_positions, dtype=np.float64)
    if intensities.ndim != 3:
        msg = "I_stack must have shape (num_positions, ny, nx)."
        raise ValueError(msg)
    if positions.shape != (intensities.shape[0], 2):
        msg = "scan_positions must have shape (num_positions, 2)."
        raise ValueError(msg)
    if num_iters < 0:
        msg = "num_iters must be non-negative."
        raise ValueError(msg)

    shape = intensities.shape[1:]
    if init_probe is None:
        mean_amp = np.sqrt(np.maximum(intensities.mean(axis=0), 0.0))
        probe = mean_amp.astype(np.complex128)
    else:
        probe = np.asarray(init_probe, dtype=np.complex128).copy()
    if init_object is None:
        obj = np.ones(shape, dtype=np.complex128)
    else:
        obj = np.asarray(init_object, dtype=np.complex128).copy()
    if probe.shape != shape or obj.shape != shape:
        msg = "init_probe and init_object must match detector frame shape."
        raise ValueError(msg)

    measured_amp = np.sqrt(np.maximum(intensities, 0.0))
    loss_curve: list[float] = []

    iterator = trange(num_iters, desc="ePIE", leave=False) if num_iters > 0 else []
    eps = np.finfo(float).eps
    for _ in iterator:
        accum_loss = 0.0
        for idx, position_xy in enumerate(positions):
            obj_shifted = _shift_object(obj, position_xy, dx)
            probe_old = probe.copy()
            exit_wave = probe * obj_shifted
            U_det = angular_spectrum_propagate(exit_wave, dx, wavelength, z_BC)
            pred_amp = np.abs(U_det)
            corrected_det = measured_amp[idx] * np.exp(1j * np.angle(U_det))
            corrected_exit = angular_spectrum_propagate(
                corrected_det, dx, wavelength, -z_BC
            )
            delta = corrected_exit - exit_wave

            probe += (
                beta_probe
                * np.conj(obj_shifted)
                / (np.max(np.abs(obj_shifted)) ** 2 + eps)
                * delta
            )

            obj_delta_shifted = (
                beta_object
                * np.conj(probe_old)
                / (np.max(np.abs(probe_old)) ** 2 + eps)
                * delta
            )
            obj += _unshift_delta(obj_delta_shifted, position_xy, dx)

            accum_loss += float(
                np.linalg.norm(pred_amp - measured_amp[idx])
                / (np.linalg.norm(measured_amp[idx]) + eps)
            )
        loss_curve.append(accum_loss / len(positions))

    return {
        "P_B_rec": probe,
        "B_rec": obj,
        "loss_curve": np.asarray(loss_curve, dtype=np.float64),
        "metadata": {
            "algorithm": "minimal_epie_scaffold",
            "integer_pixel_shifts_only": True,
            "num_iters": num_iters,
            "todo": "Add patch support, subpixel shifts, constraints, and robust normalization.",
        },
    }
