"""Extended ptychographic iterative engine (ePIE) reconstruction."""

from __future__ import annotations

from collections.abc import Callable
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


def _relative_amplitude_error(
    predicted: NDArray[np.floating], measured: NDArray[np.floating]
) -> float:
    """Return a stable relative L2 error without relying on BLAS."""

    difference_energy = float(np.sum(np.abs(predicted - measured) ** 2))
    measured_energy = float(np.sum(np.abs(measured) ** 2))
    return float(np.sqrt(difference_energy / (measured_energy + np.finfo(float).eps)))


def _field_l2_norm(field: NDArray[np.complexfloating]) -> float:
    """Return an L2 norm without relying on platform BLAS/LAPACK loading."""

    return float(np.sqrt(np.sum(np.abs(field) ** 2)))


def _apply_amplitude_bounds(
    field: NDArray[np.complex128], bounds: tuple[float, float]
) -> NDArray[np.complex128]:
    """Project a complex transmission onto amplitude bounds."""

    lower, upper = float(bounds[0]), float(bounds[1])
    if lower < 0 or upper < lower:
        msg = "object_amplitude_bounds must satisfy 0 <= min <= max."
        raise ValueError(msg)
    amplitude = np.clip(np.abs(field), lower, upper)
    return (amplitude * np.exp(1j * np.angle(field))).astype(np.complex128)


def _evaluate_data_fidelity(
    probe: NDArray[np.complex128],
    obj: NDArray[np.complex128],
    measured_amp: NDArray[np.float64],
    positions: NDArray[np.float64],
    dx: float | tuple[float, float],
    wavelength: float,
    z_BC: float,
) -> float:
    """Evaluate mean relative amplitude error with frozen estimates."""

    total = 0.0
    for index, position_xy in enumerate(positions):
        shifted_object = _shift_object(obj, position_xy, dx)
        detector_field = angular_spectrum_propagate(
            probe * shifted_object, dx, wavelength, z_BC
        )
        total += _relative_amplitude_error(np.abs(detector_field), measured_amp[index])
    return total / len(positions)


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
    update_probe: bool = True,
    shuffle_positions: bool = True,
    seed: int | None = None,
    object_amplitude_bounds: tuple[float, float] | None = None,
    probe_l2_norm_target: float | None = None,
    probe_constraint: Callable[
        [NDArray[np.complex128]], NDArray[np.complex128]
    ]
    | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run a compact full-field ePIE reconstruction.

    Algorithm sketch:
    1. For each scan position, shift the current object estimate.
    2. Form the exit wave `psi = probe * object_j`.
    3. Forward propagate to the detector.
    4. Replace predicted amplitude with measured amplitude.
    5. Backpropagate to the object/probe plane.
    6. Update object and, when requested, probe using normalized ePIE steps.
    7. Record a relative amplitude loss.

    Set `update_probe=False` and provide `init_probe` for known-probe PIE. The
    current implementation assumes full-field periodic integer shifts. Scan
    positions are `(x, y)` in meters and `dx` is the object-plane sampling.
    Subpixel shifts, finite object patches, position refinement and detector
    masks remain future extensions.
    """

    intensities = np.asarray(I_stack, dtype=np.float64)
    positions = np.asarray(scan_positions, dtype=np.float64)
    if intensities.ndim != 3:
        msg = "I_stack must have shape (num_positions, ny, nx)."
        raise ValueError(msg)
    if intensities.shape[0] == 0:
        msg = "I_stack must contain at least one scan position."
        raise ValueError(msg)
    if positions.shape != (intensities.shape[0], 2):
        msg = "scan_positions must have shape (num_positions, 2)."
        raise ValueError(msg)
    if num_iters < 0:
        msg = "num_iters must be non-negative."
        raise ValueError(msg)
    if beta_probe < 0 or beta_object < 0:
        msg = "beta_probe and beta_object must be non-negative."
        raise ValueError(msg)
    if not np.all(np.isfinite(intensities)) or np.any(intensities < 0):
        msg = "I_stack must contain finite, non-negative intensities."
        raise ValueError(msg)
    if not np.all(np.isfinite(positions)):
        msg = "scan_positions must contain finite values."
        raise ValueError(msg)
    if not update_probe and init_probe is None:
        msg = "Known-probe reconstruction requires init_probe."
        raise ValueError(msg)
    if object_amplitude_bounds is not None:
        _apply_amplitude_bounds(
            np.ones((1, 1), dtype=np.complex128), object_amplitude_bounds
        )
    if probe_l2_norm_target is not None and (
        not np.isfinite(probe_l2_norm_target) or probe_l2_norm_target <= 0
    ):
        msg = "probe_l2_norm_target must be finite and positive."
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
    if probe_l2_norm_target is not None:
        probe_norm = _field_l2_norm(probe)
        if probe_norm <= np.finfo(float).eps:
            msg = "Cannot normalize a zero-energy initial probe."
            raise ValueError(msg)
        probe *= probe_l2_norm_target / probe_norm
    if probe_constraint is not None:
        constrained = np.asarray(probe_constraint(probe.copy()), dtype=np.complex128)
        if constrained.shape != shape or not np.all(np.isfinite(constrained)):
            msg = "probe_constraint must return a finite field matching probe shape."
            raise ValueError(msg)
        probe = constrained
        if probe_l2_norm_target is not None:
            probe_norm = _field_l2_norm(probe)
            if probe_norm <= np.finfo(float).eps:
                msg = "probe_constraint returned a zero-energy field."
                raise ValueError(msg)
            probe *= probe_l2_norm_target / probe_norm

    measured_amp = np.sqrt(np.maximum(intensities, 0.0))
    initial_data_fidelity_loss = _evaluate_data_fidelity(
        probe, obj, measured_amp, positions, dx, wavelength, z_BC
    )
    loss_curve: list[float] = []
    iterator = (
        trange(num_iters, desc="ePIE", leave=False, disable=not show_progress)
        if num_iters > 0
        else []
    )
    rng = np.random.default_rng(seed)
    eps = np.finfo(float).eps
    for _ in iterator:
        accum_loss = 0.0
        order = (
            rng.permutation(len(positions))
            if shuffle_positions
            else np.arange(len(positions))
        )
        for idx in order:
            position_xy = positions[idx]
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

            if update_probe:
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

            accum_loss += _relative_amplitude_error(pred_amp, measured_amp[idx])
        if object_amplitude_bounds is not None:
            obj = _apply_amplitude_bounds(obj, object_amplitude_bounds)
        if probe_l2_norm_target is not None:
            probe_norm = _field_l2_norm(probe)
            if probe_norm <= eps:
                msg = "Probe update produced a zero-energy field."
                raise FloatingPointError(msg)
            probe *= probe_l2_norm_target / probe_norm
        if probe_constraint is not None:
            constrained = np.asarray(
                probe_constraint(probe.copy()), dtype=np.complex128
            )
            if constrained.shape != shape or not np.all(np.isfinite(constrained)):
                msg = (
                    "probe_constraint must return a finite field matching probe shape."
                )
                raise ValueError(msg)
            probe = constrained
            if probe_l2_norm_target is not None:
                probe_norm = _field_l2_norm(probe)
                if probe_norm <= eps:
                    msg = "probe_constraint returned a zero-energy field."
                    raise FloatingPointError(msg)
                probe *= probe_l2_norm_target / probe_norm
        loss_curve.append(accum_loss / len(positions))

    final_data_fidelity_loss = _evaluate_data_fidelity(
        probe, obj, measured_amp, positions, dx, wavelength, z_BC
    )
    illumination_map = np.zeros(shape, dtype=np.float64)
    for position_xy in positions:
        illumination_map += np.real(_unshift_delta(np.abs(probe) ** 2, position_xy, dx))

    return {
        "P_B_rec": probe,
        "B_rec": obj,
        "loss_curve": np.asarray(loss_curve, dtype=np.float64),
        "initial_data_fidelity_loss": initial_data_fidelity_loss,
        "final_data_fidelity_loss": final_data_fidelity_loss,
        "illumination_map": illumination_map,
        "metadata": {
            "algorithm": "epie_full_field_integer_shift",
            "known_probe": not update_probe,
            "update_probe": update_probe,
            "shuffle_positions": shuffle_positions,
            "seed": seed,
            "beta_probe": beta_probe,
            "beta_object": beta_object,
            "object_amplitude_bounds": object_amplitude_bounds,
            "probe_l2_norm_target": probe_l2_norm_target,
            "probe_constraint": (
                getattr(probe_constraint, "__name__", type(probe_constraint).__name__)
                if probe_constraint is not None
                else None
            ),
            "integer_pixel_shifts_only": True,
            "num_iters": num_iters,
            "todo": (
                "Add finite patches, subpixel shifts, detector masks, "
                "and position refinement."
            ),
        },
    }
