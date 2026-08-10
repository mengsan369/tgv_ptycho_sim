"""Extended ptychographic iterative engine (ePIE) reconstruction."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from tgv_ptycho.forward.integer_shift import (
    shift_field_integer_pixels,
    unshift_field_delta_integer_pixels,
)
from tgv_ptycho.optics.angular_spectrum import (
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)


def _shift_object(
    obj: NDArray[np.complexfloating],
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
    boundary: str = "periodic",
    boundary_value: complex = 1.0 + 0.0j,
) -> NDArray[np.complex128]:
    return shift_field_integer_pixels(
        obj,
        position_xy,
        dx,
        boundary=boundary,
        fill_value=boundary_value,
    )


def _unshift_delta(
    delta_shifted: NDArray[np.complexfloating],
    position_xy: NDArray[np.floating],
    dx: float | tuple[float, float],
    boundary: str = "periodic",
) -> NDArray[np.complex128]:
    return unshift_field_delta_integer_pixels(
        delta_shifted, position_xy, dx, boundary=boundary
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


def _constraint_name(
    probe_constraint: Callable[
        [NDArray[np.complex128]], NDArray[np.complex128]
    ]
    | None,
) -> str | None:
    if probe_constraint is None:
        return None
    return getattr(probe_constraint, "__name__", type(probe_constraint).__name__)


def _problem_signature(
    intensities: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    dx: float | tuple[float, float],
    wavelength: float,
    z_bc: float,
    beta_probe: float,
    beta_object: float,
    update_probe: bool,
    update_object: bool,
    shuffle_positions: bool,
    seed: int | None,
    object_amplitude_bounds: tuple[float, float] | None,
    probe_l2_norm_target: float | None,
    correction_mode: str,
    denominator_mode: str,
    rpie_alpha_probe: float,
    rpie_alpha_object: float,
    object_boundary: str,
    object_boundary_value: complex,
    probe_constraint_name: str | None,
) -> str:
    """Hash the numerical problem so an incompatible state cannot be resumed."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(intensities).view(np.uint8))
    digest.update(np.ascontiguousarray(positions).view(np.uint8))
    payload = {
        "dx": list(dx) if isinstance(dx, tuple) else float(dx),
        "wavelength": float(wavelength),
        "z_bc": float(z_bc),
        "beta_probe": float(beta_probe),
        "beta_object": float(beta_object),
        "update_probe": bool(update_probe),
        "update_object": bool(update_object),
        "shuffle_positions": bool(shuffle_positions),
        "seed": seed,
        "object_amplitude_bounds": object_amplitude_bounds,
        "probe_l2_norm_target": probe_l2_norm_target,
        "correction_mode": correction_mode,
        "denominator_mode": denominator_mode,
        "rpie_alpha_probe": float(rpie_alpha_probe),
        "rpie_alpha_object": float(rpie_alpha_object),
        "object_boundary": object_boundary,
        "object_boundary_value": [
            float(complex(object_boundary_value).real),
            float(complex(object_boundary_value).imag),
        ],
        "probe_constraint": probe_constraint_name,
    }
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _optimizer_state(
    probe: NDArray[np.complex128],
    obj: NDArray[np.complex128],
    loss_curve: list[float],
    initial_data_fidelity_loss: float,
    completed_iterations: int,
    rng: np.random.Generator,
    problem_signature: str,
) -> dict[str, Any]:
    """Return an isolated state sufficient for an exact shuffled continuation."""

    return {
        "version": 1,
        "completed_iterations": int(completed_iterations),
        "P_B_rec": probe.copy(),
        "B_rec": obj.copy(),
        "loss_curve": np.asarray(loss_curve, dtype=np.float64).copy(),
        "initial_data_fidelity_loss": float(initial_data_fidelity_loss),
        "rng_bit_generator": type(rng.bit_generator).__name__,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "problem_signature": problem_signature,
    }


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
    forward_transfer: NDArray[np.complex128],
    object_boundary: str,
    object_boundary_value: complex,
) -> float:
    """Evaluate mean relative amplitude error with frozen estimates."""

    total = 0.0
    for index, position_xy in enumerate(positions):
        shifted_object = _shift_object(
            obj,
            position_xy,
            dx,
            boundary=object_boundary,
            boundary_value=object_boundary_value,
        )
        detector_field = apply_angular_spectrum_transfer(
            probe * shifted_object, forward_transfer
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
    update_object: bool = True,
    shuffle_positions: bool = True,
    seed: int | None = None,
    object_amplitude_bounds: tuple[float, float] | None = None,
    probe_l2_norm_target: float | None = None,
    probe_constraint: Callable[
        [NDArray[np.complex128]], NDArray[np.complex128]
    ]
    | None = None,
    correction_mode: Literal[
        "adjoint_residual", "legacy_inverse_difference"
    ] = "adjoint_residual",
    denominator_mode: Literal["epie", "rpie"] = "epie",
    rpie_alpha_probe: float = 0.1,
    rpie_alpha_object: float = 0.1,
    object_boundary: Literal["periodic", "constant"] = "periodic",
    object_boundary_value: complex = 1.0 + 0.0j,
    checkpoint_iters: tuple[int, ...] | None = None,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run a compact full-field ePIE reconstruction.

    Algorithm sketch:
    1. For each scan position, shift the current object estimate.
    2. Form the exit wave `psi = probe * object_j`.
    3. Forward propagate to the detector.
    4. Replace predicted amplitude with measured amplitude.
    5. Apply the exact propagation adjoint to the detector-plane residual.
    6. Update object and, when requested, probe using normalized ePIE steps.
    7. Record a relative amplitude loss.

    Set ``update_probe=False`` for known-probe PIE and ``update_object=False``
    for known-object/probe-only PIE.  The default ``adjoint_residual`` update is
    required when band-limited propagation is not unitary.  The legacy update
    is retained only for explicitly labelled regression diagnostics.

    Scan positions are ``(x, y)`` in meters.  ``object_boundary="periodic"``
    preserves the historical full-field ``np.roll`` model; ``"constant"`` is
    available only for a forward/reconstruction-matched finite-FOV control.

    ``num_iters`` is the final cumulative iteration count.  A ``resume_state``
    returned by an earlier call resumes the exact shuffled trajectory: the
    saved RNG state and loss history are continued, and the initial probe
    normalization/constraint is not applied a second time.  Checkpoint numbers
    are absolute cumulative iterations.  A callback is invoked only after a
    complete iteration and all configured projections have finished.
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
    if correction_mode not in {"adjoint_residual", "legacy_inverse_difference"}:
        msg = "Unknown correction_mode."
        raise ValueError(msg)
    if denominator_mode not in {"epie", "rpie"}:
        msg = "denominator_mode must be 'epie' or 'rpie'."
        raise ValueError(msg)
    if not 0.0 <= rpie_alpha_probe <= 1.0:
        msg = "rpie_alpha_probe must lie in [0, 1]."
        raise ValueError(msg)
    if not 0.0 <= rpie_alpha_object <= 1.0:
        msg = "rpie_alpha_object must lie in [0, 1]."
        raise ValueError(msg)
    if object_boundary not in {"periodic", "constant"}:
        msg = "object_boundary must be 'periodic' or 'constant'."
        raise ValueError(msg)
    if not np.all(np.isfinite(intensities)) or np.any(intensities < 0):
        msg = "I_stack must contain finite, non-negative intensities."
        raise ValueError(msg)
    if not np.all(np.isfinite(positions)):
        msg = "scan_positions must contain finite values."
        raise ValueError(msg)
    if resume_state is not None and (init_probe is not None or init_object is not None):
        msg = "init_probe/init_object must be omitted when resume_state is provided."
        raise ValueError(msg)
    if not update_probe and init_probe is None and resume_state is None:
        msg = "Known-probe reconstruction requires init_probe."
        raise ValueError(msg)
    if not update_object and init_object is None and resume_state is None:
        msg = "Known-object reconstruction requires init_object."
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
    checkpoints_requested = tuple(sorted(set(checkpoint_iters or ())))
    if any(value < 0 or value > num_iters for value in checkpoints_requested):
        msg = "checkpoint_iters entries must lie in [0, num_iters]."
        raise ValueError(msg)

    shape = intensities.shape[1:]
    probe_constraint_name = _constraint_name(probe_constraint)
    problem_signature = _problem_signature(
        intensities,
        positions,
        dx=dx,
        wavelength=wavelength,
        z_bc=z_BC,
        beta_probe=beta_probe,
        beta_object=beta_object,
        update_probe=update_probe,
        update_object=update_object,
        shuffle_positions=shuffle_positions,
        seed=seed,
        object_amplitude_bounds=object_amplitude_bounds,
        probe_l2_norm_target=probe_l2_norm_target,
        correction_mode=correction_mode,
        denominator_mode=denominator_mode,
        rpie_alpha_probe=rpie_alpha_probe,
        rpie_alpha_object=rpie_alpha_object,
        object_boundary=object_boundary,
        object_boundary_value=object_boundary_value,
        probe_constraint_name=probe_constraint_name,
    )
    resumed_from_iteration = 0
    if resume_state is None:
        if init_probe is None:
            mean_amp = np.sqrt(np.maximum(intensities.mean(axis=0), 0.0))
            probe = mean_amp.astype(np.complex128)
        else:
            probe = np.asarray(init_probe, dtype=np.complex128).copy()
        if init_object is None:
            obj = np.ones(shape, dtype=np.complex128)
        else:
            obj = np.asarray(init_object, dtype=np.complex128).copy()
        loss_curve: list[float] = []
    else:
        required_keys = {
            "version",
            "completed_iterations",
            "P_B_rec",
            "B_rec",
            "loss_curve",
            "initial_data_fidelity_loss",
            "rng_bit_generator",
            "rng_state",
            "problem_signature",
        }
        missing = sorted(required_keys.difference(resume_state))
        if missing:
            msg = f"resume_state is missing keys: {missing}."
            raise ValueError(msg)
        if int(resume_state["version"]) != 1:
            msg = "Unsupported resume_state version."
            raise ValueError(msg)
        resumed_from_iteration = int(resume_state["completed_iterations"])
        if resumed_from_iteration < 0 or resumed_from_iteration > num_iters:
            msg = "resume_state completed_iterations must lie in [0, num_iters]."
            raise ValueError(msg)
        if str(resume_state["problem_signature"]) != problem_signature:
            msg = "resume_state does not match this reconstruction problem."
            raise ValueError(msg)
        probe = np.asarray(resume_state["P_B_rec"], dtype=np.complex128).copy()
        obj = np.asarray(resume_state["B_rec"], dtype=np.complex128).copy()
        resumed_loss = np.asarray(resume_state["loss_curve"], dtype=np.float64)
        if resumed_loss.shape != (resumed_from_iteration,):
            msg = "resume_state loss_curve length must equal completed_iterations."
            raise ValueError(msg)
        if not np.all(np.isfinite(resumed_loss)):
            msg = "resume_state loss_curve must contain finite values."
            raise ValueError(msg)
        loss_curve = resumed_loss.tolist()
    if probe.shape != shape or obj.shape != shape:
        msg = "Probe and object state must match detector frame shape."
        raise ValueError(msg)
    if not np.all(np.isfinite(probe)) or not np.all(np.isfinite(obj)):
        msg = "Probe and object state must contain finite values."
        raise ValueError(msg)
    if resume_state is None and probe_l2_norm_target is not None:
        probe_norm = _field_l2_norm(probe)
        if probe_norm <= np.finfo(float).eps:
            msg = "Cannot normalize a zero-energy initial probe."
            raise ValueError(msg)
        probe *= probe_l2_norm_target / probe_norm
    if resume_state is None and probe_constraint is not None:
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
    forward_transfer = make_angular_spectrum_transfer(
        shape, dx, wavelength, z_BC, bandlimit=True
    )
    adjoint_transfer = np.conj(forward_transfer)
    current_data_fidelity_loss = _evaluate_data_fidelity(
        probe,
        obj,
        measured_amp,
        positions,
        dx,
        forward_transfer,
        object_boundary,
        object_boundary_value,
    )
    if resume_state is None:
        initial_data_fidelity_loss = current_data_fidelity_loss
    else:
        initial_data_fidelity_loss = float(
            resume_state["initial_data_fidelity_loss"]
        )
        if not np.isfinite(initial_data_fidelity_loss):
            msg = "resume_state initial_data_fidelity_loss must be finite."
            raise ValueError(msg)
    rng = np.random.default_rng(seed)
    if resume_state is not None:
        if str(resume_state["rng_bit_generator"]) != type(rng.bit_generator).__name__:
            msg = "resume_state RNG bit generator is incompatible."
            raise ValueError(msg)
        try:
            rng.bit_generator.state = copy.deepcopy(resume_state["rng_state"])
        except (TypeError, ValueError) as exc:
            msg = "resume_state contains an invalid RNG state."
            raise ValueError(msg) from exc
    checkpoints: dict[str, dict[str, Any]] = {}
    if resumed_from_iteration in checkpoints_requested:
        state = _optimizer_state(
            probe,
            obj,
            loss_curve,
            initial_data_fidelity_loss,
            resumed_from_iteration,
            rng,
            problem_signature,
        )
        checkpoint = {
            "P_B_rec": probe.copy(),
            "B_rec": obj.copy(),
            "data_fidelity_loss": current_data_fidelity_loss,
            "optimizer_state": state,
        }
        checkpoints[str(resumed_from_iteration)] = checkpoint
        if checkpoint_callback is not None:
            checkpoint_callback(copy.deepcopy(checkpoint))
    iterator = (
        trange(
            resumed_from_iteration,
            num_iters,
            desc="ePIE",
            leave=False,
            disable=not show_progress,
        )
        if num_iters > resumed_from_iteration
        else []
    )
    eps = np.finfo(float).eps
    for iteration_index in iterator:
        accum_loss = 0.0
        order = (
            rng.permutation(len(positions))
            if shuffle_positions
            else np.arange(len(positions))
        )
        for idx in order:
            position_xy = positions[idx]
            obj_shifted = _shift_object(
                obj,
                position_xy,
                dx,
                boundary=object_boundary,
                boundary_value=object_boundary_value,
            )
            probe_old = probe.copy()
            exit_wave = probe * obj_shifted
            U_det = apply_angular_spectrum_transfer(exit_wave, forward_transfer)
            pred_amp = np.abs(U_det)
            corrected_det = measured_amp[idx] * np.exp(1j * np.angle(U_det))
            if correction_mode == "adjoint_residual":
                delta = apply_angular_spectrum_transfer(
                    corrected_det - U_det, adjoint_transfer
                )
            else:
                corrected_exit = apply_angular_spectrum_transfer(
                    corrected_det, adjoint_transfer
                )
                delta = corrected_exit - exit_wave

            if update_probe:
                object_intensity = np.abs(obj_shifted) ** 2
                if denominator_mode == "rpie":
                    probe_denominator = (
                        (1.0 - rpie_alpha_probe) * object_intensity
                        + rpie_alpha_probe * np.max(object_intensity)
                        + eps
                    )
                else:
                    probe_denominator = np.max(object_intensity) + eps
                probe += (
                    beta_probe
                    * np.conj(obj_shifted)
                    / probe_denominator
                    * delta
                )

            if update_object:
                probe_intensity = np.abs(probe_old) ** 2
                if denominator_mode == "rpie":
                    object_denominator = (
                        (1.0 - rpie_alpha_object) * probe_intensity
                        + rpie_alpha_object * np.max(probe_intensity)
                        + eps
                    )
                else:
                    object_denominator = np.max(probe_intensity) + eps
                obj_delta_shifted = (
                    beta_object
                    * np.conj(probe_old)
                    / object_denominator
                    * delta
                )
                obj += _unshift_delta(
                    obj_delta_shifted,
                    position_xy,
                    dx,
                    boundary=object_boundary,
                )

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
        completed_iteration = int(iteration_index) + 1
        if completed_iteration in checkpoints_requested:
            checkpoint_loss = _evaluate_data_fidelity(
                probe,
                obj,
                measured_amp,
                positions,
                dx,
                forward_transfer,
                object_boundary,
                object_boundary_value,
            )
            state = _optimizer_state(
                probe,
                obj,
                loss_curve,
                initial_data_fidelity_loss,
                completed_iteration,
                rng,
                problem_signature,
            )
            checkpoint = {
                "P_B_rec": probe.copy(),
                "B_rec": obj.copy(),
                "data_fidelity_loss": checkpoint_loss,
                "optimizer_state": state,
            }
            checkpoints[str(completed_iteration)] = checkpoint
            if checkpoint_callback is not None:
                checkpoint_callback(copy.deepcopy(checkpoint))

    final_data_fidelity_loss = _evaluate_data_fidelity(
        probe,
        obj,
        measured_amp,
        positions,
        dx,
        forward_transfer,
        object_boundary,
        object_boundary_value,
    )
    illumination_map = np.zeros(shape, dtype=np.float64)
    for position_xy in positions:
        illumination_map += np.real(
            _unshift_delta(
                np.abs(probe) ** 2,
                position_xy,
                dx,
                boundary=object_boundary,
            )
        )

    completed_iterations = num_iters
    final_optimizer_state = _optimizer_state(
        probe,
        obj,
        loss_curve,
        initial_data_fidelity_loss,
        completed_iterations,
        rng,
        problem_signature,
    )

    return {
        "P_B_rec": probe,
        "B_rec": obj,
        "loss_curve": np.asarray(loss_curve, dtype=np.float64),
        "initial_data_fidelity_loss": initial_data_fidelity_loss,
        "final_data_fidelity_loss": final_data_fidelity_loss,
        "illumination_map": illumination_map,
        "checkpoints": checkpoints,
        "optimizer_state": final_optimizer_state,
        "completed_iterations": completed_iterations,
        "metadata": {
            "algorithm": "epie_adjoint_residual_integer_shift",
            "known_probe": not update_probe,
            "known_object": not update_object,
            "update_probe": update_probe,
            "update_object": update_object,
            "shuffle_positions": shuffle_positions,
            "seed": seed,
            "beta_probe": beta_probe,
            "beta_object": beta_object,
            "object_amplitude_bounds": object_amplitude_bounds,
            "probe_l2_norm_target": probe_l2_norm_target,
            "correction_mode": correction_mode,
            "denominator_mode": denominator_mode,
            "rpie_alpha_probe": rpie_alpha_probe,
            "rpie_alpha_object": rpie_alpha_object,
            "object_boundary": object_boundary,
            "object_boundary_value": object_boundary_value,
            "checkpoint_iters": checkpoints_requested,
            "resumed_from_iteration": resumed_from_iteration,
            "resume_state_version": 1,
            "problem_signature": problem_signature,
            "cached_forward_and_adjoint_transfer": True,
            "probe_constraint": (
                probe_constraint_name
            ),
            "integer_pixel_shifts_only": True,
            "num_iters": num_iters,
            "todo": (
                "Add finite patches, subpixel shifts, detector masks, "
                "and position refinement."
            ),
        },
    }
