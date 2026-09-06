"""Matched known-B/probe-only reconstruction for exp031 open propagation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.exp031 import (
    adjoint_detector_roi_to_active,
    propagate_probe_patch_batch_to_roi,
)

PatchGetter = Callable[[int], NDArray[np.complexfloating]]


def _norm(values: NDArray[np.generic]) -> float:
    return float(np.sqrt(np.sum(np.abs(values) ** 2, dtype=np.float64)))


def _inner(
    first: NDArray[np.complexfloating], second: NDArray[np.complexfloating]
) -> complex:
    return complex(np.sum(np.conj(first) * second, dtype=np.complex128))


def measurement_scaled_reference_initialization(
    reference_b: NDArray[np.complexfloating],
    reference_detector: NDArray[np.complexfloating],
    intensity_stack: NDArray[np.floating],
) -> NDArray[np.complex64]:
    """Scale the analytic reference using detector energy only."""

    reference = np.asarray(reference_b, dtype=np.complex64)
    detector_reference = np.asarray(reference_detector, dtype=np.complex64)
    measured = np.asarray(intensity_stack, dtype=np.float64)
    if measured.ndim != 3 or measured.shape[-2:] != detector_reference.shape:
        raise ValueError("intensity_stack and detector reference are inconsistent.")
    measured_energy = float(np.mean(np.sum(measured, axis=(1, 2))))
    reference_energy = float(np.sum(np.abs(detector_reference) ** 2))
    scale = np.sqrt(measured_energy / max(reference_energy, np.finfo(float).tiny))
    return np.asarray(scale * reference, dtype=np.complex64)


def reconstruct_known_b_probe_open(
    intensity_stack: NDArray[np.floating],
    patch_getter: PatchGetter,
    reference_b: NDArray[np.complexfloating],
    reference_detector: NDArray[np.complexfloating],
    transfer: NDArray[np.complexfloating],
    init_probe: NDArray[np.complexfloating],
    *,
    num_iterations: int,
    beta_probe: float,
    workers: int = -1,
) -> dict[str, Any]:
    """Run a fixed-schedule known-B probe update with the matched open model.

    The known B patches are supplied lazily, so the enlarged canvas and all
    active patches need not be materialized together.  Selection/stopping uses
    the configured iteration count and detector-amplitude loss only.
    """

    measured = np.asarray(intensity_stack, dtype=np.float64)
    reference = np.asarray(reference_b, dtype=np.complex64)
    detector_reference = np.asarray(reference_detector, dtype=np.complex64)
    probe = np.asarray(init_probe, dtype=np.complex64).copy()
    kernel = np.asarray(transfer, dtype=np.complex64)
    if measured.ndim != 3 or measured.shape[-2:] != detector_reference.shape:
        raise ValueError("intensity_stack and detector reference are inconsistent.")
    if probe.shape != reference.shape:
        raise ValueError("init_probe and reference_b must have the same shape.")
    if num_iterations < 0 or not np.isfinite(beta_probe) or beta_probe < 0:
        raise ValueError("iteration count and beta_probe must be non-negative.")
    target_amplitude = np.sqrt(np.maximum(measured, 0.0))
    loss_curve: list[float] = []
    epsilon = np.finfo(np.float32).eps
    for _ in range(int(num_iterations)):
        accumulated = 0.0
        for index in range(measured.shape[0]):
            patch = np.asarray(patch_getter(index), dtype=np.complex64)
            if patch.shape != reference.shape:
                raise ValueError("patch_getter returned an inconsistent shape.")
            predicted = propagate_probe_patch_batch_to_roi(
                probe,
                reference,
                patch,
                detector_reference,
                kernel,
                workers=workers,
            )[0]
            predicted_amplitude = np.abs(predicted)
            denominator = max(_norm(target_amplitude[index]), epsilon)
            accumulated += float(
                _norm(predicted_amplitude - target_amplitude[index]) / denominator
            )
            corrected = target_amplitude[index] * np.exp(1j * np.angle(predicted))
            detector_delta = corrected - predicted
            active_delta = adjoint_detector_roi_to_active(
                detector_delta,
                kernel,
                reference.shape,
                workers=workers,
            )[0]
            object_denominator = max(float(np.max(np.abs(patch) ** 2)), epsilon)
            probe += (
                float(beta_probe) * np.conj(patch) / object_denominator * active_delta
            )
        loss_curve.append(accumulated / measured.shape[0])
    if not np.all(np.isfinite(probe)):
        raise FloatingPointError("known-B reconstruction produced nonfinite probe.")
    return {
        "P_B_rec_raw": probe,
        "loss_curve": np.asarray(loss_curve, dtype=np.float64),
        "metadata": {
            "algorithm": "matched_open_known_B_probe_only",
            "num_iterations": int(num_iterations),
            "beta_probe": float(beta_probe),
            "selection": "fixed_measurement_only_schedule",
            "truth_used_by_optimizer": False,
            "patches_materialized_together": False,
        },
    }


def simulation_evaluation_probe_error(
    reconstructed: NDArray[np.complexfloating],
    truth: NDArray[np.complexfloating],
) -> dict[str, float]:
    """Return raw and complex-scalar-aligned errors for simulation evaluation."""

    estimate = np.asarray(reconstructed, dtype=np.complex128)
    reference = np.asarray(truth, dtype=np.complex128)
    if estimate.shape != reference.shape:
        raise ValueError("reconstructed and truth probes must share a shape.")
    truth_norm = max(_norm(reference), np.finfo(float).tiny)
    raw = float(_norm(estimate - reference) / truth_norm)
    denominator = _inner(estimate, estimate)
    scale = _inner(estimate, reference) / max(abs(denominator), np.finfo(float).tiny)
    aligned = scale * estimate
    return {
        "raw_relative_l2": raw,
        "complex_scalar_aligned_relative_l2_simulation_evaluation_only": float(
            _norm(aligned - reference) / truth_norm
        ),
        "alignment_scale_abs_simulation_evaluation_only": float(abs(scale)),
        "alignment_phase_rad_simulation_evaluation_only": float(np.angle(scale)),
    }


__all__ = [
    "measurement_scaled_reference_initialization",
    "reconstruct_known_b_probe_open",
    "simulation_evaluation_probe_error",
]
