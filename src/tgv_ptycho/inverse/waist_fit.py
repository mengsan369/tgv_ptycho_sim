"""Reusable one-parameter fitting controls for TGV waist signatures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating]


def raw_complex_probe_loss(
    candidate: ComplexArray,
    target: ComplexArray,
) -> float:
    """Return the unaligned normalized squared complex-field residual."""

    candidate_values = np.asarray(candidate)
    target_values = np.asarray(target)
    if candidate_values.shape != target_values.shape:
        raise ValueError("candidate and target must have the same shape.")
    if candidate_values.ndim != 2 or min(candidate_values.shape) <= 0:
        raise ValueError("candidate and target must be nonempty 2D fields.")
    if not np.iscomplexobj(candidate_values) or not np.iscomplexobj(target_values):
        raise ValueError("candidate and target must be complex fields.")
    if not np.all(np.isfinite(candidate_values)) or not np.all(
        np.isfinite(target_values)
    ):
        raise ValueError("candidate and target must contain only finite values.")
    denominator = float(np.sum(np.abs(target_values) ** 2, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("target must have positive finite complex-field energy.")
    numerator = float(
        np.sum(np.abs(candidate_values - target_values) ** 2, dtype=np.float64)
    )
    return numerator / denominator


def replay_error_metrics(
    candidate: ComplexArray,
    target: ComplexArray,
) -> dict[str, float]:
    """Return preregistered raw, amplitude, and phase-sensitive replay errors."""

    candidate_values = np.asarray(candidate, dtype=np.complex128)
    target_values = np.asarray(target, dtype=np.complex128)
    raw_relative = float(
        np.sqrt(raw_complex_probe_loss(candidate_values, target_values))
    )
    target_norm = float(
        np.sqrt(np.sum(np.abs(target_values) ** 2, dtype=np.float64))
    )
    amplitude_relative = float(
        np.sqrt(
            np.sum(
                (np.abs(candidate_values) - np.abs(target_values)) ** 2,
                dtype=np.float64,
            )
        )
        / target_norm
    )
    phase_residual = np.abs(target_values) * (
        np.exp(1j * np.angle(candidate_values))
        - np.exp(1j * np.angle(target_values))
    )
    phase_sensitive_relative = float(
        np.sqrt(np.sum(np.abs(phase_residual) ** 2, dtype=np.float64))
        / target_norm
    )
    return {
        "raw_complex_relative_l2": raw_relative,
        "amplitude_relative_l2": amplitude_relative,
        "amplitude_weighted_phase_sensitive_relative_l2": (
            phase_sensitive_relative
        ),
    }


def extract_profile_minimum(
    diameters_m: Sequence[float] | NDArray[np.floating],
    losses: Sequence[float] | NDArray[np.floating],
    *,
    uniqueness_tolerance: float,
) -> dict[str, Any]:
    """Extract a deterministic minimum and preregistered uniqueness evidence."""

    diameter_values = np.asarray(diameters_m, dtype=np.float64)
    loss_values = np.asarray(losses, dtype=np.float64)
    if (
        diameter_values.ndim != 1
        or loss_values.shape != diameter_values.shape
        or len(diameter_values) < 2
    ):
        raise ValueError("profile arrays must be same-shaped 1D arrays.")
    if not np.all(np.isfinite(diameter_values)) or not np.all(
        np.isfinite(loss_values)
    ):
        raise ValueError("profile arrays must contain only finite values.")
    if np.any(np.diff(diameter_values) <= 0.0) or np.any(loss_values < 0.0):
        raise ValueError("diameters must increase and losses must be nonnegative.")
    if not np.isfinite(uniqueness_tolerance) or uniqueness_tolerance < 0.0:
        raise ValueError("uniqueness_tolerance must be finite and nonnegative.")
    minimum_index = int(np.argmin(loss_values))
    minimum_loss = float(loss_values[minimum_index])
    tied_indices = np.flatnonzero(
        loss_values <= minimum_loss + uniqueness_tolerance
    )
    order = np.argsort(loss_values, kind="stable")
    second_index = int(order[1])
    return {
        "minimum_index": minimum_index,
        "minimum_d_waist_m": float(diameter_values[minimum_index]),
        "minimum_loss": minimum_loss,
        "second_best_index": second_index,
        "second_best_d_waist_m": float(diameter_values[second_index]),
        "second_best_loss": float(loss_values[second_index]),
        "unique": bool(len(tied_indices) == 1),
        "indices_within_uniqueness_tolerance": tied_indices.astype(np.int64),
    }


def finite_difference_controls(
    target: ComplexArray,
    center: ComplexArray,
    minus_h1: ComplexArray,
    plus_h1: ComplexArray,
    minus_h2: ComplexArray,
    plus_h2: ComplexArray,
    *,
    h1_m: float,
    h2_m: float,
) -> dict[str, Any]:
    """Return central-Jacobian, curvature, and step-convergence controls."""

    if not np.isfinite(h1_m) or not np.isfinite(h2_m):
        raise ValueError("finite-difference steps must be finite.")
    if h1_m <= h2_m or h2_m <= 0.0:
        raise ValueError("finite-difference steps must satisfy h1 > h2 > 0.")
    target_values = np.asarray(target, dtype=np.complex128)
    fields = [
        np.asarray(values, dtype=np.complex128)
        for values in (center, minus_h1, plus_h1, minus_h2, plus_h2)
    ]
    if any(values.shape != target_values.shape for values in fields):
        raise ValueError("all finite-difference fields must match target shape.")
    if any(not np.all(np.isfinite(values)) for values in fields):
        raise ValueError("finite-difference fields must contain finite values.")
    center_values, minus1, plus1, minus2, plus2 = fields
    jacobian_h1 = (plus1 - minus1) / (2.0 * h1_m)
    jacobian_h2 = (plus2 - minus2) / (2.0 * h2_m)
    target_norm = float(np.sqrt(np.sum(np.abs(target_values) ** 2)))
    jacobian_h1_norm = float(np.sqrt(np.sum(np.abs(jacobian_h1) ** 2)))
    jacobian_h2_norm = float(np.sqrt(np.sum(np.abs(jacobian_h2) ** 2)))
    convergence = float(
        np.sqrt(np.sum(np.abs(jacobian_h1 - jacobian_h2) ** 2))
        / max(jacobian_h2_norm, np.finfo(np.float64).eps)
    )
    center_loss = raw_complex_probe_loss(center_values, target_values)
    loss_minus_h1 = raw_complex_probe_loss(minus1, target_values)
    loss_plus_h1 = raw_complex_probe_loss(plus1, target_values)
    curvature = (loss_plus_h1 - 2.0 * center_loss + loss_minus_h1) / (
        h1_m**2
    )
    return {
        "h1_m": float(h1_m),
        "h2_m": float(h2_m),
        "jacobian_h1": np.asarray(jacobian_h1, dtype=np.complex128),
        "jacobian_h2": np.asarray(jacobian_h2, dtype=np.complex128),
        "normalized_jacobian_h1_per_m": jacobian_h1_norm / target_norm,
        "normalized_jacobian_h2_per_m": jacobian_h2_norm / target_norm,
        "jacobian_step_relative_l2": convergence,
        "loss_curvature_per_m2": float(curvature),
        "center_loss": float(center_loss),
        "minus_h1_loss": float(loss_minus_h1),
        "plus_h1_loss": float(loss_plus_h1),
        "minus_h2_loss": raw_complex_probe_loss(minus2, target_values),
        "plus_h2_loss": raw_complex_probe_loss(plus2, target_values),
    }


def equal_budget_bounded_pattern_search(
    objective: Callable[[float], tuple[float, int]],
    *,
    bounds_m: tuple[float, float],
    start_m: float,
    initial_step_m: float,
    evaluation_budget: int,
) -> dict[str, Any]:
    """Run the fixed-budget bounded one-dimensional pattern search."""

    lower, upper = (float(value) for value in bounds_m)
    if not np.all(np.isfinite([lower, upper, start_m, initial_step_m])):
        raise ValueError("bounds, start, and initial step must be finite.")
    if lower >= upper or not lower <= start_m <= upper:
        raise ValueError("start must lie inside increasing bounds.")
    if initial_step_m <= 0.0:
        raise ValueError("initial_step_m must be positive.")
    if evaluation_budget < 3 or evaluation_budget % 2 != 1:
        raise ValueError("evaluation_budget must be an odd integer >= 3.")

    evaluated_d: list[float] = []
    evaluated_loss: list[float] = []
    cache_indices: list[int] = []

    def evaluate(diameter_m: float) -> tuple[float, int]:
        loss, cache_index = objective(float(diameter_m))
        if not np.isfinite(loss) or loss < 0.0:
            raise ValueError("objective must return a finite nonnegative loss.")
        evaluated_d.append(float(diameter_m))
        evaluated_loss.append(float(loss))
        cache_indices.append(int(cache_index))
        return float(loss), int(cache_index)

    incumbent = float(start_m)
    incumbent_loss, _ = evaluate(incumbent)
    step = float(initial_step_m)
    incumbent_track = [incumbent]
    incumbent_loss_track = [incumbent_loss]
    step_track = [step]
    while len(evaluated_d) < evaluation_budget:
        left = float(np.clip(incumbent - step, lower, upper))
        right = float(np.clip(incumbent + step, lower, upper))
        left_loss, _ = evaluate(left)
        right_loss, _ = evaluate(right)
        new_candidates = sorted(
            [(left_loss, left), (right_loss, right)], key=lambda item: item
        )
        best_new_loss, best_new_d = new_candidates[0]
        if best_new_loss < incumbent_loss:
            incumbent = best_new_d
            incumbent_loss = best_new_loss
        else:
            step *= 0.5
        incumbent_track.append(incumbent)
        incumbent_loss_track.append(incumbent_loss)
        step_track.append(step)

    return {
        "start_m": float(start_m),
        "bounds_m": np.asarray([lower, upper], dtype=np.float64),
        "initial_step_m": float(initial_step_m),
        "evaluation_budget": int(evaluation_budget),
        "evaluation_count": len(evaluated_d),
        "evaluated_d_waist_m": np.asarray(evaluated_d, dtype=np.float64),
        "evaluated_loss": np.asarray(evaluated_loss, dtype=np.float64),
        "evaluated_cache_index": np.asarray(cache_indices, dtype=np.int64),
        "incumbent_d_waist_m": np.asarray(
            incumbent_track, dtype=np.float64
        ),
        "incumbent_loss": np.asarray(
            incumbent_loss_track, dtype=np.float64
        ),
        "step_m": np.asarray(step_track, dtype=np.float64),
        "final_estimate_m": float(incumbent),
        "final_loss": float(incumbent_loss),
        "stopping_reason": "evaluation_budget",
    }


@dataclass
class _ProbeCache:
    generator: Callable[[float], ComplexArray]
    target_shape: tuple[int, int]
    values: dict[float, int] = field(default_factory=dict)
    diameters_m: list[float] = field(default_factory=list)
    probes: list[NDArray[np.complex128]] = field(default_factory=list)

    def get(self, diameter_m: float) -> tuple[NDArray[np.complex128], int]:
        key = float(np.float64(diameter_m))
        index = self.values.get(key)
        if index is not None:
            return self.probes[index], index
        generated = np.asarray(self.generator(key))
        if generated.shape != self.target_shape:
            raise ValueError("candidate generator returned the wrong shape.")
        if generated.dtype != np.complex128:
            raise ValueError("candidate generator must return complex128.")
        if not np.all(np.isfinite(generated)):
            raise ValueError("candidate generator returned non-finite values.")
        index = len(self.probes)
        self.values[key] = index
        self.diameters_m.append(key)
        self.probes.append(generated.copy())
        return self.probes[index], index


def _validate_fit_inputs(
    target: NDArray[np.generic],
    *,
    true_waist_m: float,
    bounds_m: tuple[float, float],
    coarse_grid_m: NDArray[np.float64],
    starts_m: NDArray[np.float64],
) -> NDArray[np.complex128]:
    if target.dtype != np.complex128 or target.ndim != 2:
        raise ValueError("target_probe must be a 2D complex128 field.")
    if not np.all(np.isfinite(target)) or float(np.sum(np.abs(target) ** 2)) <= 0:
        raise ValueError("target_probe must be finite with positive energy.")
    lower, upper = bounds_m
    if not np.all(np.isfinite([lower, upper, true_waist_m])) or lower >= upper:
        raise ValueError("bounds and true waist must be finite and ordered.")
    if not lower < true_waist_m < upper:
        raise ValueError("true waist must lie strictly inside bounds.")
    if (
        coarse_grid_m.ndim != 1
        or len(coarse_grid_m) < 3
        or np.any(np.diff(coarse_grid_m) <= 0.0)
        or coarse_grid_m[0] != lower
        or coarse_grid_m[-1] != upper
    ):
        raise ValueError("coarse grid must increase from lower to upper bound.")
    if starts_m.ndim != 1 or len(starts_m) < 2:
        raise ValueError("at least two optimizer starts are required.")
    if np.any(starts_m <= lower) or np.any(starts_m >= upper):
        raise ValueError("optimizer starts must lie strictly inside bounds.")
    return np.asarray(target, dtype=np.complex128)


def fit_waist_from_probe(
    target_probe: ComplexArray,
    candidate_generator: Callable[[float], ComplexArray],
    *,
    true_waist_m: float,
    bounds_m: tuple[float, float],
    coarse_grid_m: Sequence[float],
    fine_half_width_m: float,
    fine_step_m: float,
    finite_difference_steps_m: tuple[float, float],
    optimizer_starts_m: Sequence[float],
    optimizer_initial_step_m: float,
    optimizer_evaluation_budget: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Run the preregistered raw-complex oracle waist fit."""

    target_input = np.asarray(target_probe)
    coarse = np.asarray(coarse_grid_m, dtype=np.float64)
    starts = np.asarray(optimizer_starts_m, dtype=np.float64)
    target = _validate_fit_inputs(
        target_input,
        true_waist_m=true_waist_m,
        bounds_m=bounds_m,
        coarse_grid_m=coarse,
        starts_m=starts,
    )
    required_thresholds = {
        "replay_raw_relative_l2_max",
        "replay_amplitude_relative_l2_max",
        "replay_phase_sensitive_relative_l2_max",
        "deterministic_repeat_relative_l2_max",
        "profile_uniqueness_loss_floor",
        "profile_absolute_error_m_max",
        "profile_minimum_loss_max",
        "profile_second_to_floor_ratio_min",
        "signature_to_floor_ratio_min",
        "normalized_jacobian_per_m_min",
        "jacobian_step_relative_l2_max",
        "optimizer_absolute_error_m_max",
        "optimizer_relative_error_max",
        "optimizer_profile_agreement_m_max",
        "optimizer_spread_m_max",
        "boundary_margin_m",
    }
    missing = sorted(required_thresholds - set(thresholds))
    if missing:
        raise ValueError(f"thresholds are missing keys: {missing}")
    if fine_half_width_m <= 0.0 or fine_step_m <= 0.0:
        raise ValueError("fine grid width and step must be positive.")

    replay = np.asarray(candidate_generator(float(true_waist_m)))
    repeat = np.asarray(candidate_generator(float(true_waist_m)))
    if replay.shape != target.shape or repeat.shape != target.shape:
        raise ValueError("replay generator returned the wrong shape.")
    if replay.dtype != np.complex128 or repeat.dtype != np.complex128:
        raise ValueError("replay generator must return complex128.")
    replay_metrics = replay_error_metrics(replay, target)
    repeat_relative = float(np.sqrt(raw_complex_probe_loss(repeat, replay)))
    replay_metrics["deterministic_repeat_relative_l2"] = repeat_relative
    replay_pass = bool(
        replay_metrics["raw_complex_relative_l2"]
        <= thresholds["replay_raw_relative_l2_max"]
        and replay_metrics["amplitude_relative_l2"]
        <= thresholds["replay_amplitude_relative_l2_max"]
        and replay_metrics[
            "amplitude_weighted_phase_sensitive_relative_l2"
        ]
        <= thresholds["replay_phase_sensitive_relative_l2_max"]
        and repeat_relative
        <= thresholds["deterministic_repeat_relative_l2_max"]
        and np.all(np.isfinite(replay))
        and np.all(np.isfinite(repeat))
    )
    cache = _ProbeCache(candidate_generator, tuple(target.shape))
    cache.values[float(np.float64(true_waist_m))] = 0
    cache.diameters_m.append(float(true_waist_m))
    cache.probes.append(np.asarray(replay, dtype=np.complex128).copy())
    result: dict[str, Any] = {
        "status": "Inconclusive",
        "interpretation": "artifact_operator_handoff_not_closed",
        "stage_a_replay_pass": replay_pass,
        "replay": replay_metrics,
        "stages_completed": ["A"],
    }
    if not replay_pass:
        result["cache"] = {
            "D_waist_m": np.asarray(cache.diameters_m, dtype=np.float64),
            "P_B_candidate": np.stack(cache.probes),
        }
        return result

    def evaluate(diameter_m: float) -> tuple[float, int]:
        lower, upper = bounds_m
        if not np.isfinite(diameter_m) or not lower <= diameter_m <= upper:
            raise ValueError("candidate D_waist lies outside registered bounds.")
        probe, cache_index = cache.get(diameter_m)
        return raw_complex_probe_loss(probe, target), cache_index

    coarse_losses: list[float] = []
    coarse_indices: list[int] = []
    for diameter in coarse:
        loss, cache_index = evaluate(float(diameter))
        coarse_losses.append(loss)
        coarse_indices.append(cache_index)
    coarse_loss_values = np.asarray(coarse_losses, dtype=np.float64)
    coarse_minimum_index = int(np.argmin(coarse_loss_values))
    coarse_minimum_m = float(coarse[coarse_minimum_index])
    lower_fine = max(bounds_m[0], coarse_minimum_m - fine_half_width_m)
    upper_fine = min(bounds_m[1], coarse_minimum_m + fine_half_width_m)
    fine_count = int(np.rint((upper_fine - lower_fine) / fine_step_m))
    if not np.isclose(
        lower_fine + fine_count * fine_step_m,
        upper_fine,
        rtol=0.0,
        atol=32.0 * np.finfo(np.float64).eps * max(1.0, abs(upper_fine)),
    ):
        raise ValueError("registered fine interval is not divisible by its step.")
    fine = lower_fine + np.arange(fine_count + 1) * fine_step_m
    fine_losses: list[float] = []
    fine_indices: list[int] = []
    for diameter in fine:
        loss, cache_index = evaluate(float(diameter))
        fine_losses.append(loss)
        fine_indices.append(cache_index)
    fine_loss_values = np.asarray(fine_losses, dtype=np.float64)

    floor_relative = max(
        replay_metrics["raw_complex_relative_l2"],
        repeat_relative,
        np.finfo(np.float64).eps,
    )
    floor_loss = floor_relative**2
    uniqueness_tolerance = max(
        float(thresholds["profile_uniqueness_loss_floor"]),
        100.0 * floor_loss,
    )
    minimum = extract_profile_minimum(
        fine, fine_loss_values, uniqueness_tolerance=uniqueness_tolerance
    )
    minimum_m = float(minimum["minimum_d_waist_m"])
    minimum_loss = float(minimum["minimum_loss"])
    second_ratio = float(
        minimum["second_best_loss"]
        / max(floor_loss, np.finfo(np.float64).eps)
    )
    profile_absolute_error = abs(minimum_m - true_waist_m)
    profile_interior = bool(
        minimum_m >= bounds_m[0] + fine_step_m
        and minimum_m <= bounds_m[1] - fine_step_m
    )
    profile_pass = bool(
        minimum["unique"]
        and profile_interior
        and profile_absolute_error
        <= thresholds["profile_absolute_error_m_max"]
        and minimum_loss <= thresholds["profile_minimum_loss_max"]
        and second_ratio
        >= thresholds["profile_second_to_floor_ratio_min"]
    )

    h1_m, h2_m = finite_difference_steps_m
    fd_fields: dict[str, NDArray[np.complex128]] = {}
    fd_indices: dict[str, int] = {}
    for name, diameter in {
        "minus_h1": true_waist_m - h1_m,
        "plus_h1": true_waist_m + h1_m,
        "minus_h2": true_waist_m - h2_m,
        "plus_h2": true_waist_m + h2_m,
    }.items():
        field_values, cache_index = cache.get(float(diameter))
        fd_fields[name] = field_values
        fd_indices[name] = cache_index
    fd = finite_difference_controls(
        target,
        replay,
        fd_fields["minus_h1"],
        fd_fields["plus_h1"],
        fd_fields["minus_h2"],
        fd_fields["plus_h2"],
        h1_m=h1_m,
        h2_m=h2_m,
    )
    one_sided_signature = min(
        float(np.sqrt(fd["minus_h2_loss"])),
        float(np.sqrt(fd["plus_h2_loss"])),
    )
    signature_to_floor = one_sided_signature / floor_relative
    jacobian_pass = bool(
        np.isfinite(fd["normalized_jacobian_h1_per_m"])
        and fd["normalized_jacobian_h1_per_m"]
        >= thresholds["normalized_jacobian_per_m_min"]
        and np.isfinite(fd["loss_curvature_per_m2"])
        and fd["loss_curvature_per_m2"] > 0.0
        and fd["jacobian_step_relative_l2"]
        <= thresholds["jacobian_step_relative_l2_max"]
        and signature_to_floor
        >= thresholds["signature_to_floor_ratio_min"]
    )

    branches: dict[str, dict[str, Any]] = {}
    for branch_index, start_m in enumerate(starts):
        branch = equal_budget_bounded_pattern_search(
            evaluate,
            bounds_m=bounds_m,
            start_m=float(start_m),
            initial_step_m=optimizer_initial_step_m,
            evaluation_budget=optimizer_evaluation_budget,
        )
        final_estimate = float(branch["final_estimate_m"])
        absolute_error = abs(final_estimate - true_waist_m)
        relative_error = absolute_error / true_waist_m
        profile_agreement = abs(final_estimate - minimum_m)
        boundary_hit = bool(
            final_estimate <= bounds_m[0] + thresholds["boundary_margin_m"]
            or final_estimate >= bounds_m[1] - thresholds["boundary_margin_m"]
        )
        branch.update(
            {
                "absolute_error_m": absolute_error,
                "relative_error": relative_error,
                "profile_agreement_m": profile_agreement,
                "boundary_hit": boundary_hit,
                "pass": bool(
                    branch["evaluation_count"] == optimizer_evaluation_budget
                    and branch["stopping_reason"] == "evaluation_budget"
                    and absolute_error
                    <= thresholds["optimizer_absolute_error_m_max"]
                    and relative_error
                    <= thresholds["optimizer_relative_error_max"]
                    and profile_agreement
                    <= thresholds["optimizer_profile_agreement_m_max"]
                    and not boundary_hit
                ),
            }
        )
        branches[f"start_{branch_index:02d}"] = branch
    estimates = np.asarray(
        [branch["final_estimate_m"] for branch in branches.values()],
        dtype=np.float64,
    )
    representative_estimate = float(np.median(estimates))
    optimizer_spread = float(np.max(estimates) - np.min(estimates))
    optimizer_profile_agreement = abs(representative_estimate - minimum_m)
    optimizer_pass = bool(
        all(bool(branch["pass"]) for branch in branches.values())
        and optimizer_spread <= thresholds["optimizer_spread_m_max"]
        and optimizer_profile_agreement
        <= thresholds["optimizer_profile_agreement_m_max"]
    )
    best_probe, best_cache_index = cache.get(representative_estimate)
    residual = np.asarray(best_probe - target, dtype=np.complex128)

    if not jacobian_pass:
        status = "Inconclusive"
        interpretation = "local_jacobian_numerical_control_not_closed"
    elif not profile_pass or not optimizer_pass:
        status = "Failed"
        interpretation = "registered_oracle_fit_gate_failed"
    else:
        status = "Passed"
        interpretation = "single_parameter_oracle_fit_passed_working_model_only"
    fd["cache_indices"] = fd_indices
    fd["one_sided_signature_relative_l2"] = one_sided_signature
    fd["signature_to_replay_floor_ratio"] = signature_to_floor
    fd["pass"] = jacobian_pass
    result.update(
        {
            "status": status,
            "interpretation": interpretation,
            "stages_completed": ["A", "B", "C"],
            "profile": {
                "coarse_d_waist_m": coarse,
                "coarse_loss": coarse_loss_values,
                "coarse_cache_index": np.asarray(
                    coarse_indices, dtype=np.int64
                ),
                "coarse_minimum_index": coarse_minimum_index,
                "fine_d_waist_m": fine,
                "fine_loss": fine_loss_values,
                "fine_cache_index": np.asarray(fine_indices, dtype=np.int64),
                "minimum": minimum,
                "absolute_error_m": profile_absolute_error,
                "interior": profile_interior,
                "second_best_to_replay_floor_loss_ratio": second_ratio,
                "replay_floor_relative_l2": floor_relative,
                "replay_floor_loss": floor_loss,
                "uniqueness_tolerance": uniqueness_tolerance,
                "pass": profile_pass,
            },
            "finite_difference": fd,
            "optimizer": {
                "algorithm": "fixed_budget_bounded_pattern_search",
                "branches": branches,
                "final_estimates_m": estimates,
                "representative_estimate_m": representative_estimate,
                "spread_m": optimizer_spread,
                "representative_profile_agreement_m": (
                    optimizer_profile_agreement
                ),
                "equal_budget": bool(
                    all(
                        branch["evaluation_count"]
                        == optimizer_evaluation_budget
                        for branch in branches.values()
                    )
                ),
                "pass": optimizer_pass,
            },
            "P_B_best_raw": np.asarray(best_probe, dtype=np.complex128),
            "residual_field_raw": residual,
            "best_cache_index": best_cache_index,
            "cache": {
                "D_waist_m": np.asarray(cache.diameters_m, dtype=np.float64),
                "P_B_candidate": np.stack(cache.probes).astype(
                    np.complex128, copy=False
                ),
            },
        }
    )
    return result
