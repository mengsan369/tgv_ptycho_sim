"""Evaluation and status helpers for exp044 reconstruction evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.exp040 import relative_l2
from tgv_ptycho.recon.exp043 import align_pair_simulation_only

ComplexArray = NDArray[np.complexfloating]


def global_phase_probe_evaluation_simulation_only(
    probe: ComplexArray, probe_true: ComplexArray
) -> dict[str, Any]:
    """Evaluate a frozen known-B probe with a truth-derived unit phase only."""

    recovered = np.asarray(probe, dtype=np.complex128)
    truth = np.asarray(probe_true, dtype=np.complex128)
    if recovered.shape != truth.shape or not np.all(np.isfinite(recovered)):
        raise ValueError("probe and probe_true must be finite and same-shaped.")
    inner = np.sum(np.conj(recovered) * truth, dtype=np.complex128)
    factor = np.complex128(1.0 + 0.0j if abs(inner) == 0.0 else inner / abs(inner))
    aligned = np.asarray(factor * recovered, dtype=np.complex128)
    return {
        "P_B_rec_global_phase_aligned": aligned,
        "global_phase_factor": factor,
        "raw_probe_relative_l2": relative_l2(recovered, truth),
        "aligned_probe_relative_l2": relative_l2(aligned, truth),
        "simulation_evaluation_only": True,
        "loaded_after_raw_result_frozen": True,
        "enters_optimizer": False,
        "enters_selection": False,
        "enters_stopping": False,
    }


def _weighted_relative_l2(
    recovered: NDArray[np.generic],
    truth: NDArray[np.generic],
    weights: NDArray[np.floating],
) -> float:
    recovered_values = np.asarray(recovered)
    truth_values = np.asarray(truth)
    weight_values = np.asarray(weights, dtype=np.float64)
    if (
        recovered_values.shape != truth_values.shape
        or weight_values.shape != truth_values.shape
    ):
        raise ValueError("weighted error inputs must have the same shape.")
    if np.any(weight_values < 0.0) or not np.all(np.isfinite(weight_values)):
        raise ValueError("weights must be finite and nonnegative.")
    numerator = float(
        np.sum(weight_values * np.abs(recovered_values - truth_values) ** 2)
    )
    denominator = float(np.sum(weight_values * np.abs(truth_values) ** 2))
    return float(np.sqrt(numerator / max(denominator, 1.0e-300)))


def blind_component_evaluation_simulation_only(
    operator: Any,
    result: Mapping[str, Any],
    probe_true: ComplexArray,
    b_true: ComplexArray,
    coverage_map: NDArray[np.integer],
    *,
    illumination_coverage_map: NDArray[np.floating] | None = None,
) -> dict[str, Any]:
    """Evaluate observed/full B and scan-exit products after raw freeze."""

    probe = np.asarray(result["P_B_rec_raw"], dtype=np.complex128)
    b_rec = np.asarray(result["B_rec_raw"], dtype=np.complex128)
    p_true = np.asarray(probe_true, dtype=np.complex128)
    b_truth = np.asarray(b_true, dtype=np.complex128)
    mask = np.asarray(operator.support_mask, dtype=np.bool_)
    coverage = np.asarray(coverage_map, dtype=np.float64)
    if (
        b_rec.shape != b_truth.shape
        or b_rec.shape != mask.shape
        or coverage.shape != mask.shape
    ):
        raise ValueError("B truth/result/mask/coverage shapes must match.")
    aligned = align_pair_simulation_only(
        probe, b_rec, p_true, b_truth, mask
    )
    p_aligned = np.asarray(
        aligned["P_B_rec_reciprocal_gain_aligned"], dtype=np.complex128
    )
    b_aligned = np.asarray(
        aligned["B_rec_reciprocal_gain_aligned"], dtype=np.complex128
    )
    observed_weights = np.where(mask, coverage, 0.0)
    if illumination_coverage_map is None:
        illumination = observed_weights
    else:
        illumination = np.where(
            mask, np.asarray(illumination_coverage_map, dtype=np.float64), 0.0
        )
    positive = coverage[mask]
    if positive.size == 0 or np.min(positive) <= 0.0:
        raise ValueError("The registered observable mask must have positive coverage.")
    first_quartile = float(np.quantile(positive, 0.25, method="higher"))
    low_mask = mask & (coverage <= first_quartile)

    coverage_levels: dict[str, float] = {}
    for value in np.unique(coverage[mask]).astype(np.int64):
        level_mask = mask & (coverage == value)
        coverage_levels[str(int(value))] = relative_l2(
            b_aligned[level_mask], b_truth[level_mask]
        )

    recovered_exit = []
    truth_exit = []
    per_scan = []
    recovered_probe_open = operator.probe_open(p_aligned)
    truth_probe_open = operator.probe_open(p_true)
    positions = (
        operator.positions_m
        if hasattr(operator, "positions_m")
        else operator.reference.positions_m
    )
    for index in range(len(positions)):
        recovered = recovered_probe_open * (
            1.0 + operator.shifted_modulation(b_aligned - 1.0, index)
        )
        truth_value = truth_probe_open * (
            1.0 + operator.shifted_modulation(b_truth - 1.0, index)
        )
        recovered_exit.append(recovered)
        truth_exit.append(truth_value)
        per_scan.append(relative_l2(recovered, truth_value))
    recovered_stack = np.stack(recovered_exit)
    truth_stack = np.stack(truth_exit)
    exit_error = relative_l2(recovered_stack, truth_stack)
    per_scan_array = np.asarray(per_scan, dtype=np.float64)

    return {
        **aligned,
        "B_full_master_relative_l2": relative_l2(b_aligned, b_truth),
        "B_observed_union_relative_l2": relative_l2(
            b_aligned[mask], b_truth[mask]
        ),
        "B_coverage_weighted_relative_l2": _weighted_relative_l2(
            b_aligned, b_truth, observed_weights
        ),
        "B_illumination_coverage_weighted_relative_l2": _weighted_relative_l2(
            b_aligned, b_truth, illumination
        ),
        "B_low_coverage_edge_relative_l2": relative_l2(
            b_aligned[low_mask], b_truth[low_mask]
        ),
        "B_error_by_coverage_multiplicity": coverage_levels,
        "low_coverage_threshold": first_quartile,
        "low_coverage_pixel_count": int(np.count_nonzero(low_mask)),
        "observable_pixel_count": int(np.count_nonzero(mask)),
        "never_observed_pixel_count": int(np.count_nonzero(~mask)),
        "exit_wave_product_relative_l2": exit_error,
        "coverage_weighted_exit_product_relative_l2": float(
            np.sqrt(
                np.average(
                    per_scan_array**2,
                    weights=np.maximum(
                        np.asarray(
                            [np.sum(np.abs(value) ** 2) for value in truth_exit],
                            dtype=np.float64,
                        ),
                        1.0e-300,
                    ),
                )
            )
        ),
        "per_scan_exit_product_relative_l2": per_scan_array,
        "simulation_evaluation_only": True,
        "loaded_after_raw_result_frozen": True,
        "enters_optimizer": False,
        "enters_selection": False,
        "enters_stopping": False,
    }


def determine_exp044_status(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, str]:
    """Apply the preregistered hierarchical exp044 status matrix."""

    if not bool(metrics["forward"]["all_gates_passed"]):
        return {
            "scientific_status": "Inconclusive",
            "status_reason": "source_forward_or_operator_not_closed",
        }
    if not bool(metrics["known_b"]["all_case_gates_passed"]):
        return {
            "scientific_status": "Inconclusive",
            "status_reason": "known_b_probe_recovery_not_closed",
        }
    primary = metrics["blind"]["C3"]
    measurement_ok = (
        float(primary["maximum_detector_relative_residual"])
        <= float(thresholds["blind_detector_relative_residual_max"])
        and float(primary["maximum_pairwise_prediction_relative_l2"])
        <= float(thresholds["repeat_prediction_relative_l2_max"])
        and bool(primary["all_loss_nonincreasing"])
        and bool(primary["all_finite"])
        and bool(primary["fixed_final_selection"])
    )
    if not measurement_ok:
        return {
            "scientific_status": "Failed",
            "status_reason": "blind_measurement_reconstruction_not_closed",
        }
    evaluation = primary["representative_simulation_evaluation_only"]
    component_ok = (
        float(evaluation["probe_aligned_relative_l2"])
        <= float(thresholds["blind_probe_aligned_relative_l2_max"])
        and float(evaluation["B_coverage_weighted_relative_l2"])
        <= float(thresholds["blind_coverage_weighted_B_relative_l2_max"])
        and float(evaluation["exit_wave_product_relative_l2"])
        <= float(thresholds["blind_exit_wave_product_relative_l2_max"])
        and float(evaluation["relative_improvement_vs_exp043_B"])
        >= float(thresholds["required_relative_improvement_vs_exp043_B_min"])
        and float(evaluation["relative_improvement_vs_exp043_exit"])
        >= float(thresholds["required_relative_improvement_vs_exp043_exit_min"])
    )
    if not component_ok:
        return {
            "scientific_status": "Failed",
            "status_reason": (
                "measurement_consistent_but_component_recovery_non_identifiable"
            ),
        }
    return {
        "scientific_status": "Passed",
        "status_reason": "all_registered_measurement_design_gates_passed",
    }


__all__ = [
    "blind_component_evaluation_simulation_only",
    "determine_exp044_status",
    "global_phase_probe_evaluation_simulation_only",
]
