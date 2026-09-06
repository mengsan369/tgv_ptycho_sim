from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.exp040 import center_crop, relative_l2  # noqa: E402
from tgv_ptycho.forward.exp044 import (  # noqa: E402
    FiniteMasterBlindProbeBOperator,
    FiniteMasterKnownBProbeOperator,
    build_localized_multislice_probe,
    build_nested_master_b,
    center_embed,
    gaussian_incident_field,
    operator_control_metrics,
    random_direction_gain_metrics,
)
from tgv_ptycho.forward.scan_windows import (  # noqa: E402
    extract_scan_windows,
    make_scan_window_plan,
    scan_window_adjoint_relative_error,
    scatter_add_scan_windows,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.recon.exp042 import (  # noqa: E402
    detector_relative_residual,
    reconstruct_known_b_probe_damped_gn_cg,
)
from tgv_ptycho.recon.exp043 import (  # noqa: E402
    MatchedBlindProbeBOperator,
    alternating_blind_reconstruction,
    detector_amplitude_relative_residual,
    load_exp042_authoritative_source,
    make_b_initialization,
    pairwise_blind_stability,
    sha256_array_bytes,
    sha256_file,
)
from tgv_ptycho.recon.exp044 import (  # noqa: E402
    blind_component_evaluation_simulation_only,
    determine_exp044_status,
    global_phase_probe_evaluation_simulation_only,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp044 finite-master-B/localized-illumination study."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=("preflight", "development", "formal"),
        default=None,
    )
    return parser.parse_args()


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment",
        "execution",
        "source",
        "design_matrix",
        "optics",
        "sample_a",
        "illumination",
        "sample_b",
        "scan",
        "probe_parameterization",
        "detector",
        "loss",
        "development",
        "known_b",
        "blind",
        "conditioning",
        "formal_thresholds",
        "artifact_contract",
        "output",
    }
    if set(config) != required:
        raise ValueError(f"exp044 root config keys changed: {set(config) ^ required}.")
    if config["experiment"].get("id") != "exp044":
        raise ValueError("experiment.id must be exp044.")
    if config["source"]["exp042"].get("reference_validated") is not False:
        raise ValueError("reference_validated must remain false.")
    if config["source"]["exp042"].get("full_tgv_reference_authorized") is not False:
        raise ValueError("full_tgv_reference_authorized must remain false.")
    ids = [str(case["id"]) for case in config["design_matrix"]]
    if ids != ["C0", "C1", "C2", "C3"]:
        raise ValueError("The registered C0/C1/C2/C3 design matrix changed.")
    if config["illumination"].get("waist_plane") != "sample_A_entrance":
        raise ValueError("The selected Gaussian must be defined at A entrance.")
    if config["illumination"].get("power_convention") != (
        "fixed_total_analytic_gaussian_power"
    ):
        raise ValueError("The primary Gaussian power convention changed.")
    if config["sample_b"].get("reconstruction_active_mask") != (
        "illuminated_scan_union"
    ):
        raise ValueError("The large-master active mask must be the scan union.")
    if config["blind"].get("checkpoint_selection_rule") != (
        "fixed_final_outer_sweep"
    ):
        raise ValueError("Blind checkpoint selection must remain fixed-final.")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping at {path}.")
    return payload


def _hdf5_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _hdf5_ready(child) for key, child in value.items()}
    if isinstance(value, list) and value and all(
        isinstance(child, dict) for child in value
    ):
        return {
            f"item_{index:03d}": _hdf5_ready(child)
            for index, child in enumerate(value)
        }
    return value


def _loss_nonincreasing(values: Any) -> bool:
    array = np.asarray(values, dtype=np.float64)
    scale = max(float(np.max(np.abs(array))), 1.0)
    tolerance = 16.0 * np.finfo(np.float64).eps * scale
    return bool(np.all(np.diff(array) <= tolerance))


def _center_transparent_embed(
    transmission: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    return np.asarray(1.0 + center_embed(transmission - 1.0, shape))


def _illumination_coverage(
    probe_open: np.ndarray,
    plan: Any,
) -> np.ndarray:
    patch = center_crop(np.abs(probe_open) ** 2, plan.window_shape)
    patches = np.repeat(patch[None, :, :], len(plan.starts_yx), axis=0)
    return np.asarray(scatter_add_scan_windows(patches, plan), dtype=np.float64)


def _legacy_coverage(mask: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(mask, dtype=np.int32) * int(count)


def _build_cases(config: dict[str, Any], source: Any) -> dict[str, dict[str, Any]]:
    positions = source.scan_positions
    dx = float(config["sample_b"]["dx_m"])
    master = build_nested_master_b(config)
    master_true = np.asarray(master["B_master_true"], dtype=np.complex128)
    plan = make_scan_window_plan(
        tuple(config["sample_b"]["master_shape"]),
        tuple(config["sample_b"]["interaction_window_shape"]),
        positions,
        dx,
    )
    source_reference = source.operator
    plane = build_localized_multislice_probe(config, incident_type="plane_wave")
    plane_legacy_reference = replace(
        source_reference,
        homogeneous_probe_native=np.asarray(
            plane["P_B_reference_native"], dtype=np.complex128
        ),
        homogeneous_probe_open=np.asarray(
            plane["P_B_reference_open"], dtype=np.complex128
        ),
        homogeneous_detector_open=np.asarray(
            plane["P_detector_reference_open"], dtype=np.complex128
        ),
        finite_b_modulation_open=np.asarray(
            source.B_true - 1.0, dtype=np.complex128
        ),
        transfer_bc=np.asarray(plane["transfer_BC"], dtype=np.complex128),
    )
    plane_legacy_blind = MatchedBlindProbeBOperator(
        plane_legacy_reference, source.support_mask
    )
    plane_blind = FiniteMasterBlindProbeBOperator(
        positions_m=positions,
        node_dx_m=source_reference.node_dx_m,
        quadrature_factor=source_reference.quadrature_factor,
        native_shape=source_reference.native_shape,
        open_shape=source_reference.open_shape,
        detector_roi_shape=source_reference.detector_roi_shape,
        homogeneous_probe_native=np.asarray(
            plane["P_B_reference_native"], dtype=np.complex128
        ),
        homogeneous_probe_open=np.asarray(
            plane["P_B_reference_open"], dtype=np.complex128
        ),
        homogeneous_detector_open=np.asarray(
            plane["P_detector_reference_open"], dtype=np.complex128
        ),
        transfer_bc=np.asarray(plane["transfer_BC"], dtype=np.complex128),
        scan_plan=plan,
        support_mask=plan.visited_region,
    )
    plane_modulation = plane_blind.project_modulation(master_true - 1.0)
    plane_known = FiniteMasterKnownBProbeOperator(plane_blind, plane_modulation)

    localized = build_localized_multislice_probe(config)
    localized_reference = replace(
        source_reference,
        homogeneous_probe_native=np.asarray(
            localized["P_B_reference_native"], dtype=np.complex128
        ),
        homogeneous_probe_open=np.asarray(
            localized["P_B_reference_open"], dtype=np.complex128
        ),
        homogeneous_detector_open=np.asarray(
            localized["P_detector_reference_open"], dtype=np.complex128
        ),
        finite_b_modulation_open=np.asarray(
            source.B_true - 1.0, dtype=np.complex128
        ),
        transfer_bc=np.asarray(localized["transfer_BC"], dtype=np.complex128),
    )
    localized_legacy_blind = MatchedBlindProbeBOperator(
        localized_reference, source.support_mask
    )
    localized_large_blind = FiniteMasterBlindProbeBOperator(
        positions_m=positions,
        node_dx_m=source_reference.node_dx_m,
        quadrature_factor=source_reference.quadrature_factor,
        native_shape=source_reference.native_shape,
        open_shape=source_reference.open_shape,
        detector_roi_shape=source_reference.detector_roi_shape,
        homogeneous_probe_native=np.asarray(
            localized["P_B_reference_native"], dtype=np.complex128
        ),
        homogeneous_probe_open=np.asarray(
            localized["P_B_reference_open"], dtype=np.complex128
        ),
        homogeneous_detector_open=np.asarray(
            localized["P_detector_reference_open"], dtype=np.complex128
        ),
        transfer_bc=np.asarray(localized["transfer_BC"], dtype=np.complex128),
        scan_plan=plan,
        support_mask=plan.visited_region,
    )
    localized_large_modulation = localized_large_blind.project_modulation(
        master_true - 1.0
    )
    localized_large_known = FiniteMasterKnownBProbeOperator(
        localized_large_blind, localized_large_modulation
    )
    legacy_coverage = _legacy_coverage(source.support_mask, len(positions))
    cases = {
        "C0": {
            "label": "open_multislice_plane_wave_legacy_B_baseline",
            "operator": plane_legacy_blind,
            "known_operator": plane_legacy_reference,
            "probe_true": plane["P_B_native"],
            "B_true": source.B_true,
            "modulation_true": source.B_true - 1.0,
            "I_stack": plane_legacy_reference.predict_stack(
                plane["P_B_native"]
            ),
            "coverage_map": legacy_coverage,
            "illumination_coverage_map": (
                legacy_coverage
                * np.abs(plane_legacy_blind.probe_open(plane["P_B_native"]))
                ** 2
            ),
            "large_canvas": False,
            "localized": False,
        },
        "C1": {
            "label": "large_canvas_only",
            "operator": plane_blind,
            "known_operator": plane_known,
            "probe_true": plane["P_B_native"],
            "B_true": master_true,
            "modulation_true": plane_modulation,
            "I_stack": plane_known.predict_stack(plane["P_B_native"]),
            "coverage_map": plan.coverage_map,
            "illumination_coverage_map": _illumination_coverage(
                plane_blind.probe_open(plane["P_B_native"]), plan
            ),
            "large_canvas": True,
            "localized": False,
        },
        "C2": {
            "label": "localized_illumination_only",
            "operator": localized_legacy_blind,
            "known_operator": localized_reference,
            "probe_true": localized["P_B_native"],
            "B_true": source.B_true,
            "modulation_true": source.B_true - 1.0,
            "I_stack": localized_reference.predict_stack(
                localized["P_B_native"]
            ),
            "coverage_map": legacy_coverage,
            "illumination_coverage_map": (
                legacy_coverage
                * np.abs(localized_legacy_blind.probe_open(
                    localized["P_B_native"]
                ))
                ** 2
            ),
            "large_canvas": False,
            "localized": True,
        },
        "C3": {
            "label": "combined_primary",
            "operator": localized_large_blind,
            "known_operator": localized_large_known,
            "probe_true": localized["P_B_native"],
            "B_true": master_true,
            "modulation_true": localized_large_modulation,
            "I_stack": localized_large_known.predict_stack(
                localized["P_B_native"]
            ),
            "coverage_map": plan.coverage_map,
            "illumination_coverage_map": _illumination_coverage(
                localized_large_blind.probe_open(localized["P_B_native"]), plan
            ),
            "large_canvas": True,
            "localized": True,
        },
    }
    for case in cases.values():
        case["observable_mask"] = np.asarray(
            case["operator"].support_mask, dtype=np.bool_
        )
    cases["_shared"] = {
        "master": master,
        "scan_plan": plan,
        "localized": localized,
        "open_plane": plane,
    }
    return cases


def _support_and_padding_controls(
    config: dict[str, Any],
    source: Any,
    cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    control_shape = tuple(config["sample_a"]["support_control_open_shape"])
    localized_control = build_localized_multislice_probe(
        config, open_shape=control_shape
    )
    zero = build_localized_multislice_probe(config, zero_contrast=True)
    large_spot = build_localized_multislice_probe(
        config,
        diameter_1e2_intensity_m=float(
            config["illumination"]["large_spot_limit_diameter_m"]
        ),
    )
    primary = cases["_shared"]["localized"]
    master = cases["_shared"]["master"]["B_master_true"]
    positions = source.scan_positions
    window_shape = tuple(config["sample_b"]["interaction_window_shape"])
    control_plan = make_scan_window_plan(
        control_shape,
        window_shape,
        positions,
        float(config["sample_b"]["dx_m"]),
    )
    control_blind = FiniteMasterBlindProbeBOperator(
        positions_m=positions,
        node_dx_m=float(config["sample_b"]["dx_m"]),
        quadrature_factor=int(config["detector"]["quadrature_factor"]),
        native_shape=tuple(config["sample_a"]["native_shape"]),
        open_shape=control_shape,
        detector_roi_shape=tuple(config["detector"]["native_roi_shape"]),
        homogeneous_probe_native=np.asarray(
            localized_control["P_B_reference_native"], dtype=np.complex128
        ),
        homogeneous_probe_open=np.asarray(
            localized_control["P_B_reference_open"], dtype=np.complex128
        ),
        homogeneous_detector_open=np.asarray(
            localized_control["P_detector_reference_open"], dtype=np.complex128
        ),
        transfer_bc=np.asarray(
            localized_control["transfer_BC"], dtype=np.complex128
        ),
        scan_plan=control_plan,
        support_mask=control_plan.visited_region,
    )
    control_master = _center_transparent_embed(master, control_shape)
    control_modulation = control_blind.project_modulation(control_master - 1.0)
    control_known = FiniteMasterKnownBProbeOperator(
        control_blind, control_modulation
    )
    control_intensity = control_known.predict_stack(
        localized_control["P_B_native"]
    )
    flat, _ = gaussian_incident_field(
        tuple(config["sample_a"]["open_shape"]),
        float(config["sample_a"]["dx_m"]),
        float(config["illumination"]["large_spot_limit_diameter_m"]),
        float(config["illumination"]["analytic_total_power_reference_m2"]),
    )
    flatness = float(np.max(np.abs(flat / np.mean(flat) - 1.0)))
    plane_bridge = relative_l2(
        large_spot["P_B_native"] / np.mean(flat),
        cases["_shared"]["open_plane"]["P_B_native"],
    )
    return {
        "localized_probe_support_control_relative_l2": relative_l2(
            primary["P_B_native"], localized_control["P_B_native"]
        ),
        "detector_padding_control_relative_l2": relative_l2(
            cases["C3"]["I_stack"], control_intensity
        ),
        "zero_contrast_relative_l2": zero["controls"][
            "zero_contrast_relative_l2"
        ],
        "zero_contrast_reference_plus_residual_relative_l2": zero[
            "controls"
        ]["reference_plus_residual_A_to_B_relative_l2"],
        "large_spot_incident_flatness_relative_max": flatness,
        "large_spot_to_exact_open_plane_probe_relative_l2": plane_bridge,
        "control_open_shape": control_shape,
        "control_minimum_scan_margin_m": control_plan.minimum_margin_m,
        "control_I_stack": control_intensity,
        "localized_control": localized_control,
    }


def _dynamic_metrics(case: dict[str, Any]) -> dict[str, float]:
    intensity = np.asarray(case["I_stack"], dtype=np.float64)
    positive = intensity[intensity > 0.0]
    detector_total = float(np.sum(intensity))
    incident_power = float(
        np.sum(np.abs(case["operator"].probe_open(case["probe_true"])) ** 2)
    )
    return {
        "detector_total_intensity": detector_total,
        "detector_mean_intensity": float(np.mean(intensity)),
        "detector_peak_intensity": float(np.max(intensity)),
        "detector_minimum_positive_intensity": float(np.min(positive)),
        "detector_peak_to_mean": float(np.max(intensity) / np.mean(intensity)),
        "detector_dynamic_range_peak_to_minimum_positive": float(
            np.max(intensity) / np.min(positive)
        ),
        "detector_total_per_discrete_probe_energy": detector_total
        / max(incident_power, 1.0e-300),
    }


def _forward_evidence(
    config: dict[str, Any],
    source: Any,
    cases: dict[str, dict[str, Any]],
    support: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = config["formal_thresholds"]
    plan = cases["_shared"]["scan_plan"]
    scan_adjoint = scan_window_adjoint_relative_error(
        plan, seed=int(config["conditioning"]["random_seed"])
    )
    metrics: dict[str, Any] = {}
    controls_hdf5: dict[str, Any] = {}
    for case_id in ("C0", "C1", "C2", "C3"):
        case = cases[case_id]
        controls = operator_control_metrics(
            case["known_operator"],
            case["operator"],
            case["probe_true"],
            case["modulation_true"],
            case["I_stack"],
            seed=int(config["conditioning"]["random_seed"])
            + int(case_id[-1]),
        )
        gate = bool(
            controls["forward_replay_relative_l2"]
            <= float(thresholds["forward_replay_relative_l2_max"])
            and controls["propagation_linear_adjoint_relative_error"]
            <= float(
                thresholds["propagation_linear_adjoint_relative_error_max"]
            )
            and controls[
                "combined_intensity_jacobian_real_adjoint_relative_error"
            ]
            <= float(thresholds["intensity_jacobian_adjoint_relative_error_max"])
            and controls[
                "combined_intensity_jacobian_directional_relative_error"
            ]
            <= float(
                thresholds[
                    "combined_loss_directional_gradient_relative_error_max"
                ]
            )
            and controls["intensity_all_finite"]
            and controls["intensity_nonnegative"]
        )
        gains = random_direction_gain_metrics(
            case["operator"],
            case["probe_true"],
            case["modulation_true"],
            seed=int(config["conditioning"]["random_seed"])
            + 10 * int(case_id[-1]),
            count=int(config["conditioning"]["random_direction_count"]),
        )
        controls_hdf5[case_id] = {"operator": controls, "conditioning": gains}
        controls["random_direction_gain"] = {
            key: {
                metric: value
                for metric, value in gains[key].items()
                if metric != "values"
            }
            for key in ("probe", "sample_b_phase", "joint")
        }
        if case["large_canvas"]:
            gate = bool(
                gate
                and scan_adjoint
                <= float(thresholds["scan_window_adjoint_relative_error_max"])
                and plan.minimum_margin_m
                >= float(thresholds["minimum_physical_margin_m"])
            )
        if case["localized"]:
            power = cases["_shared"]["localized"]["power"]
            localized_controls = cases["_shared"]["localized"]["controls"]
            gate = bool(
                gate
                and power["captured_power_fraction"]
                >= float(thresholds["localized_captured_power_fraction_min"])
                and localized_controls["A_residual_edge_energy_fraction"]
                <= float(thresholds["source_residual_edge_energy_fraction_max"])
                and localized_controls[
                    "reference_plus_residual_A_to_B_relative_l2"
                ]
                <= float(thresholds["forward_replay_relative_l2_max"])
                and support["localized_probe_support_control_relative_l2"]
                <= float(
                    thresholds[
                        "localized_probe_support_control_relative_l2_max"
                    ]
                )
                and support["detector_padding_control_relative_l2"]
                <= float(thresholds["detector_padding_control_relative_l2_max"])
            )
        metrics[case_id] = {
            "label": case["label"],
            "controls": controls,
            "dynamic_range": _dynamic_metrics(case),
            "observable_pixel_count": int(
                np.count_nonzero(case["observable_mask"])
            ),
            "never_observed_pixel_count": int(
                np.count_nonzero(~case["observable_mask"])
            ),
            "coverage_minimum_positive": int(
                np.min(case["coverage_map"][case["observable_mask"]])
            ),
            "coverage_maximum": int(np.max(case["coverage_map"])),
            "all_gates_passed": gate,
        }
    localized_controls = cases["_shared"]["localized"]["controls"]
    metrics["shared"] = {
        "exp043_authoritative_bridge": {
            "forward_replay_relative_l2": source.replay[
                "independent_forward_replay_relative_l2"
            ],
            "open_plane_probe_relative_l2_vs_exp043_native_A_source": relative_l2(
                cases["_shared"]["open_plane"]["P_B_native"], source.P_B_true
            ),
            "hash_locked": True,
        },
        "scan_window_adjoint_relative_error": scan_adjoint,
        "minimum_physical_margin_m": plan.minimum_margin_m,
        "minimum_physical_margin_px": plan.minimum_margin_px,
        "all_scan_windows_inside": bool(np.all(plan.fully_inside)),
        "center_alignment_offset_xy_m": plan.center_alignment_offset_xy_m,
        "localized_power": cases["_shared"]["localized"]["power"],
        "localized_source_controls": localized_controls,
        "support_and_padding_controls": {
            key: value
            for key, value in support.items()
            if key not in {"control_I_stack", "localized_control"}
        },
        "all_gates_passed": bool(
            all(metrics[case]["all_gates_passed"] for case in ("C0", "C1", "C2", "C3"))
        ),
    }
    controls_hdf5["shared"] = support
    return metrics, controls_hdf5


def _known_b_settings(
    config: dict[str, Any], stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    reconstruction = {
        "known_sample_b": True,
        "update_sample_b": False,
        "truth_used_by_optimizer": False,
        **config["known_b"]["armijo"],
    }
    control = deepcopy(config["known_b"]["optimizer"])
    if stage == "development":
        control.update(config["development"]["known_b"])
    return reconstruction, control


def _known_b_reconstruction(
    config: dict[str, Any],
    source: Any,
    cases: dict[str, dict[str, Any]],
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = config["formal_thresholds"]
    summaries: dict[str, Any] = {}
    results: dict[str, Any] = {}
    reconstruction, control = _known_b_settings(config, stage)
    selected = ("C0", "C3") if stage == "development" else (
        "C0",
        "C1",
        "C2",
        "C3",
    )

    bridge_prediction = source.operator.predict_stack(source.authoritative_P_B_rec)
    bridge_evaluation = global_phase_probe_evaluation_simulation_only(
        source.authoritative_P_B_rec, source.P_B_true
    )
    bridge_residual = detector_relative_residual(bridge_prediction, source.I_stack)
    bridge_gate = bool(
        bridge_residual
        <= float(thresholds["known_b_detector_relative_residual_max"])
        and bridge_evaluation["aligned_probe_relative_l2"]
        <= float(thresholds["known_b_probe_aligned_relative_l2_max"])
    )
    summaries["E43"] = {
        "source": "exp042_authoritative_raw_bridge",
        "final_detector_relative_residual": bridge_residual,
        "final_detector_amplitude_relative_residual": (
            detector_amplitude_relative_residual(bridge_prediction, source.I_stack)
        ),
        "aligned_probe_relative_l2_simulation_only": bridge_evaluation[
            "aligned_probe_relative_l2"
        ],
        "authoritative_raw_sha256": sha256_array_bytes(
            source.authoritative_P_B_rec
        ),
        "gates_passed": bridge_gate,
    }
    results["E43"] = {
        "P_B_rec_raw": source.authoritative_P_B_rec,
        "prediction_final": bridge_prediction,
        "simulation_evaluation_only": bridge_evaluation,
        "source": "exp042_authoritative_raw_bridge",
    }
    for case_id in selected:
        case = cases[case_id]
        start = time.perf_counter()
        result = reconstruct_known_b_probe_damped_gn_cg(
            case["known_operator"],
            case["I_stack"],
            case["known_operator"].homogeneous_probe_native,
            reconstruction,
            control,
        )
        runtime = time.perf_counter() - start
        raw = np.asarray(result["P_B_rec"], dtype=np.complex128).copy()
        prediction = case["known_operator"].predict_stack(raw)
        residual = detector_relative_residual(prediction, case["I_stack"])
        amplitude_residual = detector_amplitude_relative_residual(
            prediction, case["I_stack"]
        )
        evaluation = global_phase_probe_evaluation_simulation_only(
            raw, case["probe_true"]
        )
        gate = bool(
            residual
            <= float(thresholds["known_b_detector_relative_residual_max"])
            and evaluation["aligned_probe_relative_l2"]
            <= float(thresholds["known_b_probe_aligned_relative_l2_max"])
            and _loss_nonincreasing(result["loss_curve"])
            and np.all(np.isfinite(raw))
        )
        summaries[case_id] = {
            "initial_loss": float(result["loss_curve"][0]),
            "final_loss": float(result["loss_curve"][-1]),
            "final_detector_relative_residual": residual,
            "final_detector_amplitude_relative_residual": amplitude_residual,
            "aligned_probe_relative_l2_simulation_only": evaluation[
                "aligned_probe_relative_l2"
            ],
            "iterations_completed": int(result["iterations_completed"]),
            "operator_action_budget_units": int(
                result["operator_action_budget_units"]
            ),
            "loss_nonincreasing": _loss_nonincreasing(result["loss_curve"]),
            "runtime_seconds": runtime,
            "gates_passed": gate,
        }
        payload = dict(result)
        payload["P_B_rec_raw"] = payload.pop("P_B_rec")
        payload["prediction_independent_replay"] = prediction
        payload["simulation_evaluation_only"] = evaluation
        results[case_id] = payload
    summaries["executed_cases"] = list(selected)
    summaries["all_case_gates_passed"] = bool(
        summaries["E43"]["gates_passed"]
        and all(summaries[case]["gates_passed"] for case in selected)
    )
    return summaries, results


def _blind_settings(config: dict[str, Any], stage: str) -> dict[str, Any]:
    settings = deepcopy(config["blind"])
    if stage == "development":
        settings["outer_sweeps"] = int(config["development"]["blind"]["outer_sweeps"])
    return settings


def _branch_summary(result: dict[str, Any], runtime: float) -> dict[str, Any]:
    last = result["block_records"][-1]
    return {
        "initial_loss": float(result["loss_curve"][0]),
        "final_loss": float(result["loss_curve"][-1]),
        "final_detector_relative_residual": float(
            result["detector_relative_residual_curve"][-1]
        ),
        "final_detector_amplitude_relative_residual": float(
            result["detector_amplitude_relative_residual_curve"][-1]
        ),
        "loss_nonincreasing": _loss_nonincreasing(result["loss_curve"]),
        "outer_sweeps_completed": int(result["outer_sweeps_completed"]),
        "stopping_reason": str(result["stopping_reason"]),
        "operator_action_budget_units": int(result["operator_action_budget_units"]),
        "final_probe_gradient_l2_norm": float(
            last["probe"]["gradient_l2_norm_curve"][-1]
        ),
        "final_B_gradient_l2_norm": float(
            last["sample_b"]["gradient_l2_norm_curve"][-1]
        ),
        "probe_norm_final_to_initial_ratio": float(
            result["gauge_probe_l2_norm_curve"][-1]
            / result["gauge_probe_l2_norm_curve"][0]
        ),
        "runtime_seconds": runtime,
        "fixed_final_selection": bool(
            result["checkpoint_selection_rule"] == "fixed_final_outer_sweep"
        ),
        "all_finite": bool(
            all(
                np.all(np.isfinite(result[key]))
                for key in (
                    "P_B_rec_raw",
                    "B_rec_raw",
                    "prediction_final",
                    "loss_curve",
                )
            )
        ),
    }


def _compact_blind_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    compact_records = []
    for record in payload["block_records"]:
        compact = {"outer_sweep": record["outer_sweep"]}
        for name in ("probe", "sample_b"):
            compact[name] = {
                key: value
                for key, value in record[name].items()
                if key not in {"probe", "modulation", "prediction_final"}
            }
        compact_records.append(compact)
    payload["block_records"] = compact_records
    payload["gauge"] = payload.pop("canonicalization")
    return payload


def _blind_reconstruction(
    config: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    known: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = _blind_settings(config, stage)
    if stage == "development":
        selected = ("C0", "C3")
    else:
        selected = ("C0", "C1", "C2", "C3")
    all_summaries: dict[str, Any] = {}
    all_results: dict[str, Any] = {}
    for case_id in selected:
        case = cases[case_id]
        measurement_control_closed = bool(
            known[case_id]["final_detector_relative_residual"]
            <= float(
                config["formal_thresholds"][
                    "known_b_detector_relative_residual_max"
                ]
            )
            and known[case_id]["loss_nonincreasing"]
        )
        if not measurement_control_closed:
            all_summaries[case_id] = {
                "executed": False,
                "reason": "known_B_measurement_gate_not_closed",
            }
            continue
        initializations = config["blind"]["initializations"]
        if case_id != "C3" or stage == "development":
            initializations = initializations[:1]
        branch_results: dict[str, Any] = {}
        branch_summaries: dict[str, Any] = {}
        evaluations: dict[str, Any] = {}
        for initialization in initializations:
            name = str(initialization["name"])
            init_modulation = make_b_initialization(
                case["observable_mask"], initialization
            )
            start = time.perf_counter()
            result = alternating_blind_reconstruction(
                case["operator"],
                case["I_stack"],
                case["known_operator"].homogeneous_probe_native,
                init_modulation,
                settings,
                seed_offset=int(initialization["seed"]),
            )
            runtime = time.perf_counter() - start
            raw_probe = np.asarray(result["P_B_rec_raw"]).copy()
            raw_b = np.asarray(result["B_rec_raw"]).copy()
            raw_prediction = np.asarray(result["prediction_final"]).copy()
            frozen_hashes = {
                "P_B_rec_raw_sha256": sha256_array_bytes(raw_probe),
                "B_rec_raw_sha256": sha256_array_bytes(raw_b),
                "prediction_final_sha256": sha256_array_bytes(raw_prediction),
            }
            evaluation = blind_component_evaluation_simulation_only(
                case["operator"],
                result,
                case["probe_true"],
                case["B_true"],
                case["coverage_map"],
                illumination_coverage_map=case["illumination_coverage_map"],
            )
            branch_results[name] = {
                **_compact_blind_result(result),
                "raw_freeze_hashes": frozen_hashes,
                "simulation_evaluation_only": evaluation,
            }
            branch_summaries[name] = _branch_summary(result, runtime)
            evaluations[name] = evaluation
        stability = pairwise_blind_stability(branch_results)
        representative_name = str(config["blind"]["representative_branch"])
        evaluation = evaluations[representative_name]
        evaluation_summary = {
            key: value
            for key, value in evaluation.items()
            if np.isscalar(value) and key != "complex_factor"
        }
        evaluation_summary["relative_improvement_vs_exp043_B"] = (
            1.0
            - float(evaluation["B_coverage_weighted_relative_l2"])
            / 0.401164927
        )
        evaluation_summary["relative_improvement_vs_exp043_exit"] = (
            1.0
            - float(evaluation["exit_wave_product_relative_l2"])
            / 0.305120617
        )
        all_summaries[case_id] = {
            "executed": True,
            "representative_branch": representative_name,
            "branches": branch_summaries,
            "maximum_detector_relative_residual": max(
                value["final_detector_relative_residual"]
                for value in branch_summaries.values()
            ),
            "maximum_pairwise_prediction_relative_l2": float(
                stability["maximum_pairwise_prediction_relative_l2"]
            ),
            "all_loss_nonincreasing": all(
                value["loss_nonincreasing"] for value in branch_summaries.values()
            ),
            "all_finite": all(
                value["all_finite"] for value in branch_summaries.values()
            ),
            "fixed_final_selection": all(
                value["fixed_final_selection"]
                for value in branch_summaries.values()
            ),
            "representative_simulation_evaluation_only": evaluation_summary,
        }
        all_results[case_id] = {
            "branches": branch_results,
            "representative_branch": representative_name,
            "representative": branch_results[representative_name],
            "stability": stability,
        }
    return all_summaries, all_results


def _load_c0_blind_bridge(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = config["source"]["exp043"]
    run = PROJECT_ROOT / settings["run"]
    hdf5_path = run / settings["hdf5"]
    metrics_path = run / settings["metrics"]
    if sha256_file(hdf5_path) != str(settings["file_sha256"]["hdf5"]):
        raise ValueError("exp043 HDF5 bridge SHA256 mismatch.")
    if sha256_file(metrics_path) != str(settings["file_sha256"]["metrics"]):
        raise ValueError("exp043 metrics bridge SHA256 mismatch.")
    source_metrics = _load_json(metrics_path)
    with h5py.File(hdf5_path, "r") as h5:
        root = h5["/entry/reconstruction/blind_joint/representative"]
        result = {
            key: np.asarray(root[key][...])
            for key in (
                "P_B_init_raw",
                "B_init_raw",
                "P_B_rec_raw",
                "B_rec_raw",
                "prediction_final",
                "loss_curve",
                "detector_relative_residual_curve",
                "detector_amplitude_relative_residual_curve",
            )
        }
    blind = source_metrics["blind_primary"]
    summary = {
        "executed": False,
        "source": "hash_locked_exp043_authoritative_bridge",
        "representative_branch": blind["representative_branch"],
        "maximum_detector_relative_residual": blind[
            "maximum_branch_detector_relative_residual"
        ],
        "maximum_pairwise_prediction_relative_l2": blind[
            "maximum_pairwise_prediction_relative_l2"
        ],
        "representative_simulation_evaluation_only": {
            "probe_aligned_relative_l2": blind[
                "representative_probe_aligned_relative_l2_simulation_only"
            ],
            "B_coverage_weighted_relative_l2": blind[
                "representative_B_aligned_active_relative_l2_simulation_only"
            ],
            "exit_wave_product_relative_l2": blind[
                "representative_exit_wave_product_relative_l2_simulation_only"
            ],
            "relative_improvement_vs_exp043_B": 0.0,
            "relative_improvement_vs_exp043_exit": 0.0,
        },
    }
    return summary, result


def _save_figures(
    run_dir: Path,
    config: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    forward: dict[str, Any],
    known_results: dict[str, Any],
    blind_results: dict[str, Any],
    blind_metrics: dict[str, Any],
) -> None:
    names = list(config["output"]["figure_filenames"])
    if len(names) != 6:
        raise ValueError("The exp044 figure contract requires six figures.")
    figures = run_dir / "figures"
    master = cases["C3"]["B_true"]
    coverage = cases["C3"]["coverage_map"]
    illum_coverage = cases["C3"]["illumination_coverage_map"]
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    panels = (
        (np.angle(master), "finite master B phase", "twilight"),
        (coverage, "scan-window coverage", "viridis"),
        (cases["C3"]["observable_mask"], "observable B mask", "gray"),
        (np.log10(illum_coverage + 1.0e-30), "log10 illumination coverage", "magma"),
    )
    for axis, (values, title, cmap) in zip(axes, panels, strict=True):
        image = axis.imshow(values, origin="lower", cmap=cmap)
        axis.set_title(title, fontsize=9)
        figure.colorbar(image, ax=axis, shrink=0.72)
    figure.tight_layout()
    figure.savefig(figures / names[0], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    totals = [
        forward[case]["dynamic_range"]["detector_total_intensity"]
        for case in ("C0", "C1", "C2", "C3")
    ]
    axes[0].bar(("C0", "C1", "C2", "C3"), totals)
    axes[0].set(title="detector total intensity", ylabel="a.u.")
    power = cases["_shared"]["localized"]["power"]
    axes[1].bar(
        ("analytic", "captured"),
        (power["analytic_total_power"], power["captured_power"]),
    )
    axes[1].set(title="Gaussian power", ylabel="m$^2$ equivalent")
    axes[2].imshow(
        np.abs(cases["_shared"]["localized"]["P_B_native"]),
        origin="lower",
        cmap="viridis",
    )
    axes[2].set_title("localized P_B amplitude")
    figure.tight_layout()
    figure.savefig(figures / names[1], dpi=160)
    plt.close(figure)

    executed_known = [
        case for case in ("C0", "C1", "C2", "C3") if case in known_results
    ]
    if not executed_known:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "known-B stage not executed during forward preflight",
            ha="center",
        )
        figure.savefig(figures / names[2], dpi=160)
        plt.close(figure)
        executed_known = []
    else:
        figure, axes = plt.subplots(
            len(executed_known), 3, figsize=(10, 3 * len(executed_known))
        )
        axes = np.atleast_2d(axes)
        for row, case_id in enumerate(executed_known):
            truth = cases[case_id]["probe_true"]
            rec = known_results[case_id]["P_B_rec_raw"]
            evaluation = global_phase_probe_evaluation_simulation_only(rec, truth)
            aligned = evaluation["P_B_rec_global_phase_aligned"]
            for axis, values, title in zip(
                axes[row],
                (np.abs(truth), np.abs(rec), np.abs(aligned - truth)),
                (f"{case_id} truth |P|", "raw rec |P|", "aligned |error|"),
                strict=True,
            ):
                image = axis.imshow(values, origin="lower", cmap="viridis")
                axis.set_title(title, fontsize=8)
                figure.colorbar(image, ax=axis, shrink=0.68)
        figure.tight_layout()
        figure.savefig(figures / names[2], dpi=160)
        plt.close(figure)

    if "C3" in blind_results:
        representative = blind_results["C3"]["representative"]
        evaluation = representative["simulation_evaluation_only"]
        p_values = (
            np.abs(cases["C3"]["probe_true"]),
            np.abs(representative["P_B_rec_raw"]),
            np.abs(
                evaluation["P_B_rec_reciprocal_gain_aligned"]
                - cases["C3"]["probe_true"]
            ),
        )
        mask = cases["C3"]["observable_mask"]
        b_aligned = evaluation["B_rec_reciprocal_gain_aligned"]
        b_values = (
            np.where(mask, np.angle(cases["C3"]["B_true"]), np.nan),
            np.where(mask, np.angle(representative["B_rec_raw"]), np.nan),
            np.where(mask, np.abs(b_aligned - cases["C3"]["B_true"]), np.nan),
        )
        figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        for axis, values, title in zip(
            axes[0], p_values, ("P truth", "P raw rec", "P aligned error"), strict=True
        ):
            image = axis.imshow(values, origin="lower", cmap="viridis")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, shrink=0.7)
        for axis, values, title in zip(
            axes[1],
            b_values,
            ("B truth phase", "B raw phase", "B aligned error"),
            strict=True,
        ):
            cmap = "twilight" if "phase" in title else "viridis"
            image = axis.imshow(values, origin="lower", cmap=cmap)
            axis.set_title(title)
            figure.colorbar(image, ax=axis, shrink=0.7)
        figure.tight_layout()
        figure.savefig(figures / names[3], dpi=160)
        plt.close(figure)

        target = cases["C3"]["I_stack"][0]
        prediction = representative["prediction_final"][0]
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for axis, values, title, cmap in zip(
            axes,
            (target, prediction, prediction - target),
            ("target intensity", "blind prediction", "prediction - target"),
            ("magma", "magma", "coolwarm"),
            strict=True,
        ):
            image = axis.imshow(values, origin="lower", cmap=cmap)
            axis.set_title(title)
            figure.colorbar(image, ax=axis, shrink=0.72)
        figure.tight_layout()
        figure.savefig(figures / names[4], dpi=160)
        plt.close(figure)
    else:
        for index in (3, 4):
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.axis("off")
            axis.text(0.5, 0.5, "blind stage not executed", ha="center")
            figure.savefig(figures / names[index], dpi=160)
            plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    labels = ("probe", "coverage-weighted B", "exit product")
    baseline = (0.206064417, 0.401164927, 0.305120617)
    axis.bar(np.arange(3) - 0.18, baseline, 0.36, label="exp043 C0")
    if "C3" in blind_metrics and blind_metrics["C3"].get("executed"):
        evaluation = blind_metrics["C3"][
            "representative_simulation_evaluation_only"
        ]
        current = (
            evaluation["probe_aligned_relative_l2"],
            evaluation["B_coverage_weighted_relative_l2"],
            evaluation["exit_wave_product_relative_l2"],
        )
        axis.bar(np.arange(3) + 0.18, current, 0.36, label="exp044 C3")
    axis.set_xticks(np.arange(3), labels, rotation=10)
    axis.set_ylabel("simulation-only relative L2")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / names[5], dpi=160)
    plt.close(figure)


def _all_numeric_finite(group: h5py.Group) -> bool:
    valid = True

    def visitor(_name: str, value: h5py.Group | h5py.Dataset) -> None:
        nonlocal valid
        if isinstance(value, h5py.Dataset) and value.dtype.kind in "biufc":
            valid = valid and bool(np.all(np.isfinite(value[...])))

    group.visititems(visitor)
    return valid


def _validate_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    output = config["output"]
    hdf5_path = run_dir / "outputs" / output["hdf5_filename"]
    required = [
        "/entry/data/I_stack",
        "/entry/data/scan_positions",
        "/entry/data/cases/C0/I_stack",
        "/entry/data/cases/C3/I_stack",
        "/entry/data/exp043_bridge/I_stack",
        "/entry/sample/sample_b/designs/C3/coverage_multiplicity",
        "/entry/sample/sample_b/designs/C3/observable_B_mask",
        "/entry/truth/cases/C3/P_B_true",
        "/entry/truth/cases/C3/B_master_true",
        "/entry/reconstruction/source_provenance",
        "/entry/reconstruction/design_matrix",
        "/entry/reconstruction/forward_controls",
        "/entry/metadata",
        "/entry/metrics",
    ]
    with h5py.File(hdf5_path, "r") as h5:
        children = sorted(h5["entry"].keys())
        expected = sorted(config["artifact_contract"]["required_entry_children"])
        if children != expected:
            raise RuntimeError("HDF5 /entry children violate the registered contract.")
        missing = [path for path in required if path not in h5]
        if missing:
            raise RuntimeError(f"Missing required HDF5 paths: {missing}.")
        if "calibration" in h5["entry"] or "preprocessing" in h5["entry"]:
            raise RuntimeError(
                "Simulation output fabricated calibration/preprocessing."
            )
        if not _all_numeric_finite(h5["entry"]):
            raise RuntimeError("HDF5 contains non-finite numeric data.")
        if sha256_array_bytes(h5["/entry/data/exp043_bridge/I_stack"][...]) != (
            config["source"]["exp042"]["dataset_sha256"]["I_stack"]
        ):
            raise RuntimeError("Persisted C0 source intensity bytes changed.")
        h5_status = h5["/entry/metrics/scientific_status"][()]
        if isinstance(h5_status, bytes):
            h5_status = h5_status.decode("utf-8")
        if str(h5_status) != str(metrics["scientific_status"]):
            raise RuntimeError("JSON/HDF5 scientific status mismatch.")
    figure_hashes = {}
    for name in output["figure_filenames"]:
        path = run_dir / "figures" / name
        with Image.open(path) as figure:
            figure.verify()
        figure_hashes[name] = sha256_file(path)
    return {
        "required_hdf5_paths_present": True,
        "entry_children_exact": True,
        "source_dataset_bytes_exact": True,
        "all_numeric_hdf5_datasets_finite": True,
        "json_hdf5_status_consistent": True,
        "figure_count": len(figure_hashes),
        "figure_sha256": figure_hashes,
    }


def run(config_path: Path, *, stage: str | None = None) -> Path:
    start = time.perf_counter()
    config = load_config(config_path)
    _validate_config(config)
    execution_stage = stage or str(config["execution"]["default_stage"])
    output = config["output"]
    run_name = f"{output['run_name']}_{execution_stage}"
    run_dir = make_run_dir(PROJECT_ROOT / output["root"], run_name)
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "artifacts_validated": False,
            "execution_stage": execution_stage,
            "created_at_utc": created_at_utc(),
        },
    )
    try:
        source = load_exp042_authoritative_source(
            PROJECT_ROOT, config["source"]["exp042"]
        )
        cases = _build_cases(config, source)
        support = _support_and_padding_controls(config, source, cases)
        forward_metrics, forward_hdf5 = _forward_evidence(
            config, source, cases, support
        )
        known_metrics: dict[str, Any] = {
            "executed": False,
            "all_case_gates_passed": False,
        }
        known_results: dict[str, Any] = {}
        blind_metrics: dict[str, Any] = {}
        blind_results: dict[str, Any] = {}
        c0_blind_metrics, c0_blind_result = _load_c0_blind_bridge(config)
        blind_metrics["E43"] = c0_blind_metrics
        blind_results["E43"] = c0_blind_result
        if execution_stage != "preflight" and forward_metrics["shared"][
            "all_gates_passed"
        ]:
            known_metrics, known_results = _known_b_reconstruction(
                config, source, cases, execution_stage
            )
            known_metrics["executed"] = True
            new_blind_metrics, new_blind_results = _blind_reconstruction(
                config, cases, known_metrics, execution_stage
            )
            blind_metrics.update(new_blind_metrics)
            blind_results.update(new_blind_results)

        if execution_stage == "formal":
            status = determine_exp044_status(
                {
                    "forward": forward_metrics["shared"],
                    "known_b": known_metrics,
                    "blind": blind_metrics,
                },
                config["formal_thresholds"],
            )
        else:
            status = {
                "scientific_status": "Development",
                "status_reason": (
                    f"{execution_stage}_scientific_gates_not_authoritative"
                ),
            }
        runtime = time.perf_counter() - start
        metrics = {
            "experiment_id": "exp044",
            "execution_stage": execution_stage,
            **status,
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "design_matrix": {
                item["id"]: item for item in config["design_matrix"]
            },
            "forward": forward_metrics,
            "known_b": known_metrics,
            "blind": blind_metrics,
            "exp043_baseline": {
                "detector_relative_residual": 0.0273522822,
                "probe_aligned_relative_l2": 0.206064417,
                "B_active_relative_l2": 0.401164927,
                "exit_wave_product_relative_l2": 0.305120617,
            },
            "truth_use": {
                "initialization": False,
                "optimizer": False,
                "checkpoint_selection": False,
                "stopping": False,
                "case_selection": False,
                "simulation_evaluation_loaded_after_raw_freeze": True,
            },
            "runtime_seconds": runtime,
            "peak_memory_estimate_bytes": int(
                40
                * np.prod(config["sample_a"]["support_control_open_shape"])
                * np.dtype(np.complex128).itemsize
            ),
        }
        metadata = {
            "experiment_id": "exp044",
            "created_at_utc": created_at_utc(),
            "execution_stage": execution_stage,
            "git_commit": get_git_commit(PROJECT_ROOT),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "source_experiments": ["exp031", "exp040", "exp042", "exp043"],
            "selected_forward": (
                "A_entrance_Gaussian_scalar_multislice_plus_finite_master_B_windows"
            ),
            "known_b_method": config["known_b"]["method"],
            "blind_method": config["blind"]["algorithm"],
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "project_hdf5_schema_changed": False,
        }
        plan = cases["_shared"]["scan_plan"]
        sample = {
            "sample_a": {
                **config["sample_a"],
                "model_boundary": "selected_exp040_scalar_working_model",
            },
            "sample_b": {
                **config["sample_b"],
                "phase_cells_rad": cases["_shared"]["master"][
                    "phase_cells_rad"
                ],
                "scan_windows": {
                    "starts_yx": plan.starts_yx,
                    "stops_yx": plan.stops_yx,
                    "margins_px_left_right_top_bottom": plan.margins_px,
                    "shifts_xy_px": plan.shifts_xy_px,
                    "actual_C3_patches": extract_scan_windows(
                        cases["C3"]["B_true"], plan
                    ),
                },
                "designs": {
                    case_id: {
                        "coverage_multiplicity": cases[case_id]["coverage_map"],
                        "illumination_weighted_coverage": cases[case_id][
                            "illumination_coverage_map"
                        ],
                        "observable_B_mask": cases[case_id]["observable_mask"],
                        "never_observed_B_mask": ~cases[case_id][
                            "observable_mask"
                        ],
                    }
                    for case_id in ("C0", "C1", "C2", "C3")
                },
            },
        }
        localized = cases["_shared"]["localized"]
        truth = {
            "identity": "simulation truth under selected exp040 scalar working model",
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "use": "forward controls and postfreeze simulation evaluation only",
            "cases": {
                case_id: {
                    "P_B_true": cases[case_id]["probe_true"],
                    "B_master_true": cases[case_id]["B_true"],
                }
                for case_id in ("C0", "C1", "C2", "C3")
            },
            "localized_fields": {
                key: localized[key]
                for key in (
                    "incident_open",
                    "U_A_exit_open",
                    "U_A_homogeneous_open",
                    "U_A_residual_open",
                    "P_B_open",
                    "P_B_reference_open",
                    "P_B_residual_open",
                    "P_B_native",
                    "P_B_reference_native",
                    "z_m",
                    "slice_widths_m",
                    "D_z_m",
                )
            },
        }
        reconstruction = {
            "source_provenance": {
                "exp042": source.provenance,
                "exp043": config["source"]["exp043"],
                "exp031_config_sha256": (
                    "71057707EF5453582C4DB32C4341DD4B208110E62D91FB4BC4D0CCCFC5B86E64"
                ),
                "exp040_R8_config_current_worktree_sha256": (
                    "9EBF713F3E7D9F803517932E480CC4AF5300347644D2D84D9F48257519B96BEB"
                ),
            },
            "design_matrix": {
                item["id"]: item for item in config["design_matrix"]
            },
            "beam_definition": config["illumination"],
            "power_normalization": localized["power"],
            "master_B_geometry": config["sample_b"],
            "forward_controls": forward_hdf5,
            "known_B_probe_reconstruction": known_results,
            "blind_joint_reconstruction": blind_results,
            "gauge": {
                "measurement_only": (
                    "identity_under_frozen_reference_and_transparent_exterior"
                ),
                "simulation_only": "postfreeze_truth_derived_reciprocal_gain",
            },
            "comparisons_to_exp043": metrics["exp043_baseline"],
        }
        data = {
            "cases": {
                case_id: {"I_stack": cases[case_id]["I_stack"]}
                for case_id in ("C0", "C1", "C2", "C3")
            },
            "exp043_bridge": {"I_stack": source.I_stack},
        }
        instrument = {
            "optics": config["optics"],
            "detector": config["detector"],
            "scan": config["scan"],
            "beam": config["illumination"],
        }
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_ptycho_hdf5(
            run_dir / "outputs" / output["hdf5_filename"],
            I_stack=cases["C3"]["I_stack"],
            scan_positions=source.scan_positions,
            data=_hdf5_ready(data),
            instrument=_hdf5_ready(instrument),
            sample=_hdf5_ready(sample),
            truth=_hdf5_ready(truth),
            reconstruction=_hdf5_ready(reconstruction),
            config_yaml=config_to_yaml(config),
            metadata=_hdf5_ready(metadata),
            metrics=_hdf5_ready(metrics),
        )
        _save_figures(
            run_dir,
            config,
            cases,
            forward_metrics,
            known_results,
            blind_results,
            blind_metrics,
        )
        audit = _validate_artifacts(run_dir, config, metrics)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "artifacts_validated": True,
                "completed_at_utc": created_at_utc(),
                "execution_stage": execution_stage,
                "scientific_status": metrics["scientific_status"],
                "status_reason": metrics["status_reason"],
                "runtime_seconds": runtime,
                "root_config_sha256": sha256_file(config_path),
                "persisted_config_sha256": sha256_file(run_dir / "config.yaml"),
                "metadata_sha256": sha256_file(run_dir / "metadata.json"),
                "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                "hdf5_sha256": sha256_file(
                    run_dir / "outputs" / output["hdf5_filename"]
                ),
                **audit,
            },
        )
    except Exception as error:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "execution_stage": execution_stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return run_dir


def main() -> None:
    args = _parse_args()
    print(run(args.config.resolve(), stage=args.stage))


if __name__ == "__main__":
    main()
