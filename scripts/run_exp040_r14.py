"""Run the pre-registered exp040 R14 bounded-memory solver scaling audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.helmholtz_axisymmetric import (  # noqa: E402
    PeakRSSMonitor,
    solve_sparse_direct,
)
from tgv_ptycho.forward.helmholtz_benchmarks import (  # noqa: E402
    axial_plane_wave_pml_benchmark,
    axisymmetric_modal_nodal_error,
    make_axisymmetric_pml_modal_problem,
)
from tgv_ptycho.forward.helmholtz_iterative import (  # noqa: E402
    build_csl_ilu_preconditioner,
    build_two_level_ras_csl_preconditioner,
    solve_restarted_gmres,
    sparse_storage_bytes,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.viz.plot_exp040_r14 import (  # noqa: E402
    EXP040_R14_FIGURE_FILENAMES,
    save_exp040_r14_figures,
)

REGISTERED_CONFIG_SHA256 = "TO_BE_LOCKED_AFTER_PREFLIGHT"
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require_exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} differs from the R14 registration.")


def scientific_contract_sha256(config: Mapping[str, Any]) -> str:
    """Hash only the pre-registered R14 scientific sections."""

    contract = {
        key: config[key]
        for key in (
            "axial_pml",
            "solver_scaling",
            "solvers",
            "memory_projection",
            "thresholds",
            "conditional_execution",
        )
    }
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_r14_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the registered R14 scientific contract."""

    _require_exact(config["experiment"]["id"], "exp040", "experiment.id")
    stage = str(config["experiment"]["stage"])
    if stage not in {"R14", "R14B"}:
        raise ValueError("experiment.stage differs from an R14 registration.")
    _require_exact(
        config["experiment"]["scientific_result"],
        True,
        "experiment.scientific_result",
    )
    axial = config["axial_pml"]
    _require_exact(
        axial["fixed_case_order"],
        ["air_upward", "glass_downward"],
        "axial case order",
    )
    if stage == "R14B":
        _require_exact(
            axial["reuse_registered_controls"],
            True,
            "axial reuse flag",
        )
        _require_exact(
            axial["cases"]["air_upward"]["source"],
            "initial_r14_preflight_metrics",
            "air control source",
        )
        _require_exact(
            axial["cases"]["glass_downward"]["source"],
            "r14a_glass_h24_pml2_checkpoint",
            "glass control source",
        )
    scaling = config["solver_scaling"]
    _require_exact(scaling["fixed_discretization_source"], "R13_h0p5_p4", "h,p")
    _require_exact(scaling["degree"], 4, "degree")
    _require_exact(scaling["element_size_ratio"], 0.5, "element ratio")
    _require_exact(scaling["operator_wavenumber"], 8.85787, "operator k")
    _require_exact(
        scaling["fixed_case_order"],
        ["core4", "core8", "core16", "core32", "core64"],
        "scaling case order",
    )
    _require_exact(
        scaling["modal_families"],
        ["nearest_primary", "next_index_offset"],
        "modal families",
    )
    _require_exact(
        [
            scaling["cases"][case]["expected_active_unknowns"]
            for case in scaling["fixed_case_order"]
        ],
        [1880, 5688, 19448, 71544, 274040],
        "active unknown sequence",
    )
    solvers = config["solvers"]
    _require_exact(
        solvers["fixed_order"],
        ["csl_ilu_gmres", "two_level_ras_csl_gmres"],
        "solver order",
    )
    _require_exact(solvers["gmres"]["restart"], 40, "GMRES restart")
    _require_exact(
        solvers["gmres"]["maximum_inner_iterations"], 400, "GMRES maximum"
    )
    _require_exact(
        solvers["complex_shifted_laplacian"]["imaginary_mass_shift"],
        0.5,
        "CSL shift",
    )
    _require_exact(
        solvers["csl_ilu_gmres"]["full_lu_forbidden"], True, "full ILU flag"
    )
    _require_exact(
        solvers["two_level_ras_csl_gmres"]["core_block_shape_nodes"],
        [64, 64],
        "RAS block",
    )
    _require_exact(
        config["memory_projection"]["coarse_projection_rule"],
        "loglog_least_squares_all_scales_with_largest_linear_floor",
        "coarse projection rule",
    )
    conditional = config["conditional_execution"]
    for key in (
        "full_tgv_execution_enabled",
        "rerun_r12",
        "rerun_r13",
        "rerun_cartesian",
        "scalar_cross_model_enabled",
        "vector_model_enabled",
    ):
        _require_exact(conditional[key], False, key)
    expected_thresholds = {
        "solve_true_relative_residual_max": 1.0e-8,
        "analytic_field_weighted_relative_l2_max": 1.0e-2,
        "direct_agreement_relative_l2_max": 1.0e-6,
        "maximum_gmres_inner_iterations": 300,
        "maximum_iteration_growth_core64_over_core8": 4.0,
        "maximum_projected_peak_gib": 10.0,
        "axial_pml_incoming_to_outgoing_ratio_max": 1.0e-3,
        "axial_pml_outgoing_impedance_residual_max": 1.0e-3,
        "axial_pml_dense_field_relative_l2_max": 1.0e-3,
        "require_all_finite": True,
    }
    _require_exact(dict(config["thresholds"]), expected_thresholds, "thresholds")
    _require_exact(
        config["output"]["figure_filenames"],
        list(EXP040_R14_FIGURE_FILENAMES),
        "figure filenames",
    )


def _load_and_validate_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    r13_dir = PROJECT_ROOT / str(provenance["r13_run"])
    paths = {
        "r13_metrics": r13_dir / "metrics.json",
        "r13_hdf5": r13_dir / "outputs" / "exp040_r13.h5",
        "r12_cartesian_checkpoint": PROJECT_ROOT
        / str(provenance["r12_cartesian_checkpoint"]),
    }
    expected = {
        "r13_metrics": str(provenance["r13_metrics_sha256"]),
        "r13_hdf5": str(provenance["r13_hdf5_sha256"]),
        "r12_cartesian_checkpoint": str(
            provenance["r12_cartesian_checkpoint_sha256"]
        ),
    }
    actual = {key: _sha256(path) for key, path in paths.items()}
    if actual != expected:
        raise ValueError("R13/R12 provenance differs from R14 registration.")
    with paths["r13_metrics"].open("r", encoding="utf-8") as handle:
        r13_metrics = json.load(handle)
    if r13_metrics["physical_k_pollution"]["selected_candidate_id"] != str(
        provenance["r13_selected_candidate"]
    ):
        raise ValueError("R13 selected candidate differs.")
    if r13_metrics["physical_k_pollution"][
        "selected_candidate_projected_unknowns"
    ] != int(provenance["r13_selected_candidate_projected_unknowns"]):
        raise ValueError("R13 projected unknown count differs.")
    required_preflight = (
        "preflight_run",
        "preflight_metrics_sha256",
        "preflight_hdf5_sha256",
        "scientific_contract_sha256",
    )
    if any(key not in provenance for key in required_preflight):
        raise ValueError("R14 formal config has not been locked to preflight.")
    if str(provenance["scientific_contract_sha256"]) != scientific_contract_sha256(
        config
    ):
        raise ValueError("R14 scientific contract differs from preflight lock.")
    preflight_dir = PROJECT_ROOT / str(provenance["preflight_run"])
    preflight_metrics_path = preflight_dir / "metrics.json"
    preflight_hdf5_path = preflight_dir / "outputs" / "exp040_r14_preflight.h5"
    if _sha256(preflight_metrics_path) != str(
        provenance["preflight_metrics_sha256"]
    ):
        raise ValueError("R14 preflight metrics hash differs.")
    if _sha256(preflight_hdf5_path) != str(
        provenance["preflight_hdf5_sha256"]
    ):
        raise ValueError("R14 preflight HDF5 hash differs.")
    with preflight_metrics_path.open("r", encoding="utf-8") as handle:
        preflight_metrics = json.load(handle)
    if preflight_metrics["formal_r14_allowed"] is not True:
        raise ValueError("R14 preflight did not authorize formal execution.")
    return {
        "hashes": {
            **actual,
            "preflight_metrics": _sha256(preflight_metrics_path),
            "preflight_hdf5": _sha256(preflight_hdf5_path),
        },
        "r13_metrics": r13_metrics,
        "preflight_metrics": preflight_metrics,
    }


_AXIAL_ARRAY_KEYS = {
    "measurement_coordinates_m",
    "incoming_to_outgoing_ratio",
    "outgoing_impedance_residual",
    "dense_coordinates_m",
    "dense_field",
    "dense_truth",
}
_ERROR_ARRAY_KEYS = {
    "radial_coordinates",
    "center_numerical_trace",
    "center_truth_trace",
}


def _without_keys(value: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in keys}


def _run_axial_pml(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    axial = config["axial_pml"]
    metrics: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    for case_id in axial["fixed_case_order"]:
        case = axial["cases"][case_id]
        result = axial_plane_wave_pml_benchmark(
            wavelength_m=float(axial["wavelength_m"]),
            refractive_index=float(case["refractive_index"]),
            direction=int(case["direction"]),
            physical_core_length_m=float(axial["physical_core_length_m"]),
            pml_thickness_m=float(axial["pml_thickness_m"]),
            element_size_m=float(axial["element_size_m"]),
            degree=int(axial["degree"]),
            quadrature_order=int(axial["quadrature_order"]),
            pml_polynomial_order=int(axial["polynomial_order"]),
            pml_target_one_way_amplitude=float(
                axial["target_one_way_amplitude"]
            ),
            measurement_fractions=np.asarray(
                axial["measurement_fractions"], dtype=np.float64
            ),
            dense_comparison_count=int(axial["dense_comparison_count"]),
        )
        metrics[case_id] = _without_keys(result, _AXIAL_ARRAY_KEYS)
        arrays[case_id] = {key: result[key] for key in _AXIAL_ARRAY_KEYS}
    return metrics, arrays


def _load_reused_axial_pml(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load hash-locked R14B axial controls without recomputing them."""

    provenance = config["provenance"]
    initial_dir = PROJECT_ROOT / str(provenance["initial_preflight_run"])
    initial_metrics_path = initial_dir / "metrics.json"
    initial_hdf5_path = (
        initial_dir / "outputs" / "exp040_r14_preflight.h5"
    )
    r14a_dir = PROJECT_ROOT / str(provenance["r14a_run"])
    r14a_metrics_path = r14a_dir / "metrics.json"
    r14a_hdf5_path = r14a_dir / "outputs" / "exp040_r14a.h5"
    checkpoint_path = r14a_dir / "checkpoints" / "glass_h24_pml2.npz"
    expected_hashes = {
        "initial_preflight_metrics": str(
            provenance["initial_preflight_metrics_sha256"]
        ),
        "initial_preflight_hdf5": str(
            provenance["initial_preflight_hdf5_sha256"]
        ),
        "r14a_metrics": str(provenance["r14a_metrics_sha256"]),
        "r14a_hdf5": str(provenance["r14a_hdf5_sha256"]),
        "r14a_checkpoint": str(provenance["r14a_checkpoint_sha256"]),
    }
    actual_hashes = {
        "initial_preflight_metrics": _sha256(initial_metrics_path),
        "initial_preflight_hdf5": _sha256(initial_hdf5_path),
        "r14a_metrics": _sha256(r14a_metrics_path),
        "r14a_hdf5": _sha256(r14a_hdf5_path),
        "r14a_checkpoint": _sha256(checkpoint_path),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("R14B axial-control provenance differs.")
    with initial_metrics_path.open("r", encoding="utf-8") as handle:
        initial = json.load(handle)
    with r14a_metrics_path.open("r", encoding="utf-8") as handle:
        r14a = json.load(handle)
    if initial["gates"]["axial_pml_smoke_pass"] is not False:
        raise ValueError("R14B initial axial provenance differs.")
    if r14a["corrected_axial_control_eligible"] is not True:
        raise ValueError("R14A did not authorize corrected axial reuse.")
    checkpoint = np.load(checkpoint_path, allow_pickle=False)
    if str(checkpoint["case_id"]) != "glass_h24_pml2":
        raise ValueError("R14A checkpoint case differs.")
    checkpoint_metrics = json.loads(str(checkpoint["metrics_json"]))
    r14a_metrics = r14a["new_cases"]["glass_h24_pml2"]
    if checkpoint_metrics != r14a_metrics:
        raise ValueError("R14A checkpoint metrics differ from its run metrics.")
    arrays = {
        "glass_downward": {
            key: np.asarray(checkpoint[key]) for key in _AXIAL_ARRAY_KEYS
        }
    }
    if not all(
        np.all(np.isfinite(value))
        for value in arrays["glass_downward"].values()
    ):
        raise ValueError("R14A checkpoint contains non-finite axial arrays.")
    metrics = {
        "air_upward": dict(
            initial["axial_pml_smoke"]["cases"]["air_upward"]
        ),
        "glass_downward": dict(r14a_metrics),
    }
    return metrics, arrays


def _modal_problem_kwargs(
    config: Mapping[str, Any], case_id: str, offset: int
) -> dict[str, Any]:
    scaling = config["solver_scaling"]
    return {
        "degree": int(scaling["degree"]),
        "element_size": float(scaling["element_size_ratio"]),
        "core_extent": float(scaling["cases"][case_id]["core_extent"]),
        "pml_thickness": float(scaling["pml_thickness"]),
        "operator_wavenumber": float(scaling["operator_wavenumber"]),
        "pml_polynomial_order": int(scaling["pml_polynomial_order"]),
        "pml_target_one_way_amplitude": float(
            scaling["pml_target_one_way_amplitude"]
        ),
        "interface_fraction_of_core_radius": float(
            scaling["interface_fraction_of_core_radius"]
        ),
        "interface_inner_n2": float(scaling["interface_inner_n2"]),
        "interface_outer_n2": float(scaling["interface_outer_n2"]),
        "target_radial_modal_wavenumber": float(
            scaling["target_radial_modal_wavenumber"]
        ),
        "target_axial_modal_wavenumber": float(
            scaling["target_axial_modal_wavenumber"]
        ),
        "modal_index_offset": int(offset),
        "quadrature_order": max(12, 2 * int(scaling["degree"]) + 2),
    }


def _build_preconditioner(
    solver_id: str,
    shifted_matrix: Any,
    active_shape: tuple[int, int],
    solver_config: Mapping[str, Any],
):
    if solver_id == "csl_ilu_gmres":
        settings = solver_config[solver_id]
        return build_csl_ilu_preconditioner(
            shifted_matrix,
            drop_tolerance=float(settings["drop_tolerance"]),
            fill_factor=float(settings["fill_factor"]),
            drop_rule=str(settings["drop_rule"]),
            permc_spec=str(settings["permc_spec"]),
        )
    if solver_id == "two_level_ras_csl_gmres":
        settings = solver_config[solver_id]
        return build_two_level_ras_csl_preconditioner(
            shifted_matrix,
            active_shape=active_shape,
            core_block_shape_nodes=tuple(settings["core_block_shape_nodes"]),
            overlap_nodes=int(settings["overlap_nodes"]),
        )
    raise ValueError(f"unknown R14 solver: {solver_id}")


def _relative_l2(test: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(test - reference)
        / max(np.linalg.norm(reference), np.finfo(float).eps)
    )


def _run_solver(
    config: Mapping[str, Any],
    solver_id: str,
    original_problem: Mapping[str, Any],
    offset_problem: Mapping[str, Any],
    shifted_matrix: Any,
    direct_solutions: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    solver_config = config["solvers"]
    gmres_config = solver_config["gmres"]
    active_shape = tuple(original_problem["controls"]["active_shape"])
    with PeakRSSMonitor() as monitor:
        try:
            preconditioner = _build_preconditioner(
                solver_id, shifted_matrix, active_shape, solver_config
            )
        except Exception as error:
            return (
                {
                    "setup_succeeded": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "process_peak_rss_bytes": monitor.peak_rss_bytes,
                },
                {},
            )
        modal_metrics: dict[str, Any] = {}
        modal_arrays: dict[str, Any] = {}
        for family, problem in (
            ("nearest_primary", original_problem),
            ("next_index_offset", offset_problem),
        ):
            try:
                solution, gmres_controls = solve_restarted_gmres(
                    original_problem["matrix"],
                    problem["rhs"],
                    preconditioner.operator,
                    relative_tolerance=float(gmres_config["relative_tolerance"]),
                    absolute_tolerance=float(gmres_config["absolute_tolerance"]),
                    restart=int(gmres_config["restart"]),
                    maximum_inner_iterations=int(
                        gmres_config["maximum_inner_iterations"]
                    ),
                )
                error_controls = axisymmetric_modal_nodal_error(
                    solution,
                    problem["grid"],
                    problem["exact_field"],
                    core_extent=float(problem["controls"]["core_extent"]),
                )
                direct_agreement = (
                    None
                    if family not in direct_solutions
                    else _relative_l2(solution, direct_solutions[family])
                )
                modal_metrics[family] = {
                    "solve_succeeded": True,
                    "gmres": gmres_controls,
                    "analytic_weighted_relative_l2": error_controls[
                        "weighted_relative_l2"
                    ],
                    "direct_agreement_relative_l2": direct_agreement,
                    "all_finite": bool(
                        gmres_controls["all_finite"]
                        and error_controls["all_finite"]
                    ),
                }
                modal_arrays[family] = {
                    key: error_controls[key] for key in _ERROR_ARRAY_KEYS
                }
                modal_arrays[family]["preconditioned_residual_history"] = (
                    gmres_controls["preconditioned_residual_history"]
                )
            except Exception as error:
                modal_metrics[family] = {
                    "solve_succeeded": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                modal_arrays[family] = {}
    preconditioner_controls = dict(preconditioner.controls)
    maximum_krylov = max(
        (
            int(value["gmres"]["krylov_basis_storage_bytes"])
            for value in modal_metrics.values()
            if value.get("solve_succeeded") is True
        ),
        default=0,
    )
    original_bytes = sparse_storage_bytes(original_problem["matrix"])
    shifted_bytes = sparse_storage_bytes(shifted_matrix)
    if solver_id == "csl_ilu_gmres":
        setup_bytes = (
            original_bytes
            + shifted_bytes
            + int(preconditioner_controls["factor_storage_bytes"])
        )
        solve_bytes = (
            original_bytes
            + int(preconditioner_controls["factor_storage_bytes"])
            + maximum_krylov
        )
    else:
        setup_bytes = original_bytes + int(
            preconditioner_controls["total_preconditioner_storage_bytes"]
        )
        solve_bytes = setup_bytes + maximum_krylov
    metrics = {
        "setup_succeeded": True,
        "preconditioner": preconditioner_controls,
        "modal_results": modal_metrics,
        "storage": {
            "original_matrix_bytes": original_bytes,
            "shifted_matrix_bytes": shifted_bytes,
            "maximum_krylov_basis_bytes": maximum_krylov,
            "conservative_peak_model_bytes": max(setup_bytes, solve_bytes),
        },
        "process_peak_rss_bytes": monitor.peak_rss_bytes,
    }
    del preconditioner
    gc.collect()
    return metrics, modal_arrays


def _project_solver_memory(
    config: Mapping[str, Any],
    solver_id: str,
    cases: Mapping[str, Any],
) -> dict[str, Any]:
    projection = config["memory_projection"]
    full_unknowns = int(projection["full_tgv_active_unknowns"])
    largest_unknowns = int(
        config["solver_scaling"]["cases"]["core64"]["expected_active_unknowns"]
    )
    safety = float(projection["safety_factor"])
    largest = cases["core64"]["solvers"][solver_id]
    storage = largest["storage"]
    if solver_id == "csl_ilu_gmres":
        projected_bytes_before_safety = (
            float(storage["conservative_peak_model_bytes"])
            * full_unknowns
            / largest_unknowns
        )
        coarse_projection = 0.0
    else:
        controls = largest["preconditioner"]
        coarse_bytes = float(controls["coarse_factor_storage_bytes"])
        noncoarse_largest = (
            float(storage["conservative_peak_model_bytes"]) - coarse_bytes
        )
        full_shape = tuple(int(value) for value in projection["full_tgv_active_shape"])
        block_shape = tuple(
            int(value)
            for value in config["solvers"][solver_id]["core_block_shape_nodes"]
        )
        full_blocks = math.ceil(full_shape[0] / block_shape[0]) * math.ceil(
            full_shape[1] / block_shape[1]
        )
        largest_blocks = int(controls["block_count"])
        block_counts = np.asarray(
            [
                cases[case_id]["solvers"][solver_id]["preconditioner"][
                    "block_count"
                ]
                for case_id in config["solver_scaling"]["fixed_case_order"]
            ],
            dtype=np.float64,
        )
        coarse_factor_bytes = np.asarray(
            [
                cases[case_id]["solvers"][solver_id]["preconditioner"][
                    "coarse_factor_storage_bytes"
                ]
                for case_id in config["solver_scaling"]["fixed_case_order"]
            ],
            dtype=np.float64,
        )
        design = np.column_stack(
            (np.ones(block_counts.size), np.log(block_counts))
        )
        intercept, exponent = np.linalg.lstsq(
            design, np.log(coarse_factor_bytes), rcond=None
        )[0]
        fitted_coarse_projection = float(
            np.exp(intercept + exponent * np.log(full_blocks))
        )
        largest_linear_projection = float(
            coarse_bytes * full_blocks / largest_blocks
        )
        coarse_projection = max(
            fitted_coarse_projection, largest_linear_projection
        )
        projected_bytes_before_safety = (
            noncoarse_largest * full_unknowns / largest_unknowns
            + coarse_projection
        )
    projected_bytes = safety * projected_bytes_before_safety
    return {
        "largest_case_peak_model_bytes": int(
            storage["conservative_peak_model_bytes"]
        ),
        "full_tgv_active_unknowns": full_unknowns,
        "coarse_fill_projected_bytes_before_safety": float(coarse_projection),
        "coarse_fit_exponent": (
            None if solver_id == "csl_ilu_gmres" else float(exponent)
        ),
        "coarse_fit_prediction_bytes": (
            0.0
            if solver_id == "csl_ilu_gmres"
            else float(fitted_coarse_projection)
        ),
        "coarse_largest_linear_floor_bytes": (
            0.0
            if solver_id == "csl_ilu_gmres"
            else float(largest_linear_projection)
        ),
        "safety_factor": safety,
        "projected_peak_bytes": float(projected_bytes),
        "projected_peak_gib": float(projected_bytes / 1024**3),
    }


def _summarize_solver(
    config: Mapping[str, Any],
    solver_id: str,
    cases: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["thresholds"]
    case_ids = config["solver_scaling"]["fixed_case_order"]
    families = config["solver_scaling"]["modal_families"]
    setup_all = all(
        cases[case]["solvers"][solver_id].get("setup_succeeded") is True
        for case in case_ids
    )
    if not setup_all:
        return {
            "solver_gate_pass": False,
            "setup_all_scales": False,
            "failure_reason": "preconditioner_setup_failed",
            "projected_peak_gib": None,
            "maximum_largest_case_iterations": None,
            "iteration_growth_core64_over_core8": None,
        }
    modal_results = [
        cases[case]["solvers"][solver_id]["modal_results"][family]
        for case in case_ids
        for family in families
    ]
    solve_all = all(value.get("solve_succeeded") is True for value in modal_results)
    if not solve_all:
        return {
            "solver_gate_pass": False,
            "setup_all_scales": True,
            "solve_all_modal_families": False,
            "failure_reason": "gmres_execution_failed",
            "projected_peak_gib": None,
            "maximum_largest_case_iterations": None,
            "iteration_growth_core64_over_core8": None,
        }
    convergence_pass = all(
        value["gmres"]["converged"]
        and value["gmres"]["true_relative_residual"]
        <= float(thresholds["solve_true_relative_residual_max"])
        and value["gmres"]["inner_iteration_count"]
        <= int(thresholds["maximum_gmres_inner_iterations"])
        and value["all_finite"]
        for value in modal_results
    )
    accuracy_pass = all(
        value["analytic_weighted_relative_l2"]
        <= float(thresholds["analytic_field_weighted_relative_l2_max"])
        for value in modal_results
    )
    direct_pass = all(
        value["direct_agreement_relative_l2"] is None
        or value["direct_agreement_relative_l2"]
        <= float(thresholds["direct_agreement_relative_l2_max"])
        for value in modal_results
    )
    core8_iterations = max(
        cases["core8"]["solvers"][solver_id]["modal_results"][family]["gmres"]
        ["inner_iteration_count"]
        for family in families
    )
    core64_iterations = max(
        cases["core64"]["solvers"][solver_id]["modal_results"][family]["gmres"]
        ["inner_iteration_count"]
        for family in families
    )
    iteration_growth = float(core64_iterations / max(core8_iterations, 1))
    iteration_growth_pass = bool(
        iteration_growth
        <= float(thresholds["maximum_iteration_growth_core64_over_core8"])
    )
    projection = _project_solver_memory(config, solver_id, cases)
    memory_pass = bool(
        projection["projected_peak_gib"]
        <= float(thresholds["maximum_projected_peak_gib"])
        and max(
            cases[case]["solvers"][solver_id]["process_peak_rss_bytes"]
            for case in case_ids
        )
        <= float(config["memory_projection"]["maximum_formal_process_peak_gib"])
        * 1024**3
    )
    solver_gate = bool(
        convergence_pass
        and accuracy_pass
        and direct_pass
        and iteration_growth_pass
        and memory_pass
    )
    return {
        "solver_gate_pass": solver_gate,
        "setup_all_scales": True,
        "solve_all_modal_families": True,
        "convergence_pass": convergence_pass,
        "accuracy_pass": accuracy_pass,
        "direct_agreement_pass": direct_pass,
        "iteration_growth_pass": iteration_growth_pass,
        "memory_pass": memory_pass,
        "maximum_largest_case_iterations": core64_iterations,
        "iteration_growth_core64_over_core8": iteration_growth,
        **projection,
    }


def _execute(
    config: Mapping[str, Any], progress: ProgressCallback
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = config["thresholds"]
    if config["axial_pml"].get("reuse_registered_controls") is True:
        axial_metrics, axial_arrays = _load_reused_axial_pml(config)
        axial_source = "hash_locked_provenance_reuse"
    else:
        axial_metrics, axial_arrays = _run_axial_pml(config)
        axial_source = "computed_in_formal_run"
    axial_case_pass = {
        case_id: bool(
            value["maximum_incoming_to_outgoing_ratio"]
            <= float(thresholds["axial_pml_incoming_to_outgoing_ratio_max"])
            and value["maximum_outgoing_impedance_residual"]
            <= float(thresholds["axial_pml_outgoing_impedance_residual_max"])
            and value["dense_field_relative_l2"]
            <= float(thresholds["axial_pml_dense_field_relative_l2_max"])
            and value["solver_controls"]["relative_residual"]
            <= float(thresholds["solve_true_relative_residual_max"])
            and value["all_finite"]
        )
        for case_id, value in axial_metrics.items()
    }
    axial_gate = bool(all(axial_case_pass.values()))
    progress("axial_pml_completed", {"gate_pass": axial_gate})

    scaling = config["solver_scaling"]
    solver_ids = config["solvers"]["fixed_order"]
    case_metrics: dict[str, Any] = {}
    solver_arrays: dict[str, Any] = {}
    maximum_process_peak = 0
    all_matrix_finite = True
    maximum_matrix_repeat_error = 0.0
    for case_id in scaling["fixed_case_order"]:
        progress("scaling_case_started", {"case_id": case_id})
        primary = make_axisymmetric_pml_modal_problem(
            **_modal_problem_kwargs(config, case_id, 0)
        )
        offset = make_axisymmetric_pml_modal_problem(
            **_modal_problem_kwargs(config, case_id, 1)
        )
        shifted = make_axisymmetric_pml_modal_problem(
            **_modal_problem_kwargs(config, case_id, 0),
            imaginary_mass_shift=float(
                config["solvers"]["complex_shifted_laplacian"][
                    "imaginary_mass_shift"
                ]
            ),
        )
        expected_unknowns = int(
            scaling["cases"][case_id]["expected_active_unknowns"]
        )
        if primary["matrix"].shape != (expected_unknowns, expected_unknowns):
            raise RuntimeError(f"{case_id} active unknown count differs.")
        matrix_difference = primary["matrix"] - offset["matrix"]
        matrix_repeat_error = (
            0.0
            if matrix_difference.nnz == 0
            else float(np.max(np.abs(matrix_difference.data)))
        )
        maximum_matrix_repeat_error = max(
            maximum_matrix_repeat_error, matrix_repeat_error
        )
        all_matrix_finite = bool(
            all_matrix_finite
            and primary["controls"]["all_finite"]
            and offset["controls"]["all_finite"]
            and shifted["controls"]["all_finite"]
        )
        direct_solutions: dict[str, np.ndarray] = {}
        direct_controls: dict[str, Any] = {}
        if case_id in scaling["direct_verification_cases"]:
            for family, problem in (
                ("nearest_primary", primary),
                ("next_index_offset", offset),
            ):
                direct, controls = solve_sparse_direct(
                    primary["matrix"], problem["rhs"]
                )
                direct_solutions[family] = direct
                direct_controls[family] = controls
        solvers_for_case: dict[str, Any] = {}
        arrays_for_case: dict[str, Any] = {}
        for solver_id in solver_ids:
            progress(
                "solver_case_started",
                {"case_id": case_id, "solver_id": solver_id},
            )
            solver_metrics, modal_arrays = _run_solver(
                config,
                solver_id,
                primary,
                offset,
                shifted["matrix"],
                direct_solutions,
            )
            solvers_for_case[solver_id] = solver_metrics
            arrays_for_case[solver_id] = modal_arrays
            maximum_process_peak = max(
                maximum_process_peak,
                int(solver_metrics.get("process_peak_rss_bytes", 0)),
            )
            progress(
                "solver_case_completed",
                {
                    "case_id": case_id,
                    "solver_id": solver_id,
                    "setup_succeeded": solver_metrics["setup_succeeded"],
                },
            )
        case_metrics[case_id] = {
            "core_extent": float(scaling["cases"][case_id]["core_extent"]),
            "active_unknowns": expected_unknowns,
            "active_shape": primary["controls"]["active_shape"],
            "matrix_nnz": int(primary["matrix"].nnz),
            "matrix_storage_bytes": sparse_storage_bytes(primary["matrix"]),
            "matrix_repeat_max_abs_error": matrix_repeat_error,
            "primary_mode_controls": _without_keys(
                primary["controls"], {"assembly_controls"}
            ),
            "offset_mode_controls": _without_keys(
                offset["controls"], {"assembly_controls"}
            ),
            "assembly_controls": primary["controls"]["assembly_controls"],
            "direct_controls": direct_controls,
            "solvers": solvers_for_case,
        }
        solver_arrays[case_id] = arrays_for_case
        del primary, offset, shifted, direct_solutions
        gc.collect()
        progress("scaling_case_completed", {"case_id": case_id})

    solver_summaries = {
        solver_id: _summarize_solver(config, solver_id, case_metrics)
        for solver_id in solver_ids
    }
    passing_solvers = [
        solver_id
        for solver_id in solver_ids
        if solver_summaries[solver_id]["solver_gate_pass"]
    ]
    selected_solver = passing_solvers[0] if passing_solvers else None
    solver_candidate_found = selected_solver is not None
    hard_controls_pass = bool(
        all_matrix_finite
        and maximum_matrix_repeat_error <= 1.0e-12
        and maximum_process_peak
        <= float(config["memory_projection"]["maximum_formal_process_peak_gib"])
        * 1024**3
    )
    benchmark_passed = bool(
        axial_gate and solver_candidate_found and hard_controls_pass
    )
    if benchmark_passed:
        interpretation = "r14_solver_closed__full_reference_preregistration_allowed"
    elif not axial_gate:
        interpretation = "r14_axial_pml_not_closed"
    elif not solver_candidate_found:
        interpretation = "r14_no_scalable_scipy_solver"
    else:
        interpretation = "r14_hard_controls_failed"
    metrics = {
        "version": str(config["experiment"]["stage"]),
        "scientific_result": True,
        "status": "Passed" if benchmark_passed else "Failed",
        "interpretation_code": interpretation,
        "benchmark_validated": benchmark_passed,
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "next_full_reference_preregistration_allowed": benchmark_passed,
        "gates": {
            "axial_pml_gate_pass": axial_gate,
            "solver_candidate_found": solver_candidate_found,
            "hard_controls_pass": hard_controls_pass,
        },
        "hard_controls": {
            "all_matrix_finite": all_matrix_finite,
            "maximum_matrix_repeat_max_abs_error": maximum_matrix_repeat_error,
            "maximum_process_peak_rss_bytes": maximum_process_peak,
        },
        "axial_pml": {
            "control_source": axial_source,
            "fixed_case_order": list(config["axial_pml"]["fixed_case_order"]),
            "cases": axial_metrics,
            "case_gate_pass": axial_case_pass,
        },
        "solver_scaling": {
            "fixed_case_order": list(scaling["fixed_case_order"]),
            "fixed_solver_order": list(solver_ids),
            "modal_families": list(scaling["modal_families"]),
            "cases": case_metrics,
            "solver_summaries": solver_summaries,
            "passing_solver_ids": passing_solvers,
            "selected_solver_id": selected_solver,
        },
        "conditional_execution": {
            "full_tgv_executed": False,
            "r12_or_r13_rerun": False,
            "cartesian_executed": False,
            "scalar_cross_model_executed": False,
            "vector_model_executed": False,
        },
        "thresholds": dict(thresholds),
    }
    arrays = {
        "axial_pml": axial_arrays,
        "solver_scaling": solver_arrays,
    }
    return metrics, arrays


def _hdf5_safe(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(key): _hdf5_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return np.asarray([], dtype=np.float64)
        if all(isinstance(item, str) for item in value):
            return list(value)
        return [_hdf5_safe(item) for item in value]
    return value


def _write_hdf5(
    path: Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> None:
    truth = {
        "axial_pml": {
            case_id: {
                "coordinates_m": values["dense_coordinates_m"],
                "analytic_field": values["dense_truth"],
            }
            for case_id, values in arrays["axial_pml"].items()
        },
        "solver_scaling": {
            case_id: {
                solver_id: {
                    family: {
                        "radial_coordinates": family_values.get(
                            "radial_coordinates", np.asarray([])
                        ),
                        "center_truth_trace": family_values.get(
                            "center_truth_trace", np.asarray([])
                        ),
                    }
                    for family, family_values in solver_values.items()
                }
                for solver_id, solver_values in case_values.items()
            }
            for case_id, case_values in arrays["solver_scaling"].items()
        },
    }
    save_ptycho_hdf5(
        path,
        instrument=_hdf5_safe(
            {
                "axial_pml": config["axial_pml"],
                "solver_scaling": config["solver_scaling"],
                "solvers": config["solvers"],
            }
        ),
        sample=_hdf5_safe(
            {
                "kind": "analytic_helmholtz_solver_benchmark",
                "contains_canonical_tgv_field": False,
            }
        ),
        truth=_hdf5_safe(truth),
        config_yaml=config_to_yaml(dict(config)),
        metadata=_hdf5_safe(dict(metadata)),
        metrics=_hdf5_safe(dict(metrics)),
    )
    with h5py.File(path, "a") as handle:
        data = handle["entry/data"]
        axial_group = data.require_group("axial_pml")
        for case_id, values in arrays["axial_pml"].items():
            group = axial_group.require_group(case_id)
            for key in (
                "dense_coordinates_m",
                "dense_field",
                "measurement_coordinates_m",
                "incoming_to_outgoing_ratio",
                "outgoing_impedance_residual",
            ):
                group.create_dataset(key, data=np.asarray(values[key]))
        scaling_group = data.require_group("solver_scaling")
        for case_id, case_values in arrays["solver_scaling"].items():
            case_group = scaling_group.require_group(case_id)
            for solver_id, solver_values in case_values.items():
                solver_group = case_group.require_group(solver_id)
                for family, values in solver_values.items():
                    family_group = solver_group.require_group(family)
                    for key, value in values.items():
                        family_group.create_dataset(key, data=np.asarray(value))


def _progress_writer(path: Path) -> ProgressCallback:
    events: list[dict[str, Any]] = []

    def write(event: str, details: Mapping[str, Any]) -> None:
        events.append(
            {"time": created_at_utc(), "event": event, "details": dict(details)}
        )
        save_json(path, {"events": events})

    return write


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    for relative in config["output"]["required_files"]:
        path = run_dir / str(relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty R14 artifact: {relative}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["scientific_result"] is not True:
        raise RuntimeError("R14 metrics lost scientific-result status.")
    if metrics["conditional_execution"]["full_tgv_executed"] is not False:
        raise RuntimeError("R14 unexpectedly ran full TGV.")
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as handle:
        entry = handle["entry"]
        expected = {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        if set(entry) != expected:
            raise RuntimeError("R14 HDF5 entry layout differs.")
        if set(entry["data"]) != {"axial_pml", "solver_scaling"}:
            raise RuntimeError("R14 HDF5 data layout differs.")
        if "reconstruction" in entry:
            raise RuntimeError("R14 must not write reconstruction data.")
    for filename in config["output"]["figure_filenames"]:
        image = np.asarray(iio.imread(run_dir / "figures" / str(filename)))
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 100:
            raise RuntimeError(f"invalid R14 figure: {filename}")
        if not np.all(np.isfinite(image)):
            raise RuntimeError(f"non-finite R14 figure: {filename}")


def run(
    config_path: Path,
    *,
    registered_config_sha256: str | None = None,
) -> Path:
    """Execute the single formal R14 scientific run."""

    registered_hash = (
        REGISTERED_CONFIG_SHA256
        if registered_config_sha256 is None
        else str(registered_config_sha256)
    )
    source = config_path.resolve()
    if _sha256(source) != registered_hash:
        raise ValueError("R14 source config hash differs.")
    config = load_config(source)
    validate_r14_config(config)
    stage = str(config["experiment"]["stage"])
    provenance = _load_and_validate_provenance(config)
    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    state_path = run_dir / "run_state.json"
    progress = _progress_writer(run_dir / "run_progress.json")
    started = time.perf_counter()
    try:
        save_json(
            state_path,
            {
                "stage": stage,
                "state": "running",
                "scientific_result": True,
                "formal_execution_count": 1,
                "created_at": created_at_utc(),
            },
        )
        progress(
            "formal_execution_started",
            {
                "config_sha256": registered_hash,
                "scientific_contract_sha256": scientific_contract_sha256(config),
            },
        )
        metrics, arrays = _execute(config, progress)
        metrics["total_execution_elapsed_s"] = float(time.perf_counter() - started)
        metrics["formal_provenance_hashes"] = provenance["hashes"]
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": stage,
            "scientific_result": True,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit(PROJECT_ROOT),
            "source_config_sha256": registered_hash,
            "scientific_contract_sha256": scientific_contract_sha256(config),
        }
        save_config(run_dir / "config.yaml", dict(config))
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        _write_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            config,
            metadata,
            metrics,
            arrays,
        )
        save_exp040_r14_figures(run_dir / "figures", metrics, arrays)
        save_json(
            state_path,
            {
                "stage": stage,
                "state": "completed",
                "scientific_result": True,
                "formal_execution_count": 1,
                "status": metrics["status"],
                "interpretation_code": metrics["interpretation_code"],
                "selected_solver_id": metrics["solver_scaling"][
                    "selected_solver_id"
                ],
                "full_tgv_reference_authorized": False,
                "completed_at": created_at_utc(),
            },
        )
        progress(
            "formal_execution_completed",
            {
                "status": metrics["status"],
                "interpretation_code": metrics["interpretation_code"],
            },
        )
        _validate_artifacts(run_dir, config)
    except Exception:
        save_json(
            state_path,
            {
                "stage": stage,
                "state": "failed_during_execution",
                "scientific_result": True,
                "formal_execution_count": 1,
                "failed_at": created_at_utc(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"run_dir: {run_dir}", flush=True)
    print(f"status: {metrics['status']}", flush=True)
    print(f"interpretation: {metrics['interpretation_code']}", flush=True)
    print(f"gates: {metrics['gates']}", flush=True)
    print(
        f"selected_solver: {metrics['solver_scaling']['selected_solver_id']}",
        flush=True,
    )
    return run_dir


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
