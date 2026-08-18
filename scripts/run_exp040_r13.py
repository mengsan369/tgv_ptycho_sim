"""Run the pre-registered exp040 R13 reflection and pollution benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from tgv_ptycho.forward.helmholtz_axisymmetric_fem import (  # noqa: E402
    make_axisymmetric_fem_grid,
)
from tgv_ptycho.forward.helmholtz_benchmarks import (  # noqa: E402
    annular_outgoing_pml_benchmark,
    physical_k_modal_fem_benchmark,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.viz.plot_exp040_r13 import (  # noqa: E402
    EXP040_R13_FIGURE_FILENAMES,
    save_exp040_r13_figures,
)

REGISTERED_CONFIG_SHA256 = (
    "FB5BEC5C2A41D303E936AADA1A3A43EB896FF7C71E7DE06B737A825944C908B0"
)
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
        raise ValueError(f"{name} differs from the R13 registration.")


def scientific_contract_sha256(config: Mapping[str, Any]) -> str:
    """Hash only the pre-registered scientific sections of an R13 config."""

    contract = {
        key: config[key]
        for key in (
            "domain_reflection",
            "physical_k_pollution",
            "full_tgv_projection",
            "conditional_execution",
            "thresholds",
        )
    }
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_r13_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the registered R13 scientific contract."""

    _require_exact(config["experiment"]["id"], "exp040", "experiment.id")
    _require_exact(config["experiment"]["stage"], "R13", "experiment.stage")
    _require_exact(
        config["experiment"]["scientific_result"],
        True,
        "experiment.scientific_result",
    )
    domain = config["domain_reflection"]
    _require_exact(
        domain["fixed_case_order"],
        ["core48_pml2", "core60_pml2"],
        "domain fixed case order",
    )
    _require_exact(domain["degree"], 4, "domain degree")
    _require_exact(domain["quadrature_order"], 12, "domain quadrature")
    _require_exact(
        domain["common_measurement_radii_m"],
        [2.4e-5, 3.2e-5, 4.0e-5, 4.6e-5],
        "domain measurement radii",
    )
    pollution = config["physical_k_pollution"]
    _require_exact(pollution["formal_kh"], 8.85787, "formal kh")
    _require_exact(
        pollution["fixed_case_order"],
        [
            "h1_p2",
            "h1_p3",
            "h1_p4",
            "h1_p6",
            "h1_p8",
            "h0p5_p2",
            "h0p5_p3",
            "h0p5_p4",
            "h0p5_p6",
            "h0p25_p2",
            "h0p25_p3",
            "h0p25_p4",
        ],
        "pollution fixed case order",
    )
    _require_exact(
        pollution["benchmark_families"],
        ["homogeneous", "glass_air_interface"],
        "benchmark families",
    )
    conditional = config["conditional_execution"]
    for key in (
        "rerun_cartesian",
        "rerun_r12",
        "scalar_cross_model_enabled",
        "vector_model_enabled",
    ):
        _require_exact(conditional[key], False, key)
    _require_exact(
        config["full_tgv_projection"]["execution_enabled"],
        False,
        "full TGV execution",
    )
    thresholds = config["thresholds"]
    expected_thresholds = {
        "solve_relative_residual_max": 1.0e-10,
        "pml_incoming_to_outgoing_ratio_max": 1.0e-3,
        "pml_outgoing_impedance_residual_max": 1.0e-3,
        "pml_dense_field_relative_l2_max": 1.0e-3,
        "pml_flux_relative_range_max": 5.0e-3,
        "pml_pair_dense_field_relative_l2_max": 1.0e-3,
        "physical_k_family_weighted_relative_l2_max": 1.0e-2,
        "require_all_finite": True,
    }
    _require_exact(dict(thresholds), expected_thresholds, "thresholds")
    _require_exact(
        config["output"]["figure_filenames"],
        list(EXP040_R13_FIGURE_FILENAMES),
        "figure filenames",
    )


def _load_and_validate_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    run_dir = PROJECT_ROOT / str(provenance["r12_run"])
    metrics_path = run_dir / "metrics.json"
    hdf5_path = run_dir / "outputs" / "exp040_r12.h5"
    checkpoint_path = PROJECT_ROOT / str(provenance["r12_cartesian_checkpoint"])
    expected = {
        "r12_metrics": str(provenance["r12_metrics_sha256"]),
        "r12_hdf5": str(provenance["r12_hdf5_sha256"]),
        "r12_cartesian_checkpoint": str(
            provenance["r12_cartesian_checkpoint_sha256"]
        ),
    }
    actual = {
        "r12_metrics": _sha256(metrics_path),
        "r12_hdf5": _sha256(hdf5_path),
        "r12_cartesian_checkpoint": _sha256(checkpoint_path),
    }
    if actual != expected:
        raise ValueError("R12 provenance hash differs from the R13 registration.")
    with metrics_path.open("r", encoding="utf-8") as handle:
        r12_metrics = json.load(handle)
    if r12_metrics["gates"]["cartesian_reference_gate_pass"] is not True:
        raise ValueError("R12 Cartesian reference was not closed.")
    if r12_metrics["conditional_cross_model"]["executed"] is not False:
        raise ValueError("R12 cross-model state differs from registration.")
    required_preflight = (
        "preflight_run",
        "preflight_metrics_sha256",
        "preflight_hdf5_sha256",
        "scientific_contract_sha256",
    )
    if any(key not in provenance for key in required_preflight):
        raise ValueError("R13 formal config has not been locked to a preflight.")
    if str(provenance["scientific_contract_sha256"]) != scientific_contract_sha256(
        config
    ):
        raise ValueError("R13 scientific contract differs from preflight lock.")
    preflight_dir = PROJECT_ROOT / str(provenance["preflight_run"])
    preflight_metrics_path = preflight_dir / "metrics.json"
    preflight_hdf5_path = preflight_dir / "outputs" / "exp040_r13_preflight.h5"
    if _sha256(preflight_metrics_path) != str(
        provenance["preflight_metrics_sha256"]
    ):
        raise ValueError("R13 preflight metrics hash differs.")
    if _sha256(preflight_hdf5_path) != str(
        provenance["preflight_hdf5_sha256"]
    ):
        raise ValueError("R13 preflight HDF5 hash differs.")
    with preflight_metrics_path.open("r", encoding="utf-8") as handle:
        preflight_metrics = json.load(handle)
    if preflight_metrics["formal_r13_allowed"] is not True:
        raise ValueError("R13 preflight did not authorize the formal run.")
    return {
        "hashes": {**actual, "preflight_metrics": _sha256(preflight_metrics_path),
                   "preflight_hdf5": _sha256(preflight_hdf5_path)},
        "r12_metrics": r12_metrics,
        "preflight_metrics": preflight_metrics,
    }


def _relative_weighted_radial_l2(
    test: np.ndarray, reference: np.ndarray, radius: np.ndarray
) -> float:
    numerator = float(
        np.trapezoid(radius * np.abs(test - reference) ** 2, radius)
    )
    denominator = float(np.trapezoid(radius * np.abs(reference) ** 2, radius))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).eps)))


_DOMAIN_ARRAY_KEYS = {
    "measurement_radii_m",
    "dense_radii_m",
    "nodal_radii_m",
    "nodal_field",
    "measurement_field",
    "measurement_derivative",
    "dense_field",
    "dense_derivative",
    "dense_truth",
    "incoming_to_outgoing_ratio",
    "outgoing_impedance_residual",
    "radial_flux",
}
_POLLUTION_ARRAY_KEYS = {
    "radial_coordinates",
    "axial_coordinates",
    "numerical_field",
    "truth_field",
}


def _without_arrays(result: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in keys}


def _project_full_tgv_unknowns(
    config: Mapping[str, Any], *, degree: int, element_size_ratio: float
) -> int:
    projection = config["full_tgv_projection"]
    spacing = float(projection["formal_element_size_m"]) * float(
        element_size_ratio
    )
    grid = make_axisymmetric_fem_grid(
        degree=degree,
        radial_extent_m=float(projection["radial_extent_m"]),
        z_min_m=float(projection["z_min_m"]),
        z_max_m=float(projection["z_max_m"]),
        radial_element_size_m=spacing,
        axial_element_size_m=spacing,
    )
    return grid.active_unknown_count


def _execute(
    config: Mapping[str, Any], progress: ProgressCallback
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = config["thresholds"]
    domain_config = config["domain_reflection"]
    measurement = np.asarray(
        domain_config["common_measurement_radii_m"], dtype=np.float64
    )
    dense = np.linspace(
        float(domain_config["dense_comparison_min_m"]),
        float(domain_config["dense_comparison_max_m"]),
        int(domain_config["dense_comparison_count"]),
        dtype=np.float64,
    )
    domain_metrics: dict[str, Any] = {}
    domain_arrays: dict[str, Any] = {}
    all_solver_residuals: list[float] = []
    for case_id in domain_config["fixed_case_order"]:
        case = domain_config["cases"][case_id]
        progress("domain_case_started", {"case_id": case_id})
        result = annular_outgoing_pml_benchmark(
            wavelength_m=float(domain_config["wavelength_m"]),
            refractive_index=float(domain_config["refractive_index"]),
            inner_radius_m=float(domain_config["inner_radius_m"]),
            pml_start_m=float(case["pml_start_m"]),
            pml_thickness_m=float(case["pml_thickness_m"]),
            element_size_m=float(domain_config["element_size_m"]),
            degree=int(domain_config["degree"]),
            quadrature_order=int(domain_config["quadrature_order"]),
            pml_polynomial_order=int(
                domain_config["pml"]["polynomial_order"]
            ),
            pml_target_one_way_amplitude=float(
                domain_config["pml"]["target_one_way_amplitude"]
            ),
            measurement_radii_m=measurement,
            dense_radii_m=dense,
        )
        domain_metrics[case_id] = _without_arrays(result, _DOMAIN_ARRAY_KEYS)
        domain_arrays[case_id] = {
            key: result[key] for key in _DOMAIN_ARRAY_KEYS if key in result
        }
        all_solver_residuals.append(
            float(result["solver_controls"]["relative_residual"])
        )
        progress(
            "domain_case_completed",
            {
                "case_id": case_id,
                "incoming_ratio": result[
                    "maximum_incoming_to_outgoing_ratio"
                ],
                "field_l2": result["dense_field_weighted_relative_l2"],
            },
        )
    first_id, second_id = domain_config["fixed_case_order"]
    domain_pair_l2 = _relative_weighted_radial_l2(
        domain_arrays[first_id]["dense_field"],
        domain_arrays[second_id]["dense_field"],
        dense,
    )
    domain_case_pass = {
        case_id: bool(
            values["maximum_incoming_to_outgoing_ratio"]
            <= float(thresholds["pml_incoming_to_outgoing_ratio_max"])
            and values["maximum_outgoing_impedance_residual"]
            <= float(thresholds["pml_outgoing_impedance_residual_max"])
            and values["dense_field_weighted_relative_l2"]
            <= float(thresholds["pml_dense_field_relative_l2_max"])
            and values["flux_relative_range"]
            <= float(thresholds["pml_flux_relative_range_max"])
            and values["solver_controls"]["relative_residual"]
            <= float(thresholds["solve_relative_residual_max"])
            and values["all_finite"]
        )
        for case_id, values in domain_metrics.items()
    }
    domain_gate = bool(
        all(domain_case_pass.values())
        and domain_pair_l2
        <= float(thresholds["pml_pair_dense_field_relative_l2_max"])
    )

    pollution_config = config["physical_k_pollution"]
    amplitude = complex(*pollution_config["complex_amplitude"])
    pollution_metrics: dict[str, Any] = {}
    pollution_fields: dict[str, dict[str, Any]] = {}
    eligible_ids: list[str] = []
    for case_id in pollution_config["fixed_case_order"]:
        case = pollution_config["cases"][case_id]
        progress("pollution_case_started", {"case_id": case_id})
        family_metrics: dict[str, Any] = {}
        family_fields: dict[str, Any] = {}
        for family in pollution_config["benchmark_families"]:
            result = physical_k_modal_fem_benchmark(
                degree=int(case["degree"]),
                element_size_ratio=float(case["element_size_ratio"]),
                formal_kh=float(pollution_config["formal_kh"]),
                radial_extent=float(pollution_config["domain_radial_extent"]),
                axial_extent=float(pollution_config["domain_axial_extent"]),
                radial_mode=int(pollution_config["radial_mode"]),
                axial_mode=int(pollution_config["axial_mode"]),
                complex_amplitude=amplitude,
                discontinuous_mass=family == "glass_air_interface",
                interface_radius=float(pollution_config["interface_radius"]),
                homogeneous_n2=float(pollution_config["homogeneous_n2"]),
                interface_inner_n2=float(
                    pollution_config["interface_inner_n2"]
                ),
                interface_outer_n2=float(
                    pollution_config["interface_outer_n2"]
                ),
                quadrature_order=int(case["quadrature_order"]),
                evaluation_count_per_axis=int(
                    pollution_config["evaluation_count_per_axis"]
                ),
            )
            family_metrics[family] = _without_arrays(
                result, _POLLUTION_ARRAY_KEYS
            )
            family_fields[family] = {
                key: result[key]
                for key in _POLLUTION_ARRAY_KEYS
                if key in result
            }
            all_solver_residuals.append(
                float(result["solver_controls"]["relative_residual"])
            )
        projected_unknowns = _project_full_tgv_unknowns(
            config,
            degree=int(case["degree"]),
            element_size_ratio=float(case["element_size_ratio"]),
        )
        eligible = bool(
            all(
                family_metrics[family]["weighted_relative_l2"]
                <= float(
                    thresholds["physical_k_family_weighted_relative_l2_max"]
                )
                and family_metrics[family]["solver_controls"]["relative_residual"]
                <= float(thresholds["solve_relative_residual_max"])
                and family_metrics[family]["all_finite"]
                for family in pollution_config["benchmark_families"]
            )
        )
        pollution_metrics[case_id] = {
            "degree": int(case["degree"]),
            "element_size_ratio": float(case["element_size_ratio"]),
            "quadrature_order": int(case["quadrature_order"]),
            "estimated_full_tgv_active_unknowns": projected_unknowns,
            "candidate_eligible": eligible,
            **family_metrics,
        }
        pollution_fields[case_id] = family_fields
        if eligible:
            eligible_ids.append(case_id)
        progress(
            "pollution_case_completed",
            {
                "case_id": case_id,
                "homogeneous_l2": family_metrics["homogeneous"][
                    "weighted_relative_l2"
                ],
                "interface_l2": family_metrics["glass_air_interface"][
                    "weighted_relative_l2"
                ],
                "eligible": eligible,
            },
        )
    selected_id = None
    if eligible_ids:
        selected_id = min(
            eligible_ids,
            key=lambda case_id: (
                pollution_metrics[case_id]["estimated_full_tgv_active_unknowns"],
                pollution_metrics[case_id]["degree"],
                pollution_metrics[case_id]["element_size_ratio"],
            ),
        )
    candidate_found = selected_id is not None
    selected_fields = (
        pollution_fields[selected_id] if selected_id is not None else {}
    )
    maximum_solver_residual = max(all_solver_residuals)
    all_finite = bool(
        all(value["all_finite"] for value in domain_metrics.values())
        and all(
            value[family]["all_finite"]
            for value in pollution_metrics.values()
            for family in pollution_config["benchmark_families"]
        )
    )
    hard_controls_pass = bool(
        all_finite
        and maximum_solver_residual
        <= float(thresholds["solve_relative_residual_max"])
    )
    domain_gate = bool(domain_gate and hard_controls_pass)
    candidate_found = bool(candidate_found and hard_controls_pass)
    benchmark_validated = bool(domain_gate and candidate_found)
    selected_unknowns = (
        None
        if selected_id is None
        else int(
            pollution_metrics[selected_id]["estimated_full_tgv_active_unknowns"]
        )
    )
    direct_unknown_gate = bool(
        selected_unknowns is not None
        and selected_unknowns
        <= int(config["full_tgv_projection"]["direct_lu_maximum_unknowns"])
    )
    if benchmark_validated:
        interpretation = (
            "r13_benchmarks_closed__solver_preflight_required"
            if direct_unknown_gate
            else "r13_benchmarks_closed__iterative_solver_required"
        )
    elif not domain_gate and not candidate_found:
        interpretation = "r13_domain_and_physical_k_benchmarks_not_closed"
    elif not domain_gate:
        interpretation = "r13_domain_reflection_not_closed"
    else:
        interpretation = "r13_physical_k_candidate_not_found"
    metrics = {
        "version": "R13",
        "scientific_result": True,
        "status": "Passed" if benchmark_validated else "Failed",
        "interpretation_code": interpretation,
        "benchmark_validated": benchmark_validated,
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "gates": {
            "domain_reflection_gate_pass": domain_gate,
            "physical_k_candidate_found": candidate_found,
            "hard_controls_pass": hard_controls_pass,
            "direct_lu_unknown_count_gate_report_only": direct_unknown_gate,
        },
        "hard_controls": {
            "all_finite": all_finite,
            "maximum_solver_relative_residual": maximum_solver_residual,
        },
        "domain_reflection": {
            "fixed_case_order": list(domain_config["fixed_case_order"]),
            "cases": domain_metrics,
            "case_gate_pass": domain_case_pass,
            "core48_to_core60_dense_field_relative_l2": domain_pair_l2,
        },
        "physical_k_pollution": {
            "formal_kh": float(pollution_config["formal_kh"]),
            "fixed_case_order": list(pollution_config["fixed_case_order"]),
            "cases": pollution_metrics,
            "eligible_candidate_ids": eligible_ids,
            "selected_candidate_id": selected_id,
            "selected_candidate_projected_unknowns": selected_unknowns,
        },
        "full_tgv_projection": dict(config["full_tgv_projection"]),
        "conditional_execution": {
            "r12_cartesian_checkpoint_reused_by_hash_only": True,
            "cartesian_executed": False,
            "full_tgv_executed": False,
            "scalar_cross_model_executed": False,
            "vector_model_executed": False,
        },
        "thresholds": dict(thresholds),
    }
    arrays = {
        "domain_reflection": domain_arrays,
        "selected_candidate_fields": selected_fields,
    }
    return metrics, arrays


def _hdf5_safe(value: Any) -> Any:
    if value is None:
        return "null"
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
    selected = arrays["selected_candidate_fields"]
    truth: dict[str, Any] = {
        "domain_reflection": {
            case_id: {
                "dense_radii_m": values["dense_radii_m"],
                "analytic_outgoing_field": values["dense_truth"],
            }
            for case_id, values in arrays["domain_reflection"].items()
        }
    }
    if selected:
        truth["physical_k_pollution"] = {
            family: {
                "radial_coordinates": values["radial_coordinates"],
                "axial_coordinates": values["axial_coordinates"],
                "analytic_field": values["truth_field"],
            }
            for family, values in selected.items()
        }
    save_ptycho_hdf5(
        path,
        instrument=_hdf5_safe(
            {
                "domain_reflection": config["domain_reflection"],
                "physical_k_pollution": config["physical_k_pollution"],
            }
        ),
        sample=_hdf5_safe(
            {
                "kind": "analytic_helmholtz_benchmark",
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
        domain_group = data.require_group("domain_reflection")
        for case_id, values in arrays["domain_reflection"].items():
            group = domain_group.require_group(case_id)
            for key in (
                "dense_radii_m",
                "dense_field",
                "dense_derivative",
                "measurement_radii_m",
                "measurement_field",
                "measurement_derivative",
                "incoming_to_outgoing_ratio",
                "outgoing_impedance_residual",
                "radial_flux",
            ):
                group.create_dataset(key, data=np.asarray(values[key]))
        pollution_group = data.require_group("physical_k_pollution")
        case_ids = list(metrics["physical_k_pollution"]["fixed_case_order"])
        pollution_group.create_dataset(
            "case_ids",
            data=case_ids,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        pollution_group.create_dataset(
            "homogeneous_weighted_relative_l2",
            data=np.asarray(
                [
                    metrics["physical_k_pollution"]["cases"][case_id]
                    ["homogeneous"]["weighted_relative_l2"]
                    for case_id in case_ids
                ]
            ),
        )
        pollution_group.create_dataset(
            "glass_air_interface_weighted_relative_l2",
            data=np.asarray(
                [
                    metrics["physical_k_pollution"]["cases"][case_id]
                    ["glass_air_interface"]["weighted_relative_l2"]
                    for case_id in case_ids
                ]
            ),
        )
        pollution_group.create_dataset(
            "projected_full_tgv_active_unknowns",
            data=np.asarray(
                [
                    metrics["physical_k_pollution"]["cases"][case_id]
                    ["estimated_full_tgv_active_unknowns"]
                    for case_id in case_ids
                ],
                dtype=np.int64,
            ),
        )
        if selected:
            selected_group = pollution_group.require_group("selected_candidate")
            for family, values in selected.items():
                family_group = selected_group.require_group(family)
                family_group.create_dataset(
                    "numerical_field", data=np.asarray(values["numerical_field"])
                )


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
            raise RuntimeError(f"missing or empty R13 artifact: {relative}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["scientific_result"] is not True:
        raise RuntimeError("R13 metrics lost scientific-result status.")
    if metrics["conditional_execution"]["cartesian_executed"] is not False:
        raise RuntimeError("R13 unexpectedly reran Cartesian propagation.")
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
            raise RuntimeError("R13 HDF5 entry layout differs.")
        if set(entry["data"]) != {
            "domain_reflection",
            "physical_k_pollution",
        }:
            raise RuntimeError("R13 HDF5 data layout differs.")
        if "reconstruction" in entry:
            raise RuntimeError("R13 must not write reconstruction data.")
    for filename in config["output"]["figure_filenames"]:
        image = np.asarray(iio.imread(run_dir / "figures" / str(filename)))
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 100:
            raise RuntimeError(f"invalid R13 figure: {filename}")
        if not np.all(np.isfinite(image)):
            raise RuntimeError(f"non-finite R13 figure: {filename}")


def run(config_path: Path) -> Path:
    """Execute the single formal R13 scientific benchmark run."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R13 source config hash differs.")
    config = load_config(source)
    validate_r13_config(config)
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
                "stage": "R13",
                "state": "running",
                "scientific_result": True,
                "formal_execution_count": 1,
                "created_at": created_at_utc(),
            },
        )
        progress(
            "formal_execution_started",
            {
                "config_sha256": REGISTERED_CONFIG_SHA256,
                "scientific_contract_sha256": scientific_contract_sha256(config),
            },
        )
        metrics, arrays = _execute(config, progress)
        metrics["total_execution_elapsed_s"] = float(time.perf_counter() - started)
        metrics["formal_provenance_hashes"] = provenance["hashes"]
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": "R13",
            "scientific_result": True,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit(PROJECT_ROOT),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
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
        save_exp040_r13_figures(run_dir / "figures", metrics, arrays)
        save_json(
            state_path,
            {
                "stage": "R13",
                "state": "completed",
                "scientific_result": True,
                "formal_execution_count": 1,
                "status": metrics["status"],
                "interpretation_code": metrics["interpretation_code"],
                "benchmark_validated": metrics["benchmark_validated"],
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
                "stage": "R13",
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
        "selected_candidate: "
        f"{metrics['physical_k_pollution']['selected_candidate_id']}",
        flush=True,
    )
    return run_dir


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
