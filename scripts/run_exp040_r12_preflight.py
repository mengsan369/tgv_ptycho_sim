"""Run the non-scientific exp040 R12 algebra and resource preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.helmholtz_axisymmetric import (  # noqa: E402
    make_axisymmetric_grid,
)
from tgv_ptycho.forward.helmholtz_axisymmetric_fem import (  # noqa: E402
    make_axisymmetric_fem_grid,
    manufactured_fem_benchmark,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.optics.hankel import (  # noqa: E402
    make_qdht_plan,
    qdht_plan_controls,
)

REGISTERED_CONFIG_SHA256 = (
    "86C560857933727ED9A5574368E12DBE7E67CDA57E1A05AD92EE240E99E33CC3"
)


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
        raise ValueError(f"R12 preflight {name} differs from registration.")


def validate_preflight_config(config: Mapping[str, Any]) -> None:
    """Validate result-controlling R12 preflight settings."""

    _require_exact(config["experiment"]["id"], "exp040", "experiment id")
    _require_exact(
        config["experiment"]["stage"], "R12_preflight", "stage"
    )
    _require_exact(
        config["experiment"]["scientific_result"], False, "scientific role"
    )
    fv = config["formal_grid_controls"]["finite_volume_core60"]
    _require_exact(
        [fv["expected_nr"], fv["expected_nz"], fv["expected_unknowns"]],
        [744, 1296, 964224],
        "finite-volume grid",
    )
    fem_cases = config["formal_grid_controls"]["fem_core60"]["cases"]
    _require_exact(
        list(fem_cases),
        [
            {"degree": 2, "expected_unknowns": 106888},
            {"degree": 3, "expected_unknowns": 240684},
        ],
        "FEM cases",
    )
    _require_exact(
        config["resource_model"]["allow_core72_or_core96"],
        False,
        "large-core stop rule",
    )
    _require_exact(
        config["output"]["hdf5_filename"],
        "exp040_r12_preflight.h5",
        "HDF5 filename",
    )


def _hdf5_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _hdf5_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)) and any(
        isinstance(child, Mapping) for child in value
    ):
        return {
            "__sequence_encoding__": "indexed_mapping_v1",
            "length": len(value),
            "items": {
                f"{index:06d}": _hdf5_safe(child)
                for index, child in enumerate(value)
            },
        }
    return value


def _load_and_validate_r11(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    run = PROJECT_ROOT / str(provenance["r11_run"])
    paths = {
        "metrics": run / "metrics.json",
        "core48": run / "checkpoints" / "adc_fine_core48.npz",
        "chord512": run / "checkpoints" / "chord512.npz",
    }
    expected = {
        "metrics": str(provenance["r11_metrics_sha256"]),
        "core48": str(provenance["r11_core48_checkpoint_sha256"]),
        "chord512": str(provenance["r11_chord512_checkpoint_sha256"]),
    }
    hashes = {key: _sha256(path) for key, path in paths.items()}
    if hashes != expected:
        raise RuntimeError("R11 provenance hashes differ from registration.")
    with paths["metrics"].open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if (
        metrics.get("version") != "R11"
        or metrics.get("scientific_result") is not True
        or metrics.get("status") != "Failed"
        or metrics.get("interpretation_code")
        != "r11_reference_not_closed__domain_mesh_anisotropy"
    ):
        raise RuntimeError("R11 scientific provenance differs.")
    return {"run": run, "hashes": hashes, "metrics": metrics}


def _formal_grid_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    controls = config["formal_grid_controls"]
    fv = controls["finite_volume_core60"]
    fv_grid = make_axisymmetric_grid(
        dr_m=float(fv["dr_m"]),
        dz_m=float(fv["dz_m"]),
        radial_core_max_m=float(fv["radial_core_max_m"]),
        z_core_min_m=float(fv["z_core_min_m"]),
        z_core_max_m=float(fv["z_core_max_m"]),
        pml_thickness_m=float(fv["pml_thickness_m"]),
    )
    fv_actual = [fv_grid.nr, fv_grid.nz, fv_grid.unknown_count]
    fv_expected = [
        int(fv["expected_nr"]),
        int(fv["expected_nz"]),
        int(fv["expected_unknowns"]),
    ]
    fem = controls["fem_core60"]
    fem_rows: dict[str, Any] = {}
    fem_pass = True
    for case in fem["cases"]:
        degree = int(case["degree"])
        grid = make_axisymmetric_fem_grid(
            degree=degree,
            radial_extent_m=float(fem["radial_extent_m"]),
            z_min_m=float(fem["z_min_m"]),
            z_max_m=float(fem["z_max_m"]),
            radial_element_size_m=float(fem["radial_element_size_m"]),
            axial_element_size_m=float(fem["axial_element_size_m"]),
        )
        expected = int(case["expected_unknowns"])
        passed = grid.active_unknown_count == expected
        fem_pass = fem_pass and passed
        fem_rows[f"p{degree}"] = {
            "degree": degree,
            "radial_elements": grid.radial_element_count,
            "axial_elements": grid.axial_element_count,
            "radial_nodes": grid.radial_node_count,
            "axial_nodes": grid.axial_node_count,
            "active_unknowns": grid.active_unknown_count,
            "expected_unknowns": expected,
            "pass": passed,
        }
    cartesian_rows: dict[str, Any] = {}
    for case in controls["cartesian"]["cases"]:
        shape = [int(value) for value in case["shape"]]
        dx = float(controls["cartesian"]["dx_m"])
        cartesian_rows[str(case["id"])] = {
            "shape": shape,
            "dx_m": dx,
            "fov_m": [shape[0] * dx, shape[1] * dx],
        }
    return {
        "finite_volume_core60": {
            "actual": fv_actual,
            "expected": fv_expected,
            "pass": fv_actual == fv_expected,
        },
        "fem_core60": fem_rows,
        "fem_all_pass": fem_pass,
        "cartesian": cartesian_rows,
    }


def _resource_controls(
    config: Mapping[str, Any], r11_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    fine_ids = ["adc_fine_core24", "adc_fine_core36", "adc_fine_core48"]
    case_controls = r11_metrics["case_controls"]
    core_um = np.asarray(
        [case_controls[key]["radial_core_max_m"] * 1.0e6 for key in fine_ids]
    )
    factor_nnz = np.asarray(
        [
            case_controls[key]["solver_controls"]["factor_l_plus_u_nnz"]
            for key in fine_ids
        ],
        dtype=np.float64,
    )
    peak_bytes = np.asarray(
        [
            case_controls[key]["solver_controls"]["peak_rss_bytes"]
            for key in fine_ids
        ],
        dtype=np.float64,
    )
    slope, intercept = np.polyfit(core_um, factor_nnz, 1)
    predicted_factor_nnz = float(slope * 60.0 + intercept)
    conservative_bytes_per_factor_entry = float(np.max(peak_bytes / factor_nnz))
    safety_factor = 1.15
    predicted_peak_gib = (
        predicted_factor_nnz
        * conservative_bytes_per_factor_entry
        * safety_factor
        / 2.0**30
    )
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(PROJECT_ROOT)
    model = config["resource_model"]
    return {
        "fit_case_ids": fine_ids,
        "factor_nnz_linear_slope_per_um": float(slope),
        "factor_nnz_linear_intercept": float(intercept),
        "predicted_core60_factor_nnz": predicted_factor_nnz,
        "conservative_bytes_per_factor_entry": conservative_bytes_per_factor_entry,
        "safety_factor": safety_factor,
        "predicted_domain_peak_gib": float(predicted_peak_gib),
        "maximum_predicted_domain_peak_gib": float(
            model["maximum_predicted_domain_peak_gib"]
        ),
        "total_physical_memory_gib": float(memory.total / 2.0**30),
        "available_memory_gib_report_only": float(memory.available / 2.0**30),
        "free_disk_gib": float(disk.free / 2.0**30),
        "minimum_free_disk_gib": float(model["minimum_free_disk_gib"]),
        "memory_model_pass": bool(
            predicted_peak_gib
            <= float(model["maximum_predicted_domain_peak_gib"])
        ),
        "disk_pass": bool(
            disk.free / 2.0**30 >= float(model["minimum_free_disk_gib"])
        ),
        "core72_or_core96_allowed": bool(model["allow_core72_or_core96"]),
    }


def _run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _load_and_validate_r11(config)
    grids = _formal_grid_controls(config)
    benchmark_config = config["manufactured_fem"]
    benchmarks: dict[str, Any] = {}
    for case_id in benchmark_config["fixed_order"]:
        interface = str(case_id).startswith("interface")
        degree = int(str(case_id)[-1])
        print(f"preflight: manufactured {case_id}", flush=True)
        benchmarks[str(case_id)] = manufactured_fem_benchmark(
            degree=degree,
            discontinuous_mass=interface,
            element_size=float(benchmark_config["element_size"]),
            quadrature_order=int(benchmark_config["quadrature_order"]),
        )
    qdht_config = config["qdht"]
    qdht = qdht_plan_controls(
        make_qdht_plan(
            int(qdht_config["sample_count"]),
            float(qdht_config["radial_max_m"]),
            order=int(qdht_config["order"]),
        )
    )
    resource = _resource_controls(config, provenance["metrics"])
    thresholds = config["thresholds"]
    homogeneous_ratio = float(
        benchmarks["homogeneous_p3"]["weighted_relative_l2"]
        / benchmarks["homogeneous_p2"]["weighted_relative_l2"]
    )
    interface_ratio = float(
        benchmarks["interface_p3"]["weighted_relative_l2"]
        / benchmarks["interface_p2"]["weighted_relative_l2"]
    )
    maximum_residual = max(
        float(value["solver_controls"]["relative_residual"])
        for value in benchmarks.values()
    )
    gates = {
        "provenance_pass": True,
        "formal_grids_pass": bool(
            grids["finite_volume_core60"]["pass"] and grids["fem_all_pass"]
        ),
        "manufactured_solver_pass": bool(
            maximum_residual
            <= float(thresholds["solve_relative_residual_max"])
        ),
        "homogeneous_benchmark_pass": bool(
            benchmarks["homogeneous_p3"]["weighted_relative_l2"]
            <= float(thresholds["homogeneous_p3_weighted_relative_l2_max"])
            and homogeneous_ratio
            <= float(thresholds["homogeneous_p3_to_p2_ratio_max"])
        ),
        "interface_benchmark_pass": bool(
            benchmarks["interface_p3"]["weighted_relative_l2"]
            <= float(thresholds["interface_p3_weighted_relative_l2_max"])
            and interface_ratio
            <= float(thresholds["interface_p3_to_p2_ratio_max"])
        ),
        "qdht_pass": bool(
            qdht["transform_involution_probe_relative_l2"]
            <= float(thresholds["qdht_involution_relative_l2_max"])
            and qdht["scaled_parseval_relative_error"]
            <= float(thresholds["qdht_parseval_relative_error_max"])
            and qdht["physical_roundtrip_relative_l2"]
            <= float(thresholds["qdht_roundtrip_relative_l2_max"])
            and qdht["all_finite"]
        ),
        "resource_pass": bool(
            resource["memory_model_pass"] and resource["disk_pass"]
        ),
    }
    formal_allowed = all(gates.values())
    return {
        "version": "R12_preflight",
        "scientific_result": False,
        "status": "Passed" if formal_allowed else "Failed",
        "interpretation_code": (
            "r12_formal_preflight_passed"
            if formal_allowed
            else "r12_formal_preflight_failed"
        ),
        "formal_r12_allowed": formal_allowed,
        "provenance": {
            "r11_run": str(config["provenance"]["r11_run"]),
            "hashes": provenance["hashes"],
        },
        "formal_grid_controls": grids,
        "manufactured_fem": benchmarks,
        "manufactured_summary": {
            "maximum_solver_relative_residual": maximum_residual,
            "homogeneous_p3_to_p2_ratio": homogeneous_ratio,
            "interface_p3_to_p2_ratio": interface_ratio,
        },
        "qdht_controls": qdht,
        "resource_controls": resource,
        "gates": gates,
        "thresholds": dict(thresholds),
    }


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    expected = set(config["output"]["required_files"])
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"R12 preflight artifact set differs: {sorted(actual)}")
    with h5py.File(
        run_dir / "outputs" / str(config["output"]["hdf5_filename"]), "r"
    ) as handle:
        entry = handle["entry"]
        if set(entry) != {"config_yaml", "data", "instrument", "metadata", "metrics"}:
            raise RuntimeError("R12 preflight HDF5 layout differs.")
        if set(entry["data"]):
            raise RuntimeError("R12 preflight HDF5 data must be empty.")


def run(config_path: Path) -> Path:
    """Execute and persist the non-scientific R12 preflight."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R12 preflight source config hash differs.")
    config = load_config(source)
    validate_preflight_config(config)
    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    state_path = run_dir / "run_state.json"
    try:
        save_json(
            state_path,
            {
                "stage": "R12_preflight",
                "state": "running",
                "scientific_result": False,
                "formal_r12_allowed": False,
                "created_at": created_at_utc(),
            },
        )
        metrics = _run_preflight(config)
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": "R12_preflight",
            "scientific_result": False,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit(PROJECT_ROOT),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
        }
        save_config(run_dir / "config.yaml", config)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_json(
            state_path,
            {
                "stage": "R12_preflight",
                "state": "completed",
                "scientific_result": False,
                "formal_r12_allowed": bool(metrics["formal_r12_allowed"]),
                "completed_at": created_at_utc(),
            },
        )
        save_ptycho_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            instrument=_hdf5_safe(
                {"formal_grid_controls": metrics["formal_grid_controls"]}
            ),
            config_yaml=config_to_yaml(config),
            metadata=_hdf5_safe(metadata),
            metrics=_hdf5_safe(metrics),
        )
        _validate_artifacts(run_dir, config)
    except Exception:
        save_json(
            state_path,
            {
                "stage": "R12_preflight",
                "state": "failed_during_execution",
                "scientific_result": False,
                "formal_r12_allowed": False,
                "failed_at": created_at_utc(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"run_dir: {run_dir}", flush=True)
    print(f"status: {metrics['status']}", flush=True)
    print(f"formal_r12_allowed: {metrics['formal_r12_allowed']}", flush=True)
    return run_dir


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
