"""Run the non-scientific preflight for the exp040 R14 solver audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import run_exp040_r14 as formal_runner  # noqa: E402

from tgv_ptycho.forward.helmholtz_axisymmetric import (  # noqa: E402
    PeakRSSMonitor,
    solve_sparse_direct,
)
from tgv_ptycho.forward.helmholtz_axisymmetric_fem import (  # noqa: E402
    make_axisymmetric_fem_grid,
)
from tgv_ptycho.forward.helmholtz_benchmarks import (  # noqa: E402
    axisymmetric_modal_nodal_error,
    make_axisymmetric_pml_modal_problem,
)
from tgv_ptycho.forward.helmholtz_iterative import (  # noqa: E402
    build_csl_ilu_preconditioner,
    build_two_level_ras_csl_preconditioner,
    solve_restarted_gmres,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402

REGISTERED_CONFIG_SHA256 = (
    "BE215853F01FF975B05F0BF5642319AF4C4062BC44A86FAB8B909759387BB3B6"
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
        raise ValueError(f"{name} differs from the R14 preflight registration.")


def validate_preflight_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the non-scientific R14 preflight contract."""

    _require_exact(config["experiment"]["id"], "exp040", "experiment.id")
    _require_exact(
        config["experiment"]["stage"],
        "R14_preflight",
        "experiment.stage",
    )
    _require_exact(
        config["experiment"]["scientific_result"],
        False,
        "experiment.scientific_result",
    )
    backend = config["backend_controls"]
    _require_exact(backend["required_backend"], "scipy", "required backend")
    _require_exact(
        backend["full_shifted_lu_preconditioner_forbidden"],
        True,
        "full shifted LU flag",
    )
    algebra = config["algebra_controls"]
    _require_exact(algebra["random_seed"], 40014, "random seed")
    _require_exact(algebra["low_k_operator_wavenumber"], 4.0, "low k")
    _require_exact(algebra["low_k_degree"], 2, "low-k degree")
    _require_exact(
        algebra["low_k_ras_core_block_shape_nodes"],
        [16, 16],
        "low-k RAS block",
    )
    _require_exact(algebra["low_k_ras_overlap_nodes"], 2, "low-k overlap")
    resource = config["resource_controls"]
    _require_exact(
        resource["full_tgv_execution_enabled"], False, "full TGV flag"
    )
    _require_exact(
        resource["formal_cartesian_execution_enabled"],
        False,
        "Cartesian flag",
    )
    _require_exact(
        dict(config["thresholds"]),
        {
            "low_k_true_relative_residual_max": 1.0e-8,
            "low_k_direct_agreement_relative_l2_max": 1.0e-6,
            "linear_operator_repeat_relative_l2_max": 1.0e-13,
            "axial_pml_incoming_ratio_max": 1.0e-3,
            "require_all_finite": True,
        },
        "preflight thresholds",
    )


def _validate_provenance_and_contract(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], Mapping[str, Any]]:
    provenance = config["provenance"]
    r13_dir = PROJECT_ROOT / str(provenance["r13_run"])
    paths = {
        "r13_metrics": r13_dir / "metrics.json",
        "r13_hdf5": r13_dir / "outputs" / "exp040_r13.h5",
    }
    expected = {
        "r13_metrics": str(provenance["r13_metrics_sha256"]),
        "r13_hdf5": str(provenance["r13_hdf5_sha256"]),
    }
    actual = {key: _sha256(path) for key, path in paths.items()}
    if actual != expected:
        raise ValueError("R13 provenance differs from R14 registration.")
    with paths["r13_metrics"].open("r", encoding="utf-8") as handle:
        r13_metrics = json.load(handle)
    if r13_metrics["status"] != "Passed":
        raise ValueError("R13 did not pass its registered benchmark.")
    if (
        r13_metrics["physical_k_pollution"]["selected_candidate_id"]
        != "h0p5_p4"
    ):
        raise ValueError("R13 selected candidate differs.")

    formal_path = PROJECT_ROOT / str(config["formal_contract"]["config_path"])
    formal_config = load_config(formal_path)
    formal_runner.validate_r14_config(formal_config)
    contract_hash = formal_runner.scientific_contract_sha256(formal_config)
    if contract_hash != str(
        config["formal_contract"]["scientific_contract_sha256"]
    ):
        raise ValueError("R14 formal scientific contract hash differs.")
    actual["scientific_contract"] = contract_hash
    return actual, formal_config


def _backend_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    backend = config["backend_controls"]
    availability = {
        name: importlib.util.find_spec(name) is not None
        for name in ("scipy", "pyamg", "petsc4py", "pypardiso")
    }
    gate = bool(
        availability["scipy"]
        and (not backend["require_no_pyamg"] or not availability["pyamg"])
        and (
            not backend["require_no_petsc4py"]
            or not availability["petsc4py"]
        )
        and (
            not backend["require_no_pypardiso"]
            or not availability["pypardiso"]
        )
        and backend["full_shifted_lu_preconditioner_forbidden"] is True
        and backend["gpu_required"] is False
    )
    return {"availability": availability, "gate_pass": gate}


def _formal_grid_controls(formal_config: Mapping[str, Any]) -> dict[str, Any]:
    scaling = formal_config["solver_scaling"]
    cases: dict[str, Any] = {}
    for case_id in scaling["fixed_case_order"]:
        core = float(scaling["cases"][case_id]["core_extent"])
        pml = float(scaling["pml_thickness"])
        spacing = float(scaling["element_size_ratio"])
        grid = make_axisymmetric_fem_grid(
            degree=int(scaling["degree"]),
            radial_extent_m=core + pml,
            z_min_m=-pml,
            z_max_m=core + pml,
            radial_element_size_m=spacing,
            axial_element_size_m=spacing,
        )
        expected = int(scaling["cases"][case_id]["expected_active_unknowns"])
        cases[case_id] = {
            "active_unknowns": grid.active_unknown_count,
            "expected_active_unknowns": expected,
            "active_shape": [
                grid.axial_node_count - 2,
                grid.radial_node_count - 1,
            ],
            "matches_registration": grid.active_unknown_count == expected,
        }
    return {
        "cases": cases,
        "maximum_formal_active_unknowns": max(
            value["active_unknowns"] for value in cases.values()
        ),
        "all_unknown_counts_match": all(
            value["matches_registration"] for value in cases.values()
        ),
    }


def _compact_solve_controls(controls: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in controls.items()
        if key != "preconditioned_residual_history"
    }


def _relative_l2(test: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(test - reference)
        / max(np.linalg.norm(reference), np.finfo(float).eps)
    )


def _low_k_algebra(
    config: Mapping[str, Any], formal_config: Mapping[str, Any]
) -> dict[str, Any]:
    algebra = config["algebra_controls"]
    common = {
        "degree": int(algebra["low_k_degree"]),
        "element_size": float(algebra["low_k_element_size"]),
        "core_extent": float(algebra["low_k_core_extent"]),
        "pml_thickness": float(algebra["low_k_pml_thickness"]),
        "operator_wavenumber": float(algebra["low_k_operator_wavenumber"]),
        "pml_polynomial_order": int(algebra["low_k_pml_polynomial_order"]),
        "pml_target_one_way_amplitude": float(
            algebra["low_k_pml_target_one_way_amplitude"]
        ),
        "interface_fraction_of_core_radius": float(
            algebra["low_k_interface_fraction"]
        ),
        "interface_inner_n2": 4.0 / 9.0,
        "interface_outer_n2": 1.0,
        "target_radial_modal_wavenumber": float(
            algebra["low_k_target_radial_modal_wavenumber"]
        ),
        "target_axial_modal_wavenumber": float(
            algebra["low_k_target_axial_modal_wavenumber"]
        ),
        "modal_index_offset": 0,
        "quadrature_order": 8,
    }
    original = make_axisymmetric_pml_modal_problem(**common)
    shifted = make_axisymmetric_pml_modal_problem(
        **common,
        imaginary_mass_shift=float(algebra["low_k_shift_imaginary"]),
    )
    direct, direct_controls = solve_sparse_direct(
        original["matrix"], original["rhs"]
    )
    rng = np.random.default_rng(int(algebra["random_seed"]))
    probe = rng.standard_normal(direct.size) + 1j * rng.standard_normal(
        direct.size
    )
    formal_solvers = formal_config["solvers"]
    preconditioners = {
        "csl_ilu_gmres": build_csl_ilu_preconditioner(
            shifted["matrix"],
            drop_tolerance=float(
                formal_solvers["csl_ilu_gmres"]["drop_tolerance"]
            ),
            fill_factor=float(
                formal_solvers["csl_ilu_gmres"]["fill_factor"]
            ),
            drop_rule=str(formal_solvers["csl_ilu_gmres"]["drop_rule"]),
            permc_spec=str(formal_solvers["csl_ilu_gmres"]["permc_spec"]),
        ),
        "two_level_ras_csl_gmres": build_two_level_ras_csl_preconditioner(
            shifted["matrix"],
            active_shape=tuple(original["controls"]["active_shape"]),
            core_block_shape_nodes=tuple(
                algebra["low_k_ras_core_block_shape_nodes"]
            ),
            overlap_nodes=int(algebra["low_k_ras_overlap_nodes"]),
        ),
    }
    results: dict[str, Any] = {}
    for solver_id, preconditioner in preconditioners.items():
        first = preconditioner.operator @ probe
        second = preconditioner.operator @ probe
        repeat_error = _relative_l2(second, first)
        solution, solve_controls = solve_restarted_gmres(
            original["matrix"],
            original["rhs"],
            preconditioner.operator,
            relative_tolerance=1.0e-8,
            absolute_tolerance=0.0,
            restart=int(algebra["low_k_gmres_restart"]),
            maximum_inner_iterations=int(
                algebra["low_k_gmres_maximum_inner_iterations"]
            ),
        )
        error = axisymmetric_modal_nodal_error(
            solution,
            original["grid"],
            original["exact_field"],
            core_extent=float(algebra["low_k_core_extent"]),
        )
        results[solver_id] = {
            "preconditioner": dict(preconditioner.controls),
            "operator_repeat_relative_l2": repeat_error,
            "solve": _compact_solve_controls(solve_controls),
            "direct_agreement_relative_l2": _relative_l2(solution, direct),
            "analytic_weighted_relative_l2_report_only": error[
                "weighted_relative_l2"
            ],
            "all_finite": bool(
                solve_controls["all_finite"]
                and error["all_finite"]
                and np.all(np.isfinite(first))
                and np.all(np.isfinite(second))
            ),
        }
    return {
        "active_unknowns": int(original["matrix"].shape[0]),
        "matrix_nnz": int(original["matrix"].nnz),
        "direct_controls": direct_controls,
        "solvers": results,
    }


def _run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance_hashes, formal_config = _validate_provenance_and_contract(config)
    backend = _backend_controls(config)
    grids = _formal_grid_controls(formal_config)
    with PeakRSSMonitor() as monitor:
        axial_metrics, _ = formal_runner._run_axial_pml(formal_config)
        low_k = _low_k_algebra(config, formal_config)
    thresholds = config["thresholds"]
    axial_pass = bool(
        all(
            value["maximum_incoming_to_outgoing_ratio"]
            <= float(thresholds["axial_pml_incoming_ratio_max"])
            and value["all_finite"]
            for value in axial_metrics.values()
        )
    )
    algebra_pass = bool(
        all(
            value["solve"]["converged"]
            and value["solve"]["true_relative_residual"]
            <= float(thresholds["low_k_true_relative_residual_max"])
            and value["direct_agreement_relative_l2"]
            <= float(
                thresholds["low_k_direct_agreement_relative_l2_max"]
            )
            and value["operator_repeat_relative_l2"]
            <= float(thresholds["linear_operator_repeat_relative_l2_max"])
            and value["preconditioner"]["full_global_factorization"] is False
            and value["all_finite"]
            for value in low_k["solvers"].values()
        )
    )
    resource = config["resource_controls"]
    free_disk_gib = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    resource_pass = bool(
        grids["all_unknown_counts_match"]
        and grids["maximum_formal_active_unknowns"]
        <= int(resource["maximum_formal_active_unknowns"])
        and monitor.peak_rss_bytes
        <= float(resource["maximum_preflight_peak_gib"]) * 1024**3
        and free_disk_gib >= float(resource["minimum_free_disk_gib"])
        and resource["full_tgv_execution_enabled"] is False
        and resource["formal_cartesian_execution_enabled"] is False
    )
    gates = {
        "provenance_and_contract_pass": True,
        "backend_pass": backend["gate_pass"],
        "low_k_algebra_pass": algebra_pass,
        "axial_pml_smoke_pass": axial_pass,
        "resource_pass": resource_pass,
    }
    formal_allowed = bool(all(gates.values()))
    return {
        "version": "R14_preflight",
        "scientific_result": False,
        "status": "Passed" if formal_allowed else "Failed",
        "formal_r14_allowed": formal_allowed,
        "gates": gates,
        "backend_controls": backend,
        "low_k_algebra": low_k,
        "axial_pml_smoke": {
            "cases": axial_metrics,
            "scientific_result": False,
        },
        "formal_grid_controls": grids,
        "resource_controls": {
            "process_peak_rss_bytes": monitor.peak_rss_bytes,
            "free_disk_gib": free_disk_gib,
            **dict(resource),
        },
        "provenance_hashes": provenance_hashes,
        "thresholds": dict(thresholds),
    }


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    for relative in config["output"]["required_files"]:
        path = run_dir / str(relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty R14 preflight artifact: {relative}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["scientific_result"] is not False:
        raise RuntimeError("R14 preflight was mislabeled as scientific.")
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as handle:
        entry = handle["entry"]
        if set(entry) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
        }:
            raise RuntimeError("R14 preflight HDF5 layout differs.")
        if set(entry["data"]):
            raise RuntimeError("R14 preflight must not contain scientific fields.")


def run(config_path: Path) -> Path:
    """Execute and persist the one non-scientific R14 preflight."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R14 preflight source config hash differs.")
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
                "stage": "R14_preflight",
                "state": "running",
                "scientific_result": False,
                "formal_r14_allowed": False,
                "created_at": created_at_utc(),
            },
        )
        metrics = _run_preflight(config)
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": "R14_preflight",
            "scientific_result": False,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit(PROJECT_ROOT),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
        }
        save_config(run_dir / "config.yaml", dict(config))
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_json(
            state_path,
            {
                "stage": "R14_preflight",
                "state": "completed",
                "scientific_result": False,
                "formal_r14_allowed": metrics["formal_r14_allowed"],
                "completed_at": created_at_utc(),
            },
        )
        save_ptycho_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            instrument={
                "backend_controls": metrics["backend_controls"],
                "formal_grid_controls": metrics["formal_grid_controls"],
            },
            config_yaml=config_to_yaml(dict(config)),
            metadata=metadata,
            metrics=metrics,
        )
        _validate_artifacts(run_dir, config)
    except Exception:
        save_json(
            state_path,
            {
                "stage": "R14_preflight",
                "state": "failed_during_execution",
                "scientific_result": False,
                "formal_r14_allowed": False,
                "failed_at": created_at_utc(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"run_dir: {run_dir}", flush=True)
    print(f"status: {metrics['status']}", flush=True)
    print(f"formal_r14_allowed: {metrics['formal_r14_allowed']}", flush=True)
    return run_dir


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
