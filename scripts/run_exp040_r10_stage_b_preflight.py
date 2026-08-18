"""Run the non-scientific exp040 R10 Stage-B formal-grid preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
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
    assemble_cylindrical_helmholtz,
    background_interface_controls,
    make_axisymmetric_grid,
    make_background_n2,
    make_cylindrical_pml,
    make_manufactured_vector,
    solve_sparse_direct,
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
    "5EA2F6C15D2C8930BE7AC7048AFD0350DE7BEF1EABA1381D17EA6EC1DA1F08D9"
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
        raise ValueError(f"R10 Stage-B preflight {name} differs from registration.")


def validate_preflight_config(config: Mapping[str, Any]) -> None:
    """Validate every resource/science-controlling preflight value."""

    _require_exact(
        set(config),
        {
            "run",
            "experiment",
            "provenance",
            "physics",
            "grid",
            "pml",
            "solver",
            "thresholds",
            "output",
        },
        "top-level sections",
    )
    _require_exact(config["experiment"]["id"], "exp040", "experiment id")
    _require_exact(
        config["experiment"]["stage"], "R10_stage_b_preflight", "stage"
    )
    _require_exact(
        config["experiment"]["scientific_result"], False, "scientific role"
    )
    _require_exact(
        dict(config["physics"]),
        {
            "wavelength_m": 5.32e-7,
            "n_glass": 1.5,
            "n_air": 1.0,
            "interface_z_m": 1.0e-4,
            "incident_amplitude": 1.0,
        },
        "physics",
    )
    _require_exact(
        dict(config["grid"]),
        {
            "id": "fine_nominal",
            "dr_m": 8.333333333333333e-8,
            "dz_m": 8.333333333333333e-8,
            "radial_core_max_m": 2.4e-5,
            "z_core_min_m": -2.0e-6,
            "z_core_max_m": 1.02e-4,
            "pml_thickness_m": 2.0e-6,
            "expected_nr": 312,
            "expected_nz": 1296,
            "expected_unknowns": 404352,
        },
        "grid",
    )
    _require_exact(
        dict(config["pml"]),
        {
            "coordinate_stretch": "cylindrical_complex_coordinate",
            "polynomial_order": 3,
            "target_one_way_amplitude": 1.0e-8,
            "radial_reference_medium": "air",
            "lower_z_reference_medium": "glass",
            "upper_z_reference_medium": "air",
            "outer_boundary": (
                "scattered_field_zero_dirichlet_at_half_cell_distance"
            ),
        },
        "PML",
    )
    _require_exact(
        dict(config["solver"]),
        {
            "package": "scipy_splu",
            "sparse_format": "csc",
            "permc_spec": "COLAMD",
            "manufactured_vector": (
                "cos_radial_times_sin_axial_times_1_plus_0p25j"
            ),
            "zero_contrast_normalization": (
                "analytic_scattered_field_zero"
            ),
        },
        "solver",
    )
    _require_exact(
        dict(config["thresholds"]),
        {
            "available_memory_before_gib_min": 3.0,
            "factor_and_solve_wall_s_max": 180.0,
            "process_peak_rss_gib_max": 6.0,
            "solve_relative_residual_max": 1.0e-9,
            "manufactured_recovery_relative_l2_max": 1.0e-8,
            "algebra_absolute_or_relative_max": 1.0e-12,
            "require_all_finite": True,
        },
        "thresholds",
    )
    _require_exact(
        config["output"]["hdf5_filename"],
        "exp040_r10_stage_b_preflight.h5",
        "HDF5 filename",
    )


def _progress_writer(path: Path):
    payload: dict[str, Any] = {
        "purpose": "r10_stage_b_formal_grid_resource_preflight",
        "created_at": created_at_utc(),
        "events": [],
    }

    def write(event: str, details: Mapping[str, Any]) -> None:
        record = {"event": event, "at": created_at_utc(), **dict(details)}
        payload["events"].append(record)
        payload["latest_event"] = record
        save_json(path, payload)
        print(f"progress: {event} {dict(details)}", flush=True)

    return write


def _pml_algebra_error(grid, pml) -> float:
    physical_r = grid.r_centers_m < grid.radial_core_max_m
    physical_z = (grid.z_centers_m > grid.z_core_min_m) & (
        grid.z_centers_m < grid.z_core_max_m
    )
    errors = [
        np.max(np.abs(pml.r_stretch_centers[physical_r] - 1.0)),
        np.max(
            np.abs(
                pml.r_tilde_centers_m[physical_r]
                - grid.r_centers_m[physical_r]
            )
        ),
        np.max(np.abs(pml.z_stretch_centers[physical_z] - 1.0)),
        np.max(
            np.abs(
                pml.z_tilde_centers_m[physical_z]
                - grid.z_centers_m[physical_z]
            )
        ),
        abs(pml.r_tilde_faces_m[0]),
    ]
    return float(max(float(value) for value in errors))


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    expected = set(str(value) for value in config["output"]["required_files"])
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"Preflight artifact set differs: {sorted(actual)}")
    for name in (
        "metadata.json",
        "metrics.json",
        "run_state.json",
        "run_progress.json",
    ):
        with (run_dir / name).open("r", encoding="utf-8") as handle:
            json.load(handle)
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as h5:
        if set(h5["entry"]) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
        }:
            raise RuntimeError("Preflight HDF5 entry layout differs.")
        if len(h5["entry/data"]) != 0:
            raise RuntimeError("Preflight HDF5 data group must be empty.")


def run(config_path: Path) -> Path:
    """Execute the one registered non-scientific preflight."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R10 Stage-B preflight source config hash differs.")
    config = load_config(source)
    validate_preflight_config(config)
    stage_a_h5 = (
        PROJECT_ROOT
        / str(config["provenance"]["stage_a_run"])
        / "outputs"
        / "exp040_r10_stage_a_repaired.h5"
    )
    expected_stage_a_hash = str(
        config["provenance"]["stage_a_repaired_hdf5_sha256"]
    )
    if not stage_a_h5.is_file() or _sha256(stage_a_h5) != expected_stage_a_hash:
        raise RuntimeError("Locked Stage-A repaired HDF5 provenance differs.")

    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    save_config(run_dir / "config.yaml", dict(config))
    save_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "created_at": created_at_utc(),
            "scientific_result": False,
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
        },
    )
    progress = _progress_writer(run_dir / "run_progress.json")
    progress("preflight_started", {"source_config_sha256": REGISTERED_CONFIG_SHA256})
    started = time.perf_counter()
    try:
        thresholds = config["thresholds"]
        available_before = int(psutil.virtual_memory().available)
        available_before_gib = available_before / 2**30
        progress(
            "memory_checked",
            {"available_physical_memory_gib": available_before_gib},
        )
        if available_before_gib < float(
            thresholds["available_memory_before_gib_min"]
        ):
            raise RuntimeError("available_memory_below_registered_preflight_gate")

        grid_config = config["grid"]
        grid = make_axisymmetric_grid(
            dr_m=float(grid_config["dr_m"]),
            dz_m=float(grid_config["dz_m"]),
            radial_core_max_m=float(grid_config["radial_core_max_m"]),
            z_core_min_m=float(grid_config["z_core_min_m"]),
            z_core_max_m=float(grid_config["z_core_max_m"]),
            pml_thickness_m=float(grid_config["pml_thickness_m"]),
        )
        if [grid.nr, grid.nz, grid.unknown_count] != [
            int(grid_config["expected_nr"]),
            int(grid_config["expected_nz"]),
            int(grid_config["expected_unknowns"]),
        ]:
            raise RuntimeError("formal preflight grid shape differs from registration")
        physics = config["physics"]
        pml_config = config["pml"]
        pml = make_cylindrical_pml(
            grid,
            wavelength_m=float(physics["wavelength_m"]),
            n_glass=float(physics["n_glass"]),
            n_air=float(physics["n_air"]),
            polynomial_order=int(pml_config["polynomial_order"]),
            target_one_way_amplitude=float(
                pml_config["target_one_way_amplitude"]
            ),
        )
        n2 = make_background_n2(
            grid,
            interface_z_m=float(physics["interface_z_m"]),
            n_glass=float(physics["n_glass"]),
            n_air=float(physics["n_air"]),
        )
        progress(
            "matrix_assembly_started",
            {"unknown_count": grid.unknown_count},
        )
        matrix, matrix_controls = assemble_cylindrical_helmholtz(
            grid,
            pml,
            n2,
            wavelength_m=float(physics["wavelength_m"]),
        )
        progress("matrix_assembly_completed", matrix_controls)
        manufactured = make_manufactured_vector(grid)
        rhs = matrix @ manufactured
        progress("factor_solve_started", {"matrix_nnz": int(matrix.nnz)})
        solution, solver_controls = solve_sparse_direct(
            matrix, rhs, permc_spec=str(config["solver"]["permc_spec"])
        )
        progress("factor_solve_completed", solver_controls)
        recovery_error = float(
            np.linalg.norm(solution - manufactured)
            / max(np.linalg.norm(manufactured), np.finfo(float).eps)
        )
        interface_controls = background_interface_controls(
            wavelength_m=float(physics["wavelength_m"]),
            n_glass=float(physics["n_glass"]),
            n_air=float(physics["n_air"]),
            interface_z_m=float(physics["interface_z_m"]),
            incident_amplitude=float(physics["incident_amplitude"]),
        )
        pml_error = _pml_algebra_error(grid, pml)
        zero_contrast_error = 0.0
        algebra_error = max(
            float(interface_controls["value_continuity_relative_error"]),
            float(interface_controls["derivative_continuity_relative_error"]),
            pml_error,
            zero_contrast_error,
            float(matrix_controls["complex_symmetric_max_abs_error"]),
        )
        total_elapsed = time.perf_counter() - started
        all_finite = bool(
            matrix_controls["finite_data"]
            and solver_controls["all_finite"]
            and np.all(np.isfinite(solution))
            and np.isfinite(recovery_error)
        )
        controls_pass = {
            "available_memory": bool(
                available_before_gib
                >= float(thresholds["available_memory_before_gib_min"])
            ),
            "factor_and_solve_wall": bool(
                float(solver_controls["factor_and_solve_elapsed_s"])
                <= float(thresholds["factor_and_solve_wall_s_max"])
            ),
            "peak_rss": bool(
                0 <= int(solver_controls["peak_rss_bytes"])
                <= float(thresholds["process_peak_rss_gib_max"]) * 2**30
            ),
            "solve_residual": bool(
                float(solver_controls["relative_residual"])
                <= float(thresholds["solve_relative_residual_max"])
            ),
            "manufactured_recovery": bool(
                recovery_error
                <= float(thresholds["manufactured_recovery_relative_l2_max"])
            ),
            "algebra": bool(
                algebra_error
                <= float(thresholds["algebra_absolute_or_relative_max"])
            ),
            "all_finite": all_finite,
        }
        hard_pass = bool(all(controls_pass.values()))
        status = "Passed" if hard_pass else "Blocked"
        interpretation = (
            "formal_grid_preflight_passed"
            if hard_pass
            else "formal_grid_preflight_failed"
        )
        metrics = {
            "version": "R10_stage_b_preflight",
            "scientific_result": False,
            "provenance": dict(config["provenance"]),
            "grid": {
                "id": str(grid_config["id"]),
                "nr": grid.nr,
                "nz": grid.nz,
                "unknown_count": grid.unknown_count,
                "dr_m": grid.dr_m,
                "dz_m": grid.dz_m,
            },
            "pml": {
                "radial_peak_alpha": pml.radial_peak_alpha,
                "lower_z_peak_alpha": pml.lower_z_peak_alpha,
                "upper_z_peak_alpha": pml.upper_z_peak_alpha,
                "physical_core_identity_max_abs_error": pml_error,
            },
            "matrix_controls": matrix_controls,
            "solver_controls": solver_controls,
            "manufactured_recovery_relative_l2": recovery_error,
            "background_interface_controls": interface_controls,
            "zero_contrast_normalization_max_abs_error": zero_contrast_error,
            "maximum_algebra_error": algebra_error,
            "available_physical_memory_before_bytes": available_before,
            "available_physical_memory_before_gib": available_before_gib,
            "total_execution_elapsed_s": float(total_elapsed),
            "control_pass": controls_pass,
            "hard_controls_pass": hard_pass,
            "thresholds": dict(thresholds),
            "status": status,
            "interpretation_code": interpretation,
            "formal_stage_b_allowed": hard_pass,
        }
        metadata = {
            "experiment_id": "exp040",
            "diagnostic_stage": "R10_stage_b_preflight",
            "scientific_result": False,
            "run_path": str(run_dir.resolve()),
            "source_config": str(source),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
            "created_at": created_at_utc(),
            "git_commit": get_git_commit(PROJECT_ROOT) or "unavailable",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "status": status,
            "interpretation_code": interpretation,
            "formal_stage_b_allowed": hard_pass,
        }
        save_json(run_dir / "metrics.json", metrics)
        save_json(run_dir / "metadata.json", metadata)
        save_ptycho_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            instrument={
                "wavelength_m": float(physics["wavelength_m"]),
                "grid": metrics["grid"],
                "pml": metrics["pml"],
            },
            config_yaml=config_to_yaml(dict(config)),
            metadata=metadata,
            metrics=metrics,
        )
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "completed_at": created_at_utc(),
                "preflight_status": status,
                "interpretation_code": interpretation,
                "formal_stage_b_allowed": hard_pass,
                "scientific_result": False,
                "artifacts_validated": False,
            },
        )
        progress(
            "artifacts_written",
            {"preflight_status": status, "formal_stage_b_allowed": hard_pass},
        )
        _validate_artifacts(run_dir, config)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "completed_at": created_at_utc(),
                "preflight_status": status,
                "interpretation_code": interpretation,
                "formal_stage_b_allowed": hard_pass,
                "scientific_result": False,
                "artifacts_validated": True,
            },
        )
        _validate_artifacts(run_dir, config)
    except Exception as error:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed_during_execution",
                "failed_at": created_at_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "scientific_result": False,
                "formal_stage_b_allowed": False,
            },
        )
        raise

    print(f"run_dir: {run_dir.resolve()}", flush=True)
    print(f"preflight_status: {status}", flush=True)
    print(f"interpretation: {interpretation}", flush=True)
    print(f"formal_stage_b_allowed: {hard_pass}", flush=True)
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
