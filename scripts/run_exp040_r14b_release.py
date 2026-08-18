"""Run the provenance-only release lock for the exp040 R14B formal run."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import run_exp040_r14 as formal_runner  # noqa: E402

from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402

REGISTERED_CONFIG_SHA256 = (
    "F54AB11A1AA961761E2A10EEA75ABDBA30578281D5FCC7ACB3F1A0073CCB8C42"
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
        raise ValueError(f"{name} differs from the R14B release registration.")


def validate_release_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the non-scientific release contract."""

    _require_exact(config["experiment"]["id"], "exp040", "experiment.id")
    _require_exact(
        config["experiment"]["stage"], "R14B_release", "stage"
    )
    _require_exact(config["experiment"]["scientific_result"], False, "result")
    _require_exact(
        config["formal_contract"]["invariant_sections"],
        [
            "solver_scaling",
            "solvers",
            "memory_projection",
            "thresholds",
            "conditional_execution",
        ],
        "invariant sections",
    )
    resource = config["resource_controls"]
    for key in (
        "assemble_formal_matrix",
        "rerun_axial_control",
        "full_tgv_execution_enabled",
        "formal_cartesian_execution_enabled",
    ):
        _require_exact(resource[key], False, key)


def _artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    provenance = config["provenance"]
    initial = PROJECT_ROOT / str(provenance["initial_preflight_run"])
    r14a = PROJECT_ROOT / str(provenance["r14a_run"])
    return {
        "initial_preflight_metrics": initial / "metrics.json",
        "initial_preflight_hdf5": (
            initial / "outputs" / "exp040_r14_preflight.h5"
        ),
        "r14a_metrics": r14a / "metrics.json",
        "r14a_hdf5": r14a / "outputs" / "exp040_r14a.h5",
        "r14a_checkpoint": r14a
        / "checkpoints"
        / "glass_h24_pml2.npz",
    }


def _validate_release_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    paths = _artifact_paths(config)
    provenance = config["provenance"]
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
    actual_hashes = {key: _sha256(path) for key, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise ValueError("R14B release artifact provenance differs.")
    with paths["initial_preflight_metrics"].open(
        "r", encoding="utf-8"
    ) as handle:
        initial = json.load(handle)
    with paths["r14a_metrics"].open("r", encoding="utf-8") as handle:
        r14a = json.load(handle)

    contract = config["formal_contract"]
    formal_path = PROJECT_ROOT / str(contract["config_path"])
    legacy_path = PROJECT_ROOT / str(contract["legacy_config_path"])
    formal_config = load_config(formal_path)
    legacy_config = load_config(legacy_path)
    formal_runner.validate_r14_config(formal_config)
    formal_runner.validate_r14_config(legacy_config)
    contract_hash = formal_runner.scientific_contract_sha256(formal_config)
    if contract_hash != str(contract["scientific_contract_sha256"]):
        raise ValueError("R14B scientific contract hash differs.")
    invariant_equal = {
        key: formal_config[key] == legacy_config[key]
        for key in contract["invariant_sections"]
    }
    if not all(invariant_equal.values()):
        raise ValueError("R14B solver contract differs from initial R14.")

    upstream_paths = {
        "r13_metrics": PROJECT_ROOT
        / str(formal_config["provenance"]["r13_run"])
        / "metrics.json",
        "r13_hdf5": PROJECT_ROOT
        / str(formal_config["provenance"]["r13_run"])
        / "outputs"
        / "exp040_r13.h5",
        "r12_cartesian_checkpoint": PROJECT_ROOT
        / str(formal_config["provenance"]["r12_cartesian_checkpoint"]),
    }
    upstream_expected = {
        "r13_metrics": str(
            formal_config["provenance"]["r13_metrics_sha256"]
        ),
        "r13_hdf5": str(formal_config["provenance"]["r13_hdf5_sha256"]),
        "r12_cartesian_checkpoint": str(
            formal_config["provenance"][
                "r12_cartesian_checkpoint_sha256"
            ]
        ),
    }
    upstream_actual = {
        key: _sha256(path) for key, path in upstream_paths.items()
    }
    if upstream_actual != upstream_expected:
        raise ValueError("R14B R12/R13 provenance differs.")
    axial_metrics, axial_arrays = formal_runner._load_reused_axial_pml(
        formal_config
    )
    thresholds = formal_config["thresholds"]
    axial_gate = bool(
        all(
            value["maximum_incoming_to_outgoing_ratio"]
            <= float(thresholds["axial_pml_incoming_to_outgoing_ratio_max"])
            and value["maximum_outgoing_impedance_residual"]
            <= float(thresholds["axial_pml_outgoing_impedance_residual_max"])
            and value["dense_field_relative_l2"]
            <= float(thresholds["axial_pml_dense_field_relative_l2_max"])
            and value["all_finite"]
            for value in axial_metrics.values()
        )
        and set(axial_arrays) == {"glass_downward"}
    )
    initial_solver_gates = {
        key: bool(initial["gates"][key])
        for key in (
            "provenance_and_contract_pass",
            "backend_pass",
            "low_k_algebra_pass",
            "resource_pass",
        )
    }
    initial_solver_preflight_pass = bool(all(initial_solver_gates.values()))
    r14a_pass = bool(
        r14a["status"] == "Passed"
        and r14a["corrected_axial_control_eligible"] is True
        and all(r14a["gates"].values())
    )
    maximum_unknowns = max(
        int(value["expected_active_unknowns"])
        for value in formal_config["solver_scaling"]["cases"].values()
    )
    free_disk_gib = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    resource = config["resource_controls"]
    resource_pass = bool(
        maximum_unknowns <= int(resource["maximum_formal_active_unknowns"])
        and free_disk_gib >= float(resource["minimum_free_disk_gib"])
        and resource["assemble_formal_matrix"] is False
        and resource["rerun_axial_control"] is False
        and resource["full_tgv_execution_enabled"] is False
        and resource["formal_cartesian_execution_enabled"] is False
    )
    gates = {
        "artifact_hashes_pass": True,
        "initial_solver_preflight_pass": initial_solver_preflight_pass,
        "r14a_corrected_axial_pass": r14a_pass,
        "solver_contract_invariant_pass": all(invariant_equal.values()),
        "reused_axial_gate_pass": axial_gate,
        "upstream_provenance_pass": True,
        "resource_pass": resource_pass,
    }
    formal_allowed = bool(all(gates.values()))
    metrics = {
        "version": "R14B_release",
        "scientific_result": False,
        "status": "Passed" if formal_allowed else "Failed",
        "formal_r14_allowed": formal_allowed,
        "gates": gates,
        "initial_solver_gates": initial_solver_gates,
        "solver_contract_invariant": invariant_equal,
        "reused_axial_metrics": axial_metrics,
        "resource_controls": {
            "maximum_formal_active_unknowns": maximum_unknowns,
            "free_disk_gib": free_disk_gib,
            **dict(resource),
        },
        "provenance_hashes": {
            **actual_hashes,
            **upstream_actual,
            "scientific_contract": contract_hash,
        },
    }
    return metrics, formal_config


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    for relative in config["output"]["required_files"]:
        path = run_dir / str(relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty R14B release artifact: {relative}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["scientific_result"] is not False:
        raise RuntimeError("R14B release was mislabeled as scientific.")
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as handle:
        if set(handle["entry"]) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
        }:
            raise RuntimeError("R14B release HDF5 layout differs.")
        if set(handle["entry/data"]):
            raise RuntimeError("R14B release must not contain scientific fields.")


def run(config_path: Path) -> Path:
    """Execute and persist the one provenance-only R14B release."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R14B release source config hash differs.")
    config = load_config(source)
    validate_release_config(config)
    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    state_path = run_dir / "run_state.json"
    try:
        save_json(
            state_path,
            {
                "stage": "R14B_release",
                "state": "running",
                "scientific_result": False,
                "formal_r14_allowed": False,
                "created_at": created_at_utc(),
            },
        )
        metrics, formal_config = _validate_release_inputs(config)
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": "R14B_release",
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
                "stage": "R14B_release",
                "state": "completed",
                "scientific_result": False,
                "formal_r14_allowed": metrics["formal_r14_allowed"],
                "completed_at": created_at_utc(),
            },
        )
        save_ptycho_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            instrument={
                "solver_scaling": formal_config["solver_scaling"],
                "provenance_only_release": True,
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
                "stage": "R14B_release",
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
