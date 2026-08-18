"""Run the non-scientific preflight for the exp040 R13 benchmarks."""

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
from scipy.special import h1vp, h2vp, hankel1, hankel2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.run_exp040_r13 import (  # noqa: E402
    scientific_contract_sha256,
    validate_r13_config,
)

from tgv_ptycho.forward.helmholtz_axisymmetric_fem import (  # noqa: E402
    make_axisymmetric_fem_grid,
)
from tgv_ptycho.forward.helmholtz_benchmarks import (  # noqa: E402
    decompose_cylindrical_field,
    normalized_cylindrical_bases,
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

REGISTERED_CONFIG_SHA256 = (
    "5E88C8F7FCEFD794C40DFEC2C908E54E6A1A77302021300C464DEA678D64B0D8"
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
        raise ValueError(f"{name} differs from the R13 preflight registration.")


def validate_preflight_config(config: Mapping[str, Any]) -> None:
    _require_exact(config["experiment"]["id"], "exp040", "experiment.id")
    _require_exact(
        config["experiment"]["stage"], "R13_preflight", "experiment.stage"
    )
    _require_exact(
        config["experiment"]["scientific_result"],
        False,
        "experiment.scientific_result",
    )
    algebra = config["algebra_controls"]
    _require_exact(algebra["low_k_degrees"], [2, 4], "low-k degrees")
    _require_exact(algebra["low_k_modal_wavenumber"], 4.0, "low-k value")
    resource = config["resource_controls"]
    _require_exact(
        resource["formal_tgv_execution_enabled"], False, "formal TGV flag"
    )
    _require_exact(
        resource["cartesian_execution_enabled"], False, "Cartesian flag"
    )
    _require_exact(
        dict(config["thresholds"]),
        {
            "decomposition_relative_error_max": 1.0e-11,
            "derivative_relative_error_max": 1.0e-10,
            "low_k_p4_weighted_relative_l2_max": 1.0e-3,
            "low_k_p4_to_p2_error_ratio_max": 0.25,
            "solve_relative_residual_max": 1.0e-10,
            "require_all_finite": True,
        },
        "preflight thresholds",
    )


def _validate_provenance_and_contract(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], Mapping[str, Any]]:
    provenance = config["provenance"]
    r12_dir = PROJECT_ROOT / str(provenance["r12_run"])
    paths = {
        "r12_metrics": r12_dir / "metrics.json",
        "r12_hdf5": r12_dir / "outputs" / "exp040_r12.h5",
        "r12_cartesian_checkpoint": r12_dir
        / "checkpoints"
        / "chord_fov128_alias.npz",
    }
    expected = {
        "r12_metrics": str(provenance["r12_metrics_sha256"]),
        "r12_hdf5": str(provenance["r12_hdf5_sha256"]),
        "r12_cartesian_checkpoint": str(
            provenance["r12_cartesian_checkpoint_sha256"]
        ),
    }
    actual = {key: _sha256(path) for key, path in paths.items()}
    if actual != expected:
        raise ValueError("R12 provenance differs from R13 preflight registration.")
    formal_path = PROJECT_ROOT / str(config["formal_contract"]["config_path"])
    formal_config = load_config(formal_path)
    validate_r13_config(formal_config)
    contract_hash = scientific_contract_sha256(formal_config)
    if contract_hash != str(
        config["formal_contract"]["scientific_contract_sha256"]
    ):
        raise ValueError("R13 formal scientific contract hash differs.")
    actual["scientific_contract"] = contract_hash
    return actual, formal_config


def _formal_grid_controls(formal_config: Mapping[str, Any]) -> dict[str, Any]:
    domain = formal_config["domain_reflection"]
    domain_unknowns: dict[str, int] = {}
    for case_id in domain["fixed_case_order"]:
        case = domain["cases"][case_id]
        elements = int(
            np.rint(
                (
                    float(case["pml_start_m"])
                    + float(case["pml_thickness_m"])
                    - float(domain["inner_radius_m"])
                )
                / float(domain["element_size_m"])
            )
        )
        domain_unknowns[case_id] = elements * int(domain["degree"]) - 1
    pollution = formal_config["physical_k_pollution"]
    pollution_unknowns: dict[str, int] = {}
    for case_id in pollution["fixed_case_order"]:
        case = pollution["cases"][case_id]
        grid = make_axisymmetric_fem_grid(
            degree=int(case["degree"]),
            radial_extent_m=float(pollution["domain_radial_extent"]),
            z_min_m=0.0,
            z_max_m=float(pollution["domain_axial_extent"]),
            radial_element_size_m=float(case["element_size_ratio"]),
            axial_element_size_m=float(case["element_size_ratio"]),
        )
        pollution_unknowns[case_id] = grid.active_unknown_count
    maximum = max((*domain_unknowns.values(), *pollution_unknowns.values()))
    return {
        "domain_active_unknowns": domain_unknowns,
        "pollution_active_unknowns": pollution_unknowns,
        "maximum_formal_benchmark_unknowns": maximum,
    }


def _run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance_hashes, formal_config = _validate_provenance_and_contract(config)
    algebra = config["algebra_controls"]
    radius = np.asarray(algebra["sample_radii"], dtype=np.float64)
    k_value = float(algebra["cylindrical_wavenumber"])
    inner = float(algebra["inner_radius"])
    outgoing, incoming, outgoing_d, incoming_d = normalized_cylindrical_bases(
        radius,
        wavenumber_per_m=k_value,
        normalization_radius_m=inner,
    )
    expected_outgoing = complex(*algebra["synthetic_outgoing_coefficient"])
    expected_incoming = complex(*algebra["synthetic_incoming_coefficient"])
    field = expected_outgoing * outgoing + expected_incoming * incoming
    derivative = expected_outgoing * outgoing_d + expected_incoming * incoming_d
    recovered_outgoing, recovered_incoming = decompose_cylindrical_field(
        field,
        derivative,
        radius,
        wavenumber_per_m=k_value,
        normalization_radius_m=inner,
    )
    decomposition_error = float(
        max(
            np.max(np.abs(recovered_outgoing - expected_outgoing))
            / abs(expected_outgoing),
            np.max(np.abs(recovered_incoming - expected_incoming))
            / abs(expected_incoming),
        )
    )
    derivative_reference = np.concatenate(
        (
            k_value * h1vp(0, k_value * radius) / hankel1(0, k_value * inner),
            k_value * h2vp(0, k_value * radius) / hankel2(0, k_value * inner),
        )
    )
    derivative_formula = np.concatenate((outgoing_d, incoming_d))
    derivative_error = float(
        np.linalg.norm(derivative_formula - derivative_reference)
        / np.linalg.norm(derivative_reference)
    )
    low_k: dict[str, Any] = {}
    for degree in algebra["low_k_degrees"]:
        result = physical_k_modal_fem_benchmark(
            degree=int(degree),
            element_size_ratio=float(algebra["low_k_element_size"]),
            formal_kh=float(algebra["low_k_modal_wavenumber"]),
            radial_extent=float(algebra["low_k_domain_extent"]),
            axial_extent=float(algebra["low_k_domain_extent"]),
            radial_mode=int(algebra["low_k_radial_mode"]),
            axial_mode=int(algebra["low_k_axial_mode"]),
            complex_amplitude=1.0 + 0.2j,
            discontinuous_mass=False,
            interface_radius=1.73,
            homogeneous_n2=1.0,
            interface_inner_n2=4.0 / 9.0,
            interface_outer_n2=1.0,
            quadrature_order=int(algebra["quadrature_order"]),
            evaluation_count_per_axis=65,
        )
        low_k[f"p{degree}"] = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "radial_coordinates",
                "axial_coordinates",
                "numerical_field",
                "truth_field",
            }
        }
    p2_error = float(low_k["p2"]["weighted_relative_l2"])
    p4_error = float(low_k["p4"]["weighted_relative_l2"])
    p4_to_p2 = p4_error / p2_error
    grids = _formal_grid_controls(formal_config)
    free_disk_gib = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    thresholds = config["thresholds"]
    resource = config["resource_controls"]
    gates = {
        "decomposition_pass": decomposition_error
        <= float(thresholds["decomposition_relative_error_max"]),
        "derivative_pass": derivative_error
        <= float(thresholds["derivative_relative_error_max"]),
        "low_k_fem_pass": bool(
            p4_error <= float(thresholds["low_k_p4_weighted_relative_l2_max"])
            and p4_to_p2
            <= float(thresholds["low_k_p4_to_p2_error_ratio_max"])
            and max(
                float(value["solver_controls"]["relative_residual"])
                for value in low_k.values()
            )
            <= float(thresholds["solve_relative_residual_max"])
            and all(bool(value["all_finite"]) for value in low_k.values())
        ),
        "resource_pass": bool(
            grids["maximum_formal_benchmark_unknowns"]
            <= int(resource["maximum_formal_benchmark_unknowns"])
            and free_disk_gib >= float(resource["minimum_free_disk_gib"])
            and resource["formal_tgv_execution_enabled"] is False
            and resource["cartesian_execution_enabled"] is False
        ),
    }
    formal_allowed = bool(all(gates.values()))
    return {
        "version": "R13_preflight",
        "scientific_result": False,
        "status": "Passed" if formal_allowed else "Failed",
        "formal_r13_allowed": formal_allowed,
        "gates": gates,
        "algebra_controls": {
            "decomposition_relative_error": decomposition_error,
            "derivative_relative_error": derivative_error,
        },
        "low_k_fem": {
            "cases": low_k,
            "p4_to_p2_error_ratio": p4_to_p2,
        },
        "formal_grid_controls": grids,
        "resource_controls": {
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
            raise RuntimeError(f"missing or empty R13 preflight artifact: {relative}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["scientific_result"] is not False:
        raise RuntimeError("R13 preflight was mislabeled as scientific.")
    with h5py.File(
        run_dir / "outputs" / str(config["output"]["hdf5_filename"]), "r"
    ) as handle:
        entry = handle["entry"]
        if set(entry) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
        }:
            raise RuntimeError("R13 preflight HDF5 layout differs.")
        if set(entry["data"]):
            raise RuntimeError("R13 preflight must not contain scientific fields.")


def run(config_path: Path) -> Path:
    """Execute and persist the non-scientific R13 preflight."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R13 preflight source config hash differs.")
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
                "stage": "R13_preflight",
                "state": "running",
                "scientific_result": False,
                "formal_r13_allowed": False,
                "created_at": created_at_utc(),
            },
        )
        metrics = _run_preflight(config)
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": "R13_preflight",
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
                "stage": "R13_preflight",
                "state": "completed",
                "scientific_result": False,
                "formal_r13_allowed": metrics["formal_r13_allowed"],
                "completed_at": created_at_utc(),
            },
        )
        save_ptycho_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            instrument={"formal_grid_controls": metrics["formal_grid_controls"]},
            config_yaml=config_to_yaml(dict(config)),
            metadata=metadata,
            metrics=metrics,
        )
        _validate_artifacts(run_dir, config)
    except Exception:
        save_json(
            state_path,
            {
                "stage": "R13_preflight",
                "state": "failed_during_execution",
                "scientific_result": False,
                "formal_r13_allowed": False,
                "failed_at": created_at_utc(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"run_dir: {run_dir}", flush=True)
    print(f"status: {metrics['status']}", flush=True)
    print(f"formal_r13_allowed: {metrics['formal_r13_allowed']}", flush=True)
    return run_dir


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
