"""Run the pre-registered exp040 R14A axial-control attribution."""

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
import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.helmholtz_benchmarks import (  # noqa: E402
    axial_plane_wave_pml_benchmark,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.viz.plot_exp040_r14a import (  # noqa: E402
    EXP040_R14A_FIGURE_FILENAME,
    save_exp040_r14a_figure,
)

REGISTERED_CONFIG_SHA256 = (
    "DE60B9FD75981F3EDD24B4297ABC5CD0AB42253484E96537E7D692597F39607D"
)
_ARRAY_KEYS = {
    "measurement_coordinates_m",
    "incoming_to_outgoing_ratio",
    "outgoing_impedance_residual",
    "dense_coordinates_m",
    "dense_field",
    "dense_truth",
}


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
        raise ValueError(f"{name} differs from the R14A registration.")


def scientific_contract_sha256(config: Mapping[str, Any]) -> str:
    """Hash only the pre-registered R14A scientific sections."""

    contract = {
        key: config[key]
        for key in (
            "axial_attribution",
            "thresholds",
            "conditional_execution",
        )
    }
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_r14a_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the registered R14A attribution."""

    _require_exact(config["experiment"]["id"], "exp040", "experiment.id")
    _require_exact(config["experiment"]["stage"], "R14A", "stage")
    _require_exact(config["experiment"]["scientific_result"], True, "result")
    axial = config["axial_attribution"]
    _require_exact(axial["degree"], 4, "degree")
    _require_exact(axial["refractive_index"], 1.5, "refractive index")
    _require_exact(axial["direction"], -1, "direction")
    _require_exact(
        axial["fixed_case_order"],
        ["glass_h16_pml2", "glass_h24_pml2", "glass_h24_pml3"],
        "case order",
    )
    _require_exact(
        axial["mesh_convergence_case_order"],
        [
            "locked_glass_h12_pml2",
            "glass_h16_pml2",
            "glass_h24_pml2",
        ],
        "mesh order",
    )
    _require_exact(
        dict(config["thresholds"]),
        {
            "derivative_metric_convergence_order_min": 3.0,
            "field_metric_convergence_order_min": 4.0,
            "incoming_to_outgoing_ratio_max": 1.0e-3,
            "outgoing_impedance_residual_max": 1.0e-3,
            "dense_field_relative_l2_max": 1.0e-3,
            "pml2_to_pml3_raw_field_relative_l2_max": 1.0e-4,
            "direct_solver_relative_residual_max": 1.0e-10,
            "require_all_finite": True,
        },
        "thresholds",
    )
    for key, value in config["conditional_execution"].items():
        _require_exact(value, False, key)
    _require_exact(
        config["output"]["figure_filename"],
        EXP040_R14A_FIGURE_FILENAME,
        "figure filename",
    )


def _load_locked_baseline(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    run_dir = PROJECT_ROOT / str(provenance["failed_preflight_run"])
    metrics_path = run_dir / "metrics.json"
    hdf5_path = run_dir / "outputs" / "exp040_r14_preflight.h5"
    if _sha256(metrics_path) != str(
        provenance["failed_preflight_metrics_sha256"]
    ):
        raise ValueError("R14 failed-preflight metrics hash differs.")
    if _sha256(hdf5_path) != str(provenance["failed_preflight_hdf5_sha256"]):
        raise ValueError("R14 failed-preflight HDF5 hash differs.")
    with metrics_path.open("r", encoding="utf-8") as handle:
        preflight = json.load(handle)
    if preflight["formal_r14_allowed"] is not False:
        raise ValueError("R14A requires the locked failed R14 preflight.")
    if preflight["gates"]["axial_pml_smoke_pass"] is not False:
        raise ValueError("R14A axial failure provenance differs.")
    case_id = str(provenance["locked_baseline_case"])
    case = dict(preflight["axial_pml_smoke"]["cases"][case_id])
    case["element_size_m"] = float(case["kh"] / case["wavenumber_per_m"])
    return {
        "case_id": case_id,
        "metrics": case,
        "hashes": {
            "failed_preflight_metrics": _sha256(metrics_path),
            "failed_preflight_hdf5": _sha256(hdf5_path),
        },
    }


def _run_case(
    common: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    return axial_plane_wave_pml_benchmark(
        wavelength_m=float(common["wavelength_m"]),
        refractive_index=float(common["refractive_index"]),
        direction=int(common["direction"]),
        physical_core_length_m=float(common["physical_core_length_m"]),
        pml_thickness_m=float(case["pml_thickness_m"]),
        element_size_m=float(case["element_size_m"]),
        degree=int(common["degree"]),
        quadrature_order=int(common["quadrature_order"]),
        pml_polynomial_order=int(common["polynomial_order"]),
        pml_target_one_way_amplitude=float(
            common["target_one_way_amplitude"]
        ),
        measurement_fractions=np.asarray(
            common["measurement_fractions"], dtype=np.float64
        ),
        dense_comparison_count=int(common["dense_comparison_count"]),
    )


def _fit_order(element_sizes: np.ndarray, errors: np.ndarray) -> float:
    design = np.column_stack(
        (np.ones(element_sizes.size), np.log(element_sizes))
    )
    return float(np.linalg.lstsq(design, np.log(errors), rcond=None)[0][1])


def _execute(
    config: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    axial = config["axial_attribution"]
    case_metrics: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    for case_id in axial["fixed_case_order"]:
        result = _run_case(axial, axial["cases"][case_id])
        case_metrics[case_id] = {
            key: value for key, value in result.items() if key not in _ARRAY_KEYS
        }
        case_metrics[case_id]["element_size_m"] = float(
            axial["cases"][case_id]["element_size_m"]
        )
        case_metrics[case_id]["pml_thickness_m"] = float(
            axial["cases"][case_id]["pml_thickness_m"]
        )
        arrays[case_id] = {key: result[key] for key in _ARRAY_KEYS}

    baseline_metrics = baseline["metrics"]
    pml2_ids = ["glass_h16_pml2", "glass_h24_pml2"]
    convergence_cases = [baseline_metrics] + [
        case_metrics[case_id] for case_id in pml2_ids
    ]
    element_sizes = np.asarray(
        [case["element_size_m"] for case in convergence_cases]
    )
    metric_keys = (
        "maximum_incoming_to_outgoing_ratio",
        "maximum_outgoing_impedance_residual",
        "dense_field_relative_l2",
    )
    errors = {
        key: np.asarray([case[key] for case in convergence_cases])
        for key in metric_keys
    }
    orders = {key: _fit_order(element_sizes, value) for key, value in errors.items()}
    pml2_field = np.asarray(arrays["glass_h24_pml2"]["dense_field"])
    pml3_field = np.asarray(arrays["glass_h24_pml3"]["dense_field"])
    pml_pair_l2 = float(
        np.linalg.norm(pml2_field - pml3_field)
        / max(np.linalg.norm(pml3_field), np.finfo(float).eps)
    )
    thresholds = config["thresholds"]
    order_pass = bool(
        orders["maximum_incoming_to_outgoing_ratio"]
        >= float(thresholds["derivative_metric_convergence_order_min"])
        and orders["maximum_outgoing_impedance_residual"]
        >= float(thresholds["derivative_metric_convergence_order_min"])
        and orders["dense_field_relative_l2"]
        >= float(thresholds["field_metric_convergence_order_min"])
    )
    corrected_cases = ("glass_h24_pml2", "glass_h24_pml3")
    original_metric_gate_pass = bool(
        all(
            case_metrics[case_id]["maximum_incoming_to_outgoing_ratio"]
            <= float(thresholds["incoming_to_outgoing_ratio_max"])
            and case_metrics[case_id]["maximum_outgoing_impedance_residual"]
            <= float(thresholds["outgoing_impedance_residual_max"])
            and case_metrics[case_id]["dense_field_relative_l2"]
            <= float(thresholds["dense_field_relative_l2_max"])
            for case_id in corrected_cases
        )
    )
    pml_separation_pass = bool(
        pml_pair_l2
        <= float(thresholds["pml2_to_pml3_raw_field_relative_l2_max"])
    )
    hard_controls_pass = bool(
        all(
            value["solver_controls"]["relative_residual"]
            <= float(thresholds["direct_solver_relative_residual_max"])
            and value["all_finite"]
            for value in case_metrics.values()
        )
        and np.all(np.isfinite(list(orders.values())))
        and np.isfinite(pml_pair_l2)
    )
    eligible = bool(
        order_pass
        and original_metric_gate_pass
        and pml_separation_pass
        and hard_controls_pass
    )
    if eligible:
        interpretation = "q4_derivative_floor_attributed__corrected_control_eligible"
    elif not order_pass:
        interpretation = "q4_order_not_confirmed"
    elif not original_metric_gate_pass:
        interpretation = "corrected_axial_control_above_original_gate"
    elif not pml_separation_pass:
        interpretation = "pml_thickness_dependence_material"
    else:
        interpretation = "r14a_hard_controls_failed"
    metrics = {
        "version": "R14A",
        "scientific_result": True,
        "status": "Passed" if eligible else "Failed",
        "interpretation_code": interpretation,
        "corrected_axial_control_eligible": eligible,
        "gates": {
            "q4_order_attribution_pass": order_pass,
            "original_metric_gate_pass": original_metric_gate_pass,
            "pml_separation_pass": pml_separation_pass,
            "hard_controls_pass": hard_controls_pass,
        },
        "locked_baseline": baseline_metrics,
        "new_cases": case_metrics,
        "mesh_convergence": {
            "fixed_case_order": list(axial["mesh_convergence_case_order"]),
            "element_sizes_m": element_sizes,
            "incoming_to_outgoing_ratio": errors[
                "maximum_incoming_to_outgoing_ratio"
            ],
            "outgoing_impedance_residual": errors[
                "maximum_outgoing_impedance_residual"
            ],
            "dense_field_relative_l2": errors["dense_field_relative_l2"],
            "orders": orders,
        },
        "pml_separation": {
            "fixed_case_pair": ["glass_h24_pml2", "glass_h24_pml3"],
            "raw_field_relative_l2": pml_pair_l2,
        },
        "provenance_hashes": dict(baseline["hashes"]),
        "conditional_execution": {
            "formal_solver_scaling_executed": False,
            "full_tgv_executed": False,
            "r12_or_r13_rerun": False,
            "failed_preflight_rerun": False,
            "scalar_cross_model_executed": False,
            "vector_model_executed": False,
        },
        "thresholds": dict(thresholds),
    }
    return metrics, arrays


def _write_hdf5(
    path: Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> None:
    truth = {
        case_id: {
            "coordinates_m": values["dense_coordinates_m"],
            "analytic_field": values["dense_truth"],
        }
        for case_id, values in arrays.items()
    }
    save_ptycho_hdf5(
        path,
        instrument={"axial_attribution": dict(config["axial_attribution"])},
        sample={
            "kind": "analytic_axial_pml_attribution",
            "contains_canonical_tgv_field": False,
        },
        truth=truth,
        config_yaml=config_to_yaml(dict(config)),
        metadata=dict(metadata),
        metrics=dict(metrics),
    )
    with h5py.File(path, "a") as handle:
        data = handle["entry/data"].require_group("axial_attribution")
        for case_id, values in arrays.items():
            group = data.require_group(case_id)
            for key, value in values.items():
                if key != "dense_truth":
                    group.create_dataset(key, data=np.asarray(value))


def _write_checkpoint(
    path: Path, metrics: Mapping[str, Any], arrays: Mapping[str, Any]
) -> None:
    case_id = "glass_h24_pml2"
    case_metrics = metrics["new_cases"][case_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        metrics_json=np.asarray(
            json.dumps(case_metrics, sort_keys=True, default=lambda x: x.item())
        ),
        **{key: np.asarray(value) for key, value in arrays[case_id].items()},
    )


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    for relative in config["output"]["required_files"]:
        path = run_dir / str(relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty R14A artifact: {relative}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["scientific_result"] is not True:
        raise RuntimeError("R14A metrics lost scientific-result status.")
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as handle:
        if set(handle["entry"]) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }:
            raise RuntimeError("R14A HDF5 entry layout differs.")
        if set(handle["entry/data"]) != {"axial_attribution"}:
            raise RuntimeError("R14A HDF5 data layout differs.")
    image = np.asarray(
        iio.imread(run_dir / "figures" / EXP040_R14A_FIGURE_FILENAME)
    )
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 100:
        raise RuntimeError("invalid R14A figure.")
    checkpoint = np.load(
        run_dir
        / "checkpoints"
        / str(config["output"]["checkpoint_filename"]),
        allow_pickle=False,
    )
    if str(checkpoint["case_id"]) != "glass_h24_pml2":
        raise RuntimeError("R14A checkpoint case differs.")
    if not np.all(np.isfinite(checkpoint["dense_field"])):
        raise RuntimeError("R14A checkpoint field is non-finite.")


def run(config_path: Path) -> Path:
    """Execute the single formal R14A attribution run."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R14A source config hash differs.")
    config = load_config(source)
    validate_r14a_config(config)
    baseline = _load_locked_baseline(config)
    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    state_path = run_dir / "run_state.json"
    started = time.perf_counter()
    try:
        save_json(
            state_path,
            {
                "stage": "R14A",
                "state": "running",
                "scientific_result": True,
                "formal_execution_count": 1,
                "created_at": created_at_utc(),
            },
        )
        metrics, arrays = _execute(config, baseline)
        metrics["total_execution_elapsed_s"] = float(
            time.perf_counter() - started
        )
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": "R14A",
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
        _write_checkpoint(
            run_dir
            / "checkpoints"
            / str(config["output"]["checkpoint_filename"]),
            metrics,
            arrays,
        )
        save_exp040_r14a_figure(run_dir / "figures", metrics, arrays)
        save_json(
            state_path,
            {
                "stage": "R14A",
                "state": "completed",
                "scientific_result": True,
                "formal_execution_count": 1,
                "status": metrics["status"],
                "interpretation_code": metrics["interpretation_code"],
                "corrected_axial_control_eligible": metrics[
                    "corrected_axial_control_eligible"
                ],
                "completed_at": created_at_utc(),
            },
        )
        _validate_artifacts(run_dir, config)
    except Exception:
        save_json(
            state_path,
            {
                "stage": "R14A",
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
    return run_dir


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
