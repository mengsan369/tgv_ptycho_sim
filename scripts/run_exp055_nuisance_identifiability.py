from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

_CONDA_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _CONDA_DLL_DIR.is_dir():
    _path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(_CONDA_DLL_DIR) not in _path_entries:
        os.environ["PATH"] = str(_CONDA_DLL_DIR) + os.pathsep + os.environ.get(
            "PATH", ""
        )

import h5py  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tgv_ptycho.inverse.exp051 import sha256_array_bytes, sha256_file  # noqa: E402
from tgv_ptycho.inverse.exp055 import (  # noqa: E402
    load_exp055_true_probe,
    make_exp055_candidate_generator,
    run_exp055_formal,
    run_exp055_preflight,
    validate_exp055_config,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402

FROZEN_ROOT_CONFIG_SHA256 = (
    "9DD61506C948DD75D464BE176BA1C786A9885D1D845AA6C830908B5D63F5EC6B"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp055 true-probe minimal nuisance identifiability."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "formal"), required=True
    )
    return parser.parse_args()


def _metadata(
    config: dict[str, Any],
    config_path: Path,
    source: Any,
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "experiment_id": "exp055",
        "mode": mode,
        "created_at_utc": created_at_utc(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config_source_path": str(config_path),
        "config_source_sha256": sha256_file(config_path),
        "source_run": str(source.run_dir),
        "source_hdf5_path": str(source.hdf5_path),
        "source_target_hdf5_path": source.target_hdf5_path,
        "source_file_sha256": source.file_sha256,
        "source_target_dataset_sha256": source.target_dataset_sha256,
        "source_exp040_branch": config["source"]["expected_identity"][
            "source_exp040_branch"
        ],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "simulation_evaluation_only": mode == "formal",
    }


def _formal_metrics(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    runtime_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    interval = result["equivalence"]["projected_d_waist_interval"]
    branches = result["multistart"]["branches"]
    return {
        "experiment_status": result["experiment_status"],
        "interpretation": result["interpretation"],
        "stage": "true_probe",
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "primary_loss": config["fit"]["primary_loss"],
        "denominator": config["fit"]["denominator"],
        "phase_or_scale_alignment": "none",
        "parameter_order": ["d_waist_m", "d_surface_m", "z_waist_m"],
        "parameter_bounds": {
            "d_waist_m": config["parameters"]["target"]["bounds_m"],
            "d_surface_m": config["parameters"]["nuisance"][0]["bounds_m"],
            "z_waist_m": config["parameters"]["nuisance"][1]["bounds_m"],
        },
        "source_replay": result["source_replay"],
        "equivalence_relative_l2": config["fit"][
            "equivalence_relative_l2"
        ],
        "local_component_count": result["equivalence"][
            "local_component_count"
        ],
        "external_coarse_component_count": result["equivalence"][
            "external_coarse_component_count"
        ],
        "global_screened_component_count": result["equivalence"][
            "global_screened_component_count"
        ],
        "projected_d_waist_interval": interval,
        "best_candidate": result["best_candidate"],
        "multistart": {
            name: {
                "terminal_index": branch["terminal_index"],
                "terminal_component_label": branch[
                    "terminal_component_label"
                ],
                "terminal_qualifies": branch["terminal_qualifies"],
                "move_count": branch["move_count"],
                "stopping_reason": branch["stopping_reason"],
            }
            for name, branch in branches.items()
        },
        "local_svd": {
            "normalized_singular_values": result["local_diagnostic"][
                "normalized_singular_values"
            ],
            "condition_number_diagnostic": result["local_diagnostic"][
                "condition_number_diagnostic"
            ],
            "enters_status_gate": False,
        },
        "simulation_evaluation_only": result[
            "simulation_evaluation_only"
        ],
        "gates": result["gates"],
        "runtime_seconds": runtime_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "cache_entry_count": result["cache"]["entry_count"],
    }


def _preflight_metrics(
    result: dict[str, Any],
    *,
    runtime_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    return {
        "preflight_status": result["preflight_status"],
        "scientific_status": "NotEvaluated",
        "interpretation": result["interpretation"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "scientific_result_used_for_threshold_or_method_selection": False,
        "replay_relative_l2": result["replay_relative_l2"],
        "deterministic_repeat_relative_l2": result[
            "deterministic_repeat_relative_l2"
        ],
        "point_count": len(result["points_m"]),
        "gates": result["gates"],
        "runtime_seconds": runtime_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "cache_entry_count": result["cache"]["entry_count"],
    }


def _result_payload(
    config: dict[str, Any],
    source: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": {
            "run": str(source.run_dir),
            "hdf5_path": str(source.hdf5_path),
            "raw_dataset_path": source.target_hdf5_path,
            "raw_dataset_sha256": source.target_dataset_sha256,
            "file_sha256": source.file_sha256,
            "P_B_true_raw_input": source.P_B_true,
            "operator_identity": config["source"]["expected_identity"],
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "p_b_rec_used_as_primary_input": False,
        },
        "frozen_design": {
            "parameter_definitions": config["parameters"],
            "fit": config["fit"],
            "thresholds": config["thresholds"],
            "status_logic": config["status_logic"],
            "numerical_controls": {
                "runtime_seconds_max": config["output"][
                    "expected_runtime_seconds_max"
                ],
                "peak_memory_bytes_max": config["output"][
                    "expected_peak_memory_bytes_max"
                ],
                "hdf5_bytes_max": config["output"][
                    "expected_hdf5_bytes_max"
                ],
            },
        },
        "result": result,
    }


def _hdf_safe(value: Any) -> Any:
    """Convert list-of-mapping config structures into named HDF5 groups."""

    if isinstance(value, dict):
        return {str(key): _hdf_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)) and any(
        isinstance(child, dict) for child in value
    ):
        return {
            f"item_{index:03d}": _hdf_safe(child)
            for index, child in enumerate(value)
        }
    if isinstance(value, (list, tuple)):
        return [_hdf_safe(child) for child in value]
    return value


def _save_figures(
    result: dict[str, Any], figure_paths: list[Path], *, mode: str
) -> None:
    if mode == "preflight":
        points = np.asarray(result["points_m"]) * 1.0e6
        rho = np.asarray(result["raw_relative_l2"])
        plots = [
            (np.arange(len(rho)), rho, "Registered seven-point loss", "point"),
            (points[:, 0], rho, "D_waist guards", "D_waist [um]"),
            (points[:, 1], rho, "D_surface guards", "D_surface [um]"),
            (points[:, 2], rho, "z_waist guards", "z_waist [um]"),
            (
                np.arange(len(rho)),
                np.maximum(rho, np.finfo(float).tiny),
                "Finite/read-back control",
                "point",
            ),
        ]
        for path, (x, y, title, xlabel) in zip(
            figure_paths, plots, strict=True
        ):
            fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            ax.plot(x, np.maximum(y, np.finfo(float).tiny), "o-")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("raw relative L2")
            ax.set_title(f"PRE-FLIGHT ONLY — {title}")
            fig.savefig(path, dpi=150)
            plt.close(fig)
        return

    coarse = result["coarse_grid"]
    local = result["local_grid"]
    profile = result["profile"]
    interval = result["equivalence"]["projected_d_waist_interval"]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    coarse_profile = np.min(np.asarray(coarse["loss"]), axis=(1, 2))
    ax.semilogy(
        np.asarray(coarse["d_waist_m"]) * 1.0e6,
        np.maximum(coarse_profile, np.finfo(float).tiny),
        "o-",
    )
    ax.set(
        xlabel="D_waist [um]",
        ylabel="profiled raw loss",
        title="Global registered nuisance profile",
    )
    fig.savefig(figure_paths[0], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.semilogy(
        np.asarray(profile["d_waist_m"]) * 1.0e6,
        np.maximum(np.asarray(profile["minimum_loss"]), np.finfo(float).tiny),
        "o-",
    )
    if interval["connected"]:
        ax.axvspan(
            interval["lower_m"] * 1.0e6,
            interval["upper_m"] * 1.0e6,
            color="tab:green",
            alpha=0.25,
            label="projected equivalence interval",
        )
        ax.legend()
    ax.set(
        xlabel="D_waist [um]",
        ylabel="profiled raw loss",
        title=f"D_waist profile — {result['experiment_status']}",
    )
    fig.savefig(figure_paths[1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    pair = np.min(np.asarray(local["loss"]), axis=0)
    image = ax.imshow(
        np.log10(np.maximum(pair, np.finfo(float).tiny)),
        origin="lower",
        aspect="auto",
        extent=[
            local["z_waist_m"][0] * 1.0e6,
            local["z_waist_m"][-1] * 1.0e6,
            local["d_surface_m"][0] * 1.0e6,
            local["d_surface_m"][-1] * 1.0e6,
        ],
    )
    fig.colorbar(image, ax=ax, label="log10 profiled raw loss")
    ax.set(
        xlabel="z_waist [um]",
        ylabel="D_surface [um]",
        title="Nuisance-pair profile map",
    )
    fig.savefig(figure_paths[2], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for name, branch in result["multistart"]["branches"].items():
        ax.semilogy(
            np.arange(len(branch["visited_loss"])),
            np.maximum(branch["visited_loss"], np.finfo(float).tiny),
            "o-",
            label=name,
        )
    ax.legend()
    ax.set(
        xlabel="accepted lattice move",
        ylabel="raw loss",
        title="Frozen four-start lattice paths",
    )
    fig.savefig(figure_paths[3], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    singular = np.asarray(result["local_diagnostic"]["normalized_singular_values"])
    ax.semilogy(np.arange(1, len(singular) + 1), singular, "o-")
    ax.set(
        xlabel="singular index",
        ylabel="normalized singular value",
        title="Diagnostic-only local finite-change SVD",
    )
    fig.savefig(figure_paths[4], dpi=150)
    plt.close(fig)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_json_group(group: h5py.Group, payload: dict[str, Any]) -> list[str]:
    differences: list[str] = []
    for key, expected in payload.items():
        if key not in group:
            differences.append(f"missing:{group.name}/{key}")
            continue
        item = group[key]
        if isinstance(expected, dict):
            if not isinstance(item, h5py.Group):
                differences.append(f"not_group:{item.name}")
            else:
                differences.extend(_compare_json_group(item, expected))
            continue
        actual = item[...]
        if np.asarray(actual).ndim == 0:
            actual = _decode(actual.item())
        else:
            actual_array = np.asarray(
                [_decode(value) for value in np.ravel(actual)]
            ).reshape(np.asarray(actual).shape)
            actual = actual_array.tolist()
        expected_array = np.asarray(expected)
        expected_value: Any = expected
        if expected_array.ndim > 0:
            expected_value = expected_array.tolist()
        if actual != expected_value:
            differences.append(f"value:{item.name}")
    return differences


def _validate_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    source: Any,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    result: dict[str, Any],
    figure_paths: list[Path],
) -> None:
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    with h5py.File(hdf5_path, "r") as h5:
        expected_children = {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "reconstruction",
            "sample",
            "truth",
        }
        if set(h5["/entry"].keys()) != expected_children:
            raise RuntimeError("exp055 HDF5 top-level layout differs.")
        if len(h5["/entry/data"].keys()) != 0:
            raise RuntimeError("exp055 must not fabricate detector data.")
        base = "/entry/reconstruction/waist_fit/nuisance_identifiability/true_probe"
        if base not in h5:
            raise RuntimeError("exp055 result group is missing.")
        raw = np.asarray(h5[f"{base}/source/P_B_true_raw_input"][...])
        best = np.asarray(h5[f"{base}/result/P_B_best_raw"][...])
        residual = np.asarray(h5[f"{base}/result/residual_field_raw"][...])
        if not np.array_equal(raw, source.P_B_true):
            raise RuntimeError("saved raw P_B_true differs from source.")
        if sha256_array_bytes(raw) != source.target_dataset_sha256:
            raise RuntimeError("saved raw P_B_true hash differs from source.")
        if not np.array_equal(best - raw, residual):
            raise RuntimeError("saved best/raw residual identity failed.")
        if _compare_json_group(h5["/entry/metadata"], metadata):
            raise RuntimeError("external metadata and HDF5 metadata differ.")
        if _compare_json_group(h5["/entry/metrics"], metrics):
            raise RuntimeError("external metrics and HDF5 metrics differ.")
        forbidden = ("aligned", "global_phase", "i_stack", "scan_positions")
        numeric_finite = True

        def visit(name: str, item: h5py.Dataset | h5py.Group) -> None:
            nonlocal numeric_finite
            lowered = name.lower()
            if any(token in lowered for token in forbidden):
                raise RuntimeError(f"forbidden exp055 artifact path: {name}")
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "biufc":
                numeric_finite = numeric_finite and bool(
                    np.all(np.isfinite(item[...]))
                )

        h5.visititems(visit)
        if not numeric_finite:
            raise RuntimeError("exp055 HDF5 contains non-finite numeric data.")
    if not np.array_equal(
        np.asarray(result["P_B_best_raw"]) - source.P_B_true,
        np.asarray(result["residual_field_raw"]),
    ):
        raise RuntimeError("in-memory best/raw residual identity failed.")
    for path in figure_paths:
        image = plt.imread(path)
        if image.size == 0 or not np.all(np.isfinite(image)):
            raise RuntimeError(f"figure read-back failed: {path.name}")


def _write_hdf5(
    run_dir: Path,
    config: dict[str, Any],
    source: Any,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    result: dict[str, Any],
) -> Path:
    path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    identity = config["source"]["expected_identity"]
    save_ptycho_hdf5(
        path,
        instrument={
            "wavelength_m": source.source_config["optics"]["wavelength_m"],
            "internal_reference_index": source.source_config["optics"][
                "internal_reference_index"
            ],
            "external_medium_index": source.source_config["optics"][
                "external_medium_index"
            ],
            "z_AB_m": source.source_config["optics"]["z_AB_m"],
            "probe_grid": identity,
        },
        sample=_hdf_safe({
            "sample_a_nominal": source.source_config["sample_a"],
            "fitted_parameter_definitions": config["parameters"],
        }),
        truth={
            "identity": config["experiment"]["truth_identity"],
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "P_B_true": source.P_B_true,
            "D_z_m": source.D_z_m,
            "z_m": source.z_m,
            "slice_widths_m": source.slice_widths_m,
        },
        reconstruction=_hdf_safe({
            "waist_fit": {
                "nuisance_identifiability": {
                    "true_probe": _result_payload(config, source, result)
                }
            }
        }),
        config_yaml=config_to_yaml(config),
        metadata=metadata,
        metrics=metrics,
    )
    return path


def run(config_path: Path, *, mode: str) -> Path:
    resolved = config_path
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    if sha256_file(resolved) != FROZEN_ROOT_CONFIG_SHA256:
        raise RuntimeError("exp055 frozen root config SHA256 changed.")
    config = load_config(resolved)
    validate_exp055_config(config)
    output_root = Path(config["output"]["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_name = config["output"]["run_name"]
    if mode == "preflight":
        run_name += "_preflight"
    run_dir = make_run_dir(output_root, run_name)
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {"status": "running", "artifacts_validated": False, "mode": mode},
    )
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()

    def sample_memory() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)

    try:
        source = load_exp055_true_probe(config, PROJECT_ROOT)
        generator = make_exp055_candidate_generator(
            source.source_config, memory_callback=sample_memory
        )
        if mode == "preflight":
            result = run_exp055_preflight(config, source.P_B_true, generator)
        else:
            result = run_exp055_formal(
                config, source.source_config, source.P_B_true, generator
            )
        sample_memory()
        runtime_seconds = time.perf_counter() - started
        runtime_pass = bool(
            runtime_seconds
            <= float(config["output"]["expected_runtime_seconds_max"])
        )
        memory_pass = bool(
            peak_rss
            <= int(config["output"]["expected_peak_memory_bytes_max"])
        )
        result["gates"]["runtime_contract_pass"] = runtime_pass
        result["gates"]["memory_contract_pass"] = memory_pass
        if not runtime_pass or not memory_pass:
            if mode == "preflight":
                result["preflight_status"] = "PreflightFailed"
            else:
                result["experiment_status"] = "Inconclusive"
                result["interpretation"] = "nuisance_numerical_control_not_closed"
        metadata = _metadata(config, resolved, source, mode=mode)
        metrics = (
            _preflight_metrics(
                result,
                runtime_seconds=runtime_seconds,
                peak_rss_bytes=peak_rss,
            )
            if mode == "preflight"
            else _formal_metrics(
                config,
                result,
                runtime_seconds=runtime_seconds,
                peak_rss_bytes=peak_rss,
            )
        )
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        hdf5_path = _write_hdf5(
            run_dir, config, source, metadata, metrics, result
        )
        hdf5_size_pass = bool(
            hdf5_path.stat().st_size
            <= int(config["output"]["expected_hdf5_bytes_max"])
        )
        if not hdf5_size_pass:
            raise RuntimeError("exp055 HDF5 exceeded the frozen size contract.")
        result["gates"]["hdf5_size_contract_pass"] = hdf5_size_pass
        metrics["gates"] = result["gates"]
        save_json(run_dir / "metrics.json", metrics)
        hdf5_path = _write_hdf5(
            run_dir, config, source, metadata, metrics, result
        )
        figure_paths = [
            run_dir / "figures" / name
            for name in config["output"]["figure_filenames"]
        ]
        _save_figures(result, figure_paths, mode=mode)
        _validate_artifacts(
            run_dir,
            config,
            source,
            metadata,
            metrics,
            result,
            figure_paths,
        )
        state = {
            "status": "complete",
            "artifacts_validated": True,
            "mode": mode,
            "completed_at_utc": created_at_utc(),
            "runtime_seconds": runtime_seconds,
            "config_sha256": sha256_file(run_dir / "config.yaml"),
            "metadata_sha256": sha256_file(run_dir / "metadata.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
            "hdf5_sha256": sha256_file(hdf5_path),
            "hdf5_bytes": hdf5_path.stat().st_size,
            "figure_sha256": {
                path.name: sha256_file(path) for path in figure_paths
            },
            "figure_count": len(figure_paths),
        }
        if mode == "preflight":
            state["preflight_status"] = result["preflight_status"]
            state["scientific_status"] = "NotEvaluated"
        else:
            state["experiment_status"] = result["experiment_status"]
            state["interpretation"] = result["interpretation"]
        save_json(run_dir / "run_state.json", state)
    except Exception as error:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "artifacts_validated": False,
                "mode": mode,
                "failed_at_utc": created_at_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    print(f"run_dir: {run_dir}")
    if mode == "preflight":
        print(f"preflight_status: {result['preflight_status']}")
    else:
        print(f"experiment_status: {result['experiment_status']}")
        print(f"interpretation: {result['interpretation']}")
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config, mode=args.mode)


if __name__ == "__main__":
    main()
