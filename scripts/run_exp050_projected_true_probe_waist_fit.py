"""Run exp050 projected-model raw true-probe waist fitting."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.inverse.exp050 import (
    load_exp030_source,
    make_exp050_candidate_generator,
    run_exp050_estimator,
    sha256_array,
    sha256_file,
    validate_exp050_config,
)
from tgv_ptycho.io.config import config_to_yaml, load_config, save_config
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit
from tgv_ptycho.io.naming import make_run_dir
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    return parser.parse_args()


def _small_result(result: dict[str, Any], true_waist_m: float) -> dict[str, Any]:
    finite_difference = dict(result.get("finite_difference", {}))
    finite_difference.pop("jacobian_h1", None)
    finite_difference.pop("jacobian_h2", None)
    return {
        "status": result["status"],
        "interpretation": result["interpretation"],
        "replay": result["replay"],
        "replay_pass": bool(result["replay_pass"]),
        "profile": result.get("profile", {}),
        "finite_difference": finite_difference,
        "estimator_crosscheck": result.get("estimator_crosscheck", {}),
        "D_waist_true_m": true_waist_m,
        "D_waist_estimate_m": result.get("estimate_m"),
        "absolute_error_m": (
            abs(float(result["estimate_m"]) - true_waist_m)
            if "estimate_m" in result
            else None
        ),
        "candidate_cache_count": int(len(result["cache"]["D_waist_m"])),
    }


def _source_identity(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "exp030",
        "run": str(source["run_dir"].relative_to(PROJECT_ROOT)),
        "hdf5": source["paths"]["hdf5"],
        "target_dataset": "/entry/truth/P_B_true",
        "target_bytes_sha256": source["target_bytes_sha256"],
        "file_sha256": source["hashes"],
        "source_git_commit": source["metadata"]["git_commit"],
        "source_run_status": source["state"]["status"],
        "source_experiment_status": source["state"]["experiment_status"],
        "source_artifacts_validated_field_present": bool(
            source["source_artifacts_validated_field_present"]
        ),
        "target_shape_ny_nx": list(source["shape_ny_nx"]),
        "target_dtype": "complex128",
        "axis_order": source["axis_order"],
        "dx_m": source["instrument"]["dx_m"],
        "fov_yx_m": list(source["fov_yx_m"]),
        "coordinate_convention": source["coordinate_convention"],
        "coordinate_endpoints_x_m": list(source["coordinate_endpoints_x_m"]),
        "coordinate_endpoints_y_m": list(source["coordinate_endpoints_y_m"]),
        "target_dataset_units_attribute_present": bool(
            source["target_dataset_units_attribute_present"]
        ),
        "field_units_semantic": "arbitrary complex field amplitude",
        "phase_units_semantic": "rad",
        "operator": (
            "continuous_axisymmetric_fresnel_hankel_on_compact_T_minus_1"
        ),
        "reference": "infinite_plane_wave_propagated_analytically",
        "coarse_A_effective_true_used_as_forward_input": False,
        "P_B_rec_used": False,
        "detector_data_used": False,
        "sample_B_used": False,
        "scan_used": False,
    }


def _plot_profile(result: dict[str, Any], path: Path) -> None:
    profile = result["profile"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    floor = np.finfo(np.float64).tiny
    axes[0].semilogy(
        np.asarray(profile["global_d_waist_m"]) * 1e6,
        np.maximum(np.asarray(profile["global_loss"]), floor),
        "o-",
        label="registered global profile",
    )
    axes[1].semilogy(
        np.asarray(profile["fine_d_waist_m"]) * 1e6,
        np.maximum(np.asarray(profile["fine_loss"]), floor),
        "o-",
        label="registered fine profile",
    )
    interval = np.asarray(profile["reported_resolution_interval_m"]) * 1e6
    for axis in axes:
        axis.axvspan(interval[0], interval[1], color="tab:green", alpha=0.2)
        axis.set_xlabel(r"$D_{waist}$ ($\mu$m)")
        axis.set_ylabel("raw complex normalized squared L2")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_title("Global registered range")
    axes[1].set_title("Local profile and grid-resolution interval")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_best_fit(target: np.ndarray, result: dict[str, Any], path: Path) -> None:
    best = np.asarray(result["P_B_best_raw"])
    residual = np.asarray(result["residual_field_raw"])
    panels = [
        (np.abs(target), "target amplitude", "viridis"),
        (np.abs(best), "best raw amplitude", "viridis"),
        (np.abs(residual), "raw residual amplitude", "magma"),
        (np.angle(target), "target phase (rad)", "twilight"),
        (np.angle(best), "best raw phase (rad)", "twilight"),
        (
            np.angle(best * np.conj(target)),
            "raw phase difference (rad)",
            "twilight",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes.ravel(), panels, strict=True):
        image = axis.imshow(values, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_numerical_controls(result: dict[str, Any], path: Path) -> None:
    fd = result["finite_difference"]
    j1 = np.asarray(fd["jacobian_h1"])
    j2 = np.asarray(fd["jacobian_h2"])
    fields = [np.abs(j1), np.abs(j2), np.abs(j1 - j2)]
    titles = [r"$|J_{h1}|$", r"$|J_{h2}|$", r"$|J_{h1}-J_{h2}|$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, values, title in zip(axes, fields, titles, strict=True):
        image = axis.imshow(values, origin="lower", cmap="magma")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle(
        "FD step relative L2 = "
        f"{float(fd['jacobian_step_relative_l2']):.3e}; "
        "signature/floor = "
        f"{float(fd['signature_to_replay_floor_ratio']):.3e}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_crosscheck(result: dict[str, Any], path: Path) -> None:
    search = result["estimator_crosscheck"]
    lower = np.asarray(search["lower_m"]) * 1e6
    upper = np.asarray(search["upper_m"]) * 1e6
    iterations = np.arange(len(lower))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(iterations, lower, "o-", label="lower bracket")
    axes[0].plot(iterations, upper, "o-", label="upper bracket")
    axes[0].fill_between(iterations, lower, upper, alpha=0.2)
    axes[0].set_xlabel("fixed golden iteration")
    axes[0].set_ylabel(r"$D_{waist}$ ($\mu$m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].semilogy(
        np.asarray(search["evaluated_d_waist_m"]) * 1e6,
        np.maximum(np.asarray(search["evaluated_loss"]), np.finfo(float).tiny),
        "o",
    )
    axes[1].set_xlabel(r"evaluated $D_{waist}$ ($\mu$m)")
    axes[1].set_ylabel("raw loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("Bounded cross-check evaluations")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_figures(
    target: np.ndarray, result: dict[str, Any], figure_dir: Path
) -> list[Path]:
    paths = [
        figure_dir / "exp050_loss_profile.png",
        figure_dir / "exp050_best_fit.png",
        figure_dir / "exp050_numerical_controls.png",
        figure_dir / "exp050_estimator_crosscheck.png",
    ]
    _plot_profile(result, paths[0])
    _plot_best_fit(target, result, paths[1])
    _plot_numerical_controls(result, paths[2])
    _plot_crosscheck(result, paths[3])
    return paths


def _audit_artifacts(
    run_dir: Path,
    *,
    expected_status: str,
    target_hash: str,
    figure_paths: list[Path],
) -> dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    if metrics["experiment_status"] != expected_status:
        raise RuntimeError("External metrics status differs from expected status.")
    with h5py.File(hdf5_path, "r") as h5:
        expected_entry = {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "reconstruction",
            "sample",
            "truth",
        }
        if set(h5["/entry"].keys()) != expected_entry:
            raise RuntimeError("Unexpected exp050 /entry tree.")
        if list(h5["/entry/data"].keys()):
            raise RuntimeError("exp050 may not contain detector or scan data.")
        for forbidden in ("calibration", "preprocessing"):
            if forbidden in h5["/entry"]:
                raise RuntimeError(f"exp050 wrote a fake {forbidden} group.")
        target = np.asarray(h5["/entry/truth/P_B_true"])
        root = h5["/entry/reconstruction/waist_fit"]
        best = np.asarray(root["P_B_best_raw"])
        residual = np.asarray(root["residual_field_raw"])
        cache = np.asarray(root["candidate_cache/P_B_candidate"])
        best_index = int(root["best_cache_index"][()])
        if sha256_array(target) != target_hash:
            raise RuntimeError("Saved target differs from authoritative source bytes.")
        if target.dtype != np.complex128 or best.dtype != np.complex128:
            raise RuntimeError("exp050 target/best dtype contract failed.")
        if not np.array_equal(best, cache[best_index]):
            raise RuntimeError("Best probe does not match its cache index.")
        if not np.array_equal(best - target, residual):
            raise RuntimeError("Raw residual identity failed.")
        if root["source_provenance/P_B_rec_used"][()]:
            raise RuntimeError("exp050 provenance promoted forbidden P_B_rec.")
        if h5["/entry/metrics/experiment_status"].asstr()[()] != expected_status:
            raise RuntimeError("HDF5 metrics status differs from JSON.")
        finite_failures: list[str] = []

        def check_finite(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "iufc":
                if not np.all(np.isfinite(obj[()])):
                    finite_failures.append(name)

        h5.visititems(check_finite)
        if finite_failures:
            raise RuntimeError(f"Non-finite HDF5 datasets: {finite_failures[:3]}.")
        hdf5_path_count = 0

        def count_path(_name: str, _obj: h5py.Group | h5py.Dataset) -> None:
            nonlocal hdf5_path_count
            hdf5_path_count += 1

        h5.visititems(count_path)
    figure_audit: dict[str, Any] = {}
    for path in figure_paths:
        image = plt.imread(path)
        if image.ndim not in {2, 3} or not np.all(np.isfinite(image)):
            raise RuntimeError(f"Unreadable or non-finite figure: {path}.")
        figure_audit[path.name] = {
            "shape": list(image.shape),
            "sha256": sha256_file(path),
        }
    return {
        "passed": True,
        "hdf5_path_count": hdf5_path_count,
        "target_hash_verified": True,
        "json_hdf5_status_consistent": True,
        "best_cache_residual_identity_verified": True,
        "all_numeric_hdf5_datasets_finite": True,
        "figures": figure_audit,
        "metadata_status": metadata["experiment_status"],
    }


def _write_hdf5(
    path: Path,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    source: dict[str, Any],
    source_identity: dict[str, Any],
    result: dict[str, Any],
) -> None:
    reconstruction = {
        "waist_fit": {
            "design": config[metadata["mode"]],
            "source_provenance": source_identity,
            "replay": result["replay"],
            "P_B_replay_raw": result["P_B_replay_raw"],
            "P_B_replay_repeat_raw": result["P_B_replay_repeat_raw"],
            "P_B_radial_replay_raw": result["P_B_radial_replay_raw"],
            "candidate_cache": result["cache"],
            "profile": result["profile"],
            "finite_difference": result["finite_difference"],
            "estimator_crosscheck": result["estimator_crosscheck"],
            "D_waist_estimate_m": result["estimate_m"],
            "best_cache_index": result["best_cache_index"],
            "P_B_best_raw": result["P_B_best_raw"],
            "residual_field_raw": result["residual_field_raw"],
        }
    }
    save_ptycho_hdf5(
        path,
        instrument={
            **source["instrument"],
            "shape_ny_nx": source["shape_ny_nx"],
            "axis_order": "y_x",
            "field_units": "arbitrary complex field amplitude",
            "phase_units": "rad",
        },
        sample={"sample_a": config["sample_a"]},
        truth={
            "P_B_true": source["target"],
            "D_waist_true_m": config["sample_a"]["d_waist_true_m"],
            "source_operator": {
                "radial_source_r_m": source["source_radius_m"],
                "radial_source_weight_m": source["source_weights_m"],
                "P_B_radial_r_m": source["output_radius_m"],
                "P_B_radial_true": source["target_radial_probe"],
            },
        },
        reconstruction=reconstruction,
        config_yaml=config_to_yaml(config),
        metadata=metadata,
        metrics=metrics,
    )


def run(config_path: Path, *, mode: str) -> Path:
    """Run one development/preflight or authoritative formal exp050 fit."""

    config = load_config(config_path)
    validate_exp050_config(config, mode=mode)
    run_name = str(config["output"]["run_name"])
    if mode == "preflight":
        run_name += "_preflight"
    run_dir = make_run_dir(PROJECT_ROOT / str(config["run"]["output_root"]), run_name)
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {"status": "running", "artifacts_validated": False, "mode": mode},
    )
    started = time.perf_counter()
    try:
        source = load_exp030_source(config, PROJECT_ROOT)
        generator = make_exp050_candidate_generator(source)
        true_waist = float(config["sample_a"]["d_waist_true_m"])
        result = run_exp050_estimator(
            np.asarray(source["target"], dtype=np.complex128),
            np.asarray(source["target_radial_probe"], dtype=np.complex128),
            generator,
            true_waist_m=true_waist,
            design=config[mode],
        )
        scientific_status = str(result["status"])
        experiment_status = scientific_status if mode == "formal" else "Development"
        source_identity = _source_identity(source)
        metrics = {
            "experiment_status": experiment_status,
            "scientific_status_if_formal": scientific_status,
            "mode": mode,
            "interpretation": result["interpretation"],
            "source_identity": source_identity,
            "fit": _small_result(result, true_waist),
            "claim_boundary": (
                "single-parameter true-probe oracle fitting self-consistency "
                "within the exp030 2D projected working model only"
            ),
            "project_level_hdf5_schema_changed": False,
        }
        metadata = {
            "experiment": "exp050_TGV_2d_projected_true_probe_waist_fit",
            "phase": "Phase 5 projected diagnostic",
            "mode": mode,
            "experiment_status": experiment_status,
            "created_at": created_at_utc(),
            "git_commit": get_git_commit(PROJECT_ROOT) or "",
            "config_path": str(config_path),
            "dataset_type": "simulation",
            "source_identity": source_identity,
            "primary_quantity": "raw_complex_P_B",
            "truth_use": (
                "simulation evaluation only; no early stop or method selection"
            ),
            "axis_order": "y_x",
            "dx_tuple_order_if_used": "dy_dx",
            "P_B_rec_used": False,
            "detector_data_used": False,
            "sample_B_used": False,
            "scan_used": False,
            "project_level_hdf5_schema_changed": False,
        }
        save_config(run_dir / "config.yaml", config)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
        _write_hdf5(
            hdf5_path,
            config=config,
            metadata=metadata,
            metrics=metrics,
            source=source,
            source_identity=source_identity,
            result=result,
        )
        figure_paths = _save_figures(source["target"], result, run_dir / "figures")
        metrics["artifact_audit"] = _audit_artifacts(
            run_dir,
            expected_status=experiment_status,
            target_hash=source["target_bytes_sha256"],
            figure_paths=figure_paths,
        )
        save_json(run_dir / "metrics.json", metrics)
        _write_hdf5(
            hdf5_path,
            config=config,
            metadata=metadata,
            metrics=metrics,
            source=source,
            source_identity=source_identity,
            result=result,
        )
        metrics["artifact_audit"] = _audit_artifacts(
            run_dir,
            expected_status=experiment_status,
            target_hash=source["target_bytes_sha256"],
            figure_paths=figure_paths,
        )
        save_json(run_dir / "metrics.json", metrics)
        _write_hdf5(
            hdf5_path,
            config=config,
            metadata=metadata,
            metrics=metrics,
            source=source,
            source_identity=source_identity,
            result=result,
        )
        artifact_hashes = {
            "config": sha256_file(run_dir / "config.yaml"),
            "metadata": sha256_file(run_dir / "metadata.json"),
            "metrics": sha256_file(run_dir / "metrics.json"),
            "hdf5": sha256_file(hdf5_path),
            "figures": {path.name: sha256_file(path) for path in figure_paths},
        }
        elapsed = time.perf_counter() - started
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "artifacts_validated": True,
                "mode": mode,
                "experiment_status": experiment_status,
                "scientific_status_if_formal": scientific_status,
                "interpretation": result["interpretation"],
                "runtime_seconds": elapsed,
                "artifact_sha256": artifact_hashes,
            },
        )
    except Exception as exc:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "artifacts_validated": False,
                "mode": mode,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            },
        )
        raise
    print(f"Saved {mode} run to: {run_dir}")
    print(
        f"Status={experiment_status}; "
        f"estimate={float(result['estimate_m']) * 1e6:.9f} um; "
        f"replay={float(result['replay']['raw_complex_relative_l2']):.3e}"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config, mode=args.mode)


if __name__ == "__main__":
    main()
