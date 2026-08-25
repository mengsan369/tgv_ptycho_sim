from __future__ import annotations

import argparse
import json
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

from tgv_ptycho.inverse.exp051 import (  # noqa: E402
    Exp051SourceArtifact,
    load_exp051_source_true_probe,
    make_exp051_candidate_generator,
    sha256_file,
    validate_exp051_config,
)
from tgv_ptycho.inverse.waist_fit import fit_waist_from_probe  # noqa: E402
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp051 oracle waist fit.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _metadata(
    config: dict[str, Any],
    config_path: Path,
    source: Exp051SourceArtifact,
) -> dict[str, Any]:
    return {
        "experiment_id": "exp051",
        "created_at_utc": created_at_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_config": str(config_path),
        "run_role": config["experiment"]["role"],
        "scientific_claim_boundary": (
            "single-parameter oracle fit within the selected exp040 scalar "
            "working model only"
        ),
        "truth_identity": config["experiment"]["truth_identity"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "source_run": str(config["source"]["run"]),
        "source_run_resolved": str(source.run_dir),
        "source_config_path": str(source.config_path),
        "source_hdf5_path": str(source.hdf5_path),
        "source_target_hdf5_path": source.target_hdf5_path,
        "source_file_sha256": source.file_sha256,
        "source_target_dataset_sha256": source.target_dataset_sha256,
        "source_git_commit": source.source_metadata["git_commit"],
        "source_operator_branch": source.source_metadata["operator_branch"],
        "primary_input_is_raw_p_b_true": True,
        "p_b_rec_used_as_primary_input": False,
        "truth_used_by_optimizer_stopping": False,
        "phase_or_scale_alignment": "none",
    }


def _metrics_summary(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    runtime_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "experiment_status": result["status"],
        "interpretation": result["interpretation"],
        "stages_completed": result["stages_completed"],
        "stage_a_replay_pass": result["stage_a_replay_pass"],
        "exact_replay": result["replay"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "truth_identity": config["experiment"]["truth_identity"],
        "primary_input_is_raw_p_b_true": True,
        "p_b_rec_used_as_primary_input": False,
        "primary_loss": config["fit"]["primary_loss"],
        "mask": config["fit"]["mask"],
        "phase_or_scale_alignment": "none",
        "runtime_seconds": runtime_seconds,
        "peak_rss_bytes_sampled": peak_rss_bytes,
        "runtime_within_registered_contract": bool(
            runtime_seconds <= config["output"]["expected_runtime_seconds_max"]
        ),
        "memory_within_registered_contract": bool(
            peak_rss_bytes
            <= config["output"]["expected_peak_memory_bytes_max"]
        ),
    }
    if result["stage_a_replay_pass"]:
        profile = result["profile"]
        finite_difference = {
            key: value
            for key, value in result["finite_difference"].items()
            if key not in {"jacobian_h1", "jacobian_h2"}
        }
        metrics.update(
            {
                "profile": profile,
                "finite_difference": finite_difference,
                "optimizer": result["optimizer"],
                "D_waist_true_m": config["fit"]["true_d_waist_m"],
                "D_waist_estimate_m": result["optimizer"][
                    "representative_estimate_m"
                ],
                "D_waist_absolute_error_m": abs(
                    result["optimizer"]["representative_estimate_m"]
                    - config["fit"]["true_d_waist_m"]
                ),
                "candidate_cache_count": len(result["cache"]["D_waist_m"]),
            }
        )
    return metrics


def _hdf5_payload(
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    source: Exp051SourceArtifact,
    result: dict[str, Any],
) -> dict[str, Any]:
    source_config = source.source_config
    reconstruction: dict[str, Any] = {
        "waist_fit": {
            "design": {
                "fit": config["fit"],
                "thresholds": config["thresholds"],
                "primary_input": {
                    "source_run": config["source"]["run"],
                    "source_hdf5_relative_path": config["source"][
                        "hdf5_relative_path"
                    ],
                    "source_target_hdf5_path": source.target_hdf5_path,
                    "source_file_sha256": source.file_sha256,
                    "source_target_dataset_sha256": (
                        source.target_dataset_sha256
                    ),
                    "P_B_rec_used": False,
                    "phase_or_scale_alignment": "none",
                },
            },
            "exact_replay": result["replay"],
            "stage_a_replay_pass": result["stage_a_replay_pass"],
            "cache": result["cache"],
        }
    }
    waist_fit = reconstruction["waist_fit"]
    if result["stage_a_replay_pass"]:
        waist_fit.update(
            {
                "profile": result["profile"],
                "finite_difference": result["finite_difference"],
                "optimizer": result["optimizer"],
                "P_B_best_raw": result["P_B_best_raw"],
                "residual_field_raw": result["residual_field_raw"],
                "best_cache_index": result["best_cache_index"],
            }
        )
    identity = config["source"]["expected_identity"]
    return {
        "instrument": {
            "wavelength_m": source_config["optics"]["wavelength_m"],
            "internal_reference_index": source_config["optics"][
                "internal_reference_index"
            ],
            "external_medium_index": source_config["optics"][
                "external_medium_index"
            ],
            "z_AB_m": source_config["optics"]["z_AB_m"],
            "probe_grid": {
                "plane": identity["plane"],
                "axis_order": identity["axis_order"],
                "native_shape": identity["native_shape"],
                "node_dx_m": identity["node_dx_m"],
                "native_fov_m": identity["native_fov_m"],
                "coordinate_endpoints_m": identity[
                    "coordinate_endpoints_m"
                ],
                "origin": identity["origin"],
                "residual_embedding": identity["residual_embedding"],
                "fft_normalization": identity["fft_normalization"],
                "angular_spectrum_bandlimit": identity[
                    "angular_spectrum_bandlimit"
                ],
                "external_alias_control": identity[
                    "external_alias_control"
                ],
            },
        },
        "sample": {
            "sample_a": source_config["sample_a"],
            "sample_b_source_identity_only_not_used_by_fit": source_config[
                "sample_b"
            ],
        },
        "truth": {
            "identity": config["experiment"]["truth_identity"],
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "P_B_true": source.P_B_true,
            "D_waist_true_m": config["fit"]["true_d_waist_m"],
            "D_z_m": source.D_z_m,
            "z_m": source.z_m,
            "slice_widths_m": source.slice_widths_m,
        },
        "reconstruction": reconstruction,
        "config_yaml": config_to_yaml(config),
        "metadata": metadata,
        "metrics": metrics,
    }


def _save_figures(
    result: dict[str, Any], target: np.ndarray, paths: list[Path]
) -> None:
    if not result["stage_a_replay_pass"]:
        raise RuntimeError("Formal figures require the exact replay gate to pass.")
    profile = result["profile"]
    optimizer = result["optimizer"]
    fd = result["finite_difference"]

    fig, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    axis.semilogy(
        np.asarray(profile["coarse_d_waist_m"]) * 1.0e6,
        np.maximum(profile["coarse_loss"], np.finfo(np.float64).tiny),
        "o-",
        label="coarse",
    )
    axis.semilogy(
        np.asarray(profile["fine_d_waist_m"]) * 1.0e6,
        np.maximum(profile["fine_loss"], np.finfo(np.float64).tiny),
        ".-",
        label="fine",
    )
    profile_ylim = axis.get_ylim()
    axis.plot(
        [20.0, 20.0],
        profile_ylim,
        color="black",
        linestyle="--",
        label="truth",
    )
    axis.set_ylim(profile_ylim)
    axis.set_xlabel("D_waist (um)")
    axis.set_ylabel("raw normalized complex loss")
    axis.set_title(f"exp051 loss profile — {result['status']}")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)

    j1 = np.asarray(fd["jacobian_h1"])
    j2 = np.asarray(fd["jacobian_h2"])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    images = [np.abs(j1), np.abs(j2), np.abs(j1 - j2)]
    titles = ["|J(h=0.25 um)|", "|J(h=0.125 um)|", "|J(h1)-J(h2)|"]
    for axis, image, title in zip(axes, images, titles, strict=True):
        shown = axis.imshow(image, origin="lower", cmap="magma")
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.8)
    fig.suptitle(
        "Local raw-complex Jacobian; units of field per meter",
        fontsize=11,
    )
    fig.savefig(paths[1], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for name, branch in optimizer["branches"].items():
        axis.plot(
            np.arange(branch["evaluation_count"]),
            np.asarray(branch["evaluated_d_waist_m"]) * 1.0e6,
            ".-",
            alpha=0.75,
            label=name,
        )
    optimizer_xlim = axis.get_xlim()
    axis.plot(
        optimizer_xlim,
        [20.0, 20.0],
        color="black",
        linestyle="--",
        label="truth",
    )
    axis.set_xlim(optimizer_xlim)
    axis.set_xlabel("objective evaluation")
    axis.set_ylabel("evaluated D_waist (um)")
    axis.set_title("Equal-budget bounded pattern-search tracks")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(paths[2], dpi=160)
    plt.close(fig)

    best = np.asarray(result["P_B_best_raw"])
    residual = np.asarray(result["residual_field_raw"])
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.5), constrained_layout=True)
    panels = [
        (np.abs(target), "target |P_B_true|", "viridis"),
        (np.abs(best), "best raw |P_B|", "viridis"),
        (np.abs(residual), "raw residual amplitude", "magma"),
        (np.angle(best * np.conj(target)), "raw phase difference (rad)", "twilight"),
    ]
    for axis, (image, title, cmap) in zip(axes.ravel(), panels, strict=True):
        shown = axis.imshow(image, origin="lower", cmap=cmap)
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.8)
    fig.suptitle("exp051 raw target / best-fit comparison", fontsize=12)
    fig.savefig(paths[3], dpi=160)
    plt.close(fig)


def _all_numeric_hdf5_finite(group: h5py.Group) -> bool:
    finite = True

    def visitor(_: str, value: h5py.Group | h5py.Dataset) -> None:
        nonlocal finite
        if isinstance(value, h5py.Dataset) and value.dtype.kind in "biufc":
            finite = bool(finite and np.all(np.isfinite(value[...])))

    group.visititems(visitor)
    return finite


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    source: Exp051SourceArtifact,
) -> None:
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    if load_config(config_path) != config:
        raise RuntimeError("Run config.yaml differs from the executed config.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if (
        metadata["reference_validated"] is not False
        or metadata["full_tgv_reference_authorized"] is not False
        or metadata["p_b_rec_used_as_primary_input"] is not False
        or metrics["reference_validated"] is not False
        or metrics["full_tgv_reference_authorized"] is not False
        or metrics["p_b_rec_used_as_primary_input"] is not False
    ):
        raise RuntimeError("Artifact provenance flags violate the exp051 boundary.")
    required = {
        "config_yaml",
        "data",
        "instrument",
        "metadata",
        "metrics",
        "reconstruction",
        "sample",
        "truth",
    }
    with h5py.File(hdf5_path, "r") as h5:
        if set(h5["entry"]) != required:
            raise RuntimeError("Unexpected exp051 /entry tree.")
        if "calibration" in h5["entry"] or "preprocessing" in h5["entry"]:
            raise RuntimeError("exp051 must not fabricate calibration/preprocessing.")
        tree_names: list[str] = []
        h5["entry"].visit(tree_names.append)
        if any(name.rsplit("/", 1)[-1] == "P_B_rec" for name in tree_names):
            raise RuntimeError("exp051 HDF5 may not contain P_B_rec.")
        if not np.array_equal(h5["entry/truth/P_B_true"][...], source.P_B_true):
            raise RuntimeError("Saved exp051 target differs from source P_B_true.")
        if (
            bool(h5["entry/truth/reference_validated"][()])
            or bool(h5["entry/truth/full_tgv_reference_authorized"][()])
        ):
            raise RuntimeError("HDF5 truth flags were promoted.")
        root = h5["entry/reconstruction/waist_fit"]
        if _decode(h5["entry/metrics/experiment_status"][()]) != metrics[
            "experiment_status"
        ]:
            raise RuntimeError("JSON/HDF5 status differs.")
        if not np.allclose(
            h5["entry/metrics/profile/fine_loss"][...],
            np.asarray(metrics["profile"]["fine_loss"]),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("JSON/HDF5 fine profile differs.")
        if not np.array_equal(
            root["P_B_best_raw"][...] - h5["entry/truth/P_B_true"][...],
            root["residual_field_raw"][...],
        ):
            raise RuntimeError("Best-fit residual field identity failed.")
        cache_count = len(root["cache/D_waist_m"])
        for branch in root["optimizer/branches"].values():
            if int(branch["evaluation_count"][()]) != 41:
                raise RuntimeError("Optimizer branches do not have equal budget.")
            if _decode(branch["stopping_reason"][()]) != "evaluation_budget":
                raise RuntimeError("Optimizer stopping reason changed.")
            indices = branch["evaluated_cache_index"][...]
            if np.any(indices < 0) or np.any(indices >= cache_count):
                raise RuntimeError("Optimizer cache mapping is invalid.")
        if not _all_numeric_hdf5_finite(h5["entry"]):
            raise RuntimeError("HDF5 contains non-finite numeric datasets.")
    figures = sorted((run_dir / "figures").glob("*.png"))
    expected_names = sorted(config["output"]["figure_filenames"])
    if [path.name for path in figures] != expected_names:
        raise RuntimeError("Formal exp051 figure set differs from config.")
    for figure in figures:
        image = plt.imread(figure)
        if image.size == 0 or not np.all(np.isfinite(image)):
            raise RuntimeError(f"Unreadable/non-finite figure: {figure}")
    if hdf5_path.stat().st_size > config["output"]["expected_hdf5_bytes_max"]:
        raise RuntimeError("exp051 HDF5 exceeds the registered size contract.")


def run(config_path: Path) -> Path:
    resolved_config = config_path
    if not resolved_config.is_absolute():
        resolved_config = (PROJECT_ROOT / resolved_config).resolve()
    config = load_config(resolved_config)
    validate_exp051_config(config)
    output_root = Path(config["output"]["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = make_run_dir(output_root, config["output"]["run_name"])
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {"status": "running", "artifacts_validated": False},
    )
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    def sample_memory() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)

    try:
        source = load_exp051_source_true_probe(config, PROJECT_ROOT)
        candidate_generator = make_exp051_candidate_generator(
            source.source_config, memory_callback=sample_memory
        )
        fit = config["fit"]
        result = fit_waist_from_probe(
            source.P_B_true,
            candidate_generator,
            true_waist_m=float(fit["true_d_waist_m"]),
            bounds_m=tuple(float(value) for value in fit["bounds_m"]),
            coarse_grid_m=fit["coarse_grid_m"],
            fine_half_width_m=float(fit["fine_half_width_m"]),
            fine_step_m=float(fit["fine_step_m"]),
            finite_difference_steps_m=tuple(
                float(value) for value in fit["finite_difference_steps_m"]
            ),
            optimizer_starts_m=fit["optimizer"]["starts_m"],
            optimizer_initial_step_m=float(
                fit["optimizer"]["initial_step_m"]
            ),
            optimizer_evaluation_budget=int(
                fit["optimizer"]["evaluation_budget_per_start"]
            ),
            thresholds=config["thresholds"],
        )
        sample_memory()
        runtime_seconds = time.perf_counter() - started
        metadata = _metadata(config, resolved_config, source)
        metrics = _metrics_summary(
            config,
            result,
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss,
        )
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        payload = _hdf5_payload(config, metadata, metrics, source, result)
        save_ptycho_hdf5(
            run_dir / "outputs" / config["output"]["hdf5_filename"],
            **payload,
        )
        figure_paths = [
            run_dir / "figures" / filename
            for filename in config["output"]["figure_filenames"]
        ]
        _save_figures(result, source.P_B_true, figure_paths)
        _validate_artifacts(run_dir, config, source)
        state = {
            "status": "complete",
            "artifacts_validated": True,
            "completed_at_utc": created_at_utc(),
            "runtime_seconds": runtime_seconds,
            "config_sha256": sha256_file(run_dir / "config.yaml"),
            "metadata_sha256": sha256_file(run_dir / "metadata.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
            "hdf5_sha256": sha256_file(
                run_dir / "outputs" / config["output"]["hdf5_filename"]
            ),
            "figure_sha256": {
                path.name: sha256_file(path) for path in figure_paths
            },
            "figure_count": len(figure_paths),
        }
        save_json(run_dir / "run_state.json", state)
    except Exception as error:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    print(f"run_dir: {run_dir}")
    print(f"experiment_status: {result['status']}")
    print(
        "D_waist_estimate_um: "
        f"{result['optimizer']['representative_estimate_m'] * 1.0e6:.9g}"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
