"""Run the combined exp052 known-B and blind-ePIE reconstructed-probe fit."""

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

from tgv_ptycho.inverse.exp052 import (
    combine_status,
    evaluate_fit_gates,
    fit_reconstructed_probe,
    frozen_data_fidelity,
    load_exp052_sources,
    make_candidate_cache,
    replay_measurement_operator,
    run_known_b_reconstruction,
    sha256_array,
    sha256_file,
    summarize_checkpoint_fits,
    truth_evaluation,
    validate_exp052_config,
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


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.sum(np.abs(values - reference) ** 2)
            / max(float(np.sum(np.abs(reference) ** 2)), np.finfo(float).eps)
        )
    )


def _fit_summary(fit: dict[str, Any], true_waist_m: float) -> dict[str, Any]:
    return {
        "D_waist_estimate_m": float(fit["estimate_m"]),
        "reported_interval_m": np.asarray(fit["reported_interval_m"]),
        "reported_interval_width_m": float(fit["reported_interval_width_m"]),
        "absolute_error_m_simulation_evaluation_only": abs(
            float(fit["estimate_m"]) - true_waist_m
        ),
        "minimum_primary_loss": float(fit["minimum_primary_loss"]),
        "minimum_raw_loss": float(fit["minimum_raw_loss"]),
        "minimum_complex_gain_profiled_loss_diagnostic": float(
            fit["minimum_complex_gain_profiled_loss_diagnostic"]
        ),
        "global_phase_rad": float(fit["global_phase_rad"]),
        "complex_gain_diagnostic": complex(fit["complex_gain_diagnostic"]),
        "quadratic_crosscheck_estimate_m": float(
            fit["quadratic_crosscheck_estimate_m"]
        ),
        "quadratic_profile_agreement_m": float(fit["quadratic_profile_agreement_m"]),
        "local_loss_curvature_per_m2": float(fit["local_loss_curvature_per_m2"]),
        "global_local_minima_count": int(fit["global_local_minima_count"]),
        "equivalence_component_count": int(fit["equivalence_component_count"]),
        "boundary_hit": bool(fit["boundary_hit"]),
        "best_cache_index": int(fit["best_cache_index"]),
    }


def _checkpoint_fit_summary(
    fits: dict[str, dict[str, Any]], true_waist_m: float
) -> dict[str, Any]:
    return {key: _fit_summary(value, true_waist_m) for key, value in fits.items()}


def _source_identity(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    exp030 = config["source"]["exp030"]
    oracle = config["source"]["exp050_oracle"]
    return {
        "exp030": {
            "run": exp030["run"],
            "hdf5": str(source["exp030_hdf5_path"]),
            "file_sha256": exp030["sha256"],
            "dataset_bytes_sha256": source["dataset_hashes"],
            "git_commit": exp030["git_commit"],
            "shape_ny_nx": list(source["shape_ny_nx"]),
            "axis_order": "y_x",
            "scan_coordinate_order": "x_y",
            "scan_position_units_attribute_present": False,
            "dataset_units_attributes_present": False,
            "field_units_semantic": "arbitrary complex field amplitude",
            "phase_units_semantic": "rad",
            "scan_units_semantic": "m",
            "operator": (
                "continuous_axisymmetric_fresnel_hankel_T_minus_1_plus_"
                "analytic_plane_wave"
            ),
            "measurement_operator": (
                "periodic_integer_shift_plus_bandlimited_ASM_detector_intensity"
            ),
        },
        "exp050_oracle": {
            "run": oracle["run"],
            "file_sha256": oracle["sha256"],
            "frozen_source_config_sha256": oracle["frozen_source_config_sha256"],
            "frozen_body_sha256": oracle["frozen_body_sha256"],
            "estimate_m": source["oracle"]["estimate_m"],
            "interval_m": source["oracle"]["interval_m"],
        },
        "part_b_raw": {
            "selection_rule": config["part_b"]["selection_rule"],
            "probe_dataset": config["part_b"]["raw_probe_dataset"],
            "object_dataset": config["part_b"]["raw_object_dataset"],
            "probe_bytes_sha256": sha256_array(source["blind"]["P_B_rec_raw"]),
            "object_bytes_sha256": sha256_array(source["blind"]["B_rec_raw"]),
            "durable_checkpoint_sha256": source["blind"]["durable_checkpoint_sha256"],
            "durable_checkpoint_metadata": source["blind"][
                "durable_checkpoint_metadata"
            ],
        },
    }


def _plot_convergence(known: dict[str, Any], blind: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].semilogy(np.arange(1, len(known["loss_curve"]) + 1), known["loss_curve"])
    axes[0].axhline(
        known["final_data_fidelity_loss"],
        color="tab:red",
        linestyle="--",
        label="frozen full-stack residual",
    )
    axes[0].set_title("Part A known-B probe-only")
    axes[1].semilogy(np.arange(1, len(blind["loss_curve"]) + 1), blind["loss_curve"])
    axes[1].axhline(
        blind["final_data_fidelity_loss"],
        color="tab:red",
        linestyle="--",
        label="frozen full-stack residual",
    )
    for axis in axes:
        axis.set_xlabel("ePIE iteration")
        axis.set_ylabel("relative detector-amplitude residual")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_landscapes(
    known_fit: dict[str, Any],
    blind_fit: dict[str, Any],
    oracle: dict[str, Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for fit, label, color in (
        (known_fit, "known-B", "tab:blue"),
        (blind_fit, "blind ePIE", "tab:orange"),
    ):
        profile = fit["global_profile"]
        axes[0].semilogy(
            profile["D_waist_m"] * 1e6,
            np.maximum(profile["primary_loss"], np.finfo(float).tiny),
            "o-",
            label=label,
            color=color,
        )
        local = fit["refinement_profile"]
        axes[1].semilogy(
            local["D_waist_m"] * 1e6,
            np.maximum(local["primary_loss"], np.finfo(float).tiny),
            "o-",
            label=label,
            color=color,
        )
    for axis in axes:
        axis.axvspan(
            oracle["interval_m"][0] * 1e6,
            oracle["interval_m"][1] * 1e6,
            color="tab:green",
            alpha=0.18,
            label="exp050 oracle grid cell",
        )
        axis.set_xlabel(r"$D_{waist}$ ($\mu$m)")
        axis.set_ylabel("global-phase-profiled normalized squared L2")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_title("Registered global landscapes")
    axes[1].set_title("Local refinement landscapes")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_raw_probes(known: dict[str, Any], blind: dict[str, Any], path: Path) -> None:
    panels = [
        (np.abs(known["P_B_rec_raw"]), "known-B raw amplitude", "viridis"),
        (np.angle(known["P_B_rec_raw"]), "known-B raw phase (rad)", "twilight"),
        (np.abs(blind["P_B_rec_raw"]), "blind raw amplitude", "viridis"),
        (np.angle(blind["P_B_rec_raw"]), "blind raw phase (rad)", "twilight"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes.ravel(), panels, strict=True):
        image = axis.imshow(values, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("Primary targets are raw saved complex128 fields")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_residuals(
    known_fit: dict[str, Any], blind_fit: dict[str, Any], path: Path
) -> None:
    panels = [
        (
            np.abs(known_fit["P_B_best_raw"]),
            "known-B best candidate amplitude",
            "viridis",
        ),
        (
            np.abs(known_fit["residual_field_primary"]),
            "known-B primary residual amplitude",
            "magma",
        ),
        (
            np.abs(blind_fit["P_B_best_raw"]),
            "blind best candidate amplitude",
            "viridis",
        ),
        (
            np.abs(blind_fit["residual_field_primary"]),
            "blind primary residual amplitude",
            "magma",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes.ravel(), panels, strict=True):
        image = axis.imshow(values, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_gauge(
    known_fit: dict[str, Any], blind_fit: dict[str, Any], path: Path
) -> None:
    labels = ["known-B", "blind ePIE"]
    fits = [known_fit, blind_fit]
    values = np.asarray(
        [
            [
                fit["minimum_raw_loss"],
                fit["minimum_primary_loss"],
                fit["minimum_complex_gain_profiled_loss_diagnostic"],
            ]
            for fit in fits
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    width = 0.23
    x = np.arange(2)
    for index, name in enumerate(
        ("raw", "global phase primary", "complex gain diagnostic")
    ):
        axes[0].bar(x + (index - 1) * width, values[:, index], width, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("normalized squared L2")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(labels, [fit["global_phase_rad"] for fit in fits])
    axes[1].set_ylabel("profiled global phase (rad)")
    axes[1].set_title("Truth-free candidate-versus-target gauge")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_checkpoint_consistency(
    known_fits: dict[str, dict[str, Any]],
    blind_fits: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, fits, title in (
        (axes[0], known_fits, "Part A checkpoints"),
        (axes[1], blind_fits, "Part B checkpoints"),
    ):
        iterations = np.asarray([int(key) for key in fits])
        estimates = np.asarray([fits[key]["estimate_m"] for key in fits]) * 1e6
        axis.plot(iterations, estimates, "o-")
        axis.set_xlabel("iteration")
        axis.set_ylabel(r"fitted $D_{waist}$ ($\mu$m)")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_comparison(comparison: dict[str, Any], path: Path) -> None:
    labels = ["oracle", "known-B", "blind ePIE"]
    estimates = np.asarray(comparison["estimates_m"]) * 1e6
    intervals = np.asarray(comparison["intervals_m"]) * 1e6
    lower = estimates - intervals[:, 0]
    upper = intervals[:, 1] - estimates
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.errorbar(
        np.arange(3), estimates, yerr=np.vstack([lower, upper]), fmt="o", capsize=5
    )
    axis.set_xticks(np.arange(3), labels)
    axis.set_ylabel(r"$D_{waist}$ ($\mu$m)")
    axis.set_title("Oracle to known-B to blind displacement")
    axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_figures(
    run_dir: Path,
    *,
    known: dict[str, Any],
    blind: dict[str, Any],
    known_fit: dict[str, Any],
    blind_fit: dict[str, Any],
    known_checkpoint_fits: dict[str, dict[str, Any]],
    blind_checkpoint_fits: dict[str, dict[str, Any]],
    oracle: dict[str, Any],
    comparison: dict[str, Any],
) -> list[Path]:
    paths = [
        run_dir / "figures" / name
        for name in (
            "exp052_reconstruction_convergence.png",
            "exp052_loss_landscapes.png",
            "exp052_raw_probe_fields.png",
            "exp052_best_fit_residuals.png",
            "exp052_gauge_diagnostics.png",
            "exp052_checkpoint_consistency.png",
            "exp052_oracle_known_blind_comparison.png",
        )
    ]
    _plot_convergence(known, blind, paths[0])
    _plot_landscapes(known_fit, blind_fit, oracle, paths[1])
    _plot_raw_probes(known, blind, paths[2])
    _plot_residuals(known_fit, blind_fit, paths[3])
    _plot_gauge(known_fit, blind_fit, paths[4])
    _plot_checkpoint_consistency(known_checkpoint_fits, blind_checkpoint_fits, paths[5])
    _plot_comparison(comparison, paths[6])
    return paths


def _fit_hdf5_payload(value: Any) -> Any:
    """Return a copy with NumPy object/string arrays made HDF5-writer safe."""

    if isinstance(value, dict):
        return {key: _fit_hdf5_payload(child) for key, child in value.items()}
    if isinstance(value, np.ndarray) and value.dtype.kind in {"O", "U"}:
        return value.astype(str).tolist()
    return value


def _write_hdf5(
    path: Path,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    source: dict[str, Any],
    source_identity: dict[str, Any],
    shared: dict[str, Any],
    known: dict[str, Any],
    blind: dict[str, Any],
    known_fit: dict[str, Any],
    blind_fit: dict[str, Any],
    known_checkpoint_fits: dict[str, dict[str, Any]],
    blind_checkpoint_fits: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    candidate_cache: Any,
) -> None:
    known_reconstruction = {
        **known,
        "simulation_evaluation_only": known["simulation_evaluation_only"],
    }
    blind_reconstruction = {
        **blind,
        "simulation_evaluation_only": blind["simulation_evaluation_only"],
    }
    reconstruction = {
        "known_b_probe": known_reconstruction,
        "blind_epie_probe": blind_reconstruction,
        "waist_fit": {
            "gauge_contract": config["gauge"],
            "known_b": _fit_hdf5_payload(known_fit),
            "blind_epie": _fit_hdf5_payload(blind_fit),
            "checkpoint_consistency": {
                "known_b": _fit_hdf5_payload(known_checkpoint_fits),
                "blind_epie": _fit_hdf5_payload(blind_checkpoint_fits),
            },
            "candidate_cache": {
                "D_waist_m": np.asarray(candidate_cache.diameters_m, dtype=np.float64),
                "P_B_candidate": np.stack(candidate_cache.probes),
                "candidate_sha256": candidate_cache.hashes,
            },
        },
        "comparison": comparison,
        "source_operator_replay": shared,
        "source_provenance": source_identity,
    }
    save_ptycho_hdf5(
        path,
        I_stack=source["I_stack"],
        scan_positions=source["scan_positions"],
        instrument={
            **source["instrument"],
            "shape_ny_nx": source["shape_ny_nx"],
            "axis_order": "y_x",
            "scan_coordinate_order": "x_y",
            "field_units": "arbitrary complex field amplitude",
            "phase_units": "rad",
            "scan_position_units": "m",
            "source_dataset_units_attributes_present": False,
        },
        sample={
            "sample_a": config["sample_a"],
            "sample_b": {
                "B_known_input": source["B_true"],
                "source_dataset": config["source"]["datasets"]["B_true"],
                "bytes_sha256": source["dataset_hashes"]["B_true"],
                "shift_boundary": "periodic",
                "scan_shift": "integer_pixel",
            },
        },
        truth={
            "P_B_true": source["P_B_true"],
            "B_true": source["B_true"],
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


def _audit_artifacts(
    run_dir: Path,
    *,
    config: dict[str, Any],
    expected_status: str,
    source: dict[str, Any],
    figure_paths: list[Path],
) -> dict[str, Any]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    if (
        metrics["experiment_status"] != expected_status
        or metadata["experiment_status"] != expected_status
    ):
        raise RuntimeError("External exp052 statuses disagree.")
    with h5py.File(hdf5_path, "r") as h5:
        expected_entry = {
            "config_yaml",
            "data",
            "instrument",
            "sample",
            "truth",
            "reconstruction",
            "metadata",
            "metrics",
        }
        if set(h5["/entry"].keys()) != expected_entry:
            raise RuntimeError("Unexpected exp052 /entry tree.")
        if "calibration" in h5["/entry"] or "preprocessing" in h5["/entry"]:
            raise RuntimeError("exp052 wrote a fake processing group.")
        if (
            sha256_array(np.asarray(h5["/entry/data/I_stack"]))
            != source["dataset_hashes"]["I_stack"]
        ):
            raise RuntimeError("Saved detector stack differs from exp030 source bytes.")
        if (
            sha256_array(np.asarray(h5["/entry/data/scan_positions"]))
            != source["dataset_hashes"]["scan_positions"]
        ):
            raise RuntimeError("Saved scan differs from exp030 source bytes.")
        known_root = h5["/entry/reconstruction/known_b_probe"]
        blind_root = h5["/entry/reconstruction/blind_epie_probe"]
        if not np.array_equal(known_root["B_fixed"][()], source["B_true"]):
            raise RuntimeError("Part A fixed-B invariant failed in HDF5.")
        if bool(blind_root["settings/uses_simulation_truth_B_as_input"][()]) or bool(
            blind_root["settings/uses_simulation_truth_probe_as_input"][()]
        ):
            raise RuntimeError("Part B truth-free provenance failed in HDF5.")
        cache = h5["/entry/reconstruction/waist_fit/candidate_cache"]
        candidate_fields = np.asarray(cache["P_B_candidate"])
        candidate_hashes = cache["candidate_sha256"].asstr()[()]
        for index in range(len(candidate_fields)):
            if sha256_array(candidate_fields[index]) != candidate_hashes[index]:
                raise RuntimeError("Candidate cache field/hash mapping failed.")
        for arm in ("known_b", "blind_epie"):
            fit = h5[f"/entry/reconstruction/waist_fit/{arm}"]
            best = np.asarray(fit["P_B_best_raw"])
            target = np.asarray(
                known_root["P_B_rec_raw"]
                if arm == "known_b"
                else blind_root["P_B_rec_raw"]
            )
            phase = float(fit["global_phase_rad"][()])
            if not np.array_equal(best - target, fit["residual_field_raw"][()]):
                raise RuntimeError(f"{arm} raw residual identity failed.")
            expected_primary = np.exp(1j * phase) * best - target
            if not np.allclose(
                expected_primary,
                fit["residual_field_primary"][()],
                rtol=0.0,
                atol=1e-15,
            ):
                raise RuntimeError(f"{arm} primary residual identity failed.")
        numeric_failures: list[str] = []

        def finite(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if (
                isinstance(obj, h5py.Dataset)
                and obj.dtype.kind in "iufc"
                and not np.all(np.isfinite(obj[()]))
            ):
                numeric_failures.append(name)

        h5.visititems(finite)
        if numeric_failures:
            raise RuntimeError(
                f"Non-finite exp052 HDF5 datasets: {numeric_failures[:3]}."
            )
        path_count = 0

        def count(_name: str, _obj: h5py.Group | h5py.Dataset) -> None:
            nonlocal path_count
            path_count += 1

        h5.visititems(count)
        if h5["/entry/metrics/experiment_status"].asstr()[()] != expected_status:
            raise RuntimeError("JSON/HDF5 status mismatch.")
    figure_audit: dict[str, Any] = {}
    for path in figure_paths:
        image = plt.imread(path)
        if image.ndim not in {2, 3} or not np.all(np.isfinite(image)):
            raise RuntimeError(f"Unreadable exp052 figure: {path}.")
        figure_audit[path.name] = {
            "shape": list(image.shape),
            "sha256": sha256_file(path),
        }
    return {
        "passed": True,
        "hdf5_path_count": path_count,
        "all_numeric_hdf5_datasets_finite": True,
        "json_hdf5_status_consistent": True,
        "source_data_hashes_verified": True,
        "known_B_fixed_verified": True,
        "blind_truth_free_flags_verified": True,
        "candidate_cache_mapping_verified": True,
        "raw_primary_residual_identity_verified": True,
        "figures": figure_audit,
    }


def run(config_path: Path, *, mode: str) -> Path:
    """Execute one exp052 development/preflight or combined formal run."""

    config = load_config(config_path)
    validate_exp052_config(config, mode=mode)
    run_name = config["output"]["run_name"] + (
        "_preflight" if mode == "preflight" else ""
    )
    run_dir = make_run_dir(PROJECT_ROOT / config["run"]["output_root"], run_name)
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {"status": "running", "mode": mode, "artifacts_validated": False},
    )
    started = time.perf_counter()
    try:
        source = load_exp052_sources(config, PROJECT_ROOT)
        generator = make_candidate_cache(source)
        true_waist = float(config["sample_a"]["d_waist_true_m"])
        replay_first = generator.generator(true_waist)
        replay_second = generator.generator(true_waist)
        probe_replay = _relative_l2(np.asarray(replay_first["P_B"]), source["P_B_true"])
        probe_repeat = _relative_l2(
            np.asarray(replay_second["P_B"]), np.asarray(replay_first["P_B"])
        )
        measurement_replay = replay_measurement_operator(source)
        shared = {
            "source_probe_replay_relative_l2": probe_replay,
            "source_probe_deterministic_repeat_relative_l2": probe_repeat,
            **measurement_replay,
        }
        design = config[mode]
        thresholds = design["thresholds"]
        shared_pass = bool(
            probe_replay <= thresholds["source_probe_replay_relative_l2_max"]
            and probe_repeat <= thresholds["source_probe_replay_relative_l2_max"]
            and measurement_replay["intensity_relative_l2"]
            <= thresholds["measurement_intensity_replay_relative_l2_max"]
            and measurement_replay["amplitude_relative_l2"]
            <= thresholds["measurement_amplitude_replay_relative_l2_max"]
            and measurement_replay["adjoint_inner_product_relative_error"]
            <= thresholds["adjoint_inner_product_relative_error_max"]
        )
        shared["pass"] = shared_pass

        known_settings = {
            **config["part_a"],
            **design["part_a_reconstruction"],
        }
        known = run_known_b_reconstruction(
            source,
            known_settings,
            repeat=bool(design["part_a_reconstruction"]["repeat"]),
        )
        known["simulation_evaluation_only"] = truth_evaluation(
            known["P_B_rec_raw"], known["B_fixed"], source
        )
        blind = {
            **source["blind"],
            "independent_data_fidelity_loss": frozen_data_fidelity(
                source["blind"]["P_B_rec_raw"], source["blind"]["B_rec_raw"], source
            ),
        }
        blind["simulation_evaluation_only"] = truth_evaluation(
            blind["P_B_rec_raw"], blind["B_rec_raw"], source
        )

        known_residual_difference = abs(
            known["final_data_fidelity_loss"] - known["independent_data_fidelity_loss"]
        )
        blind_residual_difference = abs(
            blind["final_data_fidelity_loss"] - blind["independent_data_fidelity_loss"]
        )
        known_reconstruction_pass = bool(
            known["final_data_fidelity_loss"]
            <= thresholds["part_a_final_data_fidelity_loss_max"]
            and known["fixed_B_max_abs_change"]
            <= thresholds["part_a_fixed_B_max_abs_change_max"]
            and known_residual_difference
            <= thresholds["saved_residual_recompute_absolute_difference_max"]
            and known["repeat"]["P_B_relative_l2"]
            <= thresholds["part_a_repeat_relative_l2_max"]
            and known["repeat"]["bitwise_equal"]
            and known["settings"]["update_probe"]
            and not known["settings"]["update_object"]
        )
        blind_probe_update = _relative_l2(blind["P_B_rec_raw"], blind["P_B_init"])
        blind_object_update = _relative_l2(blind["B_rec_raw"], blind["B_init"])
        checkpoint_losses = np.asarray(
            [
                blind["checkpoints"][str(value)]["data_fidelity_loss"]
                for value in config["part_b"]["checkpoints_for_stability"]
            ]
        )
        blind_reconstruction_pass = bool(
            blind["final_data_fidelity_loss"]
            <= thresholds["part_b_final_data_fidelity_loss_max"]
            and blind_residual_difference
            <= thresholds["saved_residual_recompute_absolute_difference_max"]
            and blind_probe_update >= thresholds["part_b_probe_update_relative_l2_min"]
            and blind_object_update
            >= thresholds["part_b_object_update_relative_l2_min"]
            and np.all(np.diff(checkpoint_losses) < 0.0)
            and blind["settings"]["update_probe"]
            and blind["settings"]["update_object"]
            and not blind["settings"]["uses_simulation_truth_B_as_input"]
            and not blind["settings"]["uses_simulation_truth_probe_as_input"]
        )

        known_fit = fit_reconstructed_probe(
            known["P_B_rec_raw"], generator, design=design
        )
        blind_fit = fit_reconstructed_probe(
            blind["P_B_rec_raw"], generator, design=design
        )
        known_checkpoint_fits = summarize_checkpoint_fits(
            {key: value["P_B_rec_raw"] for key, value in known["checkpoints"].items()},
            generator,
            design=design,
        )
        blind_checkpoint_fits = summarize_checkpoint_fits(
            {key: value["P_B_rec_raw"] for key, value in blind["checkpoints"].items()},
            generator,
            design=design,
        )
        known_fit_gates = evaluate_fit_gates(
            known_fit, true_waist_m=true_waist, thresholds=thresholds
        )
        blind_fit_gates = evaluate_fit_gates(
            blind_fit, true_waist_m=true_waist, thresholds=thresholds
        )
        known_checkpoint_estimates = np.asarray(
            [value["estimate_m"] for value in known_checkpoint_fits.values()]
        )
        blind_checkpoint_estimates = np.asarray(
            [value["estimate_m"] for value in blind_checkpoint_fits.values()]
        )
        known_checkpoint_spread = float(
            np.max(known_checkpoint_estimates) - np.min(known_checkpoint_estimates)
        )
        blind_checkpoint_spread = float(
            np.max(blind_checkpoint_estimates) - np.min(blind_checkpoint_estimates)
        )
        known_fit_pass = bool(
            known_fit_gates["pass"]
            and known_checkpoint_spread
            <= thresholds["checkpoint_estimate_spread_m_max"]
        )
        blind_fit_pass = bool(
            blind_fit_gates["pass"]
            and blind_checkpoint_spread
            <= thresholds["checkpoint_estimate_spread_m_max"]
        )

        component_statuses = combine_status(
            shared_status="Passed" if shared_pass else "Inconclusive",
            part_a_reconstruction_status="Passed"
            if known_reconstruction_pass
            else "Failed",
            part_a_fit_status="Passed" if known_fit_pass else "Failed",
            part_b_reconstruction_status="Passed"
            if blind_reconstruction_pass
            else "Failed",
            part_b_fit_status="Passed" if blind_fit_pass else "Failed",
        )
        scientific_status = component_statuses["overall"]
        experiment_status = scientific_status if mode == "formal" else "Development"
        oracle_estimate = float(source["oracle"]["estimate_m"])
        known_estimate = float(known_fit["estimate_m"])
        blind_estimate = float(blind_fit["estimate_m"])
        comparison = {
            "estimates_m": np.asarray(
                [oracle_estimate, known_estimate, blind_estimate]
            ),
            "intervals_m": np.vstack(
                [
                    source["oracle"]["interval_m"],
                    known_fit["reported_interval_m"],
                    blind_fit["reported_interval_m"],
                ]
            ),
            "known_minus_oracle_m": known_estimate - oracle_estimate,
            "blind_minus_oracle_m": blind_estimate - oracle_estimate,
            "blind_minus_known_m": blind_estimate - known_estimate,
            "minimum_primary_loss": np.asarray(
                [
                    0.0,
                    known_fit["minimum_primary_loss"],
                    blind_fit["minimum_primary_loss"],
                ]
            ),
            "local_loss_curvature_per_m2": np.asarray(
                [
                    source["oracle"]["loss_curvature_per_m2"],
                    known_fit["local_loss_curvature_per_m2"],
                    blind_fit["local_loss_curvature_per_m2"],
                ]
            ),
            "checkpoint_estimate_spread_m": {
                "known_b": known_checkpoint_spread,
                "blind_epie": blind_checkpoint_spread,
            },
            "gauge_contract": config["gauge"],
            "boundary_hit": {
                "known_b": known_fit["boundary_hit"],
                "blind_epie": blind_fit["boundary_hit"],
            },
            "equivalence_component_count": {
                "known_b": known_fit["equivalence_component_count"],
                "blind_epie": blind_fit["equivalence_component_count"],
            },
        }
        source_identity = _source_identity(config, source)
        metrics = {
            "experiment_status": experiment_status,
            "scientific_status_if_formal": scientific_status,
            "mode": mode,
            "component_statuses": component_statuses,
            "shared_source_operator": shared,
            "part_a": {
                "reconstruction_status": component_statuses["part_a_reconstruction"],
                "fit_status": component_statuses["part_a_fit"],
                "overall_status": component_statuses["part_a_overall"],
                "reconstruction": {
                    "initial_data_fidelity_loss": known["initial_data_fidelity_loss"],
                    "final_data_fidelity_loss": known["final_data_fidelity_loss"],
                    "independent_data_fidelity_loss": known[
                        "independent_data_fidelity_loss"
                    ],
                    "saved_residual_absolute_difference": known_residual_difference,
                    "fixed_B_max_abs_change": known["fixed_B_max_abs_change"],
                    "repeat": known["repeat"],
                    "P_B_init_sha256": sha256_array(known["P_B_init"]),
                    "P_B_rec_raw_sha256": sha256_array(known["P_B_rec_raw"]),
                    "B_known_sha256": sha256_array(known["B_known"]),
                    "simulation_evaluation_only": known["simulation_evaluation_only"],
                },
                "fit": _fit_summary(known_fit, true_waist),
                "fit_gates": known_fit_gates,
                "checkpoint_fit": _checkpoint_fit_summary(
                    known_checkpoint_fits, true_waist
                ),
                "checkpoint_estimate_spread_m": known_checkpoint_spread,
            },
            "part_b": {
                "reconstruction_status": component_statuses["part_b_reconstruction"],
                "fit_status": component_statuses["part_b_fit"],
                "overall_status": component_statuses["part_b_overall"],
                "reconstruction": {
                    "initial_data_fidelity_loss": float(blind["loss_curve"][0]),
                    "final_data_fidelity_loss": blind["final_data_fidelity_loss"],
                    "independent_data_fidelity_loss": blind[
                        "independent_data_fidelity_loss"
                    ],
                    "saved_residual_absolute_difference": blind_residual_difference,
                    "probe_update_relative_l2": blind_probe_update,
                    "object_update_relative_l2": blind_object_update,
                    "checkpoint_losses": checkpoint_losses,
                    "P_B_init_sha256": sha256_array(blind["P_B_init"]),
                    "B_init_sha256": sha256_array(blind["B_init"]),
                    "P_B_rec_raw_sha256": sha256_array(blind["P_B_rec_raw"]),
                    "B_rec_raw_sha256": sha256_array(blind["B_rec_raw"]),
                    "truth_use_flags": {"P_B_true": False, "B_true": False},
                    "simulation_evaluation_only": blind["simulation_evaluation_only"],
                },
                "fit": _fit_summary(blind_fit, true_waist),
                "fit_gates": blind_fit_gates,
                "checkpoint_fit": _checkpoint_fit_summary(
                    blind_checkpoint_fits, true_waist
                ),
                "checkpoint_estimate_spread_m": blind_checkpoint_spread,
            },
            "comparison": comparison,
            "source_identity": source_identity,
            "claim_boundary": (
                "noiseless matched exp030 2D projected single-parameter "
                "reconstructed-probe diagnostic only"
            ),
            "project_level_hdf5_schema_changed": False,
        }
        metadata = {
            "experiment": "exp052_TGV_2d_projected_reconstructed_probe_waist_fit",
            "phase": "Phase 5 projected diagnostic",
            "mode": mode,
            "experiment_status": experiment_status,
            "created_at": created_at_utc(),
            "git_commit": get_git_commit(PROJECT_ROOT) or "",
            "config_path": str(config_path),
            "dataset_type": "simulation",
            "axis_order": "y_x",
            "scan_coordinate_order": "x_y",
            "dx_tuple_order_if_used": "dy_dx",
            "source_identity": source_identity,
            "truth_use": (
                "optimizer and source selection truth-free; truth only under "
                "simulation_evaluation_only after reconstruction and fit"
            ),
            "gauge_contract": config["gauge"],
            "formal_layout": config["experiment"]["formal_layout"],
            "project_level_hdf5_schema_changed": False,
        }
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
        _write_hdf5(
            hdf5_path,
            config=config,
            metadata=metadata,
            metrics=metrics,
            source=source,
            source_identity=source_identity,
            shared=shared,
            known=known,
            blind=blind,
            known_fit=known_fit,
            blind_fit=blind_fit,
            known_checkpoint_fits=known_checkpoint_fits,
            blind_checkpoint_fits=blind_checkpoint_fits,
            comparison=comparison,
            candidate_cache=generator,
        )
        figure_paths = _save_figures(
            run_dir,
            known=known,
            blind=blind,
            known_fit=known_fit,
            blind_fit=blind_fit,
            known_checkpoint_fits=known_checkpoint_fits,
            blind_checkpoint_fits=blind_checkpoint_fits,
            oracle=source["oracle"],
            comparison=comparison,
        )
        metrics["artifact_audit"] = _audit_artifacts(
            run_dir,
            config=config,
            expected_status=experiment_status,
            source=source,
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
            shared=shared,
            known=known,
            blind=blind,
            known_fit=known_fit,
            blind_fit=blind_fit,
            known_checkpoint_fits=known_checkpoint_fits,
            blind_checkpoint_fits=blind_checkpoint_fits,
            comparison=comparison,
            candidate_cache=generator,
        )
        metrics["artifact_audit"] = _audit_artifacts(
            run_dir,
            config=config,
            expected_status=experiment_status,
            source=source,
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
            shared=shared,
            known=known,
            blind=blind,
            known_fit=known_fit,
            blind_fit=blind_fit,
            known_checkpoint_fits=known_checkpoint_fits,
            blind_checkpoint_fits=blind_checkpoint_fits,
            comparison=comparison,
            candidate_cache=generator,
        )
        elapsed = time.perf_counter() - started
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "artifacts_validated": True,
                "mode": mode,
                "experiment_status": experiment_status,
                "scientific_status_if_formal": scientific_status,
                "component_statuses": component_statuses,
                "runtime_seconds": elapsed,
                "artifact_sha256": {
                    "config": sha256_file(run_dir / "config.yaml"),
                    "metadata": sha256_file(run_dir / "metadata.json"),
                    "metrics": sha256_file(run_dir / "metrics.json"),
                    "hdf5": sha256_file(hdf5_path),
                    "figures": {path.name: sha256_file(path) for path in figure_paths},
                },
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
        f"known={known_fit['estimate_m'] * 1e6:.9f} um; "
        f"blind={blind_fit['estimate_m'] * 1e6:.9f} um"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config, mode=args.mode)


if __name__ == "__main__":
    main()
