from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.exp040 import relative_l2  # noqa: E402
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.recon.exp043 import (  # noqa: E402
    MatchedBlindProbeBOperator,
    alternating_blind_reconstruction,
    detector_amplitude_relative_residual,
    determine_status,
    known_b_probe_control,
    known_probe_b_control,
    load_exp042_authoritative_source,
    make_b_initialization,
    operator_control_metrics,
    pairwise_blind_stability,
    sha256_array_bytes,
    sha256_file,
    simulation_evaluation_only,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp043 blind probe/B alternating block GN-CG."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment",
        "execution",
        "source",
        "loss",
        "gauge",
        "operator_controls",
        "known_b_probe_control",
        "known_probe_B_control",
        "blind",
        "formal_thresholds",
        "artifact_contract",
        "output",
    }
    if set(config) != required:
        raise ValueError(f"exp043 root config keys changed: {set(config) ^ required}.")
    if config["experiment"].get("id") != "exp043":
        raise ValueError("experiment.id must be exp043.")
    if config["source"].get("experiment") != "exp042":
        raise ValueError("exp043 must retain the exp042 source.")
    if config["source"].get("reference_validated") is not False:
        raise ValueError("reference_validated must remain false.")
    if config["source"].get("full_tgv_reference_authorized") is not False:
        raise ValueError("full_tgv_reference_authorized must remain false.")
    if config["loss"] != {
        "type": "mean_half_squared_pixel_intensity_residual",
        "uses_measured_pixel_intensity_directly": True,
        "amplitude_replacement": False,
    }:
        raise ValueError("The frozen exp042 loss boundary changed.")
    if config["gauge"].get("truth_used_by_primary_gauge") is not False:
        raise ValueError("Simulation truth is forbidden from the primary gauge.")
    blind = config["blind"]
    names = [str(item["name"]) for item in blind["initializations"]]
    if len(names) < 1 or len(names) != len(set(names)):
        raise ValueError("Blind initialization names must be nonempty and unique.")
    if blind["representative_branch"] != names[0]:
        raise ValueError(
            "The representative branch must be the preregistered first branch."
        )
    if blind.get("checkpoint_selection_rule") != "fixed_final_outer_sweep":
        raise ValueError("Checkpoint selection must remain fixed-final.")
    if config["execution"].get("stage") not in {
        "development_preflight",
        "formal",
    }:
        raise ValueError("execution.stage must be development_preflight or formal.")
    formal = config["execution"]["stage"] == "formal"
    if bool(config["execution"]["scientific_gates_enabled"]) != formal:
        raise ValueError("Scientific gates are enabled only for formal execution.")


def _loss_nonincreasing(values: Any) -> bool:
    array = np.asarray(values, dtype=np.float64)
    tolerance = 16.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(array))), 1.0)
    return bool(np.all(np.diff(array) <= tolerance))


def _phase_align_probe(
    probe: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, complex]:
    denominator = float(np.sum(np.abs(probe) ** 2))
    factor = np.sum(np.conj(probe) * truth, dtype=np.complex128) / max(
        denominator, 1e-30
    )
    return np.asarray(factor * probe, dtype=np.complex128), complex(factor)


def _branch_summary(result: dict[str, Any], runtime_seconds: float) -> dict[str, Any]:
    last_blocks = result["block_records"][-1]
    probe_block = last_blocks["probe"]
    sample_b_block = last_blocks["sample_b"]
    return {
        "initial_loss": float(result["loss_curve"][0]),
        "final_loss": float(result["loss_curve"][-1]),
        "initial_detector_relative_residual": float(
            result["detector_relative_residual_curve"][0]
        ),
        "final_detector_relative_residual": float(
            result["detector_relative_residual_curve"][-1]
        ),
        "initial_detector_amplitude_relative_residual": float(
            result["detector_amplitude_relative_residual_curve"][0]
        ),
        "final_detector_amplitude_relative_residual": float(
            result["detector_amplitude_relative_residual_curve"][-1]
        ),
        "loss_nonincreasing": _loss_nonincreasing(result["loss_curve"]),
        "outer_sweeps_completed": int(result["outer_sweeps_completed"]),
        "stopping_reason": str(result["stopping_reason"]),
        "operator_action_budget_units": int(result["operator_action_budget_units"]),
        "final_probe_block_gradient_l2_norm": float(
            probe_block["gradient_l2_norm_curve"][-1]
        ),
        "final_sample_b_block_gradient_l2_norm": float(
            sample_b_block["gradient_l2_norm_curve"][-1]
        ),
        "total_backtracking_steps": int(
            sum(
                np.sum(record[name]["backtracking_count_curve"])
                for record in result["block_records"]
                for name in ("probe", "sample_b")
            )
        ),
        "final_probe_update_relative_l2": float(
            result["probe_update_relative_l2_curve"][-1]
        ),
        "final_sample_b_update_relative_l2": float(
            result["sample_b_update_relative_l2_curve"][-1]
        ),
        "probe_norm_final_to_initial_ratio": float(
            result["gauge_probe_l2_norm_curve"][-1]
            / result["gauge_probe_l2_norm_curve"][0]
        ),
        "active_B_mean_phase_final_minus_initial_rad": float(
            result["gauge_b_active_mean_phase_curve"][-1]
            - result["gauge_b_active_mean_phase_curve"][0]
        ),
        "checkpoint_selection_rule": str(result["checkpoint_selection_rule"]),
        "runtime_seconds": runtime_seconds,
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "truth_used_by_checkpoint_selection": False,
        "truth_used_by_stopping": False,
    }


def _hdf5_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _hdf5_ready(child) for key, child in value.items()}
    if (
        isinstance(value, list)
        and value
        and all(isinstance(child, dict) for child in value)
    ):
        return {
            f"item_{index:03d}": _hdf5_ready(child) for index, child in enumerate(value)
        }
    return value


def _known_b_hdf5(result: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload["P_B_rec_raw"] = payload.pop("P_B_rec")
    payload["simulation_evaluation_only"] = evaluation
    return _hdf5_ready(payload)


def _known_probe_b_hdf5(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.pop("probe", None)
    payload.pop("modulation", None)
    return _hdf5_ready(payload)


def _blind_hdf5(result: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    canonical = dict(payload.pop("canonicalization"))
    payload["gauge"] = canonical
    payload["simulation_evaluation_only"] = evaluation
    return _hdf5_ready(payload)


def _save_figures(
    run_dir: Path,
    filenames: list[str],
    source: Any,
    p_control: dict[str, Any],
    b_control: dict[str, Any],
    blind_branches: dict[str, dict[str, Any]],
    representative_name: str,
    stability: dict[str, Any],
) -> None:
    if len(filenames) != 5:
        raise ValueError("The exp043 figure contract requires exactly five figures.")
    representative = blind_branches[representative_name]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].semilogy(p_control["loss_curve"], label="known-B P")
    axes[0].semilogy(b_control["loss_curve"], label="known-probe B")
    for name, result in blind_branches.items():
        axes[0].semilogy(result["loss_curve"], marker="o", label=f"blind {name}")
    axes[0].set(
        xlabel="iteration / outer sweep", ylabel="mean half squared intensity loss"
    )
    axes[0].legend(fontsize=7)
    axes[1].semilogy(p_control["detector_relative_residual_curve"], label="known-B P")
    axes[1].semilogy(
        b_control["detector_relative_residual_curve"], label="known-probe B"
    )
    for name, result in blind_branches.items():
        axes[1].semilogy(
            result["detector_relative_residual_curve"], marker="o", label=name
        )
    axes[1].set(
        xlabel="iteration / outer sweep", ylabel="detector intensity relative L2"
    )
    axes[1].legend(fontsize=7)
    axes[2].axis("off")
    blind_final_residual = representative["detector_relative_residual_curve"][-1]
    axes[2].text(
        0.02,
        0.98,
        "exp043 controls\n"
        f"P control exact: {p_control['authoritative_raw_exact']}\n"
        f"B final residual: {b_control['detector_relative_residual_curve'][-1]:.3e}\n"
        f"blind final residual: {blind_final_residual:.3e}\n"
        "truth is not used by optimization",
        va="top",
        family="monospace",
    )
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / filenames[0], dpi=160)
    plt.close(figure)

    p_true = source.P_B_true
    p_init = representative["P_B_init_raw"]
    p_rec = representative["P_B_rec_raw"]
    p_aligned, _ = _phase_align_probe(p_rec, p_true)
    panels = [
        (np.abs(p_true), "target amplitude", "amplitude"),
        (np.abs(p_init), "initial amplitude", "amplitude"),
        (np.abs(p_rec), "raw reconstructed amplitude", "amplitude"),
        (np.angle(p_true), "target phase", "rad"),
        (np.angle(p_rec), "raw reconstructed phase", "rad"),
        (np.abs(p_aligned - p_true), "simulation-only aligned |error|", "amplitude"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for axis, (values, title, label) in zip(axes.flat, panels, strict=True):
        image = axis.imshow(
            values, origin="lower", cmap="twilight" if label == "rad" else "viridis"
        )
        axis.set_title(title, fontsize=9)
        axis.set(xlabel="x pixel", ylabel="y pixel")
        figure.colorbar(image, ax=axis, shrink=0.76, label=label)
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / filenames[1], dpi=160)
    plt.close(figure)

    b_true = source.B_true
    b_init = representative["B_init_raw"]
    b_rec = representative["B_rec_raw"]
    b_panels = [
        (np.abs(b_true), "target B amplitude", "amplitude"),
        (np.angle(b_true), "target B phase", "rad"),
        (np.abs(b_init), "initial B amplitude", "amplitude"),
        (np.angle(b_init), "initial B phase", "rad"),
        (np.abs(b_rec), "raw reconstructed B amplitude", "amplitude"),
        (np.angle(b_rec), "raw reconstructed B phase", "rad"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (values, title, label) in zip(axes.flat, b_panels, strict=True):
        image = axis.imshow(
            values, origin="lower", cmap="twilight" if label == "rad" else "viridis"
        )
        axis.set_title(title, fontsize=9)
        axis.set(xlabel="x pixel", ylabel="y pixel")
        figure.colorbar(image, ax=axis, shrink=0.76, label=label)
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / filenames[2], dpi=160)
    plt.close(figure)

    scan_index = 0
    target = source.I_stack[scan_index]
    prediction = representative["prediction_final"][scan_index]
    residual = prediction - target
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    detector_panels = [
        (target, "target detector intensity", "intensity"),
        (prediction, "blind predicted intensity", "intensity"),
        (residual, "prediction - target", "intensity residual"),
    ]
    for axis, (values, title, label) in zip(axes, detector_panels, strict=True):
        image = axis.imshow(
            values, origin="lower", cmap="coolwarm" if "residual" in label else "magma"
        )
        axis.set_title(f"scan 0: {title}", fontsize=9)
        axis.set(xlabel="detector x pixel", ylabel="detector y pixel")
        figure.colorbar(image, ax=axis, shrink=0.76, label=label)
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / filenames[3], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(
        representative["probe_update_relative_l2_curve"], marker="o", label="probe"
    )
    axes[0, 0].plot(
        representative["sample_b_update_relative_l2_curve"], marker="s", label="B"
    )
    axes[0, 0].set(xlabel="outer sweep", ylabel="relative block update")
    axes[0, 0].legend()
    axes[0, 1].plot(representative["gauge_probe_l2_norm_curve"], label="probe L2")
    twin = axes[0, 1].twinx()
    twin.plot(
        representative["gauge_b_active_rms_amplitude_curve"],
        color="tab:orange",
        label="B RMS",
    )
    axes[0, 1].set(xlabel="outer sweep", ylabel="probe L2")
    twin.set_ylabel("active B RMS amplitude")
    matrix = np.asarray(stability["pairwise_prediction_relative_l2"])
    image = axes[1, 0].imshow(matrix, origin="lower", cmap="viridis")
    axes[1, 0].set_title("truth-free pairwise prediction relative L2")
    axes[1, 0].set_xticks(
        range(len(stability["branch_names"])), stability["branch_names"], rotation=30
    )
    axes[1, 0].set_yticks(
        range(len(stability["branch_names"])), stability["branch_names"]
    )
    figure.colorbar(image, ax=axes[1, 0], shrink=0.76)
    axes[1, 1].plot(representative["gauge_b_active_mean_phase_curve"], marker="o")
    axes[1, 1].set(xlabel="outer sweep", ylabel="active B circular-mean phase (rad)")
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / filenames[4], dpi=160)
    plt.close(figure)


def _all_numeric_finite(group: h5py.Group) -> bool:
    valid = True

    def visitor(_name: str, value: h5py.Group | h5py.Dataset) -> None:
        nonlocal valid
        if isinstance(value, h5py.Dataset) and value.dtype.kind in "biufc":
            valid = valid and bool(np.all(np.isfinite(value[...])))

    group.visititems(visitor)
    return valid


def _validate_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    required_paths = [
        "/entry/data/I_stack",
        "/entry/data/scan_positions",
        "/entry/truth/P_B_true",
        "/entry/truth/B_true",
        "/entry/reconstruction/source_provenance",
        "/entry/reconstruction/design",
        "/entry/reconstruction/operator_controls",
        "/entry/reconstruction/known_b_probe_control/P_B_rec_raw",
        "/entry/reconstruction/known_probe_B_control/B_rec_raw",
        "/entry/reconstruction/blind_joint/representative/P_B_rec_raw",
        "/entry/reconstruction/blind_joint/representative/B_rec_raw",
        "/entry/reconstruction/blind_joint/representative/P_B_rec_canonical",
        "/entry/reconstruction/blind_joint/representative/B_rec_canonical",
        "/entry/reconstruction/blind_joint/representative/simulation_evaluation_only/P_B_rec_reciprocal_gain_aligned",
        "/entry/reconstruction/blind_joint/representative/simulation_evaluation_only/B_rec_reciprocal_gain_aligned",
        "/entry/reconstruction/gauge",
        "/entry/reconstruction/checkpoints",
        "/entry/metadata",
        "/entry/metrics",
    ]
    with h5py.File(hdf5_path, "r") as h5:
        entry_children = sorted(h5["entry"].keys())
        expected_children = sorted(
            config["artifact_contract"]["required_entry_children"]
        )
        if entry_children != expected_children:
            raise RuntimeError("HDF5 /entry children violate the frozen contract.")
        if "calibration" in h5["entry"] or "preprocessing" in h5["entry"]:
            raise RuntimeError(
                "Simulation output must not fabricate calibration/preprocessing."
            )
        missing = [path for path in required_paths if path not in h5]
        if missing:
            raise RuntimeError(f"Missing required HDF5 paths: {missing}.")
        source = config["source"]
        for key in ("I_stack", "scan_positions", "P_B_true", "B_true"):
            if key in {"I_stack", "scan_positions"}:
                path = f"/entry/data/{key}"
            else:
                path = f"/entry/truth/{key}"
            digest = sha256_array_bytes(h5[path][...])
            if digest != source["dataset_sha256"][key]:
                raise RuntimeError(f"Persisted source dataset changed: {path}.")
        if not _all_numeric_finite(h5["entry"]):
            raise RuntimeError("HDF5 contains a non-finite numeric dataset.")
        h5_status = h5["/entry/metrics/scientific_status"][()]
        if isinstance(h5_status, bytes):
            h5_status = h5_status.decode("utf-8")
        if str(h5_status) != metrics["scientific_status"]:
            raise RuntimeError("JSON/HDF5 scientific status mismatch.")

    figure_hashes = {}
    for name in config["output"]["figure_filenames"]:
        path = run_dir / "figures" / name
        with Image.open(path) as image:
            image.verify()
        figure_hashes[name] = sha256_file(path)
    return {
        "required_hdf5_paths_present": True,
        "entry_children_exact": True,
        "source_dataset_bytes_exact": True,
        "all_numeric_hdf5_datasets_finite": True,
        "json_hdf5_status_consistent": True,
        "figure_count": len(figure_hashes),
        "figure_sha256": figure_hashes,
    }


def run(config_path: Path) -> Path:
    start = time.perf_counter()
    config = load_config(config_path)
    _validate_config(config)
    output = config["output"]
    run_dir = make_run_dir(PROJECT_ROOT / output["root"], output["run_name"])
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "artifacts_validated": False,
            "created_at_utc": created_at_utc(),
        },
    )

    try:
        source = load_exp042_authoritative_source(PROJECT_ROOT, config["source"])
        operator = MatchedBlindProbeBOperator(source.operator, source.support_mask)
        controls = operator_control_metrics(
            operator, source, {**config["operator_controls"], **config["gauge"]}
        )

        p_control_start = time.perf_counter()
        p_control = known_b_probe_control(source)
        p_control_runtime = time.perf_counter() - p_control_start
        p_aligned, p_factor = _phase_align_probe(p_control["P_B_rec"], source.P_B_true)
        p_evaluation = {
            "P_B_rec_complex_gain_aligned": p_aligned,
            "complex_gain": np.complex128(p_factor),
            "raw_relative_l2": relative_l2(p_control["P_B_rec"], source.P_B_true),
            "aligned_relative_l2": relative_l2(p_aligned, source.P_B_true),
            "simulation_evaluation_only": True,
            "enters_optimizer": False,
            "enters_selection": False,
            "enters_stopping": False,
        }

        b_control_start = time.perf_counter()
        b_control = known_probe_b_control(
            operator, source, config["known_probe_B_control"]
        )
        b_control_runtime = time.perf_counter() - b_control_start

        blind_branches: dict[str, dict[str, Any]] = {}
        blind_evaluations: dict[str, dict[str, Any]] = {}
        blind_summaries: dict[str, dict[str, Any]] = {}
        for initialization in config["blind"]["initializations"]:
            name = str(initialization["name"])
            b_init = make_b_initialization(source.support_mask, initialization)
            branch_start = time.perf_counter()
            result = alternating_blind_reconstruction(
                operator,
                source.I_stack,
                source.operator.homogeneous_probe_native,
                b_init,
                config["blind"],
                seed_offset=int(initialization["seed_offset"]),
            )
            branch_runtime = time.perf_counter() - branch_start
            # The raw fixed-final branch exists before any truth is read here.
            blind_branches[name] = result
            blind_summaries[name] = _branch_summary(result, branch_runtime)
            blind_evaluations[name] = simulation_evaluation_only(
                operator, result, source
            )

        representative_name = str(config["blind"]["representative_branch"])
        representative = blind_branches[representative_name]
        representative_evaluation = blind_evaluations[representative_name]
        stability = pairwise_blind_stability(blind_branches)

        tolerances = config["operator_controls"]["tolerances"]
        exp042_controls = controls["exp042_known_b_operator"]
        source_operator_gates = (
            source.replay["independent_forward_replay_relative_l2"]
            <= float(tolerances["replay_relative_l2_max"])
            and exp042_controls["linear_adjoint_relative_error"]
            <= float(tolerances["linear_adjoint_relative_error_max"])
            and exp042_controls["intensity_jacobian_adjoint_relative_error"]
            <= float(tolerances["intensity_jacobian_adjoint_relative_error_max"])
            and exp042_controls["full_loss_directional_gradient_relative_error"]
            <= float(tolerances["loss_directional_gradient_relative_error_max"])
            and controls["B_intensity_jacobian_directional_relative_error"]
            <= float(tolerances["B_jacobian_directional_relative_error_max"])
            and controls["B_intensity_jacobian_real_adjoint_relative_error"]
            <= float(tolerances["B_jacobian_adjoint_relative_error_max"])
            and controls["B_phase_jacobian_directional_relative_error"]
            <= float(tolerances["B_phase_jacobian_directional_relative_error_max"])
            and controls["B_phase_jacobian_real_adjoint_relative_error"]
            <= float(tolerances["B_phase_jacobian_adjoint_relative_error_max"])
            and controls["combined_loss_directional_gradient_relative_error"]
            <= float(
                tolerances["combined_loss_directional_gradient_relative_error_max"]
            )
            and controls["B_normal_action_symmetry_relative_error"]
            <= float(tolerances["B_normal_symmetry_relative_error_max"])
            and controls["B_normal_action_psd_quadratic_form"]
            >= float(tolerances["B_normal_psd_min"])
            and controls["B_phase_normal_action_symmetry_relative_error"]
            <= float(tolerances["B_phase_normal_symmetry_relative_error_max"])
            and controls["B_phase_normal_action_psd_quadratic_form"]
            >= float(tolerances["B_phase_normal_psd_min"])
            and controls["support_projection_exterior_max_abs"]
            <= float(tolerances["exterior_max_abs"])
        )
        p_acceptance = config["known_b_probe_control"]["acceptance"]
        p_gates = (
            p_control["authoritative_raw_relative_l2"]
            <= float(p_acceptance["authoritative_raw_relative_l2_max"])
            and p_control["detector_relative_residual_curve"][-1]
            <= float(p_acceptance["detector_relative_residual_max"])
            and (
                not bool(p_acceptance["require_byte_exact"])
                or p_control["authoritative_raw_exact"]
            )
        )
        b_acceptance = config["known_probe_B_control"]["acceptance"]
        b_gates = (
            b_control["detector_relative_residual_curve"][-1]
            <= float(b_acceptance["detector_relative_residual_max"])
            and b_control["loss_curve"][-1] / b_control["loss_curve"][0]
            <= float(b_acceptance["final_to_initial_loss_ratio_max"])
            and (
                not bool(b_acceptance["require_loss_nonincreasing"])
                or _loss_nonincreasing(b_control["loss_curve"])
            )
        )
        all_blind_finite = all(
            np.all(np.isfinite(result[key]))
            for result in blind_branches.values()
            for key in ("P_B_rec_raw", "B_rec_raw", "prediction_final", "loss_curve")
        )
        representative_amplitude_residual = detector_amplitude_relative_residual(
            representative["prediction_final"], source.I_stack
        )
        maximum_branch_detector_residual = max(
            summary["final_detector_relative_residual"]
            for summary in blind_summaries.values()
        )
        blind_metrics = {
            "representative_branch": representative_name,
            "checkpoint_selection_rule": "fixed_final_outer_sweep",
            "representative_initial_loss": float(representative["loss_curve"][0]),
            "representative_final_loss": float(representative["loss_curve"][-1]),
            "representative_detector_relative_residual": float(
                representative["detector_relative_residual_curve"][-1]
            ),
            "representative_detector_amplitude_relative_residual": (
                representative_amplitude_residual
            ),
            "representative_probe_aligned_relative_l2_simulation_only": float(
                representative_evaluation["probe_aligned_relative_l2"]
            ),
            "representative_B_aligned_active_relative_l2_simulation_only": float(
                representative_evaluation["B_aligned_active_relative_l2"]
            ),
            "representative_exit_wave_product_relative_l2_simulation_only": float(
                representative_evaluation["exit_wave_product_relative_l2"]
            ),
            "maximum_pairwise_prediction_relative_l2": float(
                stability["maximum_pairwise_prediction_relative_l2"]
            ),
            "maximum_branch_detector_relative_residual": float(
                maximum_branch_detector_residual
            ),
            "maximum_pairwise_probe_aligned_relative_l2": float(
                stability["maximum_pairwise_probe_aligned_relative_l2"]
            ),
            "all_loss_nonincreasing": all(
                _loss_nonincreasing(result["loss_curve"])
                for result in blind_branches.values()
            ),
            "all_finite": all_blind_finite,
            "all_fixed_final_selection": all(
                result["checkpoint_selection_rule"] == "fixed_final_outer_sweep"
                for result in blind_branches.values()
            ),
            "branches": blind_summaries,
        }
        gate_metrics = {
            "source_operator_gates_passed": bool(source_operator_gates),
            "known_b_probe_control_gates_passed": bool(p_gates),
            "known_probe_B_control_gates_passed": bool(b_gates),
            "blind_primary": blind_metrics,
        }
        if config["execution"]["scientific_gates_enabled"]:
            status = determine_status(gate_metrics, config["formal_thresholds"])
        else:
            status = {
                "scientific_status": "Development",
                "reason": "scientific_gates_not_enabled_for_preflight",
            }
        component_gates_pass = (
            blind_metrics["representative_probe_aligned_relative_l2_simulation_only"]
            <= float(config["formal_thresholds"]["blind_probe_aligned_relative_l2_max"])
            and blind_metrics[
                "representative_B_aligned_active_relative_l2_simulation_only"
            ]
            <= float(
                config["formal_thresholds"]["blind_B_aligned_active_relative_l2_max"]
            )
            and blind_metrics[
                "representative_exit_wave_product_relative_l2_simulation_only"
            ]
            <= float(
                config["formal_thresholds"]["blind_exit_wave_product_relative_l2_max"]
            )
        )
        measurement_fit_gate = blind_metrics[
            "maximum_branch_detector_relative_residual"
        ] <= float(config["formal_thresholds"]["blind_detector_relative_residual_max"])
        structural_resistance = bool(
            source_operator_gates
            and p_gates
            and b_gates
            and measurement_fit_gate
            and not component_gates_pass
        )
        runtime_seconds = time.perf_counter() - start
        metrics = {
            "experiment_id": "exp043",
            "execution_stage": config["execution"]["stage"],
            "scientific_status": status["scientific_status"],
            "status_reason": status["reason"],
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "source_replay": source.replay,
            "operator_controls": controls,
            "source_operator_gates_passed": bool(source_operator_gates),
            "known_b_probe_control": {
                "initial_loss": float(p_control["loss_curve"][0]),
                "final_loss": float(p_control["loss_curve"][-1]),
                "final_detector_relative_residual": float(
                    p_control["detector_relative_residual_curve"][-1]
                ),
                "final_gradient_l2_norm": float(
                    p_control["gradient_l2_norm_curve"][-1]
                ),
                "authoritative_raw_relative_l2": float(
                    p_control["authoritative_raw_relative_l2"]
                ),
                "authoritative_raw_exact": bool(p_control["authoritative_raw_exact"]),
                "operator_action_budget_units": int(
                    p_control["operator_action_budget_units"]
                ),
                "runtime_seconds": p_control_runtime,
                "simulation_evaluation_only": {
                    "raw_probe_relative_l2": p_evaluation["raw_relative_l2"],
                    "aligned_probe_relative_l2": p_evaluation["aligned_relative_l2"],
                },
                "gates_passed": bool(p_gates),
            },
            "known_probe_B_control": {
                "initial_loss": float(b_control["loss_curve"][0]),
                "final_loss": float(b_control["loss_curve"][-1]),
                "final_detector_relative_residual": float(
                    b_control["detector_relative_residual_curve"][-1]
                ),
                "final_detector_amplitude_relative_residual": float(
                    b_control["detector_amplitude_relative_residual_curve"][-1]
                ),
                "B_active_relative_l2_simulation_evaluation_only": float(
                    b_control["B_active_relative_l2_simulation_evaluation_only"]
                ),
                "final_to_initial_loss_ratio": float(
                    b_control["loss_curve"][-1] / b_control["loss_curve"][0]
                ),
                "final_gradient_l2_norm": float(
                    b_control["gradient_l2_norm_curve"][-1]
                ),
                "iterations_completed": int(b_control["iterations_completed"]),
                "stopping_reason": str(b_control["stopping_reason"]),
                "total_backtracking_steps": int(
                    np.sum(b_control["backtracking_count_curve"])
                ),
                "operator_action_budget_units": int(
                    b_control["operator_action_budget_units"]
                ),
                "runtime_seconds": b_control_runtime,
                "loss_nonincreasing": _loss_nonincreasing(b_control["loss_curve"]),
                "gates_passed": bool(b_gates),
            },
            "blind_primary": blind_metrics,
            "blind_stability": stability,
            "structural_resistance_observed": structural_resistance,
            "alternative_route_attempted": False,
            "alternative_route_reason": (
                "not_authorized_by_preflight_evidence"
                if not structural_resistance
                else (
                    "deferred_component_nonidentifiability_to_exp044_measurement_design"
                )
            ),
            "truth_use": {
                "primary_initialization": False,
                "optimizer": False,
                "checkpoint_selection": False,
                "stopping": False,
                "method_selection": False,
                "simulation_evaluation_loaded_after_raw_results": True,
            },
            "runtime_seconds": runtime_seconds,
        }
        metadata = {
            "experiment_id": "exp043",
            "created_at_utc": created_at_utc(),
            "execution_stage": config["execution"]["stage"],
            "git_commit": get_git_commit(PROJECT_ROOT),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "source_experiment": "exp042",
            "source_run": str(source.source_run),
            "operator_branch": source.provenance["operator_branch"],
            "selected_method": "alternating_block_spectrally_damped_gn_cg",
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "truth_used_by_primary": False,
            "project_hdf5_schema_changed": False,
        }

        reconstruction_branches = {
            name: _blind_hdf5(blind_branches[name], blind_evaluations[name])
            for name in blind_branches
        }
        representative_payload = reconstruction_branches[representative_name]
        reconstruction = {
            "source_provenance": source.provenance,
            "design": {
                "primary_method": config["blind"]["algorithm"],
                "changed_from_exp042": "sample_B_is_variable_and_alternated_with_probe",
                "loss": config["loss"],
                "checkpoint_selection_rule": config["blind"][
                    "checkpoint_selection_rule"
                ],
                "representative_branch_rule": "first_preregistered_initialization",
                "truth_use": metrics["truth_use"],
            },
            "operator_controls": controls,
            "known_b_probe_control": _known_b_hdf5(p_control, p_evaluation),
            "known_probe_B_control": _known_probe_b_hdf5(b_control),
            "blind_joint": {
                "branches": reconstruction_branches,
                "representative_branch": representative_name,
                "representative": representative_payload,
                "stability": stability,
            },
            "gauge": {
                **config["gauge"],
                "diagnostics": {
                    "algebraic_reciprocal_product_gauge_relative_error": controls[
                        "algebraic_reciprocal_product_gauge_relative_error"
                    ],
                    "model_representable_reciprocal_scale_prediction_change": controls[
                        "model_representable_reciprocal_scale_prediction_change"
                    ],
                },
                "primary_complex_factor": np.complex128(1.0 + 0.0j),
            },
            "checkpoints": {
                "selection_rule": "fixed_final_outer_sweep",
                "representative_P_B_history": representative["P_B_checkpoint_history"],
                "representative_B_history": representative["B_checkpoint_history"],
                "representative_loss_curve": representative["loss_curve"],
            },
        }
        source_config = source.source_config
        instrument = {
            "wavelength_m": source_config["optics"]["wavelength_m"],
            "external_medium_index": source_config["optics"]["external_medium_index"],
            "z_BC_m": source_config["optics"]["z_BC_m"],
            "probe_grid": source_config["probe_grid"],
            "detector": source_config["detector"],
            "scan": source_config["scan"],
        }
        sample = {
            "sample_b": {
                **source_config["sample_b"],
                "support_mask": source.support_mask,
                "unknown_in_primary": True,
                "transparent_exterior_fixed": True,
            }
        }
        truth = {
            "P_B_true": source.P_B_true,
            "B_true": source.B_true,
            "identity": "simulation truth under selected exp040 scalar working model",
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "use": "operator controls and postfreeze simulation evaluation only",
        }
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_ptycho_hdf5(
            run_dir / "outputs" / output["hdf5_filename"],
            I_stack=source.I_stack,
            scan_positions=source.scan_positions,
            instrument=instrument,
            sample=sample,
            truth=truth,
            reconstruction=_hdf5_ready(reconstruction),
            config_yaml=config_to_yaml(config),
            metadata=metadata,
            metrics=_hdf5_ready(metrics),
        )
        _save_figures(
            run_dir,
            list(output["figure_filenames"]),
            source,
            p_control,
            b_control,
            blind_branches,
            representative_name,
            stability,
        )
        artifact_audit = _validate_artifacts(run_dir, config, metrics)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "artifacts_validated": True,
                "completed_at_utc": created_at_utc(),
                "scientific_status": metrics["scientific_status"],
                "status_reason": metrics["status_reason"],
                "runtime_seconds": runtime_seconds,
                "config_sha256": sha256_file(run_dir / "config.yaml"),
                "metadata_sha256": sha256_file(run_dir / "metadata.json"),
                "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                "hdf5_sha256": sha256_file(
                    run_dir / "outputs" / output["hdf5_filename"]
                ),
                **artifact_audit,
            },
        )
    except Exception as error:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return run_dir


def main() -> None:
    args = _parse_args()
    run_dir = run(args.config.resolve())
    print(run_dir)


if __name__ == "__main__":
    main()
