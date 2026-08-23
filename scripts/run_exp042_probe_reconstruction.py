from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.recon.exp042 import (  # noqa: E402
    MATCHED_PIXEL_AVERAGE_READOUT,
    Q1_POINT_MISMATCH_READOUT,
    MatchedKnownBProbeOperator,
    build_matched_development_case,
    detector_quadrature_ablation_consistency_metrics,
    lanczos_dimension_convergence_diagnostic,
    make_detector_readout_operator,
    make_deterministic_complex_probe_initialization,
    operator_consistency_metrics,
    reconstruct_known_b_probe,
    simulation_evaluation_only,
    truth_free_detector_quadrature_ablation_diagnostic,
    truth_free_initialization_ablation_diagnostic,
    truth_free_initialization_stability_diagnostic,
    truth_free_local_spectral_diagnostic,
    truth_free_paired_continuation_diagnostic,
    truth_free_pairwise_conditioning_diagnostic,
    validate_exp042_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exp042 matched known-B probe baseline."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _save_figures(
    run_dir: Path,
    case: dict[str, Any],
    reconstruction: dict[str, Any],
    evaluation: dict[str, Any],
    control_reconstruction: dict[str, Any],
    control_evaluation: dict[str, Any],
    stability_diagnostic: dict[str, Any],
    conditioning_diagnostic: dict[str, Any],
    primary_continuation: dict[str, Any],
    control_continuation: dict[str, Any],
    continuation_diagnostic: dict[str, Any],
    filenames: list[str],
) -> list[Path]:
    if len(filenames) != 6:
        raise ValueError("exp042 requires exactly six registered figures.")
    figure_paths = [run_dir / "figures" / name for name in filenames]

    loss = np.asarray(reconstruction["loss_curve"], dtype=np.float64)
    residual = np.asarray(
        reconstruction["detector_relative_residual_curve"], dtype=np.float64
    )
    probe_error = np.asarray(
        evaluation["probe_global_phase_aligned_relative_l2_curve"],
        dtype=np.float64,
    )
    control_loss = np.asarray(
        control_reconstruction["loss_curve"], dtype=np.float64
    )
    control_residual = np.asarray(
        control_reconstruction["detector_relative_residual_curve"],
        dtype=np.float64,
    )
    control_probe_error = np.asarray(
        control_evaluation["probe_global_phase_aligned_relative_l2_curve"],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].semilogy(
        np.maximum(loss, np.finfo(float).tiny), label="homogeneous init"
    )
    axes[0].semilogy(
        np.maximum(control_loss, np.finfo(float).tiny),
        label="5% perturbed init",
    )
    axes[0].set(title="Pixel-intensity loss", xlabel="Iteration")
    axes[1].semilogy(
        np.maximum(residual, np.finfo(float).tiny), label="homogeneous init"
    )
    axes[1].semilogy(
        np.maximum(control_residual, np.finfo(float).tiny),
        label="5% perturbed init",
    )
    axes[1].set(title="Detector relative residual", xlabel="Iteration")
    axes[2].semilogy(
        np.maximum(probe_error, np.finfo(float).tiny),
        label="homogeneous init",
    )
    axes[2].semilogy(
        np.maximum(control_probe_error, np.finfo(float).tiny),
        label="5% perturbed init",
    )
    axes[2].set(
        title="Probe error (simulation evaluation only)", xlabel="Iteration"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_paths[0], dpi=160)
    plt.close(fig)

    truth = np.asarray(case["P_B_true"], dtype=np.complex128)
    initial = np.asarray(reconstruction["P_B_init"], dtype=np.complex128)
    aligned = np.asarray(
        evaluation["P_B_rec_global_phase_aligned"], dtype=np.complex128
    )
    fields = [truth, initial, aligned]
    labels = [
        "P_B truth",
        "Initialization",
        "Reconstruction aligned\n(simulation evaluation only)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    amplitude_max = max(float(np.max(np.abs(field))) for field in fields)
    for column, (field, label) in enumerate(zip(fields, labels, strict=True)):
        amplitude_image = axes[0, column].imshow(
            np.abs(field), cmap="viridis", vmin=0.0, vmax=amplitude_max
        )
        axes[0, column].set_title(label)
        axes[1, column].imshow(
            np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi
        )
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Phase [rad]")
    fig.colorbar(amplitude_image, ax=axes[0, :].tolist(), shrink=0.75)
    fig.suptitle("exp042 development probe comparison")
    fig.subplots_adjust(left=0.04, right=0.94, bottom=0.04, top=0.9)
    fig.savefig(figure_paths[1], dpi=160)
    plt.close(fig)

    measured = np.asarray(case["I_stack"], dtype=np.float64)
    predicted = np.asarray(
        reconstruction["prediction_final"], dtype=np.float64
    )
    scan_index = len(measured) // 2
    residual_frame = predicted[scan_index] - measured[scan_index]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    intensity_max = max(
        float(np.max(measured[scan_index])),
        float(np.max(predicted[scan_index])),
    )
    axes[0].imshow(
        measured[scan_index], cmap="magma", vmin=0.0, vmax=intensity_max
    )
    axes[0].set_title("Measured matched data")
    axes[1].imshow(
        predicted[scan_index], cmap="magma", vmin=0.0, vmax=intensity_max
    )
    axes[1].set_title("Final prediction")
    residual_limit = max(float(np.max(np.abs(residual_frame))), 1.0e-15)
    axes[2].imshow(
        residual_frame,
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[2].set_title("Prediction - measured")
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(f"Detector scan {scan_index}")
    fig.tight_layout()
    fig.savefig(figure_paths[2], dpi=160)
    plt.close(fig)

    primary_init = np.asarray(
        reconstruction["P_B_init"], dtype=np.complex128
    )
    control_init = np.asarray(
        control_reconstruction["P_B_init"], dtype=np.complex128
    )
    primary_rec = np.asarray(
        reconstruction["P_B_rec"], dtype=np.complex128
    )
    control_rec_aligned = np.asarray(
        stability_diagnostic[
            "P_B_control_rec_global_phase_aligned_to_primary_rec"
        ],
        dtype=np.complex128,
    )
    stability_fields = [
        primary_init,
        control_init,
        primary_rec,
        control_rec_aligned,
    ]
    stability_labels = [
        "Primary init",
        "Perturbed init",
        "Primary reconstruction",
        "Control aligned to primary",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.0))
    stability_amplitude_max = max(
        float(np.max(np.abs(field))) for field in stability_fields
    )
    for column, (field, label) in enumerate(
        zip(stability_fields, stability_labels, strict=True)
    ):
        amplitude_image = axes[0, column].imshow(
            np.abs(field),
            cmap="viridis",
            vmin=0.0,
            vmax=stability_amplitude_max,
        )
        axes[0, column].set_title(label)
        axes[1, column].imshow(
            np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi
        )
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Phase [rad]")
    fig.colorbar(amplitude_image, ax=axes[0, :].tolist(), shrink=0.75)
    fig.suptitle("Truth-free initialization stability diagnostic")
    fig.subplots_adjust(left=0.04, right=0.94, bottom=0.04, top=0.9)
    fig.savefig(figure_paths[3], dpi=160)
    plt.close(fig)

    random_gains = np.asarray(
        conditioning_diagnostic["random_direction_jacobian_rms_gains"],
        dtype=np.float64,
    )
    named_labels = ["Native phase", "Pairwise", "Gradient"]
    named_gains = np.asarray(
        [
            conditioning_diagnostic[
                "native_global_phase_rotation_jacobian_rms_gain"
            ],
            conditioning_diagnostic["pairwise_jacobian_rms_gain"],
            conditioning_diagnostic[
                "primary_gradient_jacobian_rms_gain"
            ],
        ],
        dtype=np.float64,
    )
    labels = [
        *named_labels,
        *(f"R{index + 1}" for index in range(len(random_gains))),
    ]
    gains = np.concatenate([named_gains, random_gains])
    colors = [
        "tab:gray",
        "tab:red",
        "tab:blue",
        *(["tab:green"] * len(random_gains)),
    ]
    fig, axis = plt.subplots(figsize=(10, 4.2))
    axis.bar(
        np.arange(len(gains)),
        np.maximum(gains, np.finfo(float).tiny),
        color=colors,
    )
    axis.axhline(
        float(np.median(random_gains)),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="random median",
    )
    axis.set_yscale("log")
    axis.set_xticks(np.arange(len(gains)), labels)
    axis.set_ylabel("RMS intensity-Jacobian gain for unit-L2 direction")
    axis.set_title("Truth-free local Jacobian conditioning controls")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_paths[4], dpi=160)
    plt.close(fig)

    primary_continuation_loss = np.asarray(
        primary_continuation["loss_curve"], dtype=np.float64
    )
    control_continuation_loss = np.asarray(
        control_continuation["loss_curve"], dtype=np.float64
    )
    primary_continuation_residual = np.asarray(
        primary_continuation["detector_relative_residual_curve"],
        dtype=np.float64,
    )
    control_continuation_residual = np.asarray(
        control_continuation["detector_relative_residual_curve"],
        dtype=np.float64,
    )
    pairwise_curve = np.asarray(
        continuation_diagnostic[
            "pairwise_global_phase_aligned_relative_l2_curve"
        ],
        dtype=np.float64,
    )
    projection_curve = np.asarray(
        continuation_diagnostic[
            "pairwise_start_direction_projection_fraction_curve"
        ],
        dtype=np.float64,
    )
    checkpoints = np.asarray(
        continuation_diagnostic["checkpoint_iterations"], dtype=np.int64
    )
    prediction_curve = np.asarray(
        continuation_diagnostic["checkpoint_prediction_relative_l2_curve"],
        dtype=np.float64,
    )
    gain_curve = np.asarray(
        continuation_diagnostic[
            "checkpoint_pairwise_jacobian_rms_gain_curve"
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes[0, 0].semilogy(
        np.maximum(primary_continuation_loss, np.finfo(float).tiny),
        label="primary continuation",
    )
    axes[0, 0].semilogy(
        np.maximum(control_continuation_loss, np.finfo(float).tiny),
        label="control continuation",
    )
    axes[0, 0].set(title="Continuation loss", xlabel="Additional iteration")
    axes[0, 1].semilogy(
        np.maximum(primary_continuation_residual, np.finfo(float).tiny),
        label="primary continuation",
    )
    axes[0, 1].semilogy(
        np.maximum(control_continuation_residual, np.finfo(float).tiny),
        label="control continuation",
    )
    axes[0, 1].set(
        title="Continuation detector residual", xlabel="Additional iteration"
    )
    axes[1, 0].plot(pairwise_curve, color="tab:red", label="aligned probe L2")
    axes[1, 0].set(
        title="Truth-free pairwise field evolution",
        xlabel="Additional iteration",
        ylabel="Aligned relative L2",
    )
    projection_axis = axes[1, 0].twinx()
    projection_axis.plot(
        projection_curve,
        color="tab:purple",
        linestyle="--",
        label="start-direction projection",
    )
    projection_axis.set_ylabel("Projection / initial projection")
    axes[1, 1].semilogy(
        checkpoints,
        np.maximum(prediction_curve, np.finfo(float).tiny),
        color="tab:orange",
        marker="o",
        label="prediction relative L2",
    )
    axes[1, 1].set(
        title="Checkpoint measurement sensitivity",
        xlabel="Additional iteration",
        ylabel="Prediction relative L2",
    )
    gain_axis = axes[1, 1].twinx()
    gain_axis.semilogy(
        checkpoints,
        np.maximum(gain_curve, np.finfo(float).tiny),
        color="tab:green",
        marker="s",
        linestyle="--",
        label="pairwise Jacobian gain",
    )
    gain_axis.set_ylabel("Pairwise Jacobian RMS gain")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")
    projection_axis.legend(fontsize=7, loc="upper right")
    gain_axis.legend(fontsize=7, loc="lower right")
    fig.suptitle("Truth-free equal-budget paired continuation")
    fig.tight_layout()
    fig.savefig(figure_paths[5], dpi=160)
    plt.close(fig)
    return figure_paths


def _metadata(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    provenance = config["provenance"]
    return {
        "experiment_id": "exp042",
        "created_at_utc": created_at_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_config": str(config_path),
        "development_status": (
            "Development baseline / No scientific pass-fail conclusion"
        ),
        "operator_branch": provenance["source_branch"],
        "development_data_origin": provenance["development_data_origin"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "known_sample_b": True,
        "probe_only_reconstruction": True,
        "initialization_stability_control": True,
        "pairwise_conditioning_diagnostic": True,
        "paired_continuation_control": True,
        "truth_used_by_optimizer": False,
        "scientific_claim_boundary": (
            "matched self-consistent inverse behavior under the frozen "
            "exp040 scalar working model only"
        ),
    }


def _reconstruction_hdf5_payload(
    reconstruction: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "P_B_init": reconstruction["P_B_init"],
        "P_B_rec": reconstruction["P_B_rec"],
        "loss_curve": reconstruction["loss_curve"],
        "detector_relative_residual_curve": reconstruction[
            "detector_relative_residual_curve"
        ],
        "gradient_l2_norm_curve": reconstruction["gradient_l2_norm_curve"],
        "accepted_step_curve": reconstruction["accepted_step_curve"],
        "proposed_step_curve": reconstruction["proposed_step_curve"],
        "gauss_newton_directional_curvature_curve": reconstruction[
            "gauss_newton_directional_curvature_curve"
        ],
        "step_scale_fallback_curve": reconstruction[
            "step_scale_fallback_curve"
        ],
        "backtracking_count_curve": reconstruction[
            "backtracking_count_curve"
        ],
        "step_scale_fallback_count": reconstruction[
            "step_scale_fallback_count"
        ],
        "total_backtracking_steps": reconstruction[
            "total_backtracking_steps"
        ],
        "iterations_completed": reconstruction["iterations_completed"],
        "stopping_reason": reconstruction["stopping_reason"],
        "algorithm": reconstruction["algorithm"],
        "truth_used_by_optimizer": False,
    }
    if evaluation is not None:
        payload["simulation_evaluation_only"] = {
            "P_B_rec_global_phase_aligned": evaluation[
                "P_B_rec_global_phase_aligned"
            ],
            "global_phase_alignment_factor": evaluation[
                "global_phase_alignment_factor"
            ],
            "probe_raw_relative_l2_curve": evaluation[
                "probe_raw_relative_l2_curve"
            ],
            "probe_global_phase_aligned_relative_l2_curve": evaluation[
                "probe_global_phase_aligned_relative_l2_curve"
            ],
        }
    return payload


def _hdf5_payload(
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    case: dict[str, Any],
    reconstruction: dict[str, Any],
    evaluation: dict[str, Any],
    control_reconstruction: dict[str, Any],
    control_evaluation: dict[str, Any],
    stability_diagnostic: dict[str, Any],
    conditioning_diagnostic: dict[str, Any],
    primary_continuation: dict[str, Any],
    control_continuation: dict[str, Any],
    continuation_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    operator = case["operator"]
    if not isinstance(operator, MatchedKnownBProbeOperator):
        raise TypeError("case operator has the wrong type.")
    return {
        "I_stack": case["I_stack"],
        "scan_positions": case["scan_positions"],
        "instrument": {
            "wavelength_m": config["optics"]["wavelength_m"],
            "internal_reference_index": config["optics"][
                "internal_reference_index"
            ],
            "external_medium_index": config["optics"][
                "external_medium_index"
            ],
            "z_AB_m": config["optics"]["z_AB_m"],
            "z_BC_m": config["optics"]["z_BC_m"],
            "probe_grid": {
                "plane": "B",
                "axis_order": ["y", "x"],
                "native_shape": operator.native_shape,
                "open_shape": operator.open_shape,
                "node_dx_m": operator.node_dx_m,
            },
            "detector": {
                "model": config["detector"]["model"],
                "quadrature_factor": operator.quadrature_factor,
                "pixel_size_m": config["detector"]["pixel_size_m"],
                "native_roi_shape": operator.detector_roi_shape,
                "quadrature_weights": np.full(
                    operator.quadrature_factor**2,
                    1.0 / operator.quadrature_factor**2,
                ),
            },
        },
        "sample": {
            "sample_a": dict(config["sample_a"]),
            "sample_b": dict(config["sample_b"]),
        },
        "truth": {
            "identity": (
                "simulation truth under the selected exp040 scalar "
                "working model"
            ),
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "P_B_true": case["P_B_true"],
            "B_true": case["B_true"],
            "B_support_true": case["B_support_true"],
            "U_A_exit_true": case["U_A_exit_true"],
            "z_m": case["z_m"],
            "slice_widths_m": case["slice_widths_m"],
            "D_z_m": case["D_z_m"],
        },
        "reconstruction": {
            **_reconstruction_hdf5_payload(reconstruction, evaluation),
            "initialization_stability_control": {
                **_reconstruction_hdf5_payload(
                    control_reconstruction, control_evaluation
                ),
                "truth_free_pairwise_diagnostic": stability_diagnostic,
                "truth_free_pairwise_conditioning": (
                    conditioning_diagnostic
                ),
            },
            "paired_continuation_control": {
                "role": config["reconstruction"][
                    "paired_continuation_control"
                ]["role"],
                "start_field_source": config["reconstruction"][
                    "paired_continuation_control"
                ]["start_fields"],
                "additional_iterations": config["reconstruction"][
                    "paired_continuation_control"
                ]["additional_iterations"],
                "truth_used_by_continuation": False,
                "primary": _reconstruction_hdf5_payload(
                    primary_continuation
                ),
                "control": _reconstruction_hdf5_payload(
                    control_continuation
                ),
                "truth_free_pairwise_diagnostic": continuation_diagnostic,
            },
        },
        "config_yaml": config_to_yaml(config),
        "metadata": metadata,
        "metrics": metrics,
    }


def _all_numeric_hdf5_finite(group: h5py.Group) -> bool:
    for value in group.values():
        if isinstance(value, h5py.Group):
            if not _all_numeric_hdf5_finite(value):
                return False
        elif value.dtype.kind in "biufc" and not np.all(np.isfinite(value[...])):
            return False
    return True


def _validate_artifacts(
    run_dir: Path, config: dict[str, Any], expected_shapes: dict[str, tuple[int, ...]]
) -> None:
    required = {
        run_dir / "config.yaml",
        run_dir / "metadata.json",
        run_dir / "metrics.json",
        run_dir / "run_state.json",
        run_dir / "outputs" / config["output"]["hdf5_filename"],
        *(
            run_dir / "figures" / name
            for name in config["output"]["figure_filenames"]
        ),
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"Missing exp042 artifacts: {missing}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["development_status"] != (
        "Development baseline / No scientific pass-fail conclusion"
    ):
        raise RuntimeError("exp042 metrics contain an invalid status.")
    if "initialization_stability_control" not in metrics:
        raise RuntimeError("Initialization stability metrics are missing.")
    if (
        "truth_free_pairwise_conditioning"
        not in metrics["initialization_stability_control"]
    ):
        raise RuntimeError("Pairwise conditioning metrics are missing.")
    if "paired_continuation_control" not in metrics:
        raise RuntimeError("Paired continuation metrics are missing.")
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
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
        if set(h5["entry"]) != expected_entry:
            raise RuntimeError("Unexpected exp042 /entry layout.")
        for path, shape in expected_shapes.items():
            if path not in h5 or h5[path].shape != shape:
                raise RuntimeError(f"Unexpected HDF5 shape at {path}.")
        if "entry/reconstruction/P_B_rec" not in h5:
            raise RuntimeError("Raw P_B_rec is missing.")
        if (
            "entry/reconstruction/simulation_evaluation_only/"
            "P_B_rec_global_phase_aligned"
        ) not in h5:
            raise RuntimeError("Simulation-only aligned probe is missing.")
        if (
            "entry/reconstruction/initialization_stability_control/P_B_rec"
            not in h5
        ):
            raise RuntimeError("Raw stability-control probe is missing.")
        if (
            "entry/reconstruction/initialization_stability_control/"
            "truth_free_pairwise_diagnostic/"
            "P_B_control_rec_global_phase_aligned_to_primary_rec"
        ) not in h5:
            raise RuntimeError("Pairwise stability diagnostic is missing.")
        if (
            "entry/reconstruction/initialization_stability_control/"
            "truth_free_pairwise_conditioning/"
            "P_B_pairwise_direction_unit_l2"
        ) not in h5:
            raise RuntimeError("Pairwise conditioning diagnostic is missing.")
        if (
            "entry/reconstruction/paired_continuation_control/primary/P_B_rec"
            not in h5
            or "entry/reconstruction/paired_continuation_control/control/P_B_rec"
            not in h5
        ):
            raise RuntimeError("Raw paired continuation probes are missing.")
        if (
            "entry/reconstruction/paired_continuation_control/"
            "truth_free_pairwise_diagnostic/"
            "P_B_pairwise_start_direction_unit_l2"
        ) not in h5:
            raise RuntimeError("Paired continuation diagnostic is missing.")
        if not _all_numeric_hdf5_finite(h5["entry"]):
            raise RuntimeError("HDF5 contains non-finite numeric data.")
    for name in config["output"]["figure_filenames"]:
        image = plt.imread(run_dir / "figures" / name)
        if image.ndim not in {2, 3} or not np.all(np.isfinite(image)):
            raise RuntimeError(f"Unreadable or non-finite figure: {name}")


def _measurement_metrics(reconstruction: dict[str, Any]) -> dict[str, Any]:
    loss_curve = np.asarray(reconstruction["loss_curve"], dtype=np.float64)
    residual_curve = np.asarray(
        reconstruction["detector_relative_residual_curve"], dtype=np.float64
    )
    gradient_curve = np.asarray(
        reconstruction["gradient_l2_norm_curve"], dtype=np.float64
    )
    curvature_curve = np.asarray(
        reconstruction["gauss_newton_directional_curvature_curve"],
        dtype=np.float64,
    )
    proposed_step_curve = np.asarray(
        reconstruction["proposed_step_curve"], dtype=np.float64
    )
    accepted_step_curve = np.asarray(
        reconstruction["accepted_step_curve"][1:], dtype=np.float64
    )
    return {
        "initial_loss": float(loss_curve[0]),
        "final_loss": float(loss_curve[-1]),
        "initial_detector_relative_residual": float(residual_curve[0]),
        "final_detector_relative_residual": float(residual_curve[-1]),
        "initial_gradient_l2_norm": float(gradient_curve[0]),
        "final_gradient_l2_norm": float(gradient_curve[-1]),
        "loss_nonincreasing": bool(np.all(np.diff(loss_curve) <= 0.0)),
        "iterations_completed": reconstruction["iterations_completed"],
        "stopping_reason": reconstruction["stopping_reason"],
        "optimizer_scaling": (
            "gauss_newton_directional_curvature_recomputed_each_iteration"
        ),
        "initial_gauss_newton_directional_curvature": float(
            curvature_curve[0]
        ),
        "final_gauss_newton_directional_curvature": float(
            curvature_curve[-1]
        ),
        "minimum_proposed_step": float(np.min(proposed_step_curve)),
        "maximum_proposed_step": float(np.max(proposed_step_curve)),
        "minimum_accepted_step": float(np.min(accepted_step_curve)),
        "maximum_accepted_step": float(np.max(accepted_step_curve)),
        "step_scale_fallback_count": reconstruction[
            "step_scale_fallback_count"
        ],
        "total_backtracking_steps": reconstruction[
            "total_backtracking_steps"
        ],
    }


def _initialization_ablation_metadata(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    provenance = config["provenance"]
    run_role = str(config["execution"]["mode"])
    family = str(
        config["verification"]["initialization_ablation"][
            "initialization_family"
        ]
    )
    return {
        "experiment_id": "exp042",
        "created_at_utc": created_at_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_config": str(config_path),
        "run_role": run_role,
        "initialization_family": family,
        "development_status": (
            "Development baseline / No scientific pass-fail conclusion"
        ),
        "operator_branch": provenance["source_branch"],
        "development_data_origin": provenance["development_data_origin"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "known_sample_b": True,
        "probe_only_reconstruction": True,
        "reconstruction_performed_in_this_run": True,
        "paired_continuation_performed_in_this_run": False,
        "local_spectral_diagnostic_performed_in_this_run": False,
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "simulation_truth_evaluation_only": True,
        "scientific_claim_boundary": (
            "initialization sensitivity of one matched noiseless development "
            "case under the frozen exp040 scalar working model only; the "
            "registered branch axis is the only changed factor"
        ),
    }


def _initialization_ablation_design_payload(
    settings: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": settings["role"],
        "initialization_family": settings["initialization_family"],
        "reference_branch": settings["reference_branch"],
        "equal_iteration_budget": settings["equal_iteration_budget"],
        "pairwise_alignment": settings["pairwise_alignment"],
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "truth_used_by_branch_selection": False,
        "truth_used_by_stopping": False,
        "scientific_thresholds_preregistered": False,
    }
    if settings["initialization_family"] == (
        "homogeneous_plus_zero_mean_complex_perturbations"
    ):
        payload["perturbation_relative_l2_to_homogeneous"] = settings[
            "perturbation_relative_l2_to_homogeneous"
        ]
    else:
        payload.update(
            {
                "sweep_axis": settings["sweep_axis"],
                "fixed_perturbation_seed": settings[
                    "fixed_perturbation_seed"
                ],
                "magnitude_levels": settings["magnitude_levels"],
                "same_perturbation_direction": True,
                "same_direction_max_unit_l2_error_tolerance": settings[
                    "same_direction_max_unit_l2_error_tolerance"
                ],
                "checkpoint_interval": settings["checkpoint_interval"],
                "figure_panels": settings["figure_panels"],
            }
        )
    return payload


def _initialization_ablation_hdf5_payload(
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    case: dict[str, Any],
    reconstructions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    operator = case["operator"]
    if not isinstance(operator, MatchedKnownBProbeOperator):
        raise TypeError("case operator has the wrong type.")
    settings = config["verification"]["initialization_ablation"]
    specifications = {
        str(branch["name"]): dict(branch) for branch in settings["branches"]
    }
    branches = {
        name: {
            "initialization_spec": specifications[name],
            **_reconstruction_hdf5_payload(
                reconstructions[name], evaluations[name]
            ),
        }
        for name in reconstructions
    }
    return {
        "I_stack": case["I_stack"],
        "scan_positions": case["scan_positions"],
        "instrument": {
            "wavelength_m": config["optics"]["wavelength_m"],
            "internal_reference_index": config["optics"][
                "internal_reference_index"
            ],
            "external_medium_index": config["optics"][
                "external_medium_index"
            ],
            "z_AB_m": config["optics"]["z_AB_m"],
            "z_BC_m": config["optics"]["z_BC_m"],
            "probe_grid": {
                "plane": "B",
                "axis_order": ["y", "x"],
                "native_shape": operator.native_shape,
                "open_shape": operator.open_shape,
                "node_dx_m": operator.node_dx_m,
            },
            "detector": {
                "model": config["detector"]["model"],
                "quadrature_factor": operator.quadrature_factor,
                "pixel_size_m": config["detector"]["pixel_size_m"],
                "native_roi_shape": operator.detector_roi_shape,
                "quadrature_weights": np.full(
                    operator.quadrature_factor**2,
                    1.0 / operator.quadrature_factor**2,
                ),
            },
        },
        "sample": {
            "sample_a": dict(config["sample_a"]),
            "sample_b": dict(config["sample_b"]),
        },
        "truth": {
            "identity": (
                "simulation truth under the selected exp040 scalar "
                "working model"
            ),
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "P_B_true": case["P_B_true"],
            "B_true": case["B_true"],
            "B_support_true": case["B_support_true"],
            "U_A_exit_true": case["U_A_exit_true"],
            "z_m": case["z_m"],
            "slice_widths_m": case["slice_widths_m"],
            "D_z_m": case["D_z_m"],
        },
        "reconstruction": {
            "initialization_ablation": {
                "design": _initialization_ablation_design_payload(settings),
                "branches": branches,
                "truth_free_pairwise_diagnostic": diagnostic,
            }
        },
        "config_yaml": config_to_yaml(config),
        "metadata": metadata,
        "metrics": metrics,
    }


def _save_initialization_ablation_figure(
    run_dir: Path,
    reconstructions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    filename: str,
) -> Path:
    figure_path = run_dir / "figures" / filename
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for name, reconstruction in reconstructions.items():
        axes[0, 0].semilogy(
            np.maximum(reconstruction["loss_curve"], np.finfo(float).tiny),
            label=name,
        )
        axes[0, 1].semilogy(
            np.maximum(
                reconstruction["detector_relative_residual_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
        evaluation = evaluations[name]
        axes[0, 2].semilogy(
            np.maximum(
                evaluation["probe_raw_relative_l2_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
        axes[1, 0].semilogy(
            np.maximum(
                evaluation["probe_global_phase_aligned_relative_l2_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
    axes[0, 0].set(title="Pixel-intensity loss", xlabel="Iteration")
    axes[0, 1].set(title="Detector relative residual", xlabel="Iteration")
    axes[0, 2].set(
        title="Raw probe error (simulation evaluation only)",
        xlabel="Iteration",
    )
    axes[1, 0].set(
        title="Phase-aligned probe error (simulation evaluation only)",
        xlabel="Iteration",
    )
    for axis in axes.ravel()[:4]:
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    branch_names = list(diagnostic["branch_names"])
    matrices = (
        np.asarray(
            diagnostic[
                "final_pairwise_global_phase_aligned_probe_relative_l2"
            ],
            dtype=np.float64,
        ),
        np.asarray(
            diagnostic["final_pairwise_prediction_relative_l2"],
            dtype=np.float64,
        ),
    )
    titles = (
        "Final phase-aligned probe distance",
        "Final detector-prediction distance",
    )
    for axis, matrix, title in zip(
        axes[1, 1:], matrices, titles, strict=True
    ):
        image = axis.imshow(matrix, origin="upper", cmap="viridis")
        axis.set_title(title)
        axis.set_xticks(range(len(branch_names)), branch_names, rotation=35)
        axis.set_yticks(range(len(branch_names)), branch_names)
        axis.tick_params(labelsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("exp042 known-B equal-budget initialization ablation")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def _save_initialization_magnitude_ablation_figure(
    run_dir: Path,
    reconstructions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    filename: str,
) -> Path:
    figure_path = run_dir / "figures" / filename
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
    for name, reconstruction in reconstructions.items():
        axes[0, 0].semilogy(
            np.maximum(reconstruction["loss_curve"], np.finfo(float).tiny),
            label=name,
        )
        axes[0, 1].semilogy(
            np.maximum(
                reconstruction["detector_relative_residual_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
        evaluation = evaluations[name]
        axes[0, 2].semilogy(
            np.maximum(
                evaluation["probe_raw_relative_l2_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
        axes[0, 3].semilogy(
            np.maximum(
                evaluation["probe_global_phase_aligned_relative_l2_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
    axes[0, 0].set(title="Pixel-intensity loss", xlabel="Iteration")
    axes[0, 1].set(title="Detector relative residual", xlabel="Iteration")
    axes[0, 2].set(
        title="Raw probe error (simulation evaluation only)",
        xlabel="Iteration",
    )
    axes[0, 3].set(
        title="Aligned probe error (simulation evaluation only)",
        xlabel="Iteration",
    )
    branch_names = list(diagnostic["branch_names"])
    reference_branch = str(diagnostic["reference_branch"])
    checkpoint_iterations = np.asarray(
        diagnostic["checkpoint_iterations"], dtype=np.int64
    )
    checkpoint_curves = (
        np.asarray(
            diagnostic[
                "checkpoint_raw_probe_relative_l2_to_reference_curve"
            ],
            dtype=np.float64,
        ),
        np.asarray(
            diagnostic[
                "checkpoint_global_phase_aligned_probe_relative_l2_to_reference_curve"
            ],
            dtype=np.float64,
        ),
        np.asarray(
            diagnostic[
                "checkpoint_prediction_relative_l2_to_reference_curve"
            ],
            dtype=np.float64,
        ),
    )
    checkpoint_titles = (
        "Raw probe distance to reference",
        "Aligned probe distance to reference",
        "Prediction distance to reference",
    )
    for axis, curve, title in zip(
        axes[1, :3], checkpoint_curves, checkpoint_titles, strict=True
    ):
        for branch_index, name in enumerate(branch_names):
            if name == reference_branch:
                continue
            axis.plot(
                checkpoint_iterations,
                curve[:, branch_index],
                marker="o",
                ms=2.5,
                label=name,
            )
        axis.set(title=title, xlabel="Iteration")
    final_matrix = np.asarray(
        diagnostic["final_pairwise_global_phase_aligned_probe_relative_l2"],
        dtype=np.float64,
    )
    image = axes[1, 3].imshow(final_matrix, origin="upper", cmap="viridis")
    axes[1, 3].set_title("Final aligned probe distance")
    axes[1, 3].set_xticks(
        range(len(branch_names)), branch_names, rotation=35
    )
    axes[1, 3].set_yticks(range(len(branch_names)), branch_names)
    axes[1, 3].tick_params(labelsize=7)
    fig.colorbar(image, ax=axes[1, 3], fraction=0.046, pad=0.04)
    for axis in axes.ravel()[:7]:
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("exp042 fixed-direction initialization-magnitude ablation")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def _validate_initialization_ablation_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    reconstructions: dict[str, dict[str, Any]],
) -> None:
    output = config["output"]
    figure_path = run_dir / "figures" / output[
        "initialization_ablation_figure_filename"
    ]
    hdf5_path = run_dir / "outputs" / output["hdf5_filename"]
    required = {
        run_dir / "config.yaml",
        run_dir / "metadata.json",
        run_dir / "metrics.json",
        run_dir / "run_state.json",
        hdf5_path,
        figure_path,
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"Missing initialization-ablation artifacts: {missing}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    expected_role = str(config["execution"]["mode"])
    settings = config["verification"]["initialization_ablation"]
    if metrics.get("run_role") != expected_role:
        raise RuntimeError("The initialization-ablation run role is invalid.")
    diagnostic = metrics.get("initialization_ablation", {}).get(
        "truth_free_pairwise_diagnostic", {}
    )
    if (
        diagnostic.get("truth_used_by_diagnostic") is not False
        or diagnostic.get("equal_optimizer_algorithm") is not True
        or diagnostic.get("equal_curve_lengths") is not True
    ):
        raise RuntimeError("The initialization-ablation controls are invalid.")
    expected_names = [
        str(branch["name"])
        for branch in config["verification"]["initialization_ablation"][
            "branches"
        ]
    ]
    if diagnostic.get("branch_names") != expected_names:
        raise RuntimeError("The registered initialization branches changed.")
    magnitude_sweep = settings["initialization_family"] == (
        "fixed_direction_magnitude_sweep"
    )
    if magnitude_sweep:
        if diagnostic[
            "maximum_initial_unit_direction_pairwise_l2_error"
        ] > float(settings["same_direction_max_unit_l2_error_tolerance"]):
            raise RuntimeError("Magnitude branches changed perturbation direction.")
        checkpoint_count = (
            int(settings["equal_iteration_budget"])
            // int(settings["checkpoint_interval"])
            + 1
        )
        expected_checkpoint_shape = (
            checkpoint_count,
            len(expected_names),
            len(expected_names),
        )
        for key in (
            "checkpoint_pairwise_raw_probe_relative_l2_curve",
            "checkpoint_pairwise_global_phase_aligned_probe_relative_l2_curve",
            "checkpoint_pairwise_prediction_relative_l2_curve",
        ):
            if np.asarray(diagnostic[key]).shape != expected_checkpoint_shape:
                raise RuntimeError(f"Unexpected checkpoint shape for {key}.")
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
        if set(h5["entry"]) != expected_entry:
            raise RuntimeError("Unexpected initialization-ablation /entry layout.")
        if h5["entry/config_yaml"].asstr()[()] != config_to_yaml(config):
            raise RuntimeError("The embedded initialization config changed.")
        root = "entry/reconstruction/initialization_ablation/branches"
        for name, result in reconstructions.items():
            branch_root = f"{root}/{name}"
            if not np.array_equal(
                h5[f"{branch_root}/P_B_init"][...], result["P_B_init"]
            ) or not np.array_equal(
                h5[f"{branch_root}/P_B_rec"][...], result["P_B_rec"]
            ):
                raise RuntimeError(f"Raw branch probes changed for {name}.")
            expected_curve_length = int(result["iterations_completed"]) + 1
            for curve_name in (
                "loss_curve",
                "detector_relative_residual_curve",
                "simulation_evaluation_only/probe_raw_relative_l2_curve",
                "simulation_evaluation_only/"
                "probe_global_phase_aligned_relative_l2_curve",
            ):
                if h5[f"{branch_root}/{curve_name}"].shape != (
                    expected_curve_length,
                ):
                    raise RuntimeError(
                        f"Unexpected curve shape for {name}/{curve_name}."
                    )
        if magnitude_sweep:
            diagnostic_root = (
                "entry/reconstruction/initialization_ablation/"
                "truth_free_pairwise_diagnostic"
            )
            pairs = (
                (
                    "checkpoint_pairwise_raw_probe_relative_l2_curve",
                    "final_pairwise_raw_probe_relative_l2",
                ),
                (
                    "checkpoint_pairwise_global_phase_aligned_probe_relative_l2_curve",
                    "final_pairwise_global_phase_aligned_probe_relative_l2",
                ),
                (
                    "checkpoint_pairwise_prediction_relative_l2_curve",
                    "final_pairwise_prediction_relative_l2",
                ),
            )
            for curve_name, final_name in pairs:
                curve = h5[f"{diagnostic_root}/{curve_name}"]
                final = h5[f"{diagnostic_root}/{final_name}"]
                if curve.shape != expected_checkpoint_shape or not np.array_equal(
                    curve[-1], final[...]
                ):
                    raise RuntimeError(
                        f"Checkpoint final state disagrees for {curve_name}."
                    )
        if not _all_numeric_hdf5_finite(h5["entry"]):
            raise RuntimeError("Initialization HDF5 contains non-finite data.")
    image = plt.imread(figure_path)
    if image.ndim not in {2, 3} or not np.all(np.isfinite(image)):
        raise RuntimeError("The initialization-ablation figure is unreadable.")


def _detector_quadrature_ablation_metadata(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    provenance = config["provenance"]
    return {
        "experiment_id": "exp042",
        "created_at_utc": created_at_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_config": str(config_path),
        "run_role": str(config["execution"]["mode"]),
        "development_status": (
            "Development baseline / No scientific pass-fail conclusion"
        ),
        "operator_branch": provenance["source_branch"],
        "development_data_origin": provenance["development_data_origin"],
        "primary_detector_data_model": config["detector"]["model"],
        "control_detector_data_model": Q1_POINT_MISMATCH_READOUT,
        "detector_operator_ablation_axis": (
            "detector_data_and_reconstruction_readout_pairing"
        ),
        "q1_mismatch_branch_role": "explicit_mismatch_diagnostic",
        "q1_matched_branch_role": (
            "simplified_matched_control_not_exp040_matched"
        ),
        "q1_branch_is_exp040_matched": False,
        "q1_q1_matched_control_included": True,
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "known_sample_b": True,
        "probe_only_reconstruction": True,
        "reconstruction_performed_in_this_run": True,
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "simulation_truth_evaluation_only": True,
        "scientific_claim_boundary": (
            "three-branch q4/q4, q4/q1, and q1/q1 detector data-model "
            "pairing control on one frozen noiseless development truth, "
            "with q1 branches remaining simplified diagnostics only"
        ),
    }


def _detector_quadrature_ablation_design_payload(
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": settings["role"],
        "primary_data_model": settings["primary_data_model"],
        "control_data_model": settings["control_data_model"],
        "sweep_axis": settings["sweep_axis"],
        "point_interpolation": settings["point_interpolation"],
        "branches": {
            str(branch["name"]): dict(branch)
            for branch in settings["branches"]
        },
        "reference_branch": settings["reference_branch"],
        "primary_q4_data_branches": settings["primary_q4_data_branches"],
        "same_truth_b_scan_initialization": True,
        "equal_optimizer_settings": True,
        "equal_iteration_budget": settings["equal_iteration_budget"],
        "checkpoint_interval": settings["checkpoint_interval"],
        "pairwise_alignment": settings["pairwise_alignment"],
        "comparison_contract": settings["comparison_contract"],
        "numerical_control_tolerances": settings[
            "numerical_control_tolerances"
        ],
        "figure_panels": settings["figure_panels"],
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "truth_used_by_branch_selection": False,
        "truth_used_by_stopping": False,
        "scientific_thresholds_preregistered": False,
        "q1_branch_is_exp040_matched": False,
        "q1_q1_matched_control_included": True,
    }


def _detector_quadrature_ablation_hdf5_payload(
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    case: dict[str, Any],
    reconstructions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    measurements: dict[str, np.ndarray],
) -> dict[str, Any]:
    operator = case["operator"]
    if not isinstance(operator, MatchedKnownBProbeOperator):
        raise TypeError("case operator has the wrong type.")
    settings = config["verification"]["detector_quadrature_ablation"]
    specifications = {
        str(branch["name"]): dict(branch) for branch in settings["branches"]
    }
    branches = {
        name: {
            "operator_spec": specifications[name],
            **_reconstruction_hdf5_payload(
                reconstructions[name], evaluations[name]
            ),
        }
        for name in reconstructions
    }
    return {
        "I_stack": case["I_stack"],
        "scan_positions": case["scan_positions"],
        "instrument": {
            "wavelength_m": config["optics"]["wavelength_m"],
            "internal_reference_index": config["optics"][
                "internal_reference_index"
            ],
            "external_medium_index": config["optics"][
                "external_medium_index"
            ],
            "z_AB_m": config["optics"]["z_AB_m"],
            "z_BC_m": config["optics"]["z_BC_m"],
            "probe_grid": {
                "plane": "B",
                "axis_order": ["y", "x"],
                "native_shape": operator.native_shape,
                "open_shape": operator.open_shape,
                "node_dx_m": operator.node_dx_m,
            },
            "detector": {
                "model": config["detector"]["model"],
                "data_model_role": "simulation_truth_measurement",
                "quadrature_factor": operator.quadrature_factor,
                "pixel_size_m": config["detector"]["pixel_size_m"],
                "native_roi_shape": operator.detector_roi_shape,
                "quadrature_weights": np.full(
                    operator.quadrature_factor**2,
                    1.0 / operator.quadrature_factor**2,
                ),
            },
        },
        "sample": {
            "sample_a": dict(config["sample_a"]),
            "sample_b": dict(config["sample_b"]),
        },
        "truth": {
            "identity": (
                "simulation truth under the selected exp040 scalar "
                "working model"
            ),
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "P_B_true": case["P_B_true"],
            "B_true": case["B_true"],
            "B_support_true": case["B_support_true"],
            "U_A_exit_true": case["U_A_exit_true"],
            "z_m": case["z_m"],
            "slice_widths_m": case["slice_widths_m"],
            "D_z_m": case["D_z_m"],
        },
        "reconstruction": {
            "detector_quadrature_ablation": {
                "design": _detector_quadrature_ablation_design_payload(
                    settings
                ),
                "control_measurements": {
                    "q1_point_control_data": {
                        "I_stack": measurements["matched_q1_point"],
                        "detector_readout": Q1_POINT_MISMATCH_READOUT,
                        "role": "simulation_matched_control_only",
                        "exp040_matched": False,
                    }
                },
                "branches": branches,
                "truth_free_pairwise_diagnostic": diagnostic,
            }
        },
        "config_yaml": config_to_yaml(config),
        "metadata": metadata,
        "metrics": metrics,
    }


def _save_detector_quadrature_ablation_figure(
    run_dir: Path,
    reconstructions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    filename: str,
) -> Path:
    figure_path = run_dir / "figures" / filename
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
    for name, reconstruction in reconstructions.items():
        axes[0, 0].semilogy(
            np.maximum(reconstruction["loss_curve"], np.finfo(float).tiny),
            label=name,
        )
        axes[0, 1].semilogy(
            np.maximum(
                reconstruction["detector_relative_residual_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
        evaluation = evaluations[name]
        axes[0, 2].semilogy(
            np.maximum(
                evaluation["probe_raw_relative_l2_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
        axes[0, 3].semilogy(
            np.maximum(
                evaluation["probe_global_phase_aligned_relative_l2_curve"],
                np.finfo(float).tiny,
            ),
            label=name,
        )
    axes[0, 0].set(
        title="Branch-data pixel-intensity loss", xlabel="Iteration"
    )
    axes[0, 1].set(title="Own-data relative residual", xlabel="Iteration")
    axes[0, 2].set(
        title="Raw probe error (simulation evaluation only)",
        xlabel="Iteration",
    )
    axes[0, 3].set(
        title="Aligned probe error (simulation evaluation only)",
        xlabel="Iteration",
    )
    branch_names = list(diagnostic["branch_names"])
    reference_branch = str(diagnostic["reference_branch"])
    checkpoint_iterations = np.asarray(
        diagnostic["checkpoint_iterations"], dtype=np.int64
    )
    checkpoint_curves = (
        np.asarray(
            diagnostic[
                "checkpoint_raw_probe_relative_l2_to_reference_curve"
            ],
            dtype=np.float64,
        ),
        np.asarray(
            diagnostic[
                "checkpoint_global_phase_aligned_probe_"
                "relative_l2_to_reference_curve"
            ],
            dtype=np.float64,
        ),
        np.asarray(
            diagnostic[
                "checkpoint_prediction_relative_l2_to_reference_curve"
            ],
            dtype=np.float64,
        ),
    )
    checkpoint_titles = (
        "Raw probe distance to matched q4",
        "Aligned probe distance to matched q4",
        "Branch-output prediction distance",
    )
    for axis, curve, title in zip(
        axes[1, :3], checkpoint_curves, checkpoint_titles, strict=True
    ):
        for branch_index, name in enumerate(branch_names):
            if name == reference_branch:
                continue
            axis.plot(
                checkpoint_iterations,
                curve[:, branch_index],
                marker="o",
                ms=2.5,
                label=name,
            )
        axis.set(title=title, xlabel="Iteration")
    final_matrix = np.asarray(
        diagnostic["final_pairwise_global_phase_aligned_probe_relative_l2"],
        dtype=np.float64,
    )
    image = axes[1, 3].imshow(final_matrix, origin="upper", cmap="viridis")
    axes[1, 3].set_title("Final aligned probe distance")
    axes[1, 3].set_xticks(
        range(len(branch_names)), branch_names, rotation=35
    )
    axes[1, 3].set_yticks(range(len(branch_names)), branch_names)
    axes[1, 3].tick_params(labelsize=7)
    fig.colorbar(image, ax=axes[1, 3], fraction=0.046, pad=0.04)
    for axis in axes.ravel()[:7]:
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(
        "exp042 detector data-model pairing: q4/q4, q4/q1, and q1/q1"
    )
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def _validate_detector_quadrature_ablation_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    reconstructions: dict[str, dict[str, Any]],
    expected_measurements: dict[str, np.ndarray],
) -> None:
    output = config["output"]
    figure_path = run_dir / "figures" / output[
        "detector_quadrature_ablation_figure_filename"
    ]
    hdf5_path = run_dir / "outputs" / output["hdf5_filename"]
    required = {
        run_dir / "config.yaml",
        run_dir / "metadata.json",
        run_dir / "metrics.json",
        run_dir / "run_state.json",
        hdf5_path,
        figure_path,
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"Missing detector-ablation artifacts: {missing}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("run_role") != "detector_quadrature_three_branch_control":
        raise RuntimeError("The detector-ablation run role is invalid.")
    settings = config["verification"]["detector_quadrature_ablation"]
    tolerances = settings["numerical_control_tolerances"]
    controls = metrics.get("operator_consistency", {})
    required_true = (
        "matched_q4_truth_replay_exact",
        "q1_self_truth_replay_exact",
        "q1_self_zero_residual_prediction_exact",
        "q1_self_truth_fixed_point_exact",
        "prediction_shapes_equal",
        "shared_linear_array_components_identity",
        "shared_linear_scalar_components_exact",
        "only_registered_readout_differs",
        "q1_self_data_all_nonnegative",
        "q1_q1_matched_control_included",
    )
    if any(controls.get(key) is not True for key in required_true):
        raise RuntimeError("A detector-ablation exact control failed.")
    if (
        controls.get("q1_on_q4_truth_replay_exact") is not False
        or controls.get("q1_branch_is_exp040_matched") is not False
    ):
        raise RuntimeError("The q1 mismatch boundary changed.")
    for metric_name, tolerance_name in (
        (
            "q1_point_readout_adjoint_relative_error",
            "point_readout_adjoint_relative_error",
        ),
        (
            "q1_intensity_jacobian_adjoint_relative_error",
            "intensity_jacobian_adjoint_relative_error",
        ),
        (
            "q1_full_loss_directional_gradient_relative_error_on_q4_data",
            "full_loss_directional_gradient_relative_error",
        ),
        (
            "q1_full_loss_directional_gradient_relative_error_on_q1_data",
            "full_loss_directional_gradient_relative_error",
        ),
    ):
        if float(controls[metric_name]) >= float(tolerances[tolerance_name]):
            raise RuntimeError(f"Detector control exceeded: {metric_name}.")
    diagnostic = metrics["detector_quadrature_ablation"][
        "truth_free_pairwise_diagnostic"
    ]
    expected_names = [
        str(branch["name"]) for branch in settings["branches"]
    ]
    if (
        diagnostic.get("branch_names") != expected_names
        or diagnostic.get("initial_probe_exact_equal") is not True
        or diagnostic.get("truth_used_by_diagnostic") is not False
        or diagnostic.get("equal_optimizer_algorithm") is not True
        or diagnostic.get("equal_curve_lengths") is not True
    ):
        raise RuntimeError("The detector pairwise diagnostic is invalid.")
    checkpoint_count = (
        int(settings["equal_iteration_budget"])
        // int(settings["checkpoint_interval"])
        + 1
    )
    expected_checkpoint_shape = (
        checkpoint_count,
        len(expected_names),
        len(expected_names),
    )
    if (
        set(expected_measurements) != set(expected_names)
        or not np.array_equal(
            expected_measurements["matched_q4"],
            expected_measurements["mismatch_q1_point"],
        )
        or np.array_equal(
            expected_measurements["matched_q4"],
            expected_measurements["matched_q1_point"],
        )
    ):
        raise RuntimeError("The registered branch measurement mapping changed.")
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
        if set(h5["entry"]) != expected_entry:
            raise RuntimeError("Unexpected detector-ablation /entry layout.")
        if h5["entry/config_yaml"].asstr()[()] != config_to_yaml(config):
            raise RuntimeError("The embedded detector-ablation config changed.")
        if not np.array_equal(
            h5["entry/data/I_stack"][...],
            expected_measurements["matched_q4"],
        ):
            raise RuntimeError("The primary q4 data changed in HDF5.")
        root = "entry/reconstruction/detector_quadrature_ablation"
        if not np.array_equal(
            h5[
                f"{root}/control_measurements/"
                "q1_point_control_data/I_stack"
            ][...],
            expected_measurements["matched_q1_point"],
        ):
            raise RuntimeError("The q1 matched-control data changed in HDF5.")
        for name, result in reconstructions.items():
            branch_root = f"{root}/branches/{name}"
            if not np.array_equal(
                h5[f"{branch_root}/P_B_init"][...], result["P_B_init"]
            ) or not np.array_equal(
                h5[f"{branch_root}/P_B_rec"][...], result["P_B_rec"]
            ):
                raise RuntimeError(f"Raw detector branch probes changed: {name}.")
        diagnostic_root = f"{root}/truth_free_pairwise_diagnostic"
        for curve_name, final_name in (
            (
                "checkpoint_pairwise_raw_probe_relative_l2_curve",
                "final_pairwise_raw_probe_relative_l2",
            ),
            (
                "checkpoint_pairwise_global_phase_aligned_probe_relative_l2_curve",
                "final_pairwise_global_phase_aligned_probe_relative_l2",
            ),
            (
                "checkpoint_pairwise_prediction_relative_l2_curve",
                "final_pairwise_prediction_relative_l2",
            ),
        ):
            curve = h5[f"{diagnostic_root}/{curve_name}"]
            final = h5[f"{diagnostic_root}/{final_name}"]
            if curve.shape != expected_checkpoint_shape or not np.array_equal(
                curve[-1], final[...]
            ):
                raise RuntimeError(
                    f"Detector checkpoint final disagrees: {curve_name}."
                )
        if not _all_numeric_hdf5_finite(h5["entry"]):
            raise RuntimeError("Detector-ablation HDF5 contains non-finite data.")
    image = plt.imread(figure_path)
    if image.ndim not in {2, 3} or not np.all(np.isfinite(image)):
        raise RuntimeError("The detector-ablation figure is unreadable.")


def _run_detector_quadrature_ablation(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    validate_exp042_config(config)
    output = config["output"]
    output_root = Path(output["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = make_run_dir(output_root, str(output["run_name"]))
    save_config(run_dir / "config.yaml", config)
    state_path = run_dir / "run_state.json"
    save_json(
        state_path,
        {
            "status": "running",
            "artifacts_validated": False,
            "started_at_utc": created_at_utc(),
        },
    )
    started = time.perf_counter()
    try:
        case = build_matched_development_case(config)
        matched_operator = case["operator"]
        if not isinstance(matched_operator, MatchedKnownBProbeOperator):
            raise TypeError("case operator has the wrong type.")
        settings = config["verification"]["detector_quadrature_ablation"]
        specifications = {
            str(branch["name"]): dict(branch)
            for branch in settings["branches"]
        }
        point_operator = make_detector_readout_operator(
            matched_operator, Q1_POINT_MISMATCH_READOUT
        )
        operators: dict[str, MatchedKnownBProbeOperator] = {}
        for name, specification in specifications.items():
            readout = str(specification["detector_readout"])
            if readout == MATCHED_PIXEL_AVERAGE_READOUT:
                operators[name] = matched_operator
            elif readout == Q1_POINT_MISMATCH_READOUT:
                operators[name] = point_operator
            else:
                raise RuntimeError(f"Unexpected detector readout: {readout}")
        reference_name = str(settings["reference_branch"])
        if operators[reference_name] is not matched_operator:
            raise RuntimeError("The q4 reference operator changed.")
        if operators["mismatch_q1_point"] is not operators["matched_q1_point"]:
            raise RuntimeError("The two q1 branches must share one operator.")
        primary_q4_data = np.asarray(case["I_stack"], dtype=np.float64)
        q1_control_data = point_operator.predict_stack(case["P_B_true"])
        data_sources = {
            "primary_q4_data": primary_q4_data,
            "q1_point_control_data": q1_control_data,
        }
        measurements = {
            name: data_sources[str(specification["data_source"])]
            for name, specification in specifications.items()
        }
        if (
            not np.array_equal(
                measurements["matched_q4"],
                measurements["mismatch_q1_point"],
            )
            or np.array_equal(
                measurements["matched_q4"],
                measurements["matched_q1_point"],
            )
        ):
            raise RuntimeError("The registered branch data pairing changed.")
        initial = matched_operator.homogeneous_probe_native.copy()
        initializations = {
            name: initial.copy() for name in specifications
        }
        if not all(
            np.array_equal(values, initial)
            for values in initializations.values()
        ):
            raise RuntimeError("Detector branches changed initialization.")
        controls = detector_quadrature_ablation_consistency_metrics(
            case, config
        )
        tolerances = settings["numerical_control_tolerances"]
        for metric_name, tolerance_name in (
            (
                "q1_point_readout_adjoint_relative_error",
                "point_readout_adjoint_relative_error",
            ),
            (
                "q1_intensity_jacobian_adjoint_relative_error",
                "intensity_jacobian_adjoint_relative_error",
            ),
            (
                "q1_full_loss_directional_gradient_relative_error_on_q4_data",
                "full_loss_directional_gradient_relative_error",
            ),
            (
                "q1_full_loss_directional_gradient_relative_error_on_q1_data",
                "full_loss_directional_gradient_relative_error",
            ),
        ):
            if float(controls[metric_name]) >= float(
                tolerances[tolerance_name]
            ):
                raise RuntimeError(
                    f"Pre-reconstruction detector control failed: {metric_name}."
                )
        required_true = (
            "matched_q4_truth_replay_exact",
            "q1_self_truth_replay_exact",
            "q1_self_zero_residual_prediction_exact",
            "q1_self_truth_fixed_point_exact",
            "prediction_shapes_equal",
            "shared_linear_array_components_identity",
            "shared_linear_scalar_components_exact",
            "only_registered_readout_differs",
            "q1_self_data_all_nonnegative",
            "q1_q1_matched_control_included",
        )
        if any(controls[key] is not True for key in required_true):
            raise RuntimeError("A pre-reconstruction exact control failed.")
        if controls["q1_on_q4_truth_replay_exact"] is not False:
            raise RuntimeError("The q1 negative control unexpectedly matched q4.")
        reconstructions = {
            name: reconstruct_known_b_probe(
                operators[name],
                measurements[name],
                initialization,
                config["reconstruction"],
            )
            for name, initialization in initializations.items()
        }
        evaluations = {
            name: simulation_evaluation_only(result, case["P_B_true"])
            for name, result in reconstructions.items()
        }
        diagnostic = truth_free_detector_quadrature_ablation_diagnostic(
            reconstructions,
            operators,
            reference_branch=reference_name,
            checkpoint_interval=int(settings["checkpoint_interval"]),
        )
        elapsed = time.perf_counter() - started
        branch_metrics: dict[str, Any] = {}
        for name, result in reconstructions.items():
            evaluation = evaluations[name]
            branch_metrics[name] = {
                "operator_spec": specifications[name],
                "measurement_fit": {
                    "data_source": specifications[name]["data_source"],
                    "data_readout": specifications[name]["data_readout"],
                    **_measurement_metrics(result),
                    "loss_curve": result["loss_curve"],
                    "detector_relative_residual_curve": result[
                        "detector_relative_residual_curve"
                    ],
                },
                "simulation_evaluation_only": {
                    "probe_raw_relative_l2_curve": evaluation[
                        "probe_raw_relative_l2_curve"
                    ],
                    "probe_global_phase_aligned_relative_l2_curve": evaluation[
                        "probe_global_phase_aligned_relative_l2_curve"
                    ],
                    "initial_probe_raw_relative_l2": evaluation[
                        "initial_probe_raw_relative_l2"
                    ],
                    "final_probe_raw_relative_l2": evaluation[
                        "final_probe_raw_relative_l2"
                    ],
                    "initial_probe_global_phase_aligned_relative_l2": evaluation[
                        "initial_probe_global_phase_aligned_relative_l2"
                    ],
                    "final_probe_global_phase_aligned_relative_l2": evaluation[
                        "final_probe_global_phase_aligned_relative_l2"
                    ],
                },
            }
        metrics = {
            "development_status": (
                "Development baseline / No scientific pass-fail conclusion"
            ),
            "run_role": str(config["execution"]["mode"]),
            "detector_quadrature_ablation": {
                "design": _detector_quadrature_ablation_design_payload(
                    settings
                ),
                "branch_count": len(reconstructions),
                "branches": branch_metrics,
                "truth_free_pairwise_diagnostic": diagnostic,
            },
            "operator_consistency": controls,
            "data_controls": {
                "same_q4_data_for_primary_and_mismatch": True,
                "primary_q4_I_stack_shape": np.asarray(
                    primary_q4_data.shape
                ),
                "q1_control_I_stack_shape": np.asarray(
                    q1_control_data.shape
                ),
                "both_stacks_all_finite": bool(
                    np.all(np.isfinite(primary_q4_data))
                    and np.all(np.isfinite(q1_control_data))
                ),
                "both_stacks_all_nonnegative": bool(
                    np.all(primary_q4_data >= 0.0)
                    and np.all(q1_control_data >= 0.0)
                ),
                "q1_control_relative_l2_to_primary_q4": controls[
                    "q1_self_data_relative_l2_to_q4"
                ],
                "primary_detector_data_model": config["detector"]["model"],
                "control_detector_data_model": Q1_POINT_MISMATCH_READOUT,
            },
            "runtime_seconds": elapsed,
            "known_sample_b": True,
            "sample_b_updated": False,
            "reconstruction_performed_in_this_run": True,
            "paired_continuation_performed_in_this_run": False,
            "local_spectral_diagnostic_performed_in_this_run": False,
            "waist_estimation_performed": False,
            "scientific_pass_fail_conclusion": False,
        }
        metadata = _detector_quadrature_ablation_metadata(config, config_path)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_ptycho_hdf5(
            run_dir / "outputs" / output["hdf5_filename"],
            **_detector_quadrature_ablation_hdf5_payload(
                config,
                metadata,
                metrics,
                case,
                reconstructions,
                evaluations,
                diagnostic,
                measurements,
            ),
        )
        figure_path = _save_detector_quadrature_ablation_figure(
            run_dir,
            reconstructions,
            evaluations,
            diagnostic,
            str(output["detector_quadrature_ablation_figure_filename"]),
        )
        _validate_detector_quadrature_ablation_artifacts(
            run_dir,
            config,
            reconstructions,
            measurements,
        )
        save_json(
            state_path,
            {
                "status": "complete",
                "artifacts_validated": True,
                "completed_at_utc": created_at_utc(),
                "runtime_seconds": elapsed,
                "config_sha256": _sha256(run_dir / "config.yaml"),
                "metrics_sha256": _sha256(run_dir / "metrics.json"),
                "hdf5_sha256": _sha256(
                    run_dir / "outputs" / output["hdf5_filename"]
                ),
                "figure_count": 1,
                "figure_sha256": _sha256(figure_path),
            },
        )
    except Exception as error:
        save_json(
            state_path,
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return run_dir


def _run_initialization_ablation(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    validate_exp042_config(config)
    output = config["output"]
    output_root = Path(output["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = make_run_dir(output_root, str(output["run_name"]))
    save_config(run_dir / "config.yaml", config)
    state_path = run_dir / "run_state.json"
    save_json(
        state_path,
        {
            "status": "running",
            "artifacts_validated": False,
            "started_at_utc": created_at_utc(),
        },
    )
    started = time.perf_counter()
    try:
        case = build_matched_development_case(config)
        operator = case["operator"]
        if not isinstance(operator, MatchedKnownBProbeOperator):
            raise TypeError("case operator has the wrong type.")
        settings = config["verification"]["initialization_ablation"]
        family = str(settings["initialization_family"])
        magnitude_sweep = family == "fixed_direction_magnitude_sweep"
        homogeneous = operator.homogeneous_probe_native.copy()
        initializations: dict[str, np.ndarray] = {}
        specifications: dict[str, dict[str, Any]] = {}
        for branch in settings["branches"]:
            name = str(branch["name"])
            specifications[name] = dict(branch)
            if branch["kind"] == "homogeneous_reference_probe":
                initialization = homogeneous.copy()
            else:
                if magnitude_sweep:
                    seed = int(settings["fixed_perturbation_seed"])
                    relative_l2_value = float(
                        branch["relative_l2_to_homogeneous"]
                    )
                else:
                    seed = int(branch["seed"])
                    relative_l2_value = float(
                        settings["perturbation_relative_l2_to_homogeneous"]
                    )
                initialization = make_deterministic_complex_probe_initialization(
                    homogeneous,
                    seed=seed,
                    relative_l2_to_primary=relative_l2_value,
                )
                repeated = make_deterministic_complex_probe_initialization(
                    homogeneous,
                    seed=seed,
                    relative_l2_to_primary=relative_l2_value,
                )
                if not np.array_equal(initialization, repeated):
                    raise RuntimeError(f"Initialization is not deterministic: {name}")
            initializations[name] = initialization
        if magnitude_sweep:
            reference_name = str(settings["reference_branch"])
            reference_initialization = initializations[reference_name]
            unit_directions = []
            for name, initialization in initializations.items():
                if name == reference_name:
                    continue
                direction = initialization - reference_initialization
                direction_norm = float(np.linalg.norm(direction.ravel()))
                if direction_norm <= np.finfo(np.float64).eps:
                    raise RuntimeError(f"Magnitude direction is zero: {name}")
                unit_directions.append(direction / direction_norm)
            maximum_direction_error = max(
                (
                    float(
                        np.linalg.norm(
                            (left - right).ravel()
                        )
                    )
                    for index, left in enumerate(unit_directions)
                    for right in unit_directions[index + 1 :]
                ),
                default=0.0,
            )
            if maximum_direction_error > float(
                settings["same_direction_max_unit_l2_error_tolerance"]
            ):
                raise RuntimeError("Magnitude branches changed direction.")
        reconstructions = {
            name: reconstruct_known_b_probe(
                operator,
                case["I_stack"],
                initialization,
                config["reconstruction"],
            )
            for name, initialization in initializations.items()
        }
        evaluations = {
            name: simulation_evaluation_only(result, case["P_B_true"])
            for name, result in reconstructions.items()
        }
        diagnostic_kwargs: dict[str, Any] = {}
        if magnitude_sweep:
            diagnostic_kwargs = {
                "operator": operator,
                "checkpoint_interval": int(settings["checkpoint_interval"]),
            }
        diagnostic = truth_free_initialization_ablation_diagnostic(
            reconstructions,
            reference_branch=str(settings["reference_branch"]),
            **diagnostic_kwargs,
        )
        replay = operator.predict_stack(case["P_B_true"])
        truth_replay_exact = bool(np.array_equal(replay, case["I_stack"]))
        if not truth_replay_exact:
            raise RuntimeError("The initialization-ablation truth replay changed.")
        elapsed = time.perf_counter() - started
        branch_metrics: dict[str, Any] = {}
        for name, result in reconstructions.items():
            evaluation = evaluations[name]
            branch_metrics[name] = {
                "initialization_spec": specifications[name],
                "measurement": {
                    **_measurement_metrics(result),
                    "loss_curve": result["loss_curve"],
                    "detector_relative_residual_curve": result[
                        "detector_relative_residual_curve"
                    ],
                },
                "simulation_evaluation_only": {
                    "probe_raw_relative_l2_curve": evaluation[
                        "probe_raw_relative_l2_curve"
                    ],
                    "probe_global_phase_aligned_relative_l2_curve": evaluation[
                        "probe_global_phase_aligned_relative_l2_curve"
                    ],
                    "initial_probe_raw_relative_l2": evaluation[
                        "initial_probe_raw_relative_l2"
                    ],
                    "final_probe_raw_relative_l2": evaluation[
                        "final_probe_raw_relative_l2"
                    ],
                    "initial_probe_global_phase_aligned_relative_l2": evaluation[
                        "initial_probe_global_phase_aligned_relative_l2"
                    ],
                    "final_probe_global_phase_aligned_relative_l2": evaluation[
                        "final_probe_global_phase_aligned_relative_l2"
                    ],
                },
            }
        diagnostic_metrics = {
            key: value
            for key, value in diagnostic.items()
            if not key.startswith("P_B_")
        }
        metrics = {
            "development_status": (
                "Development baseline / No scientific pass-fail conclusion"
            ),
            "run_role": str(config["execution"]["mode"]),
            "initialization_ablation": {
                "design": _initialization_ablation_design_payload(settings),
                "branch_count": len(reconstructions),
                "branches": branch_metrics,
                "truth_free_pairwise_diagnostic": diagnostic_metrics,
            },
            "operator_consistency": {
                "truth_replay_exact": truth_replay_exact,
                "truth_replay_relative_l2": _relative_l2(
                    replay, case["I_stack"]
                ),
                "inherited_forward_adjoint_gradient_controls_rerun": False,
            },
            "runtime_seconds": elapsed,
            "known_sample_b": True,
            "sample_b_updated": False,
            "reconstruction_performed_in_this_run": True,
            "paired_continuation_performed_in_this_run": False,
            "local_spectral_diagnostic_performed_in_this_run": False,
            "waist_estimation_performed": False,
            "scientific_pass_fail_conclusion": False,
        }
        metadata = _initialization_ablation_metadata(config, config_path)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_ptycho_hdf5(
            run_dir / "outputs" / output["hdf5_filename"],
            **_initialization_ablation_hdf5_payload(
                config,
                metadata,
                metrics,
                case,
                reconstructions,
                evaluations,
                diagnostic,
            ),
        )
        if magnitude_sweep:
            figure_path = _save_initialization_magnitude_ablation_figure(
                run_dir,
                reconstructions,
                evaluations,
                diagnostic,
                str(output["initialization_ablation_figure_filename"]),
            )
        else:
            figure_path = _save_initialization_ablation_figure(
                run_dir,
                reconstructions,
                evaluations,
                diagnostic,
                str(output["initialization_ablation_figure_filename"]),
            )
        _validate_initialization_ablation_artifacts(
            run_dir, config, reconstructions
        )
        save_json(
            state_path,
            {
                "status": "complete",
                "artifacts_validated": True,
                "completed_at_utc": created_at_utc(),
                "runtime_seconds": elapsed,
                "config_sha256": _sha256(run_dir / "config.yaml"),
                "metrics_sha256": _sha256(run_dir / "metrics.json"),
                "hdf5_sha256": _sha256(
                    run_dir / "outputs" / output["hdf5_filename"]
                ),
                "figure_count": 1,
                "figure_sha256": _sha256(figure_path),
            },
        )
    except Exception as error:
        save_json(
            state_path,
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return run_dir


def _run_full_baseline(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    validate_exp042_config(config)
    output = config["output"]
    output_root = Path(output["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = make_run_dir(output_root, str(output["run_name"]))
    save_config(run_dir / "config.yaml", config)
    state_path = run_dir / "run_state.json"
    save_json(
        state_path,
        {
            "status": "running",
            "artifacts_validated": False,
            "started_at_utc": created_at_utc(),
        },
    )
    started = time.perf_counter()
    try:
        case = build_matched_development_case(config)
        controls = operator_consistency_metrics(case, config)
        operator = case["operator"]
        if not isinstance(operator, MatchedKnownBProbeOperator):
            raise TypeError("case operator has the wrong type.")
        init_probe = operator.homogeneous_probe_native.copy()
        reconstruction = reconstruct_known_b_probe(
            operator,
            case["I_stack"],
            init_probe,
            config["reconstruction"],
        )
        stability_settings = config["reconstruction"][
            "initialization_stability_control"
        ]
        control_init_probe = make_deterministic_complex_probe_initialization(
            init_probe,
            seed=int(stability_settings["seed"]),
            relative_l2_to_primary=float(
                stability_settings["relative_l2_to_primary"]
            ),
        )
        control_init_repeat = make_deterministic_complex_probe_initialization(
            init_probe,
            seed=int(stability_settings["seed"]),
            relative_l2_to_primary=float(
                stability_settings["relative_l2_to_primary"]
            ),
        )
        control_reconstruction = reconstruct_known_b_probe(
            operator,
            case["I_stack"],
            control_init_probe,
            config["reconstruction"],
        )
        stability_diagnostic = truth_free_initialization_stability_diagnostic(
            reconstruction, control_reconstruction
        )
        conditioning_settings = config["verification"][
            "pairwise_conditioning"
        ]
        conditioning_diagnostic = (
            truth_free_pairwise_conditioning_diagnostic(
                operator,
                case["I_stack"],
                reconstruction,
                control_reconstruction,
                random_seed=int(conditioning_settings["random_seed"]),
                random_direction_count=int(
                    conditioning_settings["random_direction_count"]
                ),
            )
        )
        continuation_settings = config["reconstruction"][
            "paired_continuation_control"
        ]
        continuation_optimizer_settings = dict(config["reconstruction"])
        continuation_optimizer_settings["iterations"] = int(
            continuation_settings["additional_iterations"]
        )
        primary_continuation = reconstruct_known_b_probe(
            operator,
            case["I_stack"],
            reconstruction["P_B_rec"],
            continuation_optimizer_settings,
        )
        control_continuation = reconstruct_known_b_probe(
            operator,
            case["I_stack"],
            control_reconstruction["P_B_rec"],
            continuation_optimizer_settings,
        )
        continuation_starts_exact = bool(
            np.array_equal(
                primary_continuation["P_B_init"], reconstruction["P_B_rec"]
            )
            and np.array_equal(
                control_continuation["P_B_init"],
                control_reconstruction["P_B_rec"],
            )
        )
        if not continuation_starts_exact:
            raise RuntimeError("Continuation did not preserve raw start fields.")
        continuation_diagnostic = truth_free_paired_continuation_diagnostic(
            operator,
            primary_continuation,
            control_continuation,
            checkpoint_interval=int(
                continuation_settings["checkpoint_interval"]
            ),
        )
        evaluation = simulation_evaluation_only(
            reconstruction, case["P_B_true"]
        )
        control_evaluation = simulation_evaluation_only(
            control_reconstruction, case["P_B_true"]
        )
        elapsed = time.perf_counter() - started
        metrics = {
            "development_status": (
                "Development baseline / No scientific pass-fail conclusion"
            ),
            "operator_consistency": controls,
            "measurement_only": _measurement_metrics(reconstruction),
            "initialization_stability_control": {
                "initialization": {
                    "role": stability_settings["role"],
                    "seed": int(stability_settings["seed"]),
                    "requested_relative_l2_to_primary": float(
                        stability_settings["relative_l2_to_primary"]
                    ),
                    "actual_relative_l2_to_primary": stability_diagnostic[
                        "initial_probe_raw_relative_l2"
                    ],
                    "perturbation_mean_abs": float(
                        abs(np.mean(control_init_probe - init_probe))
                    ),
                    "deterministic_repeat_exact": bool(
                        np.array_equal(control_init_probe, control_init_repeat)
                    ),
                    "deterministic_repeat_max_abs_error": float(
                        np.max(np.abs(control_init_probe - control_init_repeat))
                    ),
                    "truth_used_by_initialization": False,
                },
                "measurement_only": _measurement_metrics(
                    control_reconstruction
                ),
                "truth_free_pairwise_diagnostic": {
                    key: value
                    for key, value in stability_diagnostic.items()
                    if not key.startswith("P_B_")
                },
                "truth_free_pairwise_conditioning": {
                    key: value
                    for key, value in conditioning_diagnostic.items()
                    if not key.startswith("P_B_")
                },
                "simulation_evaluation_only": {
                    key: value
                    for key, value in control_evaluation.items()
                    if key != "P_B_rec_global_phase_aligned"
                },
            },
            "paired_continuation_control": {
                "settings": {
                    "role": continuation_settings["role"],
                    "start_field_source": continuation_settings["start_fields"],
                    "additional_iterations": int(
                        continuation_settings["additional_iterations"]
                    ),
                    "checkpoint_interval": int(
                        continuation_settings["checkpoint_interval"]
                    ),
                    "reuse_optimizer_settings": True,
                    "start_fields_exactly_same_run_raw_final": (
                        continuation_starts_exact
                    ),
                    "truth_used_by_continuation": False,
                    "truth_used_by_diagnostic": False,
                    "scientific_thresholds_preregistered": False,
                },
                "primary_measurement_only": _measurement_metrics(
                    primary_continuation
                ),
                "control_measurement_only": _measurement_metrics(
                    control_continuation
                ),
                "truth_free_pairwise_diagnostic": {
                    key: value
                    for key, value in continuation_diagnostic.items()
                    if not key.startswith("P_B_")
                    and key != "pairwise_alignment_factor_curve"
                },
            },
            "simulation_evaluation_only": {
                key: value
                for key, value in evaluation.items()
                if key != "P_B_rec_global_phase_aligned"
            },
            "runtime_seconds": elapsed,
            "known_sample_b": True,
            "sample_b_updated": False,
            "truth_used_by_optimizer": False,
            "waist_estimation_performed": False,
            "scientific_pass_fail_conclusion": False,
        }
        metadata = _metadata(config, config_path)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_ptycho_hdf5(
            run_dir / "outputs" / output["hdf5_filename"],
            **_hdf5_payload(
                config,
                metadata,
                metrics,
                case,
                reconstruction,
                evaluation,
                control_reconstruction,
                control_evaluation,
                stability_diagnostic,
                conditioning_diagnostic,
                primary_continuation,
                control_continuation,
                continuation_diagnostic,
            ),
        )
        figure_paths = _save_figures(
            run_dir,
            case,
            reconstruction,
            evaluation,
            control_reconstruction,
            control_evaluation,
            stability_diagnostic,
            conditioning_diagnostic,
            primary_continuation,
            control_continuation,
            continuation_diagnostic,
            list(output["figure_filenames"]),
        )
        expected_shapes = {
            "entry/data/I_stack": tuple(case["I_stack"].shape),
            "entry/data/scan_positions": tuple(case["scan_positions"].shape),
            "entry/truth/P_B_true": tuple(case["P_B_true"].shape),
            "entry/truth/B_true": tuple(case["B_true"].shape),
            "entry/reconstruction/P_B_init": tuple(case["P_B_true"].shape),
            "entry/reconstruction/P_B_rec": tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/initialization_stability_control/"
                "P_B_init"
            ): tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/initialization_stability_control/"
                "P_B_rec"
            ): tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/initialization_stability_control/"
                "truth_free_pairwise_conditioning/"
                "P_B_pairwise_direction_unit_l2"
            ): tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/paired_continuation_control/"
                "primary/P_B_init"
            ): tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/paired_continuation_control/"
                "primary/P_B_rec"
            ): tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/paired_continuation_control/"
                "control/P_B_init"
            ): tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/paired_continuation_control/"
                "control/P_B_rec"
            ): tuple(case["P_B_true"].shape),
            (
                "entry/reconstruction/paired_continuation_control/"
                "truth_free_pairwise_diagnostic/"
                "P_B_pairwise_start_direction_unit_l2"
            ): tuple(case["P_B_true"].shape),
        }
        _validate_artifacts(run_dir, config, expected_shapes)
        save_json(
            state_path,
            {
                "status": "complete",
                "artifacts_validated": True,
                "completed_at_utc": created_at_utc(),
                "runtime_seconds": elapsed,
                "config_sha256": _sha256(run_dir / "config.yaml"),
                "metrics_sha256": _sha256(run_dir / "metrics.json"),
                "hdf5_sha256": _sha256(
                    run_dir / "outputs" / output["hdf5_filename"]
                ),
                "figure_count": len(figure_paths),
            },
        )
    except Exception as error:
        save_json(
            state_path,
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return run_dir


_SPECTRAL_OPERATOR_CONFIG_SECTIONS = (
    "provenance",
    "optics",
    "illumination",
    "sample_a",
    "probe_grid",
    "sample_b",
    "scan",
    "detector",
    "loss",
)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left) - np.asarray(right)
    denominator = float(np.linalg.norm(np.asarray(right).ravel()))
    return float(
        np.linalg.norm(difference.ravel())
        / max(denominator, np.finfo(np.float64).eps)
    )


def _load_spectral_source(
    config: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    settings = config["verification"]["local_spectral_diagnostic"]
    source_run = Path(settings["source_run"])
    if not source_run.is_absolute():
        source_run = PROJECT_ROOT / source_run
    source_run = source_run.resolve()
    source_config_path = source_run / "config.yaml"
    source_metrics_path = source_run / "metrics.json"
    source_hdf5_path = source_run / settings["source_hdf5_relative_path"]
    expected_hashes = {
        source_config_path: settings["source_config_sha256"],
        source_metrics_path: settings["source_metrics_sha256"],
        source_hdf5_path: settings["source_hdf5_sha256"],
    }
    for path, expected_hash in expected_hashes.items():
        if not path.is_file():
            raise RuntimeError(f"Missing registered spectral source artifact: {path}")
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Spectral source hash mismatch: {path}")
    state_path = source_run / "run_state.json"
    if not state_path.is_file():
        raise RuntimeError("The registered spectral source has no run_state.json.")
    with state_path.open("r", encoding="utf-8") as handle:
        source_state = json.load(handle)
    if (
        source_state.get("status") != "complete"
        or source_state.get("artifacts_validated") is not True
    ):
        raise RuntimeError("The registered spectral source is not validated.")
    source_config = load_config(source_config_path)
    for section in _SPECTRAL_OPERATOR_CONFIG_SECTIONS:
        if source_config.get(section) != config.get(section):
            raise RuntimeError(
                f"Spectral source operator config differs at {section}."
            )
    with source_metrics_path.open("r", encoding="utf-8") as handle:
        source_metrics = json.load(handle)
    if (
        source_metrics.get("truth_used_by_optimizer") is not False
        or source_metrics.get("scientific_pass_fail_conclusion") is not False
    ):
        raise RuntimeError("The spectral source violates the evidence boundary.")
    primary_path = settings["source_primary_probe_hdf5_path"]
    control_path = settings["source_control_probe_hdf5_path"]
    with h5py.File(source_hdf5_path, "r") as h5:
        required = {
            "entry/data/I_stack",
            "entry/data/scan_positions",
            primary_path,
            control_path,
        }
        missing = sorted(path for path in required if path not in h5)
        if missing:
            raise RuntimeError(f"Spectral source HDF5 paths are missing: {missing}")
        measured = np.asarray(h5["entry/data/I_stack"][...], dtype=np.float64)
        scan_positions = np.asarray(
            h5["entry/data/scan_positions"][...], dtype=np.float64
        )
        primary = np.asarray(h5[primary_path][...], dtype=np.complex128)
        control = np.asarray(h5[control_path][...], dtype=np.complex128)
    operator = case["operator"]
    if not isinstance(operator, MatchedKnownBProbeOperator):
        raise TypeError("case operator has the wrong type.")
    if primary.shape != operator.native_shape or control.shape != operator.native_shape:
        raise RuntimeError("Spectral source probes have the wrong shape.")
    if not all(
        np.all(np.isfinite(values))
        for values in (measured, scan_positions, primary, control)
    ):
        raise RuntimeError("Spectral source arrays must all be finite.")
    data_exact = bool(np.array_equal(measured, case["I_stack"]))
    scan_exact = bool(np.array_equal(scan_positions, case["scan_positions"]))
    if not data_exact or not scan_exact:
        raise RuntimeError("Registered source data do not exactly replay the operator.")
    return {
        "source_run": source_run,
        "source_config_path": source_config_path,
        "source_metrics_path": source_metrics_path,
        "source_hdf5_path": source_hdf5_path,
        "I_stack": measured,
        "scan_positions": scan_positions,
        "P_B_primary_raw": primary,
        "P_B_control_raw": control,
        "data_exact_replay": data_exact,
        "scan_positions_exact_replay": scan_exact,
        "data_replay_relative_l2": _relative_l2(case["I_stack"], measured),
        "scan_positions_replay_relative_l2": _relative_l2(
            case["scan_positions"], scan_positions
        ),
    }


def _load_spectral_reference(
    config: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    spectral = config["verification"]["local_spectral_diagnostic"]
    settings = spectral["dimension_convergence_control"]
    reference_run = Path(settings["reference_run"])
    if not reference_run.is_absolute():
        reference_run = PROJECT_ROOT / reference_run
    reference_run = reference_run.resolve()
    config_path = reference_run / "config.yaml"
    metrics_path = reference_run / "metrics.json"
    hdf5_path = reference_run / settings["reference_hdf5_relative_path"]
    expected_hashes = {
        config_path: settings["reference_config_sha256"],
        metrics_path: settings["reference_metrics_sha256"],
        hdf5_path: settings["reference_hdf5_sha256"],
    }
    for path, expected_hash in expected_hashes.items():
        if not path.is_file():
            raise RuntimeError(
                f"Missing registered dimension reference artifact: {path}"
            )
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Dimension reference hash mismatch: {path}")
    state_path = reference_run / "run_state.json"
    if not state_path.is_file():
        raise RuntimeError("The dimension reference has no run_state.json.")
    with state_path.open("r", encoding="utf-8") as handle:
        reference_state = json.load(handle)
    if (
        reference_state.get("status") != "complete"
        or reference_state.get("artifacts_validated") is not True
    ):
        raise RuntimeError("The dimension reference is not validated.")
    reference_config = load_config(config_path)
    for section in _SPECTRAL_OPERATOR_CONFIG_SECTIONS:
        if reference_config.get(section) != config.get(section):
            raise RuntimeError(
                f"Dimension reference operator config differs at {section}."
            )
    reference_spectral = reference_config["verification"][
        "local_spectral_diagnostic"
    ]
    reused_keys = (
        "source_run",
        "source_config_sha256",
        "source_metrics_sha256",
        "source_hdf5_relative_path",
        "source_hdf5_sha256",
        "source_primary_probe_hdf5_path",
        "source_control_probe_hdf5_path",
        "base_probe",
        "direction",
        "matrix_operator",
        "matrix_normalization",
        "start_vectors",
        "random_seed",
        "reorthogonalization",
        "reorthogonalization_passes",
        "breakdown_relative_tolerance",
        "reported_low_ritz_count",
    )
    for key in reused_keys:
        if reference_spectral.get(key) != spectral.get(key):
            raise RuntimeError(f"Dimension reference differs at spectral.{key}.")
    if int(reference_spectral["krylov_dimension"]) != int(
        settings["reference_krylov_dimension"]
    ):
        raise RuntimeError("The registered reference dimension is inconsistent.")
    with metrics_path.open("r", encoding="utf-8") as handle:
        reference_metrics = json.load(handle)
    reference_diagnostic = reference_metrics.get(
        "local_spectral_diagnostic"
    )
    if not isinstance(reference_diagnostic, dict):
        raise RuntimeError("The reference spectral metrics are missing.")
    if reference_diagnostic.get("truth_used_by_diagnostic") is not False:
        raise RuntimeError("The reference spectral metrics used truth.")
    root = "entry/reconstruction/local_spectral_diagnostic"
    with h5py.File(hdf5_path, "r") as h5:
        required = {
            f"{root}/P_B_primary_probe_raw",
            f"{root}/P_B_control_probe_raw",
            f"{root}/pairwise_start/ritz_values",
            f"{root}/random_start/ritz_values",
        }
        missing = sorted(path for path in required if path not in h5)
        if missing:
            raise RuntimeError(
                f"Dimension reference HDF5 paths are missing: {missing}"
            )
        primary_exact = bool(
            np.array_equal(
                h5[f"{root}/P_B_primary_probe_raw"][...],
                source["P_B_primary_raw"],
            )
        )
        control_exact = bool(
            np.array_equal(
                h5[f"{root}/P_B_control_probe_raw"][...],
                source["P_B_control_raw"],
            )
        )
        pairwise_ritz_exact = bool(
            np.array_equal(
                h5[f"{root}/pairwise_start/ritz_values"][...],
                np.asarray(
                    reference_diagnostic["pairwise_start"]["ritz_values"],
                    dtype=np.float64,
                ),
            )
        )
        random_ritz_exact = bool(
            np.array_equal(
                h5[f"{root}/random_start/ritz_values"][...],
                np.asarray(
                    reference_diagnostic["random_start"]["ritz_values"],
                    dtype=np.float64,
                ),
            )
        )
    if not all(
        (primary_exact, control_exact, pairwise_ritz_exact, random_ritz_exact)
    ):
        raise RuntimeError("Dimension reference HDF5 does not match its metrics.")
    return {
        "reference_run": reference_run,
        "reference_config_path": config_path,
        "reference_metrics_path": metrics_path,
        "reference_hdf5_path": hdf5_path,
        "diagnostic": reference_diagnostic,
        "operator_consistency": reference_metrics["operator_consistency"],
        "source_primary_probe_exact": primary_exact,
        "source_control_probe_exact": control_exact,
        "pairwise_ritz_metrics_hdf5_exact": pairwise_ritz_exact,
        "random_ritz_metrics_hdf5_exact": random_ritz_exact,
    }


def _spectral_metadata(
    config: dict[str, Any],
    config_path: Path,
    source: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    provenance = config["provenance"]
    return {
        "experiment_id": "exp042",
        "created_at_utc": created_at_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_config": str(config_path),
        "development_status": (
            "Development baseline / No scientific pass-fail conclusion"
        ),
        "run_role": "local_spectral_diagnostic_from_prior_run",
        "operator_branch": provenance["source_branch"],
        "source_run": str(source["source_run"]),
        "source_hdf5": str(source["source_hdf5_path"]),
        "dimension_reference_run": str(reference["reference_run"]),
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "known_sample_b": True,
        "probe_only_reconstruction": True,
        "reconstruction_performed_in_this_run": False,
        "paired_continuation_performed_in_this_run": False,
        "matrix_free_local_spectral_diagnostic": True,
        "bounded_krylov_dimension_extension": True,
        "truth_used_by_diagnostic": False,
        "scientific_claim_boundary": (
            "bounded local Ritz evidence under the frozen matched exp040 "
            "scalar working model only"
        ),
    }


def _spectral_hdf5_payload(
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    case: dict[str, Any],
    source: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    operator = case["operator"]
    if not isinstance(operator, MatchedKnownBProbeOperator):
        raise TypeError("case operator has the wrong type.")
    settings = config["verification"]["local_spectral_diagnostic"]
    return {
        "I_stack": source["I_stack"],
        "scan_positions": source["scan_positions"],
        "instrument": {
            "wavelength_m": config["optics"]["wavelength_m"],
            "internal_reference_index": config["optics"][
                "internal_reference_index"
            ],
            "external_medium_index": config["optics"][
                "external_medium_index"
            ],
            "z_AB_m": config["optics"]["z_AB_m"],
            "z_BC_m": config["optics"]["z_BC_m"],
            "probe_grid": {
                "plane": "B",
                "axis_order": ["y", "x"],
                "native_shape": operator.native_shape,
                "open_shape": operator.open_shape,
                "node_dx_m": operator.node_dx_m,
            },
            "detector": {
                "model": config["detector"]["model"],
                "quadrature_factor": operator.quadrature_factor,
                "pixel_size_m": config["detector"]["pixel_size_m"],
                "native_roi_shape": operator.detector_roi_shape,
            },
        },
        "sample": {
            "sample_a": dict(config["sample_a"]),
            "sample_b": dict(config["sample_b"]),
            "B_known": case["B_true"],
        },
        "reconstruction": {
            "local_spectral_diagnostic": {
                "source_run": str(settings["source_run"]),
                "source_hdf5_relative_path": settings[
                    "source_hdf5_relative_path"
                ],
                "source_primary_probe_hdf5_path": settings[
                    "source_primary_probe_hdf5_path"
                ],
                "source_control_probe_hdf5_path": settings[
                    "source_control_probe_hdf5_path"
                ],
                "dimension_reference_run": settings[
                    "dimension_convergence_control"
                ]["reference_run"],
                "dimension_reference_hdf5_relative_path": settings[
                    "dimension_convergence_control"
                ]["reference_hdf5_relative_path"],
                **diagnostic,
            }
        },
        "config_yaml": config_to_yaml(config),
        "metadata": metadata,
        "metrics": metrics,
    }


def _save_spectral_figure(
    run_dir: Path, diagnostic: dict[str, Any], filename: str
) -> Path:
    path = run_dir / "figures" / filename
    pairwise = diagnostic["pairwise_start"]
    random = diagnostic["random_start"]
    pairwise_values = np.asarray(pairwise["ritz_values"], dtype=np.float64)
    random_values = np.asarray(random["ritz_values"], dtype=np.float64)
    pairwise_gains = np.asarray(
        pairwise["ritz_singular_gain_estimates"], dtype=np.float64
    )
    random_gains = np.asarray(
        random["ritz_singular_gain_estimates"], dtype=np.float64
    )
    pairwise_weights = np.asarray(
        pairwise["spectral_weights"], dtype=np.float64
    )
    random_weights = np.asarray(random["spectral_weights"], dtype=np.float64)
    pairwise_residuals = np.asarray(
        pairwise["ritz_residual_l2_estimates"], dtype=np.float64
    )
    random_residuals = np.asarray(
        random["ritz_residual_l2_estimates"], dtype=np.float64
    )
    convergence = diagnostic["dimension_convergence_control"]
    pairwise_convergence = convergence["pairwise_start"]
    random_convergence = convergence["random_start"]
    convergence_labels = ["pair 12", "pair 24", "random 12", "random 24"]
    lowest_values = np.asarray(
        [
            pairwise_convergence["reference_lowest_ritz_value"],
            pairwise_convergence["extended_lowest_ritz_value"],
            random_convergence["reference_lowest_ritz_value"],
            random_convergence["extended_lowest_ritz_value"],
        ],
        dtype=np.float64,
    )
    residual_to_value = np.asarray(
        [
            pairwise_convergence["reference_lowest_ritz_residual_to_value"],
            pairwise_convergence["extended_lowest_ritz_residual_to_value"],
            random_convergence["reference_lowest_ritz_residual_to_value"],
            random_convergence["extended_lowest_ritz_residual_to_value"],
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
    axes[0, 0].semilogy(
        np.maximum(pairwise_gains, np.finfo(float).tiny),
        marker="o",
        label="pairwise start",
    )
    axes[0, 0].semilogy(
        np.maximum(random_gains, np.finfo(float).tiny),
        marker="s",
        label="random start",
    )
    axes[0, 0].set(
        title="Projected Ritz gain estimates",
        xlabel="Ascending Ritz index",
        ylabel="sqrt(max(Ritz value, 0))",
    )
    axes[0, 1].plot(pairwise_weights, marker="o", label="pairwise start")
    axes[0, 1].plot(random_weights, marker="s", label="random start")
    axes[0, 1].set(
        title="Start-vector spectral weights",
        xlabel="Ascending Ritz index",
        ylabel="Squared real overlap",
    )
    axes[1, 0].semilogy(
        np.maximum(pairwise_residuals, np.finfo(float).tiny),
        marker="o",
        label="pairwise start",
    )
    axes[1, 0].semilogy(
        np.maximum(random_residuals, np.finfo(float).tiny),
        marker="s",
        label="random start",
    )
    axes[1, 0].set(
        title="Ritz residual estimates",
        xlabel="Ascending Ritz index",
        ylabel="Residual L2 estimate",
    )
    axes[1, 1].semilogx(
        np.maximum(pairwise_values, np.finfo(float).tiny),
        np.cumsum(pairwise_weights),
        marker="o",
        label="pairwise start",
    )
    axes[1, 1].semilogx(
        np.maximum(random_values, np.finfo(float).tiny),
        np.cumsum(random_weights),
        marker="s",
        label="random start",
    )
    axes[1, 1].set(
        title="Cumulative projected spectral weight",
        xlabel="Ritz value of J^T J / N_detector",
        ylabel="Cumulative weight",
    )
    axes[0, 2].bar(
        convergence_labels,
        np.maximum(lowest_values, np.finfo(float).tiny),
        color=["tab:blue", "tab:cyan", "tab:orange", "tab:red"],
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set(
        title="Lowest Ritz value: 12 to 24",
        ylabel="Lowest projected eigenvalue",
    )
    axes[1, 2].bar(
        convergence_labels,
        np.maximum(residual_to_value, np.finfo(float).tiny),
        color=["tab:blue", "tab:cyan", "tab:orange", "tab:red"],
    )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set(
        title="Lowest Ritz residual / value",
        ylabel="Dimensionless ratio",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    fig.suptitle(
        "Truth-free matrix-free local spectral dimension convergence"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _validate_spectral_artifacts(
    run_dir: Path, config: dict[str, Any], native_shape: tuple[int, int]
) -> None:
    output = config["output"]
    figure_path = run_dir / "figures" / output["spectral_figure_filename"]
    hdf5_path = run_dir / "outputs" / output["hdf5_filename"]
    required = {
        run_dir / "config.yaml",
        run_dir / "metadata.json",
        run_dir / "metrics.json",
        run_dir / "run_state.json",
        hdf5_path,
        figure_path,
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"Missing exp042 spectral artifacts: {missing}")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    diagnostic_metrics = metrics.get("local_spectral_diagnostic", {})
    if diagnostic_metrics.get("truth_used_by_diagnostic") is not False:
        raise RuntimeError("Spectral metrics have an invalid truth-use flag.")
    convergence_metrics = diagnostic_metrics.get(
        "dimension_convergence_control", {}
    )
    if (
        convergence_metrics.get("truth_used_by_control") is not False
        or convergence_metrics.get("single_extension_only") is not True
    ):
        raise RuntimeError("Dimension convergence metrics are invalid.")
    with h5py.File(hdf5_path, "r") as h5:
        expected_entry = {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "reconstruction",
            "sample",
        }
        if set(h5["entry"]) != expected_entry:
            raise RuntimeError("Unexpected exp042 spectral /entry layout.")
        root = "entry/reconstruction/local_spectral_diagnostic"
        for name in (
            "P_B_primary_probe_raw",
            "P_B_control_probe_raw",
            "P_B_pairwise_direction_unit_l2",
            "P_B_random_start_direction_unit_l2",
        ):
            if f"{root}/{name}" not in h5 or h5[f"{root}/{name}"].shape != native_shape:
                raise RuntimeError(f"Unexpected spectral HDF5 field: {name}")
        convergence_path = f"{root}/dimension_convergence_control"
        if convergence_path not in h5:
            raise RuntimeError("Dimension convergence HDF5 group is missing.")
        if not _all_numeric_hdf5_finite(h5["entry"]):
            raise RuntimeError("Spectral HDF5 contains non-finite numeric data.")
    image = plt.imread(figure_path)
    if image.ndim not in {2, 3} or not np.all(np.isfinite(image)):
        raise RuntimeError("Unreadable or non-finite spectral figure.")


def _run_local_spectral_diagnostic(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    validate_exp042_config(config)
    output = config["output"]
    output_root = Path(output["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = make_run_dir(output_root, str(output["run_name"]))
    save_config(run_dir / "config.yaml", config)
    state_path = run_dir / "run_state.json"
    save_json(
        state_path,
        {
            "status": "running",
            "artifacts_validated": False,
            "started_at_utc": created_at_utc(),
        },
    )
    started = time.perf_counter()
    try:
        case = build_matched_development_case(config)
        source = _load_spectral_source(config, case)
        reference = _load_spectral_reference(config, source)
        operator = case["operator"]
        if not isinstance(operator, MatchedKnownBProbeOperator):
            raise TypeError("case operator has the wrong type.")
        settings = config["verification"]["local_spectral_diagnostic"]
        diagnostic = truth_free_local_spectral_diagnostic(
            operator,
            source["P_B_primary_raw"],
            source["P_B_control_raw"],
            krylov_dimension=int(settings["krylov_dimension"]),
            reorthogonalization_passes=int(
                settings["reorthogonalization_passes"]
            ),
            breakdown_relative_tolerance=float(
                settings["breakdown_relative_tolerance"]
            ),
            random_seed=int(settings["random_seed"]),
            reported_low_ritz_count=int(settings["reported_low_ritz_count"]),
        )
        convergence_settings = settings["dimension_convergence_control"]
        diagnostic["dimension_convergence_control"] = (
            lanczos_dimension_convergence_diagnostic(
                reference["diagnostic"],
                diagnostic,
                reference_krylov_dimension=int(
                    convergence_settings["reference_krylov_dimension"]
                ),
                reported_low_ritz_count=int(
                    settings["reported_low_ritz_count"]
                ),
            )
        )
        elapsed = time.perf_counter() - started
        diagnostic_metrics = {
            key: value
            for key, value in diagnostic.items()
            if not key.startswith("P_B_")
        }
        metrics = {
            "development_status": (
                "Development baseline / No scientific pass-fail conclusion"
            ),
            "run_role": "local_spectral_diagnostic_from_prior_run",
            "source_artifact": {
                "source_run": str(settings["source_run"]),
                "source_config_sha256": settings["source_config_sha256"],
                "source_metrics_sha256": settings["source_metrics_sha256"],
                "source_hdf5_sha256": settings["source_hdf5_sha256"],
                "operator_config_sections_exact": True,
                "data_exact_replay": source["data_exact_replay"],
                "scan_positions_exact_replay": source[
                    "scan_positions_exact_replay"
                ],
                "data_replay_relative_l2": source[
                    "data_replay_relative_l2"
                ],
                "scan_positions_replay_relative_l2": source[
                    "scan_positions_replay_relative_l2"
                ],
            },
            "dimension_reference_artifact": {
                "reference_run": str(convergence_settings["reference_run"]),
                "reference_config_sha256": convergence_settings[
                    "reference_config_sha256"
                ],
                "reference_metrics_sha256": convergence_settings[
                    "reference_metrics_sha256"
                ],
                "reference_hdf5_sha256": convergence_settings[
                    "reference_hdf5_sha256"
                ],
                "source_primary_probe_exact": reference[
                    "source_primary_probe_exact"
                ],
                "source_control_probe_exact": reference[
                    "source_control_probe_exact"
                ],
                "pairwise_ritz_metrics_hdf5_exact": reference[
                    "pairwise_ritz_metrics_hdf5_exact"
                ],
                "random_ritz_metrics_hdf5_exact": reference[
                    "random_ritz_metrics_hdf5_exact"
                ],
            },
            "operator_consistency": {
                "intensity_jacobian_real_adjoint_relative_error": (
                    reference["operator_consistency"][
                        "intensity_jacobian_real_adjoint_relative_error"
                    ]
                ),
                "intensity_jacobian_real_adjoint_rerun": False,
                "normal_operator_definition": (
                    "real_intensity_jacobian_transpose_times_jacobian"
                ),
                "normal_operator_normalization": "mean_over_detector_stack",
                "inherited_forward_controls_rerun": False,
            },
            "local_spectral_diagnostic": diagnostic_metrics,
            "runtime_seconds": elapsed,
            "known_sample_b": True,
            "sample_b_updated": False,
            "reconstruction_performed_in_this_run": False,
            "paired_continuation_performed_in_this_run": False,
            "truth_used_by_diagnostic": False,
            "waist_estimation_performed": False,
            "scientific_pass_fail_conclusion": False,
        }
        metadata = _spectral_metadata(config, config_path, source, reference)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_ptycho_hdf5(
            run_dir / "outputs" / output["hdf5_filename"],
            **_spectral_hdf5_payload(
                config, metadata, metrics, case, source, diagnostic
            ),
        )
        figure_path = _save_spectral_figure(
            run_dir, diagnostic, str(output["spectral_figure_filename"])
        )
        _validate_spectral_artifacts(run_dir, config, operator.native_shape)
        save_json(
            state_path,
            {
                "status": "complete",
                "artifacts_validated": True,
                "completed_at_utc": created_at_utc(),
                "runtime_seconds": elapsed,
                "config_sha256": _sha256(run_dir / "config.yaml"),
                "metrics_sha256": _sha256(run_dir / "metrics.json"),
                "hdf5_sha256": _sha256(
                    run_dir / "outputs" / output["hdf5_filename"]
                ),
                "figure_count": 1,
                "figure_sha256": _sha256(figure_path),
            },
        )
    except Exception as error:
        save_json(
            state_path,
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return run_dir


def run(config_path: Path) -> Path:
    """Dispatch the current registered exp042 execution mode."""

    config = load_config(config_path.resolve())
    mode = config.get("execution", {}).get("mode")
    if mode in {
        "initialization_ablation_matched",
        "initialization_magnitude_ablation_matched",
    }:
        return _run_initialization_ablation(config_path)
    if mode == "detector_quadrature_three_branch_control":
        return _run_detector_quadrature_ablation(config_path)
    if mode == "local_spectral_diagnostic_from_prior_run":
        return _run_local_spectral_diagnostic(config_path)
    return _run_full_baseline(config_path)


def main() -> None:
    args = _parse_args()
    run_dir = run(args.config)
    print(f"run_dir: {run_dir}")
    print(
        "development_status: "
        "Development baseline / No scientific pass-fail conclusion"
    )
    print("artifacts_validated: true")


if __name__ == "__main__":
    main()
