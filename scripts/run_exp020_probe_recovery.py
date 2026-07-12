"""Run exp020: blind ePIE probe/B recovery and backpropagation to sample A."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.scan import add_integer_pixel_jitter, make_grid_scan
from tgv_ptycho.forward.scheme_probe_B import simulate_probe_B_forward
from tgv_ptycho.inverse.backprop_A import recover_thin_phase_A
from tgv_ptycho.inverse.metrics import (
    align_affine_phase_and_complex_gain,
    amplitude_rmse,
    complex_relative_error,
    phase_rmse,
)
from tgv_ptycho.io.config import config_to_yaml, load_config, save_config
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit
from tgv_ptycho.io.naming import make_run_dir
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5
from tgv_ptycho.objects.sample_a import make_smooth_random_thin_phase
from tgv_ptycho.objects.sample_b import make_random_phase_object
from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.fields import make_plane_wave
from tgv_ptycho.recon.epie import epie_reconstruct
from tgv_ptycho.recon.initialization import (
    initialize_probe_by_detector_backpropagation,
)
from tgv_ptycho.viz.plot_field import plot_complex_field
from tgv_ptycho.viz.plot_recon import (
    plot_loss_curve,
    save_diffraction_montage,
    save_reconstruction_comparison,
    save_scan_positions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _shape(config: dict[str, Any]) -> tuple[int, int]:
    shape_cfg = config.get("optics", {}).get("shape", [96, 96])
    if not isinstance(shape_cfg, (list, tuple)) or len(shape_cfg) != 2:
        msg = "optics.shape must contain [ny, nx]."
        raise ValueError(msg)
    return int(shape_cfg[0]), int(shape_cfg[1])


def _make_scan(config: dict[str, Any], dx_m: float) -> tuple[np.ndarray, np.ndarray]:
    scan_cfg = config.get("scan", {})
    if scan_cfg.get("type", "jittered_grid") != "jittered_grid":
        msg = "exp020 supports only jittered_grid scans."
        raise ValueError(msg)
    regular = make_grid_scan(
        int(scan_cfg["num_x"]),
        int(scan_cfg["num_y"]),
        float(scan_cfg["step_m"]),
        center=bool(scan_cfg.get("center", True)),
    )
    jittered = add_integer_pixel_jitter(
        regular,
        dx_m,
        int(scan_cfg.get("max_jitter_px", 0)),
        seed=int(scan_cfg["jitter_seed"]),
    )
    if len(np.unique(jittered, axis=0)) != len(jittered):
        msg = "Jittered scan contains duplicate positions; adjust scan parameters."
        raise ValueError(msg)
    return regular, jittered


def _make_sample_b(config: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    sample_cfg = config.get("sample_b", {})
    if sample_cfg.get("type", "random_phase") != "random_phase":
        msg = "exp020 energy normalization currently requires phase-only sample B."
        raise ValueError(msg)
    return make_random_phase_object(
        shape,
        phase_range=float(sample_cfg["phase_range_rad"]),
        seed=int(sample_cfg["seed"]),
        feature_size_px=int(sample_cfg.get("feature_size_px", 1)),
    )


def _initial_object(
    shape: tuple[int, int], phase_std_rad: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase = rng.normal(scale=phase_std_rad, size=shape)
    return np.exp(1j * phase).astype(np.complex128)


def run(config_path: Path) -> Path:
    """Execute exp020 simulation, constrained blind ePIE, and persistence."""

    config = load_config(config_path)
    run_cfg = config.get("run", {})
    optics_cfg = config.get("optics", {})
    sample_a_cfg = config.get("sample_a", {})
    recon_cfg = config.get("reconstruction", {})
    output_cfg = config.get("output", {})

    run_name = str(run_cfg.get("name", config_path.stem))
    output_root = PROJECT_ROOT / str(run_cfg.get("output_root", "runs"))
    run_dir = make_run_dir(output_root, run_name)
    figures_dir = run_dir / "figures"
    outputs_dir = run_dir / "outputs"

    shape = _shape(config)
    dx_m = float(optics_cfg["dx_m"])
    wavelength_m = float(optics_cfg["wavelength_m"])
    z_AB_m = float(optics_cfg["z_AB_m"])
    z_BC_m = float(optics_cfg["z_BC_m"])
    detector_pixel_size_m = float(optics_cfg.get("detector_pixel_size_m", dx_m))
    medium_index = float(optics_cfg.get("medium_index", 1.0))
    if medium_index != 1.0:
        msg = "exp020 currently requires homogeneous propagation with n=1.0."
        raise ValueError(msg)
    if detector_pixel_size_m != dx_m:
        msg = "exp020 currently requires detector_pixel_size_m == dx_m."
        raise ValueError(msg)

    incident_field = make_plane_wave(
        shape,
        dx_m,
        wavelength_m,
        amplitude=float(config.get("illumination", {}).get("amplitude", 1.0)),
    )
    A_true, A_phase_true, A_support_mask = make_smooth_random_thin_phase(
        shape,
        dx_m,
        radius=float(sample_a_cfg["radius_m"]),
        phase_rms=float(sample_a_cfg["phase_rms_rad"]),
        correlation_length=float(sample_a_cfg["correlation_length_m"]),
        seed=int(sample_a_cfg["seed"]),
    )
    A_reference_mask = ~A_support_mask
    B_true = _make_sample_b(config, shape)
    _regular_positions, scan_positions = _make_scan(config, dx_m)

    I_stack, P_B_true, _, forward_metadata = simulate_probe_B_forward(
        A_true,
        B_true,
        scan_positions,
        dx_m,
        wavelength_m,
        z_AB=z_AB_m,
        z_BC=z_BC_m,
        incident_field=incident_field,
        noise_config=config.get("noise"),
    )
    P_B_init = initialize_probe_by_detector_backpropagation(
        I_stack, dx_m, wavelength_m, z_BC_m
    )
    B_init = _initial_object(
        shape,
        float(recon_cfg.get("initial_object_phase_std_rad", 0.02)),
        int(recon_cfg["initial_object_seed"]),
    )
    probe_l2_norm_target = float(
        np.sqrt(np.mean(np.sum(I_stack, axis=(1, 2))))
    )

    def thin_phase_A_probe_constraint(probe: np.ndarray) -> np.ndarray:
        recovered = recover_thin_phase_A(
            probe,
            incident_field,
            A_reference_mask,
            dx_m,
            wavelength_m,
            z_AB_m,
        )
        A_phase_only = np.asarray(recovered["A_rec_phase_only"])
        return angular_spectrum_propagate(
            A_phase_only * incident_field,
            dx_m,
            wavelength_m,
            z_AB_m,
        )

    result = epie_reconstruct(
        I_stack,
        scan_positions,
        dx=dx_m,
        wavelength=wavelength_m,
        z_BC=z_BC_m,
        num_iters=int(recon_cfg["num_iters"]),
        beta_probe=float(recon_cfg["beta_probe"]),
        beta_object=float(recon_cfg["beta_object"]),
        init_probe=P_B_init,
        init_object=B_init,
        update_probe=True,
        shuffle_positions=bool(recon_cfg.get("shuffle_positions", True)),
        seed=int(recon_cfg["shuffle_seed"]),
        object_amplitude_bounds=(1.0, 1.0),
        probe_l2_norm_target=probe_l2_norm_target,
        probe_constraint=thin_phase_A_probe_constraint,
        show_progress=bool(recon_cfg.get("show_progress", True)),
    )
    P_B_rec = np.asarray(result["P_B_rec"], dtype=np.complex128)
    B_rec = np.asarray(result["B_rec"], dtype=np.complex128)
    loss_curve = np.asarray(result["loss_curve"], dtype=np.float64)
    illumination_map = np.asarray(result["illumination_map"], dtype=np.float64)

    A_result = recover_thin_phase_A(
        P_B_rec,
        incident_field,
        A_reference_mask,
        dx_m,
        wavelength_m,
        z_AB_m,
    )
    A_rec_raw = np.asarray(A_result["A_rec_raw"], dtype=np.complex128)
    A_rec_reference = np.asarray(
        A_result["A_rec_reference_corrected"], dtype=np.complex128
    )
    A_rec_phase_only = np.asarray(A_result["A_rec_phase_only"], dtype=np.complex128)

    threshold = float(
        config.get("metrics", {}).get("illumination_threshold_fraction", 0.05)
    )
    illuminated_mask = illumination_map >= threshold * float(np.max(illumination_map))
    P_B_eval, P_gain, P_ramp = align_affine_phase_and_complex_gain(
        P_B_rec, P_B_true
    )
    B_eval, B_gain, B_ramp = align_affine_phase_and_complex_gain(
        B_rec, B_true, illuminated_mask
    )
    A_eval, A_gain, A_ramp = align_affine_phase_and_complex_gain(
        A_rec_phase_only, A_true
    )

    initial_loss = float(result["initial_data_fidelity_loss"])
    final_loss = float(result["final_data_fidelity_loss"])
    metrics = {
        "num_scan_positions": int(len(scan_positions)),
        "num_iterations": int(len(loss_curve)),
        "initial_data_fidelity_loss": initial_loss,
        "final_data_fidelity_loss": final_loss,
        "first_iteration_sequential_loss": float(loss_curve[0]),
        "last_iteration_sequential_loss": float(loss_curve[-1]),
        "minimum_sequential_loss": float(np.min(loss_curve)),
        "loss_reduction_factor": initial_loss / max(final_loss, np.finfo(float).eps),
        "P_B_complex_relative_error_simulation_evaluation_only": (
            complex_relative_error(P_B_eval, P_B_true)
        ),
        "B_complex_relative_error_illuminated_simulation_evaluation_only": (
            complex_relative_error(B_eval, B_true, illuminated_mask)
        ),
        "B_wrapped_phase_rmse_rad_illuminated_simulation_evaluation_only": (
            phase_rmse(np.angle(B_eval), np.angle(B_true), illuminated_mask)
        ),
        "A_phase_only_complex_relative_error_truth_free_reference_correction": (
            complex_relative_error(A_rec_phase_only, A_true)
        ),
        "A_wrapped_phase_rmse_rad_active_truth_free_reference_correction": (
            phase_rmse(
                np.angle(A_rec_phase_only), np.angle(A_true), A_support_mask
            )
        ),
        "A_wrapped_phase_rmse_rad_blank_reference": phase_rmse(
            np.angle(A_rec_phase_only), np.angle(A_true), A_reference_mask
        ),
        "A_complex_relative_error_simulation_evaluation_only": (
            complex_relative_error(A_eval, A_true)
        ),
        "A_reference_corrected_amplitude_rmse": amplitude_rmse(
            A_rec_reference, A_true
        ),
        "probe_l2_norm_target_from_measurements": probe_l2_norm_target,
        "probe_l2_norm_final": float(np.sqrt(np.sum(np.abs(P_B_rec) ** 2))),
        "illuminated_pixel_fraction": float(np.mean(illuminated_mask)),
        "A_active_pixel_fraction": float(np.mean(A_support_mask)),
        "scan_max_jitter_px": int(config.get("scan", {}).get("max_jitter_px", 0)),
        "I_stack_min": float(np.min(I_stack)),
        "I_stack_max": float(np.max(I_stack)),
        "I_stack_mean": float(np.mean(I_stack)),
    }
    evaluation_alignment = {
        "P_B_complex_gain": P_gain,
        "P_B_phase_ramp_yx_rad_per_px": P_ramp,
        "B_complex_gain": B_gain,
        "B_phase_ramp_yx_rad_per_px": B_ramp,
        "A_complex_gain": A_gain,
        "A_phase_ramp_yx_rad_per_px": A_ramp,
    }
    metadata = {
        "run_name": run_name,
        "experiment": "exp020_A_thin_phase_probe_recovery",
        "phase": "Phase 2",
        "dataset_type": "simulation",
        "created_at": created_at_utc(),
        "git_commit": get_git_commit(PROJECT_ROOT) or "",
        "config_path": str(config_path),
        "shape_ny_nx": list(shape),
        "algorithm": "constrained_blind_epie_with_A_plane_projection",
        "base_algorithm": result["metadata"]["algorithm"],
        "known_probe": False,
        "probe_updated": True,
        "A_constraint_uses_truth": False,
        "A_constraint_prior": "pure_phase_with_known_blank_reference",
        "algorithm_reference_doi": "10.1016/j.ultramic.2009.05.012",
        "scan_coordinate_order": "x_y",
        "scan_position_unit": "m",
        "dx_tuple_order_if_used": "dy_dx",
        "integer_pixel_shifts_only": True,
        "periodic_object_boundary": True,
    }

    if bool(output_cfg.get("save_png", True)):
        plot_complex_field(
            A_true,
            figures_dir / "A_true_amp_phase.png",
            title="Sample A truth",
            dx=dx_m,
        )
        plot_complex_field(
            P_B_true,
            figures_dir / "P_B_true_amp_phase.png",
            title="B-plane probe truth",
            dx=dx_m,
        )
        plot_complex_field(
            P_B_rec,
            figures_dir / "P_B_rec_raw_amp_phase.png",
            title="Raw recovered probe",
            dx=dx_m,
        )
        save_reconstruction_comparison(
            A_true,
            A_rec_phase_only,
            figures_dir / "A_truth_reconstruction_error.png",
            dx=dx_m,
            field_label="A",
        )
        save_reconstruction_comparison(
            P_B_true,
            P_B_eval,
            figures_dir / "P_B_truth_reconstruction_error_eval_only.png",
            dx=dx_m,
            field_label="P_B",
        )
        save_reconstruction_comparison(
            B_true,
            B_eval,
            figures_dir / "B_truth_reconstruction_error_eval_only.png",
            dx=dx_m,
            mask=illuminated_mask,
            field_label="B",
        )
        plot_loss_curve(loss_curve, figures_dir / "loss_curve.png")
        save_scan_positions(scan_positions, figures_dir / "scan_positions.png")
        save_diffraction_montage(
            I_stack,
            figures_dir / "detector_frames.png",
            dx=detector_pixel_size_m,
        )

    save_config(run_dir / "config.yaml", config)
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "metrics.json", metrics)
    if bool(output_cfg.get("save_hdf5", True)):
        save_ptycho_hdf5(
            outputs_dir / "exp020_full_pipeline.h5",
            I_stack=I_stack,
            scan_positions=scan_positions,
            instrument={
                "wavelength": wavelength_m,
                "dx": dx_m,
                "z_AB": z_AB_m,
                "z_BC": z_BC_m,
                "detector_pixel_size": detector_pixel_size_m,
                "medium_index": medium_index,
            },
            sample={
                "sample_A_type": "smooth_random_thin_phase",
                "sample_A_parameters": sample_a_cfg,
                "sample_A_support_mask_known_prior": A_support_mask,
                "sample_A_reference_mask_known_prior": A_reference_mask,
                "sample_B_type": "random_phase",
                "sample_B_parameters": config.get("sample_b", {}),
            },
            truth={
                "incident_field_true": incident_field,
                "A_true": A_true,
                "A_phase_true": A_phase_true,
                "P_B_true": P_B_true,
                "B_true": B_true,
            },
            reconstruction={
                "P_B_init": P_B_init,
                "P_B_rec": P_B_rec,
                "B_init": B_init,
                "B_rec": B_rec,
                "field_after_A_rec": A_result["field_after_A_rec"],
                "A_rec_raw": A_rec_raw,
                "A_rec_reference_corrected": A_rec_reference,
                "A_rec_phase_only": A_rec_phase_only,
                "loss_curve": loss_curve,
                "initial_data_fidelity_loss": initial_loss,
                "final_data_fidelity_loss": final_loss,
                "illumination_map": illumination_map,
                "illuminated_mask": illuminated_mask,
                "reference_correction": A_result["reference_correction"],
                "settings": result["metadata"],
                "simulation_evaluation_only": {
                    "P_B_rec_aligned_to_truth": P_B_eval,
                    "B_rec_aligned_to_truth": B_eval,
                    "A_rec_aligned_to_truth": A_eval,
                    "alignment_parameters": evaluation_alignment,
                },
            },
            config_yaml=config_to_yaml(config),
            metadata={**metadata, "forward_model": forward_metadata},
            metrics=metrics,
        )

    print(f"Saved run to: {run_dir}")
    A_phase_rmse = metrics[
        "A_wrapped_phase_rmse_rad_active_truth_free_reference_correction"
    ]
    print(
        "Final loss: "
        f"{final_loss:.6e}; A phase RMSE (truth-free reference correction): "
        f"{A_phase_rmse:.6e} rad"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
