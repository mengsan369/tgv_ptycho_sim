"""Run the exp010 known-probe ePIE simulation and reconstruction."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.scan import make_grid_scan
from tgv_ptycho.forward.scheme_probe_B import simulate_probe_B_forward
from tgv_ptycho.inverse.metrics import (
    align_global_phase,
    amplitude_rmse,
    complex_relative_error,
    phase_rmse,
)
from tgv_ptycho.io.config import config_to_yaml, load_config, save_config
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit
from tgv_ptycho.io.naming import make_run_dir
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5
from tgv_ptycho.objects.sample_b import (
    make_random_amp_phase_object,
    make_random_phase_object,
)
from tgv_ptycho.optics.fields import make_gaussian_field
from tgv_ptycho.recon.epie import epie_reconstruct
from tgv_ptycho.viz.plot_field import plot_complex_field
from tgv_ptycho.viz.plot_recon import (
    plot_loss_curve,
    save_diffraction_montage,
    save_reconstruction_comparison,
    save_scan_positions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the exp010 YAML config.",
    )
    return parser.parse_args()


def _make_probe(
    config: dict[str, Any],
    shape: tuple[int, int],
    dx_m: float,
) -> np.ndarray:
    probe_cfg = config.get("probe", {})
    probe_type = probe_cfg.get("type", "gaussian")
    if probe_type != "gaussian":
        msg = f"Unsupported known probe type: {probe_type}"
        raise ValueError(msg)
    return make_gaussian_field(
        shape,
        dx_m,
        waist=float(probe_cfg["waist_m"]),
        amplitude=float(probe_cfg.get("amplitude", 1.0)),
    )


def _make_sample_b(
    config: dict[str, Any], shape: tuple[int, int], default_seed: int
) -> np.ndarray:
    sample_cfg = config.get("sample_b", {})
    sample_type = sample_cfg.get("type", "random_amp_phase")
    seed = int(sample_cfg.get("seed", default_seed))
    feature_size_px = int(sample_cfg.get("feature_size_px", 1))
    phase_range = float(sample_cfg.get("phase_range_rad", np.pi))
    if sample_type == "random_phase":
        return make_random_phase_object(
            shape,
            phase_range=phase_range,
            seed=seed,
            feature_size_px=feature_size_px,
        )
    if sample_type == "random_amp_phase":
        amp_range_cfg = sample_cfg.get("amp_range", [0.5, 1.0])
        amp_range = (float(amp_range_cfg[0]), float(amp_range_cfg[1]))
        return make_random_amp_phase_object(
            shape,
            amp_range=amp_range,
            phase_range=phase_range,
            seed=seed,
            feature_size_px=feature_size_px,
        )
    msg = f"Unsupported sample B type: {sample_type}"
    raise ValueError(msg)


def _make_positions(config: dict[str, Any]) -> np.ndarray:
    scan_cfg = config.get("scan", {})
    if scan_cfg.get("type", "grid") != "grid":
        msg = "exp010 currently supports only grid scans."
        raise ValueError(msg)
    return make_grid_scan(
        int(scan_cfg["num_x"]),
        int(scan_cfg["num_y"]),
        float(scan_cfg["step_m"]),
        center=bool(scan_cfg.get("center", True)),
    )


def _as_amplitude_bounds(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        msg = "reconstruction.object_amplitude_bounds must have two values."
        raise ValueError(msg)
    return float(value[0]), float(value[1])


def run(config_path: Path) -> Path:
    """Execute forward simulation, known-probe ePIE, metrics, and persistence."""

    config = load_config(config_path)
    run_cfg = config.get("run", {})
    optics_cfg = config.get("optics", {})
    recon_cfg = config.get("reconstruction", {})
    output_cfg = config.get("output", {})
    seed = int(run_cfg.get("seed", 1234))

    run_name = str(run_cfg.get("name", config_path.stem))
    output_root = PROJECT_ROOT / str(run_cfg.get("output_root", "runs"))
    run_dir = make_run_dir(output_root, run_name)
    figures_dir = run_dir / "figures"
    outputs_dir = run_dir / "outputs"

    shape_cfg = optics_cfg.get("shape", [96, 96])
    shape = (int(shape_cfg[0]), int(shape_cfg[1]))
    dx_m = float(optics_cfg["dx_m"])
    wavelength_m = float(optics_cfg["wavelength_m"])
    z_BC_m = float(optics_cfg["z_BC_m"])
    detector_pixel_size_m = float(optics_cfg.get("detector_pixel_size_m", dx_m))
    medium_index = float(optics_cfg.get("medium_index", 1.0))
    if medium_index != 1.0:
        msg = "exp010 forward/reconstruction currently requires medium_index=1.0."
        raise ValueError(msg)

    probe_true = _make_probe(config, shape, dx_m)
    B_true = _make_sample_b(config, shape, seed)
    scan_positions = _make_positions(config)
    I_stack, P_B_true, _, forward_metadata = simulate_probe_B_forward(
        probe_true,
        B_true,
        scan_positions,
        dx_m,
        wavelength_m,
        z_AB=0.0,
        z_BC=z_BC_m,
        incident_field=np.ones(shape, dtype=np.complex128),
        noise_config=config.get("noise"),
    )

    B_init = np.ones(shape, dtype=np.complex128)
    amplitude_bounds = _as_amplitude_bounds(recon_cfg.get("object_amplitude_bounds"))
    result = epie_reconstruct(
        I_stack,
        scan_positions,
        dx=dx_m,
        wavelength=wavelength_m,
        z_BC=z_BC_m,
        num_iters=int(recon_cfg.get("num_iters", 100)),
        beta_probe=float(recon_cfg.get("beta_probe", 0.0)),
        beta_object=float(recon_cfg.get("beta_object", 0.8)),
        init_probe=P_B_true,
        init_object=B_init,
        update_probe=False,
        shuffle_positions=bool(recon_cfg.get("shuffle_positions", True)),
        seed=int(recon_cfg.get("seed", seed)),
        object_amplitude_bounds=amplitude_bounds,
        show_progress=bool(recon_cfg.get("show_progress", True)),
    )
    B_rec = np.asarray(result["B_rec"], dtype=np.complex128)
    P_B_rec = np.asarray(result["P_B_rec"], dtype=np.complex128)
    loss_curve = np.asarray(result["loss_curve"], dtype=np.float64)
    illumination_map = np.asarray(result["illumination_map"], dtype=np.float64)

    threshold_fraction = float(
        config.get("metrics", {}).get("illumination_threshold_fraction", 0.05)
    )
    illumination_mask = illumination_map >= (
        threshold_fraction * float(np.max(illumination_map))
    )
    B_rec_aligned, alignment_phase = align_global_phase(
        B_rec, B_true, illumination_mask
    )
    B_init_aligned, _ = align_global_phase(B_init, B_true, illumination_mask)

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
        "initial_B_complex_relative_error_aligned_illuminated": complex_relative_error(
            B_init_aligned, B_true, illumination_mask
        ),
        "final_B_complex_relative_error_aligned_illuminated": complex_relative_error(
            B_rec_aligned, B_true, illumination_mask
        ),
        "final_B_amplitude_rmse_illuminated": amplitude_rmse(
            B_rec_aligned, B_true, illumination_mask
        ),
        "final_B_wrapped_phase_rmse_rad_illuminated": phase_rmse(
            np.angle(B_rec_aligned), np.angle(B_true), illumination_mask
        ),
        "probe_relative_error": complex_relative_error(P_B_rec, P_B_true),
        "global_phase_alignment_rad_truth_only": alignment_phase,
        "illuminated_pixel_fraction": float(np.mean(illumination_mask)),
        "illumination_threshold_fraction": threshold_fraction,
        "I_stack_min": float(np.min(I_stack)),
        "I_stack_max": float(np.max(I_stack)),
        "I_stack_mean": float(np.mean(I_stack)),
    }

    metadata = {
        "run_name": run_name,
        "experiment": "exp010_epie_known_probe",
        "phase": "Phase 1",
        "dataset_type": "simulation",
        "created_at": created_at_utc(),
        "git_commit": get_git_commit(PROJECT_ROOT) or "",
        "config_path": str(config_path),
        "shape": list(shape),
        "known_probe": True,
        "probe_updated": False,
        "algorithm": result["metadata"]["algorithm"],
        "algorithm_reference_doi": "10.1016/j.ultramic.2009.05.012",
        "scan_coordinate_order": "x_y",
        "scan_position_unit": "m",
        "integer_pixel_shifts_only": True,
        "periodic_object_boundary": True,
    }

    if bool(output_cfg.get("save_png", True)):
        plot_complex_field(
            P_B_true,
            figures_dir / "known_probe_amp_phase.png",
            title="Known B-plane probe",
            dx=dx_m,
        )
        save_reconstruction_comparison(
            B_true,
            B_rec_aligned,
            figures_dir / "B_truth_reconstruction_error.png",
            dx=dx_m,
            mask=illumination_mask,
        )
        plot_loss_curve(loss_curve, figures_dir / "loss_curve.png")
        save_scan_positions(
            scan_positions,
            figures_dir / "scan_positions.png",
        )
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
            outputs_dir / "epie_known_probe.h5",
            I_stack=I_stack,
            scan_positions=scan_positions,
            instrument={
                "wavelength": wavelength_m,
                "dx": dx_m,
                "z_AB": 0.0,
                "z_BC": z_BC_m,
                "detector_pixel_size": detector_pixel_size_m,
                "medium_index": medium_index,
            },
            sample={
                "sample_A_type": "none_known_probe_directly_defined",
                "sample_B_type": config.get("sample_b", {}).get(
                    "type", "random_amp_phase"
                ),
                "sample_B_parameters": config.get("sample_b", {}),
                "tgv_parameters": {},
            },
            truth={
                "P_B_true": P_B_true,
                "B_true": B_true,
            },
            reconstruction={
                "P_B_rec": P_B_rec,
                "B_init": B_init,
                "B_rec": B_rec,
                "B_rec_aligned_to_truth": B_rec_aligned,
                "loss_curve": loss_curve,
                "initial_data_fidelity_loss": initial_loss,
                "final_data_fidelity_loss": final_loss,
                "illumination_map": illumination_map,
                "illuminated_mask": illumination_mask,
                "settings": result["metadata"],
            },
            config_yaml=config_to_yaml(config),
            metadata={**metadata, "forward_model": forward_metadata},
            metrics=metrics,
        )

    print(f"Saved run to: {run_dir}")
    print(
        "Final loss: "
        f"{metrics['final_data_fidelity_loss']:.6e}; "
        "aligned B relative error: "
        f"{metrics['final_B_complex_relative_error_aligned_illuminated']:.6e}"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
