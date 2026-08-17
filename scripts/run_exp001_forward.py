"""Run forward simulations from YAML configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.io.config import config_to_yaml, load_config, save_config
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit
from tgv_ptycho.io.naming import make_run_dir
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5
from tgv_ptycho.objects.tgv2d import make_thin_phase_disk
from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.fields import make_circular_aperture, make_plane_wave
from tgv_ptycho.viz.plot_field import plot_complex_field, save_intensity_image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def _make_sample_transmission(
    config: dict,
    shape: tuple[int, int],
    dx_m: float,
) -> np.ndarray:
    sample_cfg = config.get("sample", {})
    sample_type = sample_cfg.get("type", "thin_phase_disk")
    if sample_type == "thin_phase_disk":
        return make_thin_phase_disk(
            shape,
            dx_m,
            float(sample_cfg["diameter_m"]),
            float(sample_cfg.get("phase_shift_rad", 1.0)),
        )
    if sample_type == "circular_aperture":
        aperture = make_circular_aperture(
            shape,
            dx_m,
            float(sample_cfg["diameter_m"]) / 2.0,
        )
        return aperture.astype(np.complex128)
    msg = f"Unsupported sample type for run_exp001_forward.py: {sample_type}"
    raise ValueError(msg)


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    run_cfg = config.get("run", {})
    optics_cfg = config.get("optics", {})
    field_cfg = config.get("incident_field", {})

    run_name = run_cfg.get("name", config_path.stem)
    output_root = PROJECT_ROOT / run_cfg.get("output_root", "runs")
    run_dir = make_run_dir(output_root, run_name)
    figures_dir = run_dir / "figures"
    outputs_dir = run_dir / "outputs"

    shape = tuple(int(v) for v in optics_cfg.get("shape", [256, 256]))
    dx_m = float(optics_cfg["dx_m"])
    wavelength_m = float(optics_cfg["wavelength_m"])
    z_m = float(optics_cfg["propagation_distance_m"])
    medium_index = float(optics_cfg.get("medium_index", 1.0))

    U0 = make_plane_wave(
        shape,
        dx_m,
        wavelength_m,
        theta_x=float(field_cfg.get("theta_x_rad", 0.0)),
        theta_y=float(field_cfg.get("theta_y_rad", 0.0)),
        amplitude=float(field_cfg.get("amplitude", 1.0)),
    )
    sample_transmission = _make_sample_transmission(config, shape, dx_m)
    U_after_sample = U0 * sample_transmission
    U_z = angular_spectrum_propagate(
        U_after_sample,
        dx_m,
        wavelength_m,
        z_m,
        n=medium_index,
    )
    intensity = np.abs(U_z) ** 2

    plot_complex_field(
        U_z,
        save_path=figures_dir / "propagated_field_amp_phase.png",
        title="exp001 propagated field",
        dx=dx_m,
    )
    save_intensity_image(
        intensity,
        figures_dir / "intensity.png",
        dx=dx_m,
        title="Detector intensity",
    )
    plt.close("all")

    metadata = {
        "run_name": run_name,
        "created_at": created_at_utc(),
        "git_commit": get_git_commit(PROJECT_ROOT) or "",
        "config_path": str(config_path),
        "shape": list(shape),
        "dx_m": dx_m,
        "wavelength_m": wavelength_m,
        "propagation_distance_m": z_m,
        "medium_index": medium_index,
        "sample_type": config.get("sample", {}).get("type", "unknown"),
    }
    input_energy = float(np.sum(np.abs(U_after_sample) ** 2))
    output_energy = float(np.sum(intensity))
    metrics = {
        "input_energy": input_energy,
        "output_energy": output_energy,
        "energy_ratio_output_over_input": output_energy / input_energy,
        "intensity_min": float(intensity.min()),
        "intensity_max": float(intensity.max()),
        "intensity_mean": float(intensity.mean()),
    }

    save_config(run_dir / "config.yaml", config)
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "metrics.json", metrics)

    if config.get("output", {}).get("save_hdf5", True):
        config_yaml = config_to_yaml(config)
        save_ptycho_hdf5(
            outputs_dir / "propagation_sanity.h5",
            I_stack=intensity[None, :, :],
            scan_positions=np.zeros((1, 2), dtype=np.float64),
            instrument={
                "wavelength": wavelength_m,
                "dx": dx_m,
                "z_AB": 0.0,
                "z_BC": z_m,
                "detector_pixel_size": dx_m,
            },
            sample={
                "sample_A_type": config.get("sample", {}).get("type", "unknown"),
                "sample_B_type": "none",
                "tgv_parameters": {},
            },
            truth={
                "incident_probe_true": U0,
                "A_true": sample_transmission,
                "U_after_sample_true": U_after_sample,
                "P_B_true": U_after_sample,
                "U_detector_true": U_z,
                "I_detector_true": intensity,
            },
            config_yaml=config_yaml,
            metadata=metadata,
            metrics=metrics,
        )

    print(f"Saved run to: {run_dir}")
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
