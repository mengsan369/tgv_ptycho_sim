from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from tgv_ptycho.io.save_load import save_ptycho_hdf5


def _test_output_path(prefix: str) -> Path:
    output_dir = Path("runs") / "_pytest_hdf5_layout"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_{uuid4().hex}.h5"


def test_experimental_hdf5_omits_truth_and_keeps_calibration() -> None:
    output_path = _test_output_path("experimental")
    save_ptycho_hdf5(
        output_path,
        I_stack=np.ones((2, 8, 8), dtype=np.float64),
        scan_positions=np.zeros((2, 2), dtype=np.float64),
        instrument={"wavelength": 532e-9, "dx": 2e-6},
        reconstruction={"loss_curve": np.array([], dtype=np.float64)},
        calibration={"camera": {"detector_pixel_size": 2e-6}},
        preprocessing={"dark_subtracted": True},
        config_yaml="run:\n  name: experimental\n",
        metadata={"dataset_type": "experimental"},
        metrics={"num_frames": 2},
    )

    with h5py.File(output_path, "r") as h5:
        assert "/entry/data/I_stack" in h5
        assert "/entry/data/scan_positions" in h5
        assert "/entry/instrument" in h5
        assert "/entry/reconstruction" in h5
        assert "/entry/calibration" in h5
        assert "/entry/preprocessing" in h5
        assert "/entry/config_yaml" in h5
        assert "/entry/metadata/dataset_type" in h5
        assert "/entry/metrics/num_frames" in h5
        assert "/entry/truth" not in h5
        assert bool(h5["/entry/preprocessing/dark_subtracted"][()]) is True


def test_simulation_hdf5_keeps_truth_and_can_omit_calibration() -> None:
    output_path = _test_output_path("simulation")
    save_ptycho_hdf5(
        output_path,
        I_stack=np.ones((1, 8, 8), dtype=np.float64),
        scan_positions=np.zeros((1, 2), dtype=np.float64),
        instrument={"wavelength": 532e-9, "dx": 2e-6},
        truth={"P_B_true": np.ones((8, 8), dtype=np.complex128)},
        config_yaml="run:\n  name: simulation\n",
        metadata={"dataset_type": "simulation"},
        metrics={"relative_error": 0.0},
    )

    with h5py.File(output_path, "r") as h5:
        assert "/entry/data/I_stack" in h5
        assert "/entry/instrument" in h5
        assert "/entry/truth/P_B_true" in h5
        assert "/entry/config_yaml" in h5
        assert "/entry/metadata/dataset_type" in h5
        assert "/entry/metrics/relative_error" in h5
        assert "/entry/calibration" not in h5
        assert "/entry/preprocessing" not in h5


def test_exp020_hdf5_can_store_raw_and_evaluation_only_results() -> None:
    output_path = _test_output_path("exp020")
    field = np.ones((8, 8), dtype=np.complex128)
    save_ptycho_hdf5(
        output_path,
        I_stack=np.ones((2, 8, 8), dtype=np.float64),
        scan_positions=np.zeros((2, 2), dtype=np.float64),
        instrument={"wavelength": 532e-9, "dx": 2e-6, "z_AB": 1e-3},
        sample={"sample_A_type": "smooth_random_thin_phase"},
        truth={"A_true": field, "P_B_true": field, "B_true": field},
        reconstruction={
            "P_B_rec": field,
            "B_rec": field,
            "A_rec_raw": field,
            "A_rec_phase_only": field,
            "simulation_evaluation_only": {
                "A_rec_aligned_to_truth": field,
            },
        },
        config_yaml="run:\n  name: exp020\n",
        metadata={"dataset_type": "simulation"},
        metrics={"final_data_fidelity_loss": 0.0},
    )

    with h5py.File(output_path, "r") as h5:
        assert "/entry/truth/A_true" in h5
        assert "/entry/reconstruction/A_rec_raw" in h5
        assert "/entry/reconstruction/A_rec_phase_only" in h5
        assert (
            "/entry/reconstruction/simulation_evaluation_only/"
            "A_rec_aligned_to_truth" in h5
        )


def test_exp030_hdf5_keeps_baseline_and_parameter_sweep_shapes() -> None:
    output_path = _test_output_path("exp030")
    num_cases, num_positions, ny, nx = 3, 5, 12, 14
    field = np.ones((ny, nx), dtype=np.complex128)
    stack = np.ones((num_positions, ny, nx), dtype=np.float64)
    save_ptycho_hdf5(
        output_path,
        I_stack=stack,
        scan_positions=np.zeros((num_positions, 2), dtype=np.float64),
        instrument={"wavelength": 532e-9, "dx": 1e-6},
        sample={"sample_A_type": "single_axisymmetric_TGV"},
        truth={
            "fill_path_length_m": np.ones((ny, nx), dtype=np.float64),
            "A_effective_true": field,
            "parameter_sweep": {
                "d_waist_m": np.asarray([9e-6, 10e-6, 11e-6]),
                "A_effective_true": np.stack([field] * num_cases),
                "P_B_true": np.stack([field] * num_cases),
                "I_stack_true": np.stack([stack] * num_cases),
            },
        },
        reconstruction={
            "cases": {
                "baseline": {
                    "P_B_rec_raw": field,
                    "B_rec_raw": field,
                    "loss_curve": np.asarray([0.2, 0.1], dtype=np.float64),
                    "simulation_evaluation_only": {
                        "P_B_rec_aligned_to_truth": field,
                    },
                }
            }
        },
        config_yaml="run:\n  name: exp030\n",
        metadata={"dataset_type": "simulation"},
        metrics={"model_validation": {"zero_contrast_max_abs_error": 0.0}},
    )

    with h5py.File(output_path, "r") as h5:
        assert h5["/entry/data/I_stack"].shape == (num_positions, ny, nx)
        assert h5["/entry/truth/parameter_sweep/d_waist_m"].shape == (
            num_cases,
        )
        assert h5["/entry/truth/parameter_sweep/A_effective_true"].shape == (
            num_cases,
            ny,
            nx,
        )
        assert h5["/entry/truth/parameter_sweep/P_B_true"].dtype == np.complex128
        assert h5["/entry/truth/parameter_sweep/I_stack_true"].shape == (
            num_cases,
            num_positions,
            ny,
            nx,
        )
        assert h5[
            "/entry/reconstruction/cases/baseline/P_B_rec_raw"
        ].shape == (ny, nx)
        assert (
            "/entry/reconstruction/cases/baseline/"
            "simulation_evaluation_only/P_B_rec_aligned_to_truth"
        ) in h5
