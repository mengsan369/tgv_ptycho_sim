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
