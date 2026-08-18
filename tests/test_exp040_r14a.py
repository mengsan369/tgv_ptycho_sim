from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scripts.run_exp040_r14a as runner

from tgv_ptycho.io.config import load_config
from tgv_ptycho.viz.plot_exp040_r14a import save_exp040_r14a_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/exp040_TGV_3d_multislice_r14a.yaml"
)


def test_r14a_config_baseline_and_order_estimator() -> None:
    config = load_config(CONFIG_PATH)
    runner.validate_r14a_config(config)
    assert runner._sha256(CONFIG_PATH) == runner.REGISTERED_CONFIG_SHA256
    assert runner.scientific_contract_sha256(config) == (
        "1305D2D4D46E4AE8FB5F974340F3B639928DF876DAA2FC3AB1B9D5C8DE318AE7"
    )
    baseline = runner._load_locked_baseline(config)
    assert baseline["metrics"]["maximum_incoming_to_outgoing_ratio"] > 1e-3
    element_sizes = np.asarray([1 / 12, 1 / 16, 1 / 24], dtype=np.float64)
    errors = 2.0 * element_sizes**4
    assert np.isclose(runner._fit_order(element_sizes, errors), 4.0)


def _synthetic_contract(config):
    coordinate = -np.linspace(0.0, 4.0e-6, 41)
    truth = np.exp(-1j * coordinate / 5.32e-7)
    arrays = {
        case_id: {
            "measurement_coordinates_m": -np.asarray([1e-6, 2e-6, 3e-6]),
            "incoming_to_outgoing_ratio": np.full(3, 1e-5),
            "outgoing_impedance_residual": np.full(3, 2e-5),
            "dense_coordinates_m": coordinate,
            "dense_field": truth.copy(),
            "dense_truth": truth,
        }
        for case_id in config["axial_attribution"]["fixed_case_order"]
    }
    case_metrics = {
        case_id: {
            "all_finite": True,
            "maximum_incoming_to_outgoing_ratio": 1e-5,
            "maximum_outgoing_impedance_residual": 2e-5,
            "dense_field_relative_l2": 3e-6,
            "solver_controls": {"relative_residual": 1e-14},
        }
        for case_id in arrays
    }
    metrics = {
        "scientific_result": True,
        "status": "Passed",
        "new_cases": case_metrics,
        "mesh_convergence": {
            "element_sizes_m": np.asarray([1 / 12, 1 / 16, 1 / 24]) * 1e-6,
            "incoming_to_outgoing_ratio": np.asarray([1e-3, 3e-4, 6e-5]),
            "outgoing_impedance_residual": np.asarray([2e-3, 6e-4, 1.2e-4]),
            "dense_field_relative_l2": np.asarray([1e-4, 2e-5, 2e-6]),
            "orders": {
                "maximum_incoming_to_outgoing_ratio": 4.0,
                "maximum_outgoing_impedance_residual": 4.0,
                "dense_field_relative_l2": 5.0,
            },
        },
        "pml_separation": {"raw_field_relative_l2": 0.0},
        "thresholds": dict(config["thresholds"]),
    }
    return metrics, arrays


def test_r14a_hdf5_checkpoint_and_plot_accept_synthetic_contract(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    metrics, arrays = _synthetic_contract(config)
    hdf5_path = tmp_path / "synthetic.h5"
    checkpoint_path = tmp_path / "glass_h24_pml2.npz"
    runner._write_hdf5(
        hdf5_path,
        config,
        {"synthetic": True},
        metrics,
        arrays,
    )
    runner._write_checkpoint(checkpoint_path, metrics, arrays)
    figure = save_exp040_r14a_figure(tmp_path / "figures", metrics, arrays)

    with h5py.File(hdf5_path, "r") as handle:
        assert set(handle["entry/data"]) == {"axial_attribution"}
        assert set(handle["entry/truth"]) == set(arrays)
    checkpoint = np.load(checkpoint_path, allow_pickle=False)
    assert str(checkpoint["case_id"]) == "glass_h24_pml2"
    assert np.all(np.isfinite(checkpoint["dense_field"]))
    assert figure.stat().st_size > 0
