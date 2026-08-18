from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scripts.run_exp040_r13 as runner
import scripts.run_exp040_r13_preflight as preflight

from tgv_ptycho.io.config import load_config
from tgv_ptycho.viz.plot_exp040_r13 import save_exp040_r13_figures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/exp040_TGV_3d_multislice_r13.yaml"
)
PREFLIGHT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/exp040_TGV_3d_multislice_r13_preflight.yaml"
)


def test_r13_scientific_contract_and_preflight_are_locked() -> None:
    config = load_config(CONFIG_PATH)
    runner.validate_r13_config(config)
    assert runner.scientific_contract_sha256(config) == (
        "B37A16F0E6D4F91B498BE4240C2D83A8CA727259A6A38B2B35FEB25EEF51EBB0"
    )
    assert runner._sha256(CONFIG_PATH) == runner.REGISTERED_CONFIG_SHA256
    assert runner._project_full_tgv_unknowns(
        config, degree=3, element_size_ratio=1.0
    ) == 240684
    provenance = runner._load_and_validate_provenance(config)
    assert provenance["preflight_metrics"]["formal_r13_allowed"] is True

    preflight_config = load_config(PREFLIGHT_CONFIG_PATH)
    preflight.validate_preflight_config(preflight_config)
    assert preflight._sha256(PREFLIGHT_CONFIG_PATH) == (
        preflight.REGISTERED_CONFIG_SHA256
    )


def test_r13_hdf5_and_figures_accept_synthetic_contract(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dense_radius = np.linspace(2.1e-5, 4.6e-5, 101)
    measurement_radius = np.asarray([2.4e-5, 3.2e-5, 4.0e-5, 4.6e-5])
    truth = np.exp(1j * dense_radius / 5.32e-7)
    domain_arrays = {
        case_id: {
            "dense_radii_m": dense_radius,
            "dense_field": truth.copy(),
            "dense_derivative": 1j * truth / 5.32e-7,
            "dense_truth": truth.copy(),
            "measurement_radii_m": measurement_radius,
            "measurement_field": np.ones(4, dtype=np.complex128),
            "measurement_derivative": np.ones(4, dtype=np.complex128),
            "incoming_to_outgoing_ratio": np.full(4, 1e-5),
            "outgoing_impedance_residual": np.full(4, 2e-5),
            "radial_flux": np.ones(4),
        }
        for case_id in config["domain_reflection"]["fixed_case_order"]
    }
    pollution_cases = {}
    for case_id in config["physical_k_pollution"]["fixed_case_order"]:
        case = config["physical_k_pollution"]["cases"][case_id]
        pollution_cases[case_id] = {
            "degree": int(case["degree"]),
            "element_size_ratio": float(case["element_size_ratio"]),
            "estimated_full_tgv_active_unknowns": 100000 * int(case["degree"]),
            "candidate_eligible": int(case["degree"]) >= 4,
            "homogeneous": {"weighted_relative_l2": 1e-3},
            "glass_air_interface": {"weighted_relative_l2": 2e-3},
        }
    coordinate = np.linspace(0.0, 4.0, 17)
    field = np.ones((17, 17), dtype=np.complex128)
    selected = {
        family: {
            "radial_coordinates": coordinate,
            "axial_coordinates": coordinate,
            "numerical_field": field,
            "truth_field": field,
        }
        for family in ("homogeneous", "glass_air_interface")
    }
    metrics = {
        "scientific_result": True,
        "thresholds": dict(config["thresholds"]),
        "physical_k_pollution": {
            "fixed_case_order": list(
                config["physical_k_pollution"]["fixed_case_order"]
            ),
            "cases": pollution_cases,
            "selected_candidate_id": "h1_p4",
        },
        "full_tgv_projection": dict(config["full_tgv_projection"]),
    }
    arrays = {
        "domain_reflection": domain_arrays,
        "selected_candidate_fields": selected,
    }
    output = tmp_path / "synthetic.h5"
    runner._write_hdf5(
        output, config, {"synthetic": True}, metrics, arrays
    )
    figure_paths = save_exp040_r13_figures(
        tmp_path / "figures", metrics, arrays
    )

    with h5py.File(output, "r") as handle:
        assert set(handle["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        assert set(handle["entry/data"]) == {
            "domain_reflection",
            "physical_k_pollution",
        }
    assert len(figure_paths) == 3
    assert all(path.stat().st_size > 0 for path in figure_paths)
