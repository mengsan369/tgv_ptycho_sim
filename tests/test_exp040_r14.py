from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scripts.run_exp040_r14 as runner
import scripts.run_exp040_r14_preflight as preflight

from tgv_ptycho.io.config import load_config
from tgv_ptycho.viz.plot_exp040_r14 import save_exp040_r14_figures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/exp040_TGV_3d_multislice_r14.yaml"
)
PREFLIGHT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/exp040_TGV_3d_multislice_r14_preflight.yaml"
)
SCIENTIFIC_CONTRACT_SHA256 = (
    "269AA8FA68EB7795B2A5EB73D3F4A23B5C2C2E90382CE6613EE78253A043E2DB"
)


def test_r14_scientific_contract_and_preflight_are_locked() -> None:
    config = load_config(CONFIG_PATH)
    runner.validate_r14_config(config)
    assert runner.scientific_contract_sha256(config) == (
        SCIENTIFIC_CONTRACT_SHA256
    )

    preflight_config = load_config(PREFLIGHT_CONFIG_PATH)
    preflight.validate_preflight_config(preflight_config)
    assert preflight._sha256(PREFLIGHT_CONFIG_PATH) == (
        preflight.REGISTERED_CONFIG_SHA256
    )
    provenance, formal = preflight._validate_provenance_and_contract(
        preflight_config
    )
    assert provenance["scientific_contract"] == SCIENTIFIC_CONTRACT_SHA256
    grids = preflight._formal_grid_controls(formal)
    assert grids["all_unknown_counts_match"] is True
    assert grids["maximum_formal_active_unknowns"] == 274040


def _synthetic_contract(config):
    axial_arrays = {}
    for case_id in config["axial_pml"]["fixed_case_order"]:
        coordinate = np.linspace(0.0, 4.0e-6, 41)
        truth = np.exp(1j * coordinate / 5.32e-7)
        axial_arrays[case_id] = {
            "measurement_coordinates_m": np.asarray([1e-6, 2e-6, 3e-6]),
            "incoming_to_outgoing_ratio": np.full(3, 1e-5),
            "outgoing_impedance_residual": np.full(3, 2e-5),
            "dense_coordinates_m": coordinate,
            "dense_field": truth.copy(),
            "dense_truth": truth,
        }

    cases = {}
    solver_arrays = {}
    for case_index, case_id in enumerate(
        config["solver_scaling"]["fixed_case_order"]
    ):
        successful_modal = {
            family: {
                "solve_succeeded": True,
                "gmres": {"inner_iteration_count": 20 + case_index},
                "analytic_weighted_relative_l2": 2e-3,
            }
            for family in config["solver_scaling"]["modal_families"]
        }
        cases[case_id] = {
            "active_unknowns": config["solver_scaling"]["cases"][case_id][
                "expected_active_unknowns"
            ],
            "solvers": {
                "csl_ilu_gmres": {
                    "setup_succeeded": True,
                    "modal_results": successful_modal,
                },
                "two_level_ras_csl_gmres": {
                    "setup_succeeded": False,
                    "error_type": "SyntheticFailure",
                },
            },
        }
        coordinate = np.linspace(0.0, 4.0, 17)
        trace = np.exp(1j * coordinate)
        solver_arrays[case_id] = {
            "csl_ilu_gmres": {
                family: {
                    "radial_coordinates": coordinate,
                    "center_numerical_trace": trace,
                    "center_truth_trace": trace,
                    "preconditioned_residual_history": np.logspace(
                        0.0, -9.0, 20 + case_index
                    ),
                }
                for family in config["solver_scaling"]["modal_families"]
            },
            "two_level_ras_csl_gmres": {},
        }
    metrics = {
        "scientific_result": True,
        "thresholds": dict(config["thresholds"]),
        "solver_scaling": {
            "fixed_solver_order": list(config["solvers"]["fixed_order"]),
            "fixed_case_order": list(
                config["solver_scaling"]["fixed_case_order"]
            ),
            "modal_families": list(
                config["solver_scaling"]["modal_families"]
            ),
            "cases": cases,
            "solver_summaries": {
                "csl_ilu_gmres": {
                    "solver_gate_pass": True,
                    "projected_peak_gib": 2.0,
                    "maximum_largest_case_iterations": 24,
                },
                "two_level_ras_csl_gmres": {
                    "solver_gate_pass": False,
                    "projected_peak_gib": None,
                    "maximum_largest_case_iterations": None,
                },
            },
        },
    }
    arrays = {
        "axial_pml": axial_arrays,
        "solver_scaling": solver_arrays,
    }
    return metrics, arrays


def test_r14_hdf5_and_plots_preserve_a_solver_setup_failure(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    metrics, arrays = _synthetic_contract(config)
    output = tmp_path / "synthetic.h5"
    runner._write_hdf5(
        output,
        config,
        {"synthetic": True},
        metrics,
        arrays,
    )
    paths = save_exp040_r14_figures(tmp_path / "figures", metrics, arrays)

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
        failed = handle[
            "entry/data/solver_scaling/core4/two_level_ras_csl_gmres"
        ]
        assert len(failed) == 0
    assert len(paths) == 3
    assert all(path.stat().st_size > 0 for path in paths)


def test_r14_coarse_memory_projection_uses_measured_scaling_and_floor() -> None:
    config = load_config(CONFIG_PATH)
    block_counts = [1, 4, 9, 25, 81]
    cases = {}
    for case_id, blocks in zip(
        config["solver_scaling"]["fixed_case_order"],
        block_counts,
        strict=True,
    ):
        cases[case_id] = {
            "solvers": {
                "two_level_ras_csl_gmres": {
                    "storage": {"conservative_peak_model_bytes": 100_000_000},
                    "preconditioner": {
                        "block_count": blocks,
                        "coarse_factor_storage_bytes": 1000 * blocks,
                    },
                }
            }
        }
    projection = runner._project_solver_memory(
        config,
        "two_level_ras_csl_gmres",
        cases,
    )
    assert np.isclose(projection["coarse_fit_exponent"], 1.0)
    assert projection["coarse_fill_projected_bytes_before_safety"] >= (
        projection["coarse_largest_linear_floor_bytes"]
    )
    assert projection["projected_peak_gib"] > 0.0
