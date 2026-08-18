from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from scripts import run_exp040_multislice_forward as runner

from tgv_ptycho.forward.exp040 import (
    _r8_outcome_code,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config, save_config

R8_CONFIG_PATH = Path(
    "configs/experiments/"
    "exp040_TGV_3d_multislice_r8_unified_visibility.yaml"
)


def _tiny_r8_config() -> dict[str, Any]:
    config = deepcopy(load_config(R8_CONFIG_PATH))
    config["optics"].update(
        baseline_shape=[8, 8],
        baseline_dx_m=2.0e-6,
        z_AB_m=20.0e-6,
        z_BC_m=30.0e-6,
    )
    config["optics"]["detector"]["pixel_size_m"] = 2.0e-6
    config["sample_a"].update(
        thickness_m=8.0e-6,
        d_top_m=6.0e-6,
        d_waist_m=4.0e-6,
        d_bottom_m=6.0e-6,
        z_waist_m=4.0e-6,
    )
    config["waist_perturbation"].update(
        delta_d_waist_m=0.5e-6,
        d_waist_m=[3.5e-6, 4.0e-6, 4.5e-6],
    )
    config["multislice"]["target_dz_m"] = 2.0e-6
    config["sample_b"].update(physical_feature_size_m=0.5e-6)
    config["sample_b"]["canonical_grid"].update(
        shape=[48, 48],
        dx_m=0.5e-6,
        fov_m=[24.0e-6, 24.0e-6],
    )
    config["scan"].update(
        num_x=1,
        num_y=1,
        step_m=2.0e-6,
        max_jitter_px=0,
        jitter_quantum_m=2.0e-6,
    )
    config["convergence"]["axial"].update(
        fixed_shape=[8, 8], fixed_dx_m=2.0e-6
    )
    config["convergence"]["lateral_fixed_fov"].update(
        fov_m=[16.0e-6, 16.0e-6],
        cases=[
            {"shape": [8, 8], "dx_m": 2.0e-6},
            {"shape": [16, 16], "dx_m": 1.0e-6},
            {"shape": [32, 32], "dx_m": 0.5e-6},
        ],
        acceptance_pair_dx_m=[1.0e-6, 0.5e-6],
        comparison_grid_shape=[16, 16],
        comparison_grid_dx_m=1.0e-6,
    )
    config["convergence"]["fov"].update(
        fixed_dx_m=1.0e-6,
        shapes=[[16, 16], [20, 20], [24, 24]],
        common_center_roi_shape=[16, 16],
        common_center_roi_fov_m=[16.0e-6, 16.0e-6],
        acceptance_pair_shapes=[[20, 20], [24, 24]],
    )
    r1 = config["diagnostics_r1"]["sample_b_refinement"]
    r1["base_grid"].update(
        shape=[192, 192],
        dx_m=0.125e-6,
        fov_m=[24.0e-6, 24.0e-6],
        coarse_phase_cell_shape=[48, 48],
    )
    r1["working_grid"].update(
        shape=[256, 256],
        dx_m=0.125e-6,
        fov_m=[32.0e-6, 32.0e-6],
    )
    r1["extension_each_side_px"] = [32, 32]

    r8 = config["diagnostics_r8"]
    r8["sample_a_cases"].update(
        fov_m=[16.0e-6, 16.0e-6],
        cases=[
            {
                "id": "axial_coarse",
                "shape": [16, 16],
                "dx_m": 1.0e-6,
                "dz_m": 2.0e-6,
                "d_waist_m": 4.0e-6,
            },
            {
                "id": "common_reference",
                "shape": [16, 16],
                "dx_m": 1.0e-6,
                "dz_m": 1.0e-6,
                "d_waist_m": 4.0e-6,
            },
            {
                "id": "finest_baseline",
                "shape": [32, 32],
                "dx_m": 0.5e-6,
                "dz_m": 1.0e-6,
                "d_waist_m": 4.0e-6,
            },
            {
                "id": "waist_minus",
                "shape": [32, 32],
                "dx_m": 0.5e-6,
                "dz_m": 1.0e-6,
                "d_waist_m": 3.5e-6,
            },
            {
                "id": "waist_plus",
                "shape": [32, 32],
                "dx_m": 0.5e-6,
                "dz_m": 1.0e-6,
                "d_waist_m": 4.5e-6,
            },
        ],
    )
    r8["detector_sampling"].update(
        node_dx_m=0.5e-6,
        base_fov_m=[48.0e-6, 48.0e-6],
        base_node_shape=[96, 96],
        primary_open_fov_m=[96.0e-6, 96.0e-6],
        primary_open_node_shape=[192, 192],
        native_roi_shape=[8, 8],
    )
    r8["finite_b"]["physical_shape_m"] = [24.0e-6, 24.0e-6]
    r8["open_control"].update(
        fov_m=[[72.0e-6, 72.0e-6], [96.0e-6, 96.0e-6]],
        node_shapes=[[144, 144], [192, 192]],
    )
    r8["comparisons"].update(
        lateral_u_a_exit_common_grid=[16, 16],
        lateral_u_a_exit_common_dx_m=1.0e-6,
    )
    return config


@pytest.mark.parametrize(
    ("status", "convergence", "visibility", "expected"),
    [
        ("Failed", False, False, "unified_forward_attribution_blocked"),
        ("Inconclusive", False, True, "unified_numerical_floor_not_closed"),
        (
            "Inconclusive",
            True,
            False,
            "waist_signal_not_above_registered_floor",
        ),
        (
            "Passed",
            True,
            True,
            "waist_signal_resolved_within_registered_working_model",
        ),
    ],
)
def test_r8_outcome_table_is_frozen(
    status: str, convergence: bool, visibility: bool, expected: str
) -> None:
    assert (
        _r8_outcome_code(
            status=status,
            convergence_pass=convergence,
            visibility_pass=visibility,
        )
        == expected
    )


def test_official_and_tiny_r8_configs_validate() -> None:
    validate_exp040_config(load_config(R8_CONFIG_PATH))
    validate_exp040_config(_tiny_r8_config())


def test_tiny_r8_runs_unified_cases_and_validates_metrics() -> None:
    config = _tiny_r8_config()
    result = run_exp040_experiment(config)
    metrics = result["metrics"]["diagnostics_r8"]
    runner._validate_r8_metrics(config, metrics)

    assert all(f"diagnostics_r{stage}" not in result for stage in range(1, 8))
    assert metrics["sampling"]["interface_factor"] == 8
    assert metrics["sampling"]["case_ids"] == [
        "axial_coarse",
        "common_reference",
        "finest_baseline",
        "waist_minus",
        "waist_plus",
    ]
    assert metrics["sampling"]["scan_count"] == 1
    assert metrics["sampling"]["full_volumes_retained"] is False
    assert metrics["sampling"]["full_node_stacks_retained"] is False
    assert metrics["determinism"]["pass"] is True
    assert metrics["all_finite"] is True
    assert metrics["all_intensity_nonnegative"] is True
    assert metrics["status"] in {"Passed", "Inconclusive"}
    assert set(result["diagnostics_r8"]["selected_scan0"]) == {
        "waist_minus",
        "finest_baseline",
        "waist_plus",
    }
    assert set(result["diagnostics_r8"]["open_selected_scan0"]) == {
        "open_288",
        "open_384",
    }


def test_tiny_r8_runner_writes_hdf5_and_eleven_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_r8_config()
    config["run"].update(name="tiny_r8", output_root="runs")
    config["output"]["hdf5_filename"] = "tiny_r8.h5"
    config_path = tmp_path / "tiny_r8.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert len(list((run_dir / "figures").glob("*.png"))) == 11
    with runner.h5py.File(run_dir / "outputs" / "tiny_r8.h5", "r") as h5:
        assert "entry/metrics/diagnostics_r8" in h5
        assert "entry/truth/diagnostics_r8" not in h5
        assert h5["entry/metadata/diagnostic_stage"][()].decode() == "R8"
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }


def test_r8_rejects_post_registered_interface_change() -> None:
    config = _tiny_r8_config()
    config["diagnostics_r8"]["interface"]["factor"] = 4

    with pytest.raises(ValueError, match="q8 rule"):
        validate_exp040_config(config)
