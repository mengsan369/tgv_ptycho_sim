from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp040_multislice_forward as runner

from tgv_ptycho.forward.camera import positive_midpoint_pixel_average
from tgv_ptycho.forward.exp040 import (
    _r4_node_geometry_error,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config, save_config


def _tiny_r4_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            Path(
                "configs/experiments/"
                "exp040_TGV_3d_multislice_r4_positive_quadrature.yaml"
            )
        )
    )
    config["optics"].update(
        baseline_shape=[16, 16],
        baseline_dx_m=1.0e-6,
        z_AB_m=20.0e-6,
        z_BC_m=30.0e-6,
    )
    config["optics"]["detector"]["pixel_size_m"] = 1.0e-6
    config["sample_a"].update(
        thickness_m=8.0e-6,
        d_top_m=6.0e-6,
        d_waist_m=4.0e-6,
        d_bottom_m=6.0e-6,
        z_waist_m=4.0e-6,
    )
    config["multislice"]["target_dz_m"] = 2.0e-6
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
    config["waist_perturbation"].update(
        delta_d_waist_m=0.5e-6,
        d_waist_m=[3.5e-6, 4.0e-6, 4.5e-6],
    )
    config["convergence"]["axial"].update(
        fixed_shape=[16, 16],
        fixed_dx_m=1.0e-6,
        dz_cases_m=[4.0e-6, 2.0e-6, 1.0e-6],
        acceptance_pair_m=[2.0e-6, 1.0e-6],
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

    r1 = config["diagnostics_r1"]
    r1["sample_b_refinement"]["base_grid"].update(
        shape=[96, 96],
        dx_m=0.25e-6,
        fov_m=[24.0e-6, 24.0e-6],
        coarse_phase_cell_shape=[12, 12],
    )
    r1["sample_b_refinement"]["working_grid"].update(
        shape=[128, 128],
        dx_m=0.25e-6,
        fov_m=[32.0e-6, 32.0e-6],
    )
    r1["sample_b_refinement"]["extension_each_side_px"] = [16, 16]
    r1["refined_axial"].update(
        existing_reference_dz_m=1.0e-6,
        new_dz_m=0.5e-6,
        acceptance_pair_m=[1.0e-6, 0.5e-6],
    )
    r1["refined_lateral"].update(
        fixed_fov_m=[16.0e-6, 16.0e-6],
        existing_reference={"shape": [32, 32], "dx_m": 0.5e-6},
        new_case={"shape": [64, 64], "dx_m": 0.25e-6},
        acceptance_pair_dx_m=[0.5e-6, 0.25e-6],
        comparison_grid_shape=[32, 32],
        comparison_grid_dx_m=0.5e-6,
    )
    r1["refined_fov"].update(
        fixed_dx_m=1.0e-6,
        existing_shapes=[[16, 16], [20, 20], [24, 24]],
        new_shapes=[[28, 28], [32, 32]],
        common_center_roi_shape=[16, 16],
        acceptance_pair_shapes=[[28, 28], [32, 32]],
    )
    r1["external_padding"].update(
        source_shape=[16, 16],
        fixed_dx_m=1.0e-6,
        padded_shapes=[[16, 16], [20, 20], [24, 24], [28, 28], [32, 32]],
        common_center_roi_shape=[16, 16],
        acceptance_pair_shapes=[[28, 28], [32, 32]],
        edge_ring_width_m=2.0e-6,
    )
    config["diagnostics_r2"]["period_commensurate"].update(
        fixed_dx_m=1.0e-6,
        fov_m=[24.0e-6, 48.0e-6, 72.0e-6],
        shapes=[[24, 24], [48, 48], [72, 72]],
        period_counts=[1, 2, 3],
        base_period_shape=[24, 24],
        common_center_roi_shape=[16, 16],
        acceptance_pair_shapes=[[48, 48], [72, 72]],
    )
    config["diagnostics_r4"]["sampling"].update(
        external_fov_m=[48.0e-6, 48.0e-6],
        node_dx_m=[0.5e-6, 0.25e-6, 0.125e-6],
        node_shapes=[[96, 96], [192, 192], [384, 384]],
        native_full_shape=[48, 48],
        native_roi_shape=[16, 16],
    )
    return config


def test_positive_midpoint_average_preserves_constant_sum_and_nonnegativity() -> None:
    values = np.arange(32, dtype=np.float64).reshape(2, 4, 4)
    averaged = positive_midpoint_pixel_average(values, 2)
    expected = np.asarray(
        [
            [[2.5, 4.5], [10.5, 12.5]],
            [[18.5, 20.5], [26.5, 28.5]],
        ]
    )

    np.testing.assert_array_equal(averaged, expected)
    np.testing.assert_array_equal(
        positive_midpoint_pixel_average(np.ones((8, 12)), 4),
        np.ones((2, 3)),
    )
    assert float(np.sum(averaged) * 4) == float(np.sum(values))
    assert np.all(averaged >= 0.0)
    with pytest.raises(ValueError, match="divisible"):
        positive_midpoint_pixel_average(np.ones((7, 8)), 2)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_r4_staggered_node_geometry_is_centered(factor: int) -> None:
    assert _r4_node_geometry_error(48, 0.5e-6, factor) <= 1e-14
    offsets = ((np.arange(factor) + 0.5) / factor - 0.5) * 0.5e-6
    assert float(np.mean(offsets)) == pytest.approx(0.0, abs=1e-22)
    assert np.all(np.diff(offsets) > 0.0)


def test_official_r4_config_validates() -> None:
    config = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r4_positive_quadrature.yaml"
        )
    )
    validate_exp040_config(config)


def test_tiny_r4_streams_compact_positive_diagnostics() -> None:
    config = _tiny_r4_config()
    validate_exp040_config(config)

    result = run_exp040_experiment(config)

    assert "diagnostics_r4" in result
    assert all(f"diagnostics_r{stage}" not in result for stage in (1, 2, 3))
    metrics = result["metrics"]["diagnostics_r4"]
    assert metrics["sampling"]["scan_count"] == 1
    assert metrics["sampling"]["full_node_stacks_retained"] is False
    np.testing.assert_array_equal(metrics["sampling"]["factors"], [2, 4, 8])
    np.testing.assert_array_equal(
        metrics["sampling"]["node_shapes"], [[96, 96], [192, 192], [384, 384]]
    )
    assert metrics["canonical_b_validation"]["pass"] is True
    assert metrics["quadrature_controls"]["pass"] is True
    assert metrics["determinism"]["pass"] is True
    assert metrics["all_finite"] is True
    assert metrics["all_intensity_nonnegative"] is True
    assert metrics["status"] in {"Passed", "Inconclusive"}
    assert result["diagnostics_r4"]["selected_q8_scan0"].shape == (16, 16)


def test_tiny_r4_runner_writes_hdf5_and_ten_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_r4_config()
    config["run"]["name"] = "tiny_r4"
    config["run"]["output_root"] = "runs"
    config["output"]["hdf5_filename"] = "tiny_r4.h5"
    config_path = tmp_path / "tiny_r4.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert len(list((run_dir / "figures").glob("*.png"))) == 10
    output = run_dir / "outputs" / "tiny_r4.h5"
    with h5py.File(output, "r") as h5:
        assert "entry/metrics/diagnostics_r4" in h5
        assert "entry/truth/diagnostics_r4" not in h5
        assert h5["entry/metadata/diagnostic_stage"][()].decode() == "R4"
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }


def test_r4_rejects_post_registered_factor_change() -> None:
    config = _tiny_r4_config()
    config["diagnostics_r4"]["sampling"]["factors"] = [2, 4, 6]

    with pytest.raises(ValueError, match="factors"):
        validate_exp040_config(config)
