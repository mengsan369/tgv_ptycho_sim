from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp040_multislice_forward as runner

from tgv_ptycho.forward.exp040 import (
    _r5_boundary_ring_energy_fraction,
    _r5_outcome_code,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.forward.integer_shift import shift_field_integer_pixels
from tgv_ptycho.io.config import load_config, save_config


def _tiny_r5_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            Path(
                "configs/experiments/"
                "exp040_TGV_3d_multislice_r5_finite_support_open_boundary.yaml"
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
        shape=[192, 192],
        dx_m=0.125e-6,
        fov_m=[24.0e-6, 24.0e-6],
        coarse_phase_cell_shape=[48, 48],
    )
    r1["sample_b_refinement"]["working_grid"].update(
        shape=[256, 256],
        dx_m=0.125e-6,
        fov_m=[32.0e-6, 32.0e-6],
    )
    r1["sample_b_refinement"]["extension_each_side_px"] = [32, 32]
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
    config["diagnostics_r5"]["sampling"].update(
        node_dx_m=0.25e-6,
        base_fov_m=[48.0e-6, 48.0e-6],
        base_node_shape=[192, 192],
        padding_fov_m=[48.0e-6, 72.0e-6, 96.0e-6],
        padding_node_shapes=[[192, 192], [288, 288], [384, 384]],
        native_roi_shape=[16, 16],
        acceptance_pair_fov_m=[72.0e-6, 96.0e-6],
        boundary_ring_width_m=4.0e-6,
    )
    config["diagnostics_r5"]["finite_support"]["physical_shape_m"] = [
        24.0e-6,
        24.0e-6,
    ]
    config["diagnostics_r5"]["determinism"][
        "selected_padding_fov_m"
    ] = 96.0e-6
    return config


def test_finite_modulation_shift_has_transparent_constant_exterior() -> None:
    modulation = np.zeros((6, 6), dtype=np.complex128)
    modulation[2:4, 2:4] = 0.5j
    shifted = shift_field_integer_pixels(
        modulation,
        np.asarray([2.0, 0.0]),
        1.0,
        boundary="constant",
        fill_value=0.0j,
    )

    assert np.all(shifted[:, :2] == 0.0j)
    assert np.all((1.0 + shifted)[:, :2] == 1.0 + 0.0j)
    assert np.count_nonzero(shifted) == np.count_nonzero(modulation)


def test_r5_boundary_ring_energy_fraction() -> None:
    field = np.ones((10, 10), dtype=np.complex128)
    assert _r5_boundary_ring_energy_fraction(field, 2) == pytest.approx(0.64)
    field[2:-2, 2:-2] = 0.0
    assert _r5_boundary_ring_energy_fraction(field, 2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("status", "support", "boundary", "expected"),
    [
        ("Passed", False, False, "finite_support_and_circular_wrap_nonmaterial"),
        ("Passed", True, False, "finite_support_material"),
        ("Passed", False, True, "circular_wrap_material"),
        ("Passed", True, True, "finite_support_and_circular_wrap_material"),
        ("Inconclusive", True, True, "attribution_blocked"),
    ],
)
def test_r5_interpretation_table_is_frozen(
    status: str, support: bool, boundary: bool, expected: str
) -> None:
    assert (
        _r5_outcome_code(
            status=status,
            support_material=support,
            boundary_material=boundary,
        )
        == expected
    )


def test_official_r5_config_validates() -> None:
    config = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r5_finite_support_open_boundary.yaml"
        )
    )
    validate_exp040_config(config)


def test_tiny_r5_streams_compact_boundary_diagnostics() -> None:
    config = _tiny_r5_config()
    validate_exp040_config(config)

    result = run_exp040_experiment(config)

    assert "diagnostics_r5" in result
    assert all(f"diagnostics_r{stage}" not in result for stage in (1, 2, 3, 4))
    metrics = result["metrics"]["diagnostics_r5"]
    assert metrics["sampling"]["scan_count"] == 1
    assert metrics["sampling"]["full_node_stacks_retained"] is False
    np.testing.assert_array_equal(
        metrics["sampling"]["padding_node_shapes"],
        [[192, 192], [288, 288], [384, 384]],
    )
    assert metrics["finite_support"]["pass"] is True
    assert metrics["controls"]["pass"] is True
    assert (
        metrics["controls"]["base_open_vs_finite_circular_relative_l2"]
        <= 1.0e-12
    )
    assert metrics["determinism"]["pass"] is True
    assert metrics["all_finite"] is True
    assert metrics["all_intensity_nonnegative"] is True
    assert metrics["status"] in {"Passed", "Inconclusive"}
    selected = result["diagnostics_r5"]["selected_scan0"]
    assert set(selected) == {
        "periodic_circular_192",
        "finite_circular_192",
        "finite_open_384",
    }
    assert all(image.shape == (16, 16) for image in selected.values())


def test_tiny_r5_runner_writes_hdf5_and_eleven_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_r5_config()
    config["run"]["name"] = "tiny_r5"
    config["run"]["output_root"] = "runs"
    config["output"]["hdf5_filename"] = "tiny_r5.h5"
    config_path = tmp_path / "tiny_r5.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert len(list((run_dir / "figures").glob("*.png"))) == 11
    output = run_dir / "outputs" / "tiny_r5.h5"
    with h5py.File(output, "r") as h5:
        assert "entry/metrics/diagnostics_r5" in h5
        assert "entry/truth/diagnostics_r5" not in h5
        assert h5["entry/metadata/diagnostic_stage"][()].decode() == "R5"
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }


def test_r5_rejects_post_registered_exterior_change() -> None:
    config = _tiny_r5_config()
    config["diagnostics_r5"]["finite_support"][
        "exterior_transmission_real"
    ] = 0.0

    with pytest.raises(ValueError, match="boundary flags"):
        validate_exp040_config(config)
