from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp040_multislice_forward as runner

from tgv_ptycho.forward.camera import (
    make_square_pixel_mtf,
    periodic_square_pixel_average,
)
from tgv_ptycho.forward.exp040 import (
    _r3_aligned_bilinear_upsample,
    _r3_native_sample,
    _r3_outcome_code,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config, save_config


def _tiny_r3_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            Path(
                "configs/experiments/"
                "exp040_TGV_3d_multislice_r3_detector_path.yaml"
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
    config["diagnostics_r3"]["sampling"].update(
        external_fov_m=[48.0e-6, 48.0e-6],
        factors=[1, 2, 4],
        dx_m=[1.0e-6, 0.5e-6, 0.25e-6],
        shapes=[[48, 48], [96, 96], [192, 192]],
        native_full_shape=[48, 48],
        native_roi_shape=[16, 16],
        acceptance_pair_factors=[2, 4],
        canonical_period_count=2,
        native_sample_offsets_px=[0, 0, 1],
        physical_origin_compensation_m=[0.0, 0.25e-6, 0.125e-6],
    )
    return config


def _assert_equal_tree(left: Any, right: Any) -> None:
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left) == set(right)
        for key in left:
            _assert_equal_tree(left[key], right[key])
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(left, np.ndarray):
            assert isinstance(right, Sequence)
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                _assert_equal_tree(left_item, right_item)
            return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def test_official_r3_config_validates() -> None:
    config = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r3_detector_path.yaml"
        )
    )
    validate_exp040_config(config)


def test_periodic_square_pixel_mtf_preserves_constant_sum_and_alignment() -> None:
    rng = np.random.default_rng(20260811)
    values = rng.random((32, 40))
    averaged = periodic_square_pixel_average(values, 0.25, 1.0)
    constant = periodic_square_pixel_average(np.ones((32, 40)), 0.25, 1.0)

    np.testing.assert_allclose(constant.real, 1.0, rtol=0.0, atol=1e-14)
    assert np.max(np.abs(constant.imag)) <= 1e-14
    assert float(np.sum(averaged.real)) == pytest.approx(float(np.sum(values)))
    shifted = np.roll(values, (3, 5), axis=(0, 1))
    shifted_average = periodic_square_pixel_average(shifted, 0.25, 1.0)
    np.testing.assert_allclose(
        shifted_average,
        np.roll(averaged, (3, 5), axis=(0, 1)),
        rtol=0.0,
        atol=1e-13,
    )
    mtf = make_square_pixel_mtf(values.shape, 0.25, 1.0)
    assert mtf.shape == values.shape
    assert mtf[0, 0] == 1.0


@pytest.mark.parametrize(("factor", "offset"), [(1, 0), (2, 0), (4, 1)])
def test_r3_aligned_mapping_recovers_native_samples(
    factor: int, offset: int
) -> None:
    source = np.arange(42, dtype=np.float64).reshape(6, 7) * (1.0 + 0.5j)
    refined = _r3_aligned_bilinear_upsample(source, factor, offset)

    np.testing.assert_array_equal(_r3_native_sample(refined, factor, offset), source)


def test_r3_preserves_r0_r1_r2_and_streams_compact_diagnostics() -> None:
    config = _tiny_r3_config()
    without_r3 = deepcopy(config)
    without_r3.pop("diagnostics_r3")
    without_r3["output"].pop("save_r3_figures")

    legacy = run_exp040_experiment(without_r3)
    result = run_exp040_experiment(config)
    comparable = {
        key: value for key, value in result.items() if key != "diagnostics_r3"
    }
    comparable["metrics"] = {
        key: value
        for key, value in result["metrics"].items()
        if key != "diagnostics_r3"
    }
    _assert_equal_tree(legacy, comparable)

    metrics = result["metrics"]["diagnostics_r3"]
    assert metrics["version"] == "R3"
    assert metrics["sampling"]["scan_count"] == 1
    assert metrics["sampling"]["full_detector_stacks_retained"] is False
    np.testing.assert_array_equal(metrics["sampling"]["factors"], [1, 2, 4])
    assert metrics["canonical_b_validation"]["pass"] is True
    assert metrics["a_exit_native_recovery"]["pass"] is True
    assert metrics["alias_masks"]["pass"] is True
    assert metrics["determinism"]["pass"] is True
    assert metrics["all_finite"] is True
    assert metrics["status"] in {"Passed", "Inconclusive", "Failed"}
    assert result["metrics"]["experiment_status"] == "Inconclusive"
    assert set(result["diagnostics_r3"]["selected_scan"]) == {
        "point_sample",
        "pixel_box_average",
        "relative_difference",
    }


def test_tiny_r3_runner_writes_valid_hdf5_metrics_and_fifteen_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_r3_config()
    config["run"]["name"] = "tiny_r3"
    config["run"]["output_root"] = "runs"
    config["output"]["hdf5_filename"] = "tiny_r3.h5"
    config_path = tmp_path / "tiny_r3.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    figures = sorted((run_dir / "figures").glob("*.png"))
    assert len(figures) == 15
    output = run_dir / "outputs" / "tiny_r3.h5"
    with h5py.File(output, "r") as h5:
        assert "entry/metrics/diagnostics_r3" in h5
        assert "entry/truth/diagnostics_r3" not in h5
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"hard_checks_pass": False}, "hard_control_failure"),
        ({"probe_pass": False}, "upstream_sampling_not_converged"),
        (
            {
                "primary_pass": True,
                "point_pass": False,
                "point_vs_pixel_material": True,
            },
            "point_detector_model_defect_supported",
        ),
        ({"primary_pass": True}, "detector_path_sampling_converged"),
        (
            {"primary_pass": False, "b_exit_or_bc_material": True},
            "finite_pixel_does_not_resolve_b_bc_floor",
        ),
        ({"primary_pass": False}, "boundary_or_higher_physics_priority"),
    ],
)
def test_r3_interpretation_table_is_frozen(
    kwargs: dict[str, bool], expected: str
) -> None:
    values = {
        "hard_checks_pass": True,
        "probe_pass": True,
        "primary_pass": False,
        "point_pass": True,
        "point_vs_pixel_material": False,
        "b_exit_or_bc_material": False,
    }
    values.update(kwargs)
    assert _r3_outcome_code(**values) == expected


def test_r3_rejects_post_registered_primary_branch_change() -> None:
    config = _tiny_r3_config()
    config["diagnostics_r3"]["detector_sampling"][
        "primary_detector_branch"
    ] = "point_sample"

    with pytest.raises(ValueError, match="detector branches"):
        validate_exp040_config(config)
