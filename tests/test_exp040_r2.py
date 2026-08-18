from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tgv_ptycho.forward.exp040 import (
    _center_periodic_extend,
    _r2_interpretation_code,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config


def _tiny_r2_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            Path(
                "configs/experiments/"
                "exp040_TGV_3d_multislice_r2_boundary_alias.yaml"
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

    r2 = config["diagnostics_r2"]
    r2["period_commensurate"].update(
        fixed_dx_m=1.0e-6,
        fov_m=[24.0e-6, 48.0e-6, 72.0e-6],
        shapes=[[24, 24], [48, 48], [72, 72]],
        period_counts=[1, 2, 3],
        base_period_shape=[24, 24],
        common_center_roi_shape=[16, 16],
        acceptance_pair_shapes=[[48, 48], [72, 72]],
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


def test_official_r2_config_validates() -> None:
    config = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r2_boundary_alias.yaml"
        )
    )

    validate_exp040_config(config)


def test_center_periodic_extend_preserves_exact_center_period() -> None:
    base = np.exp(1j * np.arange(36, dtype=np.float64).reshape(6, 6))
    extended = _center_periodic_extend(base, (18, 18))

    assert extended.shape == (18, 18)
    assert np.array_equal(extended[6:12, 6:12], base)
    assert np.array_equal(extended[:6, :6], base)


def test_r2_only_adds_registered_diagnostics_and_preserves_r0_r1() -> None:
    config = _tiny_r2_config()
    without_r2 = deepcopy(config)
    without_r2.pop("diagnostics_r2")
    without_r2["output"].pop("save_r2_figures")

    legacy = run_exp040_experiment(without_r2)
    result = run_exp040_experiment(config)

    legacy_comparable = {
        key: value
        for key, value in result.items()
        if key != "diagnostics_r2"
    }
    legacy_comparable["metrics"] = {
        key: value
        for key, value in result["metrics"].items()
        if key != "diagnostics_r2"
    }
    _assert_equal_tree(legacy, legacy_comparable)

    metrics = result["metrics"]["diagnostics_r2"]
    assert metrics["version"] == "R2"
    assert set(metrics["period_aligned"]) == {
        "current_asm",
        "alias_controlled",
    }
    np.testing.assert_array_equal(
        metrics["period_aligned"]["current_asm"]["shapes"],
        [[24, 24], [48, 48], [72, 72]],
    )
    assert metrics["canonical_b_validation"]["pass"] is True
    assert metrics["a_exit_center_invariance"]["pass"] is True
    assert metrics["alias_masks"]["pass"] is True
    assert metrics["determinism"]["pass"] is True
    assert metrics["all_finite"] is True
    assert metrics["all_intensity_nonnegative"] is True
    assert metrics["status"] in {"Passed", "Inconclusive", "Failed"}
    assert result["metrics"]["experiment_status"] == "Inconclusive"
    assert result["metrics"]["diagnostics_r1"]["status"] in {
        "Passed",
        "Inconclusive",
        "Failed",
    }


@pytest.mark.parametrize(
    ("current_pass", "alias_pass", "material", "expected"),
    [
        (True, True, False, "period_aligned_fov_supported"),
        (True, True, True, "period_aligned_but_method_dependent"),
        (False, True, True, "transfer_sampling_alias_supported"),
        (False, False, False, "remaining_downstream_floor"),
        (True, False, True, "alias_control_method_conflict"),
        (False, True, False, "ambiguous_method_effect"),
    ],
)
def test_r2_interpretation_table_is_frozen(
    current_pass: bool,
    alias_pass: bool,
    material: bool,
    expected: str,
) -> None:
    assert _r2_interpretation_code(current_pass, alias_pass, material) == expected


def test_r2_rejects_noncommensurate_registered_fov() -> None:
    config = _tiny_r2_config()
    config["diagnostics_r2"]["period_commensurate"]["fov_m"][-1] = 70e-6

    with pytest.raises(ValueError, match="shapes, dx, and FOV"):
        validate_exp040_config(config)


def test_r2_requires_r1_to_remain_enabled() -> None:
    config = _tiny_r2_config()
    config["diagnostics_r1"]["enabled"] = False

    with pytest.raises(ValueError, match="requires diagnostics_r1.enabled"):
        validate_exp040_config(config)
