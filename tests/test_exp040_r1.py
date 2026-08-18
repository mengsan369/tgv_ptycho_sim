from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tgv_ptycho.forward.exp040 as exp040_module
from tgv_ptycho.forward.exp040 import (
    _center_pad,
    _make_canonical_b,
    _make_r1_canonical_b,
    _r1_status,
    resample_centered_grid,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config


def _tiny_r1_config() -> dict[str, Any]:
    path = Path(
        "configs/experiments/exp040_TGV_3d_multislice_refinement.yaml"
    )
    config = deepcopy(load_config(path))
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
        padded_shapes=[
            [16, 16],
            [20, 20],
            [24, 24],
            [28, 28],
            [32, 32],
        ],
        common_center_roi_shape=[16, 16],
        acceptance_pair_shapes=[[28, 28], [32, 32]],
        edge_ring_width_m=2.0e-6,
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


def test_official_r1_config_validates() -> None:
    config = load_config(
        Path(
            "configs/experiments/exp040_TGV_3d_multislice_refinement.yaml"
        )
    )

    validate_exp040_config(config)


def test_r1_canonical_b_reuses_seed_and_periodically_preserves_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_r1_config()
    legacy, legacy_dx = _make_canonical_b(config["sample_b"])
    generated_shapes: list[tuple[int, int]] = []
    original = exp040_module.make_random_phase_object

    def record_generation(shape: tuple[int, int], **kwargs: Any) -> np.ndarray:
        generated_shapes.append(tuple(shape))
        return original(shape, **kwargs)

    monkeypatch.setattr(
        exp040_module, "make_random_phase_object", record_generation
    )
    base, working, fine_dx, validation = _make_r1_canonical_b(
        config["sample_b"],
        config["diagnostics_r1"],
        legacy,
        legacy_dx,
    )
    mapped = resample_centered_grid(base, fine_dx, legacy.shape, legacy_dx)

    assert generated_shapes == [(96, 96)]
    assert working.shape == (128, 128)
    assert np.array_equal(working[16:-16, 16:-16], base)
    assert np.max(np.abs(mapped - legacy)) <= 1.0e-12
    assert validation["pass"] is True


def test_center_pad_changes_only_the_outside_domain() -> None:
    source = np.arange(16, dtype=np.float64).reshape(4, 4)
    padded = _center_pad(source, (8, 8))

    assert np.array_equal(padded[2:6, 2:6], source)
    assert np.count_nonzero(padded) == np.count_nonzero(source)


def test_absent_and_disabled_r1_are_legacy_compatible() -> None:
    absent_config = _tiny_r1_config()
    absent_config.pop("diagnostics_r1")
    disabled_config = _tiny_r1_config()
    disabled_config["diagnostics_r1"]["enabled"] = False

    absent = run_exp040_experiment(absent_config)
    disabled = run_exp040_experiment(disabled_config)

    assert "diagnostics_r1" not in absent
    assert "diagnostics_r1" not in absent["metrics"]
    assert "diagnostics_r1" not in disabled
    assert "diagnostics_r1" not in disabled["metrics"]
    _assert_equal_tree(absent, disabled)


def test_r1_only_adds_registered_cases_and_reports_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_r1_config()
    simulated_specs: list[tuple[tuple[int, int], float, float]] = []
    original = exp040_module._simulate_case
    original_r1 = exp040_module._run_r1_diagnostics
    legacy_status_at_r1_entry: list[str] = []

    def record_case(*args: Any, **kwargs: Any) -> dict[str, Any]:
        simulated_specs.append(
            (
                tuple(kwargs["shape"]),
                float(kwargs["dx_m"]),
                float(kwargs["dz_m"]),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(exp040_module, "_simulate_case", record_case)

    def record_legacy_status(*args: Any, **kwargs: Any) -> Any:
        legacy_metrics = args[-1]
        legacy_status_at_r1_entry.append(
            str(legacy_metrics["experiment_status"])
        )
        return original_r1(*args, **kwargs)

    monkeypatch.setattr(
        exp040_module, "_run_r1_diagnostics", record_legacy_status
    )
    result = run_exp040_experiment(config)
    diagnostics = result["diagnostics_r1"]
    metrics = result["metrics"]["diagnostics_r1"]

    assert simulated_specs.count(((16, 16), 1.0e-6, 0.5e-6)) == 1
    assert simulated_specs.count(((64, 64), 0.25e-6, 2.0e-6)) == 1
    assert simulated_specs.count(((28, 28), 1.0e-6, 2.0e-6)) == 1
    assert simulated_specs.count(((32, 32), 1.0e-6, 2.0e-6)) == 1

    refined = diagnostics["refined_convergence"]
    assert set(refined) == {"axial", "lateral", "fov"}
    for group in refined.values():
        assert set(group) == {"x_values", "U_A_exit", "P_B", "I_stack"}
        assert all(
            np.asarray(group[name]).shape == np.asarray(group["x_values"]).shape
            for name in ("U_A_exit", "P_B", "I_stack")
        )
    external = diagnostics["external_padding"]
    assert set(external) == {
        "x_values",
        "P_B",
        "I_stack",
        "U_A_exit_center_invariance",
    }
    np.testing.assert_array_equal(
        metrics["refined_convergence"]["axial"]["acceptance_pair_m"],
        [1.0e-6, 0.5e-6],
    )
    np.testing.assert_array_equal(
        metrics["refined_convergence"]["lateral"][
            "acceptance_pair_dx_m"
        ],
        [0.5e-6, 0.25e-6],
    )
    np.testing.assert_array_equal(
        metrics["refined_convergence"]["fov"][
            "acceptance_pair_shapes"
        ],
        [[28, 28], [32, 32]],
    )
    floor_components = metrics["refined_floor"]["components"]
    for output_name in ("U_A_exit", "P_B", "I_stack"):
        assert metrics["refined_floor"][output_name] == max(
            floor_components[output_name].values()
        )
    for group_name in ("axial", "lateral", "fov"):
        assert "x_values" in metrics["refined_convergence"][group_name]
    assert "U_A_exit_center_invariance" in metrics["external_padding"]
    assert metrics["all_finite"] is True
    assert metrics["all_intensity_nonnegative"] is True
    assert legacy_status_at_r1_entry == [
        result["metrics"]["experiment_status"]
    ]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((False, True, True, True), "Failed"),
        ((True, False, True, True), "Inconclusive"),
        ((True, True, False, True), "Inconclusive"),
        ((True, True, True, False), "Inconclusive"),
        ((True, True, True, True), "Passed"),
    ],
)
def test_r1_three_state_status_logic(
    flags: tuple[bool, bool, bool, bool], expected: str
) -> None:
    assert (
        _r1_status(
            hard_checks_pass=flags[0],
            refinement_pass=flags[1],
            external_pass=flags[2],
            visibility_pass=flags[3],
        )
        == expected
    )
