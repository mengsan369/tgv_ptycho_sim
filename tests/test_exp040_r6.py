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
    _r6_outcome_code,
    _r6_periodic_support_pattern,
    _r6_phase_tapered_transmission,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config, save_config


def _tiny_r6_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            Path(
                "configs/experiments/"
                "exp040_TGV_3d_multislice_r6_b_support_sensitivity.yaml"
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
    config["diagnostics_r6"]["sampling"].update(
        node_dx_m=0.25e-6,
        fov_m=[48.0e-6, 48.0e-6],
        node_shape=[192, 192],
        native_roi_shape=[16, 16],
    )
    config["diagnostics_r6"]["support_family"].update(
        support_width_m=[20.0e-6, 24.0e-6, 28.0e-6],
        edge_taper_width_m=[0.0, 1.0e-6, 2.0e-6],
        nominal_support_width_m=24.0e-6,
    )
    config["diagnostics_r6"]["determinism"]["support_width_m"] = 24.0e-6
    return config


def test_r6_support_pattern_crops_and_extends_same_realization() -> None:
    base = np.arange(64, dtype=np.float64).reshape(8, 8).astype(np.complex128)
    cropped = _r6_periodic_support_pattern(base, (6, 6))
    extended = _r6_periodic_support_pattern(base, (10, 10))

    np.testing.assert_array_equal(cropped, base[1:7, 1:7])
    np.testing.assert_array_equal(extended[1:9, 1:9], base)
    np.testing.assert_array_equal(extended[[0, -1], 1:9], base[[-1, 0]])


def test_r6_phase_taper_is_unit_modulus_with_registered_endpoints() -> None:
    phase = np.linspace(0.0, 0.8, 100).reshape(10, 10)
    pattern = np.exp(1j * phase)
    hard, hard_weights = _r6_phase_tapered_transmission(pattern, 0)
    tapered, weights = _r6_phase_tapered_transmission(pattern, 2)

    np.testing.assert_allclose(hard, pattern, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(hard_weights, np.ones((10, 10)))
    np.testing.assert_allclose(np.abs(tapered), 1.0, rtol=0.0, atol=1e-15)
    assert np.all(weights[[0, -1], :] == 0.0)
    assert np.all(weights[:, [0, -1]] == 0.0)
    assert np.all(weights[2:-2, 2:-2] == 1.0)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Passed", "periodic_b_materiality_robust_over_support_envelope"),
        ("Inconclusive", "periodic_b_materiality_support_sensitive"),
        ("Failed", "support_envelope_attribution_blocked"),
    ],
)
def test_r6_outcome_table_is_frozen(status: str, expected: str) -> None:
    assert _r6_outcome_code(status=status) == expected


def test_official_r6_config_validates() -> None:
    config = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r6_b_support_sensitivity.yaml"
        )
    )
    validate_exp040_config(config)


def test_tiny_r6_streams_all_nine_support_cases() -> None:
    config = _tiny_r6_config()
    validate_exp040_config(config)

    result = run_exp040_experiment(config)

    assert "diagnostics_r6" in result
    assert all(
        f"diagnostics_r{stage}" not in result for stage in (1, 2, 3, 4, 5)
    )
    metrics = result["metrics"]["diagnostics_r6"]
    assert metrics["sampling"]["scan_count"] == 1
    assert metrics["sampling"]["case_count"] == 9
    assert metrics["sampling"]["full_node_stacks_retained"] is False
    assert metrics["support_effects"]["relative_l2_matrix"].shape == (3, 3)
    assert metrics["nominal_sensitivity"]["relative_l2_matrix"].shape == (3, 3)
    assert metrics["controls"]["pass"] is True
    assert metrics["controls"]["nominal_r5_provenance_applicable"] is False
    assert metrics["determinism"]["pass"] is True
    assert metrics["all_finite"] is True
    assert metrics["all_intensity_nonnegative"] is True
    assert metrics["status"] in {"Passed", "Inconclusive"}
    selected = result["diagnostics_r6"]["selected_scan0"]
    assert set(selected) == {
        "periodic",
        "nominal",
        "minimum_effect",
        "maximum_effect",
    }
    assert all(image.shape == (16, 16) for image in selected.values())


def test_tiny_r6_runner_writes_hdf5_and_eleven_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_r6_config()
    config["run"]["name"] = "tiny_r6"
    config["run"]["output_root"] = "runs"
    config["output"]["hdf5_filename"] = "tiny_r6.h5"
    config_path = tmp_path / "tiny_r6.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert len(list((run_dir / "figures").glob("*.png"))) == 11
    output = run_dir / "outputs" / "tiny_r6.h5"
    with h5py.File(output, "r") as h5:
        assert "entry/metrics/diagnostics_r6" in h5
        assert "entry/truth/diagnostics_r6" not in h5
        assert h5["entry/metadata/diagnostic_stage"][()].decode() == "R6"
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }


def test_r6_rejects_post_registered_envelope_change() -> None:
    config = _tiny_r6_config()
    config["diagnostics_r6"]["support_family"]["support_width_m"][0] = 21e-6

    with pytest.raises(ValueError, match="support widths"):
        validate_exp040_config(config)
