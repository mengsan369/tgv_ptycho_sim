from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scripts import run_exp040_multislice_forward as runner

from tgv_ptycho.forward.exp040 import (
    _r9_explicit_complex_inner_product,
    _r9_outcome_code,
    make_physical_passband_mask,
    project_field_to_passband,
    resample_centered_grid,
    restrict_aligned_cell_average,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config, save_config

R9_CONFIG_PATH = Path(
    "configs/experiments/"
    "exp040_TGV_3d_multislice_r9_a_exit_attribution.yaml"
)


def test_r9_explicit_complex_inner_product_matches_vdot_and_large_exact_value(
) -> None:
    small_left = np.asarray([[1.0 + 2.0j, 3.0 - 1.0j]], dtype=np.complex128)
    small_right = np.asarray([[2.0 - 3.0j, -4.0 + 5.0j]], dtype=np.complex128)
    assert _r9_explicit_complex_inner_product(
        small_left, small_right
    ) == pytest.approx(np.vdot(small_left, small_right), rel=0.0, abs=0.0)

    large_left = np.full((256, 256), 1.0 + 2.0j, dtype=np.complex128)
    large_right = np.full((256, 256), 3.0 - 4.0j, dtype=np.complex128)
    element_count = large_left.size
    assert _r9_explicit_complex_inner_product(
        large_left, large_right
    ) == complex(-5 * element_count, -10 * element_count)


def _tiny_r9_config() -> dict[str, Any]:
    config = deepcopy(load_config(R9_CONFIG_PATH))
    config["optics"].update(
        wavelength_m=2.5e-6,
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
    r9 = config["diagnostics_r9"]
    r9["sample_a_cases"].update(
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
                "id": "axial_fine_reference",
                "shape": [16, 16],
                "dx_m": 1.0e-6,
                "dz_m": 0.5e-6,
                "d_waist_m": 4.0e-6,
            },
            {
                "id": "lateral_fine_reference",
                "shape": [32, 32],
                "dx_m": 0.5e-6,
                "dz_m": 1.0e-6,
                "d_waist_m": 4.0e-6,
            },
        ],
    )
    r9["physical_passband"]["cutoff_cycles_per_m"] = 4.0e5
    r9["lateral_restrictions"].update(
        target_shape=[16, 16], target_dx_m=1.0e-6
    )
    return config


def test_aligned_cell_average_is_conservative_and_matches_bilinear() -> None:
    rng = np.random.default_rng(20260814)
    fine = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))

    averaged = restrict_aligned_cell_average(fine, 2)
    bilinear = resample_centered_grid(fine, 0.5e-6, (16, 16), 1.0e-6)

    np.testing.assert_allclose(averaged, bilinear, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.sum(averaged) * (1.0e-6) ** 2,
        np.sum(fine) * (0.5e-6) ** 2,
        rtol=1e-14,
        atol=1e-24,
    )


def test_physical_passband_projection_preserves_constant_and_is_idempotent() -> None:
    shape = (32, 32)
    dx_m = 1.0e-6
    cutoff = 2.0e5
    checkerboard = np.indices(shape).sum(axis=0) % 2
    field = np.ones(shape, dtype=np.complex128) + checkerboard

    mask = make_physical_passband_mask(shape, dx_m, cutoff)
    first = project_field_to_passband(field, dx_m, cutoff)
    second = project_field_to_passband(first, dx_m, cutoff)
    constant = project_field_to_passband(
        np.ones(shape, dtype=np.complex128), dx_m, cutoff
    )

    assert mask.dtype == np.bool_
    assert 0 < np.count_nonzero(mask) < mask.size
    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(constant, 1.0, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    ("status", "passband", "raw", "expected"),
    [
        ("Failed", False, False, "a_exit_attribution_blocked"),
        (
            "Inconclusive",
            False,
            False,
            "external_propagating_band_discrepancy_remains",
        ),
        (
            "Passed",
            True,
            False,
            "raw_discrepancy_attributed_outside_external_propagating_gate",
        ),
        (
            "Passed",
            True,
            True,
            "raw_and_external_passband_a_exit_converged",
        ),
    ],
)
def test_r9_outcome_table_is_frozen(
    status: str, passband: bool, raw: bool, expected: str
) -> None:
    assert (
        _r9_outcome_code(
            status=status, passband_pass=passband, raw_pass=raw
        )
        == expected
    )


def test_official_and_tiny_r9_configs_validate() -> None:
    validate_exp040_config(load_config(R9_CONFIG_PATH))
    validate_exp040_config(_tiny_r9_config())


def test_tiny_r9_runs_all_cases_and_validates_metrics() -> None:
    config = _tiny_r9_config()
    progress_events: list[tuple[str, dict[str, object]]] = []

    def collect_progress(event: str, details: Mapping[str, object]) -> None:
        progress_events.append((event, dict(details)))

    result = run_exp040_experiment(
        config, progress_callback=collect_progress
    )
    metrics = result["metrics"]["diagnostics_r9"]
    runner._validate_r9_metrics(config, metrics)

    assert all(f"diagnostics_r{stage}" not in result for stage in range(1, 9))
    assert metrics["sampling"]["interface_factor"] == 8
    assert metrics["sampling"]["slice_counts"].tolist() == [4, 8, 16, 8]
    assert metrics["sampling"]["detector_path_recomputed"] is False
    assert metrics["restriction_controls"]["pass"] is True
    assert (
        metrics["spectral_controls"][
            "maximum_parseval_closure_relative_error"
        ]
        <= config["acceptance"]["algebra_relative_l2_max"]
    )
    assert metrics["all_finite"] is True
    assert metrics["hard_checks_pass"] is False
    assert metrics["status"] == "Failed"
    assert (
        metrics["outcome_flags"]["interpretation_code"]
        == "a_exit_attribution_blocked"
    )
    assert set(result["diagnostics_r9"]["selected_maps"]) == {
        "lateral_raw_bilinear",
        "lateral_raw_cell_average",
        "lateral_passband_bilinear",
        "lateral_passband_cell_average",
        "restriction_disagreement",
    }
    event_names = [event for event, _ in progress_events]
    assert event_names.count("r9_case_started") == 4
    assert event_names.count("r9_case_completed") == 4
    assert event_names[:4] == [
        "forward_started",
        "baseline_completed",
        "legacy_core_completed",
        "r9_started",
    ]
    assert event_names[-4:] == [
        "r9_postprocessing_started",
        "r9_postprocessing_completed",
        "r9_completed",
        "forward_completed",
    ]
    assert [
        details["case_id"]
        for event, details in progress_events
        if event == "r9_case_completed"
    ] == [
        "axial_coarse",
        "common_reference",
        "axial_fine_reference",
        "lateral_fine_reference",
    ]


def test_tiny_r9_runner_writes_hdf5_and_eleven_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_r9_config()
    config["run"].update(name="tiny_r9", output_root="runs")
    config["output"]["hdf5_filename"] = "tiny_r9.h5"
    config_path = tmp_path / "tiny_r9.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    progress = json.loads(
        (run_dir / "run_progress.json").read_text(encoding="utf-8")
    )
    assert progress["purpose"] == "non_scientific_execution_diagnostic"
    assert progress["latest_event"]["event"] == "artifacts_validated"
    progress_names = [event["event"] for event in progress["events"]]
    assert progress_names.count("r9_case_started") == 4
    assert progress_names.count("r9_case_completed") == 4
    assert len(list((run_dir / "figures").glob("*.png"))) == 11
    with runner.h5py.File(run_dir / "outputs" / "tiny_r9.h5", "r") as h5:
        assert "entry/metrics/diagnostics_r9" in h5
        assert "entry/truth/diagnostics_r9" not in h5
        assert h5["entry/metadata/diagnostic_stage"][()].decode() == "R9"
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }


def test_r9_rejects_post_registered_axial_reference_change() -> None:
    config = _tiny_r9_config()
    config["diagnostics_r9"]["sample_a_cases"]["cases"][2][
        "dz_m"
    ] = 0.75e-6

    with pytest.raises(ValueError, match="case relationships"):
        validate_exp040_config(config)
