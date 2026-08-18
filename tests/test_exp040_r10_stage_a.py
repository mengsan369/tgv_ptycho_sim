from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp040_r10_stage_a as runner

from tgv_ptycho.forward.exp040 import restrict_aligned_cell_average
from tgv_ptycho.io.config import load_config

CONFIG_PATH = Path(
    "configs/experiments/exp040_TGV_3d_multislice_r10_stage_a.yaml"
)


def _tiny_config() -> dict[str, Any]:
    config = deepcopy(load_config(CONFIG_PATH))
    config["physics"].update(
        wavelength_m=2.5e-6,
        sample_thickness_m=8.0e-6,
        d_top_m=6.0e-6,
        d_waist_m=4.0e-6,
        d_bottom_m=6.0e-6,
        z_waist_m=4.0e-6,
    )
    config["stage_a"].update(common_fov_m=[16.0e-6, 16.0e-6])
    config["stage_a"]["cases"] = [
        {
            "id": "current_512",
            "shape": [8, 8],
            "dx_m": 2.0e-6,
            "dz_m": 2.0e-6,
            "expected_slice_count": 4,
        },
        {
            "id": "fine_1024",
            "shape": [16, 16],
            "dx_m": 1.0e-6,
            "dz_m": 2.0e-6,
            "expected_slice_count": 4,
        },
    ]
    config["stage_a"]["physical_passband"][
        "cutoff_cycles_per_m"
    ] = 4.0e5
    return config


def test_official_config_hash_and_science_controls_are_registered() -> None:
    config = load_config(CONFIG_PATH)
    runner.validate_stage_a_config(config)
    assert runner._sha256(CONFIG_PATH) == runner.REGISTERED_CONFIG_SHA256


def test_stage_a_config_rejects_post_registered_case_change() -> None:
    config = load_config(CONFIG_PATH)
    config["stage_a"]["cases"][1]["shape"] = [768, 768]
    with pytest.raises(ValueError, match="frozen registration"):
        runner.validate_stage_a_config(config)


@pytest.mark.parametrize(
    (
        "hard_controls_pass",
        "passband_error",
        "expected_status",
        "expected_code",
        "stage_b_allowed",
    ),
    [
        (
            False,
            0.01,
            "Failed",
            "stage_a_numerical_controls_failed",
            False,
        ),
        (
            True,
            0.05,
            "Passed",
            "scalar_lateral_reference_closed",
            True,
        ),
        (
            True,
            0.0500000001,
            "Inconclusive",
            "scalar_lateral_reference_not_closed",
            False,
        ),
    ],
)
def test_stage_a_outcome_table_is_frozen(
    hard_controls_pass: bool,
    passband_error: float,
    expected_status: str,
    expected_code: str,
    stage_b_allowed: bool,
) -> None:
    outcome = runner._stage_a_outcome(
        hard_controls_pass=hard_controls_pass,
        passband_relative_l2=passband_error,
        raw_relative_l2=0.2,
        convergence_threshold=0.05,
    )
    assert outcome["status"] == expected_status
    assert outcome["interpretation_code"] == expected_code
    assert outcome["stage_b_allowed"] is stage_b_allowed
    assert outcome["raw_convergence_pass_report_only"] is False


def test_restriction_controls_are_conservative_and_aligned() -> None:
    rng = np.random.default_rng(20260815)
    fine = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    restricted = restrict_aligned_cell_average(fine, 2)
    controls = runner._restriction_controls(fine, restricted, 2)
    assert controls["shape_matches"] is True
    assert controls["constant_max_abs_error"] == 0.0
    assert controls["four_subpixel_alignment_max_abs_error"] == 0.0
    assert controls["area_weighted_complex_mean_relative_error"] <= 1.0e-14


def test_tiny_stage_a_runs_two_streamed_cases_and_is_finite() -> None:
    events: list[str] = []

    def collect(event: str, _details: dict[str, object]) -> None:
        events.append(event)

    result = runner._run_stage_a(_tiny_config(), progress_callback=collect)
    metrics = result["metrics"]
    assert metrics["sampling"]["current_shape"] == [8, 8]
    assert metrics["sampling"]["fine_shape"] == [16, 16]
    assert [case["slice_count"] for case in metrics["case_controls"]] == [4, 4]
    assert metrics["hard_controls"]["all_finite"] is True
    assert np.isfinite(metrics["comparison"]["raw_relative_l2"])
    assert np.isfinite(
        metrics["comparison"]["external_passband_relative_l2"]
    )
    assert events == [
        "case_started",
        "case_completed",
        "case_started",
        "case_completed",
        "postprocessing_started",
        "postprocessing_completed",
    ]
    assert set(result["selected_maps"]) == {
        "raw_normalized_residual",
        "passband_normalized_residual",
        "raw_difference_spectrum",
        "external_passband_mask",
    }


def _fake_stage_a_result(config: dict[str, Any]) -> dict[str, Any]:
    maps = {
        "raw_normalized_residual": np.full((8, 8), 0.2),
        "passband_normalized_residual": np.full((8, 8), 0.04),
        "raw_difference_spectrum": np.full((8, 8), -2.0),
        "external_passband_mask": np.ones((8, 8)),
    }
    outcome = {
        "status": "Passed",
        "interpretation_code": "scalar_lateral_reference_closed",
        "external_passband_convergence_pass": True,
        "raw_convergence_pass_report_only": False,
        "stage_b_allowed": True,
    }
    return {
        "metrics": {
            "version": "R10_stage_a",
            "scientific_result": True,
            "provenance": dict(config["provenance"]),
            "sampling": {
                "current_dx_m": 1.25e-7,
                "current_shape": [512, 512],
                "fine_shape": [1024, 1024],
            },
            "comparison": {
                "raw_relative_l2": 0.2,
                "external_passband_relative_l2": 0.04,
            },
            "thresholds": {"convergence_relative_l2_max": 0.05},
            "outcome": outcome,
            "status": "Passed",
        },
        "selected_maps": maps,
    }


def test_runner_contract_writes_compact_validated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = CONFIG_PATH.resolve()
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_run_stage_a",
        lambda config, progress_callback=None: _fake_stage_a_result(dict(config)),
    )

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    progress = json.loads(
        (run_dir / "run_progress.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert state["stage_b_allowed"] is True
    assert metrics["scientific_result"] is True
    assert progress["latest_event"]["event"] == "artifacts_validated"
    assert len(list((run_dir / "figures").glob("*.png"))) == 2
    with h5py.File(
        run_dir / "outputs" / "exp040_r10_stage_a.h5", "r"
    ) as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
        }
        assert len(h5["entry/data"]) == 0
        assert "entry/truth" not in h5


def test_formal_runner_does_not_use_numpy_vdot() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "np.vdot" not in source
