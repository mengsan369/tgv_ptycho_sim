from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from scripts import run_exp040_r10_stage_a_preflight as preflight

from tgv_ptycho.io.config import load_config, save_config

CONFIG_PATH = Path(
    "configs/experiments/"
    "exp040_TGV_3d_multislice_r10_stage_a_preflight.yaml"
)


def _case_result(
    case_id: str,
    *,
    projected_s: float,
    peak_rss_bytes: int = 400_000_000,
    determinism: float = 0.0,
    interface_error: float = 0.0,
    all_finite: bool = True,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "projected_full_case_elapsed_s": projected_s,
        "sampled_peak_rss_bytes": peak_rss_bytes,
        "maximum_determinism_relative_l2": determinism,
        "maximum_interface_bound_error": interface_error,
        "all_finite": all_finite,
    }


def test_official_preflight_config_is_exactly_registered() -> None:
    preflight.validate_preflight_config(load_config(CONFIG_PATH))


def test_preflight_rejects_post_registered_change() -> None:
    config = load_config(CONFIG_PATH)
    config["benchmark"]["timed_slice_count"] = 8

    with pytest.raises(ValueError, match="frozen registration"):
        preflight.validate_preflight_config(config)


@pytest.mark.parametrize(
    ("case_results", "expected_status", "expected_code"),
    [
        (
            [
                _case_result(
                    "current_512", projected_s=82.0, determinism=1.0e-10
                ),
                _case_result("fine_1024", projected_s=330.0),
            ],
            "Failed",
            "preflight_kernel_control_failed",
        ),
        (
            [
                _case_result("current_512", projected_s=120.0),
                _case_result("fine_1024", projected_s=330.0),
            ],
            "Inconclusive",
            "short_kernel_extrapolation_not_calibrated",
        ),
        (
            [
                _case_result("current_512", projected_s=82.0),
                _case_result("fine_1024", projected_s=600.0),
            ],
            "Inconclusive",
            "stage_a_cost_not_feasible_on_current_host",
        ),
        (
            [
                _case_result("current_512", projected_s=82.0),
                _case_result("fine_1024", projected_s=330.0),
            ],
            "Passed",
            "stage_a_formal_run_feasible",
        ),
    ],
)
def test_preflight_outcome_table_is_frozen(
    case_results: list[dict[str, Any]],
    expected_status: str,
    expected_code: str,
) -> None:
    outcome = preflight._preflight_outcome(
        load_config(CONFIG_PATH),
        case_results,
        available_memory_bytes=4_000_000_000,
    )

    assert outcome["status"] == expected_status
    assert outcome["interpretation_code"] == expected_code
    assert outcome["scientific_conclusion_allowed"] is False
    assert outcome["stage_b_allowed"] is False


def test_tiny_preflight_kernel_is_streamed_finite_and_deterministic() -> None:
    config = load_config(CONFIG_PATH)
    config["benchmark"].update(
        timed_slice_count=2,
        timed_repeats=2,
        rss_sampling_interval_s=0.001,
    )
    case = {
        "id": "tiny",
        "shape": [8, 8],
        "dx_m": 8.0e-6,
        "formal_slice_count": 4,
    }

    result = preflight._benchmark_case(config, case)

    assert result["timed_slice_count"] == 2
    assert result["q8_subnode_tests_timed"] == 8 * 8 * 2 * 64
    assert result["all_finite"] is True
    assert result["maximum_interface_bound_error"] == 0.0
    assert result["maximum_determinism_relative_l2"] == 0.0
    assert result["sampled_peak_rss_bytes"] > 0


def test_preflight_runner_writes_only_four_non_scientific_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    config_path = tmp_path / "preflight.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(preflight, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        preflight, "_available_memory_bytes", lambda: 4_000_000_000
    )

    def fake_benchmark(
        _config: dict[str, Any], case: dict[str, Any]
    ) -> dict[str, Any]:
        if case["id"] == "current_512":
            return _case_result("current_512", projected_s=82.0)
        return _case_result("fine_1024", projected_s=330.0)

    monkeypatch.setattr(preflight, "_benchmark_case", fake_benchmark)

    run_dir = preflight.run(config_path)

    files = {path.name for path in run_dir.rglob("*") if path.is_file()}
    assert files == {
        "config.yaml",
        "metadata.json",
        "metrics.json",
        "run_state.json",
    }
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert metrics["scientific_result"] is False
    assert metrics["outcome"]["status"] == "Passed"
    assert list((run_dir / "figures").iterdir()) == []
    assert list((run_dir / "outputs").iterdir()) == []
