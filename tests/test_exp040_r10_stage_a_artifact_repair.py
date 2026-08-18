from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
from scripts import repair_exp040_r10_stage_a_artifacts as repairer

from tgv_ptycho.io.config import save_config
from tgv_ptycho.io.save_load import save_json


def _metrics() -> dict[str, Any]:
    return {
        "version": "R10_stage_a",
        "sampling": {
            "case_ids": ["current_512", "fine_1024"],
            "current_shape": [512, 512],
            "fine_shape": [1024, 1024],
        },
        "case_controls": [
            {"id": "current_512", "slice_count": 400, "all_finite": True},
            {"id": "fine_1024", "slice_count": 400, "all_finite": True},
        ],
        "comparison": {
            "raw_relative_l2": 0.04574990331167789,
            "external_passband_relative_l2": 0.023787308028510038,
        },
        "outcome": {
            "status": "Passed",
            "interpretation_code": "scalar_lateral_reference_closed",
            "stage_b_allowed": True,
        },
    }


def _config() -> dict[str, Any]:
    return {
        "physics": {
            "wavelength_m": 5.32e-7,
            "internal_reference_index": 1.5,
            "external_medium_index": 1.0,
            "angular_spectrum_bandlimit": True,
        },
        "stage_a": {"interface": {"factor": 8}},
    }


def test_normalization_changes_only_case_control_container() -> None:
    metrics = _metrics()
    normalized = repairer._normalise_metrics_for_hdf5(metrics)
    assert isinstance(metrics["case_controls"], list)
    assert list(normalized["case_controls"]) == ["current_512", "fine_1024"]
    assert normalized["comparison"] == metrics["comparison"]
    assert normalized["outcome"] == metrics["outcome"]


def test_repaired_hdf5_is_an_exact_normalized_metrics_round_trip(
    tmp_path: Path,
) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "config.yaml").write_text("physics: {}\n", encoding="utf-8")
    normalized = repairer._normalise_metrics_for_hdf5(_metrics())
    path = repairer._write_repaired_hdf5(
        tmp_path,
        config=_config(),
        metadata={"diagnostic_stage": "R10_stage_a"},
        normalized_metrics=normalized,
    )
    validation = repairer._validate_repaired_hdf5(path, normalized)
    assert validation["full_metrics_exact_round_trip"] is True
    assert validation["case_control_ids"] == ["current_512", "fine_1024"]
    assert validation["stage_b_allowed"] is True


def test_repair_preserves_registered_inputs_and_writes_separate_record(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "registered_run"
    (run_dir / "outputs").mkdir(parents=True)
    save_config(run_dir / "config.yaml", _config())
    save_json(run_dir / "metadata.json", {"diagnostic_stage": "R10_stage_a"})
    save_json(run_dir / "metrics.json", _metrics())
    save_json(
        run_dir / "run_state.json",
        {
            "status": "failed_during_execution",
            "error_type": "TypeError",
            "error": "Object dtype has no native HDF5 equivalent",
            "formal_attempt_retained": True,
        },
    )
    save_json(
        run_dir / "run_progress.json",
        {
            "events": [
                {
                    "event": "case_completed",
                    "case_id": "current_512",
                    "slice_count": 400,
                },
                {
                    "event": "case_completed",
                    "case_id": "fine_1024",
                    "slice_count": 400,
                },
                {"event": "artifacts_writing_started"},
            ],
            "latest_event": {"event": "artifacts_writing_started"},
        },
    )
    with h5py.File(run_dir / "outputs" / "exp040_r10_stage_a.h5", "w"):
        pass
    registered = {
        relative: repairer._sha256(run_dir / relative)
        for relative in (
            "config.yaml",
            "metadata.json",
            "metrics.json",
            "run_progress.json",
            "run_state.json",
            "outputs/exp040_r10_stage_a.h5",
        )
    }
    monkeypatch.setattr(repairer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        repairer, "REGISTERED_RUN_RELATIVE", Path("registered_run")
    )
    monkeypatch.setattr(repairer, "REGISTERED_INPUT_SHA256", registered)

    repaired_path = repairer.repair(run_dir)

    assert repaired_path.name == "exp040_r10_stage_a_repaired.h5"
    record = json.loads(
        (run_dir / "artifact_repair.json").read_text(encoding="utf-8")
    )
    assert record["scientific_recomputation"] is False
    assert record["forward_or_fft_called"] is False
    assert record["formal_run_state_preserved"] is True
    assert record["original_input_sha256_after_repair"] == registered
    assert record["validation"]["full_metrics_exact_round_trip"] is True


def test_repair_script_has_no_scientific_execution_import_or_call() -> None:
    source = Path(repairer.__file__).read_text(encoding="utf-8")
    forbidden = (
        "tgv_ptycho.forward",
        "tgv_ptycho.optics",
        "np.fft",
        "_r7_streamed_tgv_exit",
        "_r9_comparison_metrics",
        "save_exp040_r10_stage_a_figures",
    )
    assert all(value not in source for value in forbidden)
