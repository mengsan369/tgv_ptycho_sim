from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "repair_exp040_r10_stage_b_artifacts.py"
RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp040_TGV_3d_multislice_r10_stage_b_20260815_181819"
)


def _load_repair():
    spec = importlib.util.spec_from_file_location("r10_stage_b_repair", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_failed_run_hashes_and_replay_are_exact() -> None:
    repair = _load_repair()
    metrics, state = repair._verify_registered_inputs(RUN_DIR)
    assert state["status"] == "failed_during_execution"
    helmholtz, multislice = repair._load_checkpoints(RUN_DIR)
    runner = repair._formal_runner()
    config = repair.load_config(RUN_DIR / "config.yaml")
    result = repair._replay_postprocessing(
        runner, config, helmholtz, multislice, metrics
    )
    assert all(result["replay_checks"].values())
    assert result["metrics"]["status"] == "Failed"
    assert (
        result["metrics"]["interpretation_code"]
        == "helmholtz_reference_not_validated"
    )
    with pytest.raises(RuntimeError, match="forbidden"):
        runner._solve_helmholtz_case(None, None)


def test_repaired_artifact_contract_in_temporary_directory(tmp_path) -> None:
    repair = _load_repair()
    metrics, _ = repair._verify_registered_inputs(RUN_DIR)
    helmholtz, multislice = repair._load_checkpoints(RUN_DIR)
    runner = repair._formal_runner()
    config = repair.load_config(RUN_DIR / "config.yaml")
    metadata = repair._load_json(RUN_DIR / "metadata.json")
    result = repair._replay_postprocessing(
        runner, config, helmholtz, multislice, metrics
    )
    plot_result = dict(result)
    radial = dict(result["radial_profiles"])
    radial["multislice_passband"] = radial[
        "multislice_fine_1024_passband"
    ]
    plot_result["radial_profiles"] = radial
    figures = runner.save_exp040_r10_stage_b_figures(
        plot_result, tmp_path / "figures"
    )
    hdf5_path = tmp_path / "repaired.h5"
    runner._write_hdf5(
        hdf5_path,
        config=config,
        metadata=metadata,
        result=result,
    )
    validation = repair._validate_outputs(hdf5_path, figures, result)
    assert validation["full_metrics_exact_round_trip"] is True
    assert validation["selected_complex_fields_exact_round_trip"] is True
    assert validation["figure_count"] == 3


def test_repair_rejects_any_other_run(tmp_path) -> None:
    repair = _load_repair()
    with pytest.raises(ValueError, match="only accepts"):
        repair._verify_exact_run_dir(tmp_path)
