from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp051_q8_plateau_interval_fit as runner

from tgv_ptycho.inverse.exp051 import (
    Exp051SourceArtifact,
    load_exp051_source_true_probe,
    make_exp051_candidate_generator,
)
from tgv_ptycho.inverse.exp051_plateau_interval import (
    PriorQ8LocalControlArtifact,
    expand_connected_cell_component,
    fixed_membership_bisection,
    load_prior_q8_local_control,
    q8_cell_index,
    q8_cell_midpoint,
    run_q8_plateau_interval_fit,
    validate_exp051_plateau_interval_config,
)
from tgv_ptycho.io.config import load_config, save_config

CONFIG_PATH = Path(
    "configs/experiments/"
    "exp051_TGV_3d_multislice_q8_plateau_interval_fit.yaml"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registered() -> tuple[
    dict[str, Any], Exp051SourceArtifact, PriorQ8LocalControlArtifact
]:
    config = load_config(CONFIG_PATH)
    validate_exp051_plateau_interval_config(config)
    source = load_exp051_source_true_probe(config, PROJECT_ROOT)
    prior = load_prior_q8_local_control(config, PROJECT_ROOT)
    return config, source, prior


@pytest.fixture(scope="module")
def real_preflight_result(
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorQ8LocalControlArtifact
    ],
) -> dict[str, Any]:
    config, source, _ = registered
    generator = make_exp051_candidate_generator(source.source_config)
    return run_q8_plateau_interval_fit(
        config, source.source_config, source.P_B_true, generator
    )


def test_config_prior_provenance_and_forbidden_primary_input(
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorQ8LocalControlArtifact
    ],
) -> None:
    config, source, prior = registered
    assert source.target_hdf5_path == "/entry/truth/P_B_true"
    assert "P_B_rec" not in source.target_hdf5_path
    assert prior.run_state["status"] == "complete"
    assert prior.run_state["artifacts_validated"] is True
    assert prior.metrics["diagnostic_status"] == "DiagnosticPassed"
    assert prior.metrics["reference_validated"] is False
    assert prior.metrics["full_tgv_reference_authorized"] is False

    changed = deepcopy(config)
    changed["plateau_interval"]["boundary_bisection_iterations"] = 25
    with pytest.raises(ValueError, match="frozen design"):
        validate_exp051_plateau_interval_config(changed)
    forbidden = deepcopy(config)
    forbidden["source"]["target_hdf5_path"] = (
        "/entry/reconstruction/matched_q4/P_B_rec"
    )
    with pytest.raises(ValueError, match="P_B_true"):
        validate_exp051_plateau_interval_config(forbidden)
    promoted = deepcopy(config)
    promoted["experiment"]["full_tgv_reference_authorized"] = True
    with pytest.raises(ValueError, match="must remain false"):
        validate_exp051_plateau_interval_config(promoted)


def test_connected_component_disconnected_alias_cap_and_bounds() -> None:
    members = {2, 3, 4, 7}
    component = expand_connected_cell_component(
        lambda index: index in members,
        cell_count=10,
        seed_cell=3,
        max_cells_per_direction=8,
    )
    assert component["left_cell_index"] == 2
    assert component["right_cell_index"] == 4
    assert component["left_outside_cell_index"] == 1
    assert component["right_outside_cell_index"] == 5
    assert 7 not in component["queried_cell_index"]
    assert component["left_cap_hit"] is False
    assert component["right_cap_hit"] is False

    capped = expand_connected_cell_component(
        lambda _: True,
        cell_count=20,
        seed_cell=10,
        max_cells_per_direction=2,
    )
    assert capped["left_cap_hit"] is True
    assert capped["right_cap_hit"] is True
    adjacent = np.asarray([1.0, np.nextafter(1.0, 2.0), 2.0])
    assert q8_cell_midpoint(0, adjacent) == 1.0
    with pytest.raises(ValueError, match="outside q8"):
        q8_cell_index(-1.0, np.asarray([0.0, 1.0, 2.0]))


def test_fixed_budget_membership_bisection() -> None:
    evaluated: list[float] = []

    def objective(value: float) -> tuple[float, int]:
        evaluated.append(value)
        return (0.0 if value >= 0.0 else 1.0), len(evaluated) - 1

    control = fixed_membership_bisection(
        objective,
        inside_m=1.0,
        outside_m=-1.0,
        tau_relative_l2=1.0e-3,
        iterations=24,
    )
    assert control["iterations"] == 24
    assert len(control["evaluated_d_waist_m"]) == 24
    assert control["bracket_lower_m"] <= 0.0 <= control["bracket_upper_m"]
    assert control["bracket_width_m"] == pytest.approx(2.0 / 2**24)
    assert control["final_inside_loss"] == 0.0
    assert control["final_outside_loss"] == 1.0


def test_real_q8_interval_replay_thresholds_multistart_and_endpoint_controls(
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorQ8LocalControlArtifact
    ],
    real_preflight_result: dict[str, Any],
) -> None:
    config, _, _ = registered
    result = real_preflight_result
    assert result["experiment_status"] == "Passed"
    assert result["interpretation"] == (
        "q8_plateau_interval_single_parameter_oracle_fit_passed"
    )
    assert result["replay"]["raw_complex_relative_l2"] == 0.0
    assert result["replay"]["deterministic_repeat_relative_l2"] == 0.0
    assert result["tau"]["tau_primary"] == 1.0e-12
    assert result["tau"]["tau_low"] == 1.0e-13
    assert result["tau"]["tau_high"] == pytest.approx(1.0e-11)
    interval = result["reported_interval"]
    assert interval["width_m"] <= 1.25e-7
    assert interval["truth_inside_simulation_evaluation_only"] is True
    assert interval["truth_to_interval_distance_m"] == 0.0
    assert interval["lower_m"] < float(config["fit"]["true_d_waist_m"])
    assert interval["upper_m"] > float(config["fit"]["true_d_waist_m"])
    assert result["gates"]["threshold_stability_pass"] is True
    assert result["gates"]["geometry_endpoint_convention_pass"] is True
    assert result["gates"]["boundary_bisection_pass"] is True
    assert result["gates"]["multi_start_interval_agreement_pass"] is True
    assert result["gates"]["screened_profile_uniqueness_pass"] is True
    assert result["reference_validated"] is False
    assert result["full_tgv_reference_authorized"] is False
    assert result["p_b_rec_used_as_primary_input"] is False
    assert len(result["optimizer"]["branches"]) == 4
    for branch in result["optimizer"]["branches"].values():
        assert branch["evaluation_count"] == 41
        assert branch["stopping_reason"] == "evaluation_budget"
        assert branch["final_seed_qualifies"] is True
    for branch in result["bisection"].values():
        assert branch["lower"]["iterations"] == 24
        assert branch["upper"]["iterations"] == 24
        assert branch["lower"]["pass"] is True
        assert branch["upper"]["pass"] is True
    cache = result["cache"]
    assert cache["P_B_candidate"].dtype == np.complex128
    assert cache["P_B_candidate"].shape[1:] == (96, 96)
    assert len(cache["q8_cell_index"]) == len(cache["D_waist_m"])
    assert "finite_difference" not in result


def test_runner_hdf5_layout_and_artifact_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorQ8LocalControlArtifact
    ],
    real_preflight_result: dict[str, Any],
) -> None:
    config, source, prior = registered
    local_config = deepcopy(config)
    local_config["output"]["root"] = "runs"
    local_config["output"]["run_name"] = "tiny_exp051_q8_plateau"
    config_path = tmp_path / "plateau.yaml"
    save_config(config_path, local_config)

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_git_commit", lambda *_: "TEST")
    monkeypatch.setattr(
        runner, "load_exp051_source_true_probe", lambda *_: source
    )
    monkeypatch.setattr(runner, "load_prior_q8_local_control", lambda *_: prior)
    monkeypatch.setattr(
        runner,
        "make_exp051_candidate_generator",
        lambda *_args, **_kwargs: lambda _: source.P_B_true,
    )
    monkeypatch.setattr(
        runner,
        "run_q8_plateau_interval_fit",
        lambda *_args, **_kwargs: deepcopy(real_preflight_result),
    )
    run_dir = runner.run(config_path)
    state = json.loads(
        (run_dir / "run_state.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert state["figure_count"] == 3
    assert metrics["experiment_status"] == "Passed"
    assert metrics["reference_validated"] is False
    assert metrics["full_tgv_reference_authorized"] is False
    assert metrics["p_b_rec_used_as_primary_input"] is False
    hdf5_path = run_dir / "outputs" / local_config["output"]["hdf5_filename"]
    with h5py.File(hdf5_path, "r") as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "reconstruction",
            "sample",
            "truth",
        }
        assert "calibration" not in h5["entry"]
        assert "preprocessing" not in h5["entry"]
        names: list[str] = []
        h5["entry"].visit(names.append)
        assert not any(name.rsplit("/", 1)[-1] == "P_B_rec" for name in names)
        root = h5["entry/reconstruction/waist_fit/plateau_interval"]
        assert root["cache/P_B_candidate"].shape[1:] == (96, 96)
        assert root["geometry_control/node_count_stack/seed_midpoint"].shape == (
            100,
            96,
            96,
        )
        assert len(root["optimizer/branches"]) == 4
        assert all(
            int(branch["evaluation_count"][()]) == 41
            for branch in root["optimizer/branches"].values()
        )
        assert bool(h5["entry/truth/reference_validated"][()]) is False
        assert bool(
            h5["entry/truth/full_tgv_reference_authorized"][()]
        ) is False
