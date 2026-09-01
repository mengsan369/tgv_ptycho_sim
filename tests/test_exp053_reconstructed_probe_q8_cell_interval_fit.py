from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp053_reconstructed_probe_q8_cell_interval_fit as runner

from tgv_ptycho.inverse.exp051 import (
    load_exp051_source_true_probe,
    make_exp051_candidate_generator,
)
from tgv_ptycho.inverse.exp051_plateau_interval import (
    q8_cell_index,
    validate_exp051_plateau_interval_config,
)
from tgv_ptycho.inverse.exp053 import (
    Exp051OracleIntervalArtifact,
    Exp053SourceArtifact,
    load_exp051_oracle_interval,
    load_exp053_source_reconstructed_probe,
    make_exp053_candidate_generator,
    run_reconstructed_probe_q8_interval_fit,
    validate_exp053_config,
)
from tgv_ptycho.inverse.waist_fit import raw_complex_probe_loss
from tgv_ptycho.io.config import load_config, save_config

CONFIG_PATH = Path(
    "configs/experiments/"
    "exp053_TGV_3d_multislice_reconstructed_probe_q8_cell_interval_fit.yaml"
)
EXP051_CONFIG_PATH = Path(
    "configs/experiments/"
    "exp051_TGV_3d_multislice_q8_plateau_interval_fit.yaml"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registered() -> tuple[
    dict[str, Any], Exp053SourceArtifact, Exp051OracleIntervalArtifact
]:
    config = load_config(CONFIG_PATH)
    validate_exp053_config(config)
    source = load_exp053_source_reconstructed_probe(config, PROJECT_ROOT)
    oracle = load_exp051_oracle_interval(config, PROJECT_ROOT)
    return config, source, oracle


@pytest.fixture(scope="module")
def real_preflight_result(
    registered: tuple[
        dict[str, Any], Exp053SourceArtifact, Exp051OracleIntervalArtifact
    ],
) -> dict[str, Any]:
    config, source, oracle = registered
    generator = make_exp053_candidate_generator(source.source_config)
    return run_reconstructed_probe_q8_interval_fit(
        config,
        source.source_config,
        source.P_B_rec_raw,
        source.P_B_true_operator_reference,
        oracle,
        generator,
    )


def test_source_loader_accepts_only_raw_matched_q4_and_validates_identity(
    registered: tuple[
        dict[str, Any], Exp053SourceArtifact, Exp051OracleIntervalArtifact
    ],
) -> None:
    config, source, oracle = registered
    assert source.target_hdf5_path == (
        "/entry/reconstruction/exp053_feedback_control/branches/"
        "spectrally_damped_gn_cg/P_B_rec"
    )
    assert source.target_dataset_sha256 == (
        "194C7B950F8DCF2BF94212A6F63270296BC9EC79AE0C04ED557C864F5FE8EB07"
    )
    assert source.P_B_rec_raw.shape == (96, 96)
    assert source.P_B_rec_raw.dtype == np.complex128
    assert np.all(np.isfinite(source.P_B_rec_raw))
    assert source.source_state["status"] == "complete"
    assert source.source_state["artifacts_validated"] is True
    assert source.source_metadata["reference_validated"] is False
    assert source.source_metadata["full_tgv_reference_authorized"] is False
    assert oracle.run_state["status"] == "complete"
    assert oracle.run_state["artifacts_validated"] is True
    assert oracle.metrics["experiment_status"] == "Passed"
    assert oracle.lower_m == 1.9999972701052885e-5
    assert oracle.upper_m == 2.0000143340468626e-5
    optimizer = config["fit"]["optimizer"]
    assert optimizer["evaluation_budget_per_start"] == 43
    assert optimizer["numerical_closure_control"] == {
        "baseline_evaluation_budget_per_start": 41,
        "additional_complete_pattern_updates_per_start": 1,
        "evaluations_per_pattern_update": 2,
        "added_evaluations_per_start": 2,
        "selection_basis": "existing_raw_target_loss_and_41_call_tracks_only",
        "equal_budget_all_starts": True,
        "truth_used_for_method_parameter_or_stopping_selection": False,
    }
    boundary = config["reconstruction_aware_interval"][
        "source_exact_boundary_control"
    ]
    assert boundary == {
        "method": "bounded_float64_nextafter_transition_scan",
        "max_ulps_each_direction": 8,
        "lower_boundary_definition": (
            "first_representable_member_after_outside"
        ),
        "upper_boundary_definition": (
            "first_representable_outside_after_member"
        ),
        "membership_contract": (
            "tau_primary_exact_field_and_node_count_all_agree"
        ),
        "primary_boundary_reference": (
            "source_exact_float64_membership_transition"
        ),
        "analytic_breakpoint_role": (
            "retained_algebraic_diagnostic_only"
        ),
        "truth_used_for_mapping_selection_or_stopping": False,
    }

    forbidden_paths = [
        "/entry/truth/P_B_true",
        (
            "/entry/reconstruction/detector_quadrature_ablation/branches/"
            "matched_q4/simulation_evaluation_only/"
            "P_B_rec_global_phase_aligned"
        ),
        (
            "/entry/reconstruction/detector_quadrature_ablation/branches/"
            "mismatch_q1_point/P_B_rec"
        ),
        (
            "/entry/reconstruction/detector_quadrature_ablation/branches/"
            "matched_q1_point/P_B_rec"
        ),
    ]
    for path in forbidden_paths:
        changed = deepcopy(config)
        changed["source"]["target_hdf5_path"] = path
        with pytest.raises(ValueError, match="matched_q4 raw P_B_rec"):
            validate_exp053_config(changed)
    promoted = deepcopy(config)
    promoted["experiment"]["full_tgv_reference_authorized"] = True
    with pytest.raises(ValueError, match="must remain false"):
        validate_exp053_config(promoted)
    wrong_grid = deepcopy(config)
    wrong_grid["source"]["expected_identity"]["native_shape"] = [64, 64]
    with pytest.raises(ValueError, match="plane/grid/branch"):
        validate_exp053_config(wrong_grid)
    post_hoc_budget = deepcopy(config)
    post_hoc_budget["fit"]["optimizer"]["evaluation_budget_per_start"] = 45
    with pytest.raises(ValueError, match="multi-start changed"):
        validate_exp053_config(post_hoc_budget)
    post_hoc_ulp_budget = deepcopy(config)
    post_hoc_ulp_budget["reconstruction_aware_interval"][
        "source_exact_boundary_control"
    ]["max_ulps_each_direction"] = 9
    with pytest.raises(ValueError, match="frozen design changed"):
        validate_exp053_config(post_hoc_ulp_budget)


def test_operator_replay_is_separate_from_reconstructed_target_mismatch(
    real_preflight_result: dict[str, Any],
) -> None:
    result = real_preflight_result
    assert result["candidate_operator_replay"][
        "raw_complex_relative_l2"
    ] == 0.0
    assert result["candidate_operator_replay"][
        "deterministic_repeat_relative_l2"
    ] == 0.0
    assert result["reconstructed_target_mismatch"][
        "raw_complex_relative_l2"
    ] == pytest.approx(0.10107841397548276, abs=5.0e-16)
    threshold = result["field_equivalence_threshold"]
    assert threshold["tau_primary"] == 1.0e-12
    assert threshold["tau_low"] == 1.0e-13
    assert threshold["tau_high"] == pytest.approx(1.0e-11)
    assert threshold["derived_from_reconstruction_error"] is False
    assert result["primary_input_is_raw_matched_q4_p_b_rec"] is True
    assert result["truth_used_by_fitter"] is False
    assert result["phase_or_scale_alignment"] == "none"


def test_real_reconstructed_interval_multistart_boundary_and_bias(
    real_preflight_result: dict[str, Any],
    registered: tuple[
        dict[str, Any], Exp053SourceArtifact, Exp051OracleIntervalArtifact
    ],
) -> None:
    result = real_preflight_result
    config = registered[0]
    assert result["experiment_status"] == "Passed"
    assert result["interpretation"] == (
        "reconstructed_probe_q8_cell_interval_fit_passed"
    )
    interval = result["reported_interval"]
    analytic_lower = 2.000301349202269e-5
    analytic_upper = 2.000327036622095e-5
    assert interval["analytic_lower_m"] == analytic_lower
    assert interval["analytic_upper_m"] == analytic_upper
    assert interval["lower_m"] == np.nextafter(analytic_lower, -np.inf)
    assert interval["upper_m"] == pytest.approx(
        analytic_upper, abs=1.0e-20
    )
    assert interval["boundary_semantics"] == (
        "source_exact_float64_membership_transition"
    )
    assert interval["lower_closed"] is True
    assert interval["upper_closed"] is False
    assert interval["width_m"] <= 1.25e-7
    edges = np.asarray(result["partition"]["cell_edges_m"])
    assert q8_cell_index(interval["analytic_lower_m"], edges) == (
        interval["left_cell_index"]
    )
    assert q8_cell_index(interval["analytic_upper_m"], edges) == (
        interval["right_cell_index"] + 1
    )
    source_boundary = result["source_exact_boundary_control"]
    assert source_boundary["pass"] is True
    assert source_boundary[
        "truth_used_for_mapping_selection_or_stopping"
    ] is False
    lower_mapping = source_boundary["boundaries"]["lower"]
    upper_mapping = source_boundary["boundaries"]["upper"]
    for mapping in (lower_mapping, upper_mapping):
        assert mapping["pass"] is True
        assert mapping["membership_triplet_agreement"] is True
        assert mapping["unique_monotone_transition"] is True
        assert mapping["scan_cap_hit"] is False
        assert len(mapping["diameter_m"]) == 17
    assert lower_mapping["source_exact_ulp_offset_from_analytic"] == -1
    assert upper_mapping["source_exact_ulp_offset_from_analytic"] == 0
    assert lower_mapping["source_exact_breakpoint_m"] == interval["lower_m"]
    assert upper_mapping["source_exact_breakpoint_m"] == interval["upper_m"]
    optimizer = result["optimizer"]
    assert optimizer["equal_budget"] is True
    assert optimizer["all_final_seeds_qualify"] is True
    assert optimizer["interval_agreement"] is True
    assert optimizer["numerical_closure_control"] == config["fit"][
        "optimizer"
    ]["numerical_closure_control"]
    assert len(optimizer["branches"]) == 4
    for branch in optimizer["branches"].values():
        assert branch["evaluation_count"] == 43
        assert branch["stopping_reason"] == "evaluation_budget"
        assert branch["final_seed_qualifies"] is True
    assert set(result["bisection"]) == set(optimizer["branches"])
    assert all(
        set(branch) == {"lower", "upper"}
        and all(side["iterations"] == 24 for side in branch.values())
        for branch in result["bisection"].values()
    )
    for branch in result["bisection"].values():
        lower = branch["lower"]
        upper = branch["upper"]
        assert lower["pass"] is True
        assert lower["analytic_breakpoint_in_bracket"] is False
        assert lower["source_exact_breakpoint_in_bracket"] is True
        assert lower["source_exact_mapping_pass"] is True
        assert lower["source_exact_ulp_offset_from_analytic"] == -1
        assert upper["pass"] is True
        assert upper["analytic_breakpoint_in_bracket"] is True
        assert upper["source_exact_breakpoint_in_bracket"] is True
        assert upper["source_exact_mapping_pass"] is True
        assert upper["source_exact_ulp_offset_from_analytic"] == 0

    gates = result["gates"]
    assert gates["candidate_operator_replay_pass"] is True
    assert gates["threshold_stability_pass"] is True
    assert gates["source_exact_boundary_mapping_pass"] is True
    assert gates["endpoint_half_open_convention_pass"] is True
    assert gates["boundary_bisection_pass"] is True
    assert gates["numerical_controls_pass"] is True
    assert gates["screened_minimum_uniqueness_pass"] is True
    assert gates["interval_accuracy_pass_simulation_evaluation_only"] is True
    simulation = result["simulation_evaluation_only"]
    assert simulation["truth_inside_reported_interval"] is False
    assert simulation["truth_to_interval_distance_m"] == pytest.approx(
        3.013492022687817e-9
    )
    comparison = result["oracle_interval_comparison"]
    assert comparison["intervals_overlap"] is False
    assert comparison["interval_distance_m"] == pytest.approx(
        2.8701515540631407e-9
    )
    assert comparison["endpoint_hausdorff_displacement_m"] == pytest.approx(
        3.1270257523229213e-9
    )
    loss = result["loss_comparison"]
    assert loss["L_rec_at_best_interval"] < loss["L_rec_at_true_diameter"]
    assert loss["relative_loss_improvement"] == pytest.approx(
        0.0018135631676780646
    )
    directional = simulation["directional_diagnostic"]
    assert directional["performed"] is True
    assert directional["enters_fitter"] is False
    assert set(directional["directions"]) == {
        "minus_0p125um",
        "plus_0p125um",
        "best_candidate_manifold",
    }


def test_exp051_oracle_regression_remains_exact_true_probe_replay() -> None:
    config = load_config(EXP051_CONFIG_PATH)
    validate_exp051_plateau_interval_config(config)
    source = load_exp051_source_true_probe(config, PROJECT_ROOT)
    generator = make_exp051_candidate_generator(source.source_config)
    replay = generator(float(config["fit"]["true_d_waist_m"]))
    assert raw_complex_probe_loss(replay, source.P_B_true) == 0.0
    assert source.target_hdf5_path == "/entry/truth/P_B_true"


def test_tiny_runner_hdf5_json_cache_and_artifact_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered: tuple[
        dict[str, Any], Exp053SourceArtifact, Exp051OracleIntervalArtifact
    ],
    real_preflight_result: dict[str, Any],
) -> None:
    config, source, oracle = registered
    local_config = deepcopy(config)
    local_config["output"]["root"] = "runs"
    local_config["output"]["run_name"] = "tiny_exp053_q8_interval"
    config_path = tmp_path / "exp053.yaml"
    save_config(config_path, local_config)

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_git_commit", lambda *_: "TEST")
    monkeypatch.setattr(
        runner,
        "load_exp053_source_reconstructed_probe",
        lambda *_: source,
    )
    monkeypatch.setattr(runner, "load_exp051_oracle_interval", lambda *_: oracle)
    monkeypatch.setattr(
        runner,
        "make_exp053_candidate_generator",
        lambda *_args, **_kwargs: lambda _: source.P_B_true_operator_reference,
    )
    monkeypatch.setattr(
        runner,
        "run_reconstructed_probe_q8_interval_fit",
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
    assert state["experiment_status"] == "Passed"
    assert state["figure_count"] == 4
    assert metrics["experiment_status"] == "Passed"
    assert metrics["primary_input_is_raw_matched_q4_p_b_rec"] is True
    assert metrics["truth_used_by_fitter"] is False
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
        root = h5[
            "/entry/reconstruction/waist_fit/"
            "reconstructed_target_q8_cell_interval"
        ]
        assert np.array_equal(
            root["source/raw_P_B_rec_input"][...], source.P_B_rec_raw
        )
        names: list[str] = []
        root.visit(names.append)
        assert not any(
            token in name.lower()
            for name in names
            for token in (
                "global_phase_aligned",
                "truth_aligned",
                "p_b_rec_aligned",
            )
        )
        assert root["cache/P_B_candidate"].shape[1:] == (96, 96)
        midpoint_index = int(root["reported_interval/midpoint_cache_index"][()])
        assert np.array_equal(
            root["P_B_best_candidate_raw"][...],
            root["cache/P_B_candidate"][midpoint_index],
        )
        assert len(root["optimizer/branches"]) == 4
        assert all(
            int(branch["evaluation_count"][()]) == 43
            for branch in root["optimizer/branches"].values()
        )
        assert int(
            root[
                "optimizer/numerical_closure_control/"
                "baseline_evaluation_budget_per_start"
            ][()]
        ) == 41
        assert bool(root["source_exact_boundary_control/pass"][()]) is True
        assert int(
            root[
                "source_exact_boundary_control/boundaries/lower/"
                "source_exact_ulp_offset_from_analytic"
            ][()]
        ) == -1
        assert int(
            root[
                "source_exact_boundary_control/boundaries/upper/"
                "source_exact_ulp_offset_from_analytic"
            ][()]
        ) == 0
        assert bool(h5["entry/truth/reference_validated"][()]) is False
        assert bool(
            h5["entry/truth/full_tgv_reference_authorized"][()]
        ) is False
