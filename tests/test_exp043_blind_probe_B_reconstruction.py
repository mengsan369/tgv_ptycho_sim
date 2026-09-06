from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp043_blind_probe_B_reconstruction as runner

from tgv_ptycho.forward.exp040 import relative_l2
from tgv_ptycho.io.config import load_config, save_config
from tgv_ptycho.io.save_load import save_json
from tgv_ptycho.recon.exp042 import (
    build_matched_development_case,
    reconstruct_known_b_probe_damped_gn_cg,
)
from tgv_ptycho.recon.exp043 import (
    Exp042SourceBundle,
    MatchedBlindProbeBOperator,
    alternating_blind_reconstruction,
    block_normal_action,
    centered_support_mask,
    determine_status,
    known_probe_b_control,
    load_exp042_authoritative_source,
    make_b_initialization,
    measurement_only_canonical_pair,
    operator_control_metrics,
    pairwise_blind_stability,
    phase_direction_to_modulation,
    phase_normal_action,
    reciprocal_exit_product_gauge_error,
    sha256_array_bytes,
    sha256_file,
)

EXP042_CONFIG = Path(
    "configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml"
)
EXP043_CONFIG = Path(
    "configs/experiments/exp043_TGV_3d_multislice_blind_probe_B_reconstruction.yaml"
)


def _tiny_exp042_config() -> dict[str, Any]:
    config = load_config(EXP042_CONFIG)
    config["sample_a"].update(
        shape=[32, 32],
        dx_m=1.5e-6,
        thickness_m=2.0e-5,
        target_dz_m=2.0e-6,
        d_top_m=1.5e-5,
        d_waist_m=1.0e-5,
        d_bottom_m=1.5e-5,
        z_waist_m=1.0e-5,
    )
    config["probe_grid"].update(
        native_shape=[32, 32],
        node_dx_m=1.5e-6,
        open_shape=[96, 96],
        native_fov_m=[4.8e-5, 4.8e-5],
        open_fov_m=[1.44e-4, 1.44e-4],
    )
    config["sample_b"].update(support_shape=[64, 64], feature_size_px=2)
    config["scan"].update(
        num_x=3,
        num_y=3,
        step_m=3.0e-6,
        max_jitter_px=0,
        jitter_quantum_m=1.5e-6,
    )
    config["detector"].update(
        node_dx_m=1.5e-6,
        pixel_size_m=6.0e-6,
        native_roi_shape=[16, 16],
        native_roi_fov_m=[9.6e-5, 9.6e-5],
    )
    return config


@pytest.fixture(scope="module")
def tiny_bundle() -> tuple[Exp042SourceBundle, MatchedBlindProbeBOperator]:
    config = _tiny_exp042_config()
    case = build_matched_development_case(config)
    support = centered_support_mask((96, 96), (64, 64))
    bundle = Exp042SourceBundle(
        source_run=Path("synthetic"),
        source_hdf5=Path("synthetic.h5"),
        source_config=config,
        operator=case["operator"],
        I_stack=case["I_stack"],
        scan_positions=case["scan_positions"],
        P_B_true=case["P_B_true"],
        B_true=case["B_true"],
        authoritative_P_B_rec=case["P_B_true"],
        support_mask=support,
        provenance={},
        replay={"independent_forward_replay_relative_l2": 0.0},
    )
    return bundle, MatchedBlindProbeBOperator(case["operator"], support)


def _block_settings() -> dict[str, Any]:
    return {
        "algorithm": "block_spectrally_damped_gn_cg_armijo",
        "outer_iterations": 1,
        "spectral_radius_power_iterations": 2,
        "spectral_radius_seed": 41,
        "damping_relative_to_spectral_radius": 1.0e-4,
        "cg_max_iterations": 2,
        "cg_relative_residual_tolerance": 0.0,
        "line_search_initial_scale": 1.0,
        "backtracking_factor": 0.5,
        "armijo_c": 1.0e-4,
        "max_backtracking_steps": 12,
        "minimum_step": 1.0e-12,
        "gradient_norm_stop": 0.0,
        "truth_used_by_optimizer": False,
        "truth_used_by_spectral_radius": False,
        "truth_used_by_damping": False,
        "truth_used_by_stopping": False,
    }


def _blind_settings() -> dict[str, Any]:
    block = _block_settings()
    sample_b_block = deepcopy(block)
    sample_b_block["parameterization"] = "phase_only_unit_modulus_on_active_support"
    return {
        "algorithm": "alternating_block_spectrally_damped_gn_cg",
        "outer_sweeps": 1,
        "checkpoint_selection_rule": "fixed_final_outer_sweep",
        "probe_block": deepcopy(block),
        "sample_b_block": sample_b_block,
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "truth_used_by_checkpoint_selection": False,
        "truth_used_by_stopping": False,
    }


def _write_tiny_source(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _tiny_exp042_config()
    case = build_matched_development_case(config)
    reconstruction = reconstruct_known_b_probe_damped_gn_cg(
        case["operator"],
        case["I_stack"],
        case["operator"].homogeneous_probe_native,
        config["reconstruction"],
        config["reconstruction"]["exp053_feedback_control"],
    )
    source_run = tmp_path / "source_run"
    output = source_run / "outputs" / "exp042_probe_reconstruction.h5"
    output.parent.mkdir(parents=True)
    save_config(source_run / "config.yaml", config)
    save_json(
        source_run / "metadata.json",
        {
            "experiment_id": "exp042",
            "operator_branch": "R8 unified q8 finite-B open q4 scalar working model",
            "git_commit": "test-commit",
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
        },
    )
    save_json(source_run / "metrics.json", {"test": True})
    save_json(
        source_run / "run_state.json",
        {"status": "complete", "artifacts_validated": True},
    )
    paths = {
        "I_stack": "/entry/data/I_stack",
        "scan_positions": "/entry/data/scan_positions",
        "P_B_true": "/entry/truth/P_B_true",
        "B_true": "/entry/truth/B_true",
        "authoritative_P_B_rec": (
            "/entry/reconstruction/exp053_feedback_control/branches/"
            "spectrally_damped_gn_cg/P_B_rec"
        ),
    }
    arrays = {
        "I_stack": case["I_stack"],
        "scan_positions": case["scan_positions"],
        "P_B_true": case["P_B_true"],
        "B_true": case["B_true"],
        "authoritative_P_B_rec": reconstruction["P_B_rec"],
    }
    with h5py.File(output, "w") as h5:
        for key, path in paths.items():
            h5.create_dataset(path, data=arrays[key])
    source = {
        "experiment": "exp042",
        "run": "source_run",
        "hdf5": "outputs/exp042_probe_reconstruction.h5",
        "config": "config.yaml",
        "metadata": "metadata.json",
        "metrics": "metrics.json",
        "run_state": "run_state.json",
        "file_sha256": {
            "hdf5": sha256_file(output),
            "config": sha256_file(source_run / "config.yaml"),
            "metadata": sha256_file(source_run / "metadata.json"),
            "metrics": sha256_file(source_run / "metrics.json"),
            "run_state": sha256_file(source_run / "run_state.json"),
        },
        "datasets": paths,
        "dataset_sha256": {
            key: sha256_array_bytes(value) for key, value in arrays.items()
        },
        "operator_branch": "R8 unified q8 finite-B open q4 scalar working model",
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
    }
    return source, case


def test_source_loader_hash_state_dataset_and_replay_identity(tmp_path: Path) -> None:
    source, case = _write_tiny_source(tmp_path)
    bundle = load_exp042_authoritative_source(tmp_path, source)

    assert np.array_equal(bundle.I_stack, case["I_stack"])
    assert np.array_equal(bundle.scan_positions, case["scan_positions"])
    assert np.array_equal(bundle.P_B_true, case["P_B_true"])
    assert np.array_equal(bundle.B_true, case["B_true"])
    assert bundle.replay["independent_forward_replay_exact"] is True
    assert bundle.replay["independent_forward_replay_relative_l2"] == 0.0
    assert bundle.provenance["axis_order"] == ["y", "x"]
    assert bundle.provenance["scan_position_columns"] == ["x", "y"]

    corrupt = deepcopy(source)
    corrupt["dataset_sha256"]["I_stack"] = "0" * 64
    with pytest.raises(ValueError, match="dataset"):
        load_exp042_authoritative_source(tmp_path, corrupt)


def test_b_jvp_adjoint_gradient_normal_and_combined_controls(
    tiny_bundle: tuple[Exp042SourceBundle, MatchedBlindProbeBOperator],
) -> None:
    bundle, operator = tiny_bundle
    metrics = operator_control_metrics(
        operator,
        bundle,
        {
            "random_seed": 12,
            "finite_difference_relative_step": 1.0e-6,
            "gauge_test_factor_real": 1.1,
            "gauge_test_factor_imag": 0.2,
        },
    )

    assert metrics["B_intensity_jacobian_directional_relative_error"] < 2e-6
    assert metrics["B_intensity_jacobian_real_adjoint_relative_error"] < 1e-11
    assert metrics["combined_loss_directional_gradient_relative_error"] < 2e-6
    assert metrics["B_normal_action_symmetry_relative_error"] < 1e-11
    assert metrics["B_normal_action_psd_quadratic_form"] >= -1e-14
    assert metrics["B_phase_jacobian_directional_relative_error"] < 2e-6
    assert metrics["B_phase_jacobian_real_adjoint_relative_error"] < 1e-11
    assert metrics["B_phase_normal_action_symmetry_relative_error"] < 1e-11
    assert metrics["B_phase_normal_action_psd_quadratic_form"] >= -1e-14
    assert metrics["support_projection_exterior_max_abs"] == 0.0
    assert metrics["truth_loss"] < 1e-30


def test_b_normal_action_support_and_invalid_inputs(
    tiny_bundle: tuple[Exp042SourceBundle, MatchedBlindProbeBOperator],
) -> None:
    bundle, operator = tiny_bundle
    direction = np.ones(operator.open_shape, dtype=np.complex128)
    action = block_normal_action(
        operator,
        bundle.P_B_true,
        bundle.B_true - 1.0,
        direction,
        "sample_b",
    )
    assert action.shape == operator.open_shape
    assert np.max(np.abs(action[~bundle.support_mask])) == 0.0
    phase_action = phase_normal_action(
        operator,
        bundle.P_B_true,
        bundle.B_true - 1.0,
        np.ones(operator.open_shape),
    )
    assert np.max(np.abs(np.imag(phase_action))) == 0.0
    assert np.max(np.abs(phase_action[~bundle.support_mask])) == 0.0
    with pytest.raises(ValueError, match="real-valued"):
        phase_direction_to_modulation(
            operator,
            bundle.B_true - 1.0,
            np.ones(operator.open_shape, dtype=np.complex128) * (1.0 + 1.0j),
        )
    with pytest.raises(ValueError, match="finite"):
        bad = bundle.B_true.copy()
        bad[0, 0] = np.nan
        operator.project_modulation(bad)
    with pytest.raises(ValueError, match="shape"):
        block_normal_action(
            operator,
            bundle.P_B_true,
            bundle.B_true - 1.0,
            np.ones((4, 4), dtype=np.complex128),
            "sample_b",
        )


def test_gauge_invariance_and_pinned_measurement_canonicalization(
    tiny_bundle: tuple[Exp042SourceBundle, MatchedBlindProbeBOperator],
) -> None:
    bundle, operator = tiny_bundle
    probe_open = operator.probe_open(bundle.P_B_true)
    error = reciprocal_exit_product_gauge_error(probe_open, bundle.B_true, 1.1 + 0.2j)
    assert error < 1e-15
    canonical = measurement_only_canonical_pair(bundle.P_B_true, bundle.B_true - 1.0)
    assert canonical["uses_simulation_truth"] is False
    assert np.array_equal(canonical["P_B_canonical"], bundle.P_B_true)
    assert np.array_equal(canonical["B_canonical"], bundle.B_true)
    before = operator.predict_stack(bundle.P_B_true, bundle.B_true - 1.0)
    after = operator.predict_stack(
        canonical["P_B_canonical"], canonical["B_canonical"] - 1.0
    )
    assert np.array_equal(before, after)


def test_known_probe_b_and_blind_are_deterministic_and_truth_free(
    tiny_bundle: tuple[Exp042SourceBundle, MatchedBlindProbeBOperator],
) -> None:
    bundle, operator = tiny_bundle
    control_settings = {
        "initialization": {
            "kind": "homogeneous_transmission",
            "seed": 4,
            "phase_rms_rad": 0.05,
        },
        "optimizer": {
            **_block_settings(),
            "parameterization": "phase_only_unit_modulus_on_active_support",
        },
    }
    control = known_probe_b_control(operator, bundle, control_settings)
    assert control["true_probe_role"] == "simulation_diagnostic_only_fixed_input"
    assert control["truth_used_by_optimizer"] is False
    assert control["loss_curve"][-1] < control["loss_curve"][0]
    assert np.max(np.abs(control["B_rec_raw"][~bundle.support_mask] - 1.0)) == 0.0
    active_amplitude_error = np.max(
        np.abs(np.abs(control["B_rec_raw"][bundle.support_mask]) - 1.0)
    )
    assert active_amplitude_error < 1e-14

    init = make_b_initialization(
        bundle.support_mask,
        {"kind": "seeded_zero_mean_phase", "seed": 5, "phase_rms_rad": 0.05},
    )
    settings = _blind_settings()
    first = alternating_blind_reconstruction(
        operator,
        bundle.I_stack,
        operator.reference.homogeneous_probe_native,
        init,
        settings,
    )
    second = alternating_blind_reconstruction(
        operator,
        bundle.I_stack,
        operator.reference.homogeneous_probe_native,
        init,
        settings,
    )
    assert np.array_equal(first["P_B_rec_raw"], second["P_B_rec_raw"])
    assert np.array_equal(first["B_rec_raw"], second["B_rec_raw"])
    assert first["loss_curve"][-1] < first["loss_curve"][0]
    assert first["checkpoint_selection_rule"] == "fixed_final_outer_sweep"
    assert first["truth_used_by_optimizer"] is False
    stability = pairwise_blind_stability({"a": first, "b": second})
    assert stability["maximum_pairwise_prediction_relative_l2"] == 0.0
    assert stability["truth_used"] is False


@pytest.mark.parametrize(
    ("source_ok", "p_ok", "b_ok", "detector", "p_error", "b_error", "expected"),
    [
        (False, True, True, 0.0, 0.0, 0.0, "Inconclusive"),
        (True, False, True, 0.0, 0.0, 0.0, "Inconclusive"),
        (True, True, True, 0.01, 0.1, 0.1, "Passed"),
        (True, True, True, 0.01, 0.4, 0.4, "Failed"),
        (True, True, True, 0.03, 0.1, 0.1, "Failed"),
    ],
)
def test_status_matrix(
    source_ok: bool,
    p_ok: bool,
    b_ok: bool,
    detector: float,
    p_error: float,
    b_error: float,
    expected: str,
) -> None:
    metrics = {
        "source_operator_gates_passed": source_ok,
        "known_b_probe_control_gates_passed": p_ok,
        "known_probe_B_control_gates_passed": b_ok,
        "blind_primary": {
            "representative_probe_aligned_relative_l2_simulation_only": p_error,
            "representative_B_aligned_active_relative_l2_simulation_only": b_error,
            "representative_exit_wave_product_relative_l2_simulation_only": max(
                p_error, b_error
            ),
            "representative_detector_relative_residual": detector,
            "maximum_branch_detector_relative_residual": detector,
            "maximum_pairwise_prediction_relative_l2": 0.0,
            "all_loss_nonincreasing": True,
            "all_finite": True,
            "all_fixed_final_selection": True,
        },
    }
    thresholds = {
        "blind_probe_aligned_relative_l2_max": 0.25,
        "blind_B_aligned_active_relative_l2_max": 0.35,
        "blind_exit_wave_product_relative_l2_max": 0.35,
        "blind_detector_relative_residual_max": 0.02,
        "repeat_prediction_relative_l2_max": 0.02,
    }
    assert determine_status(metrics, thresholds)["scientific_status"] == expected


def test_runner_writes_raw_canonical_aligned_provenance_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _write_tiny_source(tmp_path)
    config = load_config(EXP043_CONFIG)
    config["source"] = source
    config["execution"].update(
        stage="development_preflight", scientific_gates_enabled=False
    )
    config["known_probe_B_control"]["optimizer"] = _block_settings()
    config["blind"].update(
        outer_sweeps=1,
        initializations=[
            {
                "name": "homogeneous_B",
                "kind": "homogeneous_transmission",
                "seed": 8,
                "phase_rms_rad": 0.05,
                "seed_offset": 0,
            }
        ],
    )
    config["blind"]["probe_block"] = _block_settings()
    config["blind"]["sample_b_block"] = {
        **_block_settings(),
        "parameterization": "phase_only_unit_modulus_on_active_support",
    }
    config["output"].update(root="runs", run_name="tiny_exp043")
    config_path = tmp_path / "exp043.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    with (run_dir / "run_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert state["all_numeric_hdf5_datasets_finite"] is True
    assert state["figure_count"] == 5
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert metrics["scientific_status"] == "Development"
    assert metrics["truth_use"]["optimizer"] is False
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    with h5py.File(hdf5_path, "r") as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "sample",
            "truth",
            "reconstruction",
            "metadata",
            "metrics",
        }
        root = h5["entry/reconstruction/blind_joint/representative"]
        assert root["P_B_rec_raw"].dtype == np.dtype(np.complex128)
        assert root["B_rec_raw"].dtype == np.dtype(np.complex128)
        assert "P_B_rec_canonical" in root
        assert "B_rec_canonical" in root
        assert "simulation_evaluation_only" in root
        assert "source_provenance" in h5["entry/reconstruction"]
        assert "checkpoints" in h5["entry/reconstruction"]
        assert "calibration" not in h5["entry"]
        assert "preprocessing" not in h5["entry"]
        assert (
            sha256_array_bytes(h5["entry/data/I_stack"][...])
            == source["dataset_sha256"]["I_stack"]
        )
        replay = relative_l2(
            root["prediction_final"][...], h5["entry/data/I_stack"][...]
        )
        assert replay == pytest.approx(
            metrics["blind_primary"]["representative_detector_relative_residual"]
        )
