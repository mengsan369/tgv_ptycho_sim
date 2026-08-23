from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp042_probe_reconstruction as runner

from tgv_ptycho.io.config import load_config, save_config
from tgv_ptycho.io.save_load import save_json
from tgv_ptycho.recon.exp042 import (
    MATCHED_PIXEL_AVERAGE_READOUT,
    Q1_POINT_MISMATCH_READOUT,
    MatchedKnownBProbeOperator,
    build_matched_development_case,
    detector_quadrature_ablation_consistency_metrics,
    gauss_newton_directional_curvature,
    gauss_newton_step_diagnostic,
    intensity_jacobian_adjoint_relative_error,
    intensity_jacobian_directional_relative_error,
    lanczos_dimension_convergence_diagnostic,
    linear_adjoint_relative_error,
    loss_directional_gradient_relative_error,
    make_detector_readout_operator,
    make_deterministic_complex_probe_initialization,
    matrix_free_lanczos_spectral_measure,
    normalized_intensity_jacobian_sensitivity,
    operator_consistency_metrics,
    point_readout_adjoint_relative_error,
    quadrature_adjoint_relative_error,
    reconstruct_known_b_probe,
    shift_adjoint_relative_error,
    simulation_evaluation_only,
    truth_free_detector_quadrature_ablation_diagnostic,
    truth_free_initialization_ablation_diagnostic,
    truth_free_initialization_stability_diagnostic,
    truth_free_local_spectral_diagnostic,
    truth_free_paired_continuation_diagnostic,
    truth_free_pairwise_conditioning_diagnostic,
    validate_exp042_config,
)

CONFIG_PATH = Path(
    "configs/experiments/exp042_TGV_3d_multislice_probe_reconstruction.yaml"
)


@pytest.fixture(scope="module")
def development_case() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(CONFIG_PATH)
    return config, build_matched_development_case(config)


def test_exp042_config_and_shapes(
    development_case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, case = development_case
    validate_exp042_config(config)
    operator = case["operator"]
    assert isinstance(operator, MatchedKnownBProbeOperator)
    assert case["P_B_true"].shape == (96, 96)
    assert case["B_true"].shape == (256, 256)
    assert case["I_stack"].shape == (25, 32, 32)
    assert case["scan_positions"].shape == (25, 2)
    assert case["P_B_true"].dtype == np.complex128
    assert case["I_stack"].dtype == np.float64
    ablation = config["verification"]["detector_quadrature_ablation"]
    assert config["execution"]["mode"] == (
        "detector_quadrature_three_branch_control"
    )
    assert ablation["enabled"] is True
    assert ablation["equal_iteration_budget"] == 60
    assert ablation["checkpoint_interval"] == 5
    assert ablation["reference_branch"] == "matched_q4"
    assert [branch["detector_readout"] for branch in ablation["branches"]] == [
        MATCHED_PIXEL_AVERAGE_READOUT,
        Q1_POINT_MISMATCH_READOUT,
        Q1_POINT_MISMATCH_READOUT,
    ]
    assert [branch["data_readout"] for branch in ablation["branches"]] == [
        MATCHED_PIXEL_AVERAGE_READOUT,
        MATCHED_PIXEL_AVERAGE_READOUT,
        Q1_POINT_MISMATCH_READOUT,
    ]
    assert ablation["primary_q4_data_branches"] == [
        "matched_q4",
        "mismatch_q1_point",
    ]
    assert ablation["same_truth_b_scan_initialization"] is True
    assert ablation["truth_used_by_branch_selection"] is False
    assert ablation["truth_used_by_stopping"] is False
    assert ablation["q1_branch_is_exp040_matched"] is False
    assert ablation["q1_q1_matched_control_included"] is True
    stability = config["reconstruction"]["initialization_stability_control"]
    assert stability["enabled"] is False
    assert stability["truth_used_by_initialization"] is False
    assert stability["relative_l2_to_primary"] == 0.05
    conditioning = config["verification"]["pairwise_conditioning"]
    assert conditioning["truth_used_by_diagnostic"] is False
    assert conditioning["random_direction_count"] == 8
    continuation = config["reconstruction"]["paired_continuation_control"]
    assert continuation["additional_iterations"] == 60
    assert continuation["checkpoint_interval"] == 10
    assert continuation["truth_used_by_continuation"] is False
    assert continuation["truth_used_by_diagnostic"] is False
    spectral = config["verification"]["local_spectral_diagnostic"]
    assert spectral["enabled"] is False
    assert spectral["krylov_dimension"] == 24
    assert spectral["reorthogonalization_passes"] == 2
    assert spectral["truth_used_by_diagnostic"] is False
    convergence = spectral["dimension_convergence_control"]
    assert convergence["reference_krylov_dimension"] == 12
    assert convergence["extended_krylov_dimension"] == 24
    assert convergence["single_extension_only"] is True
    assert convergence["truth_used_by_control"] is False
    assert np.allclose(
        case["scan_positions"] / operator.node_dx_m,
        np.rint(case["scan_positions"] / operator.node_dx_m),
    )


def test_truth_replay_detector_controls_and_determinism(
    development_case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, case = development_case
    metrics = operator_consistency_metrics(case, config)
    assert metrics["truth_replay_relative_l2"] == 0.0
    assert metrics["deterministic_repeat_relative_l2"] == 0.0
    assert metrics["detector_constant_max_abs_error"] == 0.0
    assert metrics["detector_sum_relative_error"] == 0.0
    assert metrics["detector_node_geometry_normalized_error"] < 1.0e-12
    assert metrics["detector_minimum_intensity"] >= 0.0
    assert metrics["detector_all_nonnegative"] is True
    assert metrics["finite_b_shift_boundary_edge_max_abs_modulation"] == 0.0
    assert metrics["intensity_jacobian_directional_relative_error"] < 2.0e-6
    assert metrics["all_arrays_finite"] is True


def test_linear_shift_and_quadrature_adjoint_dot_products(
    development_case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, case = development_case
    operator = case["operator"]
    seed = int(config["verification"]["random_seed"])
    assert linear_adjoint_relative_error(operator, seed) < 1.0e-11
    assert (
        intensity_jacobian_adjoint_relative_error(
            operator, case["P_B_true"], seed + 3
        )
        < 1.0e-11
    )
    assert shift_adjoint_relative_error(operator, seed + 1) < 1.0e-12
    assert quadrature_adjoint_relative_error(operator, seed + 2) < 1.0e-12


def test_full_loss_gradient_and_zero_residual_fixed_point(
    development_case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, case = development_case
    operator = case["operator"]
    measured = case["I_stack"]
    truth = case["P_B_true"]
    seed = int(config["verification"]["random_seed"])
    rng = np.random.default_rng(seed)
    probe = operator.homogeneous_probe_native * (
        1.0
        + 0.02
        * (
            rng.normal(size=operator.native_shape)
            + 1j * rng.normal(size=operator.native_shape)
        )
    )
    error = loss_directional_gradient_relative_error(
        operator,
        probe,
        measured,
        seed=seed + 3,
        relative_step=float(
            config["verification"]["finite_difference_relative_step"]
        ),
    )
    assert error < 2.0e-6

    loss, gradient, replay = operator.loss_and_gradient(truth, measured)
    assert loss == 0.0
    assert np.max(np.abs(gradient)) == 0.0
    assert np.array_equal(replay, measured)
    assert np.array_equal(truth - gradient, truth)

    fixed_point = reconstruct_known_b_probe(
        operator,
        measured,
        truth,
        config["reconstruction"],
    )
    assert fixed_point["iterations_completed"] == 0
    assert fixed_point["stopping_reason"] == "gradient_norm"
    assert np.array_equal(fixed_point["P_B_rec"], truth)


def test_q1_point_mismatch_controls_and_truth_free_diagnostic(
    development_case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, case = development_case
    matched = case["operator"]
    point = make_detector_readout_operator(
        matched, Q1_POINT_MISMATCH_READOUT
    )
    assert point.detector_readout == Q1_POINT_MISMATCH_READOUT
    assert point.transfer_bc is matched.transfer_bc
    assert point.finite_b_modulation_open is matched.finite_b_modulation_open
    assert point.predict_stack(case["P_B_true"]).shape == case["I_stack"].shape
    assert not np.array_equal(
        point.predict_stack(case["P_B_true"]), case["I_stack"]
    )
    tolerances = config["verification"]["detector_quadrature_ablation"][
        "numerical_control_tolerances"
    ]
    controls = detector_quadrature_ablation_consistency_metrics(case, config)
    assert controls["matched_q4_truth_replay_exact"] is True
    assert controls["matched_q4_truth_replay_relative_l2"] == 0.0
    assert controls["q1_self_truth_replay_exact"] is True
    assert controls["q1_self_truth_replay_relative_l2"] == 0.0
    assert controls["q1_self_truth_loss"] == 0.0
    assert controls["q1_self_zero_residual_gradient_l2_norm"] == 0.0
    assert controls["q1_self_zero_residual_prediction_exact"] is True
    assert controls["q1_self_truth_fixed_point_exact"] is True
    assert controls["q1_on_q4_truth_replay_exact"] is False
    assert controls["q1_on_q4_truth_replay_relative_l2"] > 0.0
    assert controls["q1_self_data_relative_l2_to_q4"] > 0.0
    assert controls["q1_self_data_all_nonnegative"] is True
    assert controls["q1_deterministic_repeat_relative_l2"] == 0.0
    assert controls["prediction_shapes_equal"] is True
    assert controls["shared_linear_array_components_identity"] is True
    assert controls["shared_linear_scalar_components_exact"] is True
    assert controls["only_registered_readout_differs"] is True
    assert controls["q1_branch_is_exp040_matched"] is False
    assert controls["q1_q1_matched_control_included"] is True
    assert controls["q1_point_readout_adjoint_relative_error"] < float(
        tolerances["point_readout_adjoint_relative_error"]
    )
    assert controls["q1_intensity_jacobian_adjoint_relative_error"] < float(
        tolerances["intensity_jacobian_adjoint_relative_error"]
    )
    assert controls[
        "q1_full_loss_directional_gradient_relative_error_on_q4_data"
    ] < float(tolerances["full_loss_directional_gradient_relative_error"])
    assert controls[
        "q1_full_loss_directional_gradient_relative_error_on_q1_data"
    ] < float(tolerances["full_loss_directional_gradient_relative_error"])
    assert point_readout_adjoint_relative_error(point, 20260848) < 1.0e-12

    settings = dict(config["reconstruction"])
    settings["iterations"] = 2
    initial = matched.homogeneous_probe_native.copy()
    operators = {
        "matched_q4": matched,
        "mismatch_q1_point": point,
        "matched_q1_point": point,
    }
    q1_data = point.predict_stack(case["P_B_true"])
    measurements = {
        "matched_q4": case["I_stack"],
        "mismatch_q1_point": case["I_stack"],
        "matched_q1_point": q1_data,
    }
    reconstructions = {
        name: reconstruct_known_b_probe(
            operator, measurements[name], initial, settings
        )
        for name, operator in operators.items()
    }
    diagnostic = truth_free_detector_quadrature_ablation_diagnostic(
        reconstructions,
        operators,
        reference_branch="matched_q4",
        checkpoint_interval=1,
    )
    assert diagnostic["initial_probe_exact_equal"] is True
    assert diagnostic["initial_pairwise_raw_probe_relative_l2"][0, 1] == 0.0
    assert diagnostic["initial_pairwise_prediction_relative_l2"][0, 1] > 0.0
    assert diagnostic["initial_pairwise_prediction_relative_l2"][1, 2] == 0.0
    assert diagnostic[
        "checkpoint_pairwise_raw_probe_relative_l2_curve"
    ].shape == (3, 3, 3)
    assert np.asarray(
        diagnostic["checkpoint_pairwise_prediction_relative_l2_curve"]
    ).shape == (3, 3, 3)
    assert np.array_equal(
        diagnostic["checkpoint_pairwise_prediction_relative_l2_curve"][-1],
        diagnostic["final_pairwise_prediction_relative_l2"],
    )
    assert diagnostic["truth_used_by_diagnostic"] is False


def test_intensity_jacobian_and_gauss_newton_curvature(
    development_case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, case = development_case
    operator = case["operator"]
    seed = int(config["verification"]["random_seed"])
    rng = np.random.default_rng(seed + 20)
    probe = operator.homogeneous_probe_native * (
        1.0
        + 0.02
        * (
            rng.normal(size=operator.native_shape)
            + 1j * rng.normal(size=operator.native_shape)
        )
    )
    direction = rng.normal(size=operator.native_shape) + 1j * rng.normal(
        size=operator.native_shape
    )
    error = intensity_jacobian_directional_relative_error(
        operator,
        probe,
        direction,
        relative_step=float(
            config["verification"]["finite_difference_relative_step"]
        ),
    )
    curvature = gauss_newton_directional_curvature(
        operator, probe, direction
    )
    _, gradient, _ = operator.loss_and_gradient(probe, case["I_stack"])
    diagnostic = gauss_newton_step_diagnostic(
        operator,
        probe,
        gradient,
        fallback_step=float(config["reconstruction"]["initial_step"]),
    )
    assert error < 2.0e-6
    assert np.isfinite(curvature)
    assert curvature >= 0.0
    assert diagnostic["used_fallback"] is False
    assert diagnostic["proposed_step"] == pytest.approx(
        diagnostic["gradient_squared_norm"]
        / diagnostic["gauss_newton_directional_curvature"]
    )


def test_fully_reorthogonalized_lanczos_matches_real_diagonal_spectrum() -> None:
    real_diagonal = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    imaginary_diagonal = np.asarray([[5.0, 6.0], [7.0, 8.0]])

    def matvec(values: np.ndarray) -> np.ndarray:
        return real_diagonal * values.real + 1j * imaginary_diagonal * values.imag

    start = np.ones((2, 2), dtype=np.complex128) * (1.0 + 1.0j)
    reference = matrix_free_lanczos_spectral_measure(
        matvec,
        start,
        krylov_dimension=4,
        reorthogonalization_passes=2,
        breakdown_relative_tolerance=1.0e-13,
    )
    result = matrix_free_lanczos_spectral_measure(
        matvec,
        start,
        krylov_dimension=8,
        reorthogonalization_passes=2,
        breakdown_relative_tolerance=1.0e-13,
    )
    assert result["completed_krylov_dimension"] == 8
    assert np.allclose(result["ritz_values"], np.arange(1.0, 9.0))
    assert np.sum(result["spectral_weights"]) == pytest.approx(1.0)
    assert result["basis_real_orthogonality_max_abs_error"] < 1.0e-12
    assert result["maximum_ritz_residual_l2_estimate"] < 1.0e-10
    assert result["rayleigh_reconstruction_relative_error"] < 1.0e-12
    shared = {
        "matrix_operator": "real_intensity_jacobian_transpose_times_jacobian",
        "matrix_normalization": "mean_over_detector_stack",
        "random_seed": 7,
        "start_vectors": [
            "pairwise_direction",
            "deterministic_zero_mean_complex_gaussian",
        ],
    }
    convergence = lanczos_dimension_convergence_diagnostic(
        {**shared, "pairwise_start": reference, "random_start": reference},
        {**shared, "pairwise_start": result, "random_start": result},
        reference_krylov_dimension=4,
        reported_low_ritz_count=2,
    )
    assert convergence["both_recurrence_prefixes_exact"] is True
    assert convergence["pairwise_start"]["prefix_ritz_values_exact"] is True
    assert convergence["pairwise_start"][
        "prefix_spectral_weights_exact"
    ] is True
    assert convergence["truth_used_by_control"] is False


def test_known_b_recovery_decreases_loss_and_probe_error(
    development_case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, case = development_case
    operator = case["operator"]
    settings = dict(config["reconstruction"])
    settings["iterations"] = 8
    result = reconstruct_known_b_probe(
        operator,
        case["I_stack"],
        operator.homogeneous_probe_native,
        settings,
    )
    stability_settings = settings["initialization_stability_control"]
    control_init = make_deterministic_complex_probe_initialization(
        operator.homogeneous_probe_native,
        seed=int(stability_settings["seed"]),
        relative_l2_to_primary=float(
            stability_settings["relative_l2_to_primary"]
        ),
    )
    control_init_repeat = make_deterministic_complex_probe_initialization(
        operator.homogeneous_probe_native,
        seed=int(stability_settings["seed"]),
        relative_l2_to_primary=float(
            stability_settings["relative_l2_to_primary"]
        ),
    )
    control = reconstruct_known_b_probe(
        operator,
        case["I_stack"],
        control_init,
        settings,
    )
    stability = truth_free_initialization_stability_diagnostic(
        result, control
    )
    ablation = truth_free_initialization_ablation_diagnostic(
        {"homogeneous": result, "perturbed": control},
        reference_branch="homogeneous",
        operator=operator,
        checkpoint_interval=2,
    )
    conditioning_settings = config["verification"]["pairwise_conditioning"]
    conditioning = truth_free_pairwise_conditioning_diagnostic(
        operator,
        case["I_stack"],
        result,
        control,
        random_seed=int(conditioning_settings["random_seed"]),
        random_direction_count=int(
            conditioning_settings["random_direction_count"]
        ),
    )
    conditioning_repeat = truth_free_pairwise_conditioning_diagnostic(
        operator,
        case["I_stack"],
        result,
        control,
        random_seed=int(conditioning_settings["random_seed"]),
        random_direction_count=int(
            conditioning_settings["random_direction_count"]
        ),
    )
    unscaled_settings = dict(settings)
    unscaled_settings["algorithm"] = "batch_complex_gradient_descent_armijo"
    unscaled = reconstruct_known_b_probe(
        operator,
        case["I_stack"],
        operator.homogeneous_probe_native,
        unscaled_settings,
    )
    evaluation = simulation_evaluation_only(result, case["P_B_true"])
    assert result["truth_used_by_optimizer"] is False
    assert result["sample_b_updated"] is False
    assert result["loss_curve"][-1] < result["loss_curve"][0]
    assert (
        result["detector_relative_residual_curve"][-1]
        < result["detector_relative_residual_curve"][0]
    )
    assert (
        evaluation["final_probe_global_phase_aligned_relative_l2"]
        < evaluation["initial_probe_global_phase_aligned_relative_l2"]
    )
    assert np.all(np.diff(result["loss_curve"]) <= 0.0)
    assert result["loss_curve"][-1] < unscaled["loss_curve"][-1]
    assert result["step_scale_fallback_count"] == 0
    assert np.all(
        np.isfinite(result["gauss_newton_directional_curvature_curve"])
    )
    assert np.all(result["gauss_newton_directional_curvature_curve"] >= 0.0)
    assert len(result["proposed_step_curve"]) == result["iterations_completed"]
    assert np.array_equal(control_init, control_init_repeat)
    assert np.mean(control_init - operator.homogeneous_probe_native) == (
        pytest.approx(0.0j, abs=1.0e-15)
    )
    assert stability["initial_probe_raw_relative_l2"] == pytest.approx(0.05)
    assert control["loss_curve"][-1] < control["loss_curve"][0]
    assert np.all(np.diff(control["loss_curve"]) <= 0.0)
    assert stability["truth_used_by_diagnostic"] is False
    assert ablation["truth_used_by_diagnostic"] is False
    assert ablation["equal_optimizer_algorithm"] is True
    assert ablation["equal_curve_lengths"] is True
    assert ablation["final_pairwise_raw_probe_relative_l2"].shape == (2, 2)
    assert ablation["final_pairwise_prediction_relative_l2"].shape == (2, 2)
    assert np.array_equal(ablation["checkpoint_iterations"], [0, 2, 4, 6, 8])
    assert ablation[
        "checkpoint_pairwise_raw_probe_relative_l2_curve"
    ].shape == (5, 2, 2)
    assert np.array_equal(
        ablation["checkpoint_pairwise_prediction_relative_l2_curve"][-1],
        ablation["final_pairwise_prediction_relative_l2"],
    )
    assert np.isfinite(stability["final_prediction_relative_l2"])
    assert np.isfinite(
        stability["final_probe_global_phase_aligned_relative_l2"]
    )
    pairwise_sensitivity = normalized_intensity_jacobian_sensitivity(
        operator,
        result["P_B_rec"],
        conditioning["P_B_pairwise_direction_unit_l2"],
    )
    assert conditioning["truth_used_by_diagnostic"] is False
    assert np.linalg.norm(
        conditioning["P_B_pairwise_direction_unit_l2"]
    ) == pytest.approx(1.0)
    assert conditioning["pairwise_jacobian_rms_gain"] == pytest.approx(
        pairwise_sensitivity["jacobian_rms_gain"]
    )
    assert conditioning[
        "pairwise_directional_gauss_newton_curvature"
    ] == pytest.approx(conditioning["pairwise_jacobian_rms_gain"] ** 2)
    native_phase_sensitivity = normalized_intensity_jacobian_sensitivity(
        operator,
        result["P_B_rec"],
        1j * result["P_B_rec"],
    )
    assert conditioning[
        "native_global_phase_rotation_jacobian_rms_gain"
    ] == pytest.approx(native_phase_sensitivity["jacobian_rms_gain"])
    assert conditioning[
        "native_global_phase_rotation_jacobian_rms_gain"
    ] > 0.0
    assert len(conditioning["random_direction_jacobian_rms_gains"]) == 8
    assert np.all(
        np.isfinite(conditioning["random_direction_jacobian_rms_gains"])
    )
    assert np.array_equal(
        conditioning["random_direction_jacobian_rms_gains"],
        conditioning_repeat["random_direction_jacobian_rms_gains"],
    )
    continuation_optimizer_settings = dict(settings)
    continuation_optimizer_settings["iterations"] = 2
    primary_continuation = reconstruct_known_b_probe(
        operator,
        case["I_stack"],
        result["P_B_rec"],
        continuation_optimizer_settings,
    )
    control_continuation = reconstruct_known_b_probe(
        operator,
        case["I_stack"],
        control["P_B_rec"],
        continuation_optimizer_settings,
    )
    continuation = truth_free_paired_continuation_diagnostic(
        operator,
        primary_continuation,
        control_continuation,
        checkpoint_interval=1,
    )
    assert np.array_equal(primary_continuation["P_B_init"], result["P_B_rec"])
    assert np.array_equal(control_continuation["P_B_init"], control["P_B_rec"])
    assert np.array_equal(continuation["checkpoint_iterations"], [0, 1, 2])
    assert continuation["additional_iterations"] == 2
    assert continuation["truth_used_by_continuation"] is False
    assert continuation["truth_used_by_diagnostic"] is False
    assert continuation[
        "initial_pairwise_global_phase_aligned_relative_l2"
    ] == pytest.approx(stability["final_probe_global_phase_aligned_relative_l2"])
    assert continuation[
        "pairwise_start_direction_projection_fraction_curve"
    ][0] == pytest.approx(1.0)
    assert np.all(
        np.isfinite(
            continuation["checkpoint_pairwise_jacobian_rms_gain_curve"]
        )
    )
    spectral = truth_free_local_spectral_diagnostic(
        operator,
        primary_continuation["P_B_rec"],
        control_continuation["P_B_rec"],
        krylov_dimension=2,
        reorthogonalization_passes=2,
        breakdown_relative_tolerance=1.0e-12,
        random_seed=20260845,
        reported_low_ritz_count=1,
    )
    spectral_repeat = truth_free_local_spectral_diagnostic(
        operator,
        primary_continuation["P_B_rec"],
        control_continuation["P_B_rec"],
        krylov_dimension=2,
        reorthogonalization_passes=2,
        breakdown_relative_tolerance=1.0e-12,
        random_seed=20260845,
        reported_low_ritz_count=1,
    )
    assert spectral["truth_used_by_diagnostic"] is False
    assert np.sum(spectral["pairwise_start"]["spectral_weights"]) == (
        pytest.approx(1.0)
    )
    assert spectral["pairwise_start"][
        "basis_real_orthogonality_max_abs_error"
    ] < 1.0e-12
    assert np.array_equal(
        spectral["pairwise_start"]["ritz_values"],
        spectral_repeat["pairwise_start"]["ritz_values"],
    )


def test_initialization_ablation_runner_writes_equal_budget_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    config["execution"]["mode"] = "initialization_magnitude_ablation_matched"
    config["verification"]["initialization_ablation"]["enabled"] = True
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
    config["reconstruction"]["iterations"] = 2
    config["verification"]["initialization_ablation"][
        "equal_iteration_budget"
    ] = 2
    config["verification"]["initialization_ablation"][
        "checkpoint_interval"
    ] = 1
    config["output"].update(root="runs", run_name="tiny_exp042_ablation")
    config_path = tmp_path / "tiny_exp042_ablation.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    assert len(list((run_dir / "figures").glob("*.png"))) == 1
    with (run_dir / "run_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    ablation = metrics["initialization_ablation"]
    diagnostic = ablation["truth_free_pairwise_diagnostic"]
    assert ablation["branch_count"] == 4
    assert ablation["design"]["fixed_perturbation_seed"] == 20260843
    assert diagnostic["equal_iterations_completed"] is True
    assert diagnostic["all_registered_budgets_completed"] is True
    assert diagnostic["equal_optimizer_algorithm"] is True
    assert diagnostic["all_loss_nonincreasing"] is True
    assert metrics["operator_consistency"]["truth_replay_exact"] is True
    assert metrics["local_spectral_diagnostic_performed_in_this_run"] is False
    with h5py.File(
        run_dir / "outputs" / config["output"]["hdf5_filename"], "r"
    ) as h5:
        root = h5["entry/reconstruction/initialization_ablation"]
        assert set(root["branches"]) == {
            "homogeneous_reference",
            "perturb_1pct",
            "perturb_2p5pct",
            "perturb_5pct",
        }
        assert root[
            "branches/homogeneous_reference/loss_curve"
        ].shape == (3,)
        assert root[
            "truth_free_pairwise_diagnostic/"
            "final_pairwise_prediction_relative_l2"
        ].shape == (4, 4)
        assert root[
            "truth_free_pairwise_diagnostic/"
            "checkpoint_pairwise_prediction_relative_l2_curve"
        ].shape == (3, 4, 4)
        assert "simulation_evaluation_only" in root[
            "branches/perturb_1pct"
        ]


def test_detector_quadrature_ablation_runner_writes_auditable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
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
    config["reconstruction"]["iterations"] = 2
    detector_settings = config["verification"][
        "detector_quadrature_ablation"
    ]
    detector_settings["equal_iteration_budget"] = 2
    detector_settings["checkpoint_interval"] = 1
    config["output"].update(root="runs", run_name="tiny_exp042_detector")
    config_path = tmp_path / "tiny_exp042_detector.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    with (run_dir / "run_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert len(list((run_dir / "figures").glob("*.png"))) == 1
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    ablation = metrics["detector_quadrature_ablation"]
    diagnostic = ablation["truth_free_pairwise_diagnostic"]
    controls = metrics["operator_consistency"]
    assert ablation["branch_count"] == 3
    assert set(ablation["branches"]) == {
        "matched_q4",
        "mismatch_q1_point",
        "matched_q1_point",
    }
    assert diagnostic["initial_probe_exact_equal"] is True
    assert diagnostic["equal_optimizer_algorithm"] is True
    assert np.asarray(
        diagnostic["checkpoint_pairwise_prediction_relative_l2_curve"]
    ).shape == (3, 3, 3)
    assert controls["matched_q4_truth_replay_exact"] is True
    assert controls["q1_self_truth_fixed_point_exact"] is True
    assert controls["q1_on_q4_truth_replay_exact"] is False
    assert controls["q1_branch_is_exp040_matched"] is False
    assert controls["q1_q1_matched_control_included"] is True
    assert metrics["data_controls"][
        "same_q4_data_for_primary_and_mismatch"
    ] is True
    with h5py.File(
        run_dir / "outputs" / config["output"]["hdf5_filename"], "r"
    ) as h5:
        root = h5["entry/reconstruction/detector_quadrature_ablation"]
        assert set(root["branches"]) == {
            "matched_q4",
            "mismatch_q1_point",
            "matched_q1_point",
        }
        assert root["branches/matched_q4/P_B_init"].shape == (32, 32)
        assert root["branches/mismatch_q1_point/P_B_rec"].shape == (32, 32)
        assert root["branches/matched_q1_point/P_B_rec"].shape == (32, 32)
        assert root[
            "control_measurements/q1_point_control_data/I_stack"
        ].shape == (9, 16, 16)
        assert root[
            "truth_free_pairwise_diagnostic/"
            "checkpoint_pairwise_prediction_relative_l2_curve"
        ].shape == (3, 3, 3)


def test_spectral_runner_writes_auditable_hdf5_and_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    config["execution"]["mode"] = "local_spectral_diagnostic_from_prior_run"
    config["reconstruction"]["initialization_stability_control"][
        "enabled"
    ] = True
    config["reconstruction"]["paired_continuation_control"]["enabled"] = True
    config["verification"]["pairwise_conditioning"]["enabled"] = True
    config["verification"]["local_spectral_diagnostic"]["enabled"] = True
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
    config["sample_b"].update(
        support_shape=[64, 64],
        feature_size_px=2,
    )
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
    config["reconstruction"]["iterations"] = 2
    config["reconstruction"]["paired_continuation_control"].update(
        additional_iterations=2,
        checkpoint_interval=1,
    )
    spectral_settings = config["verification"]["local_spectral_diagnostic"]
    config["output"].update(root="runs", run_name="tiny_exp042")
    case = build_matched_development_case(config)
    operator = case["operator"]
    primary = operator.homogeneous_probe_native.copy()
    control = make_deterministic_complex_probe_initialization(
        primary, seed=20260845, relative_l2_to_primary=0.03
    )
    source_run = tmp_path / "source_run"
    source_hdf5 = source_run / "outputs" / "exp042_probe_reconstruction.h5"
    source_hdf5.parent.mkdir(parents=True)
    source_config_path = source_run / "config.yaml"
    source_metrics_path = source_run / "metrics.json"
    save_config(source_config_path, deepcopy(config))
    save_json(
        source_metrics_path,
        {
            "truth_used_by_optimizer": False,
            "scientific_pass_fail_conclusion": False,
        },
    )
    save_json(
        source_run / "run_state.json",
        {"status": "complete", "artifacts_validated": True},
    )
    with h5py.File(source_hdf5, "w") as h5:
        h5.create_dataset("entry/data/I_stack", data=case["I_stack"])
        h5.create_dataset(
            "entry/data/scan_positions", data=case["scan_positions"]
        )
        h5.create_dataset(
            "entry/reconstruction/paired_continuation_control/"
            "primary/P_B_rec",
            data=primary,
        )
        h5.create_dataset(
            "entry/reconstruction/paired_continuation_control/"
            "control/P_B_rec",
            data=control,
        )
    spectral_settings.update(
        source_run=str(source_run),
        source_config_sha256=runner._sha256(source_config_path),
        source_metrics_sha256=runner._sha256(source_metrics_path),
        source_hdf5_sha256=runner._sha256(source_hdf5),
    )
    reference_diagnostic = truth_free_local_spectral_diagnostic(
        operator,
        primary,
        control,
        krylov_dimension=12,
        reorthogonalization_passes=2,
        breakdown_relative_tolerance=1.0e-12,
        random_seed=int(spectral_settings["random_seed"]),
        reported_low_ritz_count=int(
            spectral_settings["reported_low_ritz_count"]
        ),
    )
    reference_run = tmp_path / "reference_spectral_run"
    reference_hdf5 = (
        reference_run / "outputs" / "exp042_probe_reconstruction.h5"
    )
    reference_hdf5.parent.mkdir(parents=True)
    reference_config = deepcopy(config)
    reference_spectral = reference_config["verification"][
        "local_spectral_diagnostic"
    ]
    reference_spectral["krylov_dimension"] = 12
    reference_spectral.pop("dimension_convergence_control")
    reference_config_path = reference_run / "config.yaml"
    reference_metrics_path = reference_run / "metrics.json"
    save_config(reference_config_path, reference_config)
    reference_adjoint_error = intensity_jacobian_adjoint_relative_error(
        operator, primary, int(spectral_settings["random_seed"]) + 1
    )
    save_json(
        reference_metrics_path,
        {
            "local_spectral_diagnostic": reference_diagnostic,
            "operator_consistency": {
                "intensity_jacobian_real_adjoint_relative_error": (
                    reference_adjoint_error
                )
            },
        },
    )
    save_json(
        reference_run / "run_state.json",
        {"status": "complete", "artifacts_validated": True},
    )
    with h5py.File(reference_hdf5, "w") as h5:
        root = h5.require_group(
            "entry/reconstruction/local_spectral_diagnostic"
        )
        root.create_dataset("P_B_primary_probe_raw", data=primary)
        root.create_dataset("P_B_control_probe_raw", data=control)
        root.require_group("pairwise_start").create_dataset(
            "ritz_values",
            data=reference_diagnostic["pairwise_start"]["ritz_values"],
        )
        root.require_group("random_start").create_dataset(
            "ritz_values",
            data=reference_diagnostic["random_start"]["ritz_values"],
        )
    convergence_settings = spectral_settings["dimension_convergence_control"]
    convergence_settings.update(
        reference_run=str(reference_run),
        reference_config_sha256=runner._sha256(reference_config_path),
        reference_metrics_sha256=runner._sha256(reference_metrics_path),
        reference_hdf5_sha256=runner._sha256(reference_hdf5),
    )
    config_path = tmp_path / "tiny_exp042.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    assert (run_dir / "run_state.json").is_file()
    assert len(list((run_dir / "figures").glob("*.png"))) == 1
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert metrics["source_artifact"]["data_exact_replay"] is True
    assert metrics["source_artifact"][
        "scan_positions_exact_replay"
    ] is True
    assert metrics["operator_consistency"][
        "intensity_jacobian_real_adjoint_relative_error"
    ] < 1.0e-11
    assert metrics["operator_consistency"][
        "intensity_jacobian_real_adjoint_rerun"
    ] is False
    spectral_metrics = metrics["local_spectral_diagnostic"]
    assert spectral_metrics["truth_used_by_diagnostic"] is False
    assert spectral_metrics["pairwise_start"][
        "completed_krylov_dimension"
    ] == 24
    assert spectral_metrics["random_start"][
        "completed_krylov_dimension"
    ] == 24
    convergence_metrics = spectral_metrics["dimension_convergence_control"]
    assert convergence_metrics["both_recurrence_prefixes_exact"] is True
    assert convergence_metrics["single_extension_only"] is True
    with h5py.File(
        run_dir / "outputs" / config["output"]["hdf5_filename"], "r"
    ) as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "reconstruction",
            "sample",
        }
        assert h5["entry/data/I_stack"].shape == (9, 16, 16)
        assert h5["entry/sample/B_known"].shape == (96, 96)
        root = h5["entry/reconstruction/local_spectral_diagnostic"]
        assert root["P_B_primary_probe_raw"].shape == (32, 32)
        assert root["P_B_control_probe_raw"].shape == (32, 32)
        assert root["pairwise_start/ritz_values"].shape == (24,)
        assert root["random_start/ritz_values"].shape == (24,)
        assert "dimension_convergence_control/pairwise_start" in root
        assert np.sum(root["pairwise_start/spectral_weights"][...]) == (
            pytest.approx(1.0)
        )
