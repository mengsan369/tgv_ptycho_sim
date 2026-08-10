from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scripts.run_exp030_effective_phase import (
    _adjoint_radial_interpolation,
    _apply_radial_interpolation,
    _blind_checkpoint_evaluation,
    _build_radial_fresnel_operator,
    _detector_pixel_average,
    _estimate_radial_operator_norm_squared,
    _fresnel_hankel_radial,
    _load_blind_long_checkpoint,
    _make_radial_adjoint_constraint,
    _make_radial_output_range_constraint,
    _radial_fresnel_full_field,
    _radial_fresnel_linear_forward,
    _radial_fresnel_weighted_adjoint,
    _radial_interpolation_plan,
    _reconstruct_case,
    _run_blind_long_study,
    _run_optimizer_study,
    _save_blind_long_checkpoint,
    _simulate_probe_detector,
)

from tgv_ptycho.inverse.observability import (
    analyze_local_observability,
    central_finite_difference,
    gauge_project_complex_derivative,
    normalized_complex_sensitivity,
    relative_l2,
    successive_relative_changes,
)


def test_finite_difference_has_expected_shape_dtype_and_finite_values() -> None:
    minus = np.ones((12, 10), dtype=np.complex128)
    plus = minus + (2e-6) * (0.5 + 0.2j)
    derivative = central_finite_difference(minus, plus, 1e-6)

    assert derivative.shape == minus.shape
    assert derivative.dtype == np.complex128
    assert np.all(np.isfinite(derivative))
    assert np.allclose(derivative, 0.5 + 0.2j)


def test_gauge_projection_removes_global_phase_and_scale() -> None:
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(16, 18)) + 1j * rng.normal(size=(16, 18))
    gauge_derivative = (0.3 + 0.7j) * reference
    projected = gauge_project_complex_derivative(gauge_derivative, reference)

    assert np.sqrt(np.sum(np.abs(projected) ** 2)) < 1e-12

    step = 1e-4
    minus = reference * np.exp(-1j * step)
    plus = reference * np.exp(1j * step)
    _, sensitivity = normalized_complex_sensitivity(
        minus, plus, reference, step, parameter_scale=1.0
    )
    assert sensitivity < 1e-10


def test_local_observability_outputs_scaled_correlation_and_singular_values() -> None:
    rng = np.random.default_rng(5)
    reference = np.exp(1j * rng.normal(size=(14, 16)))
    derivatives = {
        "d_waist": rng.normal(size=reference.shape)
        + 1j * rng.normal(size=reference.shape),
        "surface": rng.normal(size=reference.shape)
        + 1j * rng.normal(size=reference.shape),
        "phase_scale": rng.normal(size=reference.shape)
        + 1j * rng.normal(size=reference.shape),
    }
    result = analyze_local_observability(
        reference,
        derivatives,
        {"d_waist": 1e-5, "surface": 2e-5, "phase_scale": 1.0},
    )

    correlation = np.asarray(result["normalized_column_correlation"])
    singular_values = np.asarray(result["singular_values"])
    assert correlation.shape == (3, 3)
    assert singular_values.shape == (3,)
    assert np.all(np.isfinite(correlation))
    assert np.all(singular_values[:-1] >= singular_values[1:])
    assert np.allclose(np.diag(correlation), 1.0)


def test_successive_relative_changes_uses_finer_value_as_reference() -> None:
    changes = successive_relative_changes([8.0, 10.0, 10.5])

    assert changes.dtype == np.float64
    assert np.allclose(changes, [0.2, 0.5 / 10.5])


def test_detector_pixel_average_preserves_constant_and_known_block_mean() -> None:
    values = np.asarray(
        [[1.0, 3.0, 2.0, 4.0], [5.0, 7.0, 6.0, 8.0]],
        dtype=np.float64,
    )
    averaged = _detector_pixel_average(values, 0.5, 1.0)

    assert averaged.shape == (1, 2)
    assert averaged.dtype == np.float64
    assert np.allclose(averaged, [[4.0, 5.0]])
    assert np.allclose(
        _detector_pixel_average(np.ones((3, 4, 4)), 0.5, 1.0), 1.0
    )


def test_fresnel_hankel_zero_contrast_is_exact_plane_wave() -> None:
    radius = (np.arange(100, dtype=np.float64) + 0.5) * 20e-9
    weights = np.full(radius.shape, 20e-9, dtype=np.float64)
    output_r, probes = _fresnel_hankel_radial(
        radius,
        weights,
        np.ones((1, radius.size), dtype=np.complex128),
        output_radius_max_m=4e-6,
        output_step_m=0.2e-6,
        wavelength_m=532e-9,
        propagation_distance_m=1e-3,
        medium_index=1.0,
        incident_amplitude=1.0,
    )
    expected = np.exp(1j * 2.0 * np.pi / 532e-9 * 1e-3)

    assert output_r.ndim == 1
    assert probes.shape == (1, output_r.size)
    assert probes.dtype == np.complex128
    assert np.all(np.isfinite(probes))
    assert np.max(np.abs(probes - expected)) <= 1e-14


def test_fresnel_hankel_gaussian_matches_closed_form() -> None:
    wavelength = 532e-9
    distance = 0.5e-3
    waist = 3e-6
    amplitude = 0.03 + 0.02j
    step = 10e-9
    support = 6.0 * waist
    count = int(np.ceil(support / step))
    actual_step = support / count
    radius = (np.arange(count, dtype=np.float64) + 0.5) * actual_step
    weights = np.full(count, actual_step, dtype=np.float64)
    transmission = 1.0 + amplitude * np.exp(-(radius / waist) ** 2)
    output_r, probes = _fresnel_hankel_radial(
        radius,
        weights,
        transmission[None, :],
        output_radius_max_m=12e-6,
        output_step_m=0.25e-6,
        wavelength_m=wavelength,
        propagation_distance_m=distance,
        medium_index=1.0,
        incident_amplitude=1.0,
    )

    wavenumber = 2.0 * np.pi / wavelength
    coefficient = 1.0 / waist**2 - 1j * wavenumber / (2.0 * distance)
    analytic_delta = (
        np.exp(1j * wavenumber * distance)
        * np.exp(1j * wavenumber * output_r**2 / (2.0 * distance))
        * (2.0 * np.pi / (1j * wavelength * distance))
        * amplitude
        / (2.0 * coefficient)
        * np.exp(
            -(wavenumber * output_r / distance) ** 2
            / (4.0 * coefficient)
        )
    )
    numeric_delta = probes[0] - np.exp(1j * wavenumber * distance)

    assert relative_l2(numeric_delta, analytic_delta) < 1e-4


def _small_radial_operator() -> dict[str, object]:
    source_step = 40e-9
    source_radius = (np.arange(30, dtype=np.float64) + 0.5) * source_step
    source_weights = np.full(source_radius.shape, source_step)
    output_radius = np.arange(0.0, 2.4e-6 + 0.1e-6, 0.1e-6)
    return _build_radial_fresnel_operator(
        source_radius,
        source_weights,
        output_radius,
        shape=(12, 14),
        dx_m=0.2e-6,
        center_xy_m=(0.0, 0.0),
        wavelength_m=532e-9,
        propagation_distance_m=0.8e-3,
        medium_index=1.0,
        incident_amplitude=1.0,
    )


def test_radial_cartesian_interpolation_has_exact_euclidean_adjoint() -> None:
    coordinate = np.linspace(0.0, 3.0e-6, 31)
    plan = _radial_interpolation_plan(
        coordinate, (10, 12), 0.25e-6, (0.0, 0.0)
    )
    rng = np.random.default_rng(31)
    radial = rng.normal(size=coordinate.shape) + 1j * rng.normal(
        size=coordinate.shape
    )
    cartesian = rng.normal(size=(10, 12)) + 1j * rng.normal(size=(10, 12))

    left = np.vdot(_apply_radial_interpolation(radial, plan), cartesian)
    right = np.vdot(radial, _adjoint_radial_interpolation(cartesian, plan))
    relative_error = abs(left - right) / max(abs(left), abs(right))

    assert relative_error <= 1e-12


def test_fresnel_hankel_cartesian_operator_has_weighted_adjoint() -> None:
    operator = _small_radial_operator()
    rng = np.random.default_rng(32)
    source = rng.normal(size=30) + 1j * rng.normal(size=30)
    cartesian = rng.normal(size=(12, 14)) + 1j * rng.normal(size=(12, 14))

    forward = _radial_fresnel_linear_forward(operator, source)
    adjoint = _radial_fresnel_weighted_adjoint(operator, cartesian)
    left = float(operator["dx_m"]) ** 2 * np.vdot(forward, cartesian)
    right = np.sum(
        np.asarray(operator["source_measure_m2"])
        * np.conj(source)
        * adjoint
    )
    relative_error = abs(left - right) / max(abs(left), abs(right))

    assert relative_error <= 1e-10


def test_radial_adjoint_constraint_preserves_model_fixed_point() -> None:
    operator = _small_radial_operator()
    source_radius = np.asarray(operator["source_radius_m"])
    transmission = np.exp(
        1j * 0.4 * np.exp(-(source_radius / 0.6e-6) ** 2)
    )
    target = _radial_fresnel_full_field(operator, transmission)
    norm_squared = _estimate_radial_operator_norm_squared(operator, 5, 33)
    constraint, state = _make_radial_adjoint_constraint(
        operator,
        operator_norm_squared=norm_squared,
        application_interval=1,
        internal_steps=1,
        step_scale=0.5,
        max_backtracking_steps=6,
        initial_transmission=transmission,
    )

    constrained = constraint(target)
    constrained_twice = constraint(constrained)

    assert relative_l2(constrained, target) <= 1e-12
    assert relative_l2(constrained_twice, constrained) <= 1e-12
    assert np.max(
        np.abs(np.abs(np.asarray(state["source_transmission"])) - 1.0)
    ) <= 1e-12


def test_radial_adjoint_constraint_step_does_not_increase_objective() -> None:
    operator = _small_radial_operator()
    source_radius = np.asarray(operator["source_radius_m"])
    target_transmission = np.exp(
        1j * 0.2 * np.cos(np.pi * source_radius / source_radius[-1])
    )
    target = _radial_fresnel_full_field(operator, target_transmission)
    norm_squared = _estimate_radial_operator_norm_squared(operator, 5, 34)
    constraint, state = _make_radial_adjoint_constraint(
        operator,
        operator_norm_squared=norm_squared,
        application_interval=1,
        internal_steps=2,
        step_scale=0.5,
        max_backtracking_steps=8,
    )

    constrained = constraint(target)
    before = np.asarray(state["objective_before"])
    after = np.asarray(state["objective_after"])

    assert constrained.shape == target.shape
    assert constrained.dtype == np.complex128
    assert np.all(np.isfinite(constrained))
    assert np.all(after <= before + 1e-26)


def test_radial_output_range_constraint_preserves_forward_truth() -> None:
    operator = _small_radial_operator()
    source_radius = np.asarray(operator["source_radius_m"])
    transmission = np.exp(
        1j * 0.3 * np.exp(-(source_radius / 0.7e-6) ** 2)
    )
    target = _radial_fresnel_full_field(operator, transmission)
    constraint, state = _make_radial_output_range_constraint(
        operator["interpolation"], 1e-13
    )

    constrained = constraint(target)
    constrained_twice = constraint(constrained)

    assert relative_l2(constrained, target) <= 1e-10
    assert relative_l2(constrained_twice, constrained) <= 1e-10
    assert state["call_count"] == 2
    assert state["active_radial_node_count"] <= state["radial_node_count"]


def test_known_b_probe_only_ablation_keeps_truth_b_fixed() -> None:
    shape = (8, 8)
    rng = np.random.default_rng(35)
    probe = np.exp(1j * rng.normal(scale=0.1, size=shape))
    sample_b = np.exp(1j * rng.normal(scale=0.2, size=shape))
    intensity = np.abs(probe * sample_b)[None, ...] ** 2
    config = {
        "optics": {
            "wavelength_m": 532e-9,
            "z_AB_m": 0.2e-3,
            "z_BC_m": 0.0,
        },
        "reconstruction": {
            "beta_probe": 0.08,
            "beta_object": 0.5,
            "initial_object_phase_std_rad": 0.02,
            "initial_object_seed": 41,
            "shuffle_positions": False,
            "shuffle_seed": 42,
            "object_amplitude_bounds": [1.0, 1.0],
            "show_progress": False,
            "operator_consistency_ablation": {},
        },
    }

    result = _reconstruct_case(
        config,
        intensity,
        np.asarray([[0.0, 0.0]]),
        np.ones(shape, dtype=np.complex128),
        np.ones(shape, dtype=bool),
        0.5e-6,
        variant_id="known_b_probe_only",
        num_iters=2,
        sample_b_true_diagnostic=sample_b,
        probe_true_diagnostic=probe,
    )

    assert result["uses_simulation_truth_B_as_input"] is True
    assert result["uses_simulation_truth_probe_as_input"] is False
    assert result["fixed_object_max_abs_change"] <= 1e-12
    assert np.allclose(
        result["B_fixed_simulation_diagnostic_only"], sample_b, atol=1e-12
    )


def test_optimizer_study_records_selected_checkpoint_trajectory() -> None:
    shape = (8, 8)
    dx = 1.0e-6
    rng = np.random.default_rng(45)
    probe = np.exp(1j * rng.normal(scale=0.1, size=shape))
    sample_b = np.exp(1j * rng.normal(scale=0.2, size=shape))
    positions = np.asarray([[0.0, 0.0], [1.0e-6, -1.0e-6]])
    transfer = np.ones(shape, dtype=np.complex128)
    config = {
        "optics": {
            "wavelength_m": 532e-9,
            "z_AB_m": 0.2e-3,
            "z_BC_m": 0.0,
            "detector_pixel_size_m": dx,
        },
        "noise": {"type": "none"},
        "reconstruction": {
            "correction_mode": "adjoint_residual",
            "denominator_mode": "epie",
            "object_boundary": "periodic",
            "beta_probe": 0.1,
            "beta_object": 0.5,
            "initial_object_phase_std_rad": 0.02,
            "initial_object_seed": 46,
            "shuffle_positions": False,
            "shuffle_seed": 47,
            "object_amplitude_bounds": [1.0, 1.0],
            "show_progress": False,
            "operator_consistency_ablation": {
                "optimizer_study": {
                    "correction_mode": "adjoint_residual",
                    "runtime_budget_s": 100.0,
                    "planning_seconds_per_iteration": 0.01,
                    "known_probe_object_only": {
                        "diagnostic_num_iters": 2,
                        "beta_object": 0.5,
                        "checkpoint_iterations": [1, 2],
                    },
                    "known_b_probe_only": {
                        "beta_probe_candidates": [0.05, 0.1],
                        "screening_num_iters": 1,
                        "normalization_screening_num_iters": 1,
                        "normalization_modes": ["none", "measurement_energy"],
                        "selected_num_iters": 3,
                        "checkpoint_iterations": [1, 2, 3],
                    },
                    "boundary_controls": {"diagnostic_num_iters": 1},
                }
            },
        },
    }
    intensity = _simulate_probe_detector(
        config, probe, sample_b, positions, dx, transfer
    )

    output, metrics, _, _, _ = _run_optimizer_study(
        config,
        intensity,
        positions,
        np.ones(shape, dtype=np.complex128),
        np.ones(shape, dtype=bool),
        dx,
        probe,
        sample_b,
        transfer,
    )

    selected = output["known_b_probe_only"]["selected_trajectory"]
    assert set(selected["checkpoints"]) == {"1", "2", "3"}
    assert metrics["known_b_probe_only"]["selected_trajectory_status"] == (
        "executed"
    )
    assert metrics["status"] == "executed"


def test_blind_long_checkpoint_round_trip_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(48)
    probe = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    object_b = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    rng_state = np.random.default_rng(49).bit_generator.state
    checkpoint = {
        "optimizer_state": {
            "version": 1,
            "completed_iterations": 3,
            "P_B_rec": probe,
            "B_rec": object_b,
            "loss_curve": np.asarray([0.8, 0.3, 0.125]),
            "initial_data_fidelity_loss": 1.25,
            "rng_bit_generator": "PCG64",
            "rng_state": rng_state,
            "problem_signature": "unit-test-problem-signature",
        },
        "constraint_state": {
            "call_count": 7,
            "last_relative_change": 2.5e-4,
        },
        "data_fidelity_loss": 0.125,
    }
    evaluation = {
        "P_B_aligned_complex_relative_error": 2.0e-5,
        "aligned_probe_error_to_true_case_separation_ratio": 0.8,
        "frozen_loss_to_true_detector_amplitude_separation_ratio": 0.4,
    }
    path = tmp_path / "baseline_iter_000003.h5"

    _save_blind_long_checkpoint(
        path,
        checkpoint,
        case_id="baseline",
        runner_signature="unit-test-runner-signature",
        evaluation_metrics=evaluation,
    )
    loaded = _load_blind_long_checkpoint(path)

    state = loaded["optimizer_state"]
    assert loaded["case_id"] == "baseline"
    assert loaded["runner_signature"] == "unit-test-runner-signature"
    assert len(loaded["source_sha256"]) == 64
    assert state["version"] == 1
    assert state["completed_iterations"] == 3
    assert state["P_B_rec"].dtype == np.complex128
    assert state["B_rec"].dtype == np.complex128
    assert np.array_equal(state["P_B_rec"], probe)
    assert np.array_equal(state["B_rec"], object_b)
    assert np.array_equal(state["loss_curve"], [0.8, 0.3, 0.125])
    assert state["initial_data_fidelity_loss"] == pytest.approx(1.25)
    assert state["rng_bit_generator"] == "PCG64"
    assert state["rng_state"] == rng_state
    assert state["problem_signature"] == "unit-test-problem-signature"
    assert loaded["data_fidelity_loss"] == pytest.approx(0.125)
    assert loaded["constraint_state"] == {
        "call_count": 7,
        "last_relative_change": pytest.approx(2.5e-4),
    }
    assert loaded["simulation_evaluation_only"] == pytest.approx(evaluation)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _save_blind_long_checkpoint(
            path,
            checkpoint,
            case_id="baseline",
            runner_signature="unit-test-runner-signature",
            evaluation_metrics=evaluation,
        )


def test_blind_checkpoint_evaluation_produces_finite_gate_ratios() -> None:
    rng = np.random.default_rng(50)
    probe_true = rng.normal(size=(5, 6)) + 1j * rng.normal(size=(5, 6))
    object_true = rng.normal(size=(5, 6)) + 1j * rng.normal(size=(5, 6))
    probe_rec = probe_true + 1e-3 * (
        rng.normal(size=probe_true.shape) + 1j * rng.normal(size=probe_true.shape)
    )
    object_rec = object_true + 2e-3 * (
        rng.normal(size=object_true.shape) + 1j * rng.normal(size=object_true.shape)
    )
    checkpoint = {
        "optimizer_state": {
            "P_B_rec": probe_rec,
            "B_rec": object_rec,
            "loss_curve": np.asarray([0.1, 0.08, 0.06, 0.05]),
        },
        "data_fidelity_loss": 0.05,
    }
    probe_separation = 0.2
    detector_separation = 0.1

    metrics = _blind_checkpoint_evaluation(
        checkpoint,
        probe_true=probe_true,
        object_true=object_true,
        true_probe_case_separation=probe_separation,
        true_detector_amplitude_separation=detector_separation,
    )

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics[
        "aligned_probe_error_to_true_case_separation_ratio"
    ] == pytest.approx(
        metrics["P_B_aligned_complex_relative_error"] / probe_separation
    )
    assert metrics[
        "frozen_loss_to_true_detector_amplitude_separation_ratio"
    ] == pytest.approx(0.5)
    assert metrics["loss_tail_slope_per_iteration"] < 0.0

    zero_scale_metrics = _blind_checkpoint_evaluation(
        checkpoint,
        probe_true=probe_true,
        object_true=object_true,
        true_probe_case_separation=0.0,
        true_detector_amplitude_separation=0.0,
    )
    assert all(np.isfinite(value) for value in zero_scale_metrics.values())


def test_blind_long_study_runs_checkpoint_gate_and_matched_cases(
    tmp_path: Path,
) -> None:
    shape = (8, 8)
    dx = 1.0e-6
    rng = np.random.default_rng(51)
    sample_b = np.exp(1j * rng.normal(scale=0.1, size=shape))
    yy, xx = np.indices(shape, dtype=np.float64)
    radius = np.sqrt((yy - 3.5) ** 2 + (xx - 3.5) ** 2)
    baseline_probe = np.exp(1j * 0.03 * radius).astype(np.complex128)
    minus_probe = (baseline_probe * np.exp(-1j * 1e-4 * radius)).astype(
        np.complex128
    )
    plus_probe = (baseline_probe * np.exp(1j * 1e-4 * radius)).astype(
        np.complex128
    )
    probes = {
        "baseline": baseline_probe,
        "waist_minus": minus_probe,
        "waist_plus": plus_probe,
    }
    positions = np.asarray([[0.0, 0.0], [dx, -dx]], dtype=np.float64)
    config = {
        "optics": {
            "shape": list(shape),
            "dx_m": dx,
            "wavelength_m": 532e-9,
            "z_AB_m": 0.0,
            "z_BC_m": 0.0,
            "detector_pixel_size_m": dx,
        },
        "noise": {"type": "none"},
        "reconstruction": {
            "correction_mode": "adjoint_residual",
            "denominator_mode": "epie",
            "rpie_alpha_probe": 0.1,
            "rpie_alpha_object": 0.1,
            "object_boundary": "periodic",
            "beta_probe": 0.08,
            "beta_object": 0.5,
            "initial_object_phase_std_rad": 0.02,
            "initial_object_seed": 52,
            "shuffle_positions": True,
            "shuffle_seed": 53,
            "object_amplitude_bounds": [1.0, 1.0],
            "show_progress": False,
            "operator_consistency_ablation": {
                "sensitivity_floor_to_signal_gate": 1.0,
                "radial_output_range_constraint": {"ridge_fraction": 1e-12},
                "blind_long_study": {
                    "enabled": True,
                    "variant_id": "blind_radial_output_range_constraint",
                    "selected_num_iters": 3,
                    "checkpoint_iterations": [1, 2, 3],
                    "sensitivity_floor_to_signal_gate": 1e12,
                    "beta_probe": 0.08,
                    "beta_object": 0.5,
                    "normalization_mode": "none",
                    "object_boundary": "periodic",
                },
            },
        },
    }
    transfer = np.ones(shape, dtype=np.complex128)
    intensities = {
        case_id: _simulate_probe_detector(
            config, probe, sample_b, positions, dx, transfer
        )
        for case_id, probe in probes.items()
    }
    interpolation = _radial_interpolation_plan(
        np.linspace(0.0, 8.0e-6, 40), shape, dx, (0.0, 0.0)
    )

    output, metrics, fields, losses, labels = _run_blind_long_study(
        config,
        run_dir=tmp_path,
        resume_checkpoint_path=None,
        scan_positions=positions,
        incident=np.ones(shape, dtype=np.complex128),
        reference_mask=np.ones(shape, dtype=bool),
        dx_m=dx,
        radial_operator={"interpolation": interpolation},
        radial_operator_norm_squared=1.0,
        sample_b=sample_b,
        intensity_by_case=intensities,
        probe_true_by_case=probes,
        true_probe_case_separation=1.0,
        true_detector_amplitude_separation=1.0,
        delta_d_waist_m=1e-9,
        nominal_d_waist_m=30e-6,
        true_normalized_probe_sensitivity=0.5,
    )

    assert output["status"] == "executed_with_matched_sensitivity_cases"
    assert metrics["selected_sensitivity_check"][
        "earliest_passing_checkpoint_iteration"
    ] == 1
    assert set(output["baseline"]["checkpoints"]) == {"1", "2", "3"}
    assert set(output["cases"]) == {"baseline", "waist_minus", "waist_plus"}
    assert len(fields) == len(losses) == len(labels) == 3
    for iteration in (1, 2, 3):
        assert (
            tmp_path
            / "checkpoints"
            / "blind_long"
            / "baseline"
            / f"iter_{iteration:04d}.h5"
        ).is_file()

    source_checkpoint = (
        tmp_path
        / "checkpoints"
        / "blind_long"
        / "baseline"
        / "iter_0002.h5"
    )
    resumed_output, resumed_metrics, _, _, _ = _run_blind_long_study(
        config,
        run_dir=tmp_path / "resumed",
        resume_checkpoint_path=source_checkpoint,
        scan_positions=positions,
        incident=np.ones(shape, dtype=np.complex128),
        reference_mask=np.ones(shape, dtype=bool),
        dx_m=dx,
        radial_operator={"interpolation": interpolation},
        radial_operator_norm_squared=1.0,
        sample_b=sample_b,
        intensity_by_case=intensities,
        probe_true_by_case=probes,
        true_probe_case_separation=1.0,
        true_detector_amplitude_separation=1.0,
        delta_d_waist_m=1e-9,
        nominal_d_waist_m=30e-6,
        true_normalized_probe_sensitivity=0.5,
    )

    assert resumed_metrics["resume_provenance"]["resumed"] is True
    assert np.array_equal(
        resumed_output["baseline"]["P_B_rec_raw"],
        output["baseline"]["P_B_rec_raw"],
    )
    assert np.array_equal(
        resumed_output["baseline"]["B_rec_raw"],
        output["baseline"]["B_rec_raw"],
    )
    assert np.array_equal(
        resumed_output["baseline"]["loss_curve"],
        output["baseline"]["loss_curve"],
    )
