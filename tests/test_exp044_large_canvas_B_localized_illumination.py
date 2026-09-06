from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from tgv_ptycho.forward.exp040 import relative_l2
from tgv_ptycho.forward.exp044 import (
    FiniteMasterBlindProbeBOperator,
    FiniteMasterKnownBProbeOperator,
    build_localized_multislice_probe,
    build_nested_master_b,
    gaussian_incident_field,
    operator_control_metrics,
)
from tgv_ptycho.forward.scan_windows import make_scan_window_plan
from tgv_ptycho.objects.sample_b import make_random_phase_object
from tgv_ptycho.recon.exp042 import reconstruct_known_b_probe_damped_gn_cg
from tgv_ptycho.recon.exp043 import (
    alternating_blind_reconstruction,
    make_b_initialization,
)
from tgv_ptycho.recon.exp044 import (
    blind_component_evaluation_simulation_only,
    determine_exp044_status,
    global_phase_probe_evaluation_simulation_only,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "exp044_TGV_3d_large_canvas_B_localized_illumination.yaml"
)


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _tiny_operator() -> tuple[
    FiniteMasterBlindProbeBOperator,
    FiniteMasterKnownBProbeOperator,
    np.ndarray,
    np.ndarray,
]:
    positions = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    plan = make_scan_window_plan((12, 12), (8, 8), positions, 1.0)
    homogeneous_open = np.ones((12, 12), dtype=np.complex128)
    frequencies = np.fft.fftfreq(12)
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    transfer = np.exp(-0.8j * np.pi * (fx**2 + fy**2))
    blind = FiniteMasterBlindProbeBOperator(
        positions_m=positions,
        node_dx_m=1.0,
        quadrature_factor=2,
        native_shape=(4, 4),
        open_shape=(12, 12),
        detector_roi_shape=(4, 4),
        homogeneous_probe_native=np.ones((4, 4), dtype=np.complex128),
        homogeneous_probe_open=homogeneous_open,
        homogeneous_detector_open=homogeneous_open,
        transfer_bc=np.asarray(transfer, dtype=np.complex128),
        scan_plan=plan,
        support_mask=plan.visited_region,
    )
    rng = np.random.default_rng(44)
    phase = np.where(plan.visited_region, 0.25 * rng.normal(size=(12, 12)), 0.0)
    modulation = np.where(plan.visited_region, np.exp(1j * phase) - 1.0, 0.0j)
    y, x = np.mgrid[-2:2, -2:2]
    probe = np.exp(-0.08 * (x**2 + y**2)) * np.exp(0.05j * x)
    known = FiniteMasterKnownBProbeOperator(blind, modulation)
    return blind, known, np.asarray(probe, dtype=np.complex128), modulation


def test_gaussian_power_diameter_and_large_spot_limit() -> None:
    field, metrics = gaussian_incident_field(
        (512, 512), 0.25e-6, 48e-6, 2.304e-9
    )
    assert metrics["captured_power_fraction"] == pytest.approx(1.0, abs=2e-7)
    center = field.shape[0] // 2
    offset = int(round(24e-6 / 0.25e-6))
    ratio = abs(field[center, center + offset]) ** 2 / metrics["center_intensity"]
    assert ratio == pytest.approx(np.exp(-2.0), rel=0.03)
    large, _ = gaussian_incident_field((48, 48), 0.5e-6, 400e-6, 1.0)
    normalized = large / np.mean(large)
    assert float(np.max(np.abs(normalized - 1.0))) < 0.005


def test_nested_master_preserves_legacy_core_and_geometry(config: dict) -> None:
    nested = build_nested_master_b(config)
    master = nested["B_master_true"]
    core = nested["B_core_true"]
    legacy = make_random_phase_object(
        (192, 192), phase_range=0.8, seed=20260840, feature_size_px=4
    )
    assert master.shape == (256, 256)
    assert np.array_equal(core, legacy)
    assert np.array_equal(master[32:224, 32:224], legacy)
    positions = np.asarray(
        [[-9e-6, -9e-6], [0.0, 0.0], [9e-6, 9e-6]], dtype=np.float64
    )
    plan = make_scan_window_plan((256, 256), (192, 192), positions, 0.5e-6)
    assert plan.minimum_margin_m == pytest.approx(7e-6)
    assert np.all(plan.fully_inside)
    assert np.count_nonzero(plan.unvisited_region) > 0


def test_localized_multislice_zero_contrast_and_decomposition(config: dict) -> None:
    tiny = deepcopy(config)
    tiny["sample_a"].update(
        {
            "native_shape": [16, 16],
            "open_shape": [32, 32],
            "dx_m": 2.0e-6,
            "thickness_m": 8.0e-6,
            "target_dz_m": 2.0e-6,
            "interface_factor": 2,
            "d_top_m": 12.0e-6,
            "d_waist_m": 8.0e-6,
            "d_bottom_m": 12.0e-6,
            "z_waist_m": 4.0e-6,
        }
    )
    tiny["illumination"]["spot_diameter_1e2_intensity_m"] = 24.0e-6
    tiny["illumination"]["analytic_total_power_reference_m2"] = 1.0e-9
    zero = build_localized_multislice_probe(tiny, zero_contrast=True)
    nonzero_a = build_localized_multislice_probe(tiny)
    nonzero_b = build_localized_multislice_probe(tiny)
    assert zero["controls"]["zero_contrast_relative_l2"] <= 1e-12
    assert zero["controls"]["reference_plus_residual_A_to_B_relative_l2"] <= 1e-12
    assert np.array_equal(nonzero_a["P_B_native"], nonzero_b["P_B_native"])
    assert nonzero_a["P_B_native"].shape == (16, 16)
    assert nonzero_a["controls"]["all_finite"]


def test_exact_open_plane_control_is_uniform_and_repeatable(config: dict) -> None:
    tiny = deepcopy(config)
    tiny["sample_a"].update(
        {
            "native_shape": [16, 16],
            "open_shape": [32, 32],
            "dx_m": 2.0e-6,
            "thickness_m": 8.0e-6,
            "target_dz_m": 2.0e-6,
            "interface_factor": 2,
            "d_top_m": 12.0e-6,
            "d_waist_m": 8.0e-6,
            "d_bottom_m": 12.0e-6,
            "z_waist_m": 4.0e-6,
        }
    )
    first = build_localized_multislice_probe(tiny, incident_type="plane_wave")
    second = build_localized_multislice_probe(tiny, incident_type="plane_wave")
    assert np.all(first["incident_open"] == 1.0)
    assert first["power"]["is_exact_plane_wave"]
    assert np.array_equal(first["P_B_native"], second["P_B_native"])


def test_formal_config_preserves_causal_matrix_and_truth_boundary(config: dict) -> None:
    cases = {item["id"]: item for item in config["design_matrix"]}
    assert list(cases) == ["C0", "C1", "C2", "C3"]
    assert cases["C0"]["authoritative_bridge"] == "exp043_hash_locked_E43"
    assert cases["C0"]["large_canvas"] is False
    assert cases["C1"]["large_canvas"] is True
    assert cases["C0"]["illumination"] == cases["C1"]["illumination"]
    assert cases["C2"]["large_canvas"] is False
    assert cases["C3"]["large_canvas"] is True
    assert cases["C2"]["illumination"] == cases["C3"]["illumination"]
    assert config["known_b"]["truth_used_by_optimizer"] is False
    assert config["blind"]["truth_used_by_checkpoint_selection"] is False


def test_initial_append_only_body_prefix_is_byte_exact() -> None:
    path = (
        ROOT
        / "docs"
        / "experiment_design"
        / "exp044_TGV_3d_large_canvas_B_localized_illumination.md"
    )
    body = path.read_bytes()
    start = body.index(b"## 1.")
    prefix = body[start : start + 9816]
    assert len(prefix) == 9816
    assert hashlib.sha256(prefix).hexdigest().upper() == (
        "D64FB6D496F99BCF39D80AB44558C7ED86DCDED4960815E156AC7E6A1CBC0EA1"
    )


def test_invalid_incident_type_fails_fast(config: dict) -> None:
    with pytest.raises(ValueError, match="incident_type"):
        build_localized_multislice_probe(config, incident_type="B_plane_aperture")


def test_master_operator_extract_scatter_jacobian_and_replay() -> None:
    blind, known, probe, modulation = _tiny_operator()
    measured = blind.predict_stack(probe, modulation)
    controls = operator_control_metrics(
        known, blind, probe, modulation, measured, seed=45
    )
    assert controls["forward_replay_relative_l2"] == 0.0
    assert controls["propagation_linear_adjoint_relative_error"] <= 1e-12
    assert (
        controls["combined_intensity_jacobian_real_adjoint_relative_error"]
        <= 1e-11
    )
    assert controls["combined_intensity_jacobian_directional_relative_error"] <= 2e-6
    assert controls["repeat_prediction_relative_l2"] == 0.0
    assert controls["intensity_nonnegative"]


def test_known_b_and_blind_reconstruction_contracts() -> None:
    blind, known, probe_true, modulation_true = _tiny_operator()
    measured = blind.predict_stack(probe_true, modulation_true)
    reconstruction = {
        "known_sample_b": True,
        "update_sample_b": False,
        "truth_used_by_optimizer": False,
        "backtracking_factor": 0.5,
        "armijo_c": 1e-4,
        "max_backtracking_steps": 8,
        "minimum_step": 1e-12,
        "gradient_norm_stop": 0.0,
    }
    control = {
        "control_algorithm": "batch_spectrally_damped_gauss_newton_cg_armijo",
        "control_outer_iterations": 2,
        "spectral_radius_power_iterations": 2,
        "spectral_radius_seed": 46,
        "damping_relative_to_spectral_radius": 1e-3,
        "cg_max_iterations": 2,
        "cg_relative_residual_tolerance": 0.0,
        "line_search_initial_scale": 1.0,
        "truth_used_by_spectral_radius": False,
        "truth_used_by_damping": False,
        "truth_used_by_cg": False,
        "truth_used_by_branch_selection": False,
        "truth_used_by_stopping": False,
    }
    frozen_b = known.modulation.copy()
    known_result = reconstruct_known_b_probe_damped_gn_cg(
        known,
        measured,
        known.homogeneous_probe_native,
        reconstruction,
        control,
    )
    assert np.array_equal(known.modulation, frozen_b)
    assert known_result["loss_curve"][-1] <= known_result["loss_curve"][0]
    evaluation = global_phase_probe_evaluation_simulation_only(
        known_result["P_B_rec"], probe_true
    )
    assert evaluation["enters_optimizer"] is False

    settings = {
        "algorithm": "alternating_block_spectrally_damped_gn_cg",
        "outer_sweeps": 2,
        "checkpoint_selection_rule": "fixed_final_outer_sweep",
        "probe_block": {
            "outer_iterations": 1,
            "spectral_radius_power_iterations": 2,
            "spectral_radius_seed": 47,
            "damping_relative_to_spectral_radius": 1e-3,
            "cg_max_iterations": 2,
            "cg_relative_residual_tolerance": 0.0,
            "line_search_initial_scale": 1.0,
            "backtracking_factor": 0.5,
            "armijo_c": 1e-4,
            "max_backtracking_steps": 8,
            "minimum_step": 1e-12,
            "gradient_norm_stop": 0.0,
            "truth_used_by_optimizer": False,
            "truth_used_by_spectral_radius": False,
            "truth_used_by_damping": False,
            "truth_used_by_stopping": False,
        },
        "sample_b_block": {
            "outer_iterations": 1,
            "spectral_radius_power_iterations": 2,
            "spectral_radius_seed": 48,
            "damping_relative_to_spectral_radius": 1e-3,
            "cg_max_iterations": 2,
            "cg_relative_residual_tolerance": 0.0,
            "line_search_initial_scale": 1.0,
            "backtracking_factor": 0.5,
            "armijo_c": 1e-4,
            "max_backtracking_steps": 8,
            "minimum_step": 1e-12,
            "gradient_norm_stop": 0.0,
            "parameterization": "phase_only_unit_modulus_on_active_support",
            "truth_used_by_optimizer": False,
            "truth_used_by_spectral_radius": False,
            "truth_used_by_damping": False,
            "truth_used_by_stopping": False,
        },
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "truth_used_by_checkpoint_selection": False,
        "truth_used_by_stopping": False,
    }
    init_b = make_b_initialization(
        blind.support_mask,
        {"kind": "homogeneous_transmission", "seed": 49, "phase_rms_rad": 0.05},
    )
    blind_result = alternating_blind_reconstruction(
        blind,
        measured,
        known.homogeneous_probe_native,
        init_b,
        settings,
    )
    assert not np.array_equal(
        blind_result["P_B_rec_raw"], known.homogeneous_probe_native
    )
    assert np.any(np.abs(blind_result["B_rec_raw"][blind.support_mask] - 1.0) > 0)
    assert np.all(blind_result["B_rec_raw"][~blind.support_mask] == 1.0)
    evaluated = blind_component_evaluation_simulation_only(
        blind,
        blind_result,
        probe_true,
        1.0 + modulation_true,
        blind.scan_plan.coverage_map,
    )
    assert evaluated["observable_pixel_count"] == np.count_nonzero(blind.support_mask)
    assert evaluated["never_observed_pixel_count"] > 0
    assert np.all(np.isfinite(evaluated["per_scan_exit_product_relative_l2"]))


@pytest.mark.parametrize(
    ("forward", "known", "measurement_residual", "b_error", "expected"),
    [
        (False, True, 0.01, 0.1, "source_forward_or_operator_not_closed"),
        (True, False, 0.01, 0.1, "known_b_probe_recovery_not_closed"),
        (True, True, 0.05, 0.1, "blind_measurement_reconstruction_not_closed"),
        (
            True,
            True,
            0.01,
            0.4,
            "measurement_consistent_but_component_recovery_non_identifiable",
        ),
        (True, True, 0.01, 0.2, "all_registered_measurement_design_gates_passed"),
    ],
)
def test_status_matrix(
    forward: bool,
    known: bool,
    measurement_residual: float,
    b_error: float,
    expected: str,
) -> None:
    thresholds = {
        "blind_detector_relative_residual_max": 0.04,
        "repeat_prediction_relative_l2_max": 0.01,
        "blind_probe_aligned_relative_l2_max": 0.25,
        "blind_coverage_weighted_B_relative_l2_max": 0.35,
        "blind_exit_wave_product_relative_l2_max": 0.25,
        "required_relative_improvement_vs_exp043_B_min": 0.1,
        "required_relative_improvement_vs_exp043_exit_min": 0.1,
    }
    metrics = {
        "forward": {"all_gates_passed": forward},
        "known_b": {"all_case_gates_passed": known},
        "blind": {
            "C3": {
                "maximum_detector_relative_residual": measurement_residual,
                "maximum_pairwise_prediction_relative_l2": 0.0,
                "all_loss_nonincreasing": True,
                "all_finite": True,
                "fixed_final_selection": True,
                "representative_simulation_evaluation_only": {
                    "probe_aligned_relative_l2": 0.2,
                    "B_coverage_weighted_relative_l2": b_error,
                    "exit_wave_product_relative_l2": 0.2,
                    "relative_improvement_vs_exp043_B": 0.2,
                    "relative_improvement_vs_exp043_exit": 0.2,
                },
            }
        },
    }
    status = determine_exp044_status(metrics, thresholds)
    assert status["status_reason"] == expected


def test_invalid_master_operator_shapes_fail_fast() -> None:
    blind, _, _, _ = _tiny_operator()
    with pytest.raises(ValueError, match="exact scan-window union"):
        FiniteMasterBlindProbeBOperator(
            **{
                **blind.__dict__,
                "support_mask": np.ones(blind.open_shape, dtype=bool),
            }
        )
    assert relative_l2(np.ones((2, 2)), np.ones((2, 2))) == 0.0
