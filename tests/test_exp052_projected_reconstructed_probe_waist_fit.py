from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp052_projected_reconstructed_probe_waist_fit as runner

from tgv_ptycho.forward.scheme_probe_B import simulate_probe_B_forward
from tgv_ptycho.inverse.exp050 import sha256_array
from tgv_ptycho.inverse.exp052 import (
    CandidateCache,
    combine_status,
    evaluate_fit_gates,
    fit_reconstructed_probe,
    load_exp052_sources,
    make_candidate_cache,
    profile_global_phase,
    replay_measurement_operator,
    run_known_b_reconstruction,
    validate_exp052_config,
)
from tgv_ptycho.io.config import load_config, save_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/exp052_TGV_2d_projected_reconstructed_probe_waist_fit.yaml"
)


def _synthetic_generator(
    true_waist_m: float,
    shape: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, Any]:
    yy, xx = np.indices(shape, dtype=np.float64)
    base = (
        0.8
        + 0.1 * yy / max(shape[0] - 1, 1)
        + 1j * (0.2 + 0.1 * xx / max(shape[1] - 1, 1))
    ).astype(np.complex128)
    direction = 0.3 + yy / max(shape[0] - 1, 1) + 1j * (0.4 + xx / max(shape[1] - 1, 1))

    def generate(diameter_m: float) -> dict[str, np.ndarray]:
        delta = (float(diameter_m) - true_waist_m) / true_waist_m
        return {"P_B": (base + delta * direction).astype(np.complex128)}

    return base, generate


def _synthetic_design(true_waist_m: float) -> dict[str, Any]:
    return {
        "bounds_m": [true_waist_m - 1.0e-6, true_waist_m + 1.0e-6],
        "global_step_m": 0.5e-6,
        "fine_half_width_m": 0.5e-6,
        "fine_step_m": 0.1e-6,
        "refine_half_width_m": 0.1e-6,
        "refine_step_m": 0.05e-6,
        "loss_tie_tolerance": 1.0e-20,
    }


def _fit_thresholds() -> dict[str, float]:
    return {
        "absolute_error_m_max": 0.1e-6,
        "truth_to_interval_distance_m_max": 0.1e-6,
        "interval_width_m_max": 0.1e-6,
        "method_agreement_m_max": 0.1e-6,
    }


def test_config_rejects_aligned_3d_and_truth_selected_sources() -> None:
    config = load_config(CONFIG_PATH)
    validate_exp052_config(config, mode="formal")

    aligned = deepcopy(config)
    aligned["source"]["exp030"]["target_dataset"] = (
        "/entry/reconstruction/simulation_evaluation_only/P_B_rec_aligned_to_truth"
    )
    with pytest.raises(ValueError, match="P_B_true"):
        validate_exp052_config(aligned, mode="formal")

    source_3d = deepcopy(config)
    source_3d["source"]["exp030"]["run"] = "runs/exp042_forbidden"
    with pytest.raises(ValueError, match="forbidden"):
        validate_exp052_config(source_3d, mode="formal")

    selected_checkpoint = deepcopy(config)
    selected_checkpoint["part_b"]["selection_rule"] = "best_truth_error_checkpoint"
    with pytest.raises(ValueError, match="fixed non-truth-selected"):
        validate_exp052_config(selected_checkpoint, mode="formal")

    wrong_gauge = deepcopy(config)
    wrong_gauge["gauge"]["primary"] = "truth_aligned"
    with pytest.raises(ValueError, match="global phase"):
        validate_exp052_config(wrong_gauge, mode="formal")


def test_authoritative_source_hashes_replay_and_blind_provenance() -> None:
    config = load_config(CONFIG_PATH)
    source = load_exp052_sources(config, PROJECT_ROOT)

    assert source["I_stack"].shape == (49, 384, 384)
    assert source["I_stack"].dtype == np.float64
    assert source["scan_positions"].shape == (49, 2)
    assert source["B_true"].dtype == np.complex128
    assert source["P_B_true"].dtype == np.complex128
    assert source["dataset_hashes"] == config["source"]["dataset_bytes_sha256"]
    assert source["target_dataset_units_attribute_present"] is False

    replay = replay_measurement_operator(source)
    assert replay["intensity_relative_l2"] == 0.0
    assert replay["amplitude_relative_l2"] == 0.0
    assert replay["adjoint_inner_product_relative_error"] < 1.0e-12

    cache = make_candidate_cache(source)
    replay_probe, first_index = cache.get(config["sample_a"]["d_waist_true_m"])
    repeated_probe, second_index = cache.get(config["sample_a"]["d_waist_true_m"])
    assert first_index == second_index == 0
    assert np.array_equal(replay_probe, repeated_probe)
    error = np.sqrt(
        np.sum(np.abs(replay_probe - source["P_B_true"]) ** 2)
        / np.sum(np.abs(source["P_B_true"]) ** 2)
    )
    assert error < 1.0e-12

    blind = source["blind"]
    assert blind["settings"]["update_probe"]
    assert blind["settings"]["update_object"]
    assert not blind["settings"]["uses_simulation_truth_B_as_input"]
    assert not blind["settings"]["uses_simulation_truth_probe_as_input"]
    assert not np.array_equal(blind["P_B_rec_raw"], blind["P_B_init"])
    assert not np.array_equal(blind["B_rec_raw"], blind["B_init"])
    assert np.array_equal(
        blind["P_B_rec_raw"], blind["checkpoints"]["1000"]["P_B_rec_raw"]
    )
    assert blind["durable_checkpoint_metadata"]["completed_iterations"] == 1000
    assert blind["durable_checkpoint_metadata"]["rng_bit_generator"] == "PCG64"
    assert source["oracle"]["estimate_m"] == pytest.approx(33.3e-6)


def test_authoritative_source_rejects_dataset_hash_mismatch() -> None:
    config = load_config(CONFIG_PATH)
    invalid = deepcopy(config)
    invalid["source"]["dataset_bytes_sha256"]["I_stack"] = "0" * 64
    with pytest.raises(ValueError, match="I_stack"):
        load_exp052_sources(invalid, PROJECT_ROOT)


def test_global_phase_is_primary_and_complex_scale_is_diagnostic() -> None:
    rng = np.random.default_rng(52)
    candidate = (rng.normal(size=(6, 7)) + 1j * rng.normal(size=(6, 7))).astype(
        np.complex128
    )
    phase = 0.83
    phase_target = np.exp(1j * phase) * candidate
    metrics = profile_global_phase(candidate, phase_target.astype(np.complex128))
    assert metrics["raw_loss"] > 0.1
    assert metrics["primary_loss"] < 1.0e-28
    assert metrics["global_phase_rad"] == pytest.approx(phase)
    assert np.allclose(
        metrics["global_phase_gain"] * candidate - phase_target,
        metrics["primary_residual"],
    )

    scaled_target = (2.0 * np.exp(1j * phase) * candidate).astype(np.complex128)
    scaled = profile_global_phase(candidate, scaled_target)
    assert scaled["primary_loss"] > 0.2
    assert scaled["complex_gain_profiled_loss_diagnostic"] < 1.0e-28

    with pytest.raises(ValueError, match="complex128"):
        profile_global_phase(candidate.real, phase_target.astype(np.complex128))


def test_profile_cache_refinement_equivalence_and_boundary_logic() -> None:
    truth = 3.0e-6
    target, generator = _synthetic_generator(truth)
    cache = CandidateCache(generator, target.shape)
    fit = fit_reconstructed_probe(
        (np.exp(0.4j) * target).astype(np.complex128),
        cache,
        design=_synthetic_design(truth),
    )
    assert fit["estimate_m"] == pytest.approx(truth)
    assert fit["global_local_minima_count"] == 1
    assert fit["equivalence_component_count"] == 1
    assert fit["equivalence_components_m"].shape == (1, 2)
    assert not fit["boundary_hit"]
    assert fit["local_loss_curvature_per_m2"] > 0.0
    assert np.array_equal(
        fit["P_B_best_raw"] - (np.exp(0.4j) * target).astype(np.complex128),
        fit["residual_field_raw"],
    )
    gates = evaluate_fit_gates(fit, true_waist_m=truth, thresholds=_fit_thresholds())
    assert gates["pass"]
    equality_thresholds = {
        "absolute_error_m_max": abs(fit["estimate_m"] - truth),
        "truth_to_interval_distance_m_max": 0.0,
        "interval_width_m_max": 0.05e-6,
        "method_agreement_m_max": fit["quadratic_profile_agreement_m"],
    }
    equality_gates = evaluate_fit_gates(
        fit, true_waist_m=truth, thresholds=equality_thresholds
    )
    assert equality_gates["pass"]
    assert equality_gates["interval_width_pass"]

    boundary_truth = 2.0e-6
    boundary_target, boundary_generator = _synthetic_generator(boundary_truth)
    boundary = fit_reconstructed_probe(
        boundary_target,
        CandidateCache(boundary_generator, boundary_target.shape),
        design=_synthetic_design(3.0e-6),
    )
    assert boundary["estimate_m"] == pytest.approx(boundary_truth)
    assert boundary["boundary_hit"]
    assert boundary["local_loss_curvature_per_m2"] == 0.0
    assert not evaluate_fit_gates(
        boundary,
        true_waist_m=boundary_truth,
        thresholds=_fit_thresholds(),
    )["pass"]

    def two_component_generator(diameter_m: float) -> dict[str, np.ndarray]:
        diameter_nm = int(round(float(diameter_m) * 1.0e9))
        distance_nm = min(abs(diameter_nm - 2950), abs(diameter_nm - 3050))
        return {
            "P_B": (target + (distance_nm / 1000.0) * np.ones_like(target)).astype(
                np.complex128
            )
        }

    multi_design = _synthetic_design(truth)
    multi_design["fine_step_m"] = 0.05e-6
    multi = fit_reconstructed_probe(
        target,
        CandidateCache(two_component_generator, target.shape),
        design=multi_design,
    )
    assert multi["equivalence_component_count"] == 2
    assert multi["equivalence_components_m"].shape == (2, 2)


def test_known_b_tiny_reconstruction_keeps_object_fixed_and_repeats() -> None:
    rng = np.random.default_rng(520)
    shape = (8, 8)
    dx = 1.0e-6
    wavelength = 532.0e-9
    z_bc = 0.2e-3
    probe = (0.8 + 0.2 * rng.random(shape)) * np.exp(
        1j * rng.normal(scale=0.2, size=shape)
    )
    sample_b = np.exp(1j * rng.normal(scale=0.4, size=shape)).astype(np.complex128)
    positions = np.asarray([[0.0, 0.0], [dx, 0.0], [0.0, -dx]], dtype=np.float64)
    intensity, _probe_true, _, _ = simulate_probe_B_forward(
        probe.astype(np.complex128),
        sample_b,
        positions,
        dx,
        wavelength,
        z_AB=0.0,
        z_BC=z_bc,
        incident_field=np.ones(shape, dtype=np.complex128),
    )
    source = {
        "I_stack": intensity,
        "scan_positions": positions,
        "B_true": sample_b,
        "instrument": {
            "dx_m": dx,
            "wavelength_m": wavelength,
            "z_BC_m": z_bc,
            "medium_index": 1.0,
        },
    }
    settings = {
        "num_iters": 2,
        "beta_probe": 0.2,
        "shuffle_positions": True,
        "seed": 521,
        "checkpoint_iterations": [1, 2],
    }
    result = run_known_b_reconstruction(source, settings, repeat=True)
    assert np.array_equal(result["B_known"], sample_b)
    assert np.array_equal(result["B_fixed"], sample_b)
    assert result["fixed_B_max_abs_change"] == 0.0
    assert result["repeat"]["bitwise_equal"]
    assert set(result["checkpoints"]) == {"1", "2"}
    assert result["checkpoints"]["1"]["loss_curve"].shape == (1,)
    assert result["checkpoints"]["2"]["optimizer_state"]["completed_iterations"] == 2
    assert result["checkpoints"]["2"]["optimizer_state"]["rng_bit_generator"] == (
        "PCG64"
    )
    assert result["settings"]["update_probe"]
    assert not result["settings"]["update_object"]
    assert result["final_data_fidelity_loss"] == pytest.approx(
        result["independent_data_fidelity_loss"], abs=1.0e-15
    )
    assert np.all(np.isfinite(result["loss_curve"]))


@pytest.mark.parametrize(
    (
        "shared",
        "a_recon",
        "a_fit",
        "b_recon",
        "b_fit",
        "expected_a",
        "expected_b",
        "expected_overall",
    ),
    [
        (
            "Passed",
            "Passed",
            "Passed",
            "Passed",
            "Passed",
            "Passed",
            "Passed",
            "Passed",
        ),
        (
            "Passed",
            "Passed",
            "Passed",
            "Failed",
            "Passed",
            "Passed",
            "Failed",
            "Failed",
        ),
        (
            "Passed",
            "Inconclusive",
            "Passed",
            "Passed",
            "Passed",
            "Inconclusive",
            "Passed",
            "Inconclusive",
        ),
        (
            "Inconclusive",
            "Passed",
            "Passed",
            "Failed",
            "Failed",
            "Inconclusive",
            "Inconclusive",
            "Inconclusive",
        ),
    ],
)
def test_status_matrix(
    shared: str,
    a_recon: str,
    a_fit: str,
    b_recon: str,
    b_fit: str,
    expected_a: str,
    expected_b: str,
    expected_overall: str,
) -> None:
    result = combine_status(
        shared_status=shared,
        part_a_reconstruction_status=a_recon,
        part_a_fit_status=a_fit,
        part_b_reconstruction_status=b_recon,
        part_b_fit_status=b_fit,
    )
    assert result["part_a_overall"] == expected_a
    assert result["part_b_overall"] == expected_b
    assert result["overall"] == expected_overall


def test_tiny_combined_runner_writes_auditable_raw_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = 3.0e-6
    target, generator = _synthetic_generator(truth)
    shape = target.shape
    phase_known = np.exp(0.4j)
    phase_blind = np.exp(-0.7j)
    known_target = (phase_known * target).astype(np.complex128)
    blind_target = (phase_blind * target).astype(np.complex128)
    sample_b = np.exp(0.1j * np.indices(shape, dtype=np.float64)[0]).astype(
        np.complex128
    )
    object_init = np.ones(shape, dtype=np.complex128)
    object_rec = (sample_b * np.exp(0.02j)).astype(np.complex128)
    intensity = np.ones((2, *shape), dtype=np.float64)
    positions = np.asarray([[0.0, 0.0], [1.0e-6, 0.0]], dtype=np.float64)

    config = load_config(CONFIG_PATH)
    config["run"]["output_root"] = str(tmp_path)
    config["output"]["run_name"] = "exp052_tiny"
    config["sample_a"]["d_waist_true_m"] = truth
    config["formal"] = {
        **_synthetic_design(truth),
        "part_a_reconstruction": {
            "num_iters": 2,
            "checkpoint_iterations": [1, 2],
            "repeat": True,
        },
        "thresholds": {
            "source_probe_replay_relative_l2_max": 1.0e-12,
            "measurement_intensity_replay_relative_l2_max": 1.0e-12,
            "measurement_amplitude_replay_relative_l2_max": 1.0e-12,
            "adjoint_inner_product_relative_error_max": 1.0e-12,
            "saved_residual_recompute_absolute_difference_max": 1.0e-12,
            "part_a_final_data_fidelity_loss_max": 1.0,
            "part_a_fixed_B_max_abs_change_max": 0.0,
            "part_a_repeat_relative_l2_max": 0.0,
            "part_b_final_data_fidelity_loss_max": 1.0,
            "part_b_probe_update_relative_l2_min": 1.0e-6,
            "part_b_object_update_relative_l2_min": 1.0e-6,
            "checkpoint_estimate_spread_m_max": 0.1e-6,
            **_fit_thresholds(),
        },
    }
    config["part_b"]["checkpoints_for_stability"] = [1, 2]
    config_path = tmp_path / "exp052_tiny.yaml"
    save_config(config_path, config)

    blind_checkpoints = {
        "1": {
            "P_B_rec_raw": blind_target,
            "B_rec_raw": object_rec,
            "data_fidelity_loss": 2.0e-3,
        },
        "2": {
            "P_B_rec_raw": blind_target,
            "B_rec_raw": object_rec,
            "data_fidelity_loss": 1.0e-3,
        },
    }
    source = {
        "I_stack": intensity,
        "scan_positions": positions,
        "B_true": sample_b,
        "P_B_true": target,
        "shape_ny_nx": shape,
        "instrument": {
            "wavelength_m": 532.0e-9,
            "dx_m": 1.0e-6,
            "z_AB_m": 1.0e-3,
            "z_BC_m": 1.0e-3,
            "detector_pixel_size_m": 1.0e-6,
            "medium_index": 1.0,
        },
        "exp030_hdf5_path": Path("synthetic_exp030.h5"),
        "dataset_hashes": {
            "I_stack": sha256_array(intensity),
            "scan_positions": sha256_array(positions),
            "B_true": sha256_array(sample_b),
            "P_B_true": sha256_array(target),
        },
        "source_radius_m": np.linspace(0.1e-6, 1.0e-6, 5),
        "source_weights_m": np.full(5, 0.2e-6),
        "output_radius_m": np.linspace(0.0, 1.0e-6, 5),
        "target_radial_probe": np.ones(5, dtype=np.complex128),
        "oracle": {
            "estimate_m": truth,
            "interval_m": np.asarray([truth - 0.025e-6, truth + 0.025e-6]),
            "interval_width_m": 0.05e-6,
            "loss_curvature_per_m2": 1.0,
        },
        "blind": {
            "P_B_init": (0.8 * target).astype(np.complex128),
            "B_init": object_init,
            "P_B_rec_raw": blind_target,
            "B_rec_raw": object_rec,
            "loss_curve": np.asarray([1.0e-2, 1.0e-3]),
            "final_data_fidelity_loss": 1.0e-3,
            "settings": {
                "update_probe": True,
                "update_object": True,
                "uses_simulation_truth_B_as_input": False,
                "uses_simulation_truth_probe_as_input": False,
                "problem_signature": "synthetic-signature",
                "rng_bit_generator": "PCG64",
            },
            "checkpoints": blind_checkpoints,
            "durable_checkpoint_path": "synthetic_checkpoint.h5",
            "durable_checkpoint_sha256": "1" * 64,
            "durable_checkpoint_metadata": {
                "completed_iterations": 2,
                "problem_signature": "synthetic-signature",
                "rng_bit_generator": "PCG64",
            },
        },
    }
    known_checkpoints = {
        "1": {
            "P_B_rec_raw": known_target,
            "B_fixed": sample_b,
            "data_fidelity_loss": 2.0e-3,
            "independent_data_fidelity_loss": 2.0e-3,
        },
        "2": {
            "P_B_rec_raw": known_target,
            "B_fixed": sample_b,
            "data_fidelity_loss": 1.0e-3,
            "independent_data_fidelity_loss": 1.0e-3,
        },
    }
    known = {
        "P_B_init": (0.7 * target).astype(np.complex128),
        "B_known": sample_b,
        "P_B_rec_raw": known_target,
        "B_fixed": sample_b,
        "loss_curve": np.asarray([1.0e-2, 1.0e-3]),
        "initial_data_fidelity_loss": 1.0e-2,
        "final_data_fidelity_loss": 1.0e-3,
        "independent_data_fidelity_loss": 1.0e-3,
        "fixed_B_max_abs_change": 0.0,
        "settings": {"update_probe": True, "update_object": False},
        "optimizer_state": {
            "completed_iterations": 2,
            "problem_signature": "known-signature",
            "rng_bit_generator": "PCG64",
            "rng_state_json": "{}",
        },
        "checkpoints": known_checkpoints,
        "repeat": {
            "executed": True,
            "P_B_relative_l2": 0.0,
            "B_relative_l2": 0.0,
            "loss_curve_relative_l2": 0.0,
            "final_loss_absolute_difference": 0.0,
            "bitwise_equal": True,
        },
    }

    monkeypatch.setattr(runner, "load_exp052_sources", lambda *_args: source)
    monkeypatch.setattr(
        runner,
        "make_candidate_cache",
        lambda _source: CandidateCache(generator, shape),
    )
    monkeypatch.setattr(
        runner,
        "replay_measurement_operator",
        lambda _source: {
            "intensity_relative_l2": 0.0,
            "amplitude_relative_l2": 0.0,
            "adjoint_inner_product_relative_error": 0.0,
        },
    )
    monkeypatch.setattr(
        runner, "run_known_b_reconstruction", lambda *_args, **_kwargs: known
    )
    monkeypatch.setattr(runner, "frozen_data_fidelity", lambda *_args: 1.0e-3)
    monkeypatch.setattr(
        runner,
        "truth_evaluation",
        lambda *_args: {
            "probe_global_phase_profiled_relative_l2": 0.0,
            "probe_complex_gain_profiled_relative_l2_diagnostic": 0.0,
            "object_global_phase_profiled_relative_l2": 0.0,
            "object_complex_gain_profiled_relative_l2_diagnostic": 0.0,
        },
    )

    run_dir = runner.run(config_path, mode="formal")
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert metrics["experiment_status"] == "Passed"
    assert len(metrics["artifact_audit"]["figures"]) == 7

    hdf5_path = run_dir / "outputs/exp052_projected_reconstructed_probe_waist_fit.h5"
    with h5py.File(hdf5_path, "r") as h5:
        assert set(h5["/entry"].keys()) == {
            "config_yaml",
            "data",
            "instrument",
            "sample",
            "truth",
            "reconstruction",
            "metadata",
            "metrics",
        }
        assert "calibration" not in h5["/entry"]
        assert "preprocessing" not in h5["/entry"]
        assert np.array_equal(
            h5["/entry/reconstruction/known_b_probe/P_B_rec_raw"][()],
            known_target,
        )
        assert np.array_equal(
            h5["/entry/reconstruction/blind_epie_probe/P_B_rec_raw"][()],
            blind_target,
        )
        assert np.array_equal(
            h5["/entry/reconstruction/blind_epie_probe/B_rec_raw"][()],
            object_rec,
        )
        assert (
            "/entry/reconstruction/waist_fit/known_b/P_B_rec_aligned_to_truth" not in h5
        )
        assert h5["/entry/data/I_stack"].dtype == np.float64
        assert h5["/entry/data/scan_positions"].shape == (2, 2)
