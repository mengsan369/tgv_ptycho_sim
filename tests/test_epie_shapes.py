from __future__ import annotations

import copy

import numpy as np
import pytest

from tgv_ptycho.forward.scan import make_grid_scan
from tgv_ptycho.forward.scheme_probe_B import simulate_probe_B_forward
from tgv_ptycho.objects.sample_b import make_random_amp_phase_object
from tgv_ptycho.optics.fields import make_gaussian_field
from tgv_ptycho.recon.epie import epie_reconstruct


def _make_resume_test_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    rng = np.random.default_rng(940)
    shape = (8, 8)
    intensities = rng.uniform(0.2, 1.8, size=(3, *shape))
    positions = np.asarray(
        [[0.0, 0.0], [1.0e-6, 0.0], [0.0, -1.0e-6]], dtype=np.float64
    )
    probe = (
        rng.uniform(0.7, 1.1, size=shape)
        * np.exp(1j * rng.normal(scale=0.2, size=shape))
    ).astype(np.complex128)
    obj = (
        rng.uniform(0.8, 1.0, size=shape)
        * np.exp(1j * rng.normal(scale=0.3, size=shape))
    ).astype(np.complex128)
    kwargs: dict[str, object] = {
        "dx": 1.0e-6,
        "wavelength": 532.0e-9,
        "z_BC": 0.4e-3,
        "beta_probe": 0.08,
        "beta_object": 0.12,
        "shuffle_positions": True,
        "seed": 941,
        "object_amplitude_bounds": (0.7, 1.2),
        "probe_l2_norm_target": 7.0,
        "show_progress": False,
    }
    return intensities, positions, probe, obj, kwargs


def test_epie_zero_iterations_returns_expected_shapes() -> None:
    I_stack = np.ones((3, 16, 16), dtype=np.float64)
    positions = np.zeros((3, 2), dtype=np.float64)
    result = epie_reconstruct(
        I_stack,
        positions,
        dx=1e-6,
        wavelength=532e-9,
        z_BC=1e-3,
        num_iters=0,
        init_probe=np.ones((16, 16), dtype=np.complex128),
        init_object=np.ones((16, 16), dtype=np.complex128),
    )
    assert result["P_B_rec"].shape == (16, 16)
    assert result["B_rec"].shape == (16, 16)
    assert result["loss_curve"].shape == (0,)


def test_known_probe_epie_keeps_probe_fixed_and_reduces_loss() -> None:
    shape = (32, 32)
    dx = 2e-6
    wavelength = 532e-9
    z_BC = 2e-3
    probe = make_gaussian_field(shape, dx, waist=12e-6)
    B_true = make_random_amp_phase_object(
        shape,
        amp_range=(0.8, 1.0),
        phase_range=0.5,
        seed=11,
        feature_size_px=2,
    )
    positions = make_grid_scan(5, 5, step=4e-6)
    I_stack, P_B_true, _, _ = simulate_probe_B_forward(
        probe,
        B_true,
        positions,
        dx,
        wavelength,
        z_AB=0.0,
        z_BC=z_BC,
        incident_field=np.ones(shape, dtype=np.complex128),
    )

    result = epie_reconstruct(
        I_stack,
        positions,
        dx,
        wavelength,
        z_BC,
        num_iters=12,
        beta_object=0.8,
        init_probe=P_B_true,
        init_object=np.ones(shape, dtype=np.complex128),
        update_probe=False,
        seed=12,
        object_amplitude_bounds=(0.0, 1.2),
        show_progress=False,
    )

    assert np.array_equal(result["P_B_rec"], P_B_true)
    assert result["loss_curve"].shape == (12,)
    assert result["loss_curve"][-1] < 0.5 * result["loss_curve"][0]
    assert result["illumination_map"].shape == shape


def test_blind_epie_enforces_probe_l2_norm() -> None:
    I_stack = np.ones((2, 8, 8), dtype=np.float64)
    positions = np.zeros((2, 2), dtype=np.float64)
    target = 5.0
    result = epie_reconstruct(
        I_stack,
        positions,
        dx=1e-6,
        wavelength=532e-9,
        z_BC=0.0,
        num_iters=1,
        beta_probe=0.1,
        beta_object=0.1,
        init_probe=np.ones((8, 8), dtype=np.complex128),
        init_object=np.ones((8, 8), dtype=np.complex128),
        update_probe=True,
        object_amplitude_bounds=(1.0, 1.0),
        probe_l2_norm_target=target,
        show_progress=False,
    )

    probe_norm = np.sqrt(np.sum(np.abs(result["P_B_rec"]) ** 2))
    assert np.isclose(probe_norm, target)


def test_blind_epie_applies_probe_constraint() -> None:
    def real_only(probe: np.ndarray) -> np.ndarray:
        return np.abs(probe).astype(np.complex128)

    result = epie_reconstruct(
        np.ones((1, 8, 8), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        dx=1e-6,
        wavelength=532e-9,
        z_BC=0.0,
        num_iters=1,
        init_probe=np.exp(0.5j) * np.ones((8, 8), dtype=np.complex128),
        probe_constraint=real_only,
        show_progress=False,
    )

    assert np.allclose(np.imag(result["P_B_rec"]), 0.0)
    assert result["metadata"]["probe_constraint"] == "real_only"


def test_adjoint_residual_keeps_exact_truth_fixed_with_bandlimit() -> None:
    shape = (32, 32)
    dx = 0.25e-6
    wavelength = 532e-9
    z_bc = 1.0e-3
    rng = np.random.default_rng(43)
    probe = np.exp(1j * rng.normal(scale=0.2, size=shape))
    sample_b = np.exp(1j * rng.normal(scale=0.4, size=shape))
    positions = make_grid_scan(3, 3, step=2.0e-6)
    intensity, probe_true, _, _ = simulate_probe_B_forward(
        probe,
        sample_b,
        positions,
        dx,
        wavelength,
        z_AB=0.0,
        z_BC=z_bc,
        incident_field=np.ones(shape, dtype=np.complex128),
    )

    object_only = epie_reconstruct(
        intensity,
        positions,
        dx,
        wavelength,
        z_bc,
        num_iters=1,
        beta_object=0.5,
        init_probe=probe_true,
        init_object=sample_b,
        update_probe=False,
        update_object=True,
        shuffle_positions=False,
        object_amplitude_bounds=(1.0, 1.0),
        show_progress=False,
    )
    probe_only = epie_reconstruct(
        intensity,
        positions,
        dx,
        wavelength,
        z_bc,
        num_iters=1,
        beta_probe=0.2,
        init_probe=probe_true,
        init_object=sample_b,
        update_probe=True,
        update_object=False,
        shuffle_positions=False,
        show_progress=False,
    )

    object_change = np.linalg.norm(object_only["B_rec"] - sample_b) / np.linalg.norm(
        sample_b
    )
    probe_change = np.linalg.norm(probe_only["P_B_rec"] - probe_true) / np.linalg.norm(
        probe_true
    )
    assert object_change <= 1e-11
    assert probe_change <= 1e-11
    assert object_only["final_data_fidelity_loss"] <= 1e-11
    assert probe_only["final_data_fidelity_loss"] <= 1e-11


def test_epie_records_requested_checkpoints() -> None:
    result = epie_reconstruct(
        np.ones((1, 8, 8), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        dx=1e-6,
        wavelength=532e-9,
        z_BC=0.0,
        num_iters=3,
        init_probe=np.ones((8, 8), dtype=np.complex128),
        init_object=np.ones((8, 8), dtype=np.complex128),
        checkpoint_iters=(0, 2, 3),
        show_progress=False,
    )

    assert set(result["checkpoints"]) == {"0", "2", "3"}
    assert result["checkpoints"]["2"]["P_B_rec"].shape == (8, 8)
    assert np.isfinite(result["checkpoints"]["3"]["data_fidelity_loss"])


def test_epie_resume_matches_uninterrupted_shuffled_trajectory_exactly() -> None:
    intensities, positions, probe, obj, kwargs = _make_resume_test_problem()
    uninterrupted = epie_reconstruct(
        intensities,
        positions,
        num_iters=6,
        init_probe=probe,
        init_object=obj,
        **kwargs,
    )
    first_segment = epie_reconstruct(
        intensities,
        positions,
        num_iters=2,
        init_probe=probe,
        init_object=obj,
        **kwargs,
    )
    resumed = epie_reconstruct(
        intensities,
        positions,
        num_iters=6,
        resume_state=first_segment["optimizer_state"],
        **kwargs,
    )

    assert np.array_equal(resumed["P_B_rec"], uninterrupted["P_B_rec"])
    assert np.array_equal(resumed["B_rec"], uninterrupted["B_rec"])
    assert np.array_equal(resumed["loss_curve"], uninterrupted["loss_curve"])
    assert (
        resumed["final_data_fidelity_loss"]
        == uninterrupted["final_data_fidelity_loss"]
    )
    assert resumed["metadata"]["resumed_from_iteration"] == 2
    assert resumed["completed_iterations"] == 6


def test_epie_checkpoint_callback_reports_isolated_iteration_state() -> None:
    intensities, positions, probe, obj, kwargs = _make_resume_test_problem()
    reference = epie_reconstruct(
        intensities,
        positions,
        num_iters=4,
        init_probe=probe,
        init_object=obj,
        checkpoint_iters=(0, 2, 4),
        **kwargs,
    )
    observed: list[tuple[int, int]] = []

    def mutate_snapshot(checkpoint: dict[str, object]) -> None:
        state = checkpoint["optimizer_state"]
        assert isinstance(state, dict)
        completed = int(state["completed_iterations"])
        loss = np.asarray(state["loss_curve"])
        observed.append((completed, int(loss.size)))
        np.asarray(checkpoint["P_B_rec"])[...] = 0.0
        np.asarray(checkpoint["B_rec"])[...] = 0.0
        np.asarray(state["P_B_rec"])[...] = 0.0
        np.asarray(state["B_rec"])[...] = 0.0
        if loss.size:
            loss[...] = np.nan

    with_callback = epie_reconstruct(
        intensities,
        positions,
        num_iters=4,
        init_probe=probe,
        init_object=obj,
        checkpoint_iters=(0, 2, 4),
        checkpoint_callback=mutate_snapshot,
        **kwargs,
    )

    assert observed == [(0, 0), (2, 2), (4, 4)]
    assert np.array_equal(with_callback["P_B_rec"], reference["P_B_rec"])
    assert np.array_equal(with_callback["B_rec"], reference["B_rec"])
    assert np.array_equal(with_callback["loss_curve"], reference["loss_curve"])
    assert np.any(with_callback["checkpoints"]["2"]["P_B_rec"] != 0.0)
    checkpoint_loss = with_callback["checkpoints"]["4"]["optimizer_state"][
        "loss_curve"
    ]
    assert np.all(np.isfinite(checkpoint_loss))


def test_epie_noop_resume_preserves_completed_state() -> None:
    intensities, positions, probe, obj, kwargs = _make_resume_test_problem()
    completed = epie_reconstruct(
        intensities,
        positions,
        num_iters=3,
        init_probe=probe,
        init_object=obj,
        **kwargs,
    )
    resumed = epie_reconstruct(
        intensities,
        positions,
        num_iters=3,
        resume_state=completed["optimizer_state"],
        checkpoint_iters=(3,),
        **kwargs,
    )

    assert np.array_equal(resumed["P_B_rec"], completed["P_B_rec"])
    assert np.array_equal(resumed["B_rec"], completed["B_rec"])
    assert np.array_equal(resumed["loss_curve"], completed["loss_curve"])
    assert (
        resumed["optimizer_state"]["rng_state"]
        == completed["optimizer_state"]["rng_state"]
    )
    assert resumed["metadata"]["resumed_from_iteration"] == 3
    assert set(resumed["checkpoints"]) == {"3"}


def test_epie_resume_rejects_problem_signature_mismatch() -> None:
    intensities, positions, probe, obj, kwargs = _make_resume_test_problem()
    first_segment = epie_reconstruct(
        intensities,
        positions,
        num_iters=2,
        init_probe=probe,
        init_object=obj,
        **kwargs,
    )
    changed_kwargs = dict(kwargs)
    changed_kwargs["beta_probe"] = 0.081

    with pytest.raises(ValueError, match="does not match"):
        epie_reconstruct(
            intensities,
            positions,
            num_iters=4,
            resume_state=first_segment["optimizer_state"],
            **changed_kwargs,
        )


def test_epie_resume_rejects_loss_history_length_mismatch() -> None:
    intensities, positions, probe, obj, kwargs = _make_resume_test_problem()
    first_segment = epie_reconstruct(
        intensities,
        positions,
        num_iters=2,
        init_probe=probe,
        init_object=obj,
        **kwargs,
    )
    invalid_state = copy.deepcopy(first_segment["optimizer_state"])
    invalid_state["loss_curve"] = np.asarray(invalid_state["loss_curve"])[0:1]

    with pytest.raises(ValueError, match="loss_curve length"):
        epie_reconstruct(
            intensities,
            positions,
            num_iters=4,
            resume_state=invalid_state,
            **kwargs,
        )


def test_constant_boundary_is_fixed_when_forward_and_inverse_match() -> None:
    shape = (16, 16)
    dx = 0.5e-6
    wavelength = 532e-9
    rng = np.random.default_rng(44)
    probe = np.exp(1j * rng.normal(scale=0.1, size=shape))
    sample_b = np.exp(1j * rng.normal(scale=0.2, size=shape))
    positions = np.asarray([[2.0e-6, -1.5e-6], [-1.0e-6, 1.5e-6]])
    intensity, probe_true, _, _ = simulate_probe_B_forward(
        probe,
        sample_b,
        positions,
        dx,
        wavelength,
        z_AB=0.0,
        z_BC=0.7e-3,
        incident_field=np.ones(shape, dtype=np.complex128),
        object_boundary="constant",
        object_boundary_value=1.0 + 0.0j,
    )

    result = epie_reconstruct(
        intensity,
        positions,
        dx,
        wavelength,
        0.7e-3,
        num_iters=1,
        init_probe=probe_true,
        init_object=sample_b,
        update_probe=True,
        update_object=False,
        object_boundary="constant",
        object_boundary_value=1.0 + 0.0j,
        shuffle_positions=False,
        show_progress=False,
    )

    change = np.linalg.norm(result["P_B_rec"] - probe_true) / np.linalg.norm(
        probe_true
    )
    assert change <= 1e-11
