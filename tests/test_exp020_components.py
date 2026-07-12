from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.scan import add_integer_pixel_jitter, make_grid_scan
from tgv_ptycho.inverse.backprop_A import recover_thin_phase_A
from tgv_ptycho.inverse.metrics import (
    align_affine_phase_and_complex_gain,
    complex_relative_error,
)
from tgv_ptycho.objects.sample_a import make_smooth_random_thin_phase
from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.fields import make_plane_wave
from tgv_ptycho.recon.initialization import (
    initialize_probe_by_detector_backpropagation,
)


def test_thin_phase_A_is_seeded_pure_phase_with_blank_reference() -> None:
    args = ((32, 40), (2e-6, 3e-6), 20e-6, 0.5, 5e-6)
    A_1, phase_1, support_1 = make_smooth_random_thin_phase(*args, seed=17)
    A_2, phase_2, support_2 = make_smooth_random_thin_phase(*args, seed=17)

    assert A_1.shape == (32, 40)
    assert A_1.dtype == np.complex128
    assert phase_1.dtype == np.float64
    assert support_1.dtype == np.bool_
    assert np.array_equal(A_1, A_2)
    assert np.array_equal(phase_1, phase_2)
    assert np.array_equal(support_1, support_2)
    assert np.allclose(np.abs(A_1), 1.0)
    assert np.allclose(A_1[~support_1], 1.0 + 0.0j)
    assert np.isclose(np.sqrt(np.mean(phase_1[support_1] ** 2)), 0.5)


def test_integer_pixel_jitter_is_seeded_and_respects_axis_sampling() -> None:
    positions = make_grid_scan(3, 2, step=(6e-6, 8e-6))
    jittered_1 = add_integer_pixel_jitter(
        positions, dx=(2e-6, 3e-6), max_jitter_px=1, seed=4
    )
    jittered_2 = add_integer_pixel_jitter(
        positions, dx=(2e-6, 3e-6), max_jitter_px=1, seed=4
    )

    offsets = jittered_1 - positions
    assert np.array_equal(jittered_1, jittered_2)
    assert np.allclose(offsets[:, 0] / 3e-6, np.rint(offsets[:, 0] / 3e-6))
    assert np.allclose(offsets[:, 1] / 2e-6, np.rint(offsets[:, 1] / 2e-6))
    assert np.max(np.abs(offsets[:, 0])) <= 3e-6
    assert np.max(np.abs(offsets[:, 1])) <= 2e-6


def test_affine_phase_gain_alignment_recovers_periodic_ambiguity() -> None:
    rng = np.random.default_rng(8)
    truth = np.exp(1j * rng.normal(scale=0.4, size=(24, 32)))
    yy, xx = np.indices(truth.shape)
    ambiguity = 1.7 * np.exp(
        1j * (0.6 - 2 * np.pi * 3 * yy / 24 + 2 * np.pi * 5 * xx / 32)
    )
    reconstruction = truth * ambiguity

    aligned, gain, ramp = align_affine_phase_and_complex_gain(reconstruction, truth)

    assert isinstance(gain, complex)
    assert len(ramp) == 2
    assert complex_relative_error(aligned, truth) < 1e-12


def test_backprop_A_reference_correction_uses_blank_region() -> None:
    shape = (32, 32)
    dx = 2e-6
    wavelength = 532e-9
    z_AB = 1e-3
    A_true, _, support = make_smooth_random_thin_phase(
        shape, dx, radius=18e-6, phase_rms=0.35, correlation_length=5e-6, seed=9
    )
    incident = make_plane_wave(shape, dx, wavelength)
    probe = angular_spectrum_propagate(A_true * incident, dx, wavelength, z_AB)
    yy, xx = np.indices(shape)
    ambiguous_probe = 1.4 * np.exp(1j * 0.7) * probe
    recovered = recover_thin_phase_A(
        ambiguous_probe,
        incident,
        ~support,
        dx,
        wavelength,
        z_AB,
    )

    A_rec = np.asarray(recovered["A_rec_phase_only"])
    assert A_rec.shape == shape
    assert A_rec.dtype == np.complex128
    assert complex_relative_error(A_rec, A_true) < 1e-10


def test_probe_initialization_has_measured_mean_frame_energy() -> None:
    rng = np.random.default_rng(3)
    I_stack = rng.random((5, 12, 10))
    probe = initialize_probe_by_detector_backpropagation(
        I_stack, dx=2e-6, wavelength=532e-9, z_BC=1e-3
    )
    target_energy = float(np.mean(np.sum(I_stack, axis=(1, 2))))

    assert probe.shape == (12, 10)
    assert probe.dtype == np.complex128
    assert np.isclose(np.sum(np.abs(probe) ** 2), target_energy)
