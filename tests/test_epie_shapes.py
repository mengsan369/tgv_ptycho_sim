from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.scan import make_grid_scan
from tgv_ptycho.forward.scheme_probe_B import simulate_probe_B_forward
from tgv_ptycho.objects.sample_b import make_random_amp_phase_object
from tgv_ptycho.optics.fields import make_gaussian_field
from tgv_ptycho.recon.epie import epie_reconstruct


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
