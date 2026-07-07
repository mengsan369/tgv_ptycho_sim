from __future__ import annotations

import numpy as np

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
