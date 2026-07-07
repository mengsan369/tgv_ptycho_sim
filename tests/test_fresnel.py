from __future__ import annotations

import numpy as np

from tgv_ptycho.optics.fresnel import fresnel_propagate


def test_fresnel_preserves_shape() -> None:
    U = np.ones((24, 32), dtype=np.complex128)
    Uz = fresnel_propagate(U, dx=2e-6, wavelength=532e-9, z=1e-3)
    assert Uz.shape == U.shape
    assert np.iscomplexobj(Uz)
