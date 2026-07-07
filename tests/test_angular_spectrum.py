from __future__ import annotations

import numpy as np

from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.fields import make_gaussian_field


def test_angular_spectrum_preserves_shape() -> None:
    U = np.ones((32, 40), dtype=np.complex128)
    Uz = angular_spectrum_propagate(U, dx=1e-6, wavelength=532e-9, z=1e-3)
    assert Uz.shape == U.shape
    assert np.iscomplexobj(Uz)


def test_angular_spectrum_roundtrip_low_frequency_field() -> None:
    U = make_gaussian_field((64, 64), dx=1e-6, waist=12e-6)
    Uz = angular_spectrum_propagate(U, dx=1e-6, wavelength=532e-9, z=2e-4)
    U_back = angular_spectrum_propagate(Uz, dx=1e-6, wavelength=532e-9, z=-2e-4)
    rel_err = np.linalg.norm(U_back - U) / np.linalg.norm(U)
    assert rel_err < 1e-10
