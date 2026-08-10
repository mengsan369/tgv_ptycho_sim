from __future__ import annotations

import numpy as np

from tgv_ptycho.optics.angular_spectrum import (
    angular_spectrum_propagate,
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)
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
    rel_err = np.sqrt(np.sum(np.abs(U_back - U) ** 2)) / np.sqrt(
        np.sum(np.abs(U) ** 2)
    )
    assert rel_err < 1e-10


def test_cached_angular_spectrum_transfer_matches_public_propagator() -> None:
    rng = np.random.default_rng(13)
    field = rng.normal(size=(31, 28)) + 1j * rng.normal(size=(31, 28))
    transfer = make_angular_spectrum_transfer(
        field.shape, (0.7e-6, 0.9e-6), 532e-9, 0.8e-3
    )

    cached = apply_angular_spectrum_transfer(field, transfer)
    direct = angular_spectrum_propagate(
        field, (0.7e-6, 0.9e-6), 532e-9, 0.8e-3
    )

    assert np.array_equal(cached, direct)


def test_bandlimited_angular_spectrum_uses_conjugate_as_adjoint() -> None:
    rng = np.random.default_rng(14)
    left = rng.normal(size=(24, 22)) + 1j * rng.normal(size=(24, 22))
    right = rng.normal(size=(24, 22)) + 1j * rng.normal(size=(24, 22))
    transfer = make_angular_spectrum_transfer(
        left.shape, 0.25e-6, 532e-9, 1.0e-3
    )

    forward_right = apply_angular_spectrum_transfer(right, transfer)
    adjoint_left = apply_angular_spectrum_transfer(left, np.conj(transfer))
    lhs = np.sum(np.conj(left) * forward_right)
    rhs = np.sum(np.conj(adjoint_left) * right)

    scale = max(abs(lhs), abs(rhs), np.finfo(float).eps)
    assert abs(lhs - rhs) / scale <= 1e-12
