from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tgv_ptycho.forward import scheme_probe_B
from tgv_ptycho.forward.integer_shift import shift_field_integer_pixels
from tgv_ptycho.forward.scheme_probe_B import (
    simulate_exit_field_B_forward,
    simulate_probe_B_forward,
)
from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate


def _fields(shape: tuple[int, int] = (10, 12)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(410)
    exit_field = (0.8 + 0.2 * rng.random(shape)) * np.exp(
        1j * rng.normal(scale=0.3, size=shape)
    )
    sample_b = np.exp(1j * rng.normal(scale=0.5, size=shape))
    return exit_field.astype(np.complex128), sample_b.astype(np.complex128)


def _positions() -> np.ndarray:
    return np.asarray(
        [[0.0, 0.0], [2.0e-6, -0.8e-6], [-1.0e-6, 1.6e-6]],
        dtype=np.float64,
    )


def test_exit_field_forward_matches_direct_chain() -> None:
    exit_field, sample_b = _fields()
    positions = _positions()
    dx = (0.8e-6, 1.0e-6)
    wavelength = 532e-9
    z_ab = 0.4e-3
    z_bc = 0.7e-3
    external_index = 1.23

    intensity, probe_b, returned_b, metadata = simulate_exit_field_B_forward(
        exit_field,
        sample_b,
        positions,
        dx,
        wavelength,
        z_ab,
        z_bc,
        external_medium_index=external_index,
        bandlimit=False,
    )

    expected_probe = angular_spectrum_propagate(
        exit_field,
        dx,
        wavelength,
        z_ab,
        n=external_index,
        bandlimit=False,
    )
    expected_frames = []
    for position_xy in positions:
        shifted_b = shift_field_integer_pixels(
            sample_b, position_xy, dx, boundary="periodic", fill_value=1.0 + 0.0j
        )
        detector_field = angular_spectrum_propagate(
            expected_probe * shifted_b,
            dx,
            wavelength,
            z_bc,
            n=external_index,
            bandlimit=False,
        )
        expected_frames.append(np.abs(detector_field) ** 2)

    assert intensity.shape == (len(positions), *exit_field.shape)
    assert intensity.dtype == np.float64
    assert probe_b.dtype == np.complex128
    assert np.allclose(probe_b, expected_probe)
    assert np.allclose(intensity, np.stack(expected_frames))
    assert np.array_equal(returned_b, sample_b)
    assert metadata["input_plane"] == "sample_A_exit"
    assert metadata["z_AB_reference_plane"] == "sample_A_exit"
    assert metadata["external_medium_index"] == external_index
    assert metadata["bandlimit"] is False


def test_exit_field_forward_is_finite_deterministic_and_index_dependent() -> None:
    exit_field, sample_b = _fields((8, 8))
    positions = _positions()
    kwargs: dict[str, Any] = {
        "dx": 0.9e-6,
        "wavelength": 532e-9,
        "z_AB": 0.3e-3,
        "z_BC": 0.5e-3,
    }

    first = simulate_exit_field_B_forward(
        exit_field,
        sample_b,
        positions,
        external_medium_index=1.0,
        **kwargs,
    )
    second = simulate_exit_field_B_forward(
        exit_field,
        sample_b,
        positions,
        external_medium_index=1.0,
        **kwargs,
    )
    changed_index = simulate_exit_field_B_forward(
        exit_field,
        sample_b,
        positions,
        external_medium_index=1.4,
        **kwargs,
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.all(np.isfinite(first[0]))
    assert np.all(np.isfinite(first[1]))
    assert not np.allclose(first[1], changed_index[1])
    assert first[3]["object_boundary"] == "periodic"
    assert first[3]["propagation_transfers_cached"] is True


def test_exit_field_forward_builds_only_ab_and_bc_transfers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_field, sample_b = _fields((8, 8))
    calls: list[float] = []
    original = scheme_probe_B.make_angular_spectrum_transfer

    def counting_transfer(*args: Any, **kwargs: Any) -> np.ndarray:
        calls.append(float(args[3]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scheme_probe_B, "make_angular_spectrum_transfer", counting_transfer
    )
    simulate_exit_field_B_forward(
        exit_field,
        sample_b,
        _positions(),
        1.0e-6,
        532e-9,
        0.2e-3,
        0.6e-3,
    )

    assert calls == [0.2e-3, 0.6e-3]


def test_exit_field_forward_alias_control_is_explicit_and_reported() -> None:
    exit_field, sample_b = _fields((32, 32))
    kwargs = {
        "scan_positions": _positions(),
        "dx": 0.5e-6,
        "wavelength": 532e-9,
        "z_AB": 0.5e-3,
        "z_BC": 1.0e-3,
    }

    current = simulate_exit_field_B_forward(exit_field, sample_b, **kwargs)
    controlled = simulate_exit_field_B_forward(
        exit_field,
        sample_b,
        **kwargs,
        alias_control=True,
    )

    assert current[3]["alias_control"] is False
    assert controlled[3]["alias_control"] is True
    assert not np.array_equal(current[1], controlled[1])
    assert np.all(np.isfinite(controlled[0]))


def test_legacy_probe_forward_matches_exit_field_helper() -> None:
    sample_a, sample_b = _fields((8, 8))
    incident = np.exp(1j * np.linspace(0.0, 0.4, 64)).reshape(8, 8)
    positions = _positions()
    args = (positions, 1.0e-6, 532e-9, 0.2e-3, 0.6e-3)

    legacy = simulate_probe_B_forward(
        sample_a,
        sample_b,
        *args,
        incident_field=incident,
    )
    direct = simulate_exit_field_B_forward(
        sample_a * incident,
        sample_b,
        *args,
    )

    assert np.array_equal(legacy[0], direct[0])
    assert np.array_equal(legacy[1], direct[1])
    assert np.array_equal(legacy[2], direct[2])
    assert legacy[3]["model"] == "scheme_probe_B"
    assert legacy[3]["z_AB_m"] == direct[3]["z_AB_m"]
    assert legacy[3]["z_BC_m"] == direct[3]["z_BC_m"]


def test_exit_field_forward_rejects_invalid_inputs() -> None:
    exit_field, sample_b = _fields((8, 8))
    positions = _positions()
    base = (sample_b, positions, 1.0e-6, 532e-9, 0.2e-3, 0.6e-3)

    with pytest.raises(ValueError, match="U_A_exit must be a 2D"):
        simulate_exit_field_B_forward(exit_field[0], *base)
    with pytest.raises(ValueError, match="B_object must be 2D"):
        simulate_exit_field_B_forward(exit_field, sample_b[:, :-1], *base[1:])
    with pytest.raises(ValueError, match="scan_positions must have shape"):
        simulate_exit_field_B_forward(
            exit_field, sample_b, np.zeros((3, 3)), *base[2:]
        )
    bad_positions = positions.copy()
    bad_positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="scan_positions must contain only finite"):
        simulate_exit_field_B_forward(exit_field, sample_b, bad_positions, *base[2:])
    with pytest.raises(ValueError, match="wavelength must be finite and positive"):
        simulate_exit_field_B_forward(
            exit_field, sample_b, positions, 1.0e-6, 0.0, 0.2e-3, 0.6e-3
        )
    with pytest.raises(ValueError, match="dx entries must be finite and positive"):
        simulate_exit_field_B_forward(
            exit_field, sample_b, positions, np.nan, 532e-9, 0.2e-3, 0.6e-3
        )
    with pytest.raises(ValueError, match="external_medium_index"):
        simulate_exit_field_B_forward(
            exit_field,
            sample_b,
            positions,
            1.0e-6,
            532e-9,
            0.2e-3,
            0.6e-3,
            external_medium_index=0.0,
        )
    with pytest.raises(ValueError, match="U_A_exit must contain only finite"):
        bad_field = exit_field.copy()
        bad_field[0, 0] = np.nan
        simulate_exit_field_B_forward(bad_field, *base)
    with pytest.raises(TypeError, match="bandlimit must be a bool"):
        simulate_exit_field_B_forward(
            exit_field,
            sample_b,
            positions,
            1.0e-6,
            532e-9,
            0.2e-3,
            0.6e-3,
            bandlimit=1,  # type: ignore[arg-type]
        )
