from __future__ import annotations

import numpy as np
import pytest

from tgv_ptycho.objects.tgv2d import make_tgv_projected_phase, make_thin_phase_disk
from tgv_ptycho.objects.tgv3d import make_tgv_refractive_index_volume
from tgv_ptycho.objects.tgv_geometry import (
    analytic_air_path_length,
    diameter_profile,
)
from tgv_ptycho.optics.fields import coordinate_grid


def test_thin_phase_disk_shape_and_values() -> None:
    obj = make_thin_phase_disk((32, 32), dx=1e-6, diameter=10e-6, phase_shift=0.7)
    assert obj.shape == (32, 32)
    assert np.iscomplexobj(obj)
    assert np.isclose(np.abs(obj).max(), 1.0)


def test_tgv_refractive_index_volume_shape_and_metadata() -> None:
    volume, metadata = make_tgv_refractive_index_volume(
        shape_xyz=(8, 32, 32),
        dx=1e-6,
        dz=5e-6,
        thickness=40e-6,
        d_top=16e-6,
        d_waist=8e-6,
        d_bottom=14e-6,
        n_glass=1.5,
        n_air=1.0,
    )
    assert volume.shape == (8, 32, 32)
    assert metadata["d_waist_m"] == 8e-6
    assert np.isclose(volume.max(), 1.5)
    assert np.isclose(volume.min(), 1.0)


def test_shared_diameter_profile_shape_dtype_endpoints_and_waist() -> None:
    z = np.asarray([0.0, 30e-6, 100e-6], dtype=np.float64)
    diameter = diameter_profile(
        z,
        thickness=100e-6,
        d_top=20e-6,
        d_waist=8e-6,
        d_bottom=16e-6,
        z_waist=30e-6,
    )

    assert diameter.shape == z.shape
    assert diameter.dtype == np.float64
    assert np.allclose(diameter, [20e-6, 8e-6, 16e-6])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"d_top": 0.0}, "diameters must be positive"),
        ({"d_waist": 21e-6}, "must not exceed"),
        ({"z_waist": 0.0}, "inside"),
        ({"z_waist": 100e-6}, "inside"),
        ({"thickness": -1.0}, "thickness must be positive"),
    ],
)
def test_shared_diameter_profile_rejects_invalid_geometry(
    overrides: dict[str, float], match: str
) -> None:
    parameters = {
        "thickness": 100e-6,
        "d_top": 20e-6,
        "d_waist": 8e-6,
        "d_bottom": 18e-6,
        "z_waist": 40e-6,
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=match):
        diameter_profile(np.asarray([0.0]), **parameters)


def test_cylinder_projected_path_reduces_to_thin_phase_disk() -> None:
    shape = (31, 33)
    dx = 1e-6
    diameter = 10e-6
    thickness = 700e-6
    wavelength = 532e-9
    result = make_tgv_projected_phase(
        shape,
        dx,
        wavelength,
        thickness,
        diameter,
        diameter,
        diameter,
        dz=20e-9,
        n_glass=1.5,
        n_air=1.0,
        integration_method="analytic",
    )
    x_grid, y_grid = coordinate_grid(shape, dx)
    expected_path = np.where(
        x_grid**2 + y_grid**2 <= (diameter / 2.0) ** 2,
        thickness,
        0.0,
    )
    phase_shift = 2.0 * np.pi / wavelength * (1.0 - 1.5) * thickness
    expected_transmission = make_thin_phase_disk(
        shape, dx, diameter, phase_shift
    )

    assert np.array_equal(result["fill_path_length_m"], expected_path)
    assert np.allclose(result["A_effective_true"], expected_transmission)


def test_symmetric_linear_waist_matches_analytic_path() -> None:
    thickness = 100e-6
    r_top = 10e-6
    r_waist = 4e-6
    radius = np.asarray([0.0, r_waist, 7e-6, r_top, 12e-6])
    path = analytic_air_path_length(
        radius,
        thickness,
        d_top=2.0 * r_top,
        d_waist=2.0 * r_waist,
        d_bottom=2.0 * r_top,
        z_waist=thickness / 2.0,
    )
    expected = np.asarray(
        [
            thickness,
            thickness,
            thickness * (r_top - 7e-6) / (r_top - r_waist),
            0.0,
            0.0,
        ]
    )

    assert path.dtype == np.float64
    assert np.allclose(path, expected)


def test_projected_phase_is_deterministic_pure_phase_and_has_reference() -> None:
    kwargs = {
        "shape": (40, 48),
        "dx": (1e-6, 1.5e-6),
        "wavelength": 532e-9,
        "thickness": 60e-6,
        "d_top": 18e-6,
        "d_waist": 10e-6,
        "d_bottom": 16e-6,
        "dz": 0.5e-6,
        "z_waist": 25e-6,
        "lateral_supersampling": 2,
    }
    first = make_tgv_projected_phase(**kwargs)
    second = make_tgv_projected_phase(**kwargs)

    assert first["fill_path_length_m"].shape == (40, 48)
    assert first["fill_path_length_m"].dtype == np.float64
    assert first["A_effective_true"].dtype == np.complex128
    assert np.array_equal(
        first["fill_path_length_m"], second["fill_path_length_m"]
    )
    assert np.allclose(np.abs(first["A_effective_true"]), 1.0)
    assert np.allclose(first["A_effective_true"][:, [0, -1]], 1.0)


def test_zero_index_contrast_is_exact_unit_transmission() -> None:
    result = make_tgv_projected_phase(
        (24, 24),
        1e-6,
        532e-9,
        50e-6,
        16e-6,
        8e-6,
        16e-6,
        1e-6,
        n_glass=1.5,
        n_air=1.5,
    )
    assert np.max(np.abs(result["A_effective_true"] - 1.0)) <= 1e-12


@pytest.mark.parametrize("dx", [0.0, -1e-6, (1e-6, 0.0)])
def test_projected_phase_rejects_non_positive_sampling(
    dx: float | tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="dx values"):
        make_tgv_projected_phase(
            (16, 16),
            dx,
            532e-9,
            50e-6,
            16e-6,
            8e-6,
            16e-6,
            1e-6,
        )


def test_projected_phase_rejects_non_positive_dz() -> None:
    with pytest.raises(ValueError, match="dz"):
        make_tgv_projected_phase(
            (16, 16),
            1e-6,
            532e-9,
            50e-6,
            16e-6,
            8e-6,
            16e-6,
            0.0,
        )
