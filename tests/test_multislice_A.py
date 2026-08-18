from __future__ import annotations

import numpy as np
import pytest

from tgv_ptycho.forward.multislice_A import (
    multislice_phase_screen_product,
    multislice_propagate_A,
    multislice_propagate_streamed_A,
)
from tgv_ptycho.objects.tgv2d import make_tgv_projected_phase
from tgv_ptycho.objects.tgv3d import make_tgv_refractive_index_volume
from tgv_ptycho.objects.tgv_geometry import diameter_profile, midpoint_z_grid
from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.fields import coordinate_grid


def _relative_error(test: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return float(np.linalg.norm(test - reference) / denominator)


def test_midpoint_grid_uses_true_short_final_slice() -> None:
    centers, widths = midpoint_z_grid(thickness=10.5e-6, dz=4.0e-6)

    np.testing.assert_allclose(widths, [4.0e-6, 4.0e-6, 2.5e-6])
    np.testing.assert_allclose(centers, [2.0e-6, 6.0e-6, 9.25e-6])
    assert np.all(widths > 0.0)
    assert float(np.sum(widths)) == pytest.approx(10.5e-6, abs=1e-20)


def test_midpoint_grid_suppresses_numerical_zero_remainder() -> None:
    thickness = np.nextafter(40.0e-6, np.inf)
    centers, widths = midpoint_z_grid(thickness=thickness, dz=5.0e-6)

    assert centers.shape == (8,)
    assert widths.shape == (8,)
    assert np.all(widths > 0.0)
    assert float(np.sum(widths)) == pytest.approx(thickness, abs=1e-20)


def test_tgv_volume_uses_shared_midpoints_widths_and_anisotropic_grid() -> None:
    shape = (3, 17, 19)
    dx = (1.2e-6, 0.8e-6)
    center_xy = (0.8e-6, -1.2e-6)
    parameters = {
        "thickness": 10.5e-6,
        "d_top": 8.0e-6,
        "d_waist": 4.0e-6,
        "d_bottom": 6.0e-6,
        "z_waist": 5.0e-6,
    }
    volume, metadata = make_tgv_refractive_index_volume(
        shape,
        dx,
        4.0e-6,
        **parameters,
        n_glass=1.5,
        n_air=1.0,
        center_xy_m=center_xy,
    )

    z_m = np.asarray(metadata["z_m"])
    widths = np.asarray(metadata["slice_thickness_m"])
    expected_diameter = diameter_profile(z_m, **parameters)
    x_grid, y_grid = coordinate_grid(shape[1:], dx)
    radius = np.sqrt((x_grid - center_xy[0]) ** 2 + (y_grid - center_xy[1]) ** 2)

    assert volume.shape == shape
    assert volume.dtype == np.float64
    assert set(np.unique(volume)) <= {1.0, 1.5}
    np.testing.assert_allclose(metadata["diameter_z_m"], expected_diameter)
    assert float(np.sum(widths)) == pytest.approx(parameters["thickness"], abs=1e-20)
    for index, diameter in enumerate(expected_diameter):
        expected_air = radius <= diameter / 2.0
        assert np.array_equal(volume[index] == 1.0, expected_air)


def test_tgv_volume_includes_voxel_centers_on_via_boundary() -> None:
    volume, _ = make_tgv_refractive_index_volume(
        (1, 9, 9),
        1.0e-6,
        5.0e-6,
        5.0e-6,
        4.0e-6,
        4.0e-6,
        4.0e-6,
        n_glass=1.5,
        n_air=1.0,
    )

    assert volume[0, 4, 6] == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"shape_xyz": (2, 9, 9)},
        {"shape_xyz": (1.5, 9, 9)},
        {"dx": 0.0},
        {"n_air": np.nan},
        {"center_xy_m": (0.0, np.inf)},
    ],
)
def test_tgv_volume_rejects_inconsistent_or_nonfinite_inputs(
    overrides: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "shape_xyz": (1, 9, 9),
        "dx": 1.0e-6,
        "dz": 5.0e-6,
        "thickness": 5.0e-6,
        "d_top": 4.0e-6,
        "d_waist": 4.0e-6,
        "d_bottom": 4.0e-6,
        "n_glass": 1.5,
        "n_air": 1.0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        make_tgv_refractive_index_volume(**arguments)  # type: ignore[arg-type]


def test_single_slice_is_centered_half_step_screen_half_step() -> None:
    rng = np.random.default_rng(40)
    shape = (12, 10)
    dx = (0.9e-6, 1.1e-6)
    wavelength = 532.0e-9
    width = 2.3e-6
    n_ref = 1.45
    incident = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    n_slice = np.full((1, *shape), n_ref, dtype=np.float64)
    n_slice[0, 3:8, 2:7] = 1.1

    first_half = angular_spectrum_propagate(
        incident, dx, wavelength, width / 2.0, n=n_ref
    )
    transmission = np.exp(
        1j * 2.0 * np.pi / wavelength * (n_slice[0] - n_ref) * width
    )
    expected = angular_spectrum_propagate(
        first_half * transmission,
        dx,
        wavelength,
        width / 2.0,
        n=n_ref,
    )
    actual = multislice_propagate_A(
        incident, n_slice, dx, width, wavelength, n_ref=n_ref
    )

    assert _relative_error(actual, expected) <= 1e-12


def test_zero_contrast_matches_reference_propagation_over_total_thickness() -> None:
    rng = np.random.default_rng(41)
    shape = (14, 16)
    dx = 1.0e-6
    wavelength = 532.0e-9
    n_ref = 1.42
    widths = np.asarray([1.0e-6, 2.0e-6, 1.5e-6])
    incident = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    volume = np.full((len(widths), *shape), n_ref)

    actual = multislice_propagate_A(
        incident, volume, dx, widths, wavelength, n_ref=n_ref
    )
    expected = angular_spectrum_propagate(
        incident,
        dx,
        wavelength,
        float(np.sum(widths)),
        n=n_ref,
    )

    assert _relative_error(actual, expected) <= 1e-12


def test_uniform_scalar_and_vector_slice_widths_are_equivalent() -> None:
    rng = np.random.default_rng(42)
    shape = (10, 12)
    incident = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    volume = rng.uniform(1.1, 1.5, size=(3, *shape))
    scalar = multislice_propagate_A(
        incident, volume, 1.0e-6, 1.2e-6, 532.0e-9
    )
    vector = multislice_propagate_A(
        incident, volume, 1.0e-6, np.full(3, 1.2e-6), 532.0e-9
    )

    assert _relative_error(scalar, vector) <= 1e-12


def test_streamed_multislice_matches_full_volume_exactly() -> None:
    rng = np.random.default_rng(420)
    incident = rng.normal(size=(10, 12)) + 1j * rng.normal(size=(10, 12))
    volume = rng.uniform(1.0, 1.5, size=(4, 10, 12))
    widths = np.asarray([0.7e-6, 1.0e-6, 0.8e-6, 1.1e-6])

    full = multislice_propagate_A(
        incident, volume, 0.9e-6, widths, 532.0e-9
    )
    streamed = multislice_propagate_streamed_A(
        incident, iter(volume), 0.9e-6, widths, 532.0e-9
    )

    np.testing.assert_array_equal(streamed, full)


def test_streamed_alias_control_matches_full_alias_control() -> None:
    rng = np.random.default_rng(421)
    incident = rng.normal(size=(12, 14)) + 1j * rng.normal(size=(12, 14))
    volume = rng.uniform(1.0, 1.5, size=(3, 12, 14))
    widths = np.asarray([0.9e-6, 1.1e-6, 0.8e-6])

    full = multislice_propagate_A(
        incident,
        volume,
        0.8e-6,
        widths,
        532.0e-9,
        alias_control=True,
    )
    streamed = multislice_propagate_streamed_A(
        incident,
        iter(volume),
        0.8e-6,
        widths,
        532.0e-9,
        alias_control=True,
    )

    np.testing.assert_array_equal(streamed, full)


def test_homogeneous_plane_wave_retains_reference_carrier() -> None:
    shape = (8, 10)
    wavelength = 532.0e-9
    n_ref = 1.5
    widths = np.asarray([0.8e-6, 1.1e-6, 0.6e-6])
    incident = np.ones(shape, dtype=np.complex128)
    volume = np.full((3, *shape), n_ref)

    actual = multislice_propagate_A(
        incident, volume, 1.0e-6, widths, wavelength, n_ref=n_ref
    )
    carrier = np.exp(1j * 2.0 * np.pi * n_ref * np.sum(widths) / wavelength)

    assert _relative_error(actual, carrier * incident) <= 1e-12


def test_no_propagation_phase_product_matches_direct_discrete_integral() -> None:
    rng = np.random.default_rng(43)
    shape = (9, 11)
    wavelength = 532.0e-9
    n_ref = 1.5
    widths = np.asarray([1.0e-6, 0.7e-6, 1.4e-6])
    incident = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    volume = rng.uniform(1.0, 1.5, size=(3, *shape))

    actual = multislice_phase_screen_product(
        incident, volume, widths, wavelength, n_ref=n_ref
    )
    phase = 2.0 * np.pi / wavelength * np.sum(
        (volume - n_ref) * widths[:, None, None], axis=0
    )
    expected = incident * np.exp(1j * phase)

    assert _relative_error(actual, expected) <= 1e-12


def test_phase_product_matches_exp030_midpoint_projected_phase() -> None:
    shape = (31, 33)
    dx = (1.0e-6, 1.2e-6)
    wavelength = 532.0e-9
    thickness = 10.5e-6
    dz = 4.0e-6
    geometry = {
        "d_top": 8.0e-6,
        "d_waist": 4.0e-6,
        "d_bottom": 6.0e-6,
        "z_waist": 5.0e-6,
    }
    centers, widths = midpoint_z_grid(thickness, dz)
    volume, _ = make_tgv_refractive_index_volume(
        (len(centers), *shape),
        dx,
        dz,
        thickness,
        **geometry,
        n_glass=1.5,
        n_air=1.0,
    )
    projected = make_tgv_projected_phase(
        shape,
        dx,
        wavelength,
        thickness,
        **geometry,
        dz=dz,
        n_glass=1.5,
        n_air=1.0,
        integration_method="midpoint",
        lateral_supersampling=1,
    )

    actual = multislice_phase_screen_product(
        np.ones(shape, dtype=np.complex128),
        volume,
        widths,
        wavelength,
        n_ref=1.5,
    )

    assert _relative_error(actual, projected["A_effective_true"]) <= 1e-12


def test_multislice_is_deterministic_and_finite() -> None:
    rng = np.random.default_rng(44)
    incident = rng.normal(size=(8, 9)) + 1j * rng.normal(size=(8, 9))
    volume = rng.uniform(1.0, 1.5, size=(2, 8, 9))
    widths = np.asarray([0.8e-6, 1.1e-6])

    first = multislice_propagate_A(
        incident, volume, (1.0e-6, 1.2e-6), widths, 532.0e-9
    )
    second = multislice_propagate_A(
        incident, volume, (1.0e-6, 1.2e-6), widths, 532.0e-9
    )

    assert np.array_equal(first, second)
    assert first.dtype == np.complex128
    assert np.all(np.isfinite(first))


@pytest.mark.parametrize(
    ("widths", "wavelength", "n_ref", "volume_value"),
    [
        ([1.0e-6], 532.0e-9, 1.5, 1.2),
        ([1.0e-6, -1.0e-6], 532.0e-9, 1.5, 1.2),
        ([1.0e-6, 1.0e-6], 0.0, 1.5, 1.2),
        ([1.0e-6, 1.0e-6], 532.0e-9, np.nan, 1.2),
        ([1.0e-6, 1.0e-6], 532.0e-9, 1.5, np.nan),
    ],
)
def test_multislice_rejects_invalid_widths_or_optical_parameters(
    widths: list[float],
    wavelength: float,
    n_ref: float,
    volume_value: float,
) -> None:
    with pytest.raises(ValueError):
        multislice_propagate_A(
            np.ones((6, 7), dtype=np.complex128),
            np.full((2, 6, 7), volume_value),
            1.0e-6,
            widths,
            wavelength,
            n_ref=n_ref,
        )


def test_multislice_rejects_non_positive_sampling() -> None:
    with pytest.raises(ValueError, match="dx entries"):
        multislice_propagate_A(
            np.ones((6, 7), dtype=np.complex128),
            np.full((1, 6, 7), 1.5),
            (1.0e-6, 0.0),
            1.0e-6,
            532.0e-9,
        )
