from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.helmholtz_axisymmetric import (
    annular_anisotropy_relative_l2,
    annular_mean_from_cartesian,
    assemble_cylindrical_helmholtz,
    background_interface_controls,
    make_axisymmetric_grid,
    make_background_n2,
    make_cylindrical_pml,
    make_manufactured_vector,
    make_tgv_n2_cell_average,
    observation_trace,
    pml_peak_alpha,
    radial_trace_to_cartesian,
    radial_weighted_relative_l2,
    solve_sparse_direct,
)


def _small_grid():
    return make_axisymmetric_grid(
        dr_m=0.25e-6,
        dz_m=0.25e-6,
        radial_core_max_m=2.0e-6,
        z_core_min_m=-0.5e-6,
        z_core_max_m=2.5e-6,
        pml_thickness_m=0.5e-6,
    )


def test_registered_grid_and_pml_coordinate_integral() -> None:
    grid = make_axisymmetric_grid(
        dr_m=1.0e-6,
        dz_m=1.0e-6,
        radial_core_max_m=4.0e-6,
        z_core_min_m=-1.0e-6,
        z_core_max_m=5.0e-6,
        pml_thickness_m=2.0e-6,
    )
    pml = make_cylindrical_pml(
        grid,
        wavelength_m=1.0e-6,
        n_glass=1.5,
        n_air=1.0,
        target_one_way_amplitude=1.0e-8,
    )
    assert (grid.nr, grid.nz, grid.unknown_count) == (6, 10, 60)
    physical = grid.r_centers_m < grid.radial_core_max_m
    np.testing.assert_array_equal(pml.r_stretch_centers[physical], 1.0 + 0.0j)
    np.testing.assert_array_equal(
        pml.r_tilde_centers_m[physical], grid.r_centers_m[physical]
    )
    k_air = 2.0 * np.pi / 1.0e-6
    attenuation_exponent = k_air * pml.r_tilde_faces_m[-1].imag
    np.testing.assert_allclose(attenuation_exponent, -np.log(1.0e-8))
    assert pml_peak_alpha(
        target_one_way_amplitude=1.0e-8,
        wavenumber_per_m=k_air,
        length_m=2.0e-6,
        polynomial_order=3,
    ) == pml.radial_peak_alpha


def test_scalar_background_is_continuous_in_value_and_derivative() -> None:
    controls = background_interface_controls(
        wavelength_m=532e-9,
        n_glass=1.5,
        n_air=1.0,
        interface_z_m=100e-6,
    )
    np.testing.assert_allclose(controls["reflection_coefficient"], 0.2)
    np.testing.assert_allclose(controls["transmission_coefficient"], 1.2)
    assert controls["value_continuity_relative_error"] <= 1.0e-15
    assert controls["derivative_continuity_relative_error"] <= 1.0e-15


def test_annular_material_fraction_conserves_subnode_volume() -> None:
    grid = make_axisymmetric_grid(
        dr_m=0.25e-6,
        dz_m=0.25e-6,
        radial_core_max_m=2.0e-6,
        z_core_min_m=-0.5e-6,
        z_core_max_m=2.5e-6,
        pml_thickness_m=0.5e-6,
    )
    n2, controls = make_tgv_n2_cell_average(
        grid,
        thickness_m=2.0e-6,
        d_top_m=2.0e-6,
        d_waist_m=1.0e-6,
        d_bottom_m=2.0e-6,
        z_waist_m=1.0e-6,
        n_glass=1.5,
        n_air=1.0,
        axial_subnodes=8,
        background_interface_z_m=2.0e-6,
    )
    assert n2.shape == (grid.nz, grid.nr)
    assert controls["fraction_bound_error"] == 0.0
    assert controls["annular_to_subnode_volume_relative_error"] <= 1.0e-15
    assert controls["subnode_to_exact_volume_relative_error_report_only"] < 1e-3


def test_small_sparse_operator_recovers_manufactured_vector() -> None:
    grid = _small_grid()
    pml = make_cylindrical_pml(
        grid,
        wavelength_m=0.8e-6,
        n_glass=1.5,
        n_air=1.0,
    )
    n2 = make_background_n2(
        grid,
        interface_z_m=2.0e-6,
        n_glass=1.5,
        n_air=1.0,
    )
    matrix, controls = assemble_cylindrical_helmholtz(
        grid, pml, n2, wavelength_m=0.8e-6
    )
    expected_max = 5 * grid.unknown_count - 2 * grid.nz - 2 * grid.nr
    assert controls["nnz"] <= expected_max
    np.testing.assert_allclose((matrix - matrix.T).data, 0.0, atol=0.0)
    manufactured = make_manufactured_vector(grid)
    solved, solve_controls = solve_sparse_direct(matrix, matrix @ manufactured)
    relative_error = np.linalg.norm(solved - manufactured) / np.linalg.norm(
        manufactured
    )
    assert solve_controls["relative_residual"] <= 1e-12
    assert relative_error <= 1e-11


def test_observation_mapping_and_annular_operators_preserve_constant() -> None:
    grid = _small_grid()
    values = np.broadcast_to(
        grid.z_centers_m[:, None].astype(np.complex128), (grid.nz, grid.nr)
    )
    trace, controls = observation_trace(
        values, grid, observation_z_m=1.0e-6
    )
    np.testing.assert_allclose(trace, 1.0e-6)
    np.testing.assert_allclose(controls["upper_weight"], 0.5)

    radial = np.ones(grid.nr, dtype=np.complex128) * (2.0 - 0.5j)
    mapped = radial_trace_to_cartesian(
        radial,
        grid.r_centers_m,
        shape=(32, 32),
        dx_m=0.125e-6,
        trace_support_radius_m=2.0e-6,
        outer_value=2.0 - 0.5j,
    )
    np.testing.assert_allclose(mapped, 2.0 - 0.5j)
    radii, means, counts = annular_mean_from_cartesian(
        mapped,
        dx_m=0.125e-6,
        bin_width_m=0.125e-6,
        maximum_radius_m=1.0e-6,
    )
    np.testing.assert_allclose(means, 2.0 - 0.5j)
    assert np.all(counts > 0)
    assert radial_weighted_relative_l2(means, means, radii) == 0.0
    anisotropy, projection = annular_anisotropy_relative_l2(
        mapped,
        dx_m=0.125e-6,
        bin_width_m=0.125e-6,
        maximum_radius_m=1.0e-6,
    )
    assert anisotropy == 0.0
    assert np.all(np.isfinite(projection[np.isfinite(projection)]))
