from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.helmholtz_axisymmetric import solve_sparse_direct
from tgv_ptycho.forward.helmholtz_benchmarks import (
    annular_outgoing_pml_benchmark,
    axial_plane_wave_pml_benchmark,
    axisymmetric_modal_nodal_error,
    decompose_cylindrical_field,
    make_annular_radial_fem_grid,
    make_axisymmetric_pml_modal_problem,
    normalized_cylindrical_bases,
    physical_k_modal_fem_benchmark,
)


def test_annular_grid_and_hankel_decomposition_are_exact() -> None:
    grid = make_annular_radial_fem_grid(
        degree=4,
        inner_radius_m=1.0,
        outer_radius_m=3.0,
        element_size_m=0.25,
    )
    assert grid.element_count == 8
    assert grid.active_unknown_count == 31

    radius = np.asarray([1.25, 1.75, 2.25])
    outgoing, incoming, outgoing_d, incoming_d = normalized_cylindrical_bases(
        radius, wavenumber_per_m=12.0, normalization_radius_m=1.0
    )
    expected_outgoing = 0.8 + 0.3j
    expected_incoming = -0.04 + 0.02j
    field = expected_outgoing * outgoing + expected_incoming * incoming
    derivative = (
        expected_outgoing * outgoing_d + expected_incoming * incoming_d
    )
    recovered_outgoing, recovered_incoming = decompose_cylindrical_field(
        field,
        derivative,
        radius,
        wavenumber_per_m=12.0,
        normalization_radius_m=1.0,
    )
    np.testing.assert_allclose(recovered_outgoing, expected_outgoing, rtol=1e-12)
    np.testing.assert_allclose(recovered_incoming, expected_incoming, rtol=1e-12)


def test_annular_pml_has_small_reflection_in_low_cost_control() -> None:
    result = annular_outgoing_pml_benchmark(
        wavelength_m=0.5,
        refractive_index=1.0,
        inner_radius_m=1.0,
        pml_start_m=2.0,
        pml_thickness_m=1.0,
        element_size_m=0.05,
        degree=4,
        quadrature_order=10,
        pml_polynomial_order=3,
        pml_target_one_way_amplitude=1e-8,
        measurement_radii_m=np.asarray([1.2, 1.5, 1.8]),
        dense_radii_m=np.linspace(1.05, 1.8, 101),
    )
    assert result["all_finite"]
    assert result["solver_controls"]["relative_residual"] < 1e-10
    assert result["maximum_incoming_to_outgoing_ratio"] < 1e-3
    assert result["maximum_outgoing_impedance_residual"] < 1e-3
    assert result["dense_field_weighted_relative_l2"] < 1e-4


def test_modal_p4_improves_over_p2_in_low_k_control() -> None:
    common = {
        "element_size_ratio": 0.5,
        "formal_kh": 4.0,
        "radial_extent": 4.0,
        "axial_extent": 4.0,
        "radial_mode": 2,
        "axial_mode": 2,
        "complex_amplitude": 1.0 + 0.2j,
        "discontinuous_mass": False,
        "interface_radius": 1.73,
        "homogeneous_n2": 1.0,
        "interface_inner_n2": 4.0 / 9.0,
        "interface_outer_n2": 1.0,
        "quadrature_order": 12,
        "evaluation_count_per_axis": 65,
    }
    p2 = physical_k_modal_fem_benchmark(degree=2, **common)
    p4 = physical_k_modal_fem_benchmark(degree=4, **common)
    assert p2["all_finite"] and p4["all_finite"]
    assert p4["weighted_relative_l2"] < 1e-3
    assert p4["weighted_relative_l2"] < 0.25 * p2["weighted_relative_l2"]


def test_stretched_modal_problem_matches_analytic_truth() -> None:
    problem = make_axisymmetric_pml_modal_problem(
        degree=4,
        element_size=0.25,
        core_extent=2.0,
        pml_thickness=0.5,
        operator_wavenumber=4.0,
        pml_polynomial_order=3,
        pml_target_one_way_amplitude=1e-6,
        interface_fraction_of_core_radius=0.4325,
        interface_inner_n2=4.0 / 9.0,
        interface_outer_n2=1.0,
        target_radial_modal_wavenumber=2.0,
        target_axial_modal_wavenumber=2.0,
        modal_index_offset=0,
        quadrature_order=12,
    )
    solution, controls = solve_sparse_direct(problem["matrix"], problem["rhs"])
    error = axisymmetric_modal_nodal_error(
        solution,
        problem["grid"],
        problem["exact_field"],
        core_extent=2.0,
    )
    assert controls["relative_residual"] < 1e-12
    assert error["weighted_relative_l2"] < 5e-3


def test_axial_plane_wave_pml_has_small_reflection() -> None:
    result = axial_plane_wave_pml_benchmark(
        wavelength_m=0.5,
        refractive_index=1.0,
        direction=-1,
        physical_core_length_m=2.0,
        pml_thickness_m=1.0,
        element_size_m=0.05,
        degree=4,
        quadrature_order=10,
        pml_polynomial_order=3,
        pml_target_one_way_amplitude=1e-8,
        measurement_fractions=np.asarray([0.2, 0.5, 0.8]),
        dense_comparison_count=101,
    )
    assert result["all_finite"]
    assert result["maximum_incoming_to_outgoing_ratio"] < 1e-3
    assert result["maximum_outgoing_impedance_residual"] < 1e-3
    assert result["dense_field_relative_l2"] < 1e-4
