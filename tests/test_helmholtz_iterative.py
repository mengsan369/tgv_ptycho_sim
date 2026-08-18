from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.helmholtz_axisymmetric import solve_sparse_direct
from tgv_ptycho.forward.helmholtz_benchmarks import (
    make_axisymmetric_pml_modal_problem,
)
from tgv_ptycho.forward.helmholtz_iterative import (
    build_csl_ilu_preconditioner,
    build_two_level_ras_csl_preconditioner,
    solve_restarted_gmres,
    sparse_storage_bytes,
)


def _low_k_pair():
    common = {
        "degree": 2,
        "element_size": 0.5,
        "core_extent": 4.0,
        "pml_thickness": 1.0,
        "operator_wavenumber": 4.0,
        "pml_polynomial_order": 3,
        "pml_target_one_way_amplitude": 1e-6,
        "interface_fraction_of_core_radius": 0.4325,
        "interface_inner_n2": 4.0 / 9.0,
        "interface_outer_n2": 1.0,
        "target_radial_modal_wavenumber": 2.0,
        "target_axial_modal_wavenumber": 2.0,
        "modal_index_offset": 0,
        "quadrature_order": 8,
    }
    original = make_axisymmetric_pml_modal_problem(**common)
    shifted = make_axisymmetric_pml_modal_problem(
        **common, imaginary_mass_shift=0.5
    )
    return original, shifted


def _solve(matrix, rhs, preconditioner):
    return solve_restarted_gmres(
        matrix,
        rhs,
        preconditioner,
        relative_tolerance=1e-8,
        absolute_tolerance=0.0,
        restart=20,
        maximum_inner_iterations=100,
    )


def test_csl_ilu_and_two_level_ras_agree_with_direct_low_k_solution() -> None:
    original, shifted = _low_k_pair()
    direct, _ = solve_sparse_direct(original["matrix"], original["rhs"])
    ilu = build_csl_ilu_preconditioner(
        shifted["matrix"],
        drop_tolerance=1e-3,
        fill_factor=4.0,
        drop_rule="basic,area",
        permc_spec="COLAMD",
    )
    ilu_solution, ilu_controls = _solve(
        original["matrix"], original["rhs"], ilu.operator
    )
    ras = build_two_level_ras_csl_preconditioner(
        shifted["matrix"],
        active_shape=tuple(original["controls"]["active_shape"]),
        core_block_shape_nodes=(16, 16),
        overlap_nodes=2,
    )
    ras_solution, ras_controls = _solve(
        original["matrix"], original["rhs"], ras.operator
    )

    for solution, controls in (
        (ilu_solution, ilu_controls),
        (ras_solution, ras_controls),
    ):
        assert controls["converged"]
        assert controls["true_relative_residual"] < 1e-8
        assert np.linalg.norm(solution - direct) / np.linalg.norm(direct) < 1e-6
    assert ilu.controls["full_global_factorization"] is False
    assert ras.controls["full_global_factorization"] is False
    assert sparse_storage_bytes(original["matrix"]) > 0


def test_preconditioner_linear_operators_are_repeatable() -> None:
    original, shifted = _low_k_pair()
    ilu = build_csl_ilu_preconditioner(
        shifted["matrix"],
        drop_tolerance=1e-3,
        fill_factor=4.0,
        drop_rule="basic,area",
        permc_spec="COLAMD",
    )
    vector = np.linspace(0.0, 1.0, original["matrix"].shape[0]).astype(
        np.complex128
    )
    first = ilu.operator @ vector
    second = ilu.operator @ vector
    np.testing.assert_allclose(first, second, rtol=1e-14, atol=1e-14)
