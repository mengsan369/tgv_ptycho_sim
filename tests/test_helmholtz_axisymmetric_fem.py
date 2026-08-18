from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.helmholtz_axisymmetric_fem import (
    gauss_lobatto_nodes,
    make_axisymmetric_fem_grid,
    manufactured_fem_benchmark,
)


def test_gauss_lobatto_nodes_and_registered_grid_counts() -> None:
    np.testing.assert_allclose(gauss_lobatto_nodes(2), [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        gauss_lobatto_nodes(3),
        [-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0],
    )
    p2 = make_axisymmetric_fem_grid(
        degree=2,
        radial_extent_m=62.0e-6,
        z_min_m=-4.0e-6,
        z_max_m=104.0e-6,
        radial_element_size_m=0.5e-6,
        axial_element_size_m=0.5e-6,
    )
    p3 = make_axisymmetric_fem_grid(
        degree=3,
        radial_extent_m=62.0e-6,
        z_min_m=-4.0e-6,
        z_max_m=104.0e-6,
        radial_element_size_m=0.5e-6,
        axial_element_size_m=0.5e-6,
    )

    assert p2.active_unknown_count == 106888
    assert p3.active_unknown_count == 240684


def test_manufactured_fem_p3_improves_homogeneous_and_interface_cases() -> None:
    homogeneous_p2 = manufactured_fem_benchmark(
        degree=2, discontinuous_mass=False
    )
    homogeneous_p3 = manufactured_fem_benchmark(
        degree=3, discontinuous_mass=False
    )
    interface_p2 = manufactured_fem_benchmark(
        degree=2, discontinuous_mass=True
    )
    interface_p3 = manufactured_fem_benchmark(
        degree=3, discontinuous_mass=True
    )

    assert homogeneous_p3["weighted_relative_l2"] <= 2.0e-4
    assert (
        homogeneous_p3["weighted_relative_l2"]
        / homogeneous_p2["weighted_relative_l2"]
        <= 0.35
    )
    assert interface_p3["weighted_relative_l2"] <= 5.0e-3
    assert (
        interface_p3["weighted_relative_l2"]
        / interface_p2["weighted_relative_l2"]
        <= 0.75
    )
    for result in (homogeneous_p2, homogeneous_p3, interface_p2, interface_p3):
        assert result["solver_controls"]["relative_residual"] <= 1.0e-10
        assert result["all_finite"] is True
