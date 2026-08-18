"""Structured high-order weak-form FEM for axisymmetric scalar Helmholtz fields."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix
from scipy.special import jn_zeros, jv

from tgv_ptycho.forward.helmholtz_axisymmetric import solve_sparse_direct

CoefficientEvaluator: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
    ],
]


@dataclass(frozen=True)
class AxisymmetricFEMGrid:
    """Tensor-product continuous Lagrange grid with a natural axis boundary."""

    degree: int
    radial_element_size_m: float
    axial_element_size_m: float
    radial_extent_m: float
    z_min_m: float
    z_max_m: float
    radial_element_count: int
    axial_element_count: int
    reference_nodes: NDArray[np.float64]
    radial_nodes_m: NDArray[np.float64]
    z_nodes_m: NDArray[np.float64]
    active_index: NDArray[np.int64]

    @property
    def radial_node_count(self) -> int:
        return int(self.radial_nodes_m.size)

    @property
    def axial_node_count(self) -> int:
        return int(self.z_nodes_m.size)

    @property
    def active_unknown_count(self) -> int:
        return int(np.max(self.active_index) + 1)


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _aligned_count(length: float, spacing: float, name: str) -> int:
    ratio = float(length) / float(spacing)
    count = int(np.rint(ratio))
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    if count <= 0 or abs(ratio - count) > tolerance:
        raise ValueError(f"{name} must be an integer multiple of its spacing.")
    return count


def gauss_lobatto_nodes(degree: int) -> NDArray[np.float64]:
    """Return the Legendre--Gauss--Lobatto interpolation nodes."""

    value = int(degree)
    if value < 1 or value != degree:
        raise ValueError("degree must be a positive integer.")
    if value == 1:
        return np.asarray([-1.0, 1.0], dtype=np.float64)
    interior = np.sort(Legendre.basis(value).deriv().roots())
    return np.concatenate(([-1.0], interior, [1.0])).astype(np.float64)


def _assembled_axis_nodes(
    start: float,
    element_size: float,
    element_count: int,
    reference_nodes: NDArray[np.float64],
) -> NDArray[np.float64]:
    pieces: list[NDArray[np.float64]] = []
    for element in range(element_count):
        left = start + element * element_size
        local = left + 0.5 * element_size * (reference_nodes + 1.0)
        pieces.append(local if element == 0 else local[1:])
    result = np.concatenate(pieces).astype(np.float64)
    result[0] = start
    result[-1] = start + element_count * element_size
    return result


def make_axisymmetric_fem_grid(
    *,
    degree: int,
    radial_extent_m: float,
    z_min_m: float,
    z_max_m: float,
    radial_element_size_m: float,
    axial_element_size_m: float,
) -> AxisymmetricFEMGrid:
    """Construct a structured Qp grid.

    The radial axis is a natural zero-flux boundary.  The outer radial face
    and both axial faces are homogeneous Dirichlet truncation boundaries.
    """

    p = int(degree)
    reference = gauss_lobatto_nodes(p)
    radial_extent = _positive(radial_extent_m, "radial_extent_m")
    hr = _positive(radial_element_size_m, "radial_element_size_m")
    hz = _positive(axial_element_size_m, "axial_element_size_m")
    z_min = float(z_min_m)
    z_max = float(z_max_m)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        raise ValueError("z limits must be finite and increasing.")
    nr_elements = _aligned_count(radial_extent, hr, "radial extent")
    nz_elements = _aligned_count(z_max - z_min, hz, "axial extent")
    radial_nodes = _assembled_axis_nodes(0.0, hr, nr_elements, reference)
    z_nodes = _assembled_axis_nodes(z_min, hz, nz_elements, reference)
    active = np.full(
        (z_nodes.size, radial_nodes.size), -1, dtype=np.int64
    )
    radial_active = radial_nodes.size - 1
    axial_active = z_nodes.size - 2
    active[1:-1, :-1] = np.arange(
        axial_active * radial_active, dtype=np.int64
    ).reshape(axial_active, radial_active)
    return AxisymmetricFEMGrid(
        degree=p,
        radial_element_size_m=hr,
        axial_element_size_m=hz,
        radial_extent_m=radial_extent,
        z_min_m=z_min,
        z_max_m=z_max,
        radial_element_count=nr_elements,
        axial_element_count=nz_elements,
        reference_nodes=reference,
        radial_nodes_m=radial_nodes,
        z_nodes_m=z_nodes,
        active_index=active,
    )


def _lagrange_values_and_derivatives(
    nodes: NDArray[np.float64], evaluation: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    count = nodes.size
    values = np.empty((evaluation.size, count), dtype=np.float64)
    derivatives = np.empty_like(values)
    for index in range(count):
        polynomial = np.poly1d([1.0])
        denominator = 1.0
        for other in range(count):
            if other == index:
                continue
            polynomial *= np.poly1d([1.0, -nodes[other]])
            denominator *= nodes[index] - nodes[other]
        polynomial /= denominator
        values[:, index] = polynomial(evaluation)
        derivatives[:, index] = np.polyder(polynomial)(evaluation)
    return values, derivatives


def assemble_axisymmetric_weak_form(
    grid: AxisymmetricFEMGrid,
    coefficient_evaluator: CoefficientEvaluator,
    *,
    quadrature_order: int,
) -> tuple[csc_matrix, NDArray[np.complex128], dict[str, Any]]:
    """Assemble ``-grad(v) A grad(u) + v M u = v f`` on the Qp grid."""

    q = int(quadrature_order)
    if q < grid.degree + 1 or q != quadrature_order:
        raise ValueError("quadrature_order must be an integer >= degree + 1.")
    quadrature_nodes, quadrature_weights = leggauss(q)
    basis, derivative = _lagrange_values_and_derivatives(
        grid.reference_nodes, quadrature_nodes
    )
    p1 = grid.degree + 1
    local_count = p1 * p1
    phi_columns: list[NDArray[np.float64]] = []
    dr_columns: list[NDArray[np.float64]] = []
    dz_columns: list[NDArray[np.float64]] = []
    for local_z in range(p1):
        for local_r in range(p1):
            phi_columns.append(
                np.outer(basis[:, local_z], basis[:, local_r]).reshape(-1)
            )
            dr_columns.append(
                (
                    np.outer(basis[:, local_z], derivative[:, local_r])
                    * (2.0 / grid.radial_element_size_m)
                ).reshape(-1)
            )
            dz_columns.append(
                (
                    np.outer(derivative[:, local_z], basis[:, local_r])
                    * (2.0 / grid.axial_element_size_m)
                ).reshape(-1)
            )
    phi = np.column_stack(phi_columns)
    derivative_r = np.column_stack(dr_columns)
    derivative_z = np.column_stack(dz_columns)
    physical_weights = (
        np.outer(quadrature_weights, quadrature_weights).reshape(-1)
        * grid.radial_element_size_m
        * grid.axial_element_size_m
        / 4.0
    )

    element_count = grid.radial_element_count * grid.axial_element_count
    maximum_entries = element_count * local_count * local_count
    rows = np.empty(maximum_entries, dtype=np.int32)
    columns = np.empty(maximum_entries, dtype=np.int32)
    entries = np.empty(maximum_entries, dtype=np.complex128)
    rhs = np.zeros(grid.active_unknown_count, dtype=np.complex128)
    position = 0
    all_coefficients_finite = True
    started = time.perf_counter()
    for element_z in range(grid.axial_element_count):
        z_left = grid.z_min_m + element_z * grid.axial_element_size_m
        z_q = z_left + 0.5 * grid.axial_element_size_m * (
            quadrature_nodes + 1.0
        )
        z_ids = element_z * grid.degree + np.arange(p1)
        for element_r in range(grid.radial_element_count):
            r_left = element_r * grid.radial_element_size_m
            r_q = r_left + 0.5 * grid.radial_element_size_m * (
                quadrature_nodes + 1.0
            )
            r_ids = element_r * grid.degree + np.arange(p1)
            r_mesh, z_mesh = np.meshgrid(r_q, z_q)
            ar, az, mass, source = coefficient_evaluator(
                r_mesh.reshape(-1), z_mesh.reshape(-1)
            )
            coefficients = tuple(
                np.asarray(value, dtype=np.complex128).reshape(-1)
                for value in (ar, az, mass, source)
            )
            if any(value.shape != (q * q,) for value in coefficients):
                raise ValueError("coefficient evaluator returned the wrong shape.")
            if not all(np.all(np.isfinite(value)) for value in coefficients):
                all_coefficients_finite = False
                raise ValueError("coefficient evaluator returned non-finite values.")
            ar_value, az_value, mass_value, source_value = coefficients
            local_matrix = -derivative_r.T @ (
                (physical_weights * ar_value)[:, None] * derivative_r
            )
            local_matrix -= derivative_z.T @ (
                (physical_weights * az_value)[:, None] * derivative_z
            )
            local_matrix += phi.T @ (
                (physical_weights * mass_value)[:, None] * phi
            )
            local_rhs = phi.T @ (physical_weights * source_value)
            active_ids = grid.active_index[np.ix_(z_ids, r_ids)].reshape(-1)
            keep = active_ids >= 0
            retained_ids = active_ids[keep].astype(np.int32, copy=False)
            retained_count = int(retained_ids.size)
            retained_matrix = local_matrix[np.ix_(keep, keep)]
            entry_count = retained_count * retained_count
            rows[position : position + entry_count] = np.repeat(
                retained_ids, retained_count
            )
            columns[position : position + entry_count] = np.tile(
                retained_ids, retained_count
            )
            entries[position : position + entry_count] = retained_matrix.reshape(-1)
            position += entry_count
            np.add.at(rhs, retained_ids, local_rhs[keep])

    matrix = coo_matrix(
        (entries[:position], (rows[:position], columns[:position])),
        shape=(grid.active_unknown_count, grid.active_unknown_count),
        dtype=np.complex128,
    ).tocsc()
    matrix.sum_duplicates()
    symmetry = matrix - matrix.transpose()
    symmetry_error = (
        0.0
        if symmetry.nnz == 0
        else float(np.max(np.abs(symmetry.data)))
    )
    controls = {
        "degree": grid.degree,
        "quadrature_order": q,
        "radial_element_count": grid.radial_element_count,
        "axial_element_count": grid.axial_element_count,
        "radial_node_count": grid.radial_node_count,
        "axial_node_count": grid.axial_node_count,
        "active_unknown_count": grid.active_unknown_count,
        "assembled_triplet_count": int(position),
        "matrix_nnz": int(matrix.nnz),
        "complex_symmetric_max_abs_error": symmetry_error,
        "rhs_l2": float(np.linalg.norm(rhs)),
        "assembly_elapsed_s": float(time.perf_counter() - started),
        "coefficient_all_finite": all_coefficients_finite,
        "matrix_rhs_all_finite": bool(
            np.all(np.isfinite(matrix.data)) and np.all(np.isfinite(rhs))
        ),
        "axis_boundary": "natural_zero_flux",
        "outer_boundaries": "homogeneous_dirichlet",
    }
    return matrix, rhs, controls


def _stretch_right(
    coordinate: NDArray[np.float64],
    *,
    start: float,
    length: float,
    peak: float,
    order: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    distance = np.maximum(coordinate - start, 0.0)
    alpha = peak * (distance / length) ** order
    integral = peak * distance ** (order + 1) / (
        (order + 1) * length**order
    )
    return 1.0 + 1j * alpha, coordinate + 1j * integral


def _stretch_two_sided(
    coordinate: NDArray[np.float64],
    *,
    core_min: float,
    core_max: float,
    length: float,
    lower_peak: float,
    upper_peak: float,
    order: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    lower = np.maximum(core_min - coordinate, 0.0)
    upper = np.maximum(coordinate - core_max, 0.0)
    alpha = lower_peak * (lower / length) ** order + upper_peak * (
        upper / length
    ) ** order
    integral = (
        upper_peak * upper ** (order + 1)
        - lower_peak * lower ** (order + 1)
    ) / ((order + 1) * length**order)
    return 1.0 + 1j * alpha, coordinate + 1j * integral


def _pml_peak(
    target: float, wavenumber: float, length: float, order: int
) -> float:
    return float(-(order + 1) * np.log(target) / (wavenumber * length))


def _diameter_at_z(
    z: NDArray[np.float64],
    *,
    thickness_m: float,
    d_top_m: float,
    d_waist_m: float,
    d_bottom_m: float,
    z_waist_m: float,
) -> NDArray[np.float64]:
    result = np.empty_like(z)
    before = z <= z_waist_m
    result[before] = d_top_m + (d_waist_m - d_top_m) * (
        z[before] / z_waist_m
    )
    result[~before] = d_waist_m + (d_bottom_m - d_waist_m) * (
        (z[~before] - z_waist_m) / (thickness_m - z_waist_m)
    )
    return result


def make_tgv_scattered_fem_evaluator(
    *,
    wavelength_m: float,
    n_glass: float,
    n_air: float,
    incident_amplitude: float,
    background_interface_z_m: float,
    sample_thickness_m: float,
    d_top_m: float,
    d_waist_m: float,
    d_bottom_m: float,
    z_waist_m: float,
    radial_core_max_m: float,
    z_core_min_m: float,
    z_core_max_m: float,
    pml_thickness_m: float,
    pml_polynomial_order: int,
    pml_target_one_way_amplitude: float,
) -> CoefficientEvaluator:
    """Build pointwise PML/material coefficients for the scattered field."""

    wavelength = _positive(wavelength_m, "wavelength_m")
    glass = _positive(n_glass, "n_glass")
    air = _positive(n_air, "n_air")
    pml_length = _positive(pml_thickness_m, "pml_thickness_m")
    radial_core = _positive(radial_core_max_m, "radial_core_max_m")
    order = int(pml_polynomial_order)
    target = float(pml_target_one_way_amplitude)
    if order < 1 or order != pml_polynomial_order:
        raise ValueError("pml_polynomial_order must be a positive integer.")
    if not 0.0 < target < 1.0:
        raise ValueError("pml target must lie in (0, 1).")
    k0 = 2.0 * np.pi / wavelength
    radial_peak = _pml_peak(target, k0 * air, pml_length, order)
    lower_peak = _pml_peak(target, k0 * glass, pml_length, order)
    upper_peak = radial_peak
    reflection = (glass - air) / (glass + air)
    transmission = 2.0 * glass / (glass + air)
    interface = float(background_interface_z_m)
    thickness = float(sample_thickness_m)

    def evaluate(
        r_m: NDArray[np.float64], z_m: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
    ]:
        r = np.asarray(r_m, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        sr, r_tilde = _stretch_right(
            r,
            start=radial_core,
            length=pml_length,
            peak=radial_peak,
            order=order,
        )
        sz, z_tilde = _stretch_two_sided(
            z,
            core_min=float(z_core_min_m),
            core_max=float(z_core_max_m),
            length=pml_length,
            lower_peak=lower_peak,
            upper_peak=upper_peak,
            order=order,
        )
        background_n2 = np.where(z < interface, glass**2, air**2)
        tgv_n2 = background_n2.copy()
        inside_axial = (z >= 0.0) & (z < thickness)
        if np.any(inside_axial):
            diameter = _diameter_at_z(
                z[inside_axial],
                thickness_m=thickness,
                d_top_m=float(d_top_m),
                d_waist_m=float(d_waist_m),
                d_bottom_m=float(d_bottom_m),
                z_waist_m=float(z_waist_m),
            )
            inside_hole = r[inside_axial] < 0.5 * diameter
            selected = np.flatnonzero(inside_axial)
            tgv_n2[selected[inside_hole]] = air**2
        kg = k0 * glass
        ka = k0 * air
        phase_interface = np.exp(1j * kg * interface)
        below = z < interface
        background_field = np.empty(z.shape, dtype=np.complex128)
        background_field[below] = float(incident_amplitude) * (
            np.exp(1j * kg * z_tilde[below])
            + reflection
            * np.exp(1j * kg * (2.0 * interface - z_tilde[below]))
        )
        background_field[~below] = (
            float(incident_amplitude)
            * transmission
            * phase_interface
            * np.exp(1j * ka * (z_tilde[~below] - interface))
        )
        jacobian = r_tilde * sr * sz
        ar = sz * r_tilde / sr
        az = r_tilde * sr / sz
        mass = k0**2 * tgv_n2 * jacobian
        source = -k0**2 * (tgv_n2 - background_n2) * jacobian * background_field
        return (
            np.asarray(ar, dtype=np.complex128),
            np.asarray(az, dtype=np.complex128),
            np.asarray(mass, dtype=np.complex128),
            np.asarray(source, dtype=np.complex128),
        )

    return evaluate


def expand_active_solution(
    values: NDArray[np.complexfloating], grid: AxisymmetricFEMGrid
) -> NDArray[np.complex128]:
    """Expand active FEM unknowns to the full nodal array."""

    active = np.asarray(values, dtype=np.complex128)
    if active.shape != (grid.active_unknown_count,):
        raise ValueError("values must match the active FEM unknown count.")
    result = np.zeros(
        (grid.axial_node_count, grid.radial_node_count), dtype=np.complex128
    )
    mask = grid.active_index >= 0
    result[mask] = active[grid.active_index[mask]]
    return result


def evaluate_fem_field(
    values: NDArray[np.complexfloating],
    grid: AxisymmetricFEMGrid,
    r_m: NDArray[np.floating],
    z_m: NDArray[np.floating] | float,
) -> NDArray[np.complex128]:
    """Interpolate a FEM solution at matching arrays of physical points."""

    nodal = expand_active_solution(values, grid)
    r_values, z_values = np.broadcast_arrays(
        np.asarray(r_m, dtype=np.float64), np.asarray(z_m, dtype=np.float64)
    )
    if (
        np.any(r_values < 0.0)
        or np.any(r_values > grid.radial_extent_m)
        or np.any(z_values < grid.z_min_m)
        or np.any(z_values > grid.z_max_m)
    ):
        raise ValueError("evaluation points must lie inside the FEM domain.")
    flat_r = r_values.reshape(-1)
    flat_z = z_values.reshape(-1)
    element_r = np.minimum(
        (flat_r / grid.radial_element_size_m).astype(np.int64),
        grid.radial_element_count - 1,
    )
    element_z = np.minimum(
        ((flat_z - grid.z_min_m) / grid.axial_element_size_m).astype(np.int64),
        grid.axial_element_count - 1,
    )
    xi_r = 2.0 * (
        flat_r - element_r * grid.radial_element_size_m
    ) / grid.radial_element_size_m - 1.0
    xi_z = 2.0 * (
        flat_z
        - (grid.z_min_m + element_z * grid.axial_element_size_m)
    ) / grid.axial_element_size_m - 1.0
    basis_r, _ = _lagrange_values_and_derivatives(grid.reference_nodes, xi_r)
    basis_z, _ = _lagrange_values_and_derivatives(grid.reference_nodes, xi_z)
    result = np.empty(flat_r.size, dtype=np.complex128)
    p1 = grid.degree + 1
    local_offsets = np.arange(p1)
    for point in range(flat_r.size):
        r_ids = element_r[point] * grid.degree + local_offsets
        z_ids = element_z[point] * grid.degree + local_offsets
        local = nodal[np.ix_(z_ids, r_ids)]
        result[point] = basis_z[point] @ local @ basis_r[point]
    return result.reshape(r_values.shape)


def manufactured_fem_benchmark(
    *,
    degree: int,
    discontinuous_mass: bool,
    element_size: float = 0.125,
    quadrature_order: int = 12,
) -> dict[str, Any]:
    """Solve a regular Bessel/sine manufactured axisymmetric problem."""

    grid = make_axisymmetric_fem_grid(
        degree=degree,
        radial_extent_m=1.0,
        z_min_m=0.0,
        z_max_m=1.0,
        radial_element_size_m=element_size,
        axial_element_size_m=element_size,
    )
    alpha = float(jn_zeros(0, 1)[0])
    k_value = 10.0
    eigenvalue = alpha**2 + np.pi**2

    def exact(r: NDArray[np.float64], z: NDArray[np.float64]):
        return jv(0, alpha * r) * np.sin(np.pi * z) * (1.0 + 0.2j)

    def evaluator(r: NDArray[np.float64], z: NDArray[np.float64]):
        n2 = (
            np.where(r < 0.43, 1.0, 2.25)
            if discontinuous_mass
            else np.ones_like(r)
        )
        field = exact(r, z)
        radial = r.astype(np.complex128)
        return (
            radial,
            radial,
            np.asarray(k_value**2 * n2 * r, dtype=np.complex128),
            np.asarray((k_value**2 * n2 - eigenvalue) * r * field, dtype=np.complex128),
        )

    matrix, rhs, assembly = assemble_axisymmetric_weak_form(
        grid, evaluator, quadrature_order=quadrature_order
    )
    solution, solver = solve_sparse_direct(matrix, rhs)
    radial = np.linspace(0.0, 1.0, 97, dtype=np.float64)
    axial = np.linspace(0.0, 1.0, 97, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(radial, axial)
    numerical = evaluate_fem_field(solution, grid, r_mesh, z_mesh)
    truth = exact(r_mesh, z_mesh)
    weight = r_mesh
    numerator = float(np.sum(weight * np.abs(numerical - truth) ** 2))
    denominator = float(np.sum(weight * np.abs(truth) ** 2))
    relative_error = float(
        np.sqrt(numerator / max(denominator, np.finfo(float).eps))
    )
    return {
        "degree": int(degree),
        "discontinuous_mass": bool(discontinuous_mass),
        "element_size": float(element_size),
        "quadrature_order": int(quadrature_order),
        "weighted_relative_l2": relative_error,
        "assembly_controls": assembly,
        "solver_controls": solver,
        "all_finite": bool(
            np.all(np.isfinite(numerical))
            and np.all(np.isfinite(truth))
            and np.isfinite(relative_error)
        ),
    }


__all__ = [
    "AxisymmetricFEMGrid",
    "assemble_axisymmetric_weak_form",
    "evaluate_fem_field",
    "expand_active_solution",
    "gauss_lobatto_nodes",
    "make_axisymmetric_fem_grid",
    "make_tgv_scattered_fem_evaluator",
    "manufactured_fem_benchmark",
]
