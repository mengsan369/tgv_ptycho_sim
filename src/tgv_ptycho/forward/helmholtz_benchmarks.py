"""Analytic Helmholtz benchmarks used by exp040 numerical-reference audits."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.special import hankel1, hankel2, jn_zeros, jv

from tgv_ptycho.forward.helmholtz_axisymmetric import solve_sparse_direct
from tgv_ptycho.forward.helmholtz_axisymmetric_fem import (
    AxisymmetricFEMGrid,
    assemble_axisymmetric_weak_form,
    evaluate_fem_field,
    expand_active_solution,
    gauss_lobatto_nodes,
    make_axisymmetric_fem_grid,
)


@dataclass(frozen=True)
class AnnularRadialFEMGrid:
    """Continuous one-dimensional Qp grid on a radial annulus."""

    degree: int
    inner_radius_m: float
    outer_radius_m: float
    element_size_m: float
    element_count: int
    reference_nodes: NDArray[np.float64]
    radial_nodes_m: NDArray[np.float64]

    @property
    def active_unknown_count(self) -> int:
        """Return the number of nodes not fixed by Dirichlet data."""

        return int(self.radial_nodes_m.size - 2)


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


def _lagrange_values_and_derivatives(
    nodes: NDArray[np.float64], evaluation: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.empty((evaluation.size, nodes.size), dtype=np.float64)
    derivatives = np.empty_like(values)
    for index in range(nodes.size):
        polynomial = np.poly1d([1.0])
        denominator = 1.0
        for other in range(nodes.size):
            if other == index:
                continue
            polynomial *= np.poly1d([1.0, -nodes[other]])
            denominator *= nodes[index] - nodes[other]
        polynomial /= denominator
        values[:, index] = polynomial(evaluation)
        derivatives[:, index] = np.polyder(polynomial)(evaluation)
    return values, derivatives


def make_annular_radial_fem_grid(
    *,
    degree: int,
    inner_radius_m: float,
    outer_radius_m: float,
    element_size_m: float,
) -> AnnularRadialFEMGrid:
    """Build an annular Legendre--Gauss--Lobatto FEM grid."""

    p = int(degree)
    if p < 1 or p != degree:
        raise ValueError("degree must be a positive integer.")
    inner = _positive(inner_radius_m, "inner_radius_m")
    outer = _positive(outer_radius_m, "outer_radius_m")
    spacing = _positive(element_size_m, "element_size_m")
    if outer <= inner:
        raise ValueError("outer_radius_m must exceed inner_radius_m.")
    element_count = _aligned_count(outer - inner, spacing, "annular extent")
    reference = gauss_lobatto_nodes(p)
    pieces: list[NDArray[np.float64]] = []
    for element in range(element_count):
        left = inner + element * spacing
        local = left + 0.5 * spacing * (reference + 1.0)
        pieces.append(local if element == 0 else local[1:])
    nodes = np.concatenate(pieces).astype(np.float64)
    nodes[0] = inner
    nodes[-1] = outer
    return AnnularRadialFEMGrid(
        degree=p,
        inner_radius_m=inner,
        outer_radius_m=outer,
        element_size_m=spacing,
        element_count=element_count,
        reference_nodes=reference,
        radial_nodes_m=nodes,
    )


def _pml_peak(
    *, target_one_way_amplitude: float, wavenumber: float,
    length: float, polynomial_order: int,
) -> float:
    target = float(target_one_way_amplitude)
    if not 0.0 < target < 1.0:
        raise ValueError("target_one_way_amplitude must lie in (0, 1).")
    order = int(polynomial_order)
    if order < 1 or order != polynomial_order:
        raise ValueError("polynomial_order must be a positive integer.")
    return float(
        -(order + 1) * np.log(target)
        / (_positive(wavenumber, "wavenumber") * _positive(length, "length"))
    )


def _radial_pml_coordinates(
    radius: NDArray[np.float64], *, pml_start: float, pml_length: float,
    peak: float, polynomial_order: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    distance = np.maximum(radius - pml_start, 0.0)
    alpha = peak * (distance / pml_length) ** polynomial_order
    integral = (
        peak
        * distance ** (polynomial_order + 1)
        / ((polynomial_order + 1) * pml_length**polynomial_order)
    )
    return (
        np.asarray(1.0 + 1j * alpha, dtype=np.complex128),
        np.asarray(radius + 1j * integral, dtype=np.complex128),
    )


def _assemble_annular_outgoing_problem(
    grid: AnnularRadialFEMGrid,
    *,
    wavenumber: float,
    pml_start_m: float,
    pml_thickness_m: float,
    pml_polynomial_order: int,
    pml_target_one_way_amplitude: float,
    quadrature_order: int,
) -> tuple[Any, NDArray[np.complex128], dict[str, Any]]:
    q = int(quadrature_order)
    if q < grid.degree + 1 or q != quadrature_order:
        raise ValueError("quadrature_order must be an integer >= degree + 1.")
    pml_start = float(pml_start_m)
    pml_length = _positive(pml_thickness_m, "pml_thickness_m")
    if not grid.inner_radius_m < pml_start < grid.outer_radius_m:
        raise ValueError("pml_start_m must lie in the annular domain.")
    if not np.isclose(pml_start + pml_length, grid.outer_radius_m):
        raise ValueError("PML must end at the annular outer boundary.")
    peak = _pml_peak(
        target_one_way_amplitude=pml_target_one_way_amplitude,
        wavenumber=wavenumber,
        length=pml_length,
        polynomial_order=pml_polynomial_order,
    )
    q_nodes, q_weights = leggauss(q)
    basis, derivative_reference = _lagrange_values_and_derivatives(
        grid.reference_nodes, q_nodes
    )
    derivative = derivative_reference * (2.0 / grid.element_size_m)
    weights = q_weights * grid.element_size_m / 2.0
    p1 = grid.degree + 1
    local_entries = p1 * p1
    maximum_entries = grid.element_count * local_entries
    rows = np.empty(maximum_entries, dtype=np.int32)
    columns = np.empty(maximum_entries, dtype=np.int32)
    entries = np.empty(maximum_entries, dtype=np.complex128)
    rhs = np.zeros(grid.active_unknown_count, dtype=np.complex128)
    boundary_values = np.zeros(grid.radial_nodes_m.size, dtype=np.complex128)
    boundary_values[0] = 1.0 + 0.0j
    active_index = np.full(grid.radial_nodes_m.size, -1, dtype=np.int64)
    active_index[1:-1] = np.arange(grid.active_unknown_count)
    position = 0
    started = time.perf_counter()
    for element in range(grid.element_count):
        left = grid.inner_radius_m + element * grid.element_size_m
        radius_q = left + 0.5 * grid.element_size_m * (q_nodes + 1.0)
        stretch, radius_tilde = _radial_pml_coordinates(
            radius_q,
            pml_start=pml_start,
            pml_length=pml_length,
            peak=peak,
            polynomial_order=pml_polynomial_order,
        )
        radial_stiffness = radius_tilde / stretch
        radial_mass = wavenumber**2 * radius_tilde * stretch
        local_matrix = -derivative.T @ (
            (weights * radial_stiffness)[:, None] * derivative
        )
        local_matrix += basis.T @ ((weights * radial_mass)[:, None] * basis)
        global_nodes = element * grid.degree + np.arange(p1)
        local_active = active_index[global_nodes]
        keep = local_active >= 0
        fixed = ~keep
        retained = local_active[keep].astype(np.int32, copy=False)
        if np.any(fixed):
            rhs[retained] -= (
                local_matrix[np.ix_(keep, fixed)]
                @ boundary_values[global_nodes[fixed]]
            )
        retained_matrix = local_matrix[np.ix_(keep, keep)]
        count = retained.size
        entry_count = count * count
        rows[position : position + entry_count] = np.repeat(retained, count)
        columns[position : position + entry_count] = np.tile(retained, count)
        entries[position : position + entry_count] = retained_matrix.reshape(-1)
        position += entry_count
    matrix = coo_matrix(
        (entries[:position], (rows[:position], columns[:position])),
        shape=(grid.active_unknown_count, grid.active_unknown_count),
        dtype=np.complex128,
    ).tocsc()
    matrix.sum_duplicates()
    symmetry = matrix - matrix.transpose()
    symmetry_error = (
        0.0 if symmetry.nnz == 0 else float(np.max(np.abs(symmetry.data)))
    )
    controls = {
        "active_unknown_count": grid.active_unknown_count,
        "element_count": grid.element_count,
        "matrix_nnz": int(matrix.nnz),
        "complex_symmetric_max_abs_error": symmetry_error,
        "pml_peak_alpha": peak,
        "assembly_elapsed_s": float(time.perf_counter() - started),
        "all_finite": bool(
            np.all(np.isfinite(matrix.data)) and np.all(np.isfinite(rhs))
        ),
    }
    return matrix, rhs, controls


def evaluate_annular_fem_field_and_derivative(
    nodal_values: NDArray[np.complexfloating],
    grid: AnnularRadialFEMGrid,
    radius_m: NDArray[np.floating],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate an annular FEM field and its physical radial derivative."""

    nodal = np.asarray(nodal_values, dtype=np.complex128)
    if nodal.shape != grid.radial_nodes_m.shape:
        raise ValueError("nodal_values must match the annular FEM nodes.")
    radius = np.asarray(radius_m, dtype=np.float64)
    if np.any(radius < grid.inner_radius_m) or np.any(
        radius > grid.outer_radius_m
    ):
        raise ValueError("evaluation radii must lie inside the annulus.")
    flat = radius.reshape(-1)
    element = np.minimum(
        ((flat - grid.inner_radius_m) / grid.element_size_m).astype(np.int64),
        grid.element_count - 1,
    )
    xi = 2.0 * (
        flat - (grid.inner_radius_m + element * grid.element_size_m)
    ) / grid.element_size_m - 1.0
    basis, derivative = _lagrange_values_and_derivatives(
        grid.reference_nodes, xi
    )
    derivative *= 2.0 / grid.element_size_m
    values = np.empty(flat.size, dtype=np.complex128)
    gradients = np.empty_like(values)
    offsets = np.arange(grid.degree + 1)
    for point in range(flat.size):
        local = nodal[element[point] * grid.degree + offsets]
        values[point] = basis[point] @ local
        gradients[point] = derivative[point] @ local
    return values.reshape(radius.shape), gradients.reshape(radius.shape)


def normalized_cylindrical_bases(
    radius_m: NDArray[np.floating], *, wavenumber_per_m: float,
    normalization_radius_m: float,
) -> tuple[
    NDArray[np.complex128], NDArray[np.complex128],
    NDArray[np.complex128], NDArray[np.complex128],
]:
    """Return normalized outgoing/incoming Hankel fields and derivatives."""

    radius = np.asarray(radius_m, dtype=np.float64)
    k_value = _positive(wavenumber_per_m, "wavenumber_per_m")
    inner = _positive(normalization_radius_m, "normalization_radius_m")
    outgoing_scale = hankel1(0, k_value * inner)
    incoming_scale = hankel2(0, k_value * inner)
    outgoing = hankel1(0, k_value * radius) / outgoing_scale
    incoming = hankel2(0, k_value * radius) / incoming_scale
    outgoing_derivative = -k_value * hankel1(1, k_value * radius) / outgoing_scale
    incoming_derivative = -k_value * hankel2(1, k_value * radius) / incoming_scale
    return tuple(
        np.asarray(value, dtype=np.complex128)
        for value in (
            outgoing,
            incoming,
            outgoing_derivative,
            incoming_derivative,
        )
    )


def decompose_cylindrical_field(
    field: NDArray[np.complexfloating],
    derivative: NDArray[np.complexfloating],
    radius_m: NDArray[np.floating],
    *,
    wavenumber_per_m: float,
    normalization_radius_m: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Decompose local field/derivative data into H1 outgoing and H2 incoming."""

    values, gradients, radius = np.broadcast_arrays(
        np.asarray(field, dtype=np.complex128),
        np.asarray(derivative, dtype=np.complex128),
        np.asarray(radius_m, dtype=np.float64),
    )
    outgoing, incoming, outgoing_d, incoming_d = normalized_cylindrical_bases(
        radius,
        wavenumber_per_m=wavenumber_per_m,
        normalization_radius_m=normalization_radius_m,
    )
    determinant = outgoing * incoming_d - incoming * outgoing_d
    coefficient_outgoing = (
        values * incoming_d - incoming * gradients
    ) / determinant
    coefficient_incoming = (
        outgoing * gradients - values * outgoing_d
    ) / determinant
    return coefficient_outgoing, coefficient_incoming


def annular_outgoing_pml_benchmark(
    *,
    wavelength_m: float,
    refractive_index: float,
    inner_radius_m: float,
    pml_start_m: float,
    pml_thickness_m: float,
    element_size_m: float,
    degree: int,
    quadrature_order: int,
    pml_polynomial_order: int,
    pml_target_one_way_amplitude: float,
    measurement_radii_m: NDArray[np.floating],
    dense_radii_m: NDArray[np.floating],
) -> dict[str, Any]:
    """Solve and audit a pure outgoing cylindrical wave truncated by a PML."""

    k_value = (
        2.0
        * np.pi
        * _positive(refractive_index, "refractive_index")
        / _positive(wavelength_m, "wavelength_m")
    )
    outer = float(pml_start_m) + float(pml_thickness_m)
    grid = make_annular_radial_fem_grid(
        degree=degree,
        inner_radius_m=inner_radius_m,
        outer_radius_m=outer,
        element_size_m=element_size_m,
    )
    matrix, rhs, assembly = _assemble_annular_outgoing_problem(
        grid,
        wavenumber=k_value,
        pml_start_m=pml_start_m,
        pml_thickness_m=pml_thickness_m,
        pml_polynomial_order=pml_polynomial_order,
        pml_target_one_way_amplitude=pml_target_one_way_amplitude,
        quadrature_order=quadrature_order,
    )
    solution, solver = solve_sparse_direct(matrix, rhs)
    nodal = np.zeros(grid.radial_nodes_m.size, dtype=np.complex128)
    nodal[0] = 1.0 + 0.0j
    nodal[1:-1] = solution
    measurement = np.asarray(measurement_radii_m, dtype=np.float64)
    dense = np.asarray(dense_radii_m, dtype=np.float64)
    if np.any(measurement >= pml_start_m) or np.any(dense >= pml_start_m):
        raise ValueError("all diagnostics must be sampled before the PML.")
    measured_field, measured_derivative = evaluate_annular_fem_field_and_derivative(
        nodal, grid, measurement
    )
    dense_field, dense_derivative = evaluate_annular_fem_field_and_derivative(
        nodal, grid, dense
    )
    outgoing_coefficient, incoming_coefficient = decompose_cylindrical_field(
        measured_field,
        measured_derivative,
        measurement,
        wavenumber_per_m=k_value,
        normalization_radius_m=inner_radius_m,
    )
    truth, _, truth_derivative, _ = normalized_cylindrical_bases(
        dense,
        wavenumber_per_m=k_value,
        normalization_radius_m=inner_radius_m,
    )
    measured_outgoing, _, measured_outgoing_d, _ = normalized_cylindrical_bases(
        measurement,
        wavenumber_per_m=k_value,
        normalization_radius_m=inner_radius_m,
    )
    impedance = measured_outgoing_d / measured_outgoing
    impedance_residual = np.abs(
        measured_derivative - impedance * measured_field
    ) / np.maximum(k_value * np.abs(measured_field), np.finfo(float).eps)
    incoming_ratio = np.abs(incoming_coefficient) / np.maximum(
        np.abs(outgoing_coefficient), np.finfo(float).eps
    )
    flux = measurement * np.imag(
        np.conj(measured_field) * measured_derivative
    )
    flux_relative_range = float(
        np.ptp(flux) / max(abs(float(np.mean(flux))), np.finfo(float).eps)
    )
    numerator = float(
        np.trapezoid(dense * np.abs(dense_field - truth) ** 2, dense)
    )
    denominator = float(np.trapezoid(dense * np.abs(truth) ** 2, dense))
    field_relative_l2 = float(
        np.sqrt(numerator / max(denominator, np.finfo(float).eps))
    )
    all_arrays = (
        nodal,
        measured_field,
        measured_derivative,
        dense_field,
        dense_derivative,
        truth,
        truth_derivative,
        outgoing_coefficient,
        incoming_coefficient,
        incoming_ratio,
        impedance_residual,
        flux,
    )
    return {
        "wavenumber_per_m": k_value,
        "kh": float(k_value * grid.element_size_m),
        "degree": grid.degree,
        "element_size_m": grid.element_size_m,
        "pml_start_m": float(pml_start_m),
        "pml_thickness_m": float(pml_thickness_m),
        "measurement_radii_m": measurement,
        "dense_radii_m": dense,
        "nodal_radii_m": grid.radial_nodes_m,
        "nodal_field": nodal,
        "measurement_field": measured_field,
        "measurement_derivative": measured_derivative,
        "dense_field": dense_field,
        "dense_derivative": dense_derivative,
        "dense_truth": truth,
        "incoming_to_outgoing_ratio": incoming_ratio,
        "outgoing_impedance_residual": impedance_residual,
        "radial_flux": flux,
        "maximum_incoming_to_outgoing_ratio": float(np.max(incoming_ratio)),
        "maximum_outgoing_impedance_residual": float(
            np.max(impedance_residual)
        ),
        "flux_relative_range": flux_relative_range,
        "dense_field_weighted_relative_l2": field_relative_l2,
        "assembly_controls": assembly,
        "solver_controls": solver,
        "all_finite": bool(all(np.all(np.isfinite(value)) for value in all_arrays)),
    }


def physical_k_modal_fem_benchmark(
    *,
    degree: int,
    element_size_ratio: float,
    formal_kh: float,
    radial_extent: float,
    axial_extent: float,
    radial_mode: int,
    axial_mode: int,
    complex_amplitude: complex,
    discontinuous_mass: bool,
    interface_radius: float,
    homogeneous_n2: float,
    interface_inner_n2: float,
    interface_outer_n2: float,
    quadrature_order: int,
    evaluation_count_per_axis: int,
) -> dict[str, Any]:
    """Solve a high-frequency analytic modal problem at a prescribed ``kh``."""

    spacing = _positive(element_size_ratio, "element_size_ratio")
    k_value = _positive(formal_kh, "formal_kh")
    radial_size = _positive(radial_extent, "radial_extent")
    axial_size = _positive(axial_extent, "axial_extent")
    radial_index = int(radial_mode)
    axial_index = int(axial_mode)
    if radial_index < 1 or radial_index != radial_mode:
        raise ValueError("radial_mode must be a positive integer.")
    if axial_index < 1 or axial_index != axial_mode:
        raise ValueError("axial_mode must be a positive integer.")
    grid = make_axisymmetric_fem_grid(
        degree=degree,
        radial_extent_m=radial_size,
        z_min_m=0.0,
        z_max_m=axial_size,
        radial_element_size_m=spacing,
        axial_element_size_m=spacing,
    )
    alpha = float(jn_zeros(0, radial_index)[-1])
    radial_beta = alpha / radial_size
    axial_beta = axial_index * np.pi / axial_size
    eigenvalue = radial_beta**2 + axial_beta**2
    amplitude = complex(complex_amplitude)

    def exact(r: NDArray[np.float64], z: NDArray[np.float64]):
        return (
            amplitude
            * jv(0, radial_beta * r)
            * np.sin(axial_beta * z)
        )

    def evaluator(r: NDArray[np.float64], z: NDArray[np.float64]):
        if discontinuous_mass:
            n2 = np.where(
                r < float(interface_radius),
                float(interface_inner_n2),
                float(interface_outer_n2),
            )
        else:
            n2 = np.full_like(r, float(homogeneous_n2))
        field = exact(r, z)
        radial = r.astype(np.complex128)
        return (
            radial,
            radial,
            np.asarray(k_value**2 * n2 * r, dtype=np.complex128),
            np.asarray(
                (k_value**2 * n2 - eigenvalue) * r * field,
                dtype=np.complex128,
            ),
        )

    matrix, rhs, assembly = assemble_axisymmetric_weak_form(
        grid, evaluator, quadrature_order=quadrature_order
    )
    solution, solver = solve_sparse_direct(matrix, rhs)
    count = int(evaluation_count_per_axis)
    if count < 17 or count != evaluation_count_per_axis:
        raise ValueError("evaluation_count_per_axis must be an integer >= 17.")
    radial = np.linspace(0.0, radial_size, count, dtype=np.float64)
    axial = np.linspace(0.0, axial_size, count, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(radial, axial)
    numerical = evaluate_fem_field(solution, grid, r_mesh, z_mesh)
    truth = exact(r_mesh, z_mesh)
    squared_error = r_mesh * np.abs(numerical - truth) ** 2
    squared_truth = r_mesh * np.abs(truth) ** 2
    numerator = float(
        np.trapezoid(np.trapezoid(squared_error, radial, axis=1), axial)
    )
    denominator = float(
        np.trapezoid(np.trapezoid(squared_truth, radial, axis=1), axial)
    )
    relative_error = float(
        np.sqrt(numerator / max(denominator, np.finfo(float).eps))
    )
    return {
        "degree": int(degree),
        "element_size_ratio": spacing,
        "formal_kh": k_value,
        "actual_kh": float(k_value * spacing),
        "kh_over_p": float(k_value * spacing / int(degree)),
        "radial_mode": radial_index,
        "axial_mode": axial_index,
        "modal_wavenumber": float(np.sqrt(eigenvalue)),
        "modal_to_operator_wavenumber_ratio": float(
            np.sqrt(eigenvalue) / k_value
        ),
        "discontinuous_mass": bool(discontinuous_mass),
        "weighted_relative_l2": relative_error,
        "radial_coordinates": radial,
        "axial_coordinates": axial,
        "numerical_field": numerical,
        "truth_field": truth,
        "assembly_controls": assembly,
        "solver_controls": solver,
        "all_finite": bool(
            np.all(np.isfinite(numerical))
            and np.all(np.isfinite(truth))
            and np.isfinite(relative_error)
        ),
    }


def _right_complex_stretch(
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
    return (
        np.asarray(1.0 + 1j * alpha, dtype=np.complex128),
        np.asarray(coordinate + 1j * integral, dtype=np.complex128),
    )


def _two_sided_complex_stretch(
    coordinate: NDArray[np.float64],
    *,
    core_min: float,
    core_max: float,
    length: float,
    peak: float,
    order: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    lower = np.maximum(core_min - coordinate, 0.0)
    upper = np.maximum(coordinate - core_max, 0.0)
    alpha = peak * ((lower / length) ** order + (upper / length) ** order)
    integral = peak * (upper ** (order + 1) - lower ** (order + 1)) / (
        (order + 1) * length**order
    )
    return (
        np.asarray(1.0 + 1j * alpha, dtype=np.complex128),
        np.asarray(coordinate + 1j * integral, dtype=np.complex128),
    )


def make_axisymmetric_pml_modal_problem(
    *,
    degree: int,
    element_size: float,
    core_extent: float,
    pml_thickness: float,
    operator_wavenumber: float,
    pml_polynomial_order: int,
    pml_target_one_way_amplitude: float,
    interface_fraction_of_core_radius: float,
    interface_inner_n2: float,
    interface_outer_n2: float,
    target_radial_modal_wavenumber: float,
    target_axial_modal_wavenumber: float,
    modal_index_offset: int,
    quadrature_order: int,
    imaginary_mass_shift: float = 0.0,
) -> dict[str, Any]:
    """Assemble a PML Helmholtz problem with an analytic stretched mode."""

    spacing = _positive(element_size, "element_size")
    core = _positive(core_extent, "core_extent")
    pml = _positive(pml_thickness, "pml_thickness")
    k_value = _positive(operator_wavenumber, "operator_wavenumber")
    order = int(pml_polynomial_order)
    if order < 1 or order != pml_polynomial_order:
        raise ValueError("pml_polynomial_order must be a positive integer.")
    offset = int(modal_index_offset)
    if offset < 0 or offset != modal_index_offset:
        raise ValueError("modal_index_offset must be a non-negative integer.")
    interface_fraction = float(interface_fraction_of_core_radius)
    if not 0.0 < interface_fraction < 1.0:
        raise ValueError("interface fraction must lie in (0, 1).")
    inner_n2 = _positive(interface_inner_n2, "interface_inner_n2")
    outer_n2 = _positive(interface_outer_n2, "interface_outer_n2")
    shift = float(imaginary_mass_shift)
    if not np.isfinite(shift) or shift < 0.0:
        raise ValueError("imaginary_mass_shift must be finite and non-negative.")
    peak = _pml_peak(
        target_one_way_amplitude=pml_target_one_way_amplitude,
        wavenumber=k_value * np.sqrt(max(inner_n2, outer_n2)),
        length=pml,
        polynomial_order=order,
    )
    radial_extent = core + pml
    z_min = -pml
    z_max = core + pml
    grid = make_axisymmetric_fem_grid(
        degree=degree,
        radial_extent_m=radial_extent,
        z_min_m=z_min,
        z_max_m=z_max,
        radial_element_size_m=spacing,
        axial_element_size_m=spacing,
    )
    _, radial_outer_tilde_array = _right_complex_stretch(
        np.asarray([radial_extent]),
        start=core,
        length=pml,
        peak=peak,
        order=order,
    )
    _, z_boundary_tilde = _two_sided_complex_stretch(
        np.asarray([z_min, z_max]),
        core_min=0.0,
        core_max=core,
        length=pml,
        peak=peak,
        order=order,
    )
    radial_outer_tilde = complex(radial_outer_tilde_array[0])
    z_lower_tilde = complex(z_boundary_tilde[0])
    z_upper_tilde = complex(z_boundary_tilde[1])
    target_radial = _positive(
        target_radial_modal_wavenumber, "target_radial_modal_wavenumber"
    )
    target_axial = _positive(
        target_axial_modal_wavenumber, "target_axial_modal_wavenumber"
    )
    estimated_radial_index = max(
        1, int(np.rint(target_radial * radial_extent / np.pi + 0.25))
    )
    search_count = estimated_radial_index + 8
    zeros = jn_zeros(0, search_count)
    primary_radial_index = int(
        np.argmin(np.abs(zeros - target_radial * radial_extent)) + 1
    )
    radial_index = primary_radial_index + offset
    if radial_index > zeros.size:
        zeros = jn_zeros(0, radial_index)
    alpha = float(zeros[radial_index - 1])
    primary_axial_index = max(
        1, int(np.rint(target_axial * (core + 2.0 * pml) / np.pi))
    )
    axial_index = primary_axial_index + offset
    radial_beta = alpha / radial_outer_tilde
    axial_beta = axial_index * np.pi / (z_upper_tilde - z_lower_tilde)
    modal_eigenvalue = radial_beta**2 + axial_beta**2
    interface_radius = interface_fraction * core

    def exact_field(
        r_m: NDArray[np.float64], z_m: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        r = np.asarray(r_m, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        _, r_tilde = _right_complex_stretch(
            r,
            start=core,
            length=pml,
            peak=peak,
            order=order,
        )
        _, z_tilde = _two_sided_complex_stretch(
            z,
            core_min=0.0,
            core_max=core,
            length=pml,
            peak=peak,
            order=order,
        )
        return np.asarray(
            (1.0 + 0.2j)
            * jv(0, radial_beta * r_tilde)
            * np.sin(axial_beta * (z_tilde - z_lower_tilde)),
            dtype=np.complex128,
        )

    def evaluator(
        r_m: NDArray[np.float64], z_m: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
    ]:
        r = np.asarray(r_m, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        sr, r_tilde = _right_complex_stretch(
            r,
            start=core,
            length=pml,
            peak=peak,
            order=order,
        )
        sz, z_tilde = _two_sided_complex_stretch(
            z,
            core_min=0.0,
            core_max=core,
            length=pml,
            peak=peak,
            order=order,
        )
        inside_inclusion = (
            (r < interface_radius) & (z >= 0.0) & (z <= core)
        )
        n2 = np.where(inside_inclusion, inner_n2, outer_n2)
        jacobian = r_tilde * sr * sz
        field = np.asarray(
            (1.0 + 0.2j)
            * jv(0, radial_beta * r_tilde)
            * np.sin(axial_beta * (z_tilde - z_lower_tilde)),
            dtype=np.complex128,
        )
        shifted_k2_n2 = (1.0 + 1j * shift) * k_value**2 * n2
        return (
            np.asarray(sz * r_tilde / sr, dtype=np.complex128),
            np.asarray(r_tilde * sr / sz, dtype=np.complex128),
            np.asarray(shifted_k2_n2 * jacobian, dtype=np.complex128),
            np.asarray(
                (shifted_k2_n2 - modal_eigenvalue) * jacobian * field,
                dtype=np.complex128,
            ),
        )

    matrix, rhs, assembly = assemble_axisymmetric_weak_form(
        grid, evaluator, quadrature_order=quadrature_order
    )
    controls = {
        "degree": int(degree),
        "element_size": spacing,
        "core_extent": core,
        "pml_thickness": pml,
        "operator_wavenumber": k_value,
        "imaginary_mass_shift": shift,
        "pml_peak_alpha": peak,
        "interface_radius": interface_radius,
        "radial_mode_index": radial_index,
        "axial_mode_index": axial_index,
        "primary_radial_mode_index": primary_radial_index,
        "primary_axial_mode_index": primary_axial_index,
        "radial_beta": radial_beta,
        "axial_beta": axial_beta,
        "modal_eigenvalue": modal_eigenvalue,
        "physical_modal_wavenumber_magnitude": float(
            np.sqrt(abs(radial_beta) ** 2 + abs(axial_beta) ** 2)
        ),
        "active_shape": [grid.axial_node_count - 2, grid.radial_node_count - 1],
        "assembly_controls": assembly,
        "all_finite": bool(
            np.all(np.isfinite(matrix.data)) and np.all(np.isfinite(rhs))
        ),
    }
    return {
        "grid": grid,
        "matrix": matrix,
        "rhs": rhs,
        "exact_field": exact_field,
        "controls": controls,
    }


def axisymmetric_modal_nodal_error(
    solution: NDArray[np.complexfloating],
    grid: AxisymmetricFEMGrid,
    exact_field: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.complex128]
    ],
    *,
    core_extent: float,
) -> dict[str, Any]:
    """Evaluate a PML modal solution against truth on physical-core FEM nodes."""

    nodal = expand_active_solution(np.asarray(solution), grid)
    radial_mask = grid.radial_nodes_m <= float(core_extent)
    axial_mask = (grid.z_nodes_m >= 0.0) & (
        grid.z_nodes_m <= float(core_extent)
    )
    radial = grid.radial_nodes_m[radial_mask]
    axial = grid.z_nodes_m[axial_mask]
    numerical = nodal[np.ix_(axial_mask, radial_mask)]
    r_mesh, z_mesh = np.meshgrid(radial, axial)
    truth = exact_field(r_mesh, z_mesh)
    squared_error = r_mesh * np.abs(numerical - truth) ** 2
    squared_truth = r_mesh * np.abs(truth) ** 2
    numerator = float(
        np.trapezoid(np.trapezoid(squared_error, radial, axis=1), axial)
    )
    denominator = float(
        np.trapezoid(np.trapezoid(squared_truth, radial, axis=1), axial)
    )
    relative_l2 = float(
        np.sqrt(numerator / max(denominator, np.finfo(float).eps))
    )
    center_index = int(np.argmin(np.abs(axial - 0.5 * float(core_extent))))
    return {
        "weighted_relative_l2": relative_l2,
        "radial_coordinates": radial,
        "center_z": float(axial[center_index]),
        "center_numerical_trace": numerical[center_index],
        "center_truth_trace": truth[center_index],
        "all_finite": bool(
            np.all(np.isfinite(numerical))
            and np.all(np.isfinite(truth))
            and np.isfinite(relative_l2)
        ),
    }


def axial_plane_wave_pml_benchmark(
    *,
    wavelength_m: float,
    refractive_index: float,
    direction: int,
    physical_core_length_m: float,
    pml_thickness_m: float,
    element_size_m: float,
    degree: int,
    quadrature_order: int,
    pml_polynomial_order: int,
    pml_target_one_way_amplitude: float,
    measurement_fractions: NDArray[np.floating],
    dense_comparison_count: int,
) -> dict[str, Any]:
    """Verify an upper or lower one-dimensional plane-wave PML."""

    propagation_direction = int(direction)
    if propagation_direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1.")
    core = _positive(physical_core_length_m, "physical_core_length_m")
    pml = _positive(pml_thickness_m, "pml_thickness_m")
    spacing = _positive(element_size_m, "element_size_m")
    k_value = (
        2.0 * np.pi * _positive(refractive_index, "refractive_index")
        / _positive(wavelength_m, "wavelength_m")
    )
    outer = core + pml
    grid = make_annular_radial_fem_grid(
        degree=degree,
        inner_radius_m=spacing,
        outer_radius_m=outer + spacing,
        element_size_m=spacing,
    )
    q = int(quadrature_order)
    q_nodes, q_weights = leggauss(q)
    basis, derivative_reference = _lagrange_values_and_derivatives(
        grid.reference_nodes, q_nodes
    )
    derivative = derivative_reference * (2.0 / spacing)
    weights = q_weights * spacing / 2.0
    peak = _pml_peak(
        target_one_way_amplitude=pml_target_one_way_amplitude,
        wavenumber=k_value,
        length=pml,
        polynomial_order=pml_polynomial_order,
    )
    active_index = np.full(grid.radial_nodes_m.size, -1, dtype=np.int64)
    active_index[1:-1] = np.arange(grid.active_unknown_count)
    boundary_values = np.zeros(grid.radial_nodes_m.size, dtype=np.complex128)
    boundary_values[0] = 1.0 + 0.0j
    p1 = degree + 1
    maximum_entries = grid.element_count * p1 * p1
    rows = np.empty(maximum_entries, dtype=np.int32)
    columns = np.empty(maximum_entries, dtype=np.int32)
    entries = np.empty(maximum_entries, dtype=np.complex128)
    rhs = np.zeros(grid.active_unknown_count, dtype=np.complex128)
    position = 0
    for element in range(grid.element_count):
        left_x = element * spacing
        x_q = left_x + 0.5 * spacing * (q_nodes + 1.0)
        stretch, _ = _right_complex_stretch(
            x_q,
            start=core,
            length=pml,
            peak=peak,
            order=pml_polynomial_order,
        )
        local_matrix = -derivative.T @ (
            (weights / stretch)[:, None] * derivative
        )
        local_matrix += basis.T @ (
            (weights * k_value**2 * stretch)[:, None] * basis
        )
        global_nodes = element * degree + np.arange(p1)
        local_active = active_index[global_nodes]
        keep = local_active >= 0
        fixed = ~keep
        retained = local_active[keep].astype(np.int32, copy=False)
        if np.any(fixed):
            rhs[retained] -= (
                local_matrix[np.ix_(keep, fixed)]
                @ boundary_values[global_nodes[fixed]]
            )
        retained_matrix = local_matrix[np.ix_(keep, keep)]
        count = retained.size
        entry_count = count * count
        rows[position : position + entry_count] = np.repeat(retained, count)
        columns[position : position + entry_count] = np.tile(retained, count)
        entries[position : position + entry_count] = retained_matrix.reshape(-1)
        position += entry_count
    matrix = coo_matrix(
        (entries[:position], (rows[:position], columns[:position])),
        shape=(grid.active_unknown_count, grid.active_unknown_count),
        dtype=np.complex128,
    ).tocsc()
    matrix.sum_duplicates()
    solution, solver = solve_sparse_direct(matrix, rhs)
    nodal = np.zeros(grid.radial_nodes_m.size, dtype=np.complex128)
    nodal[0] = 1.0 + 0.0j
    nodal[1:-1] = solution
    fractions = np.asarray(measurement_fractions, dtype=np.float64)
    if np.any(fractions <= 0.0) or np.any(fractions >= 1.0):
        raise ValueError("measurement fractions must lie in (0, 1).")
    measurement_x = core * fractions
    dense_x = np.linspace(spacing, core * (1.0 - 1e-12), dense_comparison_count)
    measurement_radius = measurement_x + spacing
    dense_radius = dense_x + spacing
    measured_field, measured_derivative = evaluate_annular_fem_field_and_derivative(
        nodal, grid, measurement_radius
    )
    dense_field, _ = evaluate_annular_fem_field_and_derivative(
        nodal, grid, dense_radius
    )
    outgoing = np.exp(1j * k_value * measurement_x)
    incoming = np.exp(-1j * k_value * measurement_x)
    outgoing_d = 1j * k_value * outgoing
    incoming_d = -1j * k_value * incoming
    determinant = outgoing * incoming_d - incoming * outgoing_d
    coefficient_outgoing = (
        measured_field * incoming_d - incoming * measured_derivative
    ) / determinant
    coefficient_incoming = (
        outgoing * measured_derivative - measured_field * outgoing_d
    ) / determinant
    incoming_ratio = np.abs(coefficient_incoming) / np.maximum(
        np.abs(coefficient_outgoing), np.finfo(float).eps
    )
    impedance_residual = np.abs(
        measured_derivative - 1j * k_value * measured_field
    ) / np.maximum(k_value * np.abs(measured_field), np.finfo(float).eps)
    truth = np.exp(1j * k_value * dense_x)
    field_relative_l2 = float(
        np.linalg.norm(dense_field - truth) / np.linalg.norm(truth)
    )
    physical_z = propagation_direction * dense_x
    return {
        "direction": propagation_direction,
        "refractive_index": float(refractive_index),
        "wavenumber_per_m": k_value,
        "kh": float(k_value * spacing),
        "active_unknown_count": grid.active_unknown_count,
        "measurement_coordinates_m": propagation_direction * measurement_x,
        "incoming_to_outgoing_ratio": incoming_ratio,
        "outgoing_impedance_residual": impedance_residual,
        "dense_coordinates_m": physical_z,
        "dense_field": dense_field,
        "dense_truth": truth,
        "maximum_incoming_to_outgoing_ratio": float(np.max(incoming_ratio)),
        "maximum_outgoing_impedance_residual": float(
            np.max(impedance_residual)
        ),
        "dense_field_relative_l2": field_relative_l2,
        "solver_controls": solver,
        "all_finite": bool(
            np.all(np.isfinite(dense_field))
            and np.all(np.isfinite(truth))
            and np.all(np.isfinite(incoming_ratio))
            and np.all(np.isfinite(impedance_residual))
        ),
    }


__all__ = [
    "AnnularRadialFEMGrid",
    "annular_outgoing_pml_benchmark",
    "axial_plane_wave_pml_benchmark",
    "axisymmetric_modal_nodal_error",
    "decompose_cylindrical_field",
    "evaluate_annular_fem_field_and_derivative",
    "make_annular_radial_fem_grid",
    "normalized_cylindrical_bases",
    "physical_k_modal_fem_benchmark",
    "make_axisymmetric_pml_modal_problem",
]
