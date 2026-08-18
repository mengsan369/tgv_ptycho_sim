"""Axisymmetric scattered-field Helmholtz tools for exp040 R10 Stage B."""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_CONDA_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _CONDA_DLL_DIR.is_dir():
    _path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(_CONDA_DLL_DIR) not in _path_entries:
        os.environ["PATH"] = str(_CONDA_DLL_DIR) + os.pathsep + os.environ.get(
            "PATH", ""
        )

from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.sparse import csc_matrix, diags  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402


@dataclass(frozen=True)
class AxisymmetricGrid:
    """Cell-centered axisymmetric grid with PML outside a physical core."""

    dr_m: float
    dz_m: float
    radial_core_max_m: float
    z_core_min_m: float
    z_core_max_m: float
    pml_thickness_m: float
    r_faces_m: NDArray[np.float64]
    r_centers_m: NDArray[np.float64]
    z_faces_m: NDArray[np.float64]
    z_centers_m: NDArray[np.float64]

    @property
    def nr(self) -> int:
        return int(self.r_centers_m.size)

    @property
    def nz(self) -> int:
        return int(self.z_centers_m.size)

    @property
    def unknown_count(self) -> int:
        return self.nr * self.nz


@dataclass(frozen=True)
class CylindricalPML:
    """Complex stretches and coordinates sampled at cell centers/faces."""

    r_stretch_centers: NDArray[np.complex128]
    r_stretch_faces: NDArray[np.complex128]
    r_tilde_centers_m: NDArray[np.complex128]
    r_tilde_faces_m: NDArray[np.complex128]
    z_stretch_centers: NDArray[np.complex128]
    z_stretch_faces: NDArray[np.complex128]
    z_tilde_centers_m: NDArray[np.complex128]
    z_tilde_faces_m: NDArray[np.complex128]
    radial_peak_alpha: float
    lower_z_peak_alpha: float
    upper_z_peak_alpha: float


class PeakRSSMonitor:
    """Read the OS-maintained process peak working set around a solve."""

    def __init__(self, interval_s: float = 0.02) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be positive.")
        self.interval_s = float(interval_s)
        self.peak_rss_bytes = 0
        self._process: Any | None = None

    def _read_peak(self) -> int:
        if self._process is None:
            return -1
        info = self._process.memory_info()
        return int(getattr(info, "peak_wset", info.rss))

    def __enter__(self) -> PeakRSSMonitor:
        import psutil

        self._process = psutil.Process()
        self.peak_rss_bytes = self._read_peak()
        return self

    def __exit__(self, *_args: object) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._read_peak())


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def adc5_shifted_wavenumber_squared(
    wavenumber_per_m: NDArray[np.floating] | float,
    grid_spacing_m: float,
) -> NDArray[np.float64]:
    r"""Return the fixed 2-D five-point asymptotic dispersion correction.

    The value is the midpoint between the standard five-point stencil's
    axis-aligned and diagonal plane-wave symbols at the requested physical
    wavenumber.  It approaches ``k**2`` as the square-grid spacing tends to
    zero.
    """

    spacing = _positive(grid_spacing_m, "grid_spacing_m")
    wavenumber = np.asarray(wavenumber_per_m, dtype=np.float64)
    if not np.all(np.isfinite(wavenumber)) or np.any(wavenumber <= 0.0):
        raise ValueError("wavenumber_per_m must be finite and positive.")
    axis = (
        2.0 / spacing * np.sin(0.5 * wavenumber * spacing)
    ) ** 2
    diagonal = (
        2.0
        * np.sqrt(2.0)
        / spacing
        * np.sin(wavenumber * spacing / (2.0 * np.sqrt(2.0)))
    ) ** 2
    return np.asarray(0.5 * (axis + diagonal), dtype=np.float64)


def _aligned_count(length_m: float, spacing_m: float, name: str) -> int:
    ratio = float(length_m) / float(spacing_m)
    count = int(np.rint(ratio))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    if count <= 0 or abs(ratio - count) > tolerance:
        raise ValueError(f"{name} must be an integer multiple of its spacing.")
    return count


def make_axisymmetric_grid(
    *,
    dr_m: float,
    dz_m: float,
    radial_core_max_m: float,
    z_core_min_m: float,
    z_core_max_m: float,
    pml_thickness_m: float,
) -> AxisymmetricGrid:
    """Construct the registered cell-centered cylindrical grid."""

    dr = _positive(dr_m, "dr_m")
    dz = _positive(dz_m, "dz_m")
    radial_core = _positive(radial_core_max_m, "radial_core_max_m")
    pml = _positive(pml_thickness_m, "pml_thickness_m")
    z_min = float(z_core_min_m)
    z_max = float(z_core_max_m)
    if not np.all(np.isfinite([z_min, z_max])) or z_max <= z_min:
        raise ValueError("z core limits must be finite and increasing.")

    _aligned_count(radial_core, dr, "radial core")
    _aligned_count(pml, dr, "radial PML")
    _aligned_count(z_max - z_min, dz, "axial core")
    _aligned_count(pml, dz, "axial PML")

    radial_total = radial_core + pml
    z_total_min = z_min - pml
    z_total_max = z_max + pml
    nr = _aligned_count(radial_total, dr, "total radial extent")
    nz = _aligned_count(z_total_max - z_total_min, dz, "total axial extent")
    r_faces = np.arange(nr + 1, dtype=np.float64) * dr
    z_faces = z_total_min + np.arange(nz + 1, dtype=np.float64) * dz
    r_faces[-1] = radial_total
    z_faces[-1] = z_total_max
    return AxisymmetricGrid(
        dr_m=dr,
        dz_m=dz,
        radial_core_max_m=radial_core,
        z_core_min_m=z_min,
        z_core_max_m=z_max,
        pml_thickness_m=pml,
        r_faces_m=r_faces,
        r_centers_m=0.5 * (r_faces[:-1] + r_faces[1:]),
        z_faces_m=z_faces,
        z_centers_m=0.5 * (z_faces[:-1] + z_faces[1:]),
    )


def pml_peak_alpha(
    *, target_one_way_amplitude: float, wavenumber_per_m: float, length_m: float,
    polynomial_order: int,
) -> float:
    """Return peak dimensionless stretch for a polynomial PML."""

    target = float(target_one_way_amplitude)
    k_ref = _positive(wavenumber_per_m, "wavenumber_per_m")
    length = _positive(length_m, "length_m")
    order = int(polynomial_order)
    if not 0.0 < target < 1.0:
        raise ValueError("target_one_way_amplitude must lie in (0, 1).")
    if order < 1 or order != polynomial_order:
        raise ValueError("polynomial_order must be a positive integer.")
    return float(-(order + 1) * np.log(target) / (k_ref * length))


def _right_stretch(
    coordinates_m: NDArray[np.float64], *, start_m: float, length_m: float,
    peak_alpha: float, order: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    distance = np.maximum(coordinates_m - start_m, 0.0)
    alpha = peak_alpha * (distance / length_m) ** order
    integral = (
        peak_alpha
        * distance ** (order + 1)
        / ((order + 1) * length_m**order)
    )
    stretch = 1.0 + 1j * alpha
    coordinate_tilde = coordinates_m + 1j * integral
    return stretch.astype(np.complex128), coordinate_tilde.astype(np.complex128)


def _two_sided_z_stretch(
    coordinates_m: NDArray[np.float64], *, core_min_m: float,
    core_max_m: float, length_m: float, lower_peak_alpha: float,
    upper_peak_alpha: float, order: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    lower_distance = np.maximum(core_min_m - coordinates_m, 0.0)
    upper_distance = np.maximum(coordinates_m - core_max_m, 0.0)
    lower_alpha = lower_peak_alpha * (lower_distance / length_m) ** order
    upper_alpha = upper_peak_alpha * (upper_distance / length_m) ** order
    lower_integral = (
        lower_peak_alpha
        * lower_distance ** (order + 1)
        / ((order + 1) * length_m**order)
    )
    upper_integral = (
        upper_peak_alpha
        * upper_distance ** (order + 1)
        / ((order + 1) * length_m**order)
    )
    stretch = 1.0 + 1j * (lower_alpha + upper_alpha)
    coordinate_tilde = coordinates_m + 1j * (upper_integral - lower_integral)
    return stretch.astype(np.complex128), coordinate_tilde.astype(np.complex128)


def make_cylindrical_pml(
    grid: AxisymmetricGrid,
    *,
    wavelength_m: float,
    n_glass: float,
    n_air: float,
    polynomial_order: int = 3,
    target_one_way_amplitude: float = 1.0e-8,
) -> CylindricalPML:
    """Sample the registered cubic cylindrical complex-coordinate PML."""

    wavelength = _positive(wavelength_m, "wavelength_m")
    glass = _positive(n_glass, "n_glass")
    air = _positive(n_air, "n_air")
    k0 = 2.0 * np.pi / wavelength
    radial_peak = pml_peak_alpha(
        target_one_way_amplitude=target_one_way_amplitude,
        wavenumber_per_m=k0 * air,
        length_m=grid.pml_thickness_m,
        polynomial_order=polynomial_order,
    )
    lower_peak = pml_peak_alpha(
        target_one_way_amplitude=target_one_way_amplitude,
        wavenumber_per_m=k0 * glass,
        length_m=grid.pml_thickness_m,
        polynomial_order=polynomial_order,
    )
    upper_peak = radial_peak
    sr_centers, rt_centers = _right_stretch(
        grid.r_centers_m,
        start_m=grid.radial_core_max_m,
        length_m=grid.pml_thickness_m,
        peak_alpha=radial_peak,
        order=polynomial_order,
    )
    sr_faces, rt_faces = _right_stretch(
        grid.r_faces_m,
        start_m=grid.radial_core_max_m,
        length_m=grid.pml_thickness_m,
        peak_alpha=radial_peak,
        order=polynomial_order,
    )
    sz_centers, zt_centers = _two_sided_z_stretch(
        grid.z_centers_m,
        core_min_m=grid.z_core_min_m,
        core_max_m=grid.z_core_max_m,
        length_m=grid.pml_thickness_m,
        lower_peak_alpha=lower_peak,
        upper_peak_alpha=upper_peak,
        order=polynomial_order,
    )
    sz_faces, zt_faces = _two_sided_z_stretch(
        grid.z_faces_m,
        core_min_m=grid.z_core_min_m,
        core_max_m=grid.z_core_max_m,
        length_m=grid.pml_thickness_m,
        lower_peak_alpha=lower_peak,
        upper_peak_alpha=upper_peak,
        order=polynomial_order,
    )
    return CylindricalPML(
        r_stretch_centers=sr_centers,
        r_stretch_faces=sr_faces,
        r_tilde_centers_m=rt_centers,
        r_tilde_faces_m=rt_faces,
        z_stretch_centers=sz_centers,
        z_stretch_faces=sz_faces,
        z_tilde_centers_m=zt_centers,
        z_tilde_faces_m=zt_faces,
        radial_peak_alpha=radial_peak,
        lower_z_peak_alpha=lower_peak,
        upper_z_peak_alpha=upper_peak,
    )


def scalar_interface_background(
    z_m: NDArray[np.floating] | NDArray[np.complexfloating],
    *,
    physical_z_m: NDArray[np.floating] | None = None,
    wavelength_m: float,
    n_glass: float,
    n_air: float,
    interface_z_m: float,
    incident_amplitude: float = 1.0,
) -> NDArray[np.complex128]:
    """Evaluate the analytic normal-incidence scalar glass/air background."""

    z = np.asarray(z_m, dtype=np.complex128)
    physical_z = np.real(z) if physical_z_m is None else np.asarray(physical_z_m)
    if physical_z.shape != z.shape:
        raise ValueError("physical_z_m must match z_m.")
    wavelength = _positive(wavelength_m, "wavelength_m")
    glass = _positive(n_glass, "n_glass")
    air = _positive(n_air, "n_air")
    interface = float(interface_z_m)
    amplitude = float(incident_amplitude)
    if not np.isfinite(interface) or not np.isfinite(amplitude):
        raise ValueError("interface and incident amplitude must be finite.")
    k0 = 2.0 * np.pi / wavelength
    kg = k0 * glass
    ka = k0 * air
    reflection = (kg - ka) / (kg + ka)
    transmission = 2.0 * kg / (kg + ka)
    phase_at_interface = np.exp(1j * kg * interface)
    below = physical_z < interface
    field = np.empty(z.shape, dtype=np.complex128)
    field[below] = amplitude * (
        np.exp(1j * kg * z[below])
        + reflection * np.exp(1j * kg * (2.0 * interface - z[below]))
    )
    field[~below] = (
        amplitude
        * transmission
        * phase_at_interface
        * np.exp(1j * ka * (z[~below] - interface))
    )
    return field


def background_interface_controls(
    *, wavelength_m: float, n_glass: float, n_air: float,
    interface_z_m: float, incident_amplitude: float = 1.0,
) -> dict[str, float]:
    """Return analytic value and derivative continuity errors."""

    wavelength = _positive(wavelength_m, "wavelength_m")
    glass = _positive(n_glass, "n_glass")
    air = _positive(n_air, "n_air")
    interface = float(interface_z_m)
    amplitude = float(incident_amplitude)
    k0 = 2.0 * np.pi / wavelength
    kg = k0 * glass
    ka = k0 * air
    reflection = (kg - ka) / (kg + ka)
    transmission = 2.0 * kg / (kg + ka)
    phase = amplitude * np.exp(1j * kg * interface)
    value_below = (1.0 + reflection) * phase
    value_above = transmission * phase
    derivative_below = 1j * kg * (1.0 - reflection) * phase
    derivative_above = 1j * ka * transmission * phase
    return {
        "reflection_coefficient": float(reflection),
        "transmission_coefficient": float(transmission),
        "value_continuity_relative_error": float(
            abs(value_below - value_above)
            / max(abs(value_above), np.finfo(float).eps)
        ),
        "derivative_continuity_relative_error": float(
            abs(derivative_below - derivative_above)
            / max(abs(derivative_above), np.finfo(float).eps)
        ),
    }


def make_background_n2(
    grid: AxisymmetricGrid,
    *,
    interface_z_m: float,
    n_glass: float,
    n_air: float,
) -> NDArray[np.float64]:
    """Return the flat-interface background n-squared on cell centers."""

    interface = float(interface_z_m)
    glass = _positive(n_glass, "n_glass")
    air = _positive(n_air, "n_air")
    axial = np.where(grid.z_centers_m < interface, glass**2, air**2)
    return np.broadcast_to(axial[:, None], (grid.nz, grid.nr)).copy()


def _diameter_piecewise_linear(
    z_m: NDArray[np.float64], *, thickness_m: float, d_top_m: float,
    d_waist_m: float, d_bottom_m: float, z_waist_m: float,
) -> NDArray[np.float64]:
    before = z_m <= z_waist_m
    result = np.empty_like(z_m)
    result[before] = d_top_m + (d_waist_m - d_top_m) * (
        z_m[before] / z_waist_m
    )
    result[~before] = d_waist_m + (d_bottom_m - d_waist_m) * (
        (z_m[~before] - z_waist_m) / (thickness_m - z_waist_m)
    )
    return result


def make_tgv_n2_cell_average(
    grid: AxisymmetricGrid,
    *,
    thickness_m: float,
    d_top_m: float,
    d_waist_m: float,
    d_bottom_m: float,
    z_waist_m: float,
    n_glass: float,
    n_air: float,
    axial_subnodes: int,
    background_interface_z_m: float,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Apply exact annular area integration and axial midpoint quadrature."""

    thickness = _positive(thickness_m, "thickness_m")
    d_top = _positive(d_top_m, "d_top_m")
    d_waist = _positive(d_waist_m, "d_waist_m")
    d_bottom = _positive(d_bottom_m, "d_bottom_m")
    z_waist = float(z_waist_m)
    glass = _positive(n_glass, "n_glass")
    air = _positive(n_air, "n_air")
    q = int(axial_subnodes)
    if q <= 0 or q != axial_subnodes:
        raise ValueError("axial_subnodes must be a positive integer.")
    if not 0.0 < z_waist < thickness:
        raise ValueError("z_waist_m must lie inside the sample.")
    if d_waist > min(d_top, d_bottom):
        raise ValueError("d_waist_m must not exceed surface diameters.")

    faces = grid.z_faces_m
    zero_index = int(np.argmin(np.abs(faces)))
    end_index = int(np.argmin(np.abs(faces - thickness)))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, thickness)
    if abs(faces[zero_index]) > tolerance:
        raise ValueError("z=0 must align with an axial face.")
    if abs(faces[end_index] - thickness) > tolerance or end_index <= zero_index:
        raise ValueError("sample thickness must align with an axial face.")

    n2 = make_background_n2(
        grid,
        interface_z_m=background_interface_z_m,
        n_glass=glass,
        n_air=air,
    )
    r_lo2 = grid.r_faces_m[:-1] ** 2
    r_hi2 = grid.r_faces_m[1:] ** 2
    annulus_denominator = r_hi2 - r_lo2
    fractions = np.zeros((end_index - zero_index, grid.nr), dtype=np.float64)
    subnode_volume = 0.0
    offsets = (np.arange(q, dtype=np.float64) + 0.5) / q
    for local_index, axial_index in enumerate(range(zero_index, end_index)):
        z_lo = faces[axial_index]
        z_hi = faces[axial_index + 1]
        nodes = z_lo + offsets * (z_hi - z_lo)
        diameters = _diameter_piecewise_linear(
            nodes,
            thickness_m=thickness,
            d_top_m=d_top,
            d_waist_m=d_waist,
            d_bottom_m=d_bottom,
            z_waist_m=z_waist,
        )
        fraction = np.zeros(grid.nr, dtype=np.float64)
        for diameter in diameters:
            radius2 = (0.5 * diameter) ** 2
            numerator = np.clip(
                np.minimum(radius2, r_hi2) - r_lo2,
                0.0,
                annulus_denominator,
            )
            fraction += numerator / annulus_denominator
            subnode_volume += np.pi * radius2 * (z_hi - z_lo) / q
        fractions[local_index] = fraction / q
    n2[zero_index:end_index] = (
        glass**2 + fractions * (air**2 - glass**2)
    )
    annular_areas = np.pi * annulus_denominator
    discrete_volume = float(
        np.sum(fractions * annular_areas[None, :]) * grid.dz_m
    )
    exact_integral_d2 = (
        z_waist * (d_top**2 + d_top * d_waist + d_waist**2) / 3.0
        + (thickness - z_waist)
        * (d_waist**2 + d_waist * d_bottom + d_bottom**2)
        / 3.0
    )
    exact_volume = float(0.25 * np.pi * exact_integral_d2)
    fraction_error = max(
        0.0,
        -float(np.min(fractions)),
        float(np.max(fractions)) - 1.0,
    )
    return n2, {
        "axial_subnodes": q,
        "sample_axial_cell_count": int(end_index - zero_index),
        "fraction_min": float(np.min(fractions)),
        "fraction_max": float(np.max(fractions)),
        "fraction_bound_error": float(fraction_error),
        "discrete_air_volume_m3": discrete_volume,
        "subnode_continuous_air_volume_m3": float(subnode_volume),
        "exact_continuous_air_volume_m3": exact_volume,
        "annular_to_subnode_volume_relative_error": float(
            abs(discrete_volume - subnode_volume)
            / max(abs(subnode_volume), np.finfo(float).eps)
        ),
        "subnode_to_exact_volume_relative_error_report_only": float(
            abs(subnode_volume - exact_volume)
            / max(abs(exact_volume), np.finfo(float).eps)
        ),
        "all_finite": bool(np.all(np.isfinite(n2))),
    }


def assemble_cylindrical_helmholtz(
    grid: AxisymmetricGrid,
    pml: CylindricalPML,
    n2: NDArray[np.floating],
    *,
    wavelength_m: float,
) -> tuple[csc_matrix, dict[str, Any]]:
    """Assemble the registered five-point conservative Helmholtz matrix."""

    n2_values = np.asarray(n2, dtype=np.float64)
    if n2_values.shape != (grid.nz, grid.nr):
        raise ValueError("n2 must have shape (nz, nr).")
    if not np.all(np.isfinite(n2_values)) or np.any(n2_values <= 0.0):
        raise ValueError("n2 must be finite and positive.")
    wavelength = _positive(wavelength_m, "wavelength_m")
    k0 = 2.0 * np.pi / wavelength
    rt_c = pml.r_tilde_centers_m
    rt_f = pml.r_tilde_faces_m
    sr_c = pml.r_stretch_centers
    sr_f = pml.r_stretch_faces
    sz_c = pml.z_stretch_centers
    sz_f = pml.z_stretch_faces
    radial_faces = sz_c[:, None] * (rt_f / sr_f)[None, :] / grid.dr_m**2
    axial_faces = (
        (rt_c * sr_c)[None, :] / sz_f[:, None] / grid.dz_m**2
    )
    west = radial_faces[:, :-1]
    east = radial_faces[:, 1:]
    down = axial_faces[:-1, :]
    up = axial_faces[1:, :]
    mass = (
        k0**2
        * n2_values
        * (rt_c * sr_c)[None, :]
        * sz_c[:, None]
    )
    main = mass - west - east - down - up
    main[:, -1] -= east[:, -1]
    main[0, :] -= down[0, :]
    main[-1, :] -= up[-1, :]

    unknowns = grid.unknown_count
    radial_upper = np.zeros(unknowns - 1, dtype=np.complex128)
    radial_mask = np.arange(unknowns - 1) % grid.nr != grid.nr - 1
    radial_upper[radial_mask] = east[:, :-1].reshape(-1)
    axial_upper = up[:-1, :].reshape(-1)
    matrix = diags(
        [
            axial_upper,
            radial_upper,
            main.reshape(-1),
            radial_upper,
            axial_upper,
        ],
        offsets=[-grid.nr, -1, 0, 1, grid.nr],
        shape=(unknowns, unknowns),
        dtype=np.complex128,
        format="csc",
    )
    expected_max_nnz = 5 * unknowns - 2 * grid.nz - 2 * grid.nr
    controls = {
        "shape": [unknowns, unknowns],
        "nnz": int(matrix.nnz),
        "five_point_max_nnz": int(expected_max_nnz),
        "finite_data": bool(np.all(np.isfinite(matrix.data))),
        "complex_symmetric_max_abs_error": 0.0,
        "outer_dirichlet_half_cell_flux": True,
        "axis_zero_flux": True,
    }
    return matrix, controls


def make_contrast_source(
    grid: AxisymmetricGrid,
    pml: CylindricalPML,
    n2_tgv: NDArray[np.floating],
    n2_background: NDArray[np.floating],
    *,
    wavelength_m: float,
    background_field_z: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Return the local scattered-field contrast source in flattened order."""

    tgv = np.asarray(n2_tgv, dtype=np.float64)
    background = np.asarray(n2_background, dtype=np.float64)
    field_z = np.asarray(background_field_z, dtype=np.complex128)
    if tgv.shape != (grid.nz, grid.nr) or background.shape != tgv.shape:
        raise ValueError("n2 arrays must match the grid.")
    if field_z.shape != (grid.nz,):
        raise ValueError("background_field_z must have shape (nz,).")
    k0 = 2.0 * np.pi / _positive(wavelength_m, "wavelength_m")
    jacobian = (
        (pml.r_tilde_centers_m * pml.r_stretch_centers)[None, :]
        * pml.z_stretch_centers[:, None]
    )
    source = -k0**2 * (tgv - background) * jacobian * field_z[:, None]
    return source.reshape(-1).astype(np.complex128, copy=False)


def solve_sparse_direct(
    matrix: csc_matrix,
    rhs: NDArray[np.complexfloating],
    *,
    permc_spec: str = "COLAMD",
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Factor and solve once with SuperLU, returning residual/resource controls."""

    if not isinstance(matrix, csc_matrix):
        raise TypeError("matrix must be CSC.")
    right = np.asarray(rhs, dtype=np.complex128)
    if right.ndim != 1 or right.shape[0] != matrix.shape[0]:
        raise ValueError("rhs must be a length-N complex vector.")
    if not np.all(np.isfinite(right)):
        raise ValueError("rhs must be finite.")
    started = time.perf_counter()
    with PeakRSSMonitor() as monitor:
        factor_started = time.perf_counter()
        lu = splu(matrix, permc_spec=permc_spec)
        factor_elapsed = time.perf_counter() - factor_started
        solve_started = time.perf_counter()
        solution = np.asarray(lu.solve(right), dtype=np.complex128)
        solve_elapsed = time.perf_counter() - solve_started
        factor_nnz = int(lu.L.nnz + lu.U.nnz)
    residual = matrix @ solution - right
    rhs_norm = float(np.linalg.norm(right))
    residual_norm = float(np.linalg.norm(residual))
    relative_residual = residual_norm / max(rhs_norm, np.finfo(float).eps)
    controls = {
        "solver": "scipy_splu",
        "permc_spec": permc_spec,
        "matrix_nnz": int(matrix.nnz),
        "factor_l_plus_u_nnz": factor_nnz,
        "fill_ratio": float(factor_nnz / max(matrix.nnz, 1)),
        "factor_elapsed_s": float(factor_elapsed),
        "solve_elapsed_s": float(solve_elapsed),
        "factor_and_solve_elapsed_s": float(time.perf_counter() - started),
        "peak_rss_bytes": int(monitor.peak_rss_bytes),
        "rhs_l2": rhs_norm,
        "residual_l2": residual_norm,
        "relative_residual": float(relative_residual),
        "all_finite": bool(
            np.all(np.isfinite(solution)) and np.all(np.isfinite(residual))
        ),
    }
    del lu, residual
    gc.collect()
    return solution, controls


def observation_trace(
    values: NDArray[np.complexfloating],
    grid: AxisymmetricGrid,
    *,
    observation_z_m: float,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Linearly interpolate a flattened/grid field to one physical z plane."""

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 1:
        if array.size != grid.unknown_count:
            raise ValueError("flattened values have the wrong length.")
        array = array.reshape(grid.nz, grid.nr)
    if array.shape != (grid.nz, grid.nr):
        raise ValueError("values must match (nz, nr).")
    observation = float(observation_z_m)
    centers = grid.z_centers_m
    upper = int(np.searchsorted(centers, observation, side="right"))
    if upper <= 0 or upper >= grid.nz:
        raise ValueError("observation plane must be bracketed by cell centers.")
    lower = upper - 1
    weight = (observation - centers[lower]) / (centers[upper] - centers[lower])
    trace = (1.0 - weight) * array[lower] + weight * array[upper]
    return trace.astype(np.complex128, copy=False), {
        "observation_z_m": observation,
        "lower_center_z_m": float(centers[lower]),
        "upper_center_z_m": float(centers[upper]),
        "upper_weight": float(weight),
        "lower_index": lower,
        "upper_index": upper,
    }


def make_manufactured_vector(grid: AxisymmetricGrid) -> NDArray[np.complex128]:
    """Return a smooth vector regular at the axis and zero on outer faces."""

    radial_total = float(grid.r_faces_m[-1])
    z_min = float(grid.z_faces_m[0])
    z_length = float(grid.z_faces_m[-1] - z_min)
    radial = np.cos(0.5 * np.pi * grid.r_centers_m / radial_total)
    axial = np.sin(np.pi * (grid.z_centers_m - z_min) / z_length)
    values = axial[:, None] * radial[None, :] * (1.0 + 0.25j)
    return values.reshape(-1).astype(np.complex128, copy=False)


def radial_weighted_relative_l2(
    test: NDArray[np.complexfloating],
    reference: NDArray[np.complexfloating],
    radius_m: NDArray[np.floating],
) -> float:
    """Return the fixed 2-pi-r weighted radial relative L2."""

    left = np.asarray(test, dtype=np.complex128)
    right = np.asarray(reference, dtype=np.complex128)
    radius = np.asarray(radius_m, dtype=np.float64)
    if left.shape != right.shape or left.shape != radius.shape or left.ndim != 1:
        raise ValueError("radial arrays must be matching one-dimensional arrays.")
    if np.any(radius <= 0.0) or not np.all(np.isfinite(radius)):
        raise ValueError("radius_m must be finite and positive.")
    weights = 2.0 * np.pi * radius
    numerator = float(np.sum(weights * np.abs(left - right) ** 2))
    denominator = float(np.sum(weights * np.abs(right) ** 2))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).eps)))


def radial_trace_to_cartesian(
    trace: NDArray[np.complexfloating],
    trace_radius_m: NDArray[np.floating],
    *,
    shape: tuple[int, int],
    dx_m: float,
    trace_support_radius_m: float,
    outer_value: complex = 1.0 + 0.0j,
) -> NDArray[np.complex128]:
    """Interpolate a radial trace onto the registered centered Cartesian grid."""

    values = np.asarray(trace, dtype=np.complex128)
    radius_nodes = np.asarray(trace_radius_m, dtype=np.float64)
    if values.ndim != 1 or values.shape != radius_nodes.shape:
        raise ValueError("trace and radius nodes must be matching 1D arrays.")
    if np.any(np.diff(radius_nodes) <= 0.0):
        raise ValueError("trace radii must be strictly increasing.")
    ny, nx = (int(value) for value in shape)
    spacing = _positive(dx_m, "dx_m")
    support = _positive(trace_support_radius_m, "trace_support_radius_m")
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * spacing
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * spacing
    radius = np.hypot(y[:, None], x[None, :])
    real = np.interp(
        radius.reshape(-1),
        radius_nodes,
        values.real,
        left=float(values.real[0]),
        right=float(values.real[-1]),
    )
    imag = np.interp(
        radius.reshape(-1),
        radius_nodes,
        values.imag,
        left=float(values.imag[0]),
        right=float(values.imag[-1]),
    )
    mapped = (real + 1j * imag).reshape(ny, nx)
    mapped[radius > support] = complex(outer_value)
    return mapped.astype(np.complex128, copy=False)


def annular_mean_from_cartesian(
    field: NDArray[np.complexfloating],
    *,
    dx_m: float,
    bin_width_m: float,
    maximum_radius_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.int64]]:
    """Average Cartesian pixel centers in fixed radial annular bins."""

    values = np.asarray(field, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("field must be a nonempty 2D array.")
    spacing = _positive(dx_m, "dx_m")
    width = _positive(bin_width_m, "bin_width_m")
    maximum = _positive(maximum_radius_m, "maximum_radius_m")
    bin_count = _aligned_count(maximum, width, "maximum radius")
    ny, nx = values.shape
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * spacing
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * spacing
    radius = np.hypot(y[:, None], x[None, :])
    indices = np.floor(radius / width).astype(np.int64)
    valid = indices < bin_count
    flat_indices = indices[valid]
    counts = np.bincount(flat_indices, minlength=bin_count).astype(np.int64)
    if np.any(counts == 0):
        raise RuntimeError("at least one registered annular bin is empty.")
    real_sum = np.bincount(
        flat_indices, weights=values.real[valid], minlength=bin_count
    )
    imag_sum = np.bincount(
        flat_indices, weights=values.imag[valid], minlength=bin_count
    )
    means = (real_sum + 1j * imag_sum) / counts
    centers = (np.arange(bin_count, dtype=np.float64) + 0.5) * width
    return centers, means.astype(np.complex128), counts


def annular_anisotropy_relative_l2(
    field: NDArray[np.complexfloating],
    *,
    dx_m: float,
    bin_width_m: float,
    maximum_radius_m: float,
) -> tuple[float, NDArray[np.complex128]]:
    """Compare a Cartesian field with its fixed annular-bin projection."""

    values = np.asarray(field, dtype=np.complex128)
    centers, means, _ = annular_mean_from_cartesian(
        values,
        dx_m=dx_m,
        bin_width_m=bin_width_m,
        maximum_radius_m=maximum_radius_m,
    )
    del centers
    ny, nx = values.shape
    spacing = float(dx_m)
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * spacing
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * spacing
    radius = np.hypot(y[:, None], x[None, :])
    indices = np.floor(radius / float(bin_width_m)).astype(np.int64)
    valid = indices < means.size
    projection = np.full(values.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    projection[valid] = means[indices[valid]]
    numerator = float(np.sum(np.abs(values[valid] - projection[valid]) ** 2))
    denominator = float(np.sum(np.abs(projection[valid]) ** 2))
    error = float(np.sqrt(numerator / max(denominator, np.finfo(float).eps)))
    return error, projection


def sample_centered_cartesian_on_polar_grid(
    field: NDArray[np.complexfloating],
    *,
    dx_m: float | tuple[float, float],
    radius_m: NDArray[np.floating],
    theta_rad: NDArray[np.floating],
    interpolation_order: int = 3,
) -> NDArray[np.complex128]:
    """Sample a centered Cartesian complex field on exact polar nodes."""

    values = np.asarray(field, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) <= 1:
        raise ValueError("field must be a nonempty two-dimensional array.")
    if isinstance(dx_m, tuple):
        if len(dx_m) != 2:
            raise ValueError("dx_m tuple must be (dy, dx).")
        dy, dx = (_positive(value, "dx_m") for value in dx_m)
    else:
        dy = dx = _positive(dx_m, "dx_m")
    radius = np.asarray(radius_m, dtype=np.float64)
    theta = np.asarray(theta_rad, dtype=np.float64)
    if (
        radius.ndim != 1
        or theta.ndim != 1
        or radius.size == 0
        or theta.size == 0
        or not np.all(np.isfinite(radius))
        or not np.all(np.isfinite(theta))
        or np.any(radius <= 0.0)
        or np.any(np.diff(radius) <= 0.0)
    ):
        raise ValueError("radius_m/theta_rad must be finite 1D polar nodes.")
    order = int(interpolation_order)
    if isinstance(interpolation_order, bool) or order != interpolation_order:
        raise ValueError("interpolation_order must be an integer.")
    if order < 0 or order > 5:
        raise ValueError("interpolation_order must lie in [0, 5].")

    sample_y = radius[:, None] * np.sin(theta[None, :])
    sample_x = radius[:, None] * np.cos(theta[None, :])
    row = sample_y / dy + (values.shape[0] - 1) / 2.0
    column = sample_x / dx + (values.shape[1] - 1) / 2.0
    tolerance = 64.0 * np.finfo(float).eps * max(values.shape)
    if (
        float(np.min(row)) < -tolerance
        or float(np.max(row)) > values.shape[0] - 1 + tolerance
        or float(np.min(column)) < -tolerance
        or float(np.max(column)) > values.shape[1] - 1 + tolerance
    ):
        raise ValueError("polar nodes must remain inside the Cartesian field.")
    coordinates = np.vstack([row.reshape(-1), column.reshape(-1)])
    kwargs = {
        "order": order,
        "mode": "nearest",
        "prefilter": bool(order > 1),
    }
    real = map_coordinates(values.real, coordinates, **kwargs)
    imag = map_coordinates(values.imag, coordinates, **kwargs)
    return (real + 1j * imag).reshape(radius.size, theta.size)


def polar_angular_diagnostics(
    polar_field: NDArray[np.complexfloating],
    *,
    radius_m: NDArray[np.floating],
    theta_rad: NDArray[np.floating],
    harmonics: tuple[int, ...] = (4, 8),
) -> tuple[dict[str, Any], NDArray[np.complex128]]:
    """Return fixed-radius angular residual, harmonics, and rotation control."""

    values = np.asarray(polar_field, dtype=np.complex128)
    radius = np.asarray(radius_m, dtype=np.float64)
    theta = np.asarray(theta_rad, dtype=np.float64)
    if values.shape != (radius.size, theta.size) or values.ndim != 2:
        raise ValueError("polar_field must have shape (radius, theta).")
    if not np.all(np.isfinite(values)):
        raise ValueError("polar_field must be finite.")
    if theta.size < 8 or theta.size % 8 != 0:
        raise ValueError("theta count must be a positive multiple of eight.")
    expected = 2.0 * np.pi * np.arange(theta.size) / theta.size
    angular_grid_error = float(
        np.max(np.abs(np.angle(np.exp(1j * (theta - expected)))))
    )
    if angular_grid_error > 1.0e-12:
        raise ValueError("theta_rad must be the registered uniform [0,2pi) grid.")

    angular_mean = np.mean(values, axis=1, dtype=np.complex128)
    residual = values - angular_mean[:, None]
    weights = radius[:, None]
    numerator = float(np.sum(weights * np.abs(residual) ** 2))
    denominator = float(
        theta.size * np.sum(radius * np.abs(angular_mean) ** 2)
    )
    angular_relative_l2 = float(
        np.sqrt(numerator / max(denominator, np.finfo(float).eps))
    )
    harmonic_relative_l2: dict[str, float] = {}
    mean_energy = float(np.sum(radius * np.abs(angular_mean) ** 2))
    for harmonic in harmonics:
        mode = int(harmonic)
        if mode <= 0 or mode != harmonic or mode >= theta.size // 2:
            raise ValueError("harmonics must be positive resolved integers.")
        positive = np.mean(
            values * np.exp(-1j * mode * theta)[None, :], axis=1
        )
        negative = np.mean(
            values * np.exp(1j * mode * theta)[None, :], axis=1
        )
        mode_energy = float(
            np.sum(radius * (np.abs(positive) ** 2 + np.abs(negative) ** 2))
        )
        harmonic_relative_l2[f"m{mode}"] = float(
            np.sqrt(mode_energy / max(mean_energy, np.finfo(float).eps))
        )
    shift = theta.size // 8
    rotated = np.roll(values, -shift, axis=1)
    rotation_numerator = float(
        np.sum(weights * np.abs(values - rotated) ** 2)
    )
    rotation_denominator = float(np.sum(weights * np.abs(values) ** 2))
    controls = {
        "angular_relative_l2": angular_relative_l2,
        "rotation_45deg_relative_l2": float(
            np.sqrt(
                rotation_numerator
                / max(rotation_denominator, np.finfo(float).eps)
            )
        ),
        "harmonic_relative_l2": harmonic_relative_l2,
        "radius_count": int(radius.size),
        "theta_count": int(theta.size),
        "theta_uniformity_max_abs_error_rad": angular_grid_error,
        "all_finite": True,
    }
    return controls, angular_mean


def cartesian_polar_angular_diagnostics(
    field: NDArray[np.complexfloating],
    *,
    dx_m: float | tuple[float, float],
    radius_m: NDArray[np.floating],
    theta_rad: NDArray[np.floating],
    interpolation_order: int = 3,
    harmonics: tuple[int, ...] = (4, 8),
) -> tuple[dict[str, Any], NDArray[np.complex128]]:
    """Sample a Cartesian field and return fixed-radius angular diagnostics."""

    polar = sample_centered_cartesian_on_polar_grid(
        field,
        dx_m=dx_m,
        radius_m=radius_m,
        theta_rad=theta_rad,
        interpolation_order=interpolation_order,
    )
    controls, angular_mean = polar_angular_diagnostics(
        polar,
        radius_m=radius_m,
        theta_rad=theta_rad,
        harmonics=harmonics,
    )
    controls["interpolation_order"] = int(interpolation_order)
    return controls, angular_mean


def outer_guard_rms_ratio(
    normalized_scattered_trace: NDArray[np.complexfloating],
    radius_m: NDArray[np.floating],
    *,
    inner_max_radius_m: float,
    guard_min_radius_m: float,
    guard_max_radius_m: float,
) -> float:
    """Return weighted guard RMS divided by weighted inner scattered RMS."""

    values = np.asarray(normalized_scattered_trace, dtype=np.complex128)
    radius = np.asarray(radius_m, dtype=np.float64)
    if values.shape != radius.shape or values.ndim != 1:
        raise ValueError("guard trace and radii must be matching 1D arrays.")
    inner = radius <= float(inner_max_radius_m)
    guard = (radius >= float(guard_min_radius_m)) & (
        radius <= float(guard_max_radius_m)
    )
    if not np.any(inner) or not np.any(guard):
        raise ValueError("inner and guard intervals must contain grid centers.")

    def rms(mask: NDArray[np.bool_]) -> float:
        weights = 2.0 * np.pi * radius[mask]
        return float(
            np.sqrt(
                np.sum(weights * np.abs(values[mask]) ** 2)
                / np.sum(weights)
            )
        )

    return float(rms(guard) / max(rms(inner), np.finfo(float).eps))


__all__ = [
    "AxisymmetricGrid",
    "CylindricalPML",
    "PeakRSSMonitor",
    "annular_anisotropy_relative_l2",
    "annular_mean_from_cartesian",
    "adc5_shifted_wavenumber_squared",
    "assemble_cylindrical_helmholtz",
    "background_interface_controls",
    "make_axisymmetric_grid",
    "make_background_n2",
    "make_contrast_source",
    "make_cylindrical_pml",
    "make_manufactured_vector",
    "make_tgv_n2_cell_average",
    "observation_trace",
    "outer_guard_rms_ratio",
    "cartesian_polar_angular_diagnostics",
    "polar_angular_diagnostics",
    "pml_peak_alpha",
    "radial_trace_to_cartesian",
    "radial_weighted_relative_l2",
    "sample_centered_cartesian_on_polar_grid",
    "scalar_interface_background",
    "solve_sparse_direct",
]
