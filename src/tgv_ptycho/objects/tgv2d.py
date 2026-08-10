"""Projected-phase and legacy thin-disk models for a single TGV."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.objects.tgv_geometry import (
    analytic_air_path_length,
    diameter_profile,
    midpoint_z_grid,
    validate_tgv_geometry,
)
from tgv_ptycho.optics.angular_spectrum import _normalize_dx
from tgv_ptycho.optics.fields import coordinate_grid


def make_thin_phase_disk(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    diameter: float,
    phase_shift: float,
) -> NDArray[np.complex128]:
    """Create a centered thin phase disk transmission function.

    The returned transmission is one outside the disk and
    `exp(i * phase_shift)` inside the disk. This is a 2D test object only.
    """

    if diameter <= 0:
        msg = "diameter must be positive."
        raise ValueError(msg)
    x_grid, y_grid = coordinate_grid(shape, dx)
    radius = diameter / 2.0
    inside = (x_grid**2 + y_grid**2) <= radius**2
    transmission = np.ones(shape, dtype=np.complex128)
    transmission[inside] = np.exp(1j * phase_shift)
    return transmission


def make_tgv_effective_phase_2d(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    diameter_top: float,
    diameter_bottom: float | None = None,
    phase_shift: float = 1.0,
) -> NDArray[np.complex128]:
    """Create a legacy constant-phase disk, not a waist-sensitive TGV model.

    The former arithmetic-mean diameter behavior discarded ``D(z)`` and is no
    longer supported. Use :func:`make_tgv_projected_phase` for a parameterized
    TGV. Equal top/bottom diameters remain a compatible thin-disk control.
    """

    if diameter_bottom is not None and not np.isclose(
        diameter_bottom, diameter_top, rtol=0.0, atol=0.0
    ):
        msg = (
            "Unequal top/bottom diameters require make_tgv_projected_phase; "
            "an arithmetic-mean disk is not a TGV waist model."
        )
        raise ValueError(msg)
    return make_thin_phase_disk(shape, dx, diameter_top, phase_shift)


def _midpoint_path_from_radius(
    radius_m: NDArray[np.float64],
    z_m: NDArray[np.float64],
    slice_width_m: NDArray[np.float64],
    diameter_z_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    radii = 0.5 * np.asarray(diameter_z_m, dtype=np.float64)
    order = np.argsort(radii)
    sorted_radii = radii[order]
    sorted_widths = np.asarray(slice_width_m, dtype=np.float64)[order]
    suffix_width = np.cumsum(sorted_widths[::-1])[::-1]
    if suffix_width.size:
        integration_start = float(z_m[0] - 0.5 * slice_width_m[0])
        integration_stop = float(z_m[-1] + 0.5 * slice_width_m[-1])
        suffix_width[0] = integration_stop - integration_start
    indices = np.searchsorted(sorted_radii, radius_m.ravel(), side="left")
    path = np.zeros(radius_m.size, dtype=np.float64)
    inside = indices < len(z_m)
    path[inside] = suffix_width[indices[inside]]
    return path.reshape(radius_m.shape)


def _average_subpixels(
    values: NDArray[np.float64],
    shape: tuple[int, int],
    supersampling: int,
) -> NDArray[np.float64]:
    if supersampling == 1:
        return values.astype(np.float64, copy=False)
    ny, nx = shape
    reshaped = values.reshape(ny, supersampling, nx, supersampling)
    return reshaped.mean(axis=(1, 3), dtype=np.float64)


def make_tgv_projected_phase(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    wavelength: float,
    thickness: float,
    d_top: float,
    d_waist: float,
    d_bottom: float,
    dz: float,
    *,
    z_waist: float | None = None,
    n_glass: float = 1.5,
    n_air: float = 1.0,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
    lateral_supersampling: int = 1,
    integration_method: str = "midpoint",
    phase_scale: float = 1.0,
) -> dict[str, NDArray[np.float64] | NDArray[np.complex128]]:
    """Project one axisymmetric air-filled TGV into a 2D transmission.

    Arrays use ``(ny, nx)`` order. A tuple ``dx`` is interpreted as
    ``(dy, dx)`` in meters, while ``center_xy_m`` is ``(x, y)``. The returned
    path length, relative OPD, and unwrapped phase are float64; the effective
    transmission is complex128. ``integration_method`` is either midpoint
    slice integration or the analytic piecewise-linear path.
    """

    if len(shape) != 2 or min(int(shape[0]), int(shape[1])) <= 0:
        msg = "shape must be a positive (ny, nx) tuple."
        raise ValueError(msg)
    dy_m, dx_m = _normalize_dx(dx)
    if not np.isfinite(dy_m) or not np.isfinite(dx_m) or min(dy_m, dx_m) <= 0:
        msg = "dx values must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        msg = "wavelength must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(dz) or dz <= 0.0:
        msg = "dz must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(n_glass) or not np.isfinite(n_air):
        msg = "refractive indices must be finite."
        raise ValueError(msg)
    if not np.isfinite(phase_scale):
        msg = "phase_scale must be finite."
        raise ValueError(msg)
    if not isinstance(lateral_supersampling, int) or lateral_supersampling <= 0:
        msg = "lateral_supersampling must be a positive integer."
        raise ValueError(msg)
    if len(center_xy_m) != 2 or not np.all(np.isfinite(center_xy_m)):
        msg = "center_xy_m must be a finite (x, y) tuple in meters."
        raise ValueError(msg)
    waist_depth = validate_tgv_geometry(
        thickness, d_top, d_waist, d_bottom, z_waist
    )

    z_m, slice_width_m = midpoint_z_grid(thickness, dz)
    diameter_z_m = diameter_profile(
        z_m, thickness, d_top, d_waist, d_bottom, waist_depth
    )
    ny, nx = int(shape[0]), int(shape[1])
    super_shape = (ny * lateral_supersampling, nx * lateral_supersampling)
    super_dx = (dy_m / lateral_supersampling, dx_m / lateral_supersampling)
    x_grid, y_grid = coordinate_grid(super_shape, super_dx)
    radius_m = np.sqrt(
        (x_grid - float(center_xy_m[0])) ** 2
        + (y_grid - float(center_xy_m[1])) ** 2
    )

    if integration_method == "midpoint":
        path_super = _midpoint_path_from_radius(
            radius_m, z_m, slice_width_m, diameter_z_m
        )
    elif integration_method == "analytic":
        path_super = analytic_air_path_length(
            radius_m,
            thickness,
            d_top,
            d_waist,
            d_bottom,
            waist_depth,
        )
    else:
        msg = "integration_method must be 'midpoint' or 'analytic'."
        raise ValueError(msg)

    fill_path_length_m = _average_subpixels(
        path_super, (ny, nx), lateral_supersampling
    )
    opd_relative_m = (n_air - n_glass) * fill_path_length_m
    phase_unwrapped_rad = (
        2.0 * np.pi / wavelength * opd_relative_m * phase_scale
    )
    A_effective_true = np.exp(1j * phase_unwrapped_rad).astype(np.complex128)
    return {
        "z_m": z_m.astype(np.float64, copy=False),
        "diameter_z_m": diameter_z_m.astype(np.float64, copy=False),
        "fill_path_length_m": fill_path_length_m.astype(np.float64, copy=False),
        "opd_relative_m": opd_relative_m.astype(np.float64, copy=False),
        "phase_unwrapped_rad": phase_unwrapped_rad.astype(np.float64, copy=False),
        "A_effective_true": A_effective_true,
    }
