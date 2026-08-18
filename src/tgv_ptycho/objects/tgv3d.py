"""Axisymmetric 3D TGV refractive-index phantoms."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from tgv_ptycho.objects.tgv_geometry import (
    diameter_profile as _shared_diameter_profile,
)
from tgv_ptycho.objects.tgv_geometry import midpoint_z_grid, validate_tgv_geometry
from tgv_ptycho.optics.angular_spectrum import _normalize_dx
from tgv_ptycho.optics.fields import coordinate_grid


def _diameter_profile(
    z: NDArray[np.float64],
    thickness: float,
    d_top: float,
    d_waist: float,
    d_bottom: float,
    z_waist: float,
) -> NDArray[np.float64]:
    """Compatibility wrapper around the shared public diameter profile."""

    return _shared_diameter_profile(
        z, thickness, d_top, d_waist, d_bottom, z_waist
    )


def make_tgv_refractive_index_volume(
    shape_xyz: tuple[int, int, int],
    dx: float | tuple[float, float],
    dz: float,
    thickness: float,
    d_top: float,
    d_waist: float,
    d_bottom: float,
    z_waist: float | None = None,
    n_glass: float = 1.5,
    n_air: float = 1.0,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Create an axisymmetric TGV refractive-index volume.

    Parameters
    ----------
    shape_xyz:
        Tuple `(nz, ny, nx)`. The output has this shape.
    dx:
        Lateral sampling in meters.
    dz:
        Slice spacing in meters.
    thickness:
        Physical sample thickness in meters.
    d_top, d_waist, d_bottom:
        Top, minimum waist, and bottom diameters in meters.
    z_waist:
        Waist depth in meters. Defaults to the middle of the thickness.
    n_glass, n_air:
        Refractive indices for glass matrix and air-filled via.
    center_xy_m:
        Via-axis position ``(x, y)`` in meters.

    Returns
    -------
    n_volume:
        Refractive-index array with shape `(nz, ny, nx)`.
    metadata:
        Dictionary containing the true diameter profile and TGV parameters.
    """

    if len(shape_xyz) != 3:
        msg = "shape_xyz must be (nz, ny, nx)."
        raise ValueError(msg)
    try:
        shape_values = np.asarray(shape_xyz, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "shape_xyz entries must be finite integers."
        raise ValueError(msg) from error
    if (
        not np.all(np.isfinite(shape_values))
        or np.any(shape_values != np.floor(shape_values))
        or any(isinstance(value, bool) for value in shape_xyz)
    ):
        msg = "shape_xyz entries must be integers."
        raise ValueError(msg)

    nz, ny, nx = (int(v) for v in shape_xyz)
    if min(nz, ny, nx) <= 0:
        msg = "shape_xyz entries must be positive."
        raise ValueError(msg)
    dy_m, dx_m = _normalize_dx(dx)
    if not np.all(np.isfinite([dy_m, dx_m])) or min(dy_m, dx_m) <= 0.0:
        msg = "dx values must be finite and positive."
        raise ValueError(msg)
    if not np.all(np.isfinite([n_glass, n_air])) or min(n_glass, n_air) <= 0.0:
        msg = "refractive indices must be finite and positive."
        raise ValueError(msg)
    if len(center_xy_m) != 2 or not np.all(np.isfinite(center_xy_m)):
        msg = "center_xy_m must be a finite (x, y) tuple in meters."
        raise ValueError(msg)

    waist_depth = validate_tgv_geometry(
        thickness, d_top, d_waist, d_bottom, z_waist
    )
    z, slice_widths = midpoint_z_grid(thickness, dz)
    if nz != len(z):
        msg = (
            "shape_xyz[0] must equal the slice count implied by thickness and dz "
            f"({len(z)})."
        )
        raise ValueError(msg)
    diameter_z = _diameter_profile(
        z, thickness, d_top, d_waist, d_bottom, waist_depth
    )

    normalized_dx = (dy_m, dx_m)
    x_grid, y_grid = coordinate_grid((ny, nx), normalized_dx)
    radius_grid = np.sqrt(
        (x_grid - float(center_xy_m[0])) ** 2
        + (y_grid - float(center_xy_m[1])) ** 2
    )

    n_volume = np.full((nz, ny, nx), n_glass, dtype=np.float64)
    for iz, diameter in enumerate(diameter_z):
        n_volume[iz, radius_grid <= diameter / 2.0] = n_air

    metadata: dict[str, Any] = {
        "shape_xyz": [nz, ny, nx],
        "dx_m": (
            [dy_m, dx_m] if isinstance(dx, tuple) else float(dx_m)
        ),
        "dz_m": dz,
        "slice_thickness_m": slice_widths.tolist(),
        "thickness_m": thickness,
        "d_top_m": d_top,
        "d_waist_m": d_waist,
        "d_bottom_m": d_bottom,
        "z_waist_m": waist_depth,
        "n_glass": n_glass,
        "n_air": n_air,
        "center_xy_m": [float(center_xy_m[0]), float(center_xy_m[1])],
        "z_m": z.tolist(),
        "diameter_z_m": diameter_z.tolist(),
    }
    return n_volume, metadata


def make_tgv_air_fraction_slice(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    diameter_m: float,
    subpixel_factor: int,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Return a positive midpoint estimate of TGV air area per pixel.

    ``subpixel_factor=q`` places ``q x q`` uniformly weighted midpoint nodes
    inside every lateral pixel.  ``q=1`` is exactly the historical
    voxel-center indicator, including points on the circular boundary.
    This is a cell-average representation of the analytic indicator, not a
    physical effective-medium model.
    """

    ny, nx = (int(value) for value in shape)
    if len(shape) != 2 or min(ny, nx) <= 0 or (ny, nx) != tuple(shape):
        msg = "shape must contain two positive integers."
        raise ValueError(msg)
    dy_m, dx_m = _normalize_dx(dx)
    if not np.all(np.isfinite([dy_m, dx_m])) or min(dy_m, dx_m) <= 0.0:
        msg = "dx values must be finite and positive."
        raise ValueError(msg)
    diameter = float(diameter_m)
    if not np.isfinite(diameter) or diameter <= 0.0:
        msg = "diameter_m must be finite and positive."
        raise ValueError(msg)
    if (
        isinstance(subpixel_factor, bool)
        or int(subpixel_factor) != subpixel_factor
        or int(subpixel_factor) <= 0
    ):
        msg = "subpixel_factor must be a positive integer."
        raise ValueError(msg)
    if len(center_xy_m) != 2 or not np.all(np.isfinite(center_xy_m)):
        msg = "center_xy_m must be a finite (x, y) tuple in meters."
        raise ValueError(msg)

    q = int(subpixel_factor)
    x_grid, y_grid = coordinate_grid((ny, nx), (dy_m, dx_m))
    offsets_y = ((np.arange(q, dtype=np.float64) + 0.5) / q - 0.5) * dy_m
    offsets_x = ((np.arange(q, dtype=np.float64) + 0.5) / q - 0.5) * dx_m
    radius_squared = (diameter / 2.0) ** 2
    counts = np.zeros((ny, nx), dtype=np.uint16)
    center_x, center_y = (float(value) for value in center_xy_m)
    for offset_y in offsets_y:
        y_squared = (y_grid + offset_y - center_y) ** 2
        for offset_x in offsets_x:
            counts += (
                (x_grid + offset_x - center_x) ** 2 + y_squared
                <= radius_squared
            )
    return counts.astype(np.float64) / float(q * q)


def make_tgv_air_fraction_slice_chord_quadrature(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    diameter_m: float,
    quadrature_order: int = 64,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Return a conservative circular air fraction from chord integration.

    Pixels wholly inside or outside the analytic disk are assigned exactly one
    or zero.  Only boundary pixels are integrated: a Gauss--Legendre rule in
    ``x`` integrates the exact vertical disk chord clipped to the pixel's
    ``y`` interval.  The rule estimates the analytic indicator's cell average;
    it is not a material effective-medium law.
    """

    ny, nx = (int(value) for value in shape)
    if len(shape) != 2 or min(ny, nx) <= 0 or (ny, nx) != tuple(shape):
        msg = "shape must contain two positive integers."
        raise ValueError(msg)
    dy_m, dx_m = _normalize_dx(dx)
    if not np.all(np.isfinite([dy_m, dx_m])) or min(dy_m, dx_m) <= 0.0:
        msg = "dx values must be finite and positive."
        raise ValueError(msg)
    diameter = float(diameter_m)
    if not np.isfinite(diameter) or diameter <= 0.0:
        msg = "diameter_m must be finite and positive."
        raise ValueError(msg)
    if (
        isinstance(quadrature_order, bool)
        or int(quadrature_order) != quadrature_order
        or int(quadrature_order) <= 0
    ):
        msg = "quadrature_order must be a positive integer."
        raise ValueError(msg)
    if len(center_xy_m) != 2 or not np.all(np.isfinite(center_xy_m)):
        msg = "center_xy_m must be a finite (x, y) tuple in meters."
        raise ValueError(msg)

    order = int(quadrature_order)
    center_x, center_y = (float(value) for value in center_xy_m)
    x_grid, y_grid = coordinate_grid((ny, nx), (dy_m, dx_m))
    x_relative = x_grid - center_x
    y_relative = y_grid - center_y
    half_x = 0.5 * dx_m
    half_y = 0.5 * dy_m
    radius_squared = (0.5 * diameter) ** 2

    closest_x = np.maximum(np.abs(x_relative) - half_x, 0.0)
    closest_y = np.maximum(np.abs(y_relative) - half_y, 0.0)
    closest_squared = closest_x**2 + closest_y**2
    farthest_squared = (
        (np.abs(x_relative) + half_x) ** 2
        + (np.abs(y_relative) + half_y) ** 2
    )
    full = farthest_squared <= radius_squared
    empty = closest_squared >= radius_squared
    boundary = ~(full | empty)

    fraction = np.zeros((ny, nx), dtype=np.float64)
    fraction[full] = 1.0
    if np.any(boundary):
        nodes, weights = roots_legendre(order)
        boundary_x = x_relative[boundary]
        boundary_y = y_relative[boundary]
        pixel_x_lower = boundary_x - half_x
        pixel_x_upper = boundary_x + half_x
        pixel_y_lower = boundary_y - half_y
        pixel_y_upper = boundary_y + half_y
        radius = 0.5 * diameter

        def edge_roots(edge_y: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.sqrt(np.maximum(radius_squared - edge_y**2, 0.0))

        lower_roots = edge_roots(pixel_y_lower)
        upper_roots = edge_roots(pixel_y_upper)
        candidates = np.stack(
            [
                pixel_x_lower,
                pixel_x_upper,
                np.full_like(pixel_x_lower, -radius),
                np.full_like(pixel_x_lower, radius),
                -lower_roots,
                lower_roots,
                -upper_roots,
                upper_roots,
            ],
            axis=1,
        )
        candidates = np.clip(
            candidates,
            pixel_x_lower[:, None],
            pixel_x_upper[:, None],
        )
        breakpoints = np.sort(candidates, axis=1)
        segment_lower = breakpoints[:, :-1]
        segment_upper = breakpoints[:, 1:]
        segment_half = 0.5 * (segment_upper - segment_lower)
        segment_midpoint = 0.5 * (segment_upper + segment_lower)
        sample_x = (
            segment_midpoint[:, :, None]
            + segment_half[:, :, None] * nodes[None, None, :]
        )
        half_chord = np.sqrt(
            np.maximum(radius_squared - sample_x**2, 0.0)
        )
        overlap_lower = np.maximum(
            pixel_y_lower[:, None, None], -half_chord
        )
        overlap_upper = np.minimum(
            pixel_y_upper[:, None, None], half_chord
        )
        overlap = np.maximum(overlap_upper - overlap_lower, 0.0)
        segment_integrals = segment_half * np.sum(
            overlap * weights[None, None, :], axis=2
        )
        boundary_area = np.sum(segment_integrals, axis=1)
        fraction[boundary] = boundary_area / (dx_m * dy_m)

    return np.clip(fraction, 0.0, 1.0)
