"""Shared parameterized geometry for axisymmetric single-TGV models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_tgv_geometry(
    thickness: float,
    d_top: float,
    d_waist: float,
    d_bottom: float,
    z_waist: float | None = None,
) -> float:
    """Validate one air-filled TGV in glass and return its waist depth.

    All lengths are in meters. The supported baseline is a single,
    axisymmetric via whose waist is no wider than either surface opening.
    """

    values = np.asarray([thickness, d_top, d_waist, d_bottom], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        msg = "thickness and diameters must be finite."
        raise ValueError(msg)
    if thickness <= 0.0:
        msg = "thickness must be positive."
        raise ValueError(msg)
    if min(d_top, d_waist, d_bottom) <= 0.0:
        msg = "diameters must be positive."
        raise ValueError(msg)
    if d_waist > min(d_top, d_bottom):
        msg = "d_waist must not exceed d_top or d_bottom."
        raise ValueError(msg)

    waist_depth = thickness / 2.0 if z_waist is None else float(z_waist)
    if not np.isfinite(waist_depth) or not 0.0 < waist_depth < thickness:
        msg = "z_waist must be finite and inside the sample thickness."
        raise ValueError(msg)
    return waist_depth


def diameter_profile(
    z_m: NDArray[np.floating] | list[float] | tuple[float, ...],
    thickness: float,
    d_top: float,
    d_waist: float,
    d_bottom: float,
    z_waist: float | None = None,
) -> NDArray[np.float64]:
    """Evaluate the shared piecewise-linear diameter profile ``D(z)``.

    ``z_m`` and every geometry parameter use meters. The returned float64
    array has the same shape as ``z_m`` and covers the closed interval
    ``0 <= z <= thickness``.
    """

    waist_depth = validate_tgv_geometry(
        thickness, d_top, d_waist, d_bottom, z_waist
    )
    z = np.asarray(z_m, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        msg = "z_m must contain only finite values."
        raise ValueError(msg)
    tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(thickness))
    if np.any(z < -tolerance) or np.any(z > thickness + tolerance):
        msg = "z_m must lie in the closed interval [0, thickness]."
        raise ValueError(msg)
    z = np.clip(z, 0.0, thickness)

    before = z <= waist_depth
    diameter = np.empty(z.shape, dtype=np.float64)
    diameter[before] = d_top + (d_waist - d_top) * (
        z[before] / waist_depth
    )
    diameter[~before] = d_waist + (d_bottom - d_waist) * (
        (z[~before] - waist_depth) / (thickness - waist_depth)
    )
    return diameter


def midpoint_z_grid(
    thickness: float,
    dz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return midpoint coordinates and exact slice widths in meters.

    ``dz`` is a target maximum width.  A shorter final slice is used when
    necessary, and ratios numerically indistinguishable from an integer do not
    create a zero-width remainder slice.
    """

    if not np.isfinite(thickness) or thickness <= 0.0:
        msg = "thickness must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(dz) or dz <= 0.0:
        msg = "dz must be finite and positive."
        raise ValueError(msg)
    ratio = thickness / dz
    nearest_integer = int(np.rint(ratio))
    ratio_tolerance = (
        32.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    )
    if nearest_integer >= 1 and abs(ratio - nearest_integer) <= ratio_tolerance:
        num_slices = nearest_integer
    else:
        num_slices = int(np.ceil(ratio))

    edges = np.empty(num_slices + 1, dtype=np.float64)
    edges[:-1] = np.arange(num_slices, dtype=np.float64) * dz
    edges[-1] = thickness
    widths = np.diff(edges)
    if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
        msg = "slice construction produced non-positive or non-finite widths."
        raise RuntimeError(msg)
    widths[-1] += thickness - float(np.sum(widths, dtype=np.float64))
    centers = edges[:-1] + 0.5 * widths
    return centers.astype(np.float64), widths.astype(np.float64)


def analytic_air_path_length(
    radius_m: NDArray[np.floating] | list[float] | tuple[float, ...],
    thickness: float,
    d_top: float,
    d_waist: float,
    d_bottom: float,
    z_waist: float | None = None,
) -> NDArray[np.float64]:
    """Return the exact axial air path for the piecewise-linear TGV.

    The input is radial distance from the via axis in meters. The result is
    the total measure of depths for which ``radius_m <= D(z) / 2``.
    """

    waist_depth = validate_tgv_geometry(
        thickness, d_top, d_waist, d_bottom, z_waist
    )
    radius = np.asarray(radius_m, dtype=np.float64)
    if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
        msg = "radius_m must contain finite, non-negative values."
        raise ValueError(msg)

    r_waist = 0.5 * d_waist

    def segment_path(
        outer_radius: float, segment_length: float
    ) -> NDArray[np.float64]:
        if outer_radius == r_waist:
            return np.where(radius <= r_waist, segment_length, 0.0)
        fraction = (outer_radius - radius) / (outer_radius - r_waist)
        return segment_length * np.clip(fraction, 0.0, 1.0)

    top_path = segment_path(0.5 * d_top, waist_depth)
    bottom_path = segment_path(0.5 * d_bottom, thickness - waist_depth)
    return np.asarray(top_path + bottom_path, dtype=np.float64)
