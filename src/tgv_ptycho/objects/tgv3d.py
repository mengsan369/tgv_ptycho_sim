"""Axisymmetric 3D TGV refractive-index phantoms."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.objects.tgv_geometry import (
    diameter_profile as _shared_diameter_profile,
)
from tgv_ptycho.objects.tgv_geometry import validate_tgv_geometry
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
    if dz <= 0:
        msg = "dz and thickness must be positive."
        raise ValueError(msg)

    nz, ny, nx = (int(v) for v in shape_xyz)
    if min(nz, ny, nx) <= 0:
        msg = "shape_xyz entries must be positive."
        raise ValueError(msg)
    z_waist = validate_tgv_geometry(
        thickness, d_top, d_waist, d_bottom, z_waist
    )

    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    z_clipped = np.clip(z, 0.0, thickness)
    diameter_z = _diameter_profile(
        z_clipped, thickness, d_top, d_waist, d_bottom, z_waist
    )

    x_grid, y_grid = coordinate_grid((ny, nx), dx)
    radius_grid = np.sqrt(x_grid**2 + y_grid**2)

    n_volume = np.full((nz, ny, nx), n_glass, dtype=np.float64)
    for iz, diameter in enumerate(diameter_z):
        n_volume[iz, radius_grid < diameter / 2.0] = n_air

    metadata: dict[str, Any] = {
        "shape_xyz": [nz, ny, nx],
        "dx_m": dx,
        "dz_m": dz,
        "thickness_m": thickness,
        "d_top_m": d_top,
        "d_waist_m": d_waist,
        "d_bottom_m": d_bottom,
        "z_waist_m": z_waist,
        "n_glass": n_glass,
        "n_air": n_air,
        "z_m": z_clipped.tolist(),
        "diameter_z_m": diameter_z.tolist(),
    }
    return n_volume, metadata
