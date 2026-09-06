"""Synthetic scanning sample B generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhysicalPhaseCellMap:
    """One finite phase-cell realization on a physical `(y, x)` lattice.

    ``origin_xy_m`` is the lower-left physical edge of the complete cell map.
    The map is never tiled periodically by the helpers in this module.
    """

    phase_rad: NDArray[np.float64]
    feature_size_m: float
    origin_xy_m: tuple[float, float]
    phase_range_rad: float
    seed: int

    @property
    def physical_extent_xy_m(self) -> tuple[float, float]:
        """Return full cell-map extent as `(Lx, Ly)` in meters."""

        ny, nx = self.phase_rad.shape
        return nx * self.feature_size_m, ny * self.feature_size_m


def make_physical_phase_cell_map(
    shape_cells: tuple[int, int],
    *,
    feature_size_m: float,
    phase_range_rad: float,
    seed: int,
    origin_xy_m: tuple[float, float] | None = None,
) -> PhysicalPhaseCellMap:
    """Generate one deterministic finite, aperiodic phase-cell realization."""

    if len(shape_cells) != 2 or min(shape_cells) <= 0:
        raise ValueError("shape_cells must be a positive (ny, nx) tuple.")
    feature = float(feature_size_m)
    phase_range = float(phase_range_rad)
    if not np.isfinite(feature) or feature <= 0:
        raise ValueError("feature_size_m must be finite and positive.")
    if not np.isfinite(phase_range) or phase_range < 0:
        raise ValueError("phase_range_rad must be finite and non-negative.")
    ny, nx = int(shape_cells[0]), int(shape_cells[1])
    if origin_xy_m is None:
        origin = (-0.5 * nx * feature, -0.5 * ny * feature)
    else:
        if len(origin_xy_m) != 2 or not np.all(np.isfinite(origin_xy_m)):
            raise ValueError("origin_xy_m must contain finite (x, y) edges.")
        origin = (float(origin_xy_m[0]), float(origin_xy_m[1]))
    rng = np.random.default_rng(int(seed))
    phase = (2.0 * rng.random((ny, nx)) - 1.0) * phase_range
    return PhysicalPhaseCellMap(
        phase_rad=np.asarray(phase, dtype=np.float64),
        feature_size_m=feature,
        origin_xy_m=origin,
        phase_range_rad=phase_range,
        seed=int(seed),
    )


def rasterize_physical_phase_values(
    cell_map: PhysicalPhaseCellMap,
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    *,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Rasterize phase values from the same physical cells without tiling.

    Array coordinates are pixel centers, shape is `(ny, nx)`, ``dx`` tuples
    are `(dy, dx)`, and ``center_xy_m`` is the raster physical center.
    The requested raster must be fully covered by the finite cell map.
    """

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must be a positive (ny, nx) tuple.")
    if isinstance(dx, tuple):
        if len(dx) != 2:
            raise ValueError("dx tuple must be (dy, dx) in meters.")
        dy_m, dx_m = float(dx[0]), float(dx[1])
    else:
        dy_m = dx_m = float(dx)
    if not np.isfinite(dy_m) or not np.isfinite(dx_m) or min(dy_m, dx_m) <= 0:
        raise ValueError("dx entries must be finite and positive.")
    if len(center_xy_m) != 2 or not np.all(np.isfinite(center_xy_m)):
        raise ValueError("center_xy_m must contain finite (x, y) values.")
    ny, nx = int(shape[0]), int(shape[1])
    center_x, center_y = float(center_xy_m[0]), float(center_xy_m[1])
    x = center_x + (np.arange(nx) - (nx - 1) / 2.0) * dx_m
    y = center_y + (np.arange(ny) - (ny - 1) / 2.0) * dy_m
    origin_x, origin_y = cell_map.origin_xy_m
    feature = cell_map.feature_size_m
    ix = np.floor((x - origin_x) / feature).astype(np.int64)
    iy = np.floor((y - origin_y) / feature).astype(np.int64)
    cells_ny, cells_nx = cell_map.phase_rad.shape
    outside = (
        np.any(ix < 0)
        or np.any(ix >= cells_nx)
        or np.any(iy < 0)
        or np.any(iy >= cells_ny)
    )
    if outside:
        raise ValueError("requested raster extends beyond the finite phase-cell map.")
    return np.asarray(cell_map.phase_rad[iy[:, None], ix[None, :]], dtype=np.float64)


def rasterize_physical_phase_cells(
    cell_map: PhysicalPhaseCellMap,
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    *,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.complex128]:
    """Rasterize phase-only transmission without periodic tiling or fill."""

    phase = rasterize_physical_phase_values(
        cell_map,
        shape,
        dx,
        center_xy_m=center_xy_m,
    )
    return np.exp(1j * phase).astype(np.complex128)


def _make_random_feature_map(
    shape: tuple[int, int],
    feature_size_px: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Create a uniform random map with square, piecewise-constant features."""

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        msg = "shape must be a positive (ny, nx) tuple."
        raise ValueError(msg)
    if feature_size_px <= 0:
        msg = "feature_size_px must be a positive integer."
        raise ValueError(msg)

    ny, nx = int(shape[0]), int(shape[1])
    coarse_shape = (
        int(np.ceil(ny / feature_size_px)),
        int(np.ceil(nx / feature_size_px)),
    )
    coarse = rng.random(coarse_shape)
    expanded = np.repeat(
        np.repeat(coarse, feature_size_px, axis=0), feature_size_px, axis=1
    )
    return expanded[:ny, :nx].astype(np.float64, copy=False)


def make_random_phase_object(
    shape: tuple[int, int],
    phase_range: float = np.pi,
    seed: int | None = None,
    feature_size_px: int = 1,
) -> NDArray[np.complex128]:
    """Create a random phase-only transmission object.

    `feature_size_px=1` produces independent pixels. Larger values produce
    square random features, which are useful for controlled synthetic masks.
    """

    if phase_range < 0:
        msg = "phase_range must be non-negative."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    phase = (2.0 * _make_random_feature_map(shape, feature_size_px, rng) - 1.0) * (
        phase_range
    )
    return np.exp(1j * phase).astype(np.complex128)


def make_random_amp_phase_object(
    shape: tuple[int, int],
    amp_range: tuple[float, float] = (0.5, 1.0),
    phase_range: float = np.pi,
    seed: int | None = None,
    feature_size_px: int = 1,
) -> NDArray[np.complex128]:
    """Create a random amplitude-phase transmission object.

    Amplitude and phase share the requested feature size but are generated
    from independent random maps.
    """

    amp_min, amp_max = amp_range
    if amp_min < 0 or amp_max < amp_min:
        msg = "amp_range must satisfy 0 <= min <= max."
        raise ValueError(msg)
    if phase_range < 0:
        msg = "phase_range must be non-negative."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    amplitude = amp_min + (amp_max - amp_min) * _make_random_feature_map(
        shape, feature_size_px, rng
    )
    phase = (
        2.0 * _make_random_feature_map(shape, feature_size_px, rng) - 1.0
    ) * phase_range
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


__all__ = [
    "PhysicalPhaseCellMap",
    "make_physical_phase_cell_map",
    "make_random_amp_phase_object",
    "make_random_phase_object",
    "rasterize_physical_phase_cells",
    "rasterize_physical_phase_values",
]
