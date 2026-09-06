"""Nonperiodic scan-window planning for one enlarged physical object canvas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _shape(value: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2 or min(value) <= 0:
        raise ValueError(f"{name} must be a positive (ny, nx) tuple.")
    return int(value[0]), int(value[1])


def _dy_dx(dx: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(dx, tuple):
        if len(dx) != 2:
            raise ValueError("dx tuple must be (dy, dx) in meters.")
        dy_m, dx_m = float(dx[0]), float(dx[1])
    else:
        dy_m = dx_m = float(dx)
    if not np.isfinite(dy_m) or not np.isfinite(dx_m) or min(dy_m, dx_m) <= 0:
        raise ValueError("dx entries must be finite and positive.")
    return dy_m, dx_m


@dataclass(frozen=True)
class ScanWindowPlan:
    """Integer slices mapping physical `(x, y)` positions into one large canvas.

    ``margins_px`` uses column order ``(left, right, top, bottom)``.  Shape
    tuples and array axes use ``(ny, nx)``; ``dx`` tuples use ``(dy, dx)``.
    Positive sample displacement follows the repository's historical shift
    convention: the source patch is sampled at decreasing canvas coordinate.
    """

    large_shape: tuple[int, int]
    window_shape: tuple[int, int]
    dx_yx_m: tuple[float, float]
    scan_positions_m: NDArray[np.float64]
    shifts_xy_px: NDArray[np.int64]
    starts_yx: NDArray[np.int64]
    stops_yx: NDArray[np.int64]
    margins_px: NDArray[np.int64]
    center_alignment_offset_xy_m: tuple[float, float]
    coverage_map: NDArray[np.int32]

    @property
    def minimum_margin_px(self) -> int:
        """Smallest distance, in whole pixels, from any patch to any edge."""

        return int(np.min(self.margins_px))

    @property
    def minimum_margin_m(self) -> float:
        """Smallest physical edge margin, accounting for anisotropic dx."""

        dy_m, dx_m = self.dx_yx_m
        scales = np.asarray([dx_m, dx_m, dy_m, dy_m], dtype=np.float64)
        return float(np.min(self.margins_px * scales[None, :]))

    @property
    def fully_inside(self) -> NDArray[np.bool_]:
        """Per-position flag that every patch pixel lies inside the canvas."""

        return np.all(self.margins_px >= 0, axis=1)

    @property
    def visited_region(self) -> NDArray[np.bool_]:
        """Boolean map of canvas pixels visited by at least one patch."""

        return self.coverage_map > 0

    @property
    def unvisited_region(self) -> NDArray[np.bool_]:
        """Boolean map of canvas pixels never visited by a patch."""

        return self.coverage_map == 0


def make_scan_window_plan(
    large_shape: tuple[int, int],
    window_shape: tuple[int, int],
    scan_positions: NDArray[np.floating],
    dx: float | tuple[float, float],
    *,
    integer_tolerance_px: float = 1.0e-9,
) -> ScanWindowPlan:
    """Precompute fail-fast nonperiodic slices for all physical scan positions.

    Positive x/y denotes positive sample displacement, matching
    :func:`shift_field_integer_pixels`; the corresponding unshifted source
    patch starts at decreasing x/y canvas indices.  No wrapping, fill,
    cropping, or per-position object regeneration is performed.
    """

    large_ny, large_nx = _shape(large_shape, "large_shape")
    win_ny, win_nx = _shape(window_shape, "window_shape")
    if win_ny > large_ny or win_nx > large_nx:
        raise ValueError("window_shape must fit inside large_shape.")
    dy_m, dx_m = _dy_dx(dx)
    positions = np.asarray(scan_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError("scan_positions must have nonempty shape (n, 2).")
    if not np.all(np.isfinite(positions)):
        raise ValueError("scan_positions must be finite.")
    if not np.isfinite(integer_tolerance_px) or integer_tolerance_px < 0:
        raise ValueError("integer_tolerance_px must be finite and non-negative.")

    shifts_float = np.column_stack([positions[:, 0] / dx_m, positions[:, 1] / dy_m])
    shifts = np.rint(shifts_float).astype(np.int64)
    if np.max(np.abs(shifts_float - shifts)) > integer_tolerance_px:
        raise ValueError("scan_positions must correspond to integer-pixel shifts.")

    base_y = (large_ny - win_ny) // 2
    base_x = (large_nx - win_nx) // 2
    starts = np.column_stack([base_y - shifts[:, 1], base_x - shifts[:, 0]])
    stops = starts + np.asarray([win_ny, win_nx], dtype=np.int64)
    margins = np.column_stack(
        [
            starts[:, 1],
            large_nx - stops[:, 1],
            starts[:, 0],
            large_ny - stops[:, 0],
        ]
    ).astype(np.int64)
    if np.any(margins < 0):
        bad = np.flatnonzero(np.any(margins < 0, axis=1)).tolist()
        raise ValueError(f"scan windows exceed the physical B canvas at indices {bad}.")

    coverage = np.zeros((large_ny, large_nx), dtype=np.int32)
    for (start_y, start_x), (stop_y, stop_x) in zip(starts, stops, strict=True):
        coverage[start_y:stop_y, start_x:stop_x] += 1

    actual_center_x = base_x + (win_nx - 1) / 2.0
    actual_center_y = base_y + (win_ny - 1) / 2.0
    canvas_center_x = (large_nx - 1) / 2.0
    canvas_center_y = (large_ny - 1) / 2.0
    offset = (
        float((actual_center_x - canvas_center_x) * dx_m),
        float((actual_center_y - canvas_center_y) * dy_m),
    )
    return ScanWindowPlan(
        large_shape=(large_ny, large_nx),
        window_shape=(win_ny, win_nx),
        dx_yx_m=(dy_m, dx_m),
        scan_positions_m=positions.copy(),
        shifts_xy_px=shifts,
        starts_yx=starts.astype(np.int64),
        stops_yx=stops.astype(np.int64),
        margins_px=margins,
        center_alignment_offset_xy_m=offset,
        coverage_map=coverage,
    )


def minimum_large_canvas_shape(
    window_shape: tuple[int, int],
    scan_positions: NDArray[np.floating],
    dx: float | tuple[float, float],
    *,
    guard_m_per_side: float | tuple[float, float] = 0.0,
    alignment_px_yx: int | tuple[int, int] = 1,
    preserve_centered_parity: bool = True,
) -> tuple[int, int]:
    """Return the smallest aligned canvas that contains every scan window.

    The continuous lower bound on each axis is the window extent plus scan
    span plus two guards.  Integer rounding, lattice alignment, asymmetric
    extrema, and centered odd/even parity are then resolved by validating the
    same slices used by :func:`make_scan_window_plan`.
    """

    win_ny, win_nx = _shape(window_shape, "window_shape")
    dy_m, dx_m = _dy_dx(dx)
    positions = np.asarray(scan_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError("scan_positions must have nonempty shape (n, 2).")
    if isinstance(guard_m_per_side, tuple):
        if len(guard_m_per_side) != 2:
            raise ValueError("guard tuple must be (guard_y, guard_x) in meters.")
        guard_y_m, guard_x_m = map(float, guard_m_per_side)
    else:
        guard_y_m = guard_x_m = float(guard_m_per_side)
    if min(guard_y_m, guard_x_m) < 0:
        raise ValueError("guard_m_per_side must be non-negative.")
    if isinstance(alignment_px_yx, tuple):
        if len(alignment_px_yx) != 2:
            raise ValueError("alignment tuple must be (y, x) pixels.")
        align_y, align_x = map(int, alignment_px_yx)
    else:
        align_y = align_x = int(alignment_px_yx)
    if min(align_y, align_x) <= 0:
        raise ValueError("alignment_px_yx entries must be positive.")

    shifts_float = np.column_stack([positions[:, 0] / dx_m, positions[:, 1] / dy_m])
    shifts = np.rint(shifts_float).astype(np.int64)
    if np.max(np.abs(shifts_float - shifts)) > 1.0e-9:
        raise ValueError("scan_positions must correspond to integer-pixel shifts.")
    guard_y_px = int(np.ceil(guard_y_m / dy_m - 1.0e-12))
    guard_x_px = int(np.ceil(guard_x_m / dx_m - 1.0e-12))
    spans_xy = np.ptp(shifts, axis=0)

    def aligned_axis(
        window: int,
        span: int,
        guard: int,
        alignment: int,
        axis: int,
    ) -> int:
        candidate = int(window + span + 2 * guard)
        candidate = int(np.ceil(candidate / alignment) * alignment)
        while True:
            parity_period = 2 * alignment
            if preserve_centered_parity and (candidate - window) % parity_period:
                candidate += alignment
                continue
            base = (candidate - window) // 2
            starts = base - shifts[:, axis]
            stops = starts + window
            if np.min(starts) >= guard and np.min(candidate - stops) >= guard:
                return candidate
            candidate += alignment

    large_ny = aligned_axis(win_ny, int(spans_xy[1]), guard_y_px, align_y, 1)
    large_nx = aligned_axis(win_nx, int(spans_xy[0]), guard_x_px, align_x, 0)
    return large_ny, large_nx


def extract_scan_window(
    canvas: NDArray[np.generic], plan: ScanWindowPlan, index: int
) -> NDArray[np.generic]:
    """Return a view of one planned patch from the same physical canvas."""

    values = np.asarray(canvas)
    if values.shape != plan.large_shape:
        raise ValueError("canvas shape must equal plan.large_shape.")
    position_index = int(index)
    if position_index < 0 or position_index >= plan.starts_yx.shape[0]:
        raise IndexError("scan-window index is out of range.")
    start_y, start_x = plan.starts_yx[position_index]
    stop_y, stop_x = plan.stops_yx[position_index]
    return values[start_y:stop_y, start_x:stop_x]


def extract_scan_windows(
    canvas: NDArray[np.generic], plan: ScanWindowPlan
) -> NDArray[np.generic]:
    """Materialize all planned patches with shape ``(n, window_ny, window_nx)``."""

    return np.stack(
        [
            extract_scan_window(canvas, plan, index)
            for index in range(len(plan.starts_yx))
        ]
    )


def scatter_add_scan_windows(
    patches: NDArray[np.generic],
    plan: ScanWindowPlan,
    *,
    out: NDArray[np.generic] | None = None,
) -> NDArray[np.generic]:
    """Apply the Euclidean adjoint of stacked patch extraction."""

    values = np.asarray(patches)
    expected = (plan.starts_yx.shape[0], *plan.window_shape)
    if values.shape != expected:
        raise ValueError(f"patches must have shape {expected}.")
    if out is None:
        result = np.zeros(plan.large_shape, dtype=values.dtype)
    else:
        result = np.asarray(out)
        if result.shape != plan.large_shape:
            raise ValueError("out shape must equal plan.large_shape.")
    for patch, (start_y, start_x), (stop_y, stop_x) in zip(
        values, plan.starts_yx, plan.stops_yx, strict=True
    ):
        result[start_y:stop_y, start_x:stop_x] += patch
    return result


def scan_window_adjoint_relative_error(plan: ScanWindowPlan, *, seed: int = 0) -> float:
    """Run a memory-bounded complex dot-product test for extraction/scatter."""

    rng = np.random.default_rng(seed)
    canvas = rng.normal(size=plan.large_shape) + 1j * rng.normal(size=plan.large_shape)
    scattered = np.zeros(plan.large_shape, dtype=np.complex128)
    left = 0.0j
    for index in range(len(plan.starts_yx)):
        patch = rng.normal(size=plan.window_shape) + 1j * rng.normal(
            size=plan.window_shape
        )
        extracted = extract_scan_window(canvas, plan, index)
        left += np.sum(np.conj(extracted) * patch, dtype=np.complex128)
        start_y, start_x = plan.starts_yx[index]
        stop_y, stop_x = plan.stops_yx[index]
        scattered[start_y:stop_y, start_x:stop_x] += patch
    right = np.sum(np.conj(canvas) * scattered, dtype=np.complex128)
    return float(abs(left - right) / max(abs(left), abs(right), np.finfo(float).eps))


__all__ = [
    "ScanWindowPlan",
    "extract_scan_window",
    "extract_scan_windows",
    "make_scan_window_plan",
    "minimum_large_canvas_shape",
    "scan_window_adjoint_relative_error",
    "scatter_add_scan_windows",
]
