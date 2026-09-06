from __future__ import annotations

import numpy as np
import pytest

from tgv_ptycho.forward.scan_windows import (
    extract_scan_window,
    make_scan_window_plan,
    minimum_large_canvas_shape,
    scan_window_adjoint_relative_error,
    scatter_add_scan_windows,
)


def test_rectangular_tuple_dx_and_xy_sign_convention() -> None:
    positions = np.asarray([[0.0, 0.0], [3.0, -4.0]])
    plan = make_scan_window_plan((11, 14), (5, 6), positions, (2.0, 1.0))
    canvas = np.arange(11 * 14).reshape(11, 14)
    center = extract_scan_window(canvas, plan, 0)
    shifted = extract_scan_window(canvas, plan, 1)
    assert plan.shifts_xy_px.tolist() == [[0, 0], [3, -2]]
    assert center.shape == (5, 6)
    assert shifted[0, 0] == canvas[5, 1]
    assert plan.margins_px[1].tolist() == [1, 7, 5, 1]


@pytest.mark.parametrize(
    ("large_shape", "window_shape", "expected_offset"),
    [((10, 12), (4, 6), (0.0, 0.0)), ((11, 12), (4, 5), (-0.5, -0.5))],
)
def test_odd_even_center_alignment_is_explicit(
    large_shape: tuple[int, int],
    window_shape: tuple[int, int],
    expected_offset: tuple[float, float],
) -> None:
    plan = make_scan_window_plan(
        large_shape, window_shape, np.zeros((1, 2)), (1.0, 1.0)
    )
    assert plan.center_alignment_offset_xy_m == expected_offset


def test_out_of_bounds_and_noninteger_positions_fail_fast() -> None:
    with pytest.raises(ValueError, match="exceed"):
        make_scan_window_plan((9, 9), (7, 7), np.asarray([[2.0, 0.0]]), 1.0)
    with pytest.raises(ValueError, match="integer-pixel"):
        make_scan_window_plan((9, 9), (3, 3), np.asarray([[0.25, 0.0]]), 1.0)


def test_extraction_scatter_are_adjoint_and_coverage_is_exact() -> None:
    positions = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    plan = make_scan_window_plan((7, 9), (3, 5), positions, 1.0)
    assert scan_window_adjoint_relative_error(plan, seed=31) <= 1.0e-12
    assert plan.minimum_margin_px == 1
    assert np.count_nonzero(plan.coverage_map) == 21
    assert np.all(plan.fully_inside)
    patches = np.ones((3, 3, 5), dtype=np.float64)
    scattered = scatter_add_scan_windows(patches, plan)
    assert np.array_equal(scattered, plan.coverage_map)
    assert np.array_equal(plan.visited_region, plan.coverage_map > 0)
    assert np.array_equal(plan.unvisited_region, plan.coverage_map == 0)


def test_minimum_canvas_applies_span_guard_alignment_and_parity() -> None:
    positions = np.asarray([[-24.5e-6, -24.5e-6], [24.5e-6, 24.5e-6]])
    shape = minimum_large_canvas_shape(
        (384, 384),
        positions,
        0.25e-6,
        guard_m_per_side=2.0e-6,
        alignment_px_yx=8,
    )
    assert shape == (608, 608)
    plan = make_scan_window_plan(shape, (384, 384), positions, 0.25e-6)
    assert plan.minimum_margin_m == 3.5e-6
