from __future__ import annotations

import numpy as np

from tgv_ptycho.objects.sample_b import (
    make_physical_phase_cell_map,
    make_random_phase_object,
    rasterize_physical_phase_cells,
)


def test_physical_cells_are_deterministic_and_multiresolution_consistent() -> None:
    cells_a = make_physical_phase_cell_map(
        (6, 8), feature_size_m=2.0, phase_range_rad=0.8, seed=31
    )
    cells_b = make_physical_phase_cell_map(
        (6, 8), feature_size_m=2.0, phase_range_rad=0.8, seed=31
    )
    assert np.array_equal(cells_a.phase_rad, cells_b.phase_rad)
    coarse = rasterize_physical_phase_cells(cells_a, (6, 8), 2.0)
    fine = rasterize_physical_phase_cells(cells_a, (12, 16), 1.0)
    assert np.allclose(fine.reshape(6, 2, 8, 2).mean(axis=(1, 3)), coarse)


def test_physical_cells_do_not_tile_outside_their_extent() -> None:
    cells = make_physical_phase_cell_map(
        (4, 4), feature_size_m=1.0, phase_range_rad=0.8, seed=32
    )
    try:
        rasterize_physical_phase_cells(cells, (5, 5), 1.0)
    except ValueError as exc:
        assert "beyond" in str(exc)
    else:
        raise AssertionError("finite cell map unexpectedly tiled or filled")


def test_legacy_random_phase_api_and_default_behavior_remain_available() -> None:
    first = make_random_phase_object((7, 9), phase_range=0.8, seed=4)
    second = make_random_phase_object((7, 9), phase_range=0.8, seed=4)
    assert first.shape == (7, 9)
    assert first.dtype == np.complex128
    assert np.array_equal(first, second)
    assert np.allclose(np.abs(first), 1.0)
