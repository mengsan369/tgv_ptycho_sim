from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.scan import add_position_jitter, make_grid_scan


def test_grid_scan_shape_and_centering() -> None:
    positions = make_grid_scan(num_x=3, num_y=2, step=10e-6, center=True)
    assert positions.shape == (6, 2)
    np.testing.assert_allclose(positions.mean(axis=0), [0.0, 0.0], atol=1e-18)


def test_position_jitter_shape() -> None:
    positions = make_grid_scan(num_x=2, num_y=2, step=1e-6)
    jittered = add_position_jitter(positions, sigma=1e-9, seed=1)
    assert jittered.shape == positions.shape
