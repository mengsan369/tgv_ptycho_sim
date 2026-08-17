from __future__ import annotations

import numpy as np

from tgv_ptycho.inverse.metrics import align_global_phase, complex_relative_error


def test_align_global_phase_removes_constant_phase_offset() -> None:
    rng = np.random.default_rng(4)
    true = rng.random((8, 8)) * np.exp(1j * rng.random((8, 8)))
    rec = true * np.exp(-1j * 0.73)

    aligned, phase_offset = align_global_phase(rec, true)

    assert np.isclose(phase_offset, 0.73)
    assert complex_relative_error(aligned, true) < 1e-12
