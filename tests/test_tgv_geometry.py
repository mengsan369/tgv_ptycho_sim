from __future__ import annotations

import numpy as np

from tgv_ptycho.objects.tgv2d import make_thin_phase_disk
from tgv_ptycho.objects.tgv3d import make_tgv_refractive_index_volume


def test_thin_phase_disk_shape_and_values() -> None:
    obj = make_thin_phase_disk((32, 32), dx=1e-6, diameter=10e-6, phase_shift=0.7)
    assert obj.shape == (32, 32)
    assert np.iscomplexobj(obj)
    assert np.isclose(np.abs(obj).max(), 1.0)


def test_tgv_refractive_index_volume_shape_and_metadata() -> None:
    volume, metadata = make_tgv_refractive_index_volume(
        shape_xyz=(8, 32, 32),
        dx=1e-6,
        dz=5e-6,
        thickness=40e-6,
        d_top=16e-6,
        d_waist=8e-6,
        d_bottom=14e-6,
        n_glass=1.5,
        n_air=1.0,
    )
    assert volume.shape == (8, 32, 32)
    assert metadata["d_waist_m"] == 8e-6
    assert np.isclose(volume.max(), 1.5)
    assert np.isclose(volume.min(), 1.0)
