"""Synthetic objects and TGV phantoms."""

from tgv_ptycho.objects.sample_b import (
    make_random_amp_phase_object,
    make_random_phase_object,
)
from tgv_ptycho.objects.tgv2d import make_tgv_effective_phase_2d, make_thin_phase_disk
from tgv_ptycho.objects.tgv3d import (
    make_tgv_air_fraction_slice,
    make_tgv_air_fraction_slice_chord_quadrature,
    make_tgv_refractive_index_volume,
)

__all__ = [
    "make_thin_phase_disk",
    "make_tgv_effective_phase_2d",
    "make_tgv_refractive_index_volume",
    "make_tgv_air_fraction_slice",
    "make_tgv_air_fraction_slice_chord_quadrature",
    "make_random_phase_object",
    "make_random_amp_phase_object",
]
