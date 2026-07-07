"""Refractive-index constants and simple material helpers."""

from __future__ import annotations

AIR_N = 1.0
BOROSILICATE_GLASS_N_532NM = 1.47
FUSED_SILICA_N_532NM = 1.46


def refractive_index_contrast(n_sample: float, n_ref: float) -> float:
    """Return `n_sample - n_ref`."""

    return float(n_sample) - float(n_ref)
