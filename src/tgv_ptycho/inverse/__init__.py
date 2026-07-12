"""Inverse helpers for backpropagation, fitting, and metrics."""

from tgv_ptycho.inverse.backprop_A import (
    backpropagate_probe_to_A,
    recover_thin_phase_A,
    remove_reference_phase_plane,
)
from tgv_ptycho.inverse.metrics import (
    align_affine_phase_and_complex_gain,
    align_global_phase,
    amplitude_rmse,
    complex_relative_error,
    phase_rmse,
)
from tgv_ptycho.inverse.observability import compare_probe_sensitivity

__all__ = [
    "backpropagate_probe_to_A",
    "recover_thin_phase_A",
    "remove_reference_phase_plane",
    "compare_probe_sensitivity",
    "align_global_phase",
    "align_affine_phase_and_complex_gain",
    "complex_relative_error",
    "amplitude_rmse",
    "phase_rmse",
]
