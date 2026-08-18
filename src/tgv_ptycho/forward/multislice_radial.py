"""Axisymmetric radial split-step propagation for reference diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.hankel import QDHTPlan


def radial_multislice_contrast_propagate(
    plan: QDHTPlan,
    diameters_m: NDArray[np.floating],
    slice_widths_m: NDArray[np.floating],
    *,
    wavelength_m: float,
    n_glass: float,
    n_air: float,
    post_exit_air_distance_m: float,
    bandlimit: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Propagate a TGV in a background-carrier-normalized QDHT frame.

    The uniform background is represented analytically as one.  Only the
    boundary-decaying contrast is Hankel transformed, avoiding a spurious
    zero boundary condition on the plane-wave carrier.
    """

    diameters = np.asarray(diameters_m, dtype=np.float64)
    widths = np.asarray(slice_widths_m, dtype=np.float64)
    if (
        diameters.ndim != 1
        or widths.ndim != 1
        or diameters.shape != widths.shape
        or diameters.size == 0
    ):
        raise ValueError(
            "diameters_m and slice_widths_m must match nonempty 1D arrays."
        )
    if (
        not np.all(np.isfinite(diameters))
        or not np.all(np.isfinite(widths))
        or np.any(diameters <= 0.0)
        or np.any(widths <= 0.0)
    ):
        raise ValueError("diameters and slice widths must be finite and positive.")
    wavelength = float(wavelength_m)
    glass = float(n_glass)
    air = float(n_air)
    post_distance = float(post_exit_air_distance_m)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength_m must be finite and positive.")
    if not np.isfinite(glass) or not np.isfinite(air) or min(glass, air) <= 0.0:
        raise ValueError("refractive indices must be finite and positive.")
    if not np.isfinite(post_distance) or post_distance < 0.0:
        raise ValueError("post_exit_air_distance_m must be finite and non-negative.")
    if not isinstance(bandlimit, bool):
        raise TypeError("bandlimit must be a bool.")

    k0 = 2.0 * np.pi / wavelength
    radius = plan.radial_nodes_m
    contrast = np.zeros(plan.sample_count, dtype=np.complex128)
    contrast = plan.propagate_contrast(
        contrast,
        wavelength_m=wavelength,
        distance_m=0.5 * widths[0],
        refractive_index=glass,
        bandlimit=bandlimit,
    )
    maximum_outer_phase_error = 0.0
    for index, (diameter, width) in enumerate(
        zip(diameters, widths, strict=True)
    ):
        local_index = np.where(radius < 0.5 * diameter, air, glass)
        transmission = np.exp(1j * k0 * (local_index - glass) * width)
        maximum_outer_phase_error = max(
            maximum_outer_phase_error,
            float(abs(transmission[-1] - 1.0)),
        )
        contrast = (1.0 + contrast) * transmission - 1.0
        if index + 1 < widths.size:
            distance = 0.5 * (width + widths[index + 1])
            contrast = plan.propagate_contrast(
                contrast,
                wavelength_m=wavelength,
                distance_m=float(distance),
                refractive_index=glass,
                bandlimit=bandlimit,
            )
        if progress_callback is not None:
            progress_callback(index + 1, int(widths.size))
    contrast = plan.propagate_contrast(
        contrast,
        wavelength_m=wavelength,
        distance_m=0.5 * widths[-1],
        refractive_index=glass,
        bandlimit=bandlimit,
    )
    if post_distance > 0.0:
        contrast = plan.propagate_contrast(
            contrast,
            wavelength_m=wavelength,
            distance_m=post_distance,
            refractive_index=air,
            bandlimit=bandlimit,
        )
    normalized = 1.0 + contrast
    controls = {
        "sample_count": plan.sample_count,
        "radial_max_m": plan.radial_max_m,
        "slice_count": int(widths.size),
        "sample_thickness_m": float(np.sum(widths)),
        "post_exit_air_distance_m": post_distance,
        "background_representation": "analytic_unit_carrier",
        "propagated_quantity": "normalized_contrast",
        "interface_rule": "pointwise_analytic_radius_on_qdht_nodes",
        "maximum_outer_transmission_error": maximum_outer_phase_error,
        "outer_contrast_abs": float(abs(contrast[-1])),
        "contrast_l2": float(np.linalg.norm(contrast)),
        "all_finite": bool(np.all(np.isfinite(normalized))),
    }
    return np.asarray(normalized, dtype=np.complex128), controls


__all__ = ["radial_multislice_contrast_propagate"]
