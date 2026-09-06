"""Shared continuous-radial projected forward used by exp030 handoffs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

from tgv_ptycho.objects.tgv_geometry import (
    analytic_air_path_length,
    validate_tgv_geometry,
)
from tgv_ptycho.optics.fields import coordinate_grid


def build_exp030_radial_operator(
    source_radius_m: NDArray[np.floating],
    source_weights_m: NDArray[np.floating],
    output_radius_m: NDArray[np.floating],
    *,
    shape: tuple[int, int],
    dx_m: float,
    center_xy_m: tuple[float, float],
    wavelength_m: float,
    propagation_distance_m: float,
    medium_index: float,
    incident_amplitude: float,
) -> dict[str, Any]:
    """Build the frozen exp030 Fresnel--Hankel operator.

    The operator propagates only the compact radial ``T - 1`` perturbation.
    Its infinite plane-wave reference is retained analytically. Arrays use
    SI units, and the Cartesian output has ``(ny, nx)`` axis order.
    """

    radius = np.asarray(source_radius_m, dtype=np.float64)
    weights = np.asarray(source_weights_m, dtype=np.float64)
    output_radius = np.asarray(output_radius_m, dtype=np.float64)
    if (
        radius.ndim != 1
        or radius.size < 2
        or weights.shape != radius.shape
        or not np.all(np.isfinite(radius))
        or not np.all(np.isfinite(weights))
        or np.any(radius <= 0.0)
        or np.any(weights <= 0.0)
        or np.any(np.diff(radius) <= 0.0)
    ):
        raise ValueError("source radius and weights must be positive finite 1D arrays.")
    if (
        output_radius.ndim != 1
        or output_radius.size < 2
        or not np.all(np.isfinite(output_radius))
        or output_radius[0] != 0.0
        or np.any(np.diff(output_radius) <= 0.0)
    ):
        raise ValueError("output radius must start at zero and strictly increase.")
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must contain positive (ny, nx) entries.")
    scalars = np.asarray(
        [
            dx_m,
            wavelength_m,
            propagation_distance_m,
            medium_index,
            incident_amplitude,
            *center_xy_m,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)) or min(
        dx_m, wavelength_m, propagation_distance_m, medium_index
    ) <= 0.0:
        raise ValueError(
            "forward scalars must be finite and physical lengths positive."
        )

    medium_wavelength = wavelength_m / medium_index
    wavenumber = 2.0 * np.pi / medium_wavelength
    reference = complex(
        incident_amplitude * np.exp(1j * wavenumber * propagation_distance_m)
    )
    source_chirp = np.exp(
        1j * wavenumber * radius**2 / (2.0 * propagation_distance_m)
    ).astype(np.complex128)
    source_measure = (2.0 * np.pi * radius * weights).astype(np.float64)
    output_factor = (
        reference
        * np.exp(
            1j
            * wavenumber
            * output_radius**2
            / (2.0 * propagation_distance_m)
        )
        / (1j * medium_wavelength * propagation_distance_m)
    ).astype(np.complex128)
    kernel = j0(
        wavenumber
        * output_radius[:, None]
        * radius[None, :]
        / propagation_distance_m
    ).astype(np.float64, copy=False)

    x_grid, y_grid = coordinate_grid(shape, dx_m)
    cartesian_radius = np.sqrt(
        (x_grid - center_xy_m[0]) ** 2 + (y_grid - center_xy_m[1]) ** 2
    )
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, output_radius[-1])
    if float(np.max(cartesian_radius)) > float(output_radius[-1]) + tolerance:
        raise ValueError("radial output table does not cover the Cartesian grid.")
    flat_radius = np.minimum(cartesian_radius.ravel(), output_radius[-1])
    right = np.searchsorted(output_radius, flat_radius, side="right")
    right = np.clip(right, 1, output_radius.size - 1).astype(np.int64)
    left = right - 1
    alpha = (
        (flat_radius - output_radius[left])
        / (output_radius[right] - output_radius[left])
    ).astype(np.float64)
    return {
        "source_radius_m": radius,
        "source_weights_m": weights,
        "source_measure_m2": source_measure,
        "source_chirp": source_chirp,
        "output_radius_m": output_radius,
        "output_factor": output_factor,
        "kernel": kernel,
        "interpolation_left_index": left,
        "interpolation_right_index": right,
        "interpolation_right_weight": alpha,
        "shape_ny_nx": tuple(int(value) for value in shape),
        "propagated_plane_wave_reference": reference,
        "solver": "continuous_axisymmetric_fresnel_hankel_on_compact_T_minus_1",
        "reference_rule": "infinite_plane_wave_propagated_analytically",
    }


def propagate_exp030_radial_transmission(
    operator: dict[str, Any],
    transmission: NDArray[np.complexfloating],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Propagate one radial transmission to radial and Cartesian B fields."""

    values = np.asarray(transmission)
    radius = np.asarray(operator["source_radius_m"], dtype=np.float64)
    if values.dtype != np.complex128 or values.shape != radius.shape:
        raise ValueError("transmission must be complex128 on the source-radius grid.")
    if not np.all(np.isfinite(values)):
        raise ValueError("transmission must contain only finite values.")
    compact = (
        (values - 1.0)
        * np.asarray(operator["source_chirp"], dtype=np.complex128)
        * np.asarray(operator["source_measure_m2"], dtype=np.float64)
    )
    kernel = np.asarray(operator["kernel"], dtype=np.float64)
    radial_integral = np.einsum(
        "ij,j->i", kernel, compact, optimize=False
    )
    radial = (
        complex(operator["propagated_plane_wave_reference"])
        + np.asarray(operator["output_factor"], dtype=np.complex128)
        * radial_integral
    ).astype(np.complex128, copy=False)
    left = np.asarray(operator["interpolation_left_index"], dtype=np.int64)
    right = np.asarray(operator["interpolation_right_index"], dtype=np.int64)
    alpha = np.asarray(operator["interpolation_right_weight"], dtype=np.float64)
    cartesian = ((1.0 - alpha) * radial[left] + alpha * radial[right]).reshape(
        tuple(operator["shape_ny_nx"])
    )
    return radial, cartesian.astype(np.complex128, copy=False)


def make_exp030_projected_probe(
    operator: dict[str, Any],
    *,
    thickness_m: float,
    d_top_m: float,
    d_waist_m: float,
    d_bottom_m: float,
    z_waist_m: float,
    n_glass: float,
    n_air: float,
    wavelength_m: float,
    phase_scale: float = 1.0,
) -> dict[str, NDArray[np.generic]]:
    """Generate a matched exp030 projected probe for one waist diameter."""

    validate_tgv_geometry(
        thickness_m, d_top_m, d_waist_m, d_bottom_m, z_waist_m
    )
    material_values = np.asarray(
        [n_glass, n_air, wavelength_m, phase_scale], dtype=np.float64
    )
    if not np.all(np.isfinite(material_values)) or wavelength_m <= 0.0:
        raise ValueError("material, wavelength, and phase scale must be finite.")
    radius = np.asarray(operator["source_radius_m"], dtype=np.float64)
    path = analytic_air_path_length(
        radius,
        thickness_m,
        d_top_m,
        d_waist_m,
        d_bottom_m,
        z_waist_m,
    )
    phase = (
        2.0
        * np.pi
        / wavelength_m
        * (n_air - n_glass)
        * path
        * phase_scale
    )
    transmission = np.exp(1j * phase).astype(np.complex128)
    radial, cartesian = propagate_exp030_radial_transmission(
        operator, transmission
    )
    return {
        "fill_path_length_radial_m": path,
        "phase_unwrapped_radial_rad": phase.astype(np.float64),
        "A_effective_radial": transmission,
        "P_B_radial": radial,
        "P_B": cartesian,
    }
