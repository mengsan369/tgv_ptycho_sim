"""Illumination, sensitivity, Fisher, and detector metrics for exp031."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf, erfinv

from tgv_ptycho.inverse.observability import gauge_project_complex_derivative


def gaussian_total_power(amplitude: float, waist_radius_m: float) -> float:
    """Return the analytic full-plane power of ``A0 exp(-r^2/w^2)``."""

    value = float(amplitude)
    waist = float(waist_radius_m)
    if not np.isfinite(value) or not np.isfinite(waist) or waist <= 0:
        raise ValueError("amplitude must be finite and waist_radius_m positive.")
    return float(value**2 * np.pi * waist**2 / 2.0)


def gaussian_amplitude_for_total_power(
    total_power: float, waist_radius_m: float
) -> float:
    """Return ``A0`` for a requested analytic full-plane Gaussian power."""

    power = float(total_power)
    waist = float(waist_radius_m)
    if not np.isfinite(power) or power <= 0 or not np.isfinite(waist) or waist <= 0:
        raise ValueError("total_power and waist_radius_m must be finite and positive.")
    return float(np.sqrt(2.0 * power / (np.pi * waist**2)))


def gaussian_square_captured_power_fraction(
    side_length_m: float, waist_radius_m: float
) -> float:
    """Return analytic power fraction inside a centered square."""

    side = float(side_length_m)
    waist = float(waist_radius_m)
    if not np.isfinite(side) or side <= 0 or not np.isfinite(waist) or waist <= 0:
        raise ValueError("side_length_m and waist_radius_m must be positive.")
    return float(erf(side / (np.sqrt(2.0) * waist)) ** 2)


def gaussian_square_side_for_omitted_power(
    waist_radius_m: float, omitted_power_fraction: float
) -> float:
    """Return the minimum centered-square side for a Gaussian tail target."""

    waist = float(waist_radius_m)
    omitted = float(omitted_power_fraction)
    if not np.isfinite(waist) or waist <= 0:
        raise ValueError("waist_radius_m must be finite and positive.")
    if not np.isfinite(omitted) or not 0 < omitted < 1:
        raise ValueError("omitted_power_fraction must lie strictly between 0 and 1.")
    return float(np.sqrt(2.0) * waist * erfinv(np.sqrt(1.0 - omitted)))


def aligned_square_shape(
    side_length_m: float,
    dx_m: float,
    *,
    feature_size_m: float,
) -> tuple[int, int]:
    """Round a square FOV upward to full feature cells on the raster grid."""

    side = float(side_length_m)
    dx = float(dx_m)
    feature = float(feature_size_m)
    ratio = feature / dx
    pixels_per_cell = int(np.rint(ratio))
    if (
        not np.isfinite(side)
        or side <= 0
        or not np.isfinite(dx)
        or dx <= 0
        or pixels_per_cell <= 0
        or abs(ratio - pixels_per_cell) > 1.0e-9
    ):
        raise ValueError("FOV, dx, and feature size must define integer cells.")
    pixels = int(np.ceil(side / dx - 1.0e-12))
    aligned = int(np.ceil(pixels / pixels_per_cell) * pixels_per_cell)
    return aligned, aligned


def relative_change(value: float, reference: float, epsilon: float = 1.0e-15) -> float:
    """Return ``abs(value-reference)/max(abs(reference), epsilon)``."""

    numerator = abs(float(value) - float(reference))
    denominator = max(abs(float(reference)), epsilon)
    return float(numerator / denominator)


def change_class(relative_value: float) -> str:
    """Classify a preregistered engineering relative change."""

    value = float(relative_value)
    if value < 0.05:
        return "small"
    if value < 0.20:
        return "moderate"
    return "large"


def probe_sensitivity_metrics(
    probe: NDArray[np.complexfloating],
    derivative_per_m: NDArray[np.complexfloating],
    parameter_scale_m: float,
    *,
    dx_m: float,
    incident_power: float,
) -> dict[str, float]:
    """Return absolute, gauge-removed, and photon-normalized probe metrics."""

    field = np.asarray(probe, dtype=np.complex128)
    derivative = np.asarray(derivative_per_m, dtype=np.complex128)
    if field.shape != derivative.shape:
        raise ValueError("probe and derivative_per_m must have the same shape.")
    projected = gauge_project_complex_derivative(derivative, field)
    raw = float(np.sqrt(np.sum(np.abs(derivative) ** 2, dtype=np.float64)) * dx_m)
    projected_norm = float(
        np.sqrt(np.sum(np.abs(projected) ** 2, dtype=np.float64)) * dx_m
    )
    field_norm = float(np.sqrt(np.sum(np.abs(field) ** 2, dtype=np.float64)) * dx_m)
    power = float(incident_power)
    if power <= 0:
        raise ValueError("incident_power must be positive.")
    return {
        "raw_derivative_l2_per_m": raw,
        "gauge_removed_derivative_l2_per_m": projected_norm,
        "normalized_gauge_removed_sensitivity": float(
            parameter_scale_m * projected_norm / max(field_norm, np.finfo(float).eps)
        ),
        "photon_normalized_gauge_removed_derivative_l2_per_m": float(
            projected_norm / np.sqrt(power)
        ),
    }


def detector_sensitivity_metrics(
    intensity: NDArray[np.floating],
    derivative_per_m: NDArray[np.floating],
    parameter_scale_m: float,
    *,
    pixel_area_m2: float,
    incident_power: float,
) -> dict[str, float | list[float]]:
    """Return stack/frame absolute, normalized, and photon-normalized metrics."""

    values = np.asarray(intensity, dtype=np.float64)
    derivative = np.asarray(derivative_per_m, dtype=np.float64)
    if values.shape != derivative.shape or values.ndim != 3:
        raise ValueError(
            "intensity and derivative_per_m must be same-shaped 3D stacks."
        )
    frame_raw = np.sqrt(np.sum(derivative**2, axis=(1, 2), dtype=np.float64))
    frame_ref = np.sqrt(np.sum(values**2, axis=(1, 2), dtype=np.float64))
    frame_normalized = (
        parameter_scale_m * frame_raw / np.maximum(frame_ref, np.finfo(float).eps)
    )
    stack_raw = float(np.sqrt(np.sum(derivative**2, dtype=np.float64)))
    stack_ref = float(np.sqrt(np.sum(values**2, dtype=np.float64)))
    power = float(incident_power)
    if power <= 0:
        raise ValueError("incident_power must be positive.")
    photon_derivative = derivative * float(pixel_area_m2) / power
    return {
        "raw_derivative_l2_per_m": stack_raw,
        "normalized_sensitivity": float(
            parameter_scale_m * stack_raw / max(stack_ref, np.finfo(float).eps)
        ),
        "photon_normalized_derivative_l2_per_m": float(
            np.sqrt(np.sum(photon_derivative**2, dtype=np.float64))
        ),
        "per_frame_normalized": frame_normalized.tolist(),
        "per_frame_minimum": float(np.min(frame_normalized)),
        "per_frame_median": float(np.median(frame_normalized)),
        "per_frame_maximum": float(np.max(frame_normalized)),
    }


def poisson_fisher_per_incident_photon(
    intensity: NDArray[np.floating],
    derivative_per_m: NDArray[np.floating],
    *,
    pixel_area_m2: float,
    incident_power_per_frame: float,
    mu_epsilon: float,
) -> dict[str, float | list[float]]:
    """Compute ideal independent-Poisson FI per total incident photon."""

    values = np.asarray(intensity, dtype=np.float64)
    derivative = np.asarray(derivative_per_m, dtype=np.float64)
    if values.shape != derivative.shape or values.ndim != 3:
        raise ValueError(
            "intensity and derivative_per_m must be same-shaped 3D stacks."
        )
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("intensity must be finite and nonnegative.")
    power = float(incident_power_per_frame)
    epsilon = float(mu_epsilon)
    if power <= 0 or epsilon <= 0:
        raise ValueError("incident power and mu_epsilon must be positive.")
    mu = values * pixel_area_m2 / power
    dmu = derivative * pixel_area_m2 / power
    frame_fi = np.sum(dmu**2 / np.maximum(mu, epsilon), axis=(1, 2))
    fisher = float(np.mean(frame_fi))
    return {
        "fisher_information_per_incident_photon_per_m2": fisher,
        "ideal_crlb_m_per_sqrt_incident_photon": float(
            1.0 / np.sqrt(max(fisher, np.finfo(float).tiny))
        ),
        "per_frame_fisher": frame_fi.tolist(),
        "per_frame_minimum": float(np.min(frame_fi)),
        "per_frame_median": float(np.median(frame_fi)),
        "per_frame_maximum": float(np.max(frame_fi)),
    }


def detector_dynamic_range_metrics(
    intensity: NDArray[np.floating],
    *,
    pixel_area_m2: float,
    incident_power_per_frame: float,
    low_signal_fraction_of_peak: float,
    percentile_low: float,
    percentile_high: float,
) -> dict[str, float]:
    """Return conditional detector ROI energy and dynamic-range diagnostics."""

    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 3 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("intensity must be a finite nonnegative 3D stack.")
    flat = values.ravel()
    maximum = float(np.max(flat))
    median = float(np.median(flat))
    p_low = float(np.percentile(flat, percentile_low))
    p_high = float(np.percentile(flat, percentile_high))
    ny, nx = values.shape[-2:]
    yy, xx = np.indices((ny, nx))
    radius = np.sqrt(
        ((yy - (ny - 1) / 2.0) / ny) ** 2 + ((xx - (nx - 1) / 2.0) / nx) ** 2
    )
    center = values[:, radius <= 0.10]
    tail = values[:, radius >= 0.40]
    return {
        "detector_captured_energy_fraction": float(
            np.mean(np.sum(values, axis=(1, 2)))
            * pixel_area_m2
            / incident_power_per_frame
        ),
        "maximum_to_median_ratio": float(maximum / max(median, np.finfo(float).tiny)),
        "percentile_dynamic_range": float(p_high / max(p_low, np.finfo(float).tiny)),
        "center_to_tail_ratio": float(
            np.mean(center) / max(float(np.mean(tail)), np.finfo(float).tiny)
        ),
        "low_signal_pixel_fraction": float(
            np.mean(flat < low_signal_fraction_of_peak * maximum)
        ),
        "maximum_intensity": maximum,
        "median_intensity": median,
    }


__all__ = [
    "aligned_square_shape",
    "change_class",
    "detector_dynamic_range_metrics",
    "detector_sensitivity_metrics",
    "gaussian_amplitude_for_total_power",
    "gaussian_square_captured_power_fraction",
    "gaussian_square_side_for_omitted_power",
    "gaussian_total_power",
    "poisson_fisher_per_incident_photon",
    "probe_sensitivity_metrics",
    "relative_change",
]
