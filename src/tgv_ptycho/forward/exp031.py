"""Reusable Gaussian/reference-plus-residual forward helpers for exp031."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft2, ifft2, next_fast_len
from scipy.special import j0

from tgv_ptycho.objects.tgv_geometry import analytic_air_path_length


@dataclass(frozen=True)
class RadialProbeResult:
    """Radial baseline and waist derivative for one illumination envelope."""

    source_radius_m: NDArray[np.float64]
    source_weights_m: NDArray[np.float64]
    output_radius_m: NDArray[np.float64]
    source_transmission: NDArray[np.complex128]
    probe_delta_minus: NDArray[np.complex128]
    probe_delta_baseline: NDArray[np.complex128]
    probe_delta_plus: NDArray[np.complex128]
    probe_delta_derivative_per_m: NDArray[np.complex128]
    reference_kind: str


def gaussian_reference_radial(
    radius_m: NDArray[np.floating],
    *,
    waist_radius_m: float,
    amplitude: float,
    wavelength_m: float,
    distance_m: float,
    refractive_index: float = 1.0,
) -> NDArray[np.complex128]:
    """Analytic paraxial Gaussian with its waist at distance zero."""

    radius = np.asarray(radius_m, dtype=np.float64)
    waist = float(waist_radius_m)
    if waist <= 0 or wavelength_m <= 0 or refractive_index <= 0:
        raise ValueError("waist, wavelength, and refractive_index must be positive.")
    k = 2.0 * np.pi * refractive_index / wavelength_m
    z_rayleigh = 0.5 * k * waist**2
    q_factor = 1.0 + 1j * float(distance_m) / z_rayleigh
    field = (
        float(amplitude)
        * np.exp(1j * k * float(distance_m))
        / q_factor
        * np.exp(-(radius**2) / (waist**2 * q_factor))
    )
    return np.asarray(field, dtype=np.complex128)


def plane_wave_reference(
    shape: tuple[int, int],
    *,
    amplitude: float,
    wavelength_m: float,
    distance_m: float,
    refractive_index: float = 1.0,
) -> NDArray[np.complex128]:
    """Return an analytic untilted plane-wave reference on one grid."""

    k = 2.0 * np.pi * refractive_index / wavelength_m
    value = float(amplitude) * np.exp(1j * k * float(distance_m))
    return np.full(shape, value, dtype=np.complex128)


def radial_quadrature(
    *,
    d_top_m: float,
    d_waist_m: float,
    core_step_m: float,
    transition_step_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a composite midpoint rule over the compact TGV top aperture."""

    waist_radius = 0.5 * float(d_waist_m)
    outer_radius = 0.5 * float(d_top_m)
    if not 0 < waist_radius <= outer_radius:
        raise ValueError("TGV radii must satisfy 0 < waist <= top.")

    def interval(
        start: float, stop: float, target: float
    ) -> tuple[np.ndarray, np.ndarray]:
        count = max(1, int(np.ceil((stop - start) / target)))
        edges = np.linspace(start, stop, count + 1, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)

    core_r, core_w = interval(0.0, waist_radius, float(core_step_m))
    if outer_radius > waist_radius:
        taper_r, taper_w = interval(
            waist_radius, outer_radius, float(transition_step_m)
        )
        return np.concatenate([core_r, taper_r]), np.concatenate([core_w, taper_w])
    return core_r, core_w


def make_tgv_radial_probe_result(
    *,
    tgv: dict[str, float],
    wavelength_m: float,
    z_AB_m: float,
    radial_output_max_m: float,
    radial_output_step_m: float,
    finite_difference_step_m: float,
    core_step_m: float,
    transition_step_m: float,
    gaussian_waist_radius_m: float | None,
    amplitude: float = 1.0,
    refractive_index: float = 1.0,
    output_chunk_size: int = 128,
) -> RadialProbeResult:
    """Propagate ``G_A(T-1)`` for matched minus/baseline/plus waist cases."""

    source_r, source_w = radial_quadrature(
        d_top_m=tgv["d_top_m"],
        d_waist_m=tgv["d_waist_m"],
        core_step_m=core_step_m,
        transition_step_m=transition_step_m,
    )
    output_r = np.arange(
        0.0,
        float(radial_output_max_m) + 0.5 * float(radial_output_step_m),
        float(radial_output_step_m),
        dtype=np.float64,
    )
    if output_r.size < 2:
        raise ValueError("radial output grid must contain at least two nodes.")
    step = float(finite_difference_step_m)
    waist_values = np.asarray(
        [tgv["d_waist_m"] - step, tgv["d_waist_m"], tgv["d_waist_m"] + step]
    )
    k0 = 2.0 * np.pi / float(wavelength_m)
    transmissions = []
    for waist in waist_values:
        path = analytic_air_path_length(
            source_r,
            tgv["thickness_m"],
            tgv["d_top_m"],
            float(waist),
            tgv["d_bottom_m"],
            tgv["z_waist_m"],
        )
        phase = k0 * (tgv["n_air"] - tgv["n_glass"]) * path
        transmissions.append(np.exp(1j * phase))
    transmission = np.asarray(transmissions, dtype=np.complex128)
    if gaussian_waist_radius_m is None:
        source_reference = np.full(
            source_r.shape, float(amplitude), dtype=np.complex128
        )
        reference_kind = "plane_wave"
    else:
        source_reference = gaussian_reference_radial(
            source_r,
            waist_radius_m=gaussian_waist_radius_m,
            amplitude=amplitude,
            wavelength_m=wavelength_m,
            distance_m=0.0,
            refractive_index=refractive_index,
        )
        reference_kind = "gaussian"
    source = source_reference[None, :] * (transmission - 1.0)
    medium_wavelength = float(wavelength_m) / float(refractive_index)
    k = 2.0 * np.pi / medium_wavelength
    input_factor = (
        source
        * np.exp(1j * k * source_r**2 / (2.0 * z_AB_m))[None, :]
        * (source_r * source_w)[None, :]
    )
    propagated = np.empty((3, output_r.size), dtype=np.complex128)
    constant = 2.0 * np.pi * np.exp(1j * k * z_AB_m) / (1j * medium_wavelength * z_AB_m)
    for start in range(0, output_r.size, int(output_chunk_size)):
        stop = min(output_r.size, start + int(output_chunk_size))
        rho = output_r[start:stop]
        kernel = j0(k * rho[:, None] * source_r[None, :] / z_AB_m)
        # Avoid the Windows complex-BLAS loader failure observed in this project.
        # The Bessel kernel is real, so separate real/imaginary contractions are
        # algebraically identical to the complex matrix product.
        integral = np.einsum(
            "ij,kj->ik", input_factor.real, kernel, optimize=False
        ) + 1j * np.einsum("ij,kj->ik", input_factor.imag, kernel, optimize=False)
        propagated[:, start:stop] = (
            constant * np.exp(1j * k * rho**2 / (2.0 * z_AB_m))[None, :] * integral
        )
    derivative = (propagated[2] - propagated[0]) / (2.0 * step)
    return RadialProbeResult(
        source_radius_m=source_r,
        source_weights_m=source_w,
        output_radius_m=output_r,
        source_transmission=transmission[1],
        probe_delta_minus=propagated[0],
        probe_delta_baseline=propagated[1],
        probe_delta_plus=propagated[2],
        probe_delta_derivative_per_m=derivative,
        reference_kind=reference_kind,
    )


def sample_radial_field(
    radial_coordinate_m: NDArray[np.floating],
    radial_field: NDArray[np.complexfloating],
    shape: tuple[int, int],
    dx_m: float,
) -> NDArray[np.complex128]:
    """Linearly sample one centered radial complex field on `(ny, nx)`."""

    ny, nx = int(shape[0]), int(shape[1])
    y = (np.arange(ny) - (ny - 1) / 2.0) * float(dx_m)
    x = (np.arange(nx) - (nx - 1) / 2.0) * float(dx_m)
    radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    coordinate = np.asarray(radial_coordinate_m, dtype=np.float64)
    values = np.asarray(radial_field, dtype=np.complex128)
    if radius.max() > coordinate[-1] + 1.0e-15:
        raise ValueError("radial table does not cover the Cartesian grid.")
    real = np.interp(radius.ravel(), coordinate, values.real)
    imag = np.interp(radius.ravel(), coordinate, values.imag)
    return (real + 1j * imag).reshape(shape).astype(np.complex128)


def make_open_shape(
    active_shape: tuple[int, int], dx_m: float, guard_m_per_side: float
) -> tuple[int, int]:
    """Return an FFT-friendly open grid containing physical active+guard."""

    guard_px = int(np.ceil(float(guard_m_per_side) / float(dx_m)))
    return tuple(int(next_fast_len(int(size) + 2 * guard_px)) for size in active_shape)


def center_embed_slices(
    outer_shape: tuple[int, int], inner_shape: tuple[int, int]
) -> tuple[slice, slice]:
    """Return centered insertion/crop slices and fail if inner does not fit."""

    if inner_shape[0] > outer_shape[0] or inner_shape[1] > outer_shape[1]:
        raise ValueError("inner_shape must fit inside outer_shape.")
    start_y = (outer_shape[0] - inner_shape[0]) // 2
    start_x = (outer_shape[1] - inner_shape[1]) // 2
    return (
        slice(start_y, start_y + inner_shape[0]),
        slice(start_x, start_x + inner_shape[1]),
    )


def make_asm_transfer_complex64(
    shape: tuple[int, int],
    *,
    dx_m: float,
    wavelength_m: float,
    distance_m: float,
    refractive_index: float = 1.0,
    bandlimit: bool = True,
    alias_control: bool = True,
) -> NDArray[np.complex64]:
    """Build a memory-bounded same-grid ASM transfer for open residuals."""

    ny, nx = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(ny, d=dx_m)
    fx = np.fft.fftfreq(nx, d=dx_m)
    inv_lambda = float(refractive_index) / float(wavelength_m)
    radial_sq = fy[:, None] ** 2 + fx[None, :] ** 2
    propagating = radial_sq <= inv_lambda**2
    if bandlimit and alias_control:
        du = 1.0 / (nx * dx_m)
        dv = 1.0 / (ny * dx_m)
        u_limit = inv_lambda / np.sqrt(1.0 + (2.0 * du * abs(distance_m)) ** 2)
        v_limit = inv_lambda / np.sqrt(1.0 + (2.0 * dv * abs(distance_m)) ** 2)
        propagating &= (
            fx[None, :] ** 2 / u_limit**2 + fy[:, None] ** 2 / inv_lambda**2 <= 1.0
        )
        propagating &= (
            fx[None, :] ** 2 / inv_lambda**2 + fy[:, None] ** 2 / v_limit**2 <= 1.0
        )
    kz = 2.0 * np.pi * np.sqrt(np.maximum(inv_lambda**2 - radial_sq, 0.0))
    transfer = np.zeros((ny, nx), dtype=np.complex64)
    if bandlimit:
        transfer[propagating] = np.exp(1j * kz[propagating] * distance_m).astype(
            np.complex64
        )
    else:
        transfer[:] = np.exp(1j * kz * distance_m).astype(np.complex64)
    return transfer


def propagate_residual_batch_to_roi(
    residuals: NDArray[np.complexfloating],
    transfer: NDArray[np.complexfloating],
    detector_roi_shape: tuple[int, int],
    *,
    workers: int = -1,
) -> NDArray[np.complex64]:
    """Center-pad active residuals, propagate, and return the fixed center ROI."""

    values = np.asarray(residuals, dtype=np.complex64)
    kernel = np.asarray(transfer, dtype=np.complex64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("residuals must be 2D or a batch of 2D fields.")
    active_slices = center_embed_slices(kernel.shape, values.shape[-2:])
    padded = np.zeros((values.shape[0], *kernel.shape), dtype=np.complex64)
    padded[:, active_slices[0], active_slices[1]] = values
    spectrum = fft2(padded, axes=(-2, -1), workers=workers, overwrite_x=True)
    spectrum *= kernel[None, :, :]
    propagated = ifft2(spectrum, axes=(-2, -1), workers=workers, overwrite_x=True)
    roi_slices = center_embed_slices(kernel.shape, detector_roi_shape)
    return np.asarray(propagated[:, roi_slices[0], roi_slices[1]], dtype=np.complex64)


def propagate_probe_patch_batch_to_roi(
    probes_b: NDArray[np.complexfloating],
    reference_b: NDArray[np.complexfloating],
    sample_b_patch: NDArray[np.complexfloating],
    reference_detector: NDArray[np.complexfloating],
    transfer: NDArray[np.complexfloating],
    *,
    workers: int = -1,
) -> NDArray[np.complex64]:
    """Apply the Gaussian-reference plus localized-residual B-to-C model."""

    probes = np.asarray(probes_b, dtype=np.complex64)
    if probes.ndim == 2:
        probes = probes[None, ...]
    reference = np.asarray(reference_b, dtype=np.complex64)
    patch = np.asarray(sample_b_patch, dtype=np.complex64)
    detector_reference = np.asarray(reference_detector, dtype=np.complex64)
    if probes.ndim != 3 or probes.shape[-2:] != reference.shape:
        raise ValueError("probes_b and reference_b shapes are inconsistent.")
    if patch.shape != reference.shape or detector_reference.ndim != 2:
        raise ValueError("sample patch/reference detector shapes are invalid.")
    residuals = probes * patch[None, ...] - reference[None, ...]
    propagated = propagate_residual_batch_to_roi(
        residuals,
        transfer,
        detector_reference.shape,
        workers=workers,
    )
    return propagated + detector_reference[None, ...]


def adjoint_detector_roi_to_active(
    detector_fields: NDArray[np.complexfloating],
    transfer: NDArray[np.complexfloating],
    active_shape: tuple[int, int],
    *,
    workers: int = -1,
) -> NDArray[np.complex64]:
    """Apply the exact discrete adjoint of open propagation plus ROI crop."""

    values = np.asarray(detector_fields, dtype=np.complex64)
    kernel = np.asarray(transfer, dtype=np.complex64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("detector_fields must be 2D or a batch of 2D fields.")
    roi_slices = center_embed_slices(kernel.shape, values.shape[-2:])
    padded = np.zeros((values.shape[0], *kernel.shape), dtype=np.complex64)
    padded[:, roi_slices[0], roi_slices[1]] = values
    spectrum = fft2(padded, axes=(-2, -1), workers=workers, overwrite_x=True)
    spectrum *= np.conj(kernel)[None, :, :]
    propagated = ifft2(spectrum, axes=(-2, -1), workers=workers, overwrite_x=True)
    active_slices = center_embed_slices(kernel.shape, active_shape)
    return np.asarray(
        propagated[:, active_slices[0], active_slices[1]], dtype=np.complex64
    )


def residual_edge_energy_fraction(
    residual: NDArray[np.complexfloating], ring_width_px: int
) -> float:
    """Return energy fraction in a rectangular source-edge ring."""

    values = np.asarray(residual)
    width = int(ring_width_px)
    if values.ndim != 2 or width <= 0 or 2 * width >= min(values.shape):
        raise ValueError("residual and ring_width_px define an invalid edge ring.")
    total = float(np.sum(np.abs(values) ** 2, dtype=np.float64))
    interior = values[width:-width, width:-width]
    inner = float(np.sum(np.abs(interior) ** 2, dtype=np.float64))
    return float((total - inner) / max(total, np.finfo(float).eps))


def reference_plus_residual_identity_error(
    reference: NDArray[np.complexfloating],
    probe_delta: NDArray[np.complexfloating],
    sample_b: NDArray[np.complexfloating],
) -> float:
    """Check ``P*B == G + [deltaP + P*(B-1)]`` in relative L2."""

    g = np.asarray(reference, dtype=np.complex128)
    delta = np.asarray(probe_delta, dtype=np.complex128)
    b = np.asarray(sample_b, dtype=np.complex128)
    direct = (g + delta) * b
    decomposed = g + delta + (g + delta) * (b - 1.0)
    numerator = np.sum(np.abs(direct - decomposed) ** 2, dtype=np.float64)
    denominator = np.sum(np.abs(direct) ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).eps)))


def estimate_case_peak_memory_bytes(
    active_shape: tuple[int, int], open_shape: tuple[int, int], batch_size: int = 3
) -> int:
    """Conservative peak estimate for complex64 batched FFT propagation."""

    active = int(np.prod(active_shape))
    opened = int(np.prod(open_shape))
    return int(8 * (6 * active + (2 * batch_size + 3) * opened))


def finite_metrics(payload: Any) -> bool:
    """Recursively check numeric arrays/scalars in a nested payload."""

    if isinstance(payload, dict):
        return all(finite_metrics(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(finite_metrics(value) for value in payload)
    if isinstance(payload, (np.ndarray, np.number, float, int, complex)):
        return bool(np.all(np.isfinite(payload)))
    return True


__all__ = [
    "RadialProbeResult",
    "adjoint_detector_roi_to_active",
    "center_embed_slices",
    "estimate_case_peak_memory_bytes",
    "finite_metrics",
    "gaussian_reference_radial",
    "make_asm_transfer_complex64",
    "make_open_shape",
    "make_tgv_radial_probe_result",
    "plane_wave_reference",
    "propagate_residual_batch_to_roi",
    "propagate_probe_patch_batch_to_roi",
    "radial_quadrature",
    "reference_plus_residual_identity_error",
    "residual_edge_energy_fraction",
    "sample_radial_field",
]
