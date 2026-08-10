"""Run exp030: single-TGV projected-phase probe observability."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.special import j0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.integer_shift import shift_field_integer_pixels
from tgv_ptycho.forward.noise import apply_noise
from tgv_ptycho.forward.scan import add_integer_pixel_jitter, make_grid_scan
from tgv_ptycho.inverse.backprop_A import recover_thin_phase_A
from tgv_ptycho.inverse.metrics import (
    align_affine_phase_and_complex_gain,
    complex_relative_error,
)
from tgv_ptycho.inverse.observability import (
    analyze_local_observability,
    central_finite_difference,
    compare_probe_sensitivity,
    normalized_complex_sensitivity,
    normalized_real_sensitivity,
    relative_l2,
    successive_relative_changes,
)
from tgv_ptycho.io.config import config_to_yaml, load_config, save_config
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit
from tgv_ptycho.io.naming import make_run_dir
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5
from tgv_ptycho.objects.sample_b import make_random_phase_object
from tgv_ptycho.objects.tgv2d import (
    make_tgv_projected_phase,
    make_thin_phase_disk,
)
from tgv_ptycho.objects.tgv_geometry import (
    analytic_air_path_length,
    diameter_profile,
)
from tgv_ptycho.optics.angular_spectrum import (
    angular_spectrum_propagate,
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)
from tgv_ptycho.optics.fields import coordinate_grid, make_plane_wave
from tgv_ptycho.recon.epie import epie_reconstruct
from tgv_ptycho.recon.initialization import (
    initialize_probe_by_detector_backpropagation,
)
from tgv_ptycho.viz.plot_tgv import (
    plot_diameter_profile,
    plot_effective_transmission,
    plot_intensity_sensitivity,
    plot_jacobian_correlation,
    plot_jacobian_singular_values,
    plot_loss_curves,
    plot_opd_and_phase,
    plot_phase_profiles,
    plot_probe_sensitivity_maps,
    plot_radial_profiles,
    plot_recovered_probe_cases,
    plot_sensitivity_curve,
    plot_step_convergence,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume-blind-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume an interrupted exp030 blind-long trajectory in a new "
            "timestamped run. The checkpoint is never modified."
        ),
    )
    return parser.parse_args()


def _shape(values: Any, name: str) -> tuple[int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        msg = f"{name} must contain [ny, nx]."
        raise ValueError(msg)
    shape = int(values[0]), int(values[1])
    if min(shape) <= 0:
        msg = f"{name} entries must be positive."
        raise ValueError(msg)
    return shape


def _norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(values) ** 2, dtype=np.float64)))


def _relative_change(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(abs(reference), np.finfo(float).eps))


def _make_scan(config: dict[str, Any], dx_m: float) -> np.ndarray:
    scan_cfg = config["scan"]
    if scan_cfg.get("type") != "jittered_grid":
        msg = "exp030 supports the paired jittered_grid scan used by exp020."
        raise ValueError(msg)
    regular = make_grid_scan(
        int(scan_cfg["num_x"]),
        int(scan_cfg["num_y"]),
        float(scan_cfg["step_m"]),
        center=bool(scan_cfg.get("center", True)),
    )
    jitter_quantum_m = float(scan_cfg.get("jitter_quantum_m", dx_m))
    positions = add_integer_pixel_jitter(
        regular,
        jitter_quantum_m,
        int(scan_cfg.get("max_jitter_px", 0)),
        seed=int(scan_cfg["jitter_seed"]),
    )
    if len(np.unique(positions, axis=0)) != len(positions):
        msg = "Jittered scan contains duplicate positions."
        raise ValueError(msg)
    pixel_coordinates = positions / dx_m
    if not np.allclose(pixel_coordinates, np.round(pixel_coordinates)):
        msg = "Physical scan positions must be integer shifts on the baseline grid."
        raise ValueError(msg)
    return positions


def _projected_model(
    config: dict[str, Any],
    *,
    shape: tuple[int, int],
    dx_m: float,
    dz_m: float,
    supersampling: int,
    d_waist_m: float | None = None,
    d_top_m: float | None = None,
    d_bottom_m: float | None = None,
    phase_scale: float | None = None,
    n_air: float | None = None,
    integration_method: str | None = None,
) -> dict[str, np.ndarray]:
    optics = config["optics"]
    tgv = config["tgv"]
    projected = config["projected_phase"]
    result = make_tgv_projected_phase(
        shape,
        dx_m,
        float(optics["wavelength_m"]),
        float(tgv["thickness_m"]),
        float(tgv["d_top_m"] if d_top_m is None else d_top_m),
        float(tgv["d_waist_m"] if d_waist_m is None else d_waist_m),
        float(tgv["d_bottom_m"] if d_bottom_m is None else d_bottom_m),
        dz_m,
        z_waist=float(tgv["z_waist_m"]),
        n_glass=float(tgv["n_glass"]),
        n_air=float(tgv["n_air"] if n_air is None else n_air),
        center_xy_m=tuple(float(v) for v in tgv["center_xy_m"]),
        lateral_supersampling=supersampling,
        integration_method=(
            str(projected["integration_method"])
            if integration_method is None
            else integration_method
        ),
        phase_scale=float(
            projected["phase_scale"] if phase_scale is None else phase_scale
        ),
    )
    return {key: np.asarray(value) for key, value in result.items()}


def _detector_pixel_average(
    intensity: np.ndarray,
    simulation_dx_m: float,
    detector_pixel_size_m: float,
) -> np.ndarray:
    """Area-average a simulated irradiance grid onto fixed detector pixels."""

    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim < 2:
        msg = "intensity must have at least two dimensions."
        raise ValueError(msg)
    if (
        not np.isfinite(simulation_dx_m)
        or not np.isfinite(detector_pixel_size_m)
        or simulation_dx_m <= 0.0
        or detector_pixel_size_m <= 0.0
    ):
        msg = "simulation and detector sampling must be finite and positive."
        raise ValueError(msg)
    ratio = detector_pixel_size_m / simulation_dx_m
    factor = int(round(ratio))
    if factor <= 0 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-12):
        msg = "detector pixel size must be an integer multiple of simulation dx."
        raise ValueError(msg)
    ny, nx = values.shape[-2:]
    if ny % factor != 0 or nx % factor != 0:
        msg = "simulation shape must be divisible by the detector bin factor."
        raise ValueError(msg)
    if factor == 1:
        return values.astype(np.float64, copy=False)
    reshaped = values.reshape(
        *values.shape[:-2], ny // factor, factor, nx // factor, factor
    )
    return reshaped.mean(axis=(-3, -1), dtype=np.float64)


def _angular_spectrum_transfer(
    shape: tuple[int, int],
    dx_m: float,
    wavelength_m: float,
    z_m: float,
    medium_index: float,
) -> np.ndarray:
    """Build the existing propagating-wave ASM transfer once per grid."""
    return make_angular_spectrum_transfer(
        shape,
        dx_m,
        wavelength_m,
        z_m,
        n=medium_index,
        bandlimit=True,
    )


def _simulate_probe_detector(
    config: dict[str, Any],
    probe: np.ndarray,
    sample_b: np.ndarray,
    scan_positions: np.ndarray,
    dx_m: float,
    transfer_bc: np.ndarray,
    *,
    object_boundary: str = "periodic",
    object_boundary_value: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Propagate a supplied B-plane probe to one fixed physical detector."""

    field = np.asarray(probe, dtype=np.complex128)
    object_b = np.asarray(sample_b, dtype=np.complex128)
    positions = np.asarray(scan_positions, dtype=np.float64)
    if field.shape != object_b.shape or field.shape != transfer_bc.shape:
        msg = "probe, sample B, and B-to-C transfer must share one shape."
        raise ValueError(msg)
    optics = config["optics"]
    detector_pixel = float(optics["detector_pixel_size_m"])
    detector_shape = _detector_pixel_average(
        np.zeros(field.shape, dtype=np.float64), dx_m, detector_pixel
    ).shape
    intensity_stack = np.empty(
        (len(positions), *detector_shape), dtype=np.float64
    )
    for index, position_xy in enumerate(positions):
        shifted_b = shift_field_integer_pixels(
            object_b,
            position_xy,
            dx_m,
            boundary=object_boundary,
            fill_value=object_boundary_value,
        )
        detector_field = apply_angular_spectrum_transfer(
            field * shifted_b, transfer_bc
        )
        pixel_average = _detector_pixel_average(
            np.abs(detector_field) ** 2, dx_m, detector_pixel
        )
        intensity_stack[index] = apply_noise(
            pixel_average, noise_config=config.get("noise"), seed=None
        )
    return intensity_stack


def _model_validation(
    config: dict[str, Any],
    baseline: dict[str, np.ndarray],
    shape: tuple[int, int],
    dx_m: float,
    dz_m: float,
    supersampling: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    tgv = config["tgv"]
    optics = config["optics"]
    thickness = float(tgv["thickness_m"])
    d_top = float(tgv["d_top_m"])
    d_waist = float(tgv["d_waist_m"])
    d_bottom = float(tgv["d_bottom_m"])
    z_waist = float(tgv["z_waist_m"])

    analytic = _projected_model(
        config,
        shape=shape,
        dx_m=dx_m,
        dz_m=dz_m,
        supersampling=supersampling,
        integration_method="analytic",
    )
    fill_difference = baseline["fill_path_length_m"] - analytic[
        "fill_path_length_m"
    ]

    cylinder = _projected_model(
        config,
        shape=shape,
        dx_m=dx_m,
        dz_m=dz_m,
        supersampling=1,
        d_waist_m=d_top,
        d_bottom_m=d_top,
    )
    phase_shift = (
        2.0
        * np.pi
        / float(optics["wavelength_m"])
        * (float(tgv["n_air"]) - float(tgv["n_glass"]))
        * thickness
    )
    phase_disk = make_thin_phase_disk(shape, dx_m, d_top, phase_shift)
    zero_contrast = _projected_model(
        config,
        shape=shape,
        dx_m=dx_m,
        dz_m=dz_m,
        supersampling=supersampling,
        n_air=float(tgv["n_glass"]),
    )

    x_grid, y_grid = coordinate_grid(shape, dx_m)
    center_x, center_y = (float(v) for v in tgv["center_xy_m"])
    radius = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    reference = radius >= 0.5 * max(d_top, d_bottom) + 2.0 * dx_m
    endpoint_z = np.asarray([0.0, z_waist, thickness], dtype=np.float64)
    endpoint_expected = np.asarray(
        [d_top, d_waist, d_bottom], dtype=np.float64
    )
    endpoint_actual = diameter_profile(
        endpoint_z, thickness, d_top, d_waist, d_bottom, z_waist
    )

    metrics = {
        "diameter_profile_max_abs_error_m": float(
            np.max(np.abs(endpoint_actual - endpoint_expected))
        ),
        "fill_path_analytic_max_abs_error_m": float(
            np.max(np.abs(fill_difference))
        ),
        "fill_path_analytic_rmse_m": float(
            np.sqrt(np.mean(fill_difference**2))
        ),
        "transmission_complex_relative_error": relative_l2(
            cylinder["A_effective_true"], phase_disk
        ),
        "zero_contrast_max_abs_error": float(
            np.max(np.abs(zero_contrast["A_effective_true"] - 1.0))
        ),
        "reference_region_max_abs_T_minus_1": float(
            np.max(np.abs(baseline["A_effective_true"][reference] - 1.0))
        ),
        "pure_phase_amplitude_max_abs_error": float(
            np.max(np.abs(np.abs(baseline["A_effective_true"]) - 1.0))
        ),
    }
    controls = {
        "analytic_fill_path_length_m": analytic["fill_path_length_m"],
        "reference_region_mask": reference,
    }
    return metrics, controls


def _frame_sensitivity(
    derivative: np.ndarray, baseline: np.ndarray, parameter_scale: float
) -> np.ndarray:
    numerator = np.sqrt(np.sum(derivative**2, axis=(1, 2), dtype=np.float64))
    denominator = np.sqrt(np.sum(baseline**2, axis=(1, 2), dtype=np.float64))
    return parameter_scale * numerator / np.maximum(
        denominator, np.finfo(float).eps
    )


def _projected_phase_fringe_metrics(config: dict[str, Any]) -> dict[str, float]:
    """Estimate the narrowest radial phase-fringe period in the taper."""

    optics = config["optics"]
    tgv = config["tgv"]
    projected = config["projected_phase"]
    r_top = 0.5 * float(tgv["d_top_m"])
    r_waist = 0.5 * float(tgv["d_waist_m"])
    r_bottom = 0.5 * float(tgv["d_bottom_m"])
    thickness = float(tgv["thickness_m"])
    z_waist = float(tgv["z_waist_m"])
    path_slope = 0.0
    if r_top > r_waist:
        path_slope += z_waist / (r_top - r_waist)
    if r_bottom > r_waist:
        path_slope += (thickness - z_waist) / (r_bottom - r_waist)
    contrast = abs(float(tgv["n_air"]) - float(tgv["n_glass"]))
    phase_scale = abs(float(projected["phase_scale"]))
    if path_slope <= 0.0 or contrast <= 0.0 or phase_scale <= 0.0:
        msg = "A tapered, non-zero-contrast phase profile is required."
        raise ValueError(msg)
    fringe_period = float(optics["wavelength_m"]) / (
        contrast * phase_scale * path_slope
    )
    return {
        "transition_max_abs_path_slope": float(path_slope),
        "transition_phase_fringe_period_m": float(fringe_period),
        "transition_phase_nyquist_dx_requirement_m": float(
            0.5 * fringe_period
        ),
    }


def _resolved_tgv_case(
    config: dict[str, Any], overrides: dict[str, float]
) -> dict[str, float]:
    tgv = config["tgv"]
    projected = config["projected_phase"]
    return {
        "thickness_m": float(tgv["thickness_m"]),
        "d_top_m": float(overrides.get("d_top_m", tgv["d_top_m"])),
        "d_waist_m": float(
            overrides.get("d_waist_m", tgv["d_waist_m"])
        ),
        "d_bottom_m": float(
            overrides.get("d_bottom_m", tgv["d_bottom_m"])
        ),
        "z_waist_m": float(tgv["z_waist_m"]),
        "n_glass": float(tgv["n_glass"]),
        "n_air": float(overrides.get("n_air", tgv["n_air"])),
        "phase_scale": float(
            overrides.get("phase_scale", projected["phase_scale"])
        ),
    }


def _midpoint_interval(
    start_m: float, stop_m: float, maximum_step_m: float
) -> tuple[np.ndarray, np.ndarray]:
    if stop_m <= start_m:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    count = int(np.ceil((stop_m - start_m) / maximum_step_m))
    step = (stop_m - start_m) / count
    nodes = start_m + (np.arange(count, dtype=np.float64) + 0.5) * step
    return nodes, np.full(count, step, dtype=np.float64)


def _tgv_radial_quadrature(
    config: dict[str, Any],
    cases: dict[str, dict[str, float]],
    transition_step_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build a composite radial midpoint rule focused on the taper annulus."""

    if not np.isfinite(transition_step_m) or transition_step_m <= 0.0:
        msg = "transition_step_m must be finite and positive."
        raise ValueError(msg)
    parameters = [
        _resolved_tgv_case(config, overrides) for overrides in cases.values()
    ]
    inner_stop = 0.5 * min(case["d_waist_m"] for case in parameters)
    outer_stop = 0.5 * max(
        max(case["d_top_m"], case["d_bottom_m"]) for case in parameters
    )
    inner_nodes, inner_weights = _midpoint_interval(
        0.0, inner_stop, 50.0 * transition_step_m
    )
    transition_nodes, transition_weights = _midpoint_interval(
        inner_stop, outer_stop, transition_step_m
    )
    radius = np.concatenate([inner_nodes, transition_nodes])
    weights = np.concatenate([inner_weights, transition_weights])
    diagnostics = {
        "requested_transition_step_m": float(transition_step_m),
        "maximum_actual_step_m": float(np.max(weights)),
        "inner_region_stop_m": float(inner_stop),
        "outer_support_radius_m": float(outer_stop),
        "num_radial_source_nodes": int(radius.size),
    }
    return radius, weights, diagnostics


def _fresnel_hankel_radial(
    source_radius_m: np.ndarray,
    source_weights_m: np.ndarray,
    transmissions: np.ndarray,
    *,
    output_radius_max_m: float,
    output_step_m: float,
    wavelength_m: float,
    propagation_distance_m: float,
    medium_index: float,
    incident_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate compact radial ``T-1`` perturbations with Fresnel-Hankel."""

    radius = np.asarray(source_radius_m, dtype=np.float64)
    weights = np.asarray(source_weights_m, dtype=np.float64)
    fields = np.asarray(transmissions, dtype=np.complex128)
    if fields.ndim != 2 or fields.shape[1] != radius.size:
        msg = "transmissions must have shape (num_cases, num_source_nodes)."
        raise ValueError(msg)
    if weights.shape != radius.shape:
        msg = "source radius and quadrature weights must share one shape."
        raise ValueError(msg)
    if min(wavelength_m, propagation_distance_m, medium_index) <= 0.0:
        msg = "wavelength, propagation distance, and medium index must be positive."
        raise ValueError(msg)
    if output_radius_max_m <= 0.0 or output_step_m <= 0.0:
        msg = "radial output extent and step must be positive."
        raise ValueError(msg)

    medium_wavelength = wavelength_m / medium_index
    wavenumber = 2.0 * np.pi / medium_wavelength
    count = int(np.ceil(output_radius_max_m / output_step_m)) + 2
    output_radius = np.arange(count, dtype=np.float64) * output_step_m
    compact_source = (
        (fields - 1.0).T
        * np.exp(
            1j
            * wavenumber
            * radius**2
            / (2.0 * propagation_distance_m)
        )[:, None]
        * (radius * weights)[:, None]
    )
    integral = np.empty(
        (output_radius.size, fields.shape[0]), dtype=np.complex128
    )
    for start in range(0, output_radius.size, 128):
        selection = slice(start, min(start + 128, output_radius.size))
        kernel = j0(
            wavenumber
            * output_radius[selection, None]
            * radius[None, :]
            / propagation_distance_m
        )
        # NumPy's complex GEMM is pathologically slow in the target Windows
        # environment; the explicit contraction is numerically identical.
        integral[selection] = np.einsum(
            "ij,jk->ik", kernel, compact_source, optimize=False
        )

    propagated_reference = incident_amplitude * np.exp(
        1j * wavenumber * propagation_distance_m
    )
    prefactor = (
        propagated_reference
        * np.exp(
            1j
            * wavenumber
            * output_radius**2
            / (2.0 * propagation_distance_m)
        )[:, None]
        * (2.0 * np.pi / (1j * medium_wavelength * propagation_distance_m))
    )
    radial_probes = propagated_reference + prefactor * integral
    return output_radius, radial_probes.T.astype(np.complex128, copy=False)


def _fresnel_tgv_probe_batch(
    config: dict[str, Any],
    cases: dict[str, dict[str, float]],
    *,
    transition_step_m: float,
    output_radius_max_m: float,
    output_step_m: float,
) -> dict[str, Any]:
    """Evaluate all requested TGV cases in one shared radial propagation."""

    source_radius, source_weights, diagnostics = _tgv_radial_quadrature(
        config, cases, transition_step_m
    )
    optics = config["optics"]
    transmissions: list[np.ndarray] = []
    for overrides in cases.values():
        case = _resolved_tgv_case(config, overrides)
        path = analytic_air_path_length(
            source_radius,
            case["thickness_m"],
            case["d_top_m"],
            case["d_waist_m"],
            case["d_bottom_m"],
            case["z_waist_m"],
        )
        phase = (
            2.0
            * np.pi
            / float(optics["wavelength_m"])
            * (case["n_air"] - case["n_glass"])
            * path
            * case["phase_scale"]
        )
        transmissions.append(np.exp(1j * phase).astype(np.complex128))
    transmission_array = np.stack(transmissions)
    output_radius, radial_probes = _fresnel_hankel_radial(
        source_radius,
        source_weights,
        transmission_array,
        output_radius_max_m=output_radius_max_m,
        output_step_m=output_step_m,
        wavelength_m=float(optics["wavelength_m"]),
        propagation_distance_m=float(optics["z_AB_m"]),
        medium_index=float(optics["medium_index"]),
        incident_amplitude=float(config["illumination"]["amplitude"]),
    )
    labels = list(cases)
    return {
        "case_labels": labels,
        "source_radius_m": source_radius,
        "source_weights_m": source_weights,
        "source_transmission": transmission_array,
        "probe_radius_m": output_radius,
        "probe_radial": radial_probes,
        "quadrature": diagnostics,
    }


def _radial_interpolation_plan(
    radial_coordinate_m: np.ndarray,
    shape: tuple[int, int],
    dx_m: float,
    center_xy_m: tuple[float, float],
) -> dict[str, Any]:
    """Build the exact linear interpolation used from radial to Cartesian B."""

    coordinate = np.asarray(radial_coordinate_m, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or coordinate.size < 2
        or not np.all(np.isfinite(coordinate))
        or not np.all(np.diff(coordinate) > 0.0)
    ):
        msg = "radial_coordinate_m must be finite and strictly increasing."
        raise ValueError(msg)
    if not np.isfinite(dx_m) or dx_m <= 0.0:
        msg = "dx_m must be finite and positive."
        raise ValueError(msg)
    x_grid, y_grid = coordinate_grid(shape, dx_m)
    radius = np.sqrt(
        (x_grid - center_xy_m[0]) ** 2 + (y_grid - center_xy_m[1]) ** 2
    )
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, coordinate[-1])
    if float(np.max(radius)) > float(coordinate[-1]) + tolerance:
        msg = "radial probe table does not cover the Cartesian output grid."
        raise ValueError(msg)
    flat_radius = np.minimum(radius.ravel(), coordinate[-1])
    right = np.searchsorted(coordinate, flat_radius, side="right")
    right = np.clip(right, 1, coordinate.size - 1).astype(np.int64)
    left = right - 1
    spacing = coordinate[right] - coordinate[left]
    alpha = np.clip(
        (flat_radius - coordinate[left]) / spacing, 0.0, 1.0
    ).astype(np.float64)
    return {
        "radial_coordinate_m": coordinate,
        "left_index": left,
        "right_index": right,
        "right_weight": alpha,
        "shape": shape,
    }


def _apply_radial_interpolation(
    radial_field: np.ndarray, plan: dict[str, Any]
) -> np.ndarray:
    """Apply radial-to-Cartesian linear interpolation ``S``."""

    values = np.asarray(radial_field, dtype=np.complex128)
    coordinate = np.asarray(plan["radial_coordinate_m"])
    if values.shape != coordinate.shape:
        msg = "radial_field must match the interpolation radial coordinate."
        raise ValueError(msg)
    left = np.asarray(plan["left_index"], dtype=np.int64)
    right = np.asarray(plan["right_index"], dtype=np.int64)
    alpha = np.asarray(plan["right_weight"], dtype=np.float64)
    sampled = (1.0 - alpha) * values[left] + alpha * values[right]
    return sampled.reshape(tuple(plan["shape"])).astype(
        np.complex128, copy=False
    )


def _adjoint_radial_interpolation(
    cartesian_field: np.ndarray, plan: dict[str, Any]
) -> np.ndarray:
    """Apply the Euclidean adjoint ``S^H`` by scatter-adding interpolation weights."""

    values = np.asarray(cartesian_field, dtype=np.complex128)
    shape = tuple(plan["shape"])
    if values.shape != shape:
        msg = "cartesian_field must match the interpolation output shape."
        raise ValueError(msg)
    left = np.asarray(plan["left_index"], dtype=np.int64)
    right = np.asarray(plan["right_index"], dtype=np.int64)
    alpha = np.asarray(plan["right_weight"], dtype=np.float64)
    flat = values.ravel()
    result = np.zeros(
        np.asarray(plan["radial_coordinate_m"]).shape, dtype=np.complex128
    )
    np.add.at(result, left, (1.0 - alpha) * flat)
    np.add.at(result, right, alpha * flat)
    return result


def _solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve one complex-RHS tridiagonal system without a BLAS dependency."""

    a = np.asarray(lower, dtype=np.float64)
    b = np.asarray(diagonal, dtype=np.float64)
    c = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(right_hand_side, dtype=np.complex128)
    if (
        b.ndim != 1
        or rhs.shape != b.shape
        or a.shape != (b.size - 1,)
        or c.shape != (b.size - 1,)
    ):
        msg = "Invalid tridiagonal system shapes."
        raise ValueError(msg)
    modified_upper = np.empty_like(c)
    modified_rhs = np.empty_like(rhs)
    pivot = b[0]
    if not np.isfinite(pivot) or abs(pivot) <= np.finfo(float).eps:
        msg = "Tridiagonal system has a zero or non-finite pivot."
        raise FloatingPointError(msg)
    if c.size:
        modified_upper[0] = c[0] / pivot
    modified_rhs[0] = rhs[0] / pivot
    for index in range(1, b.size):
        pivot = b[index] - a[index - 1] * modified_upper[index - 1]
        if not np.isfinite(pivot) or abs(pivot) <= np.finfo(float).eps:
            msg = "Tridiagonal system has a zero or non-finite pivot."
            raise FloatingPointError(msg)
        if index < c.size:
            modified_upper[index] = c[index] / pivot
        modified_rhs[index] = (
            rhs[index] - a[index - 1] * modified_rhs[index - 1]
        ) / pivot
    solution = np.empty_like(rhs)
    solution[-1] = modified_rhs[-1]
    for index in range(b.size - 2, -1, -1):
        solution[index] = (
            modified_rhs[index] - modified_upper[index] * solution[index + 1]
        )
    return solution


def _make_radial_output_range_constraint(
    interpolation_plan: dict[str, Any], ridge_fraction: float
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Project a Cartesian probe onto the sampled axisymmetric radial range.

    This is the regularized projector ``S (S^H S + ridge I)^-1 S^H``. It is
    deliberately a B-plane range constraint, not an inverse for the much more
    underdetermined high-resolution radial A-plane transmission.
    """

    if not np.isfinite(ridge_fraction) or ridge_fraction <= 0.0:
        msg = "ridge_fraction must be finite and positive."
        raise ValueError(msg)
    coordinate = np.asarray(interpolation_plan["radial_coordinate_m"])
    left = np.asarray(interpolation_plan["left_index"], dtype=np.int64)
    right = np.asarray(interpolation_plan["right_index"], dtype=np.int64)
    alpha = np.asarray(interpolation_plan["right_weight"], dtype=np.float64)
    left_weight = 1.0 - alpha
    diagonal = np.zeros(coordinate.shape, dtype=np.float64)
    np.add.at(diagonal, left, left_weight**2)
    np.add.at(diagonal, right, alpha**2)
    off_diagonal = np.zeros(coordinate.size - 1, dtype=np.float64)
    np.add.at(off_diagonal, left, left_weight * alpha)
    ridge = float(ridge_fraction * np.max(diagonal))
    regularized_diagonal = diagonal + ridge
    state: dict[str, Any] = {
        "call_count": 0,
        "ridge_fraction": float(ridge_fraction),
        "ridge_absolute": ridge,
        "normal_diagonal_min": float(np.min(diagonal)),
        "normal_diagonal_max": float(np.max(diagonal)),
        "active_radial_node_count": int(np.count_nonzero(diagonal > 0.0)),
        "radial_node_count": int(coordinate.size),
        "last_relative_change": 0.0,
    }

    def continuous_radial_output_range_constraint(
        probe: np.ndarray,
    ) -> np.ndarray:
        target = np.asarray(probe, dtype=np.complex128)
        radial_rhs = _adjoint_radial_interpolation(
            target, interpolation_plan
        )
        radial_solution = _solve_tridiagonal(
            off_diagonal,
            regularized_diagonal,
            off_diagonal,
            radial_rhs,
        )
        projected = _apply_radial_interpolation(
            radial_solution, interpolation_plan
        )
        state["call_count"] = int(state["call_count"]) + 1
        state["last_relative_change"] = relative_l2(projected, target)
        return projected

    return continuous_radial_output_range_constraint, state


def _build_radial_fresnel_operator(
    source_radius_m: np.ndarray,
    source_weights_m: np.ndarray,
    output_radius_m: np.ndarray,
    *,
    shape: tuple[int, int],
    dx_m: float,
    center_xy_m: tuple[float, float],
    wavelength_m: float,
    propagation_distance_m: float,
    medium_index: float,
    incident_amplitude: float,
) -> dict[str, Any]:
    """Build the linear part of the authoritative radial A-to-B operator."""

    radius = np.asarray(source_radius_m, dtype=np.float64)
    weights = np.asarray(source_weights_m, dtype=np.float64)
    output_radius = np.asarray(output_radius_m, dtype=np.float64)
    if (
        radius.ndim != 1
        or radius.size == 0
        or weights.shape != radius.shape
        or np.any(radius <= 0.0)
        or np.any(weights <= 0.0)
    ):
        msg = "source radii and quadrature weights must be positive 1D arrays."
        raise ValueError(msg)
    if min(wavelength_m, propagation_distance_m, medium_index) <= 0.0:
        msg = "wavelength, propagation distance, and medium index must be positive."
        raise ValueError(msg)
    medium_wavelength = wavelength_m / medium_index
    wavenumber = 2.0 * np.pi / medium_wavelength
    propagated_reference = incident_amplitude * np.exp(
        1j * wavenumber * propagation_distance_m
    )
    source_measure = 2.0 * np.pi * radius * weights
    source_chirp = np.exp(
        1j * wavenumber * radius**2 / (2.0 * propagation_distance_m)
    )
    output_factor = (
        propagated_reference
        * np.exp(
            1j
            * wavenumber
            * output_radius**2
            / (2.0 * propagation_distance_m)
        )
        / (1j * medium_wavelength * propagation_distance_m)
    )
    kernel = j0(
        wavenumber
        * output_radius[:, None]
        * radius[None, :]
        / propagation_distance_m
    ).astype(np.float64, copy=False)
    interpolation = _radial_interpolation_plan(
        output_radius, shape, dx_m, center_xy_m
    )
    return {
        "source_radius_m": radius,
        "source_weights_m": weights,
        "source_measure_m2": source_measure,
        "source_chirp": source_chirp,
        "output_radius_m": output_radius,
        "output_factor": output_factor,
        "kernel": kernel,
        "interpolation": interpolation,
        "propagated_reference": complex(propagated_reference),
        "dx_m": float(dx_m),
        "adjoint_convention": (
            "A_dagger=W_A^-1 F^H S^H W_B; "
            "W_A=diag(2*pi*r*dr), W_B=dx^2*I"
        ),
    }


def _radial_fresnel_linear_forward(
    operator: dict[str, Any], source_perturbation: np.ndarray
) -> np.ndarray:
    """Apply ``A=S F`` to the compact radial perturbation ``T-1``."""

    perturbation = np.asarray(source_perturbation, dtype=np.complex128)
    measure = np.asarray(operator["source_measure_m2"], dtype=np.float64)
    if perturbation.shape != measure.shape:
        msg = "source_perturbation must match the radial source nodes."
        raise ValueError(msg)
    compact = (
        perturbation
        * np.asarray(operator["source_chirp"], dtype=np.complex128)
        * measure
    )
    kernel = np.asarray(operator["kernel"], dtype=np.float64)
    radial_integral = (
        np.einsum("ij,j->i", kernel, compact.real, optimize=False)
        + 1j
        * np.einsum("ij,j->i", kernel, compact.imag, optimize=False)
    )
    radial_delta = (
        np.asarray(operator["output_factor"], dtype=np.complex128)
        * radial_integral
    )
    return _apply_radial_interpolation(
        radial_delta, operator["interpolation"]
    )


def _radial_fresnel_weighted_adjoint(
    operator: dict[str, Any], cartesian_residual: np.ndarray
) -> np.ndarray:
    """Apply the exact weighted adjoint of ``A=S F`` to a B-plane residual."""

    residual = np.asarray(cartesian_residual, dtype=np.complex128)
    interpolation = operator["interpolation"]
    if residual.shape != tuple(interpolation["shape"]):
        msg = "cartesian_residual must match the operator output shape."
        raise ValueError(msg)
    weighted_radial = _adjoint_radial_interpolation(
        float(operator["dx_m"]) ** 2 * residual, interpolation
    )
    kernel = np.asarray(operator["kernel"], dtype=np.float64)
    radial_source = (
        np.conj(np.asarray(operator["output_factor"])) * weighted_radial
    )
    contracted = (
        np.einsum("ij,i->j", kernel, radial_source.real, optimize=False)
        + 1j
        * np.einsum("ij,i->j", kernel, radial_source.imag, optimize=False)
    )
    return (
        np.conj(np.asarray(operator["source_chirp"])) * contracted
    ).astype(np.complex128, copy=False)


def _radial_fresnel_full_field(
    operator: dict[str, Any], source_transmission: np.ndarray
) -> np.ndarray:
    """Forward a radial transmission, retaining the analytic plane-wave reference."""

    transmission = np.asarray(source_transmission, dtype=np.complex128)
    return (
        complex(operator["propagated_reference"])
        + _radial_fresnel_linear_forward(operator, transmission - 1.0)
    ).astype(np.complex128, copy=False)


def _estimate_radial_operator_norm_squared(
    operator: dict[str, Any], num_iterations: int, seed: int
) -> float:
    """Estimate the largest weighted singular value squared by power iteration."""

    if num_iterations <= 0:
        msg = "num_iterations must be positive."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    measure = np.asarray(operator["source_measure_m2"], dtype=np.float64)
    vector = rng.normal(size=measure.shape) + 1j * rng.normal(
        size=measure.shape
    )
    for _ in range(num_iterations):
        norm = np.sqrt(np.sum(measure * np.abs(vector) ** 2))
        if not np.isfinite(norm) or norm <= np.finfo(float).eps:
            msg = "Power iteration encountered a zero or non-finite vector."
            raise FloatingPointError(msg)
        vector = vector / norm
        vector = _radial_fresnel_weighted_adjoint(
            operator, _radial_fresnel_linear_forward(operator, vector)
        )
    source_norm_squared = float(np.sum(measure * np.abs(vector) ** 2))
    if source_norm_squared <= np.finfo(float).eps:
        msg = "Power iteration returned a zero operator norm."
        raise FloatingPointError(msg)
    vector /= np.sqrt(source_norm_squared)
    propagated = _radial_fresnel_linear_forward(operator, vector)
    estimate = float(
        float(operator["dx_m"]) ** 2 * np.sum(np.abs(propagated) ** 2)
    )
    if not np.isfinite(estimate) or estimate <= 0.0:
        msg = "Estimated operator norm must be finite and positive."
        raise FloatingPointError(msg)
    return estimate


def _fit_global_complex_gain(
    model: np.ndarray, target: np.ndarray
) -> complex:
    """Fit the probe/object complex-scale gauge for one model field."""

    model_field = np.asarray(model, dtype=np.complex128)
    target_field = np.asarray(target, dtype=np.complex128)
    if model_field.shape != target_field.shape:
        msg = "model and target must share one shape."
        raise ValueError(msg)
    denominator = np.sum(np.conj(model_field) * model_field)
    if abs(denominator) <= np.finfo(float).eps:
        return 1.0 + 0.0j
    numerator = np.sum(np.conj(model_field) * target_field)
    return complex(numerator / denominator)


def _make_radial_adjoint_constraint(
    operator: dict[str, Any],
    *,
    operator_norm_squared: float,
    application_interval: int,
    internal_steps: int,
    step_scale: float,
    max_backtracking_steps: int,
    initial_transmission: np.ndarray | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Create a stateful pure-phase radial constraint using the paired adjoint."""

    if application_interval <= 0 or internal_steps <= 0:
        msg = "constraint interval and internal step count must be positive."
        raise ValueError(msg)
    if operator_norm_squared <= 0.0 or step_scale <= 0.0:
        msg = "operator norm and step scale must be positive."
        raise ValueError(msg)
    source_shape = np.asarray(operator["source_radius_m"]).shape
    if initial_transmission is None:
        transmission = np.ones(source_shape, dtype=np.complex128)
    else:
        initial = np.asarray(initial_transmission, dtype=np.complex128)
        if initial.shape != source_shape or not np.all(np.isfinite(initial)):
            msg = "initial_transmission must be finite and match source nodes."
            raise ValueError(msg)
        transmission = np.exp(1j * np.angle(initial)).astype(np.complex128)
    state: dict[str, Any] = {
        "source_transmission": transmission,
        "call_count": 0,
        "application_call_index": [],
        "objective_before": [],
        "objective_after": [],
        "accepted_step": [],
        "global_gain": [],
        "operator_norm_squared": float(operator_norm_squared),
        "application_interval": int(application_interval),
        "internal_steps": int(internal_steps),
        "step_scale": float(step_scale),
        "max_backtracking_steps": int(max_backtracking_steps),
    }
    dx_squared = float(operator["dx_m"]) ** 2

    def radial_adjoint_pure_phase_constraint(probe: np.ndarray) -> np.ndarray:
        target = np.asarray(probe, dtype=np.complex128)
        if target.shape != tuple(operator["interpolation"]["shape"]):
            msg = "probe must match the radial operator Cartesian shape."
            raise ValueError(msg)
        call_index = int(state["call_count"])
        state["call_count"] = call_index + 1
        if call_index % application_interval != 0:
            return target.copy()

        current = np.asarray(state["source_transmission"], dtype=np.complex128)
        last_gain = 1.0 + 0.0j
        for _ in range(internal_steps):
            model = _radial_fresnel_full_field(operator, current)
            gain = _fit_global_complex_gain(model, target)
            residual = gain * model - target
            objective_before = float(
                0.5 * dx_squared * np.sum(np.abs(residual) ** 2)
            )
            gradient = _radial_fresnel_weighted_adjoint(
                operator, np.conj(gain) * residual
            )
            phase_gradient = np.imag(np.conj(current) * gradient)
            initial_step = step_scale / max(
                abs(gain) ** 2 * operator_norm_squared,
                np.finfo(float).eps,
            )
            accepted = False
            step = initial_step
            candidate = current
            candidate_gain = gain
            objective_after = objective_before
            for _ in range(max_backtracking_steps + 1):
                proposed = np.exp(
                    1j * (np.angle(current) - step * phase_gradient)
                ).astype(np.complex128)
                proposed_model = _radial_fresnel_full_field(
                    operator, proposed
                )
                proposed_gain = _fit_global_complex_gain(
                    proposed_model, target
                )
                proposed_residual = proposed_gain * proposed_model - target
                proposed_objective = float(
                    0.5
                    * dx_squared
                    * np.sum(np.abs(proposed_residual) ** 2)
                )
                tolerance = 1e-12 * max(
                    objective_before, np.finfo(float).eps
                )
                if proposed_objective <= objective_before + tolerance:
                    candidate = proposed
                    candidate_gain = proposed_gain
                    objective_after = proposed_objective
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                step = 0.0
            current = candidate
            last_gain = candidate_gain
            state["objective_before"].append(objective_before)
            state["objective_after"].append(objective_after)
            state["accepted_step"].append(float(step))
            state["global_gain"].append(complex(last_gain))

        state["source_transmission"] = current
        state["application_call_index"].append(call_index)
        constrained_model = _radial_fresnel_full_field(operator, current)
        last_gain = _fit_global_complex_gain(constrained_model, target)
        return (last_gain * constrained_model).astype(
            np.complex128, copy=False
        )

    return radial_adjoint_pure_phase_constraint, state


def _sample_radial_probe_batch(
    radial_result: dict[str, Any],
    shape: tuple[int, int],
    dx_m: float,
    center_xy_m: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Interpolate continuous radial probes onto one Cartesian output grid."""

    radial_coordinate = np.asarray(radial_result["probe_radius_m"])
    interpolation = _radial_interpolation_plan(
        radial_coordinate, shape, dx_m, center_xy_m
    )
    sampled: dict[str, np.ndarray] = {}
    for label, radial_field in zip(
        radial_result["case_labels"],
        np.asarray(radial_result["probe_radial"]),
        strict=True,
    ):
        sampled[str(label)] = _apply_radial_interpolation(
            np.asarray(radial_field, dtype=np.complex128), interpolation
        )
    return sampled


def _waist_step_label(index: int, sign: str) -> str:
    return f"waist_step_{index:02d}_{sign}"


def _step_convergence_from_cases(
    steps_m: np.ndarray,
    probes: dict[str, np.ndarray],
    config: dict[str, Any],
    sample_b: np.ndarray,
    scan_positions: np.ndarray,
    dx_m: float,
    transfer_bc: np.ndarray,
    baseline_probe: np.ndarray,
    baseline_intensity: np.ndarray,
    d_waist_m: float,
) -> dict[str, np.ndarray]:
    """Evaluate finite-difference convergence from paired precomputed cases."""

    steps = np.asarray(steps_m, dtype=np.float64)
    if (
        steps.ndim != 1
        or steps.size < 2
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0.0)
        or not np.all(np.diff(steps) < 0.0)
    ):
        msg = "delta_d_waist_steps_m must be finite, positive, and decreasing."
        raise ValueError(msg)
    probe_values: list[float] = []
    intensity_values: list[float] = []
    for index, step in enumerate(steps):
        minus_label = _waist_step_label(index, "minus")
        plus_label = _waist_step_label(index, "plus")
        minus_intensity = _simulate_probe_detector(
            config,
            probes[minus_label],
            sample_b,
            scan_positions,
            dx_m,
            transfer_bc,
        )
        plus_intensity = _simulate_probe_detector(
            config,
            probes[plus_label],
            sample_b,
            scan_positions,
            dx_m,
            transfer_bc,
        )
        _, probe_value = normalized_complex_sensitivity(
            probes[minus_label],
            probes[plus_label],
            baseline_probe,
            float(step),
            d_waist_m,
        )
        _, intensity_value = normalized_real_sensitivity(
            minus_intensity,
            plus_intensity,
            baseline_intensity,
            float(step),
            d_waist_m,
        )
        probe_values.append(probe_value)
        intensity_values.append(intensity_value)
        del minus_intensity, plus_intensity
        gc.collect()
    probe_array = np.asarray(probe_values, dtype=np.float64)
    intensity_array = np.asarray(intensity_values, dtype=np.float64)
    return {
        "delta_d_waist_m": steps,
        "normalized_probe_sensitivity": probe_array,
        "normalized_intensity_sensitivity": intensity_array,
        "probe_successive_relative_change": successive_relative_changes(
            probe_array
        ),
        "intensity_successive_relative_change": successive_relative_changes(
            intensity_array
        ),
    }


def _initial_object(
    shape: tuple[int, int], phase_std_rad: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase = rng.normal(scale=phase_std_rad, size=shape)
    return np.exp(1j * phase).astype(np.complex128)


def _legacy_coarse_a_plane_constraint(
    probe: np.ndarray,
    incident: np.ndarray,
    reference_mask: np.ndarray,
    dx_m: float,
    wavelength_m: float,
    z_ab_m: float,
) -> np.ndarray:
    """Apply the former coarse ASM A-plane pure-phase projection."""

    recovered = recover_thin_phase_A(
        probe,
        incident,
        reference_mask,
        dx_m,
        wavelength_m,
        z_ab_m,
    )
    return angular_spectrum_propagate(
        np.asarray(recovered["A_rec_phase_only"]) * incident,
        dx_m,
        wavelength_m,
        z_ab_m,
    )


def _reconstruction_probe(reconstruction: dict[str, Any]) -> np.ndarray:
    if "P_B_rec_raw" in reconstruction:
        return np.asarray(reconstruction["P_B_rec_raw"], dtype=np.complex128)
    return np.asarray(
        reconstruction["P_B_fixed_simulation_diagnostic_only"],
        dtype=np.complex128,
    )


def _reconstruction_object(reconstruction: dict[str, Any]) -> np.ndarray:
    if "B_rec_raw" in reconstruction:
        return np.asarray(reconstruction["B_rec_raw"], dtype=np.complex128)
    return np.asarray(
        reconstruction["B_fixed_simulation_diagnostic_only"],
        dtype=np.complex128,
    )


def _reconstruct_case(
    config: dict[str, Any],
    intensity: np.ndarray,
    scan_positions: np.ndarray,
    incident: np.ndarray,
    reference_mask: np.ndarray,
    dx_m: float,
    *,
    variant_id: str,
    num_iters: int,
    sample_b_true_diagnostic: np.ndarray | None = None,
    probe_true_diagnostic: np.ndarray | None = None,
    radial_operator: dict[str, Any] | None = None,
    radial_operator_norm_squared: float | None = None,
    beta_probe_override: float | None = None,
    beta_object_override: float | None = None,
    normalization_mode: str = "variant_default",
    correction_mode_override: str | None = None,
    denominator_mode_override: str | None = None,
    object_boundary_override: str | None = None,
    checkpoint_iters: tuple[int, ...] | None = None,
    resume_state: dict[str, Any] | None = None,
    resume_constraint_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one explicitly labelled Stage D operator-consistency ablation."""

    optics = config["optics"]
    recon = config["reconstruction"]
    ablation = recon["operator_consistency_ablation"]
    wavelength = float(optics["wavelength_m"])
    z_ab = float(optics["z_AB_m"])
    z_bc = float(optics["z_BC_m"])
    measurement_probe_init = initialize_probe_by_detector_backpropagation(
        intensity, dx_m, wavelength, z_bc
    )
    blind_object_init = _initial_object(
        intensity.shape[1:],
        float(recon["initial_object_phase_std_rad"]),
        int(recon["initial_object_seed"]),
    )
    probe_norm_target = float(
        np.sqrt(np.mean(np.sum(intensity, axis=(1, 2), dtype=np.float64)))
    )

    valid_variants = {
        "known_b_probe_only",
        "known_probe_object_only",
        "blind_unconstrained_with_energy_norm",
        "blind_unconstrained",
        "blind_legacy_coarse_a_constraint",
        "blind_radial_output_range_constraint",
        "blind_radial_adjoint_constraint",
    }
    if variant_id not in valid_variants:
        msg = f"Unknown Stage D ablation variant: {variant_id}."
        raise ValueError(msg)

    uses_truth_b = variant_id == "known_b_probe_only"
    uses_truth_probe = variant_id == "known_probe_object_only"
    if uses_truth_b:
        if sample_b_true_diagnostic is None:
            msg = "known_b_probe_only requires a labelled truth-B diagnostic input."
            raise ValueError(msg)
        object_init = np.asarray(
            sample_b_true_diagnostic, dtype=np.complex128
        ).copy()
    else:
        object_init = blind_object_init
    if uses_truth_probe:
        if probe_true_diagnostic is None:
            msg = "known_probe_object_only requires a labelled truth-probe input."
            raise ValueError(msg)
        probe_init = np.asarray(probe_true_diagnostic, dtype=np.complex128).copy()
    else:
        probe_init = measurement_probe_init

    update_probe = not uses_truth_probe
    effective_update_object = not uses_truth_b
    beta_probe = float(
        recon["beta_probe"]
        if beta_probe_override is None
        else beta_probe_override
    )
    configured_beta_object = float(
        recon["beta_object"]
        if beta_object_override is None
        else beta_object_override
    )
    beta_object = 0.0 if uses_truth_b else configured_beta_object
    apply_energy_norm = variant_id in {
        "blind_unconstrained_with_energy_norm",
        "blind_legacy_coarse_a_constraint",
    }
    if normalization_mode == "none":
        probe_norm = None
    elif normalization_mode == "measurement_energy":
        probe_norm = probe_norm_target
    elif normalization_mode == "truth_probe_norm_simulation_diagnostic_only":
        if probe_true_diagnostic is None:
            msg = "Truth-probe normalization requires a labelled truth probe."
            raise ValueError(msg)
        probe_norm = _norm(np.asarray(probe_true_diagnostic))
    elif normalization_mode == "variant_default":
        probe_norm = probe_norm_target if apply_energy_norm else None
    else:
        msg = f"Unknown normalization mode: {normalization_mode}."
        raise ValueError(msg)
    probe_constraint: Callable[[np.ndarray], np.ndarray] | None = None
    radial_constraint_state: dict[str, Any] | None = None
    radial_range_state: dict[str, Any] | None = None
    if variant_id == "blind_legacy_coarse_a_constraint":

        def legacy_constraint(probe: np.ndarray) -> np.ndarray:
            return _legacy_coarse_a_plane_constraint(
                probe,
                incident,
                reference_mask,
                dx_m,
                wavelength,
                z_ab,
            )

        probe_constraint = legacy_constraint
    elif variant_id == "blind_radial_output_range_constraint":
        if radial_operator is None:
            msg = "The radial-output variant requires the shared interpolation."
            raise ValueError(msg)
        range_cfg = ablation["radial_output_range_constraint"]
        probe_constraint, radial_range_state = (
            _make_radial_output_range_constraint(
                radial_operator["interpolation"],
                float(range_cfg["ridge_fraction"]),
            )
        )
        if resume_constraint_state is not None:
            radial_range_state["call_count"] = int(
                resume_constraint_state["call_count"]
            )
            radial_range_state["last_relative_change"] = float(
                resume_constraint_state["last_relative_change"]
            )
    elif variant_id == "blind_radial_adjoint_constraint":
        if radial_operator is None or radial_operator_norm_squared is None:
            msg = "The radial-adjoint variant requires its paired operator and norm."
            raise ValueError(msg)
        radial_cfg = ablation["radial_constraint"]
        interval = int(radial_cfg["application_interval"])
        if num_iters % interval != 0:
            msg = "A radial-constraint run must end on a constraint application."
            raise ValueError(msg)
        probe_constraint, radial_constraint_state = (
            _make_radial_adjoint_constraint(
                radial_operator,
                operator_norm_squared=radial_operator_norm_squared,
                application_interval=interval,
                internal_steps=int(radial_cfg["internal_steps"]),
                step_scale=float(radial_cfg["step_scale"]),
                max_backtracking_steps=int(
                    radial_cfg["max_backtracking_steps"]
                ),
            )
        )

    bounds = tuple(float(value) for value in recon["object_amplitude_bounds"])
    def checkpoint_with_constraint_state(checkpoint: dict[str, Any]) -> None:
        if checkpoint_callback is None:
            return
        if radial_range_state is not None:
            checkpoint["constraint_state"] = {
                "call_count": int(radial_range_state["call_count"]),
                "last_relative_change": float(
                    radial_range_state["last_relative_change"]
                ),
            }
        checkpoint_callback(checkpoint)

    result = epie_reconstruct(
        intensity,
        scan_positions,
        dx=dx_m,
        wavelength=wavelength,
        z_BC=z_bc,
        num_iters=int(num_iters),
        beta_probe=beta_probe,
        beta_object=beta_object,
        init_probe=None if resume_state is not None else probe_init,
        init_object=None if resume_state is not None else object_init,
        update_probe=update_probe,
        update_object=effective_update_object,
        shuffle_positions=bool(recon.get("shuffle_positions", True)),
        seed=int(recon["shuffle_seed"]),
        object_amplitude_bounds=(bounds[0], bounds[1]),
        probe_l2_norm_target=probe_norm,
        probe_constraint=probe_constraint,
        correction_mode=(
            str(recon.get("correction_mode", "adjoint_residual"))
            if correction_mode_override is None
            else correction_mode_override
        ),
        denominator_mode=(
            str(recon.get("denominator_mode", "epie"))
            if denominator_mode_override is None
            else denominator_mode_override
        ),
        rpie_alpha_probe=float(recon.get("rpie_alpha_probe", 0.1)),
        rpie_alpha_object=float(recon.get("rpie_alpha_object", 0.1)),
        object_boundary=(
            str(recon.get("object_boundary", "periodic"))
            if object_boundary_override is None
            else object_boundary_override
        ),
        checkpoint_iters=checkpoint_iters,
        resume_state=resume_state,
        checkpoint_callback=(
            checkpoint_with_constraint_state
            if checkpoint_callback is not None
            else None
        ),
        show_progress=bool(recon.get("show_progress", False)),
    )
    probe_rec = np.asarray(result["P_B_rec"], dtype=np.complex128)
    object_rec = np.asarray(result["B_rec"], dtype=np.complex128)
    fixed_object_change = (
        float(np.max(np.abs(object_rec - object_init))) if uses_truth_b else 0.0
    )
    fixed_probe_change = (
        float(np.max(np.abs(probe_rec - probe_init)))
        if uses_truth_probe
        else 0.0
    )
    if fixed_object_change > 1e-12 or fixed_probe_change > 1e-12:
        msg = "A truth-assisted fixed-field diagnostic changed its fixed field."
        raise FloatingPointError(msg)
    a_result = recover_thin_phase_A(
        probe_rec,
        incident,
        reference_mask,
        dx_m,
        wavelength,
        z_ab,
    )
    output: dict[str, Any] = {
        "variant_id": variant_id,
        "scientific_role": (
            "simulation_diagnostic_only"
            if uses_truth_b or uses_truth_probe
            else "blind_reconstruction_ablation"
        ),
        "uses_simulation_truth_B_as_input": uses_truth_b,
        "uses_simulation_truth_probe_as_input": uses_truth_probe,
        "P_B_init": (
            np.asarray(resume_state["P_B_rec"], dtype=np.complex128)
            if resume_state is not None
            else probe_init
        ),
        "B_init": (
            np.asarray(resume_state["B_rec"], dtype=np.complex128)
            if resume_state is not None
            else object_init
        ),
        "coarse_A_plane_diagnostic": {
            "A_rec_raw": a_result["A_rec_raw"],
            "A_rec_reference_corrected": a_result[
                "A_rec_reference_corrected"
            ],
            "A_rec_phase_only": a_result["A_rec_phase_only"],
            "reference_correction": a_result["reference_correction"],
            "scientific_role": (
                "diagnostic_only_not_the_authoritative_radial_inverse"
            ),
        },
        "loss_curve": np.asarray(result["loss_curve"], dtype=np.float64),
        "initial_data_fidelity_loss": float(
            result["initial_data_fidelity_loss"]
        ),
        "final_data_fidelity_loss": float(result["final_data_fidelity_loss"]),
        "illumination_map": result["illumination_map"],
        "settings": {
            **result["metadata"],
            "variant_id": variant_id,
            "effective_update_object": effective_update_object,
            "energy_based_probe_norm_applied": bool(
                normalization_mode == "measurement_energy"
                or (normalization_mode == "variant_default" and apply_energy_norm)
            ),
            "probe_norm_constraint_applied": probe_norm is not None,
            "normalization_mode": normalization_mode,
            "uses_simulation_truth_B_as_input": uses_truth_b,
            "uses_simulation_truth_probe_as_input": uses_truth_probe,
            "resumed": resume_state is not None,
            "resumed_from_iteration": (
                int(resume_state["completed_iterations"])
                if resume_state is not None
                else 0
            ),
        },
        "probe_l2_norm_target_from_measurements": probe_norm_target,
        "fixed_object_max_abs_change": fixed_object_change,
        "fixed_probe_max_abs_change": fixed_probe_change,
    }
    if result["checkpoints"]:
        output["checkpoints"] = result["checkpoints"]
    if update_probe:
        output["P_B_rec_raw"] = probe_rec
    else:
        output["P_B_fixed_simulation_diagnostic_only"] = probe_rec
    if effective_update_object:
        output["B_rec_raw"] = object_rec
    else:
        output["B_fixed_simulation_diagnostic_only"] = object_rec
    if radial_constraint_state is not None:
        source_final = np.asarray(
            radial_constraint_state["source_transmission"],
            dtype=np.complex128,
        )
        output["radial_constraint"] = {
            "source_transmission_final": source_final,
            "source_amplitude_max_abs_error": float(
                np.max(np.abs(np.abs(source_final) - 1.0))
            ),
            "call_count": int(radial_constraint_state["call_count"]),
            "application_call_index": np.asarray(
                radial_constraint_state["application_call_index"],
                dtype=np.int64,
            ),
            "objective_before": np.asarray(
                radial_constraint_state["objective_before"], dtype=np.float64
            ),
            "objective_after": np.asarray(
                radial_constraint_state["objective_after"], dtype=np.float64
            ),
            "accepted_step": np.asarray(
                radial_constraint_state["accepted_step"], dtype=np.float64
            ),
            "global_gain": np.asarray(
                radial_constraint_state["global_gain"], dtype=np.complex128
            ),
            "operator_norm_squared": float(
                radial_constraint_state["operator_norm_squared"]
            ),
            "application_interval": int(
                radial_constraint_state["application_interval"]
            ),
            "internal_steps": int(radial_constraint_state["internal_steps"]),
            "adjoint_convention": str(radial_operator["adjoint_convention"]),
        }
    if radial_range_state is not None:
        output["radial_output_range_constraint"] = {
            "call_count": int(radial_range_state["call_count"]),
            "ridge_fraction": float(radial_range_state["ridge_fraction"]),
            "ridge_absolute": float(radial_range_state["ridge_absolute"]),
            "normal_diagonal_min": float(
                radial_range_state["normal_diagonal_min"]
            ),
            "normal_diagonal_max": float(
                radial_range_state["normal_diagonal_max"]
            ),
            "active_radial_node_count": int(
                radial_range_state["active_radial_node_count"]
            ),
            "radial_node_count": int(radial_range_state["radial_node_count"]),
            "last_relative_change": float(
                radial_range_state["last_relative_change"]
            ),
            "scientific_role": (
                "B_plane_axisymmetric_range_constraint_not_an_A_plane_inverse"
            ),
        }
    return output


def _evaluate_reconstruction_case(
    reconstruction: dict[str, Any],
    probe_true: np.ndarray,
    object_true: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Align one Stage D result to truth only for simulation evaluation."""

    probe_raw = _reconstruction_probe(reconstruction)
    object_raw = _reconstruction_object(reconstruction)
    probe_aligned, probe_gain, probe_ramp = align_affine_phase_and_complex_gain(
        probe_raw, probe_true
    )
    object_aligned, object_gain, object_ramp = (
        align_affine_phase_and_complex_gain(
            object_raw, object_true
        )
    )
    evaluation = {
        "P_B_rec_aligned_to_truth": probe_aligned,
        "B_rec_aligned_to_truth": object_aligned,
        "alignment_parameters": {
            "P_B_complex_gain": probe_gain,
            "P_B_phase_ramp_yx_rad_per_px": probe_ramp,
            "B_complex_gain": object_gain,
            "B_phase_ramp_yx_rad_per_px": object_ramp,
        },
    }
    metrics = {
        "P_B_rec_raw_relative_l2_to_truth_unaligned": relative_l2(
            probe_raw, probe_true
        ),
        "P_B_rec_aligned_complex_relative_error": complex_relative_error(
            probe_aligned, probe_true
        ),
        "B_rec_aligned_complex_relative_error": complex_relative_error(
            object_aligned, object_true
        ),
        "initial_data_fidelity_loss": float(
            reconstruction["initial_data_fidelity_loss"]
        ),
        "final_data_fidelity_loss": float(
            reconstruction["final_data_fidelity_loss"]
        ),
    }
    return evaluation, metrics


def _apply_affine_alignment(
    field: np.ndarray,
    gain: complex,
    ramp_yx_rad_per_px: tuple[float, float],
) -> np.ndarray:
    """Apply one previously estimated affine gauge transform."""

    values = np.asarray(field, dtype=np.complex128)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    ky, kx = (float(value) for value in ramp_yx_rad_per_px)
    return (
        complex(gain) * values * np.exp(1j * (ky * yy + kx * xx))
    ).astype(np.complex128)


def _loss_tail_slope(loss_curve: np.ndarray, tail_length: int = 20) -> float:
    """Return the least-squares loss slope over the final iterations."""

    values = np.asarray(loss_curve, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        return 0.0
    tail = values[-min(tail_length, values.size) :]
    coordinate = np.arange(tail.size, dtype=np.float64)
    centered = coordinate - np.mean(coordinate)
    denominator = float(np.sum(centered**2))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(centered * (tail - np.mean(tail))) / denominator)


def _checkpoint_truth_metrics(
    reconstruction: dict[str, Any],
    probe_true: np.ndarray,
    object_true: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate saved optimizer checkpoints without feeding truth into updates."""

    metrics: dict[str, dict[str, float]] = {}
    for iteration, checkpoint in reconstruction.get("checkpoints", {}).items():
        probe = np.asarray(checkpoint["P_B_rec"], dtype=np.complex128)
        object_b = np.asarray(checkpoint["B_rec"], dtype=np.complex128)
        probe_aligned, _, _ = align_affine_phase_and_complex_gain(
            probe, probe_true
        )
        object_aligned, _, _ = align_affine_phase_and_complex_gain(
            object_b, object_true
        )
        evaluation = {
            "P_B_aligned_complex_relative_error": complex_relative_error(
                probe_aligned, probe_true
            ),
            "B_aligned_complex_relative_error": complex_relative_error(
                object_aligned, object_true
            ),
        }
        checkpoint["simulation_evaluation_only"] = evaluation
        metrics[str(iteration)] = {
            **evaluation,
            "frozen_data_fidelity_loss": float(
                checkpoint["data_fidelity_loss"]
            ),
        }
    return metrics


def _reconstruction_metric_summary(
    reconstruction: dict[str, Any],
    probe_true: np.ndarray,
    object_true: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach truth-labelled evaluation and return a compact metric summary."""

    evaluation, truth_metrics = _evaluate_reconstruction_case(
        reconstruction, probe_true, object_true
    )
    checkpoint_metrics = _checkpoint_truth_metrics(
        reconstruction, probe_true, object_true
    )
    output = {
        **reconstruction,
        "simulation_evaluation_only": evaluation,
    }
    metrics = {
        "initial_data_fidelity_loss": float(
            reconstruction["initial_data_fidelity_loss"]
        ),
        "final_data_fidelity_loss": float(
            reconstruction["final_data_fidelity_loss"]
        ),
        "loss_tail_slope_per_iteration": _loss_tail_slope(
            np.asarray(reconstruction["loss_curve"])
        ),
        "simulation_evaluation_only": truth_metrics,
        "checkpoints": checkpoint_metrics,
    }
    return output, metrics


def _optimizer_case_id(prefix: str, value: float) -> str:
    token = f"{value:.6g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_{token}"


def _run_optimizer_study(
    config: dict[str, Any],
    intensity: np.ndarray,
    scan_positions: np.ndarray,
    incident: np.ndarray,
    reference_mask: np.ndarray,
    dx_m: float,
    probe_true: np.ndarray,
    object_true: np.ndarray,
    transfer_bc: np.ndarray,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
]:
    """Run bounded Stage-D optimizer diagnostics and checkpointed controls."""

    recon = config["reconstruction"]
    ablation = recon["operator_consistency_ablation"]
    study = ablation["optimizer_study"]
    optics = config["optics"]
    wavelength = float(optics["wavelength_m"])
    z_bc = float(optics["z_BC_m"])
    correction_mode = str(study.get("correction_mode", "adjoint_residual"))
    runtime_budget_s = float(study.get("runtime_budget_s", 6000.0))
    planning_seconds_per_iteration = float(
        study.get("planning_seconds_per_iteration", 2.0)
    )
    started = time.perf_counter()
    timed_iterations = 0
    timed_seconds = 0.0
    timing_metrics: dict[str, dict[str, float]] = {}

    def timed_case(
        case_id: str, *, num_iters: int, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal timed_iterations, timed_seconds
        case_started = time.perf_counter()
        result = _reconstruct_case(
            config,
            intensity=kwargs.pop("case_intensity", intensity),
            scan_positions=scan_positions,
            incident=incident,
            reference_mask=reference_mask,
            dx_m=dx_m,
            num_iters=num_iters,
            correction_mode_override=correction_mode,
            **kwargs,
        )
        elapsed = time.perf_counter() - case_started
        timing_metrics[case_id] = {
            "elapsed_s": elapsed,
            "num_iters": int(num_iters),
            "seconds_per_iteration": elapsed / max(num_iters, 1),
        }
        timed_iterations += num_iters
        timed_seconds += elapsed
        return result

    def projected_runtime_s(num_iters: int) -> float:
        seconds_per_iteration = (
            timed_seconds / timed_iterations
            if timed_iterations > 0
            else planning_seconds_per_iteration
        )
        return seconds_per_iteration * num_iters

    outputs: dict[str, Any] = {
        "scientific_role": "simulation_diagnostic_only",
        "correction_mode": correction_mode,
    }
    metrics: dict[str, Any] = {
        "status": "running",
        "scientific_role": "simulation_diagnostic_only",
        "selection_uses_truth": False,
        "runtime_budget_s": runtime_budget_s,
    }
    plot_fields: list[np.ndarray] = []
    plot_losses: list[np.ndarray] = []
    plot_labels: list[str] = []

    fixed_outputs: dict[str, Any] = {}
    fixed_metrics: dict[str, Any] = {}
    for mode in ("legacy_inverse_difference", "adjoint_residual"):
        for target in ("object_only", "probe_only"):
            fixed_started = time.perf_counter()
            update_probe = target == "probe_only"
            update_object = target == "object_only"
            result = epie_reconstruct(
                intensity,
                scan_positions,
                dx_m,
                wavelength,
                z_bc,
                num_iters=1,
                beta_probe=float(recon["beta_probe"]),
                beta_object=float(recon["beta_object"]),
                init_probe=probe_true,
                init_object=object_true,
                update_probe=update_probe,
                update_object=update_object,
                shuffle_positions=False,
                object_amplitude_bounds=(1.0, 1.0),
                correction_mode=mode,
                object_boundary="periodic",
                show_progress=False,
            )
            elapsed = time.perf_counter() - fixed_started
            case_id = f"{mode}_{target}"
            probe_change = relative_l2(result["P_B_rec"], probe_true)
            object_change = relative_l2(result["B_rec"], object_true)
            fixed_outputs[case_id] = {
                "P_B_after_one_iteration": result["P_B_rec"],
                "B_after_one_iteration": result["B_rec"],
                "loss_curve": result["loss_curve"],
                "scientific_role": "truth_initialized_fixed_point_diagnostic",
            }
            fixed_metrics[case_id] = {
                "initial_data_fidelity_loss": float(
                    result["initial_data_fidelity_loss"]
                ),
                "final_data_fidelity_loss": float(
                    result["final_data_fidelity_loss"]
                ),
                "probe_relative_change": probe_change,
                "object_relative_change": object_change,
                "elapsed_s": elapsed,
            }
    outputs["truth_fixed_point"] = fixed_outputs
    metrics["truth_fixed_point"] = fixed_metrics

    known_probe_cfg = study["known_probe_object_only"]
    known_probe_iters = int(known_probe_cfg["diagnostic_num_iters"])
    known_probe_result = timed_case(
        "known_probe_object_only",
        num_iters=known_probe_iters,
        variant_id="known_probe_object_only",
        sample_b_true_diagnostic=object_true,
        probe_true_diagnostic=probe_true,
        beta_object_override=float(
            known_probe_cfg.get("beta_object", recon["beta_object"])
        ),
        normalization_mode="none",
        object_boundary_override="periodic",
        checkpoint_iters=tuple(
            int(value) for value in known_probe_cfg["checkpoint_iterations"]
        ),
    )
    known_probe_output, known_probe_metrics = _reconstruction_metric_summary(
        known_probe_result, probe_true, object_true
    )
    outputs["known_probe_object_only"] = known_probe_output
    metrics["known_probe_object_only"] = known_probe_metrics
    plot_fields.append(_reconstruction_probe(known_probe_result))
    plot_losses.append(np.asarray(known_probe_result["loss_curve"]))
    plot_labels.append(f"known probe, {known_probe_iters} iters")

    known_b_cfg = study["known_b_probe_only"]
    screening_iters = int(known_b_cfg["screening_num_iters"])
    screening_outputs: dict[str, Any] = {}
    screening_metrics: dict[str, Any] = {}
    candidate_losses: dict[str, float] = {}
    for beta in (float(value) for value in known_b_cfg["beta_probe_candidates"]):
        case_id = _optimizer_case_id("beta", beta)
        result = timed_case(
            f"known_b_screening_{case_id}",
            num_iters=screening_iters,
            variant_id="known_b_probe_only",
            sample_b_true_diagnostic=object_true,
            probe_true_diagnostic=probe_true,
            beta_probe_override=beta,
            normalization_mode="none",
            object_boundary_override="periodic",
        )
        output, result_metrics = _reconstruction_metric_summary(
            result, probe_true, object_true
        )
        output["beta_probe"] = beta
        screening_outputs[case_id] = output
        screening_metrics[case_id] = {
            **result_metrics,
            "beta_probe": beta,
        }
        candidate_losses[case_id] = float(result["final_data_fidelity_loss"])
    selected_beta_id = min(candidate_losses, key=candidate_losses.get)
    selected_beta = float(screening_outputs[selected_beta_id]["beta_probe"])

    normalization_iters = int(known_b_cfg["normalization_screening_num_iters"])
    normalization_outputs: dict[str, Any] = {}
    normalization_metrics: dict[str, Any] = {}
    selectable_normalization_losses: dict[str, float] = {}
    for normalization_mode in (
        str(value) for value in known_b_cfg["normalization_modes"]
    ):
        result = timed_case(
            f"known_b_normalization_{normalization_mode}",
            num_iters=normalization_iters,
            variant_id="known_b_probe_only",
            sample_b_true_diagnostic=object_true,
            probe_true_diagnostic=probe_true,
            beta_probe_override=selected_beta,
            normalization_mode=normalization_mode,
            object_boundary_override="periodic",
        )
        output, result_metrics = _reconstruction_metric_summary(
            result, probe_true, object_true
        )
        normalization_outputs[normalization_mode] = output
        normalization_metrics[normalization_mode] = result_metrics
        if normalization_mode != "truth_probe_norm_simulation_diagnostic_only":
            selectable_normalization_losses[normalization_mode] = float(
                result["final_data_fidelity_loss"]
            )
    selected_normalization = min(
        selectable_normalization_losses,
        key=selectable_normalization_losses.get,
    )

    boundary_cfg = study["boundary_controls"]
    boundary_iters = int(boundary_cfg["diagnostic_num_iters"])
    constant_intensity = _simulate_probe_detector(
        config,
        probe_true,
        object_true,
        scan_positions,
        dx_m,
        transfer_bc,
        object_boundary="constant",
        object_boundary_value=1.0 + 0.0j,
    )
    boundary_outputs: dict[str, Any] = {}
    boundary_metrics: dict[str, Any] = {
        "constant_vs_periodic_I_stack_relative_l2": relative_l2(
            constant_intensity, intensity
        )
    }
    boundary_cases = {
        "periodic_data_periodic_reconstruction": (intensity, "periodic"),
        "periodic_data_constant_reconstruction_mismatch": (
            intensity,
            "constant",
        ),
        "constant_data_constant_reconstruction": (
            constant_intensity,
            "constant",
        ),
    }
    for case_id, (case_intensity, boundary_mode) in boundary_cases.items():
        result = timed_case(
            f"boundary_{case_id}",
            num_iters=boundary_iters,
            case_intensity=case_intensity,
            variant_id="known_b_probe_only",
            sample_b_true_diagnostic=object_true,
            probe_true_diagnostic=probe_true,
            beta_probe_override=selected_beta,
            normalization_mode=selected_normalization,
            object_boundary_override=boundary_mode,
        )
        output, result_metrics = _reconstruction_metric_summary(
            result, probe_true, object_true
        )
        output["data_boundary"] = (
            "constant" if case_intensity is constant_intensity else "periodic"
        )
        output["reconstruction_boundary"] = boundary_mode
        boundary_outputs[case_id] = output
        boundary_metrics[case_id] = result_metrics

    selected_num_iters = int(known_b_cfg["selected_num_iters"])
    projected_long_s = projected_runtime_s(selected_num_iters)
    elapsed_before_long_s = time.perf_counter() - started
    can_start_long = (
        elapsed_before_long_s + projected_long_s <= runtime_budget_s
    )
    if can_start_long:
        selected_result = timed_case(
            "known_b_selected_trajectory",
            num_iters=selected_num_iters,
            variant_id="known_b_probe_only",
            sample_b_true_diagnostic=object_true,
            probe_true_diagnostic=probe_true,
            beta_probe_override=selected_beta,
            normalization_mode=selected_normalization,
            object_boundary_override="periodic",
            checkpoint_iters=tuple(
                int(value)
                for value in known_b_cfg["checkpoint_iterations"]
            ),
        )
        selected_output, selected_metrics = _reconstruction_metric_summary(
            selected_result, probe_true, object_true
        )
        selected_status = "executed"
        plot_fields.append(_reconstruction_probe(selected_result))
        plot_losses.append(np.asarray(selected_result["loss_curve"]))
        plot_labels.append(f"known B, {selected_num_iters} iters")
    else:
        selected_output = {
            "status": "not_run_runtime_budget",
            "projected_runtime_s": projected_long_s,
        }
        selected_metrics = dict(selected_output)
        selected_status = "not_run_runtime_budget"

    outputs["known_b_probe_only"] = {
        "screening": screening_outputs,
        "normalization": normalization_outputs,
        "selected_trajectory": selected_output,
    }
    outputs["boundary_controls"] = boundary_outputs
    metrics["known_b_probe_only"] = {
        "screening": screening_metrics,
        "selected_beta_probe": selected_beta,
        "selected_beta_case_id": selected_beta_id,
        "beta_selection_metric": "frozen_data_fidelity_loss_measurement_only",
        "normalization": normalization_metrics,
        "selected_normalization": selected_normalization,
        "normalization_selection_uses_truth": False,
        "selected_trajectory": selected_metrics,
        "selected_trajectory_status": selected_status,
    }
    metrics["boundary_controls"] = boundary_metrics
    total_elapsed_s = time.perf_counter() - started
    metrics["timing"] = {
        "total_elapsed_s": total_elapsed_s,
        "runtime_budget_s": runtime_budget_s,
        "cases": timing_metrics,
        "observed_seconds_per_iteration": (
            timed_seconds / timed_iterations if timed_iterations > 0 else 0.0
        ),
        "projected_selected_trajectory_s_before_start": projected_long_s,
    }
    metrics["status"] = (
        "executed" if selected_status == "executed" else "partially_executed"
    )
    outputs["settings"] = {
        "selected_beta_probe": selected_beta,
        "selected_normalization": selected_normalization,
        "selection_uses_truth": False,
        "runtime_budget_s": runtime_budget_s,
    }
    return outputs, metrics, plot_fields, plot_losses, plot_labels


def _json_numpy_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    msg = f"Object of type {type(value).__name__} is not JSON serializable."
    raise TypeError(msg)


def _blind_long_runner_signature(
    config: dict[str, Any], case_id: str
) -> str:
    """Hash settings outside the reusable ePIE problem signature."""

    recon = config["reconstruction"]
    ablation = recon["operator_consistency_ablation"]
    study = ablation["blind_long_study"]
    payload = {
        "format": "exp030_blind_long_checkpoint_v1",
        "case_id": case_id,
        "variant_id": str(study["variant_id"]),
        "ridge_fraction": float(
            ablation["radial_output_range_constraint"]["ridge_fraction"]
        ),
        "beta_probe": float(study.get("beta_probe", recon["beta_probe"])),
        "beta_object": float(study.get("beta_object", recon["beta_object"])),
        "normalization_mode": str(study.get("normalization_mode", "none")),
        "correction_mode": str(recon.get("correction_mode", "adjoint_residual")),
        "denominator_mode": str(recon.get("denominator_mode", "epie")),
        "rpie_alpha_probe": float(recon.get("rpie_alpha_probe", 0.1)),
        "rpie_alpha_object": float(recon.get("rpie_alpha_object", 0.1)),
        "object_boundary": str(recon.get("object_boundary", "periodic")),
        "object_amplitude_bounds": list(recon["object_amplitude_bounds"]),
        "shuffle_positions": bool(recon.get("shuffle_positions", True)),
        "shuffle_seed": int(recon["shuffle_seed"]),
        "initial_object_phase_std_rad": float(
            recon["initial_object_phase_std_rad"]
        ),
        "initial_object_seed": int(recon["initial_object_seed"]),
        "shape": list(config["optics"]["shape"]),
        "dx_m": float(config["optics"]["dx_m"]),
        "wavelength_m": float(config["optics"]["wavelength_m"]),
        "z_BC_m": float(config["optics"]["z_BC_m"]),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_numpy_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_blind_long_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
    *,
    case_id: str,
    runner_signature: str,
    evaluation_metrics: dict[str, float],
) -> None:
    """Atomically persist one interruption-safe optimizer checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        msg = f"Refusing to overwrite an existing checkpoint: {path}."
        raise FileExistsError(msg)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    state = checkpoint["optimizer_state"]
    constraint_state = checkpoint.get("constraint_state", {})
    with h5py.File(temporary, "x") as h5:
        h5.attrs["format"] = "exp030_blind_long_checkpoint_v1"
        h5.attrs["case_id"] = case_id
        h5.attrs["runner_signature"] = runner_signature
        optimizer = h5.require_group("optimizer_state")
        optimizer.create_dataset(
            "P_B_rec", data=np.asarray(state["P_B_rec"], dtype=np.complex128)
        ).attrs["units"] = "arbitrary field amplitude"
        optimizer.create_dataset(
            "B_rec", data=np.asarray(state["B_rec"], dtype=np.complex128)
        ).attrs["units"] = "dimensionless complex transmission"
        optimizer.create_dataset(
            "loss_curve", data=np.asarray(state["loss_curve"], dtype=np.float64)
        ).attrs["units"] = "relative detector amplitude error"
        optimizer.create_dataset(
            "completed_iterations",
            data=np.int64(state["completed_iterations"]),
        ).attrs["units"] = "iteration"
        optimizer.create_dataset(
            "initial_data_fidelity_loss",
            data=np.float64(state["initial_data_fidelity_loss"]),
        )
        optimizer.create_dataset(
            "frozen_data_fidelity_loss",
            data=np.float64(checkpoint["data_fidelity_loss"]),
        )
        optimizer.create_dataset("version", data=np.int64(state["version"]))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        optimizer.create_dataset(
            "rng_bit_generator",
            data=str(state["rng_bit_generator"]),
            dtype=string_dtype,
        )
        optimizer.create_dataset(
            "rng_state_json",
            data=json.dumps(
                state["rng_state"],
                sort_keys=True,
                default=_json_numpy_default,
            ),
            dtype=string_dtype,
        )
        optimizer.create_dataset(
            "problem_signature",
            data=str(state["problem_signature"]),
            dtype=string_dtype,
        )
        constraint = h5.require_group("constraint_state")
        constraint.create_dataset(
            "call_count", data=np.int64(constraint_state.get("call_count", 0))
        )
        constraint.create_dataset(
            "last_relative_change",
            data=np.float64(constraint_state.get("last_relative_change", 0.0)),
        )
        evaluation = h5.require_group("simulation_evaluation_only")
        for name, value in evaluation_metrics.items():
            evaluation.create_dataset(name, data=np.float64(value))
        h5.flush()
    temporary.replace(path)


def _load_blind_long_checkpoint(path: Path) -> dict[str, Any]:
    """Load a durable exp030 checkpoint without modifying its source run."""

    resolved = path.resolve()
    if not resolved.is_file():
        msg = f"Blind-long checkpoint does not exist: {resolved}."
        raise FileNotFoundError(msg)
    source_sha256 = _file_sha256(resolved)
    with h5py.File(resolved, "r") as h5:
        if str(h5.attrs.get("format", "")) != "exp030_blind_long_checkpoint_v1":
            msg = f"Unsupported blind-long checkpoint format: {resolved}."
            raise ValueError(msg)
        optimizer = h5["optimizer_state"]
        state = {
            "version": int(optimizer["version"][()]),
            "completed_iterations": int(
                optimizer["completed_iterations"][()]
            ),
            "P_B_rec": np.asarray(optimizer["P_B_rec"], dtype=np.complex128),
            "B_rec": np.asarray(optimizer["B_rec"], dtype=np.complex128),
            "loss_curve": np.asarray(
                optimizer["loss_curve"], dtype=np.float64
            ),
            "initial_data_fidelity_loss": float(
                optimizer["initial_data_fidelity_loss"][()]
            ),
            "rng_bit_generator": optimizer["rng_bit_generator"].asstr()[()],
            "rng_state": json.loads(optimizer["rng_state_json"].asstr()[()]),
            "problem_signature": optimizer["problem_signature"].asstr()[()],
        }
        evaluation = {
            name: float(dataset[()])
            for name, dataset in h5["simulation_evaluation_only"].items()
        }
        constraint = h5["constraint_state"]
        constraint_state = {
            "call_count": int(constraint["call_count"][()]),
            "last_relative_change": float(
                constraint["last_relative_change"][()]
            ),
        }
        return {
            "source_path": str(resolved),
            "source_sha256": source_sha256,
            "case_id": str(h5.attrs["case_id"]),
            "runner_signature": str(h5.attrs["runner_signature"]),
            "optimizer_state": state,
            "constraint_state": constraint_state,
            "P_B_rec": state["P_B_rec"],
            "B_rec": state["B_rec"],
            "loss_curve": state["loss_curve"],
            "data_fidelity_loss": float(
                optimizer["frozen_data_fidelity_loss"][()]
            ),
            "simulation_evaluation_only": evaluation,
        }


def _checkpoint_reconstruction_payload(
    checkpoint: dict[str, Any], *, case_id: str
) -> dict[str, Any]:
    """Promote one raw checkpoint to a reconstruction case payload."""

    state = checkpoint["optimizer_state"]
    return {
        "case_id": case_id,
        "scientific_role": "blind_reconstruction_ablation",
        "uses_simulation_truth_B_as_input": False,
        "uses_simulation_truth_probe_as_input": False,
        "P_B_rec_raw": np.asarray(state["P_B_rec"], dtype=np.complex128),
        "B_rec_raw": np.asarray(state["B_rec"], dtype=np.complex128),
        "loss_curve": np.asarray(state["loss_curve"], dtype=np.float64),
        "initial_data_fidelity_loss": float(
            state["initial_data_fidelity_loss"]
        ),
        "final_data_fidelity_loss": float(checkpoint["data_fidelity_loss"]),
        "completed_iterations": int(state["completed_iterations"]),
    }


def _blind_checkpoint_evaluation(
    checkpoint: dict[str, Any],
    *,
    probe_true: np.ndarray,
    object_true: np.ndarray,
    true_probe_case_separation: float,
    true_detector_amplitude_separation: float,
) -> dict[str, float]:
    state = checkpoint["optimizer_state"]
    probe = np.asarray(state["P_B_rec"], dtype=np.complex128)
    object_b = np.asarray(state["B_rec"], dtype=np.complex128)
    probe_aligned, _, _ = align_affine_phase_and_complex_gain(probe, probe_true)
    object_aligned, _, _ = align_affine_phase_and_complex_gain(
        object_b, object_true
    )
    probe_error = complex_relative_error(probe_aligned, probe_true)
    object_error = complex_relative_error(object_aligned, object_true)
    frozen_loss = float(checkpoint["data_fidelity_loss"])
    return {
        "P_B_aligned_complex_relative_error": probe_error,
        "B_aligned_complex_relative_error": object_error,
        "aligned_probe_error_to_true_case_separation_ratio": float(
            probe_error
            / max(true_probe_case_separation, np.finfo(float).eps)
        ),
        "frozen_data_fidelity_loss": frozen_loss,
        "frozen_loss_to_true_detector_amplitude_separation_ratio": float(
            frozen_loss
            / max(true_detector_amplitude_separation, np.finfo(float).eps)
        ),
        "loss_tail_slope_per_iteration": _loss_tail_slope(
            np.asarray(state["loss_curve"], dtype=np.float64)
        ),
    }


def _run_blind_long_study(
    config: dict[str, Any],
    *,
    run_dir: Path,
    resume_checkpoint_path: Path | None,
    scan_positions: np.ndarray,
    incident: np.ndarray,
    reference_mask: np.ndarray,
    dx_m: float,
    radial_operator: dict[str, Any],
    radial_operator_norm_squared: float,
    sample_b: np.ndarray,
    intensity_by_case: dict[str, np.ndarray],
    probe_true_by_case: dict[str, np.ndarray],
    true_probe_case_separation: float,
    true_detector_amplitude_separation: float,
    delta_d_waist_m: float,
    nominal_d_waist_m: float,
    true_normalized_probe_sensitivity: float,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
]:
    """Run the resumable 200/500/1000 blind trajectory and matched cases."""

    recon = config["reconstruction"]
    ablation = recon["operator_consistency_ablation"]
    study = ablation["blind_long_study"]
    variant_id = str(study["variant_id"])
    if variant_id != "blind_radial_output_range_constraint":
        msg = "blind_long_study currently requires the radial B-range variant."
        raise ValueError(msg)
    max_num_iters = int(study["selected_num_iters"])
    checkpoint_iterations = tuple(
        sorted(set(int(value) for value in study["checkpoint_iterations"]))
    )
    if (
        not checkpoint_iterations
        or checkpoint_iterations[-1] != max_num_iters
        or checkpoint_iterations[0] <= 0
    ):
        msg = (
            "blind_long_study checkpoints must be positive and end at "
            "selected_num_iters."
        )
        raise ValueError(msg)
    gate_ratio = float(
        study.get(
            "sensitivity_floor_to_signal_gate",
            ablation["sensitivity_floor_to_signal_gate"],
        )
    )
    if gate_ratio <= 0.0:
        msg = "The blind-long signal gate must be positive."
        raise ValueError(msg)

    durable_root = run_dir / "checkpoints" / "blind_long"
    source_checkpoint = (
        _load_blind_long_checkpoint(resume_checkpoint_path)
        if resume_checkpoint_path is not None
        else None
    )
    if source_checkpoint is not None and source_checkpoint["case_id"] not in {
        "baseline",
        "waist_minus",
        "waist_plus",
    }:
        msg = "The resume checkpoint is not one of the blind-long cases."
        raise ValueError(msg)
    source_root = (
        Path(str(source_checkpoint["source_path"])).parent.parent
        if source_checkpoint is not None
        else None
    )
    resume_provenance: dict[str, Any] = {
        "resumed": source_checkpoint is not None,
    }
    if source_checkpoint is not None:
        resume_provenance.update(
            {
                "source_checkpoint_path": str(source_checkpoint["source_path"]),
                "source_checkpoint_sha256": str(
                    source_checkpoint["source_sha256"]
                ),
                "source_case_id": str(source_checkpoint["case_id"]),
                "source_completed_iteration": int(
                    source_checkpoint["optimizer_state"]["completed_iterations"]
                ),
                "source_run_directory": str(source_root.parent.parent),
            }
        )

    checkpoint_files: dict[str, dict[str, str]] = {}
    progress_path = run_dir / "blind_long_progress.json"
    started = time.perf_counter()
    timing_cases: dict[str, dict[str, float]] = {}

    def validate_loaded(
        loaded: dict[str, Any], expected_case_id: str
    ) -> dict[str, Any]:
        if str(loaded["case_id"]) != expected_case_id:
            msg = (
                f"Checkpoint case {loaded['case_id']} cannot resume "
                f"{expected_case_id}."
            )
            raise ValueError(msg)
        expected = _blind_long_runner_signature(config, expected_case_id)
        if str(loaded["runner_signature"]) != expected:
            msg = f"Checkpoint settings do not match case {expected_case_id}."
            raise ValueError(msg)
        return loaded

    def prior_checkpoint(case_id: str, iteration: int) -> dict[str, Any]:
        if source_root is None:
            msg = "No source checkpoint tree is available for continuation."
            raise FileNotFoundError(msg)
        candidate = source_root / case_id / f"iter_{iteration:04d}.h5"
        return validate_loaded(_load_blind_long_checkpoint(candidate), case_id)

    def callback_for(
        case_id: str, probe_true: np.ndarray
    ) -> Callable[[dict[str, Any]], None]:
        signature = _blind_long_runner_signature(config, case_id)

        def persist(checkpoint: dict[str, Any]) -> None:
            iteration = int(
                checkpoint["optimizer_state"]["completed_iterations"]
            )
            evaluation = _blind_checkpoint_evaluation(
                checkpoint,
                probe_true=probe_true,
                object_true=sample_b,
                true_probe_case_separation=true_probe_case_separation,
                true_detector_amplitude_separation=(
                    true_detector_amplitude_separation
                ),
            )
            checkpoint["simulation_evaluation_only"] = evaluation
            destination = (
                durable_root / case_id / f"iter_{iteration:04d}.h5"
            )
            _save_blind_long_checkpoint(
                destination,
                checkpoint,
                case_id=case_id,
                runner_signature=signature,
                evaluation_metrics=evaluation,
            )
            checkpoint_files.setdefault(case_id, {})[str(iteration)] = str(
                destination
            )
            save_json(
                progress_path,
                {
                    "status": "running",
                    "latest_case_id": case_id,
                    "latest_completed_iteration": iteration,
                    "latest_checkpoint_path": str(destination),
                    "latest_checkpoint_metrics": evaluation,
                    "resume_provenance": resume_provenance,
                },
            )

        return persist

    def run_case(
        case_id: str,
        *,
        num_iters: int,
        checkpoint_iters: tuple[int, ...],
        loaded: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if loaded is not None:
            validate_loaded(loaded, case_id)
        case_started = time.perf_counter()
        result = _reconstruct_case(
            config,
            intensity_by_case[case_id],
            scan_positions,
            incident,
            reference_mask,
            dx_m,
            variant_id=variant_id,
            num_iters=num_iters,
            sample_b_true_diagnostic=sample_b,
            probe_true_diagnostic=probe_true_by_case[case_id],
            radial_operator=radial_operator,
            radial_operator_norm_squared=radial_operator_norm_squared,
            beta_probe_override=float(
                study.get("beta_probe", recon["beta_probe"])
            ),
            beta_object_override=float(
                study.get("beta_object", recon["beta_object"])
            ),
            normalization_mode=str(study.get("normalization_mode", "none")),
            object_boundary_override=str(
                study.get("object_boundary", recon.get("object_boundary", "periodic"))
            ),
            checkpoint_iters=checkpoint_iters,
            resume_state=(loaded["optimizer_state"] if loaded is not None else None),
            resume_constraint_state=(
                loaded["constraint_state"] if loaded is not None else None
            ),
            checkpoint_callback=callback_for(
                case_id, probe_true_by_case[case_id]
            ),
        )
        elapsed = time.perf_counter() - case_started
        resumed_from = (
            int(loaded["optimizer_state"]["completed_iterations"])
            if loaded is not None
            else 0
        )
        timing_cases[case_id] = {
            "elapsed_s": elapsed,
            "target_num_iters": int(num_iters),
            "resumed_from_iteration": resumed_from,
            "iterations_executed_this_run": int(num_iters - resumed_from),
            "seconds_per_executed_iteration": (
                elapsed / max(num_iters - resumed_from, 1)
            ),
        }
        return result

    baseline_loaded: dict[str, Any] | None = None
    if source_checkpoint is not None:
        if source_checkpoint["case_id"] == "baseline":
            baseline_loaded = source_checkpoint
        else:
            baseline_loaded = prior_checkpoint("baseline", max_num_iters)
    baseline_result = run_case(
        "baseline",
        num_iters=max_num_iters,
        checkpoint_iters=checkpoint_iterations,
        loaded=baseline_loaded,
    )
    baseline_checkpoints = dict(baseline_result.get("checkpoints", {}))
    if baseline_loaded is not None:
        resumed_from = int(
            baseline_loaded["optimizer_state"]["completed_iterations"]
        )
        for iteration in checkpoint_iterations:
            if iteration < resumed_from:
                baseline_checkpoints[str(iteration)] = prior_checkpoint(
                    "baseline", iteration
                )
    baseline_result["checkpoints"] = baseline_checkpoints

    checkpoint_metrics: dict[str, dict[str, float]] = {}
    public_checkpoints: dict[str, dict[str, Any]] = {}
    earliest_passing_iteration: int | None = None
    for iteration in checkpoint_iterations:
        key = str(iteration)
        if key not in baseline_checkpoints:
            msg = f"Baseline trajectory did not produce checkpoint {iteration}."
            raise RuntimeError(msg)
        checkpoint = baseline_checkpoints[key]
        evaluation = _blind_checkpoint_evaluation(
            checkpoint,
            probe_true=probe_true_by_case["baseline"],
            object_true=sample_b,
            true_probe_case_separation=true_probe_case_separation,
            true_detector_amplitude_separation=true_detector_amplitude_separation,
        )
        checkpoint_metrics[key] = evaluation
        public_checkpoints[key] = {
            "P_B_rec": np.asarray(checkpoint["P_B_rec"], dtype=np.complex128),
            "B_rec": np.asarray(checkpoint["B_rec"], dtype=np.complex128),
            "data_fidelity_loss": float(checkpoint["data_fidelity_loss"]),
            "simulation_evaluation_only": evaluation,
        }
        if (
            earliest_passing_iteration is None
            and evaluation[
                "aligned_probe_error_to_true_case_separation_ratio"
            ]
            < gate_ratio
        ):
            earliest_passing_iteration = iteration

    baseline_evaluation, baseline_truth_metrics = _evaluate_reconstruction_case(
        baseline_result,
        probe_true_by_case["baseline"],
        sample_b,
    )
    baseline_output = {
        **baseline_result,
        "checkpoints": public_checkpoints,
        "simulation_evaluation_only": baseline_evaluation,
    }
    baseline_metrics = {
        "initial_data_fidelity_loss": float(
            baseline_result["initial_data_fidelity_loss"]
        ),
        "final_data_fidelity_loss": float(
            baseline_result["final_data_fidelity_loss"]
        ),
        "loss_tail_slope_per_iteration": _loss_tail_slope(
            np.asarray(baseline_result["loss_curve"])
        ),
        "simulation_evaluation_only": baseline_truth_metrics,
        "checkpoints": checkpoint_metrics,
    }

    plot_fields = [_reconstruction_probe(baseline_result)]
    plot_losses = [
        np.append(
            np.asarray(baseline_result["loss_curve"], dtype=np.float64),
            float(baseline_result["final_data_fidelity_loss"]),
        )
    ]
    plot_labels = [f"blind radial baseline, {max_num_iters} iters"]
    cases_output: dict[str, Any] = {}
    cases_metrics: dict[str, Any] = {}
    selected_check: dict[str, Any] = {
        "primary_gate": "aligned_probe_error_to_true_case_separation_ratio",
        "required_maximum_ratio_strict": gate_ratio,
        "earliest_passing_checkpoint_iteration": earliest_passing_iteration,
        "detector_loss_ratio_role": "secondary_diagnostic_not_primary_gate",
    }

    if earliest_passing_iteration is None:
        selected_check.update(
            {
                "status": "not_run_no_checkpoint_passed_signal_gate",
                "reason": (
                    "No 200/500/1000 baseline checkpoint reduced the aligned "
                    "probe floor below the true waist case separation."
                ),
            }
        )
    else:
        selected_iteration = earliest_passing_iteration
        baseline_checkpoint = baseline_checkpoints[str(selected_iteration)]
        baseline_case = _checkpoint_reconstruction_payload(
            baseline_checkpoint, case_id="baseline"
        )
        baseline_case_evaluation, baseline_case_metrics = (
            _evaluate_reconstruction_case(
                baseline_case,
                probe_true_by_case["baseline"],
                sample_b,
            )
        )
        cases_output["baseline"] = {
            **baseline_case,
            "simulation_evaluation_only": baseline_case_evaluation,
        }
        cases_metrics["baseline"] = baseline_case_metrics

        for case_id in ("waist_minus", "waist_plus"):
            case_loaded: dict[str, Any] | None = None
            if source_checkpoint is not None:
                if source_checkpoint["case_id"] == case_id:
                    case_loaded = source_checkpoint
                elif (
                    source_checkpoint["case_id"] == "waist_plus"
                    and case_id == "waist_minus"
                ):
                    case_loaded = prior_checkpoint(case_id, selected_iteration)
            case_result = run_case(
                case_id,
                num_iters=selected_iteration,
                checkpoint_iters=tuple(
                    iteration
                    for iteration in checkpoint_iterations
                    if iteration <= selected_iteration
                ),
                loaded=case_loaded,
            )
            case_evaluation, case_truth_metrics = _evaluate_reconstruction_case(
                case_result,
                probe_true_by_case[case_id],
                sample_b,
            )
            case_output = {
                **case_result,
                "simulation_evaluation_only": case_evaluation,
            }
            case_output.pop("checkpoints", None)
            cases_output[case_id] = case_output
            cases_metrics[case_id] = case_truth_metrics
            plot_fields.append(_reconstruction_probe(case_result))
            plot_losses.append(
                np.append(
                    np.asarray(case_result["loss_curve"], dtype=np.float64),
                    float(case_result["final_data_fidelity_loss"]),
                )
            )
            plot_labels.append(f"blind radial {case_id}, {selected_iteration} iters")

        baseline_raw = _reconstruction_probe(cases_output["baseline"])
        minus_raw = _reconstruction_probe(cases_output["waist_minus"])
        plus_raw = _reconstruction_probe(cases_output["waist_plus"])
        minus_to_baseline, minus_gain, minus_ramp = (
            align_affine_phase_and_complex_gain(minus_raw, baseline_raw)
        )
        plus_to_baseline, plus_gain, plus_ramp = (
            align_affine_phase_and_complex_gain(plus_raw, baseline_raw)
        )
        baseline_alignment = cases_output["baseline"][
            "simulation_evaluation_only"
        ]["alignment_parameters"]
        anchor_gain = complex(baseline_alignment["P_B_complex_gain"])
        anchor_ramp = tuple(
            float(value)
            for value in baseline_alignment["P_B_phase_ramp_yx_rad_per_px"]
        )
        baseline_common = np.asarray(
            cases_output["baseline"]["simulation_evaluation_only"][
                "P_B_rec_aligned_to_truth"
            ]
        )
        minus_common = _apply_affine_alignment(
            minus_to_baseline, anchor_gain, anchor_ramp
        )
        plus_common = _apply_affine_alignment(
            plus_to_baseline, anchor_gain, anchor_ramp
        )
        cases_output["baseline"]["simulation_evaluation_only"][
            "P_B_rec_common_gauge"
        ] = baseline_common
        for case_id, common, gain, ramp in (
            ("waist_minus", minus_common, minus_gain, minus_ramp),
            ("waist_plus", plus_common, plus_gain, plus_ramp),
        ):
            cases_output[case_id]["simulation_evaluation_only"].update(
                {
                    "P_B_rec_common_gauge": common,
                    "relative_alignment_to_recovered_baseline": {
                        "complex_gain": gain,
                        "phase_ramp_yx_rad_per_px": ramp,
                    },
                }
            )
        _, recovered_sensitivity = normalized_complex_sensitivity(
            minus_common,
            plus_common,
            baseline_common,
            delta_d_waist_m,
            nominal_d_waist_m,
        )
        true_minus_difference = compare_probe_sensitivity(
            probe_true_by_case["baseline"], probe_true_by_case["waist_minus"]
        )["gauge_aligned_complex_relative_l2"]
        true_plus_difference = compare_probe_sensitivity(
            probe_true_by_case["baseline"], probe_true_by_case["waist_plus"]
        )["gauge_aligned_complex_relative_l2"]
        recovered_minus_difference = compare_probe_sensitivity(
            baseline_common, minus_common
        )["gauge_aligned_complex_relative_l2"]
        recovered_plus_difference = compare_probe_sensitivity(
            baseline_common, plus_common
        )["gauge_aligned_complex_relative_l2"]
        selected_check.update(
            {
                "status": "executed_floor_gate_passed",
                "selected_iteration_count": selected_iteration,
                "selected_floor_to_signal_ratio": checkpoint_metrics[
                    str(selected_iteration)
                ]["aligned_probe_error_to_true_case_separation_ratio"],
                "normalized_recovered_probe_sensitivity": recovered_sensitivity,
                "true_probe_sensitivity": true_normalized_probe_sensitivity,
                "recovered_to_true_sensitivity_relative_deviation": (
                    _relative_change(
                        recovered_sensitivity,
                        true_normalized_probe_sensitivity,
                    )
                ),
                "true_minus_difference": true_minus_difference,
                "true_plus_difference": true_plus_difference,
                "recovered_minus_difference": recovered_minus_difference,
                "recovered_plus_difference": recovered_plus_difference,
                "sensitivity_ordering_matches_true": bool(
                    (true_minus_difference <= true_plus_difference)
                    == (recovered_minus_difference <= recovered_plus_difference)
                ),
                "gauge_evaluation_method": (
                    "case-to-recovered-baseline alignment followed by one "
                    "baseline-to-truth simulation-evaluation-only anchor"
                ),
            }
        )

    elapsed = time.perf_counter() - started
    output = {
        "status": (
            "executed_with_matched_sensitivity_cases"
            if earliest_passing_iteration is not None
            else "executed_baseline_gate_not_passed"
        ),
        "settings": {
            "variant_id": variant_id,
            "selected_num_iters": max_num_iters,
            "checkpoint_iterations": checkpoint_iterations,
            "beta_probe": float(study.get("beta_probe", recon["beta_probe"])),
            "beta_object": float(study.get("beta_object", recon["beta_object"])),
            "normalization_mode": str(study.get("normalization_mode", "none")),
            "selection_uses_truth_only_for_simulation_evaluation": True,
        },
        "baseline": baseline_output,
        "cases": cases_output,
        "selected_sensitivity_check": selected_check,
        "durable_checkpoint_files": checkpoint_files,
        "resume_provenance": resume_provenance,
    }
    metrics = {
        "status": output["status"],
        "baseline": baseline_metrics,
        "cases": cases_metrics,
        "selected_sensitivity_check": selected_check,
        "timing": {"total_elapsed_s": elapsed, "cases": timing_cases},
        "durable_checkpoint_files": checkpoint_files,
        "resume_provenance": resume_provenance,
    }
    save_json(
        progress_path,
        {
            "status": "complete",
            "selected_sensitivity_check": selected_check,
            "checkpoint_files": checkpoint_files,
            "resume_provenance": resume_provenance,
            "timing": metrics["timing"],
        },
    )
    return output, metrics, plot_fields, plot_losses, plot_labels


def run(
    config_path: Path, resume_blind_checkpoint: Path | None = None
) -> Path:
    """Execute exp030 Stages A-C and conditionally gate Stage D."""

    config = load_config(config_path)
    resolved_resume_checkpoint = (
        resume_blind_checkpoint.resolve()
        if resume_blind_checkpoint is not None
        else None
    )
    if (
        resolved_resume_checkpoint is not None
        and not resolved_resume_checkpoint.is_file()
    ):
        msg = f"Resume checkpoint does not exist: {resolved_resume_checkpoint}."
        raise FileNotFoundError(msg)
    run_cfg = config["run"]
    optics = config["optics"]
    projected_cfg = config["projected_phase"]
    sensitivity_cfg = config["sensitivity"]
    convergence_cfg = config["sampling_convergence"]
    tgv = config["tgv"]
    output_cfg = config["output"]

    run_dir = make_run_dir(
        PROJECT_ROOT / str(run_cfg.get("output_root", "runs")),
        str(run_cfg["name"]),
    )
    figures_dir = run_dir / "figures"
    outputs_dir = run_dir / "outputs"
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "config_path": str(config_path),
            "resume_blind_checkpoint": (
                str(resolved_resume_checkpoint)
                if resolved_resume_checkpoint is not None
                else None
            ),
        },
    )

    shape = _shape(optics["shape"], "optics.shape")
    dx_m = float(optics["dx_m"])
    wavelength_m = float(optics["wavelength_m"])
    z_ab_m = float(optics["z_AB_m"])
    z_bc_m = float(optics["z_BC_m"])
    medium_index = float(optics["medium_index"])
    dz_m = float(projected_cfg["dz_m"])
    supersampling = int(projected_cfg["lateral_supersampling"])
    d0 = float(tgv["d_waist_m"])
    delta_d = float(sensitivity_cfg["delta_d_waist_m"])
    delta_steps = np.asarray(
        sensitivity_cfg["delta_d_waist_steps_m"], dtype=np.float64
    )
    if delta_steps.ndim != 1 or delta_steps.size < 2:
        msg = "delta_d_waist_steps_m must contain at least two steps."
        raise ValueError(msg)
    if delta_d != float(delta_steps[-1]):
        msg = "delta_d_waist_m must equal the finest configured step."
        raise ValueError(msg)
    waist_values = np.asarray([d0 - delta_d, d0, d0 + delta_d])
    if waist_values[0] <= 0.0 or waist_values[-1] > min(
        float(tgv["d_top_m"]), float(tgv["d_bottom_m"])
    ):
        msg = "Waist finite-difference cases violate the TGV geometry."
        raise ValueError(msg)

    fine_shape = _shape(convergence_cfg["fine_shape"], "fine_shape")
    fine_dx = float(convergence_cfg["fine_dx_m"])
    if not np.isclose(fine_shape[0] * fine_dx, shape[0] * dx_m) or not np.isclose(
        fine_shape[1] * fine_dx, shape[1] * dx_m
    ):
        msg = "Fine and baseline lateral grids must cover the same FOV."
        raise ValueError(msg)
    scale_y = fine_shape[0] // shape[0]
    scale_x = fine_shape[1] // shape[1]
    if fine_shape != (shape[0] * scale_y, shape[1] * scale_x):
        msg = "fine_shape must be an integer refinement of optics.shape."
        raise ValueError(msg)

    noise_cfg = config.get("noise") or {}
    if noise_cfg.get("photon_scale") is not None or float(
        noise_cfg.get("gaussian_sigma", 0.0)
    ) > 0.0:
        msg = "exp030 convergence requires the configured noise-free forward model."
        raise ValueError(msg)

    incident = make_plane_wave(
        shape,
        dx_m,
        wavelength_m,
        amplitude=float(config["illumination"]["amplitude"]),
    )
    sample_b_cfg = config["sample_b"]
    sample_b = make_random_phase_object(
        shape,
        phase_range=float(sample_b_cfg["phase_range_rad"]),
        feature_size_px=int(sample_b_cfg["feature_size_px"]),
        seed=int(sample_b_cfg["seed"]),
    )
    fine_b = np.repeat(np.repeat(sample_b, scale_y, axis=0), scale_x, axis=1)
    scan_positions = _make_scan(config, dx_m)
    fine_pixel_coordinates = scan_positions / fine_dx
    if not np.allclose(fine_pixel_coordinates, np.round(fine_pixel_coordinates)):
        msg = "Physical scan positions must be integer shifts on the fine grid."
        raise ValueError(msg)

    models = [
        _projected_model(
            config,
            shape=shape,
            dx_m=dx_m,
            dz_m=dz_m,
            supersampling=supersampling,
            d_waist_m=float(waist),
        )
        for waist in waist_values
    ]
    _minus_model, baseline, _plus_model = models
    model_metrics, controls = _model_validation(
        config, baseline, shape, dx_m, dz_m, supersampling
    )

    fringe_metrics = _projected_phase_fringe_metrics(config)
    radial_source_step = float(
        convergence_cfg.get(
            "radial_source_transition_step_m",
            min(1.0e-9, fringe_metrics["transition_phase_fringe_period_m"] / 12.0),
        )
    )
    fine_radial_source_step = float(
        convergence_cfg.get(
            "fine_radial_source_transition_step_m", 0.5 * radial_source_step
        )
    )
    medium_wavelength = wavelength_m / medium_index
    radial_output_step = float(
        convergence_cfg.get(
            "radial_output_step_m", min(0.5 * fine_dx, medium_wavelength / 8.0)
        )
    )
    convergence_cfg["resolved_radial_forward"] = {
        "source_transition_step_m": radial_source_step,
        "fine_source_transition_step_m": fine_radial_source_step,
        "output_step_m": radial_output_step,
        "method": "continuous_axisymmetric_fresnel_hankel",
    }

    surface_step = float(sensitivity_cfg["delta_surface_diameter_m"])
    phase_step = float(sensitivity_cfg["delta_phase_scale_fraction"])
    phase_scale_nominal = float(projected_cfg["phase_scale"])
    cases: dict[str, dict[str, float]] = {
        "waist_minus": {"d_waist_m": float(waist_values[0])},
        "baseline": {},
        "waist_plus": {"d_waist_m": float(waist_values[2])},
        "surface_minus": {
            "d_top_m": float(tgv["d_top_m"]) - surface_step,
            "d_bottom_m": float(tgv["d_bottom_m"]) - surface_step,
        },
        "surface_plus": {
            "d_top_m": float(tgv["d_top_m"]) + surface_step,
            "d_bottom_m": float(tgv["d_bottom_m"]) + surface_step,
        },
        "phase_minus": {
            "phase_scale": phase_scale_nominal * (1.0 - phase_step)
        },
        "phase_plus": {
            "phase_scale": phase_scale_nominal * (1.0 + phase_step)
        },
        "zero_contrast": {"n_air": float(tgv["n_glass"])},
    }
    for index, step in enumerate(delta_steps):
        cases[_waist_step_label(index, "minus")] = {
            "d_waist_m": d0 - float(step)
        }
        cases[_waist_step_label(index, "plus")] = {
            "d_waist_m": d0 + float(step)
        }

    fine_x, fine_y = coordinate_grid(fine_shape, fine_dx)
    center_x, center_y = (float(value) for value in tgv["center_xy_m"])
    output_radius_max = float(
        np.max(np.sqrt((fine_x - center_x) ** 2 + (fine_y - center_y) ** 2))
    )
    del fine_x, fine_y
    radial_result = _fresnel_tgv_probe_batch(
        config,
        cases,
        transition_step_m=radial_source_step,
        output_radius_max_m=output_radius_max,
        output_step_m=radial_output_step,
    )
    coarse_probe_cases = _sample_radial_probe_batch(
        radial_result, shape, dx_m, (center_x, center_y)
    )
    fine_probe_cases = _sample_radial_probe_batch(
        radial_result, fine_shape, fine_dx, (center_x, center_y)
    )

    p_minus = coarse_probe_cases["waist_minus"]
    p_base = coarse_probe_cases["baseline"]
    p_plus = coarse_probe_cases["waist_plus"]
    probes = [p_minus, p_base, p_plus]
    fine_p_minus = fine_probe_cases["waist_minus"]
    fine_p_base = fine_probe_cases["baseline"]
    fine_p_plus = fine_probe_cases["waist_plus"]

    transfer_bc = _angular_spectrum_transfer(
        shape, dx_m, wavelength_m, z_bc_m, medium_index
    )
    fine_transfer_bc = _angular_spectrum_transfer(
        fine_shape, fine_dx, wavelength_m, z_bc_m, medium_index
    )
    intensity_stacks = [
        _simulate_probe_detector(
            config, probe, sample_b, scan_positions, dx_m, transfer_bc
        )
        for probe in probes
    ]
    fine_intensity_stacks = [
        _simulate_probe_detector(
            config, probe, fine_b, scan_positions, fine_dx, fine_transfer_bc
        )
        for probe in (fine_p_minus, fine_p_base, fine_p_plus)
    ]
    i_minus, i_base, i_plus = intensity_stacks
    fine_i_minus, fine_i_base, fine_i_plus = fine_intensity_stacks

    probe_derivative, normalized_probe = normalized_complex_sensitivity(
        p_minus, p_plus, p_base, delta_d, d0
    )
    intensity_derivative, normalized_intensity = normalized_real_sensitivity(
        i_minus, i_plus, i_base, delta_d, d0
    )
    _, fine_probe_norm = normalized_complex_sensitivity(
        fine_p_minus, fine_p_plus, fine_p_base, delta_d, d0
    )
    _, fine_intensity_norm = normalized_real_sensitivity(
        fine_i_minus, fine_i_plus, fine_i_base, delta_d, d0
    )
    per_frame = _frame_sensitivity(intensity_derivative, i_base, d0)
    probe_metrics: dict[str, Any] = {
        "minus_vs_baseline": compare_probe_sensitivity(p_base, p_minus),
        "plus_vs_baseline": compare_probe_sensitivity(p_base, p_plus),
        "normalized_d_waist_sensitivity": normalized_probe,
        "scaled_difference_map_max": float(
            np.max(np.abs(probe_derivative * d0))
        ),
        "scaled_difference_map_rms": float(
            np.sqrt(np.mean(np.abs(probe_derivative * d0) ** 2))
        ),
    }
    frame_energies = np.sum(i_base, axis=(1, 2), dtype=np.float64)
    intensity_metrics: dict[str, Any] = {
        "minus_vs_baseline_relative_l2": relative_l2(i_minus, i_base),
        "plus_vs_baseline_relative_l2": relative_l2(i_plus, i_base),
        "plus_minus_relative_l2_to_baseline": float(
            _norm(i_plus - i_minus)
            / max(_norm(i_base), np.finfo(float).eps)
        ),
        "normalized_intensity_finite_difference_norm": normalized_intensity,
        "per_frame_normalized_sensitivity": per_frame,
        "maximum_frame_sensitivity": float(np.max(per_frame)),
        "median_frame_sensitivity": float(np.median(per_frame)),
        "baseline_frame_energy_max_relative_deviation": float(
            np.max(np.abs(frame_energies - np.mean(frame_energies)))
            / max(float(np.mean(frame_energies)), np.finfo(float).eps)
        ),
        "detector_pixel_value_semantics": "pixel_average_irradiance",
        "baseline_detector_bin_factor": int(
            round(float(optics["detector_pixel_size_m"]) / dx_m)
        ),
        "fine_detector_bin_factor": int(
            round(float(optics["detector_pixel_size_m"]) / fine_dx)
        ),
    }

    baseline_step_convergence = _step_convergence_from_cases(
        delta_steps,
        coarse_probe_cases,
        config,
        sample_b,
        scan_positions,
        dx_m,
        transfer_bc,
        p_base,
        i_base,
        d0,
    )
    fine_step_convergence = _step_convergence_from_cases(
        delta_steps,
        fine_probe_cases,
        config,
        fine_b,
        scan_positions,
        fine_dx,
        fine_transfer_bc,
        fine_p_base,
        fine_i_base,
        d0,
    )
    step_convergence_change = max(
        float(baseline_step_convergence["probe_successive_relative_change"][-1]),
        float(
            baseline_step_convergence["intensity_successive_relative_change"][-1]
        ),
        float(fine_step_convergence["probe_successive_relative_change"][-1]),
        float(fine_step_convergence["intensity_successive_relative_change"][-1]),
    )

    fine_dz = float(convergence_cfg["fine_dz_m"])
    fine_z_model = _projected_model(
        config,
        shape=shape,
        dx_m=dx_m,
        dz_m=fine_dz,
        supersampling=supersampling,
        d_waist_m=d0,
    )
    path_dz_change = relative_l2(
        baseline["fill_path_length_m"], fine_z_model["fill_path_length_m"]
    )
    transmission_dz_change = relative_l2(
        baseline["A_effective_true"], fine_z_model["A_effective_true"]
    )

    main_cases = {
        "waist_minus": cases["waist_minus"],
        "baseline": cases["baseline"],
        "waist_plus": cases["waist_plus"],
    }
    fine_radial_result = _fresnel_tgv_probe_batch(
        config,
        main_cases,
        transition_step_m=fine_radial_source_step,
        output_radius_max_m=output_radius_max,
        output_step_m=radial_output_step,
    )
    fine_source_probes = _sample_radial_probe_batch(
        fine_radial_result, shape, dx_m, (center_x, center_y)
    )
    fine_source_intensities = [
        _simulate_probe_detector(
            config,
            fine_source_probes[label],
            sample_b,
            scan_positions,
            dx_m,
            transfer_bc,
        )
        for label in ("waist_minus", "baseline", "waist_plus")
    ]
    _, fine_source_probe_norm = normalized_complex_sensitivity(
        fine_source_probes["waist_minus"],
        fine_source_probes["waist_plus"],
        fine_source_probes["baseline"],
        delta_d,
        d0,
    )
    _, fine_source_intensity_norm = normalized_real_sensitivity(
        fine_source_intensities[0],
        fine_source_intensities[2],
        fine_source_intensities[1],
        delta_d,
        d0,
    )
    radial_source_change = max(
        _relative_change(normalized_probe, fine_source_probe_norm),
        _relative_change(normalized_intensity, fine_source_intensity_norm),
    )

    coarse_radial_result = dict(radial_result)
    coarse_indices = np.arange(
        0, len(np.asarray(radial_result["probe_radius_m"])), 2
    )
    if coarse_indices[-1] != len(np.asarray(radial_result["probe_radius_m"])) - 1:
        coarse_indices = np.append(
            coarse_indices, len(np.asarray(radial_result["probe_radius_m"])) - 1
        )
    coarse_radial_result["probe_radius_m"] = np.asarray(
        radial_result["probe_radius_m"]
    )[coarse_indices]
    coarse_radial_result["probe_radial"] = np.asarray(
        radial_result["probe_radial"]
    )[:, coarse_indices]
    coarse_output_probes = _sample_radial_probe_batch(
        coarse_radial_result, shape, dx_m, (center_x, center_y)
    )
    _, coarse_output_probe_norm = normalized_complex_sensitivity(
        coarse_output_probes["waist_minus"],
        coarse_output_probes["waist_plus"],
        coarse_output_probes["baseline"],
        delta_d,
        d0,
    )
    radial_output_change = _relative_change(
        coarse_output_probe_norm, normalized_probe
    )

    zero_index = list(radial_result["case_labels"]).index("zero_contrast")
    reference_probe = float(config["illumination"]["amplitude"]) * np.exp(
        1j * 2.0 * np.pi * medium_index / wavelength_m * z_ab_m
    )
    plane_wave_error = float(
        np.max(
            np.abs(
                np.asarray(radial_result["probe_radial"])[zero_index]
                - reference_probe
            )
        )
    )
    maximum_separation = output_radius_max + float(
        radial_result["quadrature"]["outer_support_radius_m"]
    )
    maximum_paraxial_ratio = maximum_separation / z_ab_m
    paraxial_fourth_order_phase_error = float(
        2.0
        * np.pi
        / medium_wavelength
        * maximum_separation**4
        / (8.0 * z_ab_m**3)
    )

    convergence_metrics = {
        "path_dz_relative_change": path_dz_change,
        "raster_transmission_dz_relative_change": transmission_dz_change,
        "probe_dx_relative_change": _relative_change(
            normalized_probe, fine_probe_norm
        ),
        "intensity_dx_relative_change": _relative_change(
            normalized_intensity, fine_intensity_norm
        ),
        "fine_dx_normalized_probe_sensitivity": fine_probe_norm,
        "fine_dx_normalized_intensity_sensitivity": fine_intensity_norm,
        "fine_source_normalized_probe_sensitivity": fine_source_probe_norm,
        "fine_source_normalized_intensity_sensitivity": (
            fine_source_intensity_norm
        ),
        "radial_source_convergence_relative_change": radial_source_change,
        "radial_output_interpolation_relative_change": radial_output_change,
        "delta_d_step_convergence_relative_change": step_convergence_change,
        "finite_difference_step": {
            "baseline_dx_m": dx_m,
            "fine_dx_m": fine_dx,
            "baseline_grid": baseline_step_convergence,
            "fine_grid": fine_step_convergence,
        },
    }
    model_metrics.update(fringe_metrics)
    model_metrics.update(
        {
            "a_to_b_forward_method": (
                "continuous_axisymmetric_fresnel_hankel_on_T_minus_1"
            ),
            "fresnel_plane_wave_max_abs_error": plane_wave_error,
            "radial_source_transition_step_m": radial_source_step,
            "fine_radial_source_transition_step_m": fine_radial_source_step,
            "radial_output_step_m": radial_output_step,
            "maximum_paraxial_transverse_to_z_ratio": maximum_paraxial_ratio,
            "estimated_max_fourth_order_path_phase_error_rad": (
                paraxial_fourth_order_phase_error
            ),
            "fine_dx_to_phase_nyquist_requirement_ratio": float(
                fine_dx
                / fringe_metrics["transition_phase_nyquist_dx_requirement_m"]
            ),
            "dz_convergence_relative_change": max(
                path_dz_change, transmission_dz_change
            ),
            "dx_convergence_relative_change": max(
                convergence_metrics["probe_dx_relative_change"],
                convergence_metrics["intensity_dx_relative_change"],
            ),
            "radial_source_convergence_relative_change": radial_source_change,
            "radial_output_interpolation_relative_change": radial_output_change,
            "delta_d_step_convergence_relative_change": step_convergence_change,
        }
    )
    intensity_metrics["sampling_discretization_floor_relative"] = max(
        model_metrics["dz_convergence_relative_change"],
        model_metrics["dx_convergence_relative_change"],
        model_metrics["radial_source_convergence_relative_change"],
        model_metrics["radial_output_interpolation_relative_change"],
        model_metrics["delta_d_step_convergence_relative_change"],
    )

    surface_derivative = central_finite_difference(
        coarse_probe_cases["surface_minus"],
        coarse_probe_cases["surface_plus"],
        surface_step,
    )
    phase_derivative = central_finite_difference(
        coarse_probe_cases["phase_minus"],
        coarse_probe_cases["phase_plus"],
        phase_scale_nominal * phase_step,
    )
    observability = analyze_local_observability(
        p_base,
        {
            "d_waist": central_finite_difference(p_minus, p_plus, delta_d),
            "common_surface_diameter": surface_derivative,
            "phase_scale": phase_derivative,
        },
        {
            "d_waist": d0,
            "common_surface_diameter": 0.5
            * (float(tgv["d_top_m"]) + float(tgv["d_bottom_m"])),
            "phase_scale": phase_scale_nominal,
        },
    )
    labels = list(observability["parameter_labels"])
    correlation = np.asarray(observability["normalized_column_correlation"])
    waist_index = labels.index("d_waist")
    other_indices = [index for index in range(len(labels)) if index != waist_index]
    observability["d_waist_max_abs_correlation_with_other_columns"] = float(
        np.max(np.abs(correlation[waist_index, other_indices]))
    )

    del (
        fine_probe_cases,
        fine_p_minus,
        fine_p_base,
        fine_p_plus,
        fine_intensity_stacks,
        fine_i_minus,
        fine_i_base,
        fine_i_plus,
        fine_source_probes,
        fine_source_intensities,
        coarse_output_probes,
        fine_b,
        fine_transfer_bc,
        fine_z_model,
    )
    gc.collect()

    threshold = float(convergence_cfg["relative_change_threshold"])
    controls_pass = (
        model_metrics["zero_contrast_max_abs_error"] <= 1e-12
        and model_metrics["reference_region_max_abs_T_minus_1"] <= 1e-12
        and model_metrics["pure_phase_amplitude_max_abs_error"] <= 1e-12
        and model_metrics["transmission_complex_relative_error"] <= 1e-10
        and model_metrics["fresnel_plane_wave_max_abs_error"] <= 1e-12
        and model_metrics["fill_path_analytic_max_abs_error_m"]
        <= dz_m + 32.0 * np.finfo(float).eps
    )
    convergence_pass = max(
        model_metrics["dz_convergence_relative_change"],
        model_metrics["dx_convergence_relative_change"],
        model_metrics["radial_source_convergence_relative_change"],
        model_metrics["radial_output_interpolation_relative_change"],
        model_metrics["delta_d_step_convergence_relative_change"],
    ) < threshold
    stage_a_c_status = (
        "Passed" if controls_pass and convergence_pass else "Inconclusive"
    )
    experiment_status = stage_a_c_status
    reconstruction_cfg = config["reconstruction"]
    reconstruction_enabled = bool(reconstruction_cfg.get("enabled", False))
    reconstruction_output: dict[str, Any] | None = None
    reconstruction_metrics: dict[str, Any]
    recovered_probe_fields: list[np.ndarray] = []
    recovered_loss_curves: list[np.ndarray] = []
    recovered_plot_labels: list[str] = []
    if not controls_pass:
        reconstruction_status = "not_run_model_validation_gate_failed"
        reconstruction_metrics = {
            "status": reconstruction_status,
            "executed": False,
            "reason": "Stage D requires validated Stage A analytic controls.",
        }
    elif not convergence_pass:
        reconstruction_status = "not_run_sampling_or_step_convergence_gate_failed"
        reconstruction_metrics = {
            "status": reconstruction_status,
            "executed": False,
            "reason": (
                "Stage D requires dz, radial-source, radial-output, dx, and "
                "finite-difference-step convergence below the threshold."
            ),
        }
    elif reconstruction_enabled:
        ablation_cfg = reconstruction_cfg["operator_consistency_ablation"]
        if not bool(ablation_cfg.get("enabled", True)):
            msg = "Stage D is enabled but operator_consistency_ablation is disabled."
            raise ValueError(msg)
        reconstruction_status = "executed_operator_consistency_ablation"
        variant_ids = [str(value) for value in ablation_cfg["variants"]]
        if len(variant_ids) != len(set(variant_ids)) or not variant_ids:
            msg = "Stage D ablation variants must be non-empty and unique."
            raise ValueError(msg)
        screening_num_iters = int(ablation_cfg["screening_num_iters"])
        if screening_num_iters <= 0:
            msg = "screening_num_iters must be positive."
            raise ValueError(msg)

        radial_operator = _build_radial_fresnel_operator(
            np.asarray(radial_result["source_radius_m"]),
            np.asarray(radial_result["source_weights_m"]),
            np.asarray(radial_result["probe_radius_m"]),
            shape=shape,
            dx_m=dx_m,
            center_xy_m=(center_x, center_y),
            wavelength_m=wavelength_m,
            propagation_distance_m=z_ab_m,
            medium_index=medium_index,
            incident_amplitude=float(config["illumination"]["amplitude"]),
        )
        radial_cfg = ablation_cfg["radial_constraint"]
        radial_operator_norm_squared = _estimate_radial_operator_norm_squared(
            radial_operator,
            int(radial_cfg["power_iterations"]),
            int(radial_cfg["power_iteration_seed"]),
        )
        baseline_radial_index = list(radial_result["case_labels"]).index(
            "baseline"
        )
        baseline_source_truth = np.asarray(
            radial_result["source_transmission"]
        )[baseline_radial_index]
        radial_forward_truth = _radial_fresnel_full_field(
            radial_operator, baseline_source_truth
        )

        diagnostic_rng = np.random.default_rng(
            int(ablation_cfg["adjoint_test_seed"])
        )
        source_trial = diagnostic_rng.normal(
            size=baseline_source_truth.shape
        ) + 1j * diagnostic_rng.normal(size=baseline_source_truth.shape)
        cartesian_trial = diagnostic_rng.normal(size=shape) + 1j * (
            diagnostic_rng.normal(size=shape)
        )
        trial_forward = _radial_fresnel_linear_forward(
            radial_operator, source_trial
        )
        trial_adjoint = _radial_fresnel_weighted_adjoint(
            radial_operator, cartesian_trial
        )
        inner_forward = dx_m**2 * np.sum(
            np.conj(trial_forward) * cartesian_trial
        )
        inner_adjoint = np.sum(
            np.asarray(radial_operator["source_measure_m2"])
            * np.conj(source_trial)
            * trial_adjoint
        )
        adjoint_identity_relative_error = float(
            abs(inner_forward - inner_adjoint)
            / max(
                abs(inner_forward),
                abs(inner_adjoint),
                np.finfo(float).eps,
            )
        )

        legacy_truth_probe = _legacy_coarse_a_plane_constraint(
            p_base,
            incident,
            np.asarray(controls["reference_region_mask"], dtype=bool),
            dx_m,
            wavelength_m,
            z_ab_m,
        )
        legacy_truth_intensity = _simulate_probe_detector(
            config,
            legacy_truth_probe,
            sample_b,
            scan_positions,
            dx_m,
            transfer_bc,
        )
        fixed_constraint, fixed_state = _make_radial_adjoint_constraint(
            radial_operator,
            operator_norm_squared=radial_operator_norm_squared,
            application_interval=1,
            internal_steps=int(radial_cfg["internal_steps"]),
            step_scale=float(radial_cfg["step_scale"]),
            max_backtracking_steps=int(radial_cfg["max_backtracking_steps"]),
            initial_transmission=baseline_source_truth,
        )
        radial_truth_fixed_once = fixed_constraint(p_base)
        radial_truth_fixed_twice = fixed_constraint(radial_truth_fixed_once)
        radial_truth_fixed_intensity = _simulate_probe_detector(
            config,
            radial_truth_fixed_once,
            sample_b,
            scan_positions,
            dx_m,
            transfer_bc,
        )
        range_constraint, range_state = _make_radial_output_range_constraint(
            radial_operator["interpolation"],
            float(
                ablation_cfg["radial_output_range_constraint"][
                    "ridge_fraction"
                ]
            ),
        )
        radial_range_truth_once = range_constraint(p_base)
        radial_range_truth_twice = range_constraint(radial_range_truth_once)
        radial_range_truth_intensity = _simulate_probe_detector(
            config,
            radial_range_truth_once,
            sample_b,
            scan_positions,
            dx_m,
            transfer_bc,
        )
        oracle_epie = epie_reconstruct(
            i_base,
            scan_positions,
            dx=dx_m,
            wavelength=wavelength_m,
            z_BC=z_bc_m,
            num_iters=0,
            beta_probe=0.0,
            beta_object=0.0,
            init_probe=p_base,
            init_object=sample_b,
            update_probe=False,
            shuffle_positions=False,
            seed=None,
            object_amplitude_bounds=None,
            probe_l2_norm_target=None,
            probe_constraint=None,
            show_progress=False,
        )
        energy_norm_target = float(
            np.sqrt(
                np.mean(np.sum(i_base, axis=(1, 2), dtype=np.float64))
            )
        )
        true_minus_difference = compare_probe_sensitivity(p_base, p_minus)[
            "gauge_aligned_complex_relative_l2"
        ]
        true_plus_difference = compare_probe_sensitivity(p_base, p_plus)[
            "gauge_aligned_complex_relative_l2"
        ]
        true_probe_case_separation = 0.5 * (
            true_minus_difference + true_plus_difference
        )
        true_detector_amplitude_separation = 0.5 * (
            relative_l2(np.sqrt(i_minus), np.sqrt(i_base))
            + relative_l2(np.sqrt(i_plus), np.sqrt(i_base))
        )
        operator_consistency_metrics: dict[str, Any] = {
            "oracle_truth_pair": {
                "epie_frozen_amplitude_loss": float(
                    oracle_epie["final_data_fidelity_loss"]
                ),
                "direct_intensity_relative_l2": relative_l2(
                    _simulate_probe_detector(
                        config,
                        p_base,
                        sample_b,
                        scan_positions,
                        dx_m,
                        transfer_bc,
                    ),
                    i_base,
                ),
            },
            "legacy_coarse_A_constraint_on_truth": {
                "raw_probe_relative_l2": relative_l2(
                    legacy_truth_probe, p_base
                ),
                "gauge_aligned_probe_relative_l2": compare_probe_sensitivity(
                    p_base, legacy_truth_probe
                )["gauge_aligned_complex_relative_l2"],
                "detector_amplitude_relative_l2_with_true_B": relative_l2(
                    np.sqrt(legacy_truth_intensity), np.sqrt(i_base)
                ),
            },
            "continuous_radial_operator": {
                "forward_reproduction_relative_l2": relative_l2(
                    radial_forward_truth, p_base
                ),
                "weighted_adjoint_identity_relative_error": (
                    adjoint_identity_relative_error
                ),
                "estimated_operator_norm_squared": (
                    radial_operator_norm_squared
                ),
                "truth_fixed_point_raw_relative_l2": relative_l2(
                    radial_truth_fixed_once, p_base
                ),
                "truth_fixed_point_gauge_aligned_relative_l2": (
                    compare_probe_sensitivity(
                        p_base, radial_truth_fixed_once
                    )["gauge_aligned_complex_relative_l2"]
                ),
                "truth_fixed_point_detector_amplitude_relative_l2": (
                    relative_l2(
                        np.sqrt(radial_truth_fixed_intensity), np.sqrt(i_base)
                    )
                ),
                "truth_fixed_point_idempotence_relative_l2": relative_l2(
                    radial_truth_fixed_twice, radial_truth_fixed_once
                ),
                "truth_source_amplitude_max_abs_error": float(
                    np.max(np.abs(np.abs(baseline_source_truth) - 1.0))
                ),
                "fixed_point_objective_after": float(
                    np.asarray(fixed_state["objective_after"])[-1]
                ),
                "adjoint_convention": str(
                    radial_operator["adjoint_convention"]
                ),
            },
            "continuous_radial_output_range_constraint": {
                "scientific_role": (
                    "B_plane_axisymmetric_range_constraint_not_an_A_plane_inverse"
                ),
                "truth_fixed_point_raw_relative_l2": relative_l2(
                    radial_range_truth_once, p_base
                ),
                "truth_fixed_point_gauge_aligned_relative_l2": (
                    compare_probe_sensitivity(
                        p_base, radial_range_truth_once
                    )["gauge_aligned_complex_relative_l2"]
                ),
                "truth_fixed_point_detector_amplitude_relative_l2": (
                    relative_l2(
                        np.sqrt(radial_range_truth_intensity), np.sqrt(i_base)
                    )
                ),
                "truth_fixed_point_idempotence_relative_l2": relative_l2(
                    radial_range_truth_twice, radial_range_truth_once
                ),
                "ridge_fraction": float(range_state["ridge_fraction"]),
                "ridge_absolute": float(range_state["ridge_absolute"]),
                "active_radial_node_count": int(
                    range_state["active_radial_node_count"]
                ),
                "radial_node_count": int(range_state["radial_node_count"]),
            },
            "energy_norm_control": {
                "true_probe_l2_norm": _norm(p_base),
                "detector_energy_probe_norm_target": energy_norm_target,
                "relative_bias": _relative_change(
                    energy_norm_target, _norm(p_base)
                ),
                "interpretation": (
                    "diagnostic_only; propagating-wave cutoff is not exactly unitary"
                ),
            },
            "signal_scales": {
                "true_probe_case_separation": true_probe_case_separation,
                "true_detector_amplitude_case_separation": (
                    true_detector_amplitude_separation
                ),
            },
        }

        optimizer_study_output: dict[str, Any] = {
            "status": "not_run_disabled"
        }
        optimizer_study_metrics: dict[str, Any] = {
            "status": "not_run_disabled"
        }
        optimizer_study_cfg = ablation_cfg.get("optimizer_study", {})
        if bool(optimizer_study_cfg.get("enabled", False)):
            (
                optimizer_study_output,
                optimizer_study_metrics,
                optimizer_plot_fields,
                optimizer_plot_losses,
                optimizer_plot_labels,
            ) = _run_optimizer_study(
                config,
                i_base,
                scan_positions,
                incident,
                np.asarray(controls["reference_region_mask"], dtype=bool),
                dx_m,
                p_base,
                sample_b,
                transfer_bc,
            )
            recovered_probe_fields.extend(optimizer_plot_fields)
            recovered_loss_curves.extend(optimizer_plot_losses)
            recovered_plot_labels.extend(optimizer_plot_labels)

        variant_outputs: dict[str, Any] = {}
        variant_metrics: dict[str, Any] = {}
        plot_label_by_variant = {
            "known_b_probe_only": "known B / probe only",
            "known_probe_object_only": "known probe / object only",
            "blind_unconstrained_with_energy_norm": "blind + energy norm",
            "blind_unconstrained": "blind unconstrained",
            "blind_legacy_coarse_a_constraint": "blind legacy coarse A",
            "blind_radial_output_range_constraint": "blind radial B-range",
            "blind_radial_adjoint_constraint": "blind radial A-adjoint",
        }
        for variant_id in variant_ids:
            raw_result = _reconstruct_case(
                config,
                i_base,
                scan_positions,
                incident,
                np.asarray(controls["reference_region_mask"], dtype=bool),
                dx_m,
                variant_id=variant_id,
                num_iters=screening_num_iters,
                sample_b_true_diagnostic=sample_b,
                probe_true_diagnostic=p_base,
                radial_operator=radial_operator,
                radial_operator_norm_squared=radial_operator_norm_squared,
            )
            evaluation, evaluation_metrics = _evaluate_reconstruction_case(
                raw_result, p_base, sample_b
            )
            case_output = {
                **raw_result,
                "simulation_evaluation_only": evaluation,
            }
            variant_outputs[variant_id] = {
                "scientific_role": str(raw_result["scientific_role"]),
                "uses_simulation_truth_B_as_input": bool(
                    raw_result["uses_simulation_truth_B_as_input"]
                ),
                "uses_simulation_truth_probe_as_input": bool(
                    raw_result["uses_simulation_truth_probe_as_input"]
                ),
                "cases": {"baseline": case_output},
            }
            raw_probe = _reconstruction_probe(raw_result)
            raw_object = _reconstruction_object(raw_result)
            aligned_probe_error = float(
                evaluation_metrics["P_B_rec_aligned_complex_relative_error"]
            )
            final_loss = float(raw_result["final_data_fidelity_loss"])
            metric_payload: dict[str, Any] = {
                "raw_P_B_l2_norm": _norm(raw_probe),
                "raw_B_amplitude_min": float(
                    np.min(np.abs(raw_object))
                ),
                "raw_B_amplitude_max": float(
                    np.max(np.abs(raw_object))
                ),
                "initial_data_fidelity_loss": float(
                    raw_result["initial_data_fidelity_loss"]
                ),
                "final_data_fidelity_loss": final_loss,
                "loss_reduction": float(
                    raw_result["initial_data_fidelity_loss"] - final_loss
                ),
                "loss_tail_slope_per_iteration": _loss_tail_slope(
                    np.asarray(raw_result["loss_curve"])
                ),
                "uses_simulation_truth_B_as_input": bool(
                    raw_result["uses_simulation_truth_B_as_input"]
                ),
                "uses_simulation_truth_probe_as_input": bool(
                    raw_result["uses_simulation_truth_probe_as_input"]
                ),
                "fixed_object_max_abs_change": float(
                    raw_result["fixed_object_max_abs_change"]
                ),
                "fixed_probe_max_abs_change": float(
                    raw_result["fixed_probe_max_abs_change"]
                ),
                "aligned_probe_error_to_true_case_separation_ratio": float(
                    aligned_probe_error
                    / max(true_probe_case_separation, np.finfo(float).eps)
                ),
                "final_loss_to_true_detector_amplitude_separation_ratio": float(
                    final_loss
                    / max(
                        true_detector_amplitude_separation,
                        np.finfo(float).eps,
                    )
                ),
                "simulation_evaluation_only": evaluation_metrics,
            }
            if "radial_constraint" in raw_result:
                radial_case = raw_result["radial_constraint"]
                before = np.asarray(radial_case["objective_before"])
                after = np.asarray(radial_case["objective_after"])
                metric_payload["radial_constraint"] = {
                    "application_count": int(
                        len(radial_case["application_call_index"])
                    ),
                    "source_amplitude_max_abs_error": float(
                        radial_case["source_amplitude_max_abs_error"]
                    ),
                    "final_objective_before": float(before[-1]),
                    "final_objective_after": float(after[-1]),
                    "all_projected_steps_nonincreasing": bool(
                        np.all(after <= before + 1e-24)
                    ),
                }
            if "radial_output_range_constraint" in raw_result:
                range_case = raw_result["radial_output_range_constraint"]
                metric_payload["radial_output_range_constraint"] = {
                    "call_count": int(range_case["call_count"]),
                    "ridge_fraction": float(range_case["ridge_fraction"]),
                    "last_relative_change": float(
                        range_case["last_relative_change"]
                    ),
                    "scientific_role": str(range_case["scientific_role"]),
                }
            variant_metrics[variant_id] = {
                "scientific_role": str(raw_result["scientific_role"]),
                "cases": {"baseline": metric_payload},
            }
            recovered_probe_fields.append(raw_probe)
            recovered_loss_curves.append(
                np.append(
                    np.asarray(raw_result["loss_curve"], dtype=np.float64),
                    float(raw_result["final_data_fidelity_loss"]),
                )
            )
            recovered_plot_labels.append(plot_label_by_variant[variant_id])

        selected_variant = str(ablation_cfg["selected_sensitivity_variant"])
        if selected_variant not in variant_outputs:
            msg = "selected_sensitivity_variant must be one screened variant."
            raise ValueError(msg)
        selected_baseline_metrics = variant_metrics[selected_variant]["cases"][
            "baseline"
        ]
        floor_ratio = float(
            selected_baseline_metrics[
                "aligned_probe_error_to_true_case_separation_ratio"
            ]
        )
        sensitivity_gate_ratio = float(
            ablation_cfg["sensitivity_floor_to_signal_gate"]
        )
        sensitivity_check: dict[str, Any] = {
            "selected_variant": selected_variant,
            "floor_to_signal_ratio": floor_ratio,
            "required_maximum_ratio": sensitivity_gate_ratio,
        }
        run_selected_sensitivity = bool(
            ablation_cfg.get("run_sensitivity_cases_if_floor_passes", True)
        ) and floor_ratio < sensitivity_gate_ratio
        if run_selected_sensitivity:
            selected_case_truth = {
                "waist_minus": (i_minus, p_minus),
                "waist_plus": (i_plus, p_plus),
            }
            for case_id, (case_intensity, case_probe_true) in (
                selected_case_truth.items()
            ):
                raw_result = _reconstruct_case(
                    config,
                    case_intensity,
                    scan_positions,
                    incident,
                    np.asarray(controls["reference_region_mask"], dtype=bool),
                    dx_m,
                    variant_id=selected_variant,
                    num_iters=screening_num_iters,
                    sample_b_true_diagnostic=sample_b,
                    probe_true_diagnostic=case_probe_true,
                    radial_operator=radial_operator,
                    radial_operator_norm_squared=radial_operator_norm_squared,
                )
                evaluation, evaluation_metrics = _evaluate_reconstruction_case(
                    raw_result, case_probe_true, sample_b
                )
                variant_outputs[selected_variant]["cases"][case_id] = {
                    **raw_result,
                    "simulation_evaluation_only": evaluation,
                }
                variant_metrics[selected_variant]["cases"][case_id] = {
                    "initial_data_fidelity_loss": float(
                        raw_result["initial_data_fidelity_loss"]
                    ),
                    "final_data_fidelity_loss": float(
                        raw_result["final_data_fidelity_loss"]
                    ),
                    "simulation_evaluation_only": evaluation_metrics,
                }

            selected_cases = variant_outputs[selected_variant]["cases"]
            baseline_raw = _reconstruction_probe(selected_cases["baseline"])
            minus_raw = _reconstruction_probe(selected_cases["waist_minus"])
            plus_raw = _reconstruction_probe(selected_cases["waist_plus"])
            minus_to_baseline, minus_gain, minus_ramp = (
                align_affine_phase_and_complex_gain(minus_raw, baseline_raw)
            )
            plus_to_baseline, plus_gain, plus_ramp = (
                align_affine_phase_and_complex_gain(plus_raw, baseline_raw)
            )
            baseline_evaluation = selected_cases["baseline"][
                "simulation_evaluation_only"
            ]
            baseline_alignment = baseline_evaluation["alignment_parameters"]
            anchor_gain = complex(baseline_alignment["P_B_complex_gain"])
            anchor_ramp = tuple(
                float(value)
                for value in baseline_alignment[
                    "P_B_phase_ramp_yx_rad_per_px"
                ]
            )
            baseline_common = np.asarray(
                baseline_evaluation["P_B_rec_aligned_to_truth"]
            )
            minus_common = _apply_affine_alignment(
                minus_to_baseline, anchor_gain, anchor_ramp
            )
            plus_common = _apply_affine_alignment(
                plus_to_baseline, anchor_gain, anchor_ramp
            )
            selected_cases["baseline"]["simulation_evaluation_only"][
                "P_B_rec_common_gauge"
            ] = baseline_common
            selected_cases["waist_minus"]["simulation_evaluation_only"].update(
                {
                    "P_B_rec_common_gauge": minus_common,
                    "relative_alignment_to_recovered_baseline": {
                        "complex_gain": minus_gain,
                        "phase_ramp_yx_rad_per_px": minus_ramp,
                    },
                }
            )
            selected_cases["waist_plus"]["simulation_evaluation_only"].update(
                {
                    "P_B_rec_common_gauge": plus_common,
                    "relative_alignment_to_recovered_baseline": {
                        "complex_gain": plus_gain,
                        "phase_ramp_yx_rad_per_px": plus_ramp,
                    },
                }
            )
            _, recovered_probe_sensitivity = normalized_complex_sensitivity(
                minus_common,
                plus_common,
                baseline_common,
                delta_d,
                d0,
            )
            recovered_minus_difference = compare_probe_sensitivity(
                baseline_common, minus_common
            )["gauge_aligned_complex_relative_l2"]
            recovered_plus_difference = compare_probe_sensitivity(
                baseline_common, plus_common
            )["gauge_aligned_complex_relative_l2"]
            sensitivity_check.update(
                {
                    "status": "executed_floor_gate_passed",
                    "normalized_recovered_probe_sensitivity": (
                        recovered_probe_sensitivity
                    ),
                    "true_probe_sensitivity": normalized_probe,
                    "recovered_to_true_sensitivity_relative_deviation": (
                        _relative_change(
                            recovered_probe_sensitivity, normalized_probe
                        )
                    ),
                    "true_minus_difference": true_minus_difference,
                    "true_plus_difference": true_plus_difference,
                    "recovered_minus_difference": recovered_minus_difference,
                    "recovered_plus_difference": recovered_plus_difference,
                    "sensitivity_ordering_matches_true": bool(
                        (true_minus_difference <= true_plus_difference)
                        == (
                            recovered_minus_difference
                            <= recovered_plus_difference
                        )
                    ),
                    "gauge_evaluation_method": (
                        "case-to-recovered-baseline alignment followed by one "
                        "baseline-to-truth simulation-evaluation-only anchor"
                    ),
                }
            )
        else:
            sensitivity_check.update(
                {
                    "status": "not_run_reconstruction_floor_exceeds_true_signal",
                    "reason": (
                        "Baseline reconstruction error must be below the true "
                        "plus/minus probe separation before a recovered finite "
                        "difference is scientifically interpretable."
                    ),
                }
            )

        blind_long_output: dict[str, Any] = {
            "status": "not_run_disabled"
        }
        blind_long_metrics: dict[str, Any] = {
            "status": "not_run_disabled"
        }
        blind_long_cfg = ablation_cfg.get("blind_long_study", {})
        blind_long_enabled = bool(blind_long_cfg.get("enabled", False))
        if resolved_resume_checkpoint is not None and not blind_long_enabled:
            msg = "A blind checkpoint was supplied but blind_long_study is disabled."
            raise ValueError(msg)
        if blind_long_enabled:
            (
                blind_long_output,
                blind_long_metrics,
                blind_long_plot_fields,
                blind_long_plot_losses,
                blind_long_plot_labels,
            ) = _run_blind_long_study(
                config,
                run_dir=run_dir,
                resume_checkpoint_path=resolved_resume_checkpoint,
                scan_positions=scan_positions,
                incident=incident,
                reference_mask=np.asarray(
                    controls["reference_region_mask"], dtype=bool
                ),
                dx_m=dx_m,
                radial_operator=radial_operator,
                radial_operator_norm_squared=radial_operator_norm_squared,
                sample_b=sample_b,
                intensity_by_case={
                    "baseline": i_base,
                    "waist_minus": i_minus,
                    "waist_plus": i_plus,
                },
                probe_true_by_case={
                    "baseline": p_base,
                    "waist_minus": p_minus,
                    "waist_plus": p_plus,
                },
                true_probe_case_separation=true_probe_case_separation,
                true_detector_amplitude_separation=(
                    true_detector_amplitude_separation
                ),
                delta_d_waist_m=delta_d,
                nominal_d_waist_m=d0,
                true_normalized_probe_sensitivity=normalized_probe,
            )
            recovered_probe_fields.extend(blind_long_plot_fields)
            recovered_loss_curves.extend(blind_long_plot_losses)
            recovered_plot_labels.extend(blind_long_plot_labels)

        reconstruction_metrics = {
            "status": reconstruction_status,
            "executed": True,
            "operator_consistency": operator_consistency_metrics,
            "operator_consistency_ablation": {
                "variant_ids": variant_ids,
                "screening_num_iters": screening_num_iters,
                "variants": variant_metrics,
                "selected_sensitivity_check": sensitivity_check,
                "optimizer_study": optimizer_study_metrics,
                "blind_long_study": blind_long_metrics,
            },
        }
        reconstruction_output = {
            "operator_consistency_ablation": {
                "variant_ids": variant_ids,
                "variants": variant_outputs,
                "selected_sensitivity_check": sensitivity_check,
                "optimizer_study": optimizer_study_output,
                "blind_long_study": blind_long_output,
            }
        }
        del radial_operator
        gc.collect()
    else:
        reconstruction_status = "not_run_disabled_after_sampling_convergence_passed"
        reconstruction_metrics = {
            "status": reconstruction_status,
            "executed": False,
            "reason": str(reconstruction_cfg.get("reason_if_disabled", "")),
        }

    if bool(reconstruction_metrics.get("executed", False)):
        ablation_metrics = reconstruction_metrics[
            "operator_consistency_ablation"
        ]
        blind_long_result = ablation_metrics.get("blind_long_study", {})
        selected_check = (
            blind_long_result["selected_sensitivity_check"]
            if str(blind_long_result.get("status", "")).startswith("executed_")
            else ablation_metrics["selected_sensitivity_check"]
        )
        if selected_check.get("status") == "executed_floor_gate_passed" and (
            float(
                selected_check[
                    "recovered_to_true_sensitivity_relative_deviation"
                ]
            )
            <= 0.1
            and bool(selected_check["sensitivity_ordering_matches_true"])
        ):
            stage_d_status = "Passed"
        else:
            stage_d_status = "Inconclusive"
    else:
        stage_d_status = "NotExecuted"
    experiment_status = (
        "Passed"
        if stage_a_c_status == "Passed" and stage_d_status == "Passed"
        else "Inconclusive"
    )
    reconstruction_metrics["scientific_status"] = stage_d_status

    model_metrics["analytic_controls_pass"] = bool(controls_pass)
    model_metrics["sampling_convergence_pass"] = bool(convergence_pass)
    config["experiment"]["status"] = experiment_status
    metrics: dict[str, Any] = {
        "experiment_status": experiment_status,
        "stage_status": {
            "stage_A_to_C": stage_a_c_status,
            "stage_D": stage_d_status,
        },
        "model_validation": model_metrics,
        "probe_sensitivity": probe_metrics,
        "intensity_sensitivity": intensity_metrics,
        "sampling_convergence": convergence_metrics,
        "local_observability": observability,
        "reconstruction_check": reconstruction_metrics,
    }

    metadata = {
        "run_name": str(run_cfg["name"]),
        "experiment": "exp030_TGV_2d_effective_phase",
        "phase": "Phase 3",
        "experiment_status": experiment_status,
        "stage_A_to_C_status": stage_a_c_status,
        "stage_D_status": stage_d_status,
        "dataset_type": "simulation",
        "created_at": created_at_utc(),
        "git_commit": get_git_commit(PROJECT_ROOT) or "",
        "config_path": str(config_path),
        "shape_ny_nx": list(shape),
        "dx_tuple_order_if_used": "dy_dx",
        "scan_coordinate_order": "x_y",
        "scan_position_unit": "m",
        "sample_A_scope": "single_axisymmetric_air_filled_TGV_in_glass",
        "model_boundary": (
            "projected phase; no internal diffraction, refraction, reflection, "
            "multiple scattering, polarization, roughness, tilt, or noncircularity"
        ),
        "truth_used_by_primary_blind_reconstruction": False,
        "truth_used_by_known_B_or_known_probe_ablation_controls": True,
        "truth_use_boundary": (
            "known-B/known-probe are labelled simulation diagnostics; blind "
            "variants use truth only under simulation_evaluation_only"
        ),
        "paired_cases_share_B_scan_and_seeds": True,
        "a_to_b_forward_solver": (
            "continuous_axisymmetric_fresnel_hankel_on_compact_T_minus_1"
        ),
        "a_to_b_reference_field": "infinite_plane_wave_propagated_analytically",
        "coarse_A_effective_true_role": (
            "visualization_and_projected_path_validation_not_direct_ASM_input"
        ),
        "detector_pixel_semantics": "fixed_area_average_irradiance",
        "stage_d_forward_note": (
            "operator-consistency ablation separates the exact B-to-C truth-pair "
            "floor, the legacy coarse A constraint, detector-energy normalization, "
            "blind ambiguity, and the weighted continuous-radial A-to-B adjoint"
        ),
    }

    if bool(output_cfg.get("save_png", True)):
        plot_diameter_profile(
            baseline["z_m"],
            baseline["diameter_z_m"],
            figures_dir / "diameter_profile.png",
        )
        radial_plot = np.linspace(
            0.0, 0.6 * max(float(tgv["d_top_m"]), float(tgv["d_bottom_m"])), 800
        )
        analytic_radial = analytic_air_path_length(
            radial_plot,
            float(tgv["thickness_m"]),
            float(tgv["d_top_m"]),
            d0,
            float(tgv["d_bottom_m"]),
            float(tgv["z_waist_m"]),
        )
        plot_radial_profiles(
            radial_plot,
            [analytic_radial],
            ["analytic symmetric waist"],
            figures_dir / "fill_path_radial_profile.png",
        )
        plot_effective_transmission(
            baseline["A_effective_true"],
            dx_m,
            figures_dir / "effective_transmission.png",
        )
        plot_opd_and_phase(
            baseline["opd_relative_m"],
            baseline["phase_unwrapped_rad"],
            dx_m,
            figures_dir / "opd_and_unwrapped_phase.png",
        )
        radial_phases = [
            2.0
            * np.pi
            / wavelength_m
            * (float(tgv["n_air"]) - float(tgv["n_glass"]))
            * analytic_air_path_length(
                radial_plot,
                float(tgv["thickness_m"]),
                float(tgv["d_top_m"]),
                float(waist),
                float(tgv["d_bottom_m"]),
                float(tgv["z_waist_m"]),
            )
            for waist in waist_values
        ]
        plot_phase_profiles(
            radial_plot,
            radial_phases,
            [f"D_waist={waist * 1e6:.2f} um" for waist in waist_values],
            figures_dir / "waist_sweep_phase_profiles.png",
        )
        plot_probe_sensitivity_maps(
            p_base,
            p_minus,
            p_plus,
            probe_derivative * d0,
            dx_m,
            figures_dir / "probe_sensitivity_maps.png",
        )
        representative = int(np.argmax(per_frame))
        plot_intensity_sensitivity(
            i_base[representative],
            intensity_derivative[representative] * d0,
            per_frame,
            float(optics["detector_pixel_size_m"]),
            figures_dir / "intensity_sensitivity.png",
        )
        curve = np.asarray(
            [
                compare_probe_sensitivity(p_base, probe)[
                    "gauge_aligned_complex_relative_l2"
                ]
                for probe in probes
            ]
        )
        plot_sensitivity_curve(
            waist_values, curve, figures_dir / "sensitivity_curve.png"
        )
        plot_step_convergence(
            delta_steps,
            np.asarray(
                baseline_step_convergence["normalized_probe_sensitivity"]
            ),
            np.asarray(
                baseline_step_convergence["normalized_intensity_sensitivity"]
            ),
            np.asarray(fine_step_convergence["normalized_probe_sensitivity"]),
            np.asarray(
                fine_step_convergence["normalized_intensity_sensitivity"]
            ),
            figures_dir / "delta_d_step_convergence.png",
        )
        plot_jacobian_correlation(
            correlation, labels, figures_dir / "jacobian_correlation.png"
        )
        plot_jacobian_singular_values(
            np.asarray(observability["singular_values"]),
            figures_dir / "jacobian_singular_values.png",
        )
        if reconstruction_output is not None:
            plot_recovered_probe_cases(
                recovered_probe_fields,
                recovered_plot_labels,
                dx_m,
                figures_dir / "recovered_probe_cases.png",
            )
            plot_loss_curves(
                recovered_loss_curves,
                recovered_plot_labels,
                figures_dir / "loss_curves.png",
            )

    save_config(run_dir / "config.yaml", config)
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "metrics.json", metrics)
    if bool(output_cfg.get("save_hdf5", True)):
        save_ptycho_hdf5(
            outputs_dir / "exp030_effective_phase.h5",
            I_stack=i_base,
            scan_positions=scan_positions,
            instrument={
                "wavelength": wavelength_m,
                "dx": dx_m,
                "z_AB": z_ab_m,
                "z_BC": float(optics["z_BC_m"]),
                "detector_pixel_size": float(optics["detector_pixel_size_m"]),
                "medium_index": float(optics["medium_index"]),
            },
            sample={
                "sample_A_type": str(tgv["type"]),
                "tgv_parameters": tgv,
                "effective_model": {
                    "approximation": str(projected_cfg["approximation"]),
                    "reference_index": float(tgv["n_glass"]),
                    "propagation_axis": "z",
                    "integration_method": str(
                        projected_cfg["integration_method"]
                    ),
                    "lateral_supersampling": supersampling,
                    "a_to_b_solver": (
                        "continuous_axisymmetric_fresnel_hankel_on_T_minus_1"
                    ),
                    "radial_source_transition_step_m": radial_source_step,
                    "radial_output_step_m": radial_output_step,
                },
                "sample_B_type": str(sample_b_cfg["type"]),
                "sample_B_parameters": sample_b_cfg,
            },
            truth={
                "z_m": baseline["z_m"],
                "diameter_z_m": baseline["diameter_z_m"],
                "fill_path_length_m": baseline["fill_path_length_m"],
                "opd_relative_m": baseline["opd_relative_m"],
                "phase_unwrapped_rad": baseline["phase_unwrapped_rad"],
                "A_effective_true": baseline["A_effective_true"],
                "incident_field_true": incident,
                "U_after_A_true": baseline["A_effective_true"] * incident,
                "P_B_true": p_base,
                "B_true": sample_b,
                "reference_region_mask": controls["reference_region_mask"],
                "effective_forward": {
                    "radial_source_r_m": radial_result["source_radius_m"],
                    "radial_source_weight_m": radial_result[
                        "source_weights_m"
                    ],
                    "A_effective_radial_true": np.asarray(
                        radial_result["source_transmission"]
                    )[
                        list(radial_result["case_labels"]).index("baseline")
                    ],
                    "P_B_radial_r_m": radial_result["probe_radius_m"],
                    "P_B_radial_true": np.asarray(
                        radial_result["probe_radial"]
                    )[list(radial_result["case_labels"]).index("baseline")],
                },
                "parameter_sweep": {
                    "case_id": ["waist_minus", "baseline", "waist_plus"],
                    "d_waist_m": waist_values,
                    "A_effective_true": np.stack(
                        [model["A_effective_true"] for model in models]
                    ),
                    "P_B_true": np.stack(probes),
                    "I_stack_true": np.stack(intensity_stacks),
                    "radial_source_r_m": radial_result["source_radius_m"],
                    "A_effective_radial_true": np.asarray(
                        radial_result["source_transmission"]
                    )[
                        [
                            list(radial_result["case_labels"]).index(label)
                            for label in (
                                "waist_minus",
                                "baseline",
                                "waist_plus",
                            )
                        ]
                    ],
                    "P_B_radial_r_m": radial_result["probe_radius_m"],
                    "P_B_radial_true": np.asarray(
                        radial_result["probe_radial"]
                    )[
                        [
                            list(radial_result["case_labels"]).index(label)
                            for label in (
                                "waist_minus",
                                "baseline",
                                "waist_plus",
                            )
                        ]
                    ],
                },
            },
            reconstruction=reconstruction_output,
            config_yaml=config_to_yaml(config),
            metadata=metadata,
            metrics=metrics,
        )

    save_json(
        run_dir / "run_state.json",
        {
            "status": "complete",
            "experiment_status": experiment_status,
            "stage_A_to_C_status": stage_a_c_status,
            "stage_D_status": stage_d_status,
            "config_path": str(config_path),
            "resume_blind_checkpoint": (
                str(resolved_resume_checkpoint)
                if resolved_resume_checkpoint is not None
                else None
            ),
        },
    )

    print(f"Saved run to: {run_dir}")
    print(
        f"Status: {experiment_status}; normalized probe sensitivity="
        f"{normalized_probe:.6e}; normalized intensity sensitivity="
        f"{normalized_intensity:.6e}"
    )
    print(
        "Sampling relative changes: "
        f"dz={model_metrics['dz_convergence_relative_change']:.3%}, "
        f"dx={model_metrics['dx_convergence_relative_change']:.3%}, "
        "radial source="
        f"{model_metrics['radial_source_convergence_relative_change']:.3%}, "
        "delta-D step="
        f"{model_metrics['delta_d_step_convergence_relative_change']:.3%}"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config, args.resume_blind_checkpoint)


if __name__ == "__main__":
    main()
