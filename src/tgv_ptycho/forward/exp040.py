"""Reusable orchestration helpers for the exp040 multi-slice forward study."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from tgv_ptycho.forward.camera import (
    make_square_pixel_mtf,
    positive_midpoint_pixel_average,
)
from tgv_ptycho.forward.integer_shift import shift_field_integer_pixels
from tgv_ptycho.forward.multislice_A import (
    multislice_phase_screen_product,
    multislice_propagate_A,
    multislice_propagate_streamed_A,
)
from tgv_ptycho.forward.scan import add_integer_pixel_jitter, make_grid_scan
from tgv_ptycho.forward.scheme_probe_B import simulate_exit_field_B_forward
from tgv_ptycho.objects.sample_b import make_random_phase_object
from tgv_ptycho.objects.tgv2d import make_tgv_projected_phase
from tgv_ptycho.objects.tgv3d import (
    make_tgv_air_fraction_slice,
    make_tgv_refractive_index_volume,
)
from tgv_ptycho.objects.tgv_geometry import (
    diameter_profile,
    midpoint_z_grid,
    validate_tgv_geometry,
)
from tgv_ptycho.optics.angular_spectrum import (
    angular_spectrum_propagate,
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
    make_transfer_sampling_alias_mask,
)
from tgv_ptycho.optics.fields import coordinate_grid, make_plane_wave


def relative_l2(
    test: NDArray[np.generic],
    reference: NDArray[np.generic],
) -> float:
    """Return an unaligned relative L2 difference on a common grid."""

    test_values = np.asarray(test)
    reference_values = np.asarray(reference)
    if test_values.shape != reference_values.shape:
        msg = "test and reference must have the same shape."
        raise ValueError(msg)
    numerator = float(np.sum(np.abs(test_values - reference_values) ** 2))
    denominator = float(np.sum(np.abs(reference_values) ** 2))
    epsilon = np.finfo(np.float64).eps
    return float(np.sqrt(numerator) / max(np.sqrt(denominator), epsilon))


def center_crop(
    values: NDArray[np.generic],
    target_shape: tuple[int, int],
) -> NDArray[Any]:
    """Crop the last two axes to a centered ``(ny, nx)`` region."""

    array = np.asarray(values)
    if array.ndim < 2:
        msg = "values must have at least two dimensions."
        raise ValueError(msg)
    target_y, target_x = _shape(target_shape, "target_shape")
    source_y, source_x = array.shape[-2:]
    if target_y > source_y or target_x > source_x:
        msg = "target_shape must not exceed the source spatial shape."
        raise ValueError(msg)
    start_y = (source_y - target_y) // 2
    start_x = (source_x - target_x) // 2
    if source_y - target_y != 2 * start_y:
        msg = "source and target y sizes must have aligned centers."
        raise ValueError(msg)
    if source_x - target_x != 2 * start_x:
        msg = "source and target x sizes must have aligned centers."
        raise ValueError(msg)
    return array[
        ...,
        start_y : start_y + target_y,
        start_x : start_x + target_x,
    ]


def resample_centered_grid(
    values: NDArray[np.generic],
    source_dx_m: float,
    target_shape: tuple[int, int],
    target_dx_m: float,
) -> NDArray[Any]:
    """Linearly sample the last two axes at centered target pixel centers.

    This mapping is an exp040 convergence diagnostic. It is not part of the
    detector forward model and does not define a project-wide remap standard.
    """

    array = np.asarray(values)
    if array.ndim < 2:
        msg = "values must have at least two dimensions."
        raise ValueError(msg)
    source_dx = _positive(source_dx_m, "source_dx_m")
    target_dx = _positive(target_dx_m, "target_dx_m")
    target_y, target_x = _shape(target_shape, "target_shape")
    source_y, source_x = array.shape[-2:]

    if (
        source_y == target_y
        and source_x == target_x
        and source_dx == target_dx
    ):
        return array.copy()

    target_y_m = (
        np.arange(target_y, dtype=np.float64) - (target_y - 1) / 2.0
    ) * target_dx
    target_x_m = (
        np.arange(target_x, dtype=np.float64) - (target_x - 1) / 2.0
    ) * target_dx
    source_y_index = target_y_m / source_dx + (source_y - 1) / 2.0
    source_x_index = target_x_m / source_dx + (source_x - 1) / 2.0
    yy, xx = np.meshgrid(source_y_index, source_x_index, indexing="ij")
    tolerance = 32.0 * np.finfo(np.float64).eps * max(source_y, source_x)
    if (
        np.min(yy) < -tolerance
        or np.max(yy) > source_y - 1 + tolerance
        or np.min(xx) < -tolerance
        or np.max(xx) > source_x - 1 + tolerance
    ):
        msg = "target grid extends beyond the source field of view."
        raise ValueError(msg)
    coordinates = np.asarray([yy, xx], dtype=np.float64)
    flattened = array.reshape((-1, source_y, source_x))
    output_shape = (flattened.shape[0], target_y, target_x)

    if np.iscomplexobj(array):
        sampled = np.empty(output_shape, dtype=np.complex128)
        for index, plane in enumerate(flattened):
            sampled[index] = map_coordinates(
                np.asarray(plane.real, dtype=np.float64),
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            ) + 1j * map_coordinates(
                np.asarray(plane.imag, dtype=np.float64),
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
    else:
        sampled = np.empty(output_shape, dtype=np.float64)
        for index, plane in enumerate(flattened):
            sampled[index] = map_coordinates(
                np.asarray(plane, dtype=np.float64),
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
    return sampled.reshape((*array.shape[:-2], target_y, target_x))


def restrict_aligned_cell_average(
    values: NDArray[np.generic], refinement_ratio: int
) -> NDArray[Any]:
    """Restrict aligned nested cells by a conservative complex average."""

    array = np.asarray(values)
    if array.ndim < 2:
        msg = "values must have at least two dimensions."
        raise ValueError(msg)
    if (
        isinstance(refinement_ratio, bool)
        or int(refinement_ratio) != refinement_ratio
        or int(refinement_ratio) <= 0
    ):
        msg = "refinement_ratio must be a positive integer."
        raise ValueError(msg)
    ratio = int(refinement_ratio)
    source_y, source_x = array.shape[-2:]
    if source_y % ratio != 0 or source_x % ratio != 0:
        msg = "source shape must be divisible by refinement_ratio."
        raise ValueError(msg)
    target_y = source_y // ratio
    target_x = source_x // ratio
    reshaped = array.reshape(
        (*array.shape[:-2], target_y, ratio, target_x, ratio)
    )
    return np.mean(reshaped, axis=(-3, -1))


def make_physical_passband_mask(
    shape: tuple[int, int],
    dx_m: float,
    cutoff_cycles_per_m: float,
) -> NDArray[np.bool_]:
    """Return an inclusive radial physical-frequency mask on an FFT grid."""

    ny, nx = _shape(shape, "passband shape")
    dx_value = _positive(dx_m, "passband dx_m")
    cutoff = _positive(cutoff_cycles_per_m, "passband cutoff")
    fy = np.fft.fftfreq(ny, d=dx_value)
    fx = np.fft.fftfreq(nx, d=dx_value)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    return fx_grid**2 + fy_grid**2 <= cutoff**2


def project_field_to_passband(
    field: NDArray[np.generic],
    dx_m: float,
    cutoff_cycles_per_m: float,
) -> NDArray[np.complex128]:
    """Orthogonally project one complex field onto a radial FFT passband."""

    values = np.asarray(field, dtype=np.complex128)
    if values.ndim != 2:
        msg = "field must be a two-dimensional complex array."
        raise ValueError(msg)
    mask = make_physical_passband_mask(
        values.shape, dx_m, cutoff_cycles_per_m
    )
    return np.fft.ifft2(np.fft.fft2(values) * mask).astype(
        np.complex128, copy=False
    )


def validate_exp040_config(config: Mapping[str, Any]) -> None:
    """Validate the pre-registered exp040 configuration."""

    if not isinstance(config, Mapping):
        msg = "config must be a mapping."
        raise ValueError(msg)
    run_cfg = _section(config, "run")
    if not str(run_cfg.get("name", "")).strip():
        msg = "run.name must be a non-empty string."
        raise ValueError(msg)
    if not str(run_cfg.get("output_root", "")).strip():
        msg = "run.output_root must be a non-empty path."
        raise ValueError(msg)
    optics = _section(config, "optics")
    illumination = _section(config, "illumination")
    sample_a = _section(config, "sample_a")
    multislice = _section(config, "multislice")
    sample_b = _section(config, "sample_b")
    scan = _section(config, "scan")
    perturbation = _section(config, "waist_perturbation")
    convergence = _section(config, "convergence")
    acceptance = _section(config, "acceptance")

    wavelength = _positive(optics["wavelength_m"], "optics.wavelength_m")
    del wavelength
    n_reference = _positive(
        optics["internal_reference_index"],
        "optics.internal_reference_index",
    )
    _positive(
        optics["external_medium_index"],
        "optics.external_medium_index",
    )
    baseline_shape = _shape(optics["baseline_shape"], "optics.baseline_shape")
    baseline_dx = _isotropic_dx(
        optics["baseline_dx_m"], "optics.baseline_dx_m"
    )
    _positive(optics["z_AB_m"], "optics.z_AB_m", allow_zero=True)
    _positive(optics["z_BC_m"], "optics.z_BC_m", allow_zero=True)
    if not isinstance(optics["angular_spectrum_bandlimit"], bool):
        msg = "optics.angular_spectrum_bandlimit must be boolean."
        raise ValueError(msg)
    detector = _section(optics, "detector")
    detector_pixel = _positive(
        detector["pixel_size_m"], "optics.detector.pixel_size_m"
    )
    if detector.get("pixel_integration") is not False:
        msg = "exp040 requires detector.pixel_integration=false."
        raise ValueError(msg)
    if not np.isclose(detector_pixel, baseline_dx, rtol=0.0, atol=0.0):
        msg = "detector pixel size must equal the baseline grid sampling."
        raise ValueError(msg)
    if detector.get("model") != "grid_sampled_intensity":
        msg = "exp040 requires the grid_sampled_intensity detector model."
        raise ValueError(msg)

    if illumination.get("type") != "plane_wave":
        msg = "exp040 currently requires plane-wave illumination."
        raise ValueError(msg)
    _positive(illumination["amplitude"], "illumination.amplitude")
    angles = np.asarray(
        [illumination["theta_x_rad"], illumination["theta_y_rad"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(angles)):
        msg = "illumination angles must be finite."
        raise ValueError(msg)
    if illumination.get("seed") is not None:
        msg = "deterministic plane-wave illumination must use seed=null."
        raise ValueError(msg)

    thickness = _positive(sample_a["thickness_m"], "sample_a.thickness_m")
    for name in ("d_top_m", "d_waist_m", "d_bottom_m", "z_waist_m"):
        _positive(sample_a[name], f"sample_a.{name}")
    validate_tgv_geometry(
        thickness,
        float(sample_a["d_top_m"]),
        float(sample_a["d_waist_m"]),
        float(sample_a["d_bottom_m"]),
        float(sample_a["z_waist_m"]),
    )
    n_glass = _positive(sample_a["n_glass"], "sample_a.n_glass")
    _positive(sample_a["n_air"], "sample_a.n_air")
    if not np.isclose(n_reference, n_glass, rtol=0.0, atol=0.0):
        msg = "projected control requires internal reference index=n_glass."
        raise ValueError(msg)
    center = sample_a["center_xy_m"]
    if not isinstance(center, Sequence) or len(center) != 2:
        msg = "sample_a.center_xy_m must be [x, y]."
        raise ValueError(msg)
    if not np.all(np.isfinite(np.asarray(center, dtype=np.float64))):
        msg = "sample_a.center_xy_m must be finite."
        raise ValueError(msg)
    if sample_a.get("voxelization") != "voxel_center_binary":
        msg = "exp040 requires voxel_center_binary voxelization."
        raise ValueError(msg)

    target_dz = _positive(multislice["target_dz_m"], "multislice.target_dz_m")
    if multislice.get("operator") != "centered_symmetric_split_step":
        msg = "exp040 requires centered_symmetric_split_step."
        raise ValueError(msg)
    if multislice.get("use_true_slice_widths") is not True:
        msg = "multislice.use_true_slice_widths must be true."
        raise ValueError(msg)
    if multislice.get("store_slice_fields") is not False:
        msg = "slice fields must not be stored by default."
        raise ValueError(msg)
    midpoint_z_grid(thickness, target_dz)

    canonical = _section(sample_b, "canonical_grid")
    canonical_shape = _shape(
        canonical["shape"], "sample_b.canonical_grid.shape"
    )
    canonical_dx = _positive(
        canonical["dx_m"], "sample_b.canonical_grid.dx_m"
    )
    canonical_fov = np.asarray(canonical["fov_m"], dtype=np.float64)
    if canonical_fov.shape != (2,) or not np.all(canonical_fov > 0.0):
        msg = "sample_b.canonical_grid.fov_m must contain positive [y, x]."
        raise ValueError(msg)
    expected_fov = np.asarray(canonical_shape, dtype=np.float64) * canonical_dx
    if not np.allclose(canonical_fov, expected_fov, rtol=1e-12, atol=0.0):
        msg = "canonical shape, dx, and fov are inconsistent."
        raise ValueError(msg)
    feature_size = _positive(
        sample_b["physical_feature_size_m"],
        "sample_b.physical_feature_size_m",
    )
    _require_integer_ratio(feature_size, canonical_dx, "sample B feature size")
    if not isinstance(sample_b.get("seed"), int):
        msg = "sample_b.seed must be an integer."
        raise ValueError(msg)
    phase_range = float(sample_b["phase_range_rad"])
    if not np.isfinite(phase_range) or phase_range < 0.0:
        msg = "sample_b.phase_range_rad must be finite and non-negative."
        raise ValueError(msg)
    if sample_b.get("object_boundary") != "periodic":
        msg = "the registered exp040 baseline requires periodic sample B."
        raise ValueError(msg)

    if int(scan["num_x"]) <= 0 or int(scan["num_y"]) <= 0:
        msg = "scan dimensions must be positive."
        raise ValueError(msg)
    step = _positive(scan["step_m"], "scan.step_m")
    jitter_quantum = _positive(
        scan["jitter_quantum_m"], "scan.jitter_quantum_m"
    )
    if int(scan["max_jitter_px"]) < 0:
        msg = "scan.max_jitter_px must be non-negative."
        raise ValueError(msg)
    if not isinstance(scan.get("jitter_seed"), int):
        msg = "scan.jitter_seed must be an integer."
        raise ValueError(msg)
    if scan.get("integer_pixel_shifts_only") is not True:
        msg = "exp040 requires integer-only scan shifts."
        raise ValueError(msg)
    if scan.get("type") != "jittered_grid":
        msg = "exp040 requires a jittered_grid scan."
        raise ValueError(msg)

    delta_d = _positive(
        perturbation["delta_d_waist_m"],
        "waist_perturbation.delta_d_waist_m",
    )
    case_ids = list(perturbation["case_ids"])
    waist_values = np.asarray(perturbation["d_waist_m"], dtype=np.float64)
    if case_ids != ["waist_minus", "baseline", "waist_plus"]:
        msg = "waist case_ids must be minus, baseline, plus."
        raise ValueError(msg)
    expected_waists = float(sample_a["d_waist_m"]) + delta_d * np.asarray(
        [-1.0, 0.0, 1.0]
    )
    if not np.allclose(waist_values, expected_waists, rtol=0.0, atol=1e-18):
        msg = "waist sweep values must match the configured symmetric delta."
        raise ValueError(msg)
    for waist in waist_values:
        validate_tgv_geometry(
            thickness,
            float(sample_a["d_top_m"]),
            float(waist),
            float(sample_a["d_bottom_m"]),
            float(sample_a["z_waist_m"]),
        )

    axial = _section(convergence, "axial")
    if _shape(axial["fixed_shape"], "convergence.axial.fixed_shape") != (
        baseline_shape
    ):
        msg = "axial fixed shape must equal the baseline shape."
        raise ValueError(msg)
    if _isotropic_dx(axial["fixed_dx_m"], "axial.fixed_dx_m") != baseline_dx:
        msg = "axial fixed dx must equal baseline dx."
        raise ValueError(msg)
    axial_dz = [_positive(value, "axial dz") for value in axial["dz_cases_m"]]
    if target_dz not in axial_dz:
        msg = "baseline target dz must be an axial convergence case."
        raise ValueError(msg)

    lateral = _section(convergence, "lateral_fixed_fov")
    lateral_cases = list(lateral["cases"])
    if len(lateral_cases) < 2:
        msg = "at least two lateral convergence cases are required."
        raise ValueError(msg)
    all_dx: list[float] = []
    for index, case in enumerate(lateral_cases):
        case_shape = _shape(case["shape"], f"lateral case {index} shape")
        case_dx = _isotropic_dx(case["dx_m"], f"lateral case {index} dx")
        all_dx.append(case_dx)
        fov = np.asarray(case_shape, dtype=np.float64) * case_dx
        expected = np.asarray(lateral["fov_m"], dtype=np.float64)
        if not np.allclose(fov, expected, rtol=1e-12, atol=0.0):
            msg = "all lateral cases must have the registered fixed FOV."
            raise ValueError(msg)

    fov_cfg = _section(convergence, "fov")
    fov_dx = _isotropic_dx(fov_cfg["fixed_dx_m"], "fov.fixed_dx_m")
    fov_shapes = [_shape(value, "fov shape") for value in fov_cfg["shapes"]]
    common_shape = _shape(
        fov_cfg["common_center_roi_shape"], "fov common shape"
    )
    if any(
        shape[0] < common_shape[0] or shape[1] < common_shape[1]
        for shape in fov_shapes
    ):
        msg = "FOV cases must contain the common center ROI."
        raise ValueError(msg)
    all_dx.append(fov_dx)
    all_dx.extend([baseline_dx, canonical_dx])
    for dx_value in all_dx:
        _require_integer_ratio(step, dx_value, "scan step")
        _require_integer_ratio(jitter_quantum, dx_value, "scan jitter quantum")
    max_fov = np.max(
        np.asarray(fov_shapes, dtype=np.float64) * fov_dx,
        axis=0,
    )
    if np.any(max_fov > canonical_fov * (1.0 + 1e-12)):
        msg = "canonical sample B does not cover every convergence FOV."
        raise ValueError(msg)

    _positive(
        acceptance["algebra_relative_l2_max"],
        "acceptance.algebra_relative_l2_max",
    )
    _positive(
        acceptance["convergence_relative_l2_max"],
        "acceptance.convergence_relative_l2_max",
    )
    _positive(
        acceptance["detector_visibility_signal_to_floor_min"],
        "acceptance.detector_visibility_signal_to_floor_min",
    )
    output = _section(config, "output")
    if output.get("save_hdf5") is not True or output.get("save_png") is not True:
        msg = "the registered exp040 run requires HDF5 and PNG outputs."
        raise ValueError(msg)
    if output.get("save_slice_fields") is not False:
        msg = "output.save_slice_fields must be false."
        raise ValueError(msg)
    if not str(output.get("hdf5_filename", "")).strip():
        msg = "output.hdf5_filename must be non-empty."
        raise ValueError(msg)
    diagnostics_r1 = config.get("diagnostics_r1")
    if diagnostics_r1 is not None:
        if not isinstance(diagnostics_r1, Mapping):
            msg = "diagnostics_r1 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r1.get("enabled") is True:
            _validate_r1_config(config, diagnostics_r1)
        elif diagnostics_r1.get("enabled") is not False:
            msg = "diagnostics_r1.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r2 = config.get("diagnostics_r2")
    if diagnostics_r2 is not None:
        if not isinstance(diagnostics_r2, Mapping):
            msg = "diagnostics_r2 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r2.get("enabled") is True:
            _validate_r2_config(config, diagnostics_r2)
        elif diagnostics_r2.get("enabled") is not False:
            msg = "diagnostics_r2.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r3 = config.get("diagnostics_r3")
    if diagnostics_r3 is not None:
        if not isinstance(diagnostics_r3, Mapping):
            msg = "diagnostics_r3 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r3.get("enabled") is True:
            _validate_r3_config(config, diagnostics_r3)
        elif diagnostics_r3.get("enabled") is not False:
            msg = "diagnostics_r3.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r4 = config.get("diagnostics_r4")
    if diagnostics_r4 is not None:
        if not isinstance(diagnostics_r4, Mapping):
            msg = "diagnostics_r4 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r4.get("enabled") is True:
            _validate_r4_config(config, diagnostics_r4)
        elif diagnostics_r4.get("enabled") is not False:
            msg = "diagnostics_r4.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r5 = config.get("diagnostics_r5")
    if diagnostics_r5 is not None:
        if not isinstance(diagnostics_r5, Mapping):
            msg = "diagnostics_r5 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r5.get("enabled") is True:
            _validate_r5_config(config, diagnostics_r5)
        elif diagnostics_r5.get("enabled") is not False:
            msg = "diagnostics_r5.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r6 = config.get("diagnostics_r6")
    if diagnostics_r6 is not None:
        if not isinstance(diagnostics_r6, Mapping):
            msg = "diagnostics_r6 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r6.get("enabled") is True:
            _validate_r6_config(config, diagnostics_r6)
        elif diagnostics_r6.get("enabled") is not False:
            msg = "diagnostics_r6.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r7 = config.get("diagnostics_r7")
    if diagnostics_r7 is not None:
        if not isinstance(diagnostics_r7, Mapping):
            msg = "diagnostics_r7 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r7.get("enabled") is True:
            _validate_r7_config(config, diagnostics_r7)
        elif diagnostics_r7.get("enabled") is not False:
            msg = "diagnostics_r7.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r8 = config.get("diagnostics_r8")
    if diagnostics_r8 is not None:
        if not isinstance(diagnostics_r8, Mapping):
            msg = "diagnostics_r8 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r8.get("enabled") is True:
            _validate_r8_config(config, diagnostics_r8)
        elif diagnostics_r8.get("enabled") is not False:
            msg = "diagnostics_r8.enabled must be boolean."
            raise ValueError(msg)
    diagnostics_r9 = config.get("diagnostics_r9")
    if diagnostics_r9 is not None:
        if not isinstance(diagnostics_r9, Mapping):
            msg = "diagnostics_r9 must be a mapping when present."
            raise ValueError(msg)
        if diagnostics_r9.get("enabled") is True:
            _validate_r9_config(config, diagnostics_r9)
        elif diagnostics_r9.get("enabled") is not False:
            msg = "diagnostics_r9.enabled must be boolean."
            raise ValueError(msg)


def _emit_runtime_progress(
    callback: Callable[[str, Mapping[str, Any]], None] | None,
    event: str,
    **details: Any,
) -> None:
    """Emit non-scientific execution progress without changing numerics."""

    if callback is not None:
        callback(event, details)


def run_exp040_experiment(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run all pre-registered exp040 numerical cases in memory."""

    validate_exp040_config(config)
    _emit_runtime_progress(progress_callback, "forward_started")
    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    multislice = _section(config, "multislice")
    sample_b_cfg = _section(config, "sample_b")
    scan_cfg = _section(config, "scan")

    baseline_shape = _shape(optics["baseline_shape"], "baseline shape")
    baseline_dx = _isotropic_dx(optics["baseline_dx_m"], "baseline dx")
    baseline_dz = float(multislice["target_dz_m"])
    canonical_b, canonical_dx = _make_canonical_b(sample_b_cfg)
    positions = _make_common_scan(scan_cfg)

    grid_specs = _all_grid_specs(config)
    for shape, dx_value in grid_specs:
        _validate_scan_on_grid(positions, dx_value)
        _sample_b_for_grid(canonical_b, canonical_dx, shape, dx_value)

    baseline_b = _sample_b_for_grid(
        canonical_b, canonical_dx, baseline_shape, baseline_dx
    )
    baseline = _simulate_case(
        config,
        shape=baseline_shape,
        dx_m=baseline_dx,
        dz_m=baseline_dz,
        d_waist_m=float(sample_a["d_waist_m"]),
        sample_b=baseline_b,
        scan_positions=positions,
        keep_volume=True,
    )
    _emit_runtime_progress(progress_callback, "baseline_completed")

    sweep_cases: list[dict[str, Any]] = []
    waist_cfg = _section(config, "waist_perturbation")
    for case_id, waist in zip(
        waist_cfg["case_ids"], waist_cfg["d_waist_m"], strict=True
    ):
        if case_id == "baseline":
            case = baseline
        else:
            case = _simulate_case(
                config,
                shape=baseline_shape,
                dx_m=baseline_dx,
                dz_m=baseline_dz,
                d_waist_m=float(waist),
                sample_b=baseline_b,
                scan_positions=positions,
                keep_volume=False,
            )
        sweep_cases.append(case)

    controls, control_metrics = _run_controls(
        config, baseline, positions, sweep_cases
    )
    convergence = _run_convergence_cases(
        config,
        baseline,
        canonical_b,
        canonical_dx,
        positions,
    )
    metrics = _assemble_metrics(
        config,
        baseline,
        sweep_cases,
        control_metrics,
        convergence,
    )
    _emit_runtime_progress(progress_callback, "legacy_core_completed")

    result = {
        "baseline": baseline,
        "controls": controls,
        "convergence": convergence,
        "sweep": {
            "case_ids": list(waist_cfg["case_ids"]),
            "d_waist_m": np.asarray(
                waist_cfg["d_waist_m"], dtype=np.float64
            ),
            "U_A_exit": np.stack(
                [case["U_A_exit"] for case in sweep_cases]
            ),
            "P_B": np.stack([case["P_B"] for case in sweep_cases]),
            "I_stack": np.stack([case["I_stack"] for case in sweep_cases]),
        },
        "scan_positions": positions,
        "canonical_sample_b": canonical_b,
        "canonical_sample_b_dx_m": canonical_dx,
        "metrics": metrics,
        "shared_inputs": {
            "sample_b_seed": int(sample_b_cfg["seed"]),
            "scan_seed": int(scan_cfg["jitter_seed"]),
            "same_incident_field": True,
            "same_physical_sample_b": True,
            "same_scan_positions": True,
            "same_detector_model": True,
        },
    }
    diagnostics_r1 = config.get("diagnostics_r1")
    if (
        isinstance(diagnostics_r1, Mapping)
        and diagnostics_r1.get("enabled") is True
    ):
        r1_result, r1_metrics = _run_r1_diagnostics(
            config,
            baseline,
            controls,
            convergence,
            positions,
            canonical_b,
            canonical_dx,
            metrics,
        )
        result["diagnostics_r1"] = r1_result
        metrics["diagnostics_r1"] = r1_metrics
    diagnostics_r2 = config.get("diagnostics_r2")
    if (
        isinstance(diagnostics_r2, Mapping)
        and diagnostics_r2.get("enabled") is True
    ):
        if "diagnostics_r1" not in metrics:
            msg = "R2 requires preserved diagnostics_r1 metrics."
            raise RuntimeError(msg)
        r2_result, r2_metrics = _run_r2_diagnostics(
            config,
            baseline,
            controls,
            positions,
            canonical_b,
            canonical_dx,
            _section(metrics, "diagnostics_r1"),
        )
        result["diagnostics_r2"] = r2_result
        metrics["diagnostics_r2"] = r2_metrics
    diagnostics_r3 = config.get("diagnostics_r3")
    if (
        isinstance(diagnostics_r3, Mapping)
        and diagnostics_r3.get("enabled") is True
    ):
        if "diagnostics_r1" not in metrics or "diagnostics_r2" not in metrics:
            msg = "R3 requires preserved diagnostics_r1 and diagnostics_r2 metrics."
            raise RuntimeError(msg)
        r3_result, r3_metrics = _run_r3_diagnostics(
            config,
            baseline,
            controls,
            positions,
            canonical_b,
            canonical_dx,
            _section(metrics, "diagnostics_r2"),
        )
        result["diagnostics_r3"] = r3_result
        metrics["diagnostics_r3"] = r3_metrics
    diagnostics_r4 = config.get("diagnostics_r4")
    if (
        isinstance(diagnostics_r4, Mapping)
        and diagnostics_r4.get("enabled") is True
    ):
        r4_result, r4_metrics = _run_r4_diagnostics(
            config,
            baseline,
            controls,
            positions,
            canonical_b,
            canonical_dx,
        )
        result["diagnostics_r4"] = r4_result
        metrics["diagnostics_r4"] = r4_metrics
    diagnostics_r5 = config.get("diagnostics_r5")
    if (
        isinstance(diagnostics_r5, Mapping)
        and diagnostics_r5.get("enabled") is True
    ):
        r5_result, r5_metrics = _run_r5_diagnostics(
            config,
            baseline,
            controls,
            positions,
            canonical_b,
            canonical_dx,
        )
        result["diagnostics_r5"] = r5_result
        metrics["diagnostics_r5"] = r5_metrics
    diagnostics_r6 = config.get("diagnostics_r6")
    if (
        isinstance(diagnostics_r6, Mapping)
        and diagnostics_r6.get("enabled") is True
    ):
        r6_result, r6_metrics = _run_r6_diagnostics(
            config,
            baseline,
            controls,
            positions,
            canonical_b,
            canonical_dx,
        )
        result["diagnostics_r6"] = r6_result
        metrics["diagnostics_r6"] = r6_metrics
    diagnostics_r7 = config.get("diagnostics_r7")
    if (
        isinstance(diagnostics_r7, Mapping)
        and diagnostics_r7.get("enabled") is True
    ):
        r7_result, r7_metrics = _run_r7_diagnostics(
            config,
            positions,
            canonical_b,
            canonical_dx,
        )
        result["diagnostics_r7"] = r7_result
        metrics["diagnostics_r7"] = r7_metrics
    diagnostics_r8 = config.get("diagnostics_r8")
    if (
        isinstance(diagnostics_r8, Mapping)
        and diagnostics_r8.get("enabled") is True
    ):
        r8_result, r8_metrics = _run_r8_diagnostics(
            config,
            positions,
            canonical_b,
            canonical_dx,
        )
        result["diagnostics_r8"] = r8_result
        metrics["diagnostics_r8"] = r8_metrics
    diagnostics_r9 = config.get("diagnostics_r9")
    if (
        isinstance(diagnostics_r9, Mapping)
        and diagnostics_r9.get("enabled") is True
    ):
        _emit_runtime_progress(progress_callback, "r9_started")
        r9_result, r9_metrics = _run_r9_diagnostics(
            config, progress_callback=progress_callback
        )
        result["diagnostics_r9"] = r9_result
        metrics["diagnostics_r9"] = r9_metrics
        _emit_runtime_progress(progress_callback, "r9_completed")
    _emit_runtime_progress(progress_callback, "forward_completed")
    return result


def _validate_r1_config(
    config: Mapping[str, Any], diagnostics_r1: Mapping[str, Any]
) -> None:
    """Validate the append-only exp040-R1 refinement registration."""

    if diagnostics_r1.get("version") != "R1":
        msg = "diagnostics_r1.version must be R1."
        raise ValueError(msg)
    if diagnostics_r1.get("preserve_r0_metrics_and_status") is not True:
        msg = "R1 must preserve R0 metrics and status."
        raise ValueError(msg)
    methods = _section(diagnostics_r1, "methods")
    expected_methods = {
        "lateral_b_mapping": "same_seed_piecewise_constant_fine_grid",
        "fov_b_extension": "centered_periodic_extension_of_base_96um",
        "external_field_padding": (
            "homogeneous_reference_plus_zero_padded_scattered_residual"
        ),
        "comparison_mapping": (
            "centered_bilinear_complex_field_diagnostic_only"
        ),
    }
    if any(methods.get(key) != value for key, value in expected_methods.items()):
        msg = "diagnostics_r1 methods do not match the pre-registration."
        raise ValueError(msg)

    sample_b = _section(config, "sample_b")
    scan = _section(config, "scan")
    refinement = _section(diagnostics_r1, "sample_b_refinement")
    base_grid = _section(refinement, "base_grid")
    working_grid = _section(refinement, "working_grid")
    base_shape = _shape(base_grid["shape"], "R1 B base shape")
    working_shape = _shape(working_grid["shape"], "R1 B working shape")
    base_dx = _isotropic_dx(base_grid["dx_m"], "R1 B base dx")
    working_dx = _isotropic_dx(working_grid["dx_m"], "R1 B working dx")
    if base_dx != working_dx:
        msg = "R1 B base and working grids must have the same sampling."
        raise ValueError(msg)
    for grid, shape, dx_m, name in (
        (base_grid, base_shape, base_dx, "base"),
        (working_grid, working_shape, working_dx, "working"),
    ):
        fov = np.asarray(grid["fov_m"], dtype=np.float64)
        expected_fov = np.asarray(shape, dtype=np.float64) * dx_m
        if fov.shape != (2,) or not np.allclose(
            fov, expected_fov, rtol=1e-12, atol=0.0
        ):
            msg = f"R1 B {name} shape, dx, and FOV are inconsistent."
            raise ValueError(msg)

    feature_size = float(sample_b["physical_feature_size_m"])
    base_feature_pixels = _require_integer_ratio(
        feature_size, base_dx, "R1 B physical feature size"
    )
    legacy_grid = _section(sample_b, "canonical_grid")
    legacy_shape = _shape(legacy_grid["shape"], "legacy canonical shape")
    legacy_dx = _isotropic_dx(legacy_grid["dx_m"], "legacy canonical dx")
    legacy_feature_pixels = _require_integer_ratio(
        feature_size, legacy_dx, "legacy B physical feature size"
    )
    base_cells = _shape(
        base_grid["coarse_phase_cell_shape"], "R1 coarse phase-cell shape"
    )
    if base_cells != (
        base_shape[0] // base_feature_pixels,
        base_shape[1] // base_feature_pixels,
    ) or base_cells != (
        legacy_shape[0] // legacy_feature_pixels,
        legacy_shape[1] // legacy_feature_pixels,
    ):
        msg = "R1 and R0 canonical grids must share one coarse phase-cell map."
        raise ValueError(msg)
    padding = _shape(
        refinement["extension_each_side_px"], "R1 extension_each_side_px"
    )
    if working_shape != (
        base_shape[0] + 2 * padding[0],
        base_shape[1] + 2 * padding[1],
    ):
        msg = "R1 working B shape must equal base shape plus centered padding."
        raise ValueError(msg)
    if refinement.get("extension") != "centered_periodic_tile_of_base_96um":
        msg = "R1 B must use the registered centered periodic extension."
        raise ValueError(msg)
    if refinement.get("same_seed_as_r0") is not True:
        msg = "R1 B must reuse the R0 seed."
        raise ValueError(msg)
    _positive(
        refinement["r0_mapping_max_complex_error"],
        "R1 B mapping tolerance",
    )

    legacy_convergence = _section(config, "convergence")
    axial = _section(diagnostics_r1, "refined_axial")
    existing_dz = _positive(
        axial["existing_reference_dz_m"], "R1 existing axial dz"
    )
    new_dz = _positive(axial["new_dz_m"], "R1 new axial dz")
    legacy_axial_dz = [
        float(value)
        for value in _section(legacy_convergence, "axial")["dz_cases_m"]
    ]
    if existing_dz not in legacy_axial_dz:
        msg = "R1 axial reference must be an existing R0 case."
        raise ValueError(msg)
    if new_dz in legacy_axial_dz:
        msg = "R1 axial new dz must not duplicate an R0 case."
        raise ValueError(msg)
    if not np.allclose(
        axial["acceptance_pair_m"], [existing_dz, new_dz], rtol=0.0, atol=1e-18
    ):
        msg = "R1 axial acceptance pair must be existing then new dz."
        raise ValueError(msg)

    lateral = _section(diagnostics_r1, "refined_lateral")
    lateral_existing = _section(lateral, "existing_reference")
    lateral_new = _section(lateral, "new_case")
    existing_shape = _shape(lateral_existing["shape"], "R1 lateral reference")
    existing_dx = _isotropic_dx(
        lateral_existing["dx_m"], "R1 lateral reference dx"
    )
    new_shape = _shape(lateral_new["shape"], "R1 lateral new shape")
    new_dx = _isotropic_dx(lateral_new["dx_m"], "R1 lateral new dx")
    if new_dx != base_dx:
        msg = "R1 lateral refinement must use the fine canonical-B sampling."
        raise ValueError(msg)
    fixed_fov = np.asarray(lateral["fixed_fov_m"], dtype=np.float64)
    if not np.allclose(
        np.asarray(existing_shape) * existing_dx, fixed_fov, rtol=1e-12, atol=0.0
    ) or not np.allclose(
        np.asarray(new_shape) * new_dx, fixed_fov, rtol=1e-12, atol=0.0
    ):
        msg = "R1 lateral cases must share the registered fixed FOV."
        raise ValueError(msg)
    if not np.allclose(
        lateral["acceptance_pair_dx_m"],
        [existing_dx, new_dx],
        rtol=0.0,
        atol=1e-18,
    ):
        msg = "R1 lateral acceptance pair must be existing then new dx."
        raise ValueError(msg)
    if _shape(lateral["comparison_grid_shape"], "R1 lateral comparison shape") != (
        existing_shape
    ) or _isotropic_dx(
        lateral["comparison_grid_dx_m"], "R1 lateral comparison dx"
    ) != existing_dx:
        msg = "R1 lateral comparison grid must be the existing reference grid."
        raise ValueError(msg)

    legacy_lateral_cases = _section(
        legacy_convergence, "lateral_fixed_fov"
    )["cases"]
    if not any(
        _shape(case["shape"], "legacy lateral shape") == existing_shape
        and _isotropic_dx(case["dx_m"], "legacy lateral dx") == existing_dx
        for case in legacy_lateral_cases
    ):
        msg = "R1 lateral reference must be an existing R0 case."
        raise ValueError(msg)
    if any(
        _shape(case["shape"], "legacy lateral shape") == new_shape
        and _isotropic_dx(case["dx_m"], "legacy lateral dx") == new_dx
        for case in legacy_lateral_cases
    ):
        msg = "R1 lateral new case must not duplicate an R0 case."
        raise ValueError(msg)

    fov = _section(diagnostics_r1, "refined_fov")
    legacy_fov = _section(legacy_convergence, "fov")
    existing_fov_shapes = [
        _shape(value, "R1 existing FOV shape") for value in fov["existing_shapes"]
    ]
    legacy_fov_shapes = [
        _shape(value, "legacy FOV shape") for value in legacy_fov["shapes"]
    ]
    new_fov_shapes = [
        _shape(value, "R1 new FOV shape") for value in fov["new_shapes"]
    ]
    if existing_fov_shapes != legacy_fov_shapes:
        msg = "R1 existing FOV shapes must exactly match R0."
        raise ValueError(msg)
    fov_dx = _isotropic_dx(fov["fixed_dx_m"], "R1 FOV dx")
    if fov_dx != _isotropic_dx(legacy_fov["fixed_dx_m"], "legacy FOV dx"):
        msg = "R1 FOV sampling must match R0."
        raise ValueError(msg)
    fov_pair = [
        _shape(value, "R1 FOV pair shape")
        for value in fov["acceptance_pair_shapes"]
    ]
    if fov_pair != new_fov_shapes[-2:]:
        msg = "R1 FOV acceptance pair must be the final two new shapes."
        raise ValueError(msg)
    if any(shape in legacy_fov_shapes for shape in new_fov_shapes):
        msg = "R1 new FOV shapes must not duplicate R0 cases."
        raise ValueError(msg)
    common_shape = _shape(fov["common_center_roi_shape"], "R1 FOV common ROI")
    if common_shape != _shape(
        legacy_fov["common_center_roi_shape"], "legacy FOV common ROI"
    ):
        msg = "R1 FOV common ROI must preserve R0."
        raise ValueError(msg)
    if np.any(
        np.asarray(new_fov_shapes[-1]) * fov_dx
        > np.asarray(working_shape) * working_dx
    ):
        msg = "R1 working canonical B must cover the largest FOV case."
        raise ValueError(msg)

    external = _section(diagnostics_r1, "external_padding")
    optics = _section(config, "optics")
    if _shape(external["source_shape"], "R1 external source") != _shape(
        optics["baseline_shape"], "baseline shape"
    ) or _isotropic_dx(external["fixed_dx_m"], "R1 external dx") != (
        _isotropic_dx(optics["baseline_dx_m"], "baseline dx")
    ):
        msg = "R1 external source grid must equal the R0 baseline grid."
        raise ValueError(msg)
    padded_shapes = [
        _shape(value, "R1 external padded shape")
        for value in external["padded_shapes"]
    ]
    if padded_shapes != existing_fov_shapes + new_fov_shapes:
        msg = "R1 external padded shapes must be the full R0+R1 FOV series."
        raise ValueError(msg)
    external_pair = [
        _shape(value, "R1 external pair")
        for value in external["acceptance_pair_shapes"]
    ]
    if external_pair != fov_pair:
        msg = "R1 external and full-chain FOV acceptance pairs must match."
        raise ValueError(msg)
    external_common_shape = _shape(
        external["common_center_roi_shape"], "R1 external common ROI"
    )
    source_shape = _shape(external["source_shape"], "R1 external source")
    if external_common_shape != source_shape:
        msg = "R1 external common ROI must equal the fixed source grid."
        raise ValueError(msg)
    if any(
        target_y < source_shape[0]
        or target_x < source_shape[1]
        or (target_y - source_shape[0]) % 2
        or (target_x - source_shape[1]) % 2
        for target_y, target_x in padded_shapes
    ):
        msg = "R1 external padded grids must be centered supersets of source."
        raise ValueError(msg)
    if external.get("direct_full_field_zero_padding_allowed") is not False:
        msg = "R1 forbids direct zero-padding of the full A-exit field."
        raise ValueError(msg)
    _positive(external["edge_ring_width_m"], "R1 edge ring width")
    _positive(
        external["require_a_exit_center_invariance_max"],
        "R1 A-exit center-invariance tolerance",
    )

    for sampling in (base_dx, new_dx, fov_dx):
        _require_integer_ratio(float(scan["step_m"]), sampling, "R1 scan step")
        _require_integer_ratio(
            float(scan["jitter_quantum_m"]), sampling, "R1 scan jitter quantum"
        )
    thresholds = _section(diagnostics_r1, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_acceptance_detector_visibility_signal_to_floor_min",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R1 must reuse all registered R0 acceptance thresholds."
        raise ValueError(msg)


def _validate_r2_config(
    config: Mapping[str, Any], diagnostics_r2: Mapping[str, Any]
) -> None:
    """Validate the append-only exp040-R2 diagnostic registration."""

    if diagnostics_r2.get("version") != "R2":
        msg = "diagnostics_r2.version must be R2."
        raise ValueError(msg)
    if diagnostics_r2.get("preserve_r0_r1_metrics_and_status") is not True:
        msg = "R2 must preserve R0/R1 metrics and status."
        raise ValueError(msg)
    diagnostics_r1 = config.get("diagnostics_r1")
    if not isinstance(diagnostics_r1, Mapping) or (
        diagnostics_r1.get("enabled") is not True
    ):
        msg = "R2 requires diagnostics_r1.enabled=true."
        raise ValueError(msg)

    methods = _section(diagnostics_r2, "methods")
    expected_methods = {
        "period_boundary": (
            "same_96um_canonical_period_centered_wrap_extension"
        ),
        "a_exit_padding": (
            "homogeneous_reference_plus_zero_padded_scattered_residual"
        ),
        "current_asm": "evanescent_only_same_grid_circular",
        "alias_controlled_asm": (
            "matsushima_exact_common_ellipse_same_grid"
        ),
        "comparison": "unaligned_common_center_roi_relative_l2",
    }
    if dict(methods) != expected_methods:
        msg = "diagnostics_r2 methods do not match the pre-registration."
        raise ValueError(msg)

    optics = _section(config, "optics")
    if optics.get("angular_spectrum_bandlimit") is not True:
        msg = "R2 alias control requires the propagating-wave bandlimit."
        raise ValueError(msg)
    baseline_shape = _shape(optics["baseline_shape"], "R2 baseline shape")
    baseline_dx = _isotropic_dx(optics["baseline_dx_m"], "R2 baseline dx")
    period_cfg = _section(diagnostics_r2, "period_commensurate")
    period_dx = _isotropic_dx(period_cfg["fixed_dx_m"], "R2 period dx")
    if period_dx != baseline_dx:
        msg = "R2 external dx must equal the registered baseline dx."
        raise ValueError(msg)
    shapes = [_shape(value, "R2 period shape") for value in period_cfg["shapes"]]
    fov_values = np.asarray(period_cfg["fov_m"], dtype=np.float64)
    period_counts = np.asarray(period_cfg["period_counts"], dtype=np.int64)
    if len(shapes) != 3 or fov_values.shape != (3,) or period_counts.shape != (3,):
        msg = "R2 must register exactly three period-commensurate cases."
        raise ValueError(msg)
    if not np.array_equal(period_counts, np.asarray([1, 2, 3])):
        msg = "R2 period counts must be [1, 2, 3]."
        raise ValueError(msg)
    if any(shape[0] != shape[1] for shape in shapes):
        msg = "R2 period-aligned grids must be square."
        raise ValueError(msg)
    widths = np.asarray([shape[1] * period_dx for shape in shapes])
    if not np.allclose(widths, fov_values, rtol=1e-12, atol=0.0):
        msg = "R2 shapes, dx, and FOV values are inconsistent."
        raise ValueError(msg)
    r1_refinement = _section(diagnostics_r1, "sample_b_refinement")
    base_fov = np.asarray(
        _section(r1_refinement, "base_grid")["fov_m"], dtype=np.float64
    )
    if base_fov.shape != (2,) or not np.allclose(
        fov_values,
        period_counts * base_fov[1],
        rtol=1e-12,
        atol=0.0,
    ):
        msg = "R2 FOV values must be integer multiples of the 96 um B period."
        raise ValueError(msg)
    base_period_shape = _shape(
        period_cfg["base_period_shape"], "R2 base-period shape"
    )
    if base_period_shape != shapes[0] or any(
        shape != (
            base_period_shape[0] * int(count),
            base_period_shape[1] * int(count),
        )
        for shape, count in zip(shapes, period_counts, strict=True)
    ):
        msg = "R2 shapes must contain exactly 1, 2, and 3 B periods."
        raise ValueError(msg)
    common_shape = _shape(
        period_cfg["common_center_roi_shape"], "R2 common ROI"
    )
    if common_shape != baseline_shape or any(
        shape[0] < common_shape[0] or shape[1] < common_shape[1]
        for shape in shapes
    ):
        msg = "R2 common ROI must equal and fit the baseline detector grid."
        raise ValueError(msg)
    pair = [
        _shape(value, "R2 acceptance shape")
        for value in period_cfg["acceptance_pair_shapes"]
    ]
    if pair != shapes[-2:]:
        msg = "R2 acceptance pair must be the 192 to 288 um cases."
        raise ValueError(msg)
    _positive(
        period_cfg["require_a_exit_center_invariance_max"],
        "R2 A-exit invariance tolerance",
    )
    _positive(
        period_cfg["canonical_b_mapping_max_complex_error"],
        "R2 canonical-B mapping tolerance",
    )

    alias_cfg = _section(diagnostics_r2, "alias_control")
    if list(alias_cfg.get("apply_to_stages", [])) != ["AB", "BC"]:
        msg = "R2 alias control must apply only to external AB and BC."
        raise ValueError(msg)
    required_alias_flags = {
        "exact_common_ellipse": True,
        "same_periodic_grid": True,
        "rectangular_approximation": False,
        "linear_convolution_padding": False,
        "apply_inside_sample_a": False,
    }
    if any(
        alias_cfg.get(key) is not value
        for key, value in required_alias_flags.items()
    ):
        msg = "R2 alias-control flags do not match the theory registration."
        raise ValueError(msg)
    determinism = _section(diagnostics_r2, "determinism")
    if determinism.get("repeat_largest_alias_controlled_case") is not True:
        msg = "R2 must repeat the largest alias-controlled case."
        raise ValueError(msg)

    scan = _section(config, "scan")
    _require_integer_ratio(float(scan["step_m"]), period_dx, "R2 scan step")
    _require_integer_ratio(
        float(scan["jitter_quantum_m"]), period_dx, "R2 scan jitter quantum"
    )
    thresholds = _section(diagnostics_r2, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_r1_a_exit_center_invariance_max",
            "reuse_r1_canonical_b_mapping_max_complex_error",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R2 must reuse all pre-registered thresholds."
        raise ValueError(msg)
    output = _section(config, "output")
    if output.get("save_r2_figures") is not True:
        msg = "R2 output must enable the two pre-registered figures."
        raise ValueError(msg)


def _validate_r3_config(
    config: Mapping[str, Any], diagnostics_r3: Mapping[str, Any]
) -> None:
    """Validate the append-only exp040-R3 detector-path registration."""

    if diagnostics_r3.get("version") != "R3":
        msg = "diagnostics_r3.version must be R3."
        raise ValueError(msg)
    if diagnostics_r3.get("preserve_r0_r1_r2_metrics_and_status") is not True:
        msg = "R3 must preserve R0/R1/R2 metrics and status."
        raise ValueError(msg)
    for stage in ("diagnostics_r1", "diagnostics_r2"):
        value = config.get(stage)
        if not isinstance(value, Mapping) or value.get("enabled") is not True:
            msg = f"R3 requires {stage}.enabled=true."
            raise ValueError(msg)

    expected_methods = {
        "external_fov": "two_period_192um_periodic_grid",
        "a_exit_mapping": "aligned_bilinear_residual_plus_homogeneous_reference",
        "canonical_b": "same_phase_cells_piecewise_constant_refinement",
        "ab_propagation": "matsushima_exact_common_ellipse_same_grid",
        "bc_current_asm": "evanescent_only_same_grid_circular",
        "bc_alias_controlled_asm": "matsushima_exact_common_ellipse_same_grid",
        "point_detector": "aligned_native_center_point_sample",
        "pixel_detector": "periodic_square_pixel_sinc_mtf_area_average",
        "comparison": "unaligned_native_roi_relative_l2",
    }
    if dict(_section(diagnostics_r3, "methods")) != expected_methods:
        msg = "diagnostics_r3 methods do not match the pre-registration."
        raise ValueError(msg)

    optics = _section(config, "optics")
    baseline_shape = _shape(optics["baseline_shape"], "R3 baseline shape")
    baseline_dx = _isotropic_dx(optics["baseline_dx_m"], "R3 baseline dx")
    detector = _section(optics, "detector")
    detector_pixel = _positive(
        detector["pixel_size_m"], "R3 detector pixel size"
    )
    if detector_pixel != baseline_dx:
        msg = "R3 native detector pixel must equal the baseline dx."
        raise ValueError(msg)

    sampling = _section(diagnostics_r3, "sampling")
    factors = np.asarray(sampling["factors"], dtype=np.int64)
    if not np.array_equal(factors, np.asarray([1, 2, 4], dtype=np.int64)):
        msg = "R3 sampling factors must be [1, 2, 4]."
        raise ValueError(msg)
    dx_values = np.asarray(sampling["dx_m"], dtype=np.float64)
    if dx_values.shape != (3,) or not np.allclose(
        dx_values, baseline_dx / factors, rtol=1e-12, atol=0.0
    ):
        msg = "R3 dx values must be baseline_dx / [1, 2, 4]."
        raise ValueError(msg)
    shapes = [_shape(value, "R3 sampling shape") for value in sampling["shapes"]]
    native_full = _shape(sampling["native_full_shape"], "R3 native full shape")
    native_roi = _shape(sampling["native_roi_shape"], "R3 native ROI")
    if native_roi != baseline_shape:
        msg = "R3 native ROI must equal the baseline detector shape."
        raise ValueError(msg)
    if len(shapes) != 3 or any(
        shape != (native_full[0] * int(factor), native_full[1] * int(factor))
        for shape, factor in zip(shapes, factors, strict=True)
    ):
        msg = "R3 shapes must be native_full_shape multiplied by each factor."
        raise ValueError(msg)
    fov = np.asarray(sampling["external_fov_m"], dtype=np.float64)
    if fov.shape != (2,) or not all(
        np.allclose(
            np.asarray(shape, dtype=np.float64) * dx_m,
            fov,
            rtol=1e-12,
            atol=0.0,
        )
        for shape, dx_m in zip(shapes, dx_values, strict=True)
    ):
        msg = "R3 shapes and dx must keep the registered 192 um external FOV."
        raise ValueError(msg)
    r2_period = _section(_section(config, "diagnostics_r2"), "period_commensurate")
    r2_shapes = [_shape(value, "R2 shape") for value in r2_period["shapes"]]
    if native_full != r2_shapes[1] or int(sampling["canonical_period_count"]) != 2:
        msg = "R3 native full grid must equal the R2 two-period grid."
        raise ValueError(msg)
    pair = [int(value) for value in sampling["acceptance_pair_factors"]]
    if pair != [2, 4]:
        msg = "R3 acceptance pair must be factors [2, 4]."
        raise ValueError(msg)
    offsets = np.asarray(sampling["native_sample_offsets_px"], dtype=np.int64)
    expected_offsets = (factors - 1) // 2
    if not np.array_equal(offsets, expected_offsets):
        msg = "R3 native sample offsets do not match the registered origin rule."
        raise ValueError(msg)
    compensation = np.asarray(
        sampling["physical_origin_compensation_m"], dtype=np.float64
    )
    expected_compensation = (
        (factors.astype(np.float64) - 1.0) / 2.0 - expected_offsets
    ) * dx_values
    if compensation.shape != (3,) or not np.allclose(
        compensation, expected_compensation, rtol=1e-12, atol=1e-20
    ):
        msg = "R3 physical-origin compensation does not match R3.8."
        raise ValueError(msg)

    propagation = _section(diagnostics_r3, "propagation")
    required_propagation = {
        "ab_alias_control": True,
        "bc_methods": ["current_asm", "alias_controlled"],
        "no_apodization_or_window": True,
        "stream_scans": True,
        "retain_full_detector_stack": False,
    }
    if dict(propagation) != required_propagation:
        msg = "R3 propagation flags do not match the pre-registration."
        raise ValueError(msg)
    detector_sampling = _section(diagnostics_r3, "detector_sampling")
    required_detector = {
        "branches": ["point_sample", "pixel_box_average"],
        "primary_bc_method": "alias_controlled",
        "primary_detector_branch": "pixel_box_average",
        "pixel_response": "ideal_square_sinc_mtf",
        "selected_factor": 4,
        "selected_scan_index": 0,
    }
    if dict(detector_sampling) != required_detector:
        msg = "R3 detector branches do not match the pre-registration."
        raise ValueError(msg)
    if int(detector_sampling["selected_scan_index"]) >= (
        int(_section(config, "scan")["num_x"])
        * int(_section(config, "scan")["num_y"])
    ):
        msg = "R3 selected detector scan index is out of range."
        raise ValueError(msg)

    scan = _section(config, "scan")
    for dx_m in dx_values:
        _require_integer_ratio(float(scan["step_m"]), dx_m, "R3 scan step")
        _require_integer_ratio(
            float(scan["jitter_quantum_m"]), dx_m, "R3 scan jitter quantum"
        )
        _require_integer_ratio(
            float(_section(config, "sample_b")["physical_feature_size_m"]),
            dx_m,
            "R3 B feature size",
        )

    determinism = _section(diagnostics_r3, "determinism")
    if dict(determinism) != {
        "repeat_selected_factor_primary_scan": True,
    }:
        msg = "R3 determinism control does not match the pre-registration."
        raise ValueError(msg)
    thresholds = _section(diagnostics_r3, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_acceptance_algebra_relative_l2_max_for_mapping_and_pixel",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R3 must reuse all pre-registered thresholds."
        raise ValueError(msg)
    if _section(config, "output").get("save_r3_figures") is not True:
        msg = "R3 output must enable the three pre-registered figures."
        raise ValueError(msg)


def _validate_r4_config(
    config: Mapping[str, Any], diagnostics_r4: Mapping[str, Any]
) -> None:
    """Validate the exp040-R4 positive quadrature registration."""

    if diagnostics_r4.get("version") != "R4":
        msg = "diagnostics_r4.version must be R4."
        raise ValueError(msg)
    for stage in ("diagnostics_r1", "diagnostics_r2", "diagnostics_r3"):
        value = config.get(stage)
        if not isinstance(value, Mapping) or value.get("enabled") is not False:
            msg = f"R4 requires {stage}.enabled=false to avoid recomputation."
            raise ValueError(msg)
    expected_methods = {
        "quadrature": "staggered_midpoint_uniform_positive_weights",
        "a_exit_mapping": "centered_bilinear_residual_plus_homogeneous_reference",
        "canonical_b": "same_48x48_phase_cells_piecewise_constant_nodes",
        "ab_propagation": "matsushima_exact_common_ellipse_same_grid",
        "bc_propagation": "matsushima_exact_common_ellipse_same_grid",
        "comparison": "unaligned_native_128_roi_relative_l2",
    }
    if dict(_section(diagnostics_r4, "methods")) != expected_methods:
        msg = "diagnostics_r4 methods do not match the pre-registration."
        raise ValueError(msg)
    provenance = _section(diagnostics_r4, "r3_provenance")
    expected_provenance = {
        "run": (
            "runs/exp040_TGV_3d_multislice_r3_detector_path_"
            "20260811_153852"
        ),
        "config_sha256": (
            "4B17ADD64B0633540322EA416B4C9E23BB7720A85774A8F48BA7CA95A085B4B6"
        ),
        "metrics_sha256": (
            "85BE9131F9797B6837DCFD341834F6876633403B624D1CEB4B080183E2ACF9BE"
        ),
        "status": "Failed",
        "alias_pixel_factor2_to4_relative_l2": 0.022648734882925837,
        "pixel_mtf_negative_relative_scale": 0.001089144469713379,
    }
    if dict(provenance) != expected_provenance:
        msg = "R4 R3 provenance does not match the frozen formal run."
        raise ValueError(msg)

    optics = _section(config, "optics")
    baseline_shape = _shape(optics["baseline_shape"], "R4 baseline shape")
    baseline_dx = _isotropic_dx(optics["baseline_dx_m"], "R4 baseline dx")
    sampling = _section(diagnostics_r4, "sampling")
    factors = np.asarray(sampling["factors"], dtype=np.int64)
    if not np.array_equal(factors, np.asarray([2, 4, 8])):
        msg = "R4 quadrature factors must be [2, 4, 8]."
        raise ValueError(msg)
    dx_values = np.asarray(sampling["node_dx_m"], dtype=np.float64)
    if not np.allclose(dx_values, baseline_dx / factors, rtol=1e-12, atol=0.0):
        msg = "R4 node dx values must equal pixel pitch / factor."
        raise ValueError(msg)
    native_full = _shape(sampling["native_full_shape"], "R4 native full shape")
    native_roi = _shape(sampling["native_roi_shape"], "R4 native ROI")
    if native_roi != baseline_shape:
        msg = "R4 native ROI must equal the baseline detector shape."
        raise ValueError(msg)
    shapes = [_shape(value, "R4 node shape") for value in sampling["node_shapes"]]
    if any(
        shape != (native_full[0] * int(q), native_full[1] * int(q))
        for shape, q in zip(shapes, factors, strict=True)
    ):
        msg = "R4 node shapes must equal native_full_shape times q."
        raise ValueError(msg)
    fov = np.asarray(sampling["external_fov_m"], dtype=np.float64)
    if fov.shape != (2,) or not all(
        np.allclose(np.asarray(shape) * dx, fov, rtol=1e-12, atol=0.0)
        for shape, dx in zip(shapes, dx_values, strict=True)
    ):
        msg = "R4 node grids must keep the registered 192 um FOV."
        raise ValueError(msg)
    r2_period = _section(_section(config, "diagnostics_r2"), "period_commensurate")
    if native_full != _shape(r2_period["shapes"][1], "R4 R2 comparator shape"):
        msg = "R4 native full grid must be the R2 two-period grid."
        raise ValueError(msg)
    if [int(value) for value in sampling["acceptance_pair_factors"]] != [4, 8]:
        msg = "R4 acceptance pair must be q=4 to q=8."
        raise ValueError(msg)

    if dict(_section(diagnostics_r4, "propagation")) != {
        "ab_alias_control": True,
        "bc_alias_control": True,
        "stream_scans": True,
        "retain_full_node_stacks": False,
    }:
        msg = "R4 propagation flags do not match the pre-registration."
        raise ValueError(msg)
    if dict(_section(diagnostics_r4, "quadrature")) != {
        "node_rule": "pixel_center_plus_a_half_over_q",
        "weights": "uniform_nonnegative",
        "block_average_axes": ["subpixel_y", "subpixel_x"],
    }:
        msg = "R4 quadrature flags do not match the pre-registration."
        raise ValueError(msg)
    if dict(_section(diagnostics_r4, "determinism")) != {
        "selected_factor": 8,
        "selected_scan_index": 0,
    }:
        msg = "R4 determinism control must use q8 scan 0."
        raise ValueError(msg)
    thresholds = _section(diagnostics_r4, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_acceptance_algebra_relative_l2_max",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R4 must reuse all registered thresholds."
        raise ValueError(msg)
    scan = _section(config, "scan")
    feature = float(_section(config, "sample_b")["physical_feature_size_m"])
    for dx in dx_values:
        _require_integer_ratio(float(scan["step_m"]), dx, "R4 scan step")
        _require_integer_ratio(float(scan["jitter_quantum_m"]), dx, "R4 jitter")
        _require_integer_ratio(feature, dx, "R4 B feature size")
    output = _section(config, "output")
    if output.get("save_r4_figures") is not True:
        msg = "R4 output must enable the two pre-registered figures."
        raise ValueError(msg)
    if any(
        output.get(name) is not False
        for name in ("save_r1_figures", "save_r2_figures", "save_r3_figures")
    ):
        msg = "R4 must not duplicate prior diagnostic figures."
        raise ValueError(msg)


def _validate_r5_config(
    config: Mapping[str, Any], diagnostics_r5: Mapping[str, Any]
) -> None:
    """Validate the exp040-R5 finite-support/open-boundary registration."""

    if diagnostics_r5.get("version") != "R5":
        msg = "diagnostics_r5.version must be R5."
        raise ValueError(msg)
    for stage in (
        "diagnostics_r1",
        "diagnostics_r2",
        "diagnostics_r3",
        "diagnostics_r4",
    ):
        value = config.get(stage)
        if not isinstance(value, Mapping) or value.get("enabled") is not False:
            msg = f"R5 requires {stage}.enabled=false to avoid recomputation."
            raise ValueError(msg)
    expected_methods = {
        "finite_b_support": "same_96um_phase_cells_transparent_exterior",
        "finite_b_shift": "constant_zero_shift_of_b_minus_one",
        "probe_decomposition": "homogeneous_background_plus_fixed_r4_q4_residual",
        "open_bc": "zero_padded_residual_alias_controlled_asm",
        "quadrature": "r4_q4_staggered_midpoint_uniform_positive_weights",
        "comparison": "unaligned_native_128_roi_relative_l2_registered_denominators",
    }
    if dict(_section(diagnostics_r5, "methods")) != expected_methods:
        msg = "diagnostics_r5 methods do not match the pre-registration."
        raise ValueError(msg)
    expected_provenance = {
        "run": (
            "runs/exp040_TGV_3d_multislice_r4_positive_quadrature_"
            "20260811_161412"
        ),
        "config_sha256": (
            "C9628F9D12663CBCA1FCC0BA3533A14313086E1326076329DB5BB62919631D7E"
        ),
        "metrics_sha256": (
            "F2093962EFF1C369E45145A6C61E0C600D5B6DB55E8794FE2D4F65893C09467C"
        ),
        "status": "Passed",
        "q4_to_q8_P_B_relative_l2": 4.455618330995762e-06,
        "q4_to_q8_I_stack_relative_l2": 0.00039699933862833467,
    }
    if dict(_section(diagnostics_r5, "r4_provenance")) != expected_provenance:
        msg = "R5 R4 provenance does not match the frozen formal run."
        raise ValueError(msg)

    optics = _section(config, "optics")
    detector_pixel = float(_section(optics, "detector")["pixel_size_m"])
    baseline_shape = _shape(optics["baseline_shape"], "R5 baseline shape")
    sampling = _section(diagnostics_r5, "sampling")
    factor = int(sampling["quadrature_factor"])
    if factor != 4:
        msg = "R5 quadrature factor must be q=4."
        raise ValueError(msg)
    node_dx = float(sampling["node_dx_m"])
    if not np.isclose(node_dx, detector_pixel / factor, rtol=1e-12, atol=0.0):
        msg = "R5 node dx must equal detector pixel pitch / 4."
        raise ValueError(msg)
    native_roi = _shape(sampling["native_roi_shape"], "R5 native ROI")
    if native_roi != baseline_shape:
        msg = "R5 native ROI must equal the baseline detector shape."
        raise ValueError(msg)
    base_shape = _shape(sampling["base_node_shape"], "R5 base node shape")
    base_fov = np.asarray(sampling["base_fov_m"], dtype=np.float64)
    if base_fov.shape != (2,) or not np.allclose(
        np.asarray(base_shape) * node_dx, base_fov, rtol=1e-12, atol=0.0
    ):
        msg = "R5 base node grid and FOV are inconsistent."
        raise ValueError(msg)
    if not np.allclose(base_fov, base_fov[0], rtol=1e-12, atol=0.0):
        msg = "R5 base FOV must be square."
        raise ValueError(msg)
    padding_fov = np.asarray(sampling["padding_fov_m"], dtype=np.float64)
    padding_shapes = [
        _shape(value, "R5 padding node shape")
        for value in sampling["padding_node_shapes"]
    ]
    if padding_fov.shape != (3,) or len(padding_shapes) != 3:
        msg = "R5 padding series must contain three cases."
        raise ValueError(msg)
    if not np.allclose(
        padding_fov,
        np.asarray([base_fov[0], 1.5 * base_fov[0], 2.0 * base_fov[0]]),
        rtol=1e-12,
        atol=0.0,
    ):
        msg = "R5 padding FOVs must be the registered 2/3/4-period series."
        raise ValueError(msg)
    if any(
        shape != (_require_integer_ratio(fov, node_dx, "R5 padding FOV"),) * 2
        for fov, shape in zip(padding_fov, padding_shapes, strict=True)
    ):
        msg = "R5 padding shapes disagree with FOV/node sampling."
        raise ValueError(msg)
    if padding_shapes[0] != base_shape:
        msg = "R5 first padding grid must equal the base grid."
        raise ValueError(msg)
    if not np.allclose(
        np.asarray(sampling["acceptance_pair_fov_m"], dtype=np.float64),
        padding_fov[-2:],
        rtol=1e-12,
        atol=0.0,
    ):
        msg = "R5 acceptance pair must be the final two padding FOVs."
        raise ValueError(msg)
    ring_width = float(sampling["boundary_ring_width_m"])
    ring_pixels = _require_integer_ratio(
        ring_width, node_dx, "R5 boundary ring width"
    )
    if 2 * ring_pixels >= min(base_shape):
        msg = "R5 boundary ring must leave a nonempty interior."
        raise ValueError(msg)

    support = _section(diagnostics_r5, "finite_support")
    support_shape_m = np.asarray(support["physical_shape_m"], dtype=np.float64)
    cells = np.asarray(support["canonical_phase_cells"], dtype=np.int64)
    feature_m = float(_section(config, "sample_b")["physical_feature_size_m"])
    if (
        support_shape_m.shape != (2,)
        or cells.shape != (2,)
        or not np.array_equal(cells, np.asarray([48, 48]))
        or not np.allclose(
            support_shape_m, cells * feature_m, rtol=1e-12, atol=0.0
        )
    ):
        msg = "R5 finite B must contain the same 48x48 physical phase cells."
        raise ValueError(msg)
    if not np.allclose(
        base_fov, 2.0 * support_shape_m, rtol=1e-12, atol=0.0
    ):
        msg = "R5 base FOV must be two finite-B support widths."
        raise ValueError(msg)
    if dict(support) != {
        "physical_shape_m": list(support["physical_shape_m"]),
        "canonical_phase_cells": [48, 48],
        "exterior_transmission_real": 1.0,
        "exterior_transmission_imag": 0.0,
        "modulation_shift_boundary": "constant_zero",
    }:
        msg = "R5 finite-support boundary flags are invalid."
        raise ValueError(msg)
    positions = _make_common_scan(_section(config, "scan"))
    max_scan_xy = np.max(np.abs(positions), axis=0)
    support_half_xy = support_shape_m[::-1] / 2.0
    base_half_xy = base_fov[::-1] / 2.0
    if np.any(support_half_xy + max_scan_xy >= base_half_xy):
        msg = "R5 shifted finite B support must remain inside the base FOV."
        raise ValueError(msg)
    _validate_scan_on_grid(positions, node_dx)

    expected_propagation = {
        "ab_probe_source": "fixed_r4_q4_192um",
        "bc_alias_control": True,
        "background": "full_grid_homogeneous_plane",
        "residual_padding": "centered_zero",
        "stream_scans": True,
        "retain_full_node_stacks": False,
    }
    if dict(_section(diagnostics_r5, "propagation")) != expected_propagation:
        msg = "R5 propagation flags do not match the pre-registration."
        raise ValueError(msg)
    expected_comparisons = {
        "support": ["periodic_circular_192", "finite_circular_192"],
        "boundary": ["finite_circular_192", "finite_open_384"],
        "combined": ["periodic_circular_192", "finite_open_384"],
        "denominator": "second_branch",
    }
    if dict(_section(diagnostics_r5, "comparisons")) != expected_comparisons:
        msg = "R5 comparison branches do not match the pre-registration."
        raise ValueError(msg)
    determinism = _section(diagnostics_r5, "determinism")
    if (
        float(determinism.get("selected_padding_fov_m", -1.0))
        != float(padding_fov[-1])
        or int(determinism.get("selected_scan_index", -1)) != 0
    ):
        msg = "R5 determinism control must use 384 um scan 0."
        raise ValueError(msg)
    thresholds = _section(diagnostics_r5, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_acceptance_algebra_relative_l2_max",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R5 must reuse all registered thresholds."
        raise ValueError(msg)
    output = _section(config, "output")
    if output.get("save_r5_figures") is not True:
        msg = "R5 output must enable the three pre-registered figures."
        raise ValueError(msg)
    if any(
        output.get(name) is not False
        for name in (
            "save_r1_figures",
            "save_r2_figures",
            "save_r3_figures",
            "save_r4_figures",
        )
    ):
        msg = "R5 must not duplicate prior diagnostic figures."
        raise ValueError(msg)


def _validate_r6_config(
    config: Mapping[str, Any], diagnostics_r6: Mapping[str, Any]
) -> None:
    """Validate the exp040-R6 virtual sample-B support envelope."""

    if diagnostics_r6.get("version") != "R6":
        msg = "diagnostics_r6.version must be R6."
        raise ValueError(msg)
    for stage in (
        "diagnostics_r1",
        "diagnostics_r2",
        "diagnostics_r3",
        "diagnostics_r4",
        "diagnostics_r5",
    ):
        value = config.get(stage)
        if not isinstance(value, Mapping) or value.get("enabled") is not False:
            msg = f"R6 requires {stage}.enabled=false to avoid recomputation."
            raise ValueError(msg)
    expected_methods = {
        "support_family": "same_canonical_cells_center_crop_or_periodic_extend",
        "edge_taper": "separable_raised_cosine_phase_only_unit_modulus",
        "finite_shift": "constant_zero_shift_of_b_minus_one",
        "bc_propagation": "r5_circular_alias_controlled_192um",
        "quadrature": "r4_q4_staggered_midpoint_uniform_positive_weights",
        "comparison": "unaligned_native_128_roi_relative_l2_registered_denominators",
    }
    if dict(_section(diagnostics_r6, "methods")) != expected_methods:
        msg = "diagnostics_r6 methods do not match the pre-registration."
        raise ValueError(msg)
    expected_provenance = {
        "run": (
            "runs/exp040_TGV_3d_multislice_r5_finite_support_open_boundary_"
            "20260811_163555"
        ),
        "config_sha256": (
            "A0EC579CDEB7BDB474CC3174A61FE3D3CAC8188329902E79BEF104C0F8C5249B"
        ),
        "metrics_sha256": (
            "41AA3522B146EF4063EA65DCFEB707E25E01B17688EBF3036680DE9205A85C28"
        ),
        "status": "Passed",
        "support_relative_l2": 0.38114505695745043,
        "boundary_relative_l2": 0.03266165515687117,
    }
    if dict(_section(diagnostics_r6, "r5_provenance")) != expected_provenance:
        msg = "R6 R5 provenance does not match the frozen formal run."
        raise ValueError(msg)

    optics = _section(config, "optics")
    detector_pixel = float(_section(optics, "detector")["pixel_size_m"])
    baseline_shape = _shape(optics["baseline_shape"], "R6 baseline shape")
    sampling = _section(diagnostics_r6, "sampling")
    factor = int(sampling["quadrature_factor"])
    if factor != 4:
        msg = "R6 quadrature factor must be q=4."
        raise ValueError(msg)
    node_dx = float(sampling["node_dx_m"])
    if not np.isclose(node_dx, detector_pixel / factor, rtol=1e-12, atol=0.0):
        msg = "R6 node dx must equal detector pixel pitch / 4."
        raise ValueError(msg)
    base_shape = _shape(sampling["node_shape"], "R6 node shape")
    base_fov = np.asarray(sampling["fov_m"], dtype=np.float64)
    if base_fov.shape != (2,) or not np.allclose(
        np.asarray(base_shape) * node_dx,
        base_fov,
        rtol=1e-12,
        atol=0.0,
    ):
        msg = "R6 node shape/FOV are inconsistent."
        raise ValueError(msg)
    if not np.allclose(base_fov, base_fov[0], rtol=1e-12, atol=0.0):
        msg = "R6 node FOV must be square."
        raise ValueError(msg)
    native_roi = _shape(sampling["native_roi_shape"], "R6 native ROI")
    if native_roi != baseline_shape:
        msg = "R6 native ROI must equal the baseline detector shape."
        raise ValueError(msg)

    family = _section(diagnostics_r6, "support_family")
    widths = np.asarray(family["support_width_m"], dtype=np.float64)
    tapers = np.asarray(family["edge_taper_width_m"], dtype=np.float64)
    nominal_width = float(family["nominal_support_width_m"])
    nominal_taper = float(family["nominal_edge_taper_width_m"])
    if widths.shape != (3,) or not np.allclose(
        widths / nominal_width,
        np.asarray([5.0 / 6.0, 1.0, 7.0 / 6.0]),
        rtol=1e-12,
        atol=0.0,
    ):
        msg = "R6 support widths must be the registered 80/96/112 envelope."
        raise ValueError(msg)
    if tapers.shape != (3,) or not np.allclose(
        tapers / nominal_width,
        np.asarray([0.0, 1.0 / 24.0, 1.0 / 12.0]),
        rtol=1e-12,
        atol=0.0,
    ):
        msg = "R6 taper widths must be the registered 0/4/8 envelope."
        raise ValueError(msg)
    feature_m = float(_section(config, "sample_b")["physical_feature_size_m"])
    if not np.isclose(nominal_width, 48.0 * feature_m, rtol=1e-12, atol=0.0):
        msg = "R6 nominal support must contain the same 48 phase cells."
        raise ValueError(msg)
    if not np.isclose(base_fov[0], 2.0 * nominal_width, rtol=1e-12, atol=0.0):
        msg = "R6 FOV must be twice the nominal support width."
        raise ValueError(msg)
    for width in widths:
        _require_integer_ratio(float(width), node_dx, "R6 support width")
        _require_integer_ratio(float(width), feature_m, "R6 support cells")
    for taper in tapers[1:]:
        _require_integer_ratio(float(taper), node_dx, "R6 taper width")
        _require_integer_ratio(float(taper), feature_m, "R6 taper cells")
    expected_family = {
        "support_width_m": list(family["support_width_m"]),
        "edge_taper_width_m": list(family["edge_taper_width_m"]),
        "nominal_support_width_m": nominal_width,
        "nominal_edge_taper_width_m": 0.0,
        "exterior_transmission_real": 1.0,
        "exterior_transmission_imag": 0.0,
        "case_order": "width_major_full_factorial",
    }
    if dict(family) != expected_family or nominal_taper != 0.0:
        msg = "R6 support-family boundary flags are invalid."
        raise ValueError(msg)
    positions = _make_common_scan(_section(config, "scan"))
    max_scan_xy = np.max(np.abs(positions), axis=0)
    if np.any(widths[-1] / 2.0 + max_scan_xy >= base_fov[::-1] / 2.0):
        msg = "R6 largest shifted support must remain inside the node FOV."
        raise ValueError(msg)
    _validate_scan_on_grid(positions, node_dx)

    if dict(_section(diagnostics_r6, "propagation")) != {
        "ab_probe_source": "fixed_r4_q4_192um",
        "bc_alias_control": True,
        "stream_scans": True,
        "retain_full_node_stacks": False,
    }:
        msg = "R6 propagation flags do not match the pre-registration."
        raise ValueError(msg)
    if dict(_section(diagnostics_r6, "comparisons")) != {
        "support_effect": ["periodic_circular", "finite_case_circular"],
        "support_effect_denominator": "finite_case",
        "nominal_sensitivity": ["finite_case", "nominal_96um_hard"],
        "nominal_sensitivity_denominator": "nominal",
    }:
        msg = "R6 comparisons do not match the pre-registration."
        raise ValueError(msg)
    determinism = _section(diagnostics_r6, "determinism")
    if (
        float(determinism.get("support_width_m", -1.0)) != nominal_width
        or float(determinism.get("edge_taper_width_m", -1.0)) != 0.0
        or int(determinism.get("scan_index", -1)) != 0
    ):
        msg = "R6 determinism must use nominal hard-edge scan 0."
        raise ValueError(msg)
    thresholds = _section(diagnostics_r6, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max_as_materiality",
            "reuse_acceptance_algebra_relative_l2_max",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R6 must reuse all registered thresholds."
        raise ValueError(msg)
    output = _section(config, "output")
    if output.get("save_r6_figures") is not True:
        msg = "R6 output must enable the three pre-registered figures."
        raise ValueError(msg)
    if any(
        output.get(name) is not False
        for name in (
            "save_r1_figures",
            "save_r2_figures",
            "save_r3_figures",
            "save_r4_figures",
            "save_r5_figures",
        )
    ):
        msg = "R6 must not duplicate prior diagnostic figures."
        raise ValueError(msg)


def _validate_r7_config(
    config: Mapping[str, Any], diagnostics_r7: Mapping[str, Any]
) -> None:
    """Validate the frozen exp040-R7 subvoxel-interface diagnostic."""

    if diagnostics_r7.get("version") != "R7":
        msg = "diagnostics_r7.version must be R7."
        raise ValueError(msg)
    for stage in (
        "diagnostics_r1",
        "diagnostics_r2",
        "diagnostics_r3",
        "diagnostics_r4",
        "diagnostics_r5",
        "diagnostics_r6",
    ):
        value = config.get(stage)
        if not isinstance(value, Mapping) or value.get("enabled") is not False:
            msg = f"R7 requires {stage}.enabled=false to avoid recomputation."
            raise ValueError(msg)
    expected_methods = {
        "interface": "staggered_midpoint_air_area_fraction",
        "effective_index": "linear_cell_average_of_indicator",
        "multislice": "centered_symmetric_split_step_streamed_slices",
        "finite_b": "same_nominal_96um_hard_edge_transparent_exterior",
        "finite_shift": "constant_zero_shift_of_b_minus_one",
        "open_bc": "r5_zero_padded_residual_alias_controlled_384um",
        "detector": "r4_q4_staggered_midpoint_uniform_positive_weights",
        "comparison": "unaligned_relative_l2_q8_denominator",
    }
    if dict(_section(diagnostics_r7, "methods")) != expected_methods:
        msg = "diagnostics_r7 methods do not match the pre-registration."
        raise ValueError(msg)
    expected_provenance = {
        "run": (
            "runs/exp040_TGV_3d_multislice_r6_b_support_sensitivity_"
            "20260811_200120"
        ),
        "metrics_sha256": (
            "5813692089C374892D961152250225F71CE05843828A5F8D9FBE5CEBA33B987A"
        ),
        "status": "Passed",
        "maximum_nominal_b_variation": 0.17129597874704286,
    }
    if dict(_section(diagnostics_r7, "r6_provenance")) != expected_provenance:
        msg = "R7 R6 provenance does not match the frozen formal run."
        raise ValueError(msg)

    interface = _section(diagnostics_r7, "interface")
    if dict(interface) != {
        "factors": [1, 2, 4, 8],
        "acceptance_pair_factors": [4, 8],
        "binary_factor": 1,
        "reference_factor": 8,
        "node_rule": "pixel_center_plus_a_half_over_q",
        "weights": "uniform_nonnegative",
        "axial_subnodes": 1,
    }:
        msg = "R7 interface cases do not match the pre-registration."
        raise ValueError(msg)
    sampling = _section(diagnostics_r7, "sample_a_sampling")
    shape = _shape(sampling["shape"], "R7 sample-A shape")
    dx_m = _isotropic_dx(sampling["dx_m"], "R7 sample-A dx")
    fov = np.asarray(sampling["fov_m"], dtype=np.float64)
    dz_m = _positive(sampling["dz_m"], "R7 sample-A dz")
    baseline_dx = _isotropic_dx(
        _section(config, "optics")["baseline_dx_m"], "R7 baseline dx"
    )
    registered_fov = np.asarray(
        _section(_section(config, "convergence"), "lateral_fixed_fov")["fov_m"],
        dtype=np.float64,
    )
    if (
        fov.shape != (2,)
        or not np.allclose(np.asarray(shape) * dx_m, fov, rtol=1e-12, atol=0.0)
        or not np.allclose(fov, registered_fov, rtol=1e-12, atol=0.0)
        or not np.isclose(dx_m, baseline_dx / 2.0, rtol=1e-12, atol=0.0)
        or sampling.get("stream_slices") is not True
        or sampling.get("retain_full_volumes") is not False
    ):
        msg = "R7 sample-A sampling does not match the frozen half-pitch grid."
        raise ValueError(msg)
    sample_a = _section(config, "sample_a")
    _, widths = midpoint_z_grid(float(sample_a["thickness_m"]), dz_m)
    if not np.isclose(float(np.sum(widths)), float(sample_a["thickness_m"])):
        msg = "R7 slice widths must span the exact sample-A thickness."
        raise ValueError(msg)

    detector = _section(diagnostics_r7, "detector_sampling")
    factor = int(detector["quadrature_factor"])
    node_dx = float(detector["node_dx_m"])
    pixel_m = float(_section(_section(config, "optics"), "detector")["pixel_size_m"])
    base_shape = _shape(detector["base_node_shape"], "R7 base node shape")
    open_shape = _shape(detector["open_node_shape"], "R7 open node shape")
    base_fov = np.asarray(detector["base_fov_m"], dtype=np.float64)
    open_fov = np.asarray(detector["open_fov_m"], dtype=np.float64)
    native_roi = _shape(detector["native_roi_shape"], "R7 native ROI")
    baseline_shape = _shape(_section(config, "optics")["baseline_shape"], "baseline")
    if (
        factor != 4
        or not np.isclose(node_dx, pixel_m / factor, rtol=1e-12, atol=0.0)
        or not np.allclose(
            np.asarray(base_shape) * node_dx, base_fov, rtol=1e-12, atol=0.0
        )
        or not np.allclose(
            np.asarray(open_shape) * node_dx, open_fov, rtol=1e-12, atol=0.0
        )
        or not np.allclose(open_fov, 2.0 * base_fov, rtol=1e-12, atol=0.0)
        or native_roi != baseline_shape
        or int(detector["selected_scan_index"]) != 0
        or detector.get("stream_scans") is not True
        or detector.get("retain_full_node_stacks") is not False
    ):
        msg = "R7 detector/open-grid sampling is inconsistent."
        raise ValueError(msg)
    _validate_scan_on_grid(_make_common_scan(_section(config, "scan")), node_dx)

    finite_b = _section(diagnostics_r7, "finite_b")
    feature_m = float(_section(config, "sample_b")["physical_feature_size_m"])
    support_m = 48.0 * feature_m
    if (
        list(finite_b.get("canonical_phase_cells", [])) != [48, 48]
        or not np.allclose(
            np.asarray(finite_b.get("physical_shape_m"), dtype=float),
            [support_m, support_m],
            rtol=1e-12,
            atol=0.0,
        )
        or float(finite_b.get("exterior_transmission_real", np.nan)) != 1.0
        or float(finite_b.get("exterior_transmission_imag", np.nan)) != 0.0
        or not np.allclose(base_fov, [2.0 * support_m] * 2, rtol=1e-12, atol=0.0)
    ):
        msg = "R7 finite-B definition does not match the nominal R6 case."
        raise ValueError(msg)
    comparisons = _section(diagnostics_r7, "comparisons")
    if dict(comparisons) != {
        "series_reference": "q8",
        "final_pair": ["q4", "q8"],
        "binary_effect": ["q1", "q8"],
        "denominator": "q8",
        "phase_scale_spatial_alignment": False,
    }:
        msg = "R7 comparisons do not match the pre-registration."
        raise ValueError(msg)
    if dict(_section(diagnostics_r7, "determinism")) != {
        "interface_factor": 8,
        "scan_index": 0,
    }:
        msg = "R7 determinism must use q8 scan 0."
        raise ValueError(msg)
    thresholds = _section(diagnostics_r7, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_acceptance_algebra_relative_l2_max",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R7 must reuse all registered thresholds."
        raise ValueError(msg)
    output = _section(config, "output")
    if output.get("save_r7_figures") is not True:
        msg = "R7 output must enable the three pre-registered figures."
        raise ValueError(msg)
    if any(
        output.get(f"save_r{stage}_figures") is not False
        for stage in range(1, 7)
    ):
        msg = "R7 must not recompute earlier diagnostic figures."
        raise ValueError(msg)


def _validate_r8_config(
    config: Mapping[str, Any], diagnostics_r8: Mapping[str, Any]
) -> None:
    """Validate the frozen exp040-R8 unified forward diagnostic."""

    if diagnostics_r8.get("version") != "R8":
        msg = "diagnostics_r8.version must be R8."
        raise ValueError(msg)
    for stage in range(1, 8):
        value = config.get(f"diagnostics_r{stage}")
        if not isinstance(value, Mapping) or value.get("enabled") is not False:
            msg = f"R8 requires diagnostics_r{stage}.enabled=false."
            raise ValueError(msg)
    expected_methods = {
        "interface": "q8_staggered_midpoint_air_area_fraction",
        "effective_index": "linear_cell_average_of_indicator",
        "multislice": "centered_symmetric_split_step_streamed_slices",
        "sample_a_cases": "q8_axial_lateral_and_finest_waist_sweep",
        "lateral_comparison_mapping": (
            "finest_to_coarse_centered_bilinear_complex_field"
        ),
        "a_to_detector_mapping": "centered_bilinear_scattered_residual",
        "finite_b": "same_nominal_96um_hard_edge_transparent_exterior",
        "finite_shift": "constant_zero_shift_of_b_minus_one",
        "ab_propagation": "alias_controlled_same_grid_192um",
        "open_bc": "residual_alias_controlled_288_384um",
        "detector": "q4_staggered_midpoint_uniform_positive_weights",
        "comparison": (
            "unaligned_relative_l2_registered_reference_denominators"
        ),
    }
    if dict(_section(diagnostics_r8, "methods")) != expected_methods:
        msg = "diagnostics_r8 methods do not match the pre-registration."
        raise ValueError(msg)
    expected_r7 = {
        "run": (
            "runs/exp040_TGV_3d_multislice_r7_subvoxel_interface_"
            "20260813_011329"
        ),
        "metrics_sha256": (
            "F7C26CC9B14778704C4F14B660515A8CD710B750A5C1E5EB0A868969CB4324BE"
        ),
        "hdf5_sha256": (
            "34D10D9FB506BD76FE887041788B803D74A1138E0E434140878686E649258551"
        ),
        "status": "Passed",
        "q4_to_q8_u_a_exit": 0.0198499,
        "q4_to_q8_p_b": 0.000498961,
        "q4_to_q8_i_stack": 0.000250482,
    }
    if dict(_section(diagnostics_r8, "r7_provenance")) != expected_r7:
        msg = "R8 R7 provenance does not match the formal run."
        raise ValueError(msg)
    expected_r6 = {
        "run": (
            "runs/exp040_TGV_3d_multislice_r6_b_support_sensitivity_"
            "20260811_200120"
        ),
        "metrics_sha256": (
            "5813692089C374892D961152250225F71CE05843828A5F8D9FBE5CEBA33B987A"
        ),
        "maximum_nominal_b_variation": 0.17129597874704286,
        "combined_with_r8_metrics": False,
        "used_in_r8_gate": False,
    }
    if dict(_section(diagnostics_r8, "r6_context")) != expected_r6:
        msg = "R8 must preserve the frozen R6 context without combining it."
        raise ValueError(msg)
    if dict(_section(diagnostics_r8, "interface")) != {
        "factor": 8,
        "node_rule": "pixel_center_plus_a_half_over_q",
        "weights": "uniform_nonnegative",
        "axial_subnodes": 1,
    }:
        msg = "R8 interface must remain fixed at the registered q8 rule."
        raise ValueError(msg)

    sample_cases = _section(diagnostics_r8, "sample_a_cases")
    fov = np.asarray(sample_cases["fov_m"], dtype=np.float64)
    registered_fov = np.asarray(
        _section(_section(config, "convergence"), "lateral_fixed_fov")["fov_m"],
        dtype=np.float64,
    )
    raw_cases = sample_cases.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        msg = "R8 sample_a_cases.cases must be a sequence."
        raise ValueError(msg)
    cases = list(raw_cases)
    expected_ids = [
        "axial_coarse",
        "common_reference",
        "finest_baseline",
        "waist_minus",
        "waist_plus",
    ]
    case_ids = [
        case.get("id") if isinstance(case, Mapping) else None
        for case in cases
    ]
    if case_ids != expected_ids:
        msg = "R8 sample-A case IDs/order do not match the pre-registration."
        raise ValueError(msg)
    parsed: dict[str, tuple[tuple[int, int], float, float, float]] = {}
    sample_a = _section(config, "sample_a")
    for case in cases:
        if not isinstance(case, Mapping):
            msg = "Each R8 sample-A case must be a mapping."
            raise ValueError(msg)
        case_id = str(case["id"])
        shape = _shape(case["shape"], f"R8 {case_id} shape")
        dx_m = _isotropic_dx(case["dx_m"], f"R8 {case_id} dx")
        dz_m = _positive(case["dz_m"], f"R8 {case_id} dz")
        waist_m = _positive(case["d_waist_m"], f"R8 {case_id} waist")
        if not np.allclose(
            np.asarray(shape) * dx_m, fov, rtol=1e-12, atol=0.0
        ):
            msg = f"R8 {case_id} does not preserve the registered A FOV."
            raise ValueError(msg)
        _, widths = midpoint_z_grid(float(sample_a["thickness_m"]), dz_m)
        if not np.isclose(float(np.sum(widths)), float(sample_a["thickness_m"])):
            msg = f"R8 {case_id} slice widths do not span sample A."
            raise ValueError(msg)
        parsed[case_id] = (shape, dx_m, dz_m, waist_m)
    if (
        fov.shape != (2,)
        or not np.allclose(fov, registered_fov, rtol=1e-12, atol=0.0)
    ):
        msg = "R8 sample-A FOV must equal the registered 64 um domain."
        raise ValueError(msg)
    axial_coarse = parsed["axial_coarse"]
    common = parsed["common_reference"]
    finest = parsed["finest_baseline"]
    minus = parsed["waist_minus"]
    plus = parsed["waist_plus"]
    waist_delta = float(_section(config, "waist_perturbation")["delta_d_waist_m"])
    if (
        axial_coarse[0] != common[0]
        or not np.isclose(axial_coarse[1], common[1])
        or not np.isclose(axial_coarse[2], 2.0 * common[2])
        or not np.isclose(axial_coarse[3], common[3])
        or finest[0] != (2 * common[0][0], 2 * common[0][1])
        or not np.isclose(common[1], 2.0 * finest[1])
        or not np.isclose(common[2], finest[2])
        or not np.isclose(common[3], finest[3])
        or minus[:3] != finest[:3]
        or plus[:3] != finest[:3]
        or not np.isclose(minus[3], finest[3] - waist_delta)
        or not np.isclose(plus[3], finest[3] + waist_delta)
    ):
        msg = "R8 axial/lateral/waist case relationships are inconsistent."
        raise ValueError(msg)
    expected_case_registration = {
        "axial_pair": ["axial_coarse", "common_reference"],
        "axial_reference": "common_reference",
        "lateral_pair": ["common_reference", "finest_baseline"],
        "lateral_reference": "finest_baseline",
        "waist_cases": ["waist_minus", "finest_baseline", "waist_plus"],
        "waist_reference": "finest_baseline",
        "reuse_common_cases": True,
        "stream_slices": True,
        "retain_full_volumes": False,
    }
    if any(
        sample_cases.get(key) != value
        for key, value in expected_case_registration.items()
    ):
        msg = "R8 case pairs/references do not match the pre-registration."
        raise ValueError(msg)

    detector = _section(diagnostics_r8, "detector_sampling")
    factor = int(detector["quadrature_factor"])
    node_dx = _positive(detector["node_dx_m"], "R8 detector node dx")
    pixel_m = float(_section(_section(config, "optics"), "detector")["pixel_size_m"])
    base_shape = _shape(detector["base_node_shape"], "R8 base node shape")
    primary_shape = _shape(
        detector["primary_open_node_shape"], "R8 primary open shape"
    )
    base_fov = np.asarray(detector["base_fov_m"], dtype=np.float64)
    primary_fov = np.asarray(detector["primary_open_fov_m"], dtype=np.float64)
    native_roi = _shape(detector["native_roi_shape"], "R8 native ROI")
    baseline_shape = _shape(_section(config, "optics")["baseline_shape"], "baseline")
    if (
        factor != 4
        or not np.isclose(node_dx, pixel_m / factor, rtol=1e-12, atol=0.0)
        or not np.isclose(node_dx, finest[1], rtol=1e-12, atol=0.0)
        or not np.allclose(
            np.asarray(base_shape) * node_dx,
            base_fov,
            rtol=1e-12,
            atol=0.0,
        )
        or not np.allclose(
            np.asarray(primary_shape) * node_dx,
            primary_fov,
            rtol=1e-12,
            atol=0.0,
        )
        or not np.allclose(primary_fov, 2.0 * base_fov, rtol=1e-12, atol=0.0)
        or native_roi != baseline_shape
        or int(detector["selected_scan_index"]) != 0
        or detector.get("stream_scans") is not True
        or detector.get("retain_full_node_stacks") is not False
    ):
        msg = "R8 detector/open-grid sampling is inconsistent."
        raise ValueError(msg)
    _validate_scan_on_grid(_make_common_scan(_section(config, "scan")), node_dx)

    finite_b = _section(diagnostics_r8, "finite_b")
    feature_m = float(_section(config, "sample_b")["physical_feature_size_m"])
    support_m = 48.0 * feature_m
    if (
        list(finite_b.get("canonical_phase_cells", [])) != [48, 48]
        or not np.allclose(
            np.asarray(finite_b.get("physical_shape_m"), dtype=float),
            [support_m, support_m],
            rtol=1e-12,
            atol=0.0,
        )
        or float(finite_b.get("exterior_transmission_real", np.nan)) != 1.0
        or float(finite_b.get("exterior_transmission_imag", np.nan)) != 0.0
        or not np.allclose(base_fov, [2.0 * support_m] * 2, rtol=1e-12, atol=0.0)
    ):
        msg = "R8 finite-B definition does not match the nominal case."
        raise ValueError(msg)

    open_control = _section(diagnostics_r8, "open_control")
    open_fovs = np.asarray(open_control["fov_m"], dtype=np.float64)
    open_shapes = np.asarray(open_control["node_shapes"], dtype=np.int64)
    if (
        open_control.get("case_id") != "finest_baseline"
        or open_fovs.shape != (2, 2)
        or open_shapes.shape != (2, 2)
        or not np.allclose(open_shapes * node_dx, open_fovs, rtol=1e-12, atol=0.0)
        or not np.allclose(open_fovs[0], 1.5 * base_fov, rtol=1e-12, atol=0.0)
        or not np.allclose(open_fovs[1], primary_fov, rtol=1e-12, atol=0.0)
        or open_control.get("acceptance_pair") != ["open_288", "open_384"]
        or open_control.get("reference") != "open_384"
        or open_control.get("denominator")
        != "open_384_finest_baseline_i_stack"
    ):
        msg = "R8 open-control pair does not match the pre-registration."
        raise ValueError(msg)

    comparisons = _section(diagnostics_r8, "comparisons")
    expected_comparisons = {
        "outputs": ["U_A_exit", "P_B", "I_stack"],
        "axial_denominator": "common_reference",
        "lateral_denominator": "finest_baseline",
        "waist_denominator": "finest_baseline",
        "lateral_u_a_exit_common_grid": list(common[0]),
        "lateral_u_a_exit_common_dx_m": common[1],
        "detector_floor_components": ["axial", "lateral", "open"],
        "detector_visibility_uses_full_stack": True,
        "phase_scale_spatial_alignment": False,
    }
    if dict(comparisons) != expected_comparisons:
        msg = "R8 comparisons do not match the pre-registration."
        raise ValueError(msg)
    if dict(_section(diagnostics_r8, "determinism")) != {
        "case_id": "finest_baseline",
        "scan_index": 0,
        "open_reference": "open_384",
    }:
        msg = "R8 determinism case does not match the pre-registration."
        raise ValueError(msg)
    thresholds = _section(diagnostics_r8, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_acceptance_algebra_relative_l2_max",
            "reuse_acceptance_determinism_relative_l2_max",
            "reuse_detector_visibility_signal_to_floor_min",
        )
    ):
        msg = "R8 must reuse all registered thresholds."
        raise ValueError(msg)
    output = _section(config, "output")
    if output.get("save_r8_figures") is not True:
        msg = "R8 output must enable the three pre-registered figures."
        raise ValueError(msg)
    if any(output.get(f"save_r{stage}_figures") is not False for stage in range(1, 8)):
        msg = "R8 must not recompute earlier diagnostic figures."
        raise ValueError(msg)


def _validate_r9_config(
    config: Mapping[str, Any], diagnostics_r9: Mapping[str, Any]
) -> None:
    """Validate the frozen R9 A-exit attribution registration."""

    if diagnostics_r9.get("version") != "R9":
        msg = "diagnostics_r9.version must be R9."
        raise ValueError(msg)
    expected_methods = {
        "interface": "q8_staggered_midpoint_air_area_fraction",
        "effective_index": "linear_cell_average_of_indicator",
        "multislice": "centered_symmetric_split_step_streamed_slices",
        "raw_comparison": "unaligned_total_complex_field_relative_l2",
        "passband": "exact_external_medium_propagating_disk",
        "passband_order": "project_each_native_grid_then_restrict",
        "fourier_projection": "ifft2_fft2_times_binary_mask",
        "bilinear_restriction": "centered_bilinear_complex_field",
        "conservative_restriction": "aligned_2x2_complex_cell_average",
        "spectral_attribution": "orthogonal_parseval_inside_outside_energy",
    }
    if dict(_section(diagnostics_r9, "methods")) != expected_methods:
        msg = "diagnostics_r9 methods do not match the pre-registration."
        raise ValueError(msg)
    expected_r8 = {
        "run": (
            "runs/exp040_TGV_3d_multislice_r8_unified_visibility_"
            "20260814_152034"
        ),
        "metrics_sha256": (
            "DADE29E7625B7FCEB904534B674DBAF31A37C86066A9A1870B22E9F618F3DA17"
        ),
        "hdf5_sha256": (
            "5004947CDF767F3777E4356CCBE67BA75AD27F254D3EE978690B9ECC3AD776E7"
        ),
        "status": "Inconclusive",
        "raw_axial_u_a_exit": 0.0709483058386522,
        "raw_lateral_u_a_exit": 0.20874412488300237,
    }
    if dict(_section(diagnostics_r9, "r8_provenance")) != expected_r8:
        msg = "R9 R8 provenance does not match the frozen formal run."
        raise ValueError(msg)
    if dict(_section(diagnostics_r9, "interface")) != {
        "factor": 8,
        "node_rule": "pixel_center_plus_a_half_over_q",
        "weights": "uniform_nonnegative",
        "axial_subnodes": 1,
    }:
        msg = "R9 interface must remain fixed at the registered q8 rule."
        raise ValueError(msg)

    registration = _section(diagnostics_r9, "sample_a_cases")
    fov = np.asarray(registration["fov_m"], dtype=np.float64)
    registered_fov = np.asarray(
        _section(_section(config, "convergence"), "lateral_fixed_fov")[
            "fov_m"
        ],
        dtype=np.float64,
    )
    raw_cases = registration.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(
        raw_cases, (str, bytes)
    ):
        msg = "R9 sample_a_cases.cases must be a sequence."
        raise ValueError(msg)
    cases = list(raw_cases)
    expected_ids = [
        "axial_coarse",
        "common_reference",
        "axial_fine_reference",
        "lateral_fine_reference",
    ]
    case_ids = [
        case.get("id") if isinstance(case, Mapping) else None
        for case in cases
    ]
    if case_ids != expected_ids:
        msg = "R9 sample-A case IDs/order do not match the pre-registration."
        raise ValueError(msg)
    parsed: dict[str, tuple[tuple[int, int], float, float, float]] = {}
    sample_a = _section(config, "sample_a")
    thickness = float(sample_a["thickness_m"])
    for case in cases:
        if not isinstance(case, Mapping):
            msg = "Each R9 sample-A case must be a mapping."
            raise ValueError(msg)
        case_id = str(case["id"])
        shape = _shape(case["shape"], f"R9 {case_id} shape")
        dx_m = _isotropic_dx(case["dx_m"], f"R9 {case_id} dx")
        dz_m = _positive(case["dz_m"], f"R9 {case_id} dz")
        waist_m = _positive(case["d_waist_m"], f"R9 {case_id} waist")
        if not np.allclose(
            np.asarray(shape) * dx_m, fov, rtol=1e-12, atol=0.0
        ):
            msg = f"R9 {case_id} does not preserve the registered A FOV."
            raise ValueError(msg)
        _, widths = midpoint_z_grid(thickness, dz_m)
        if not np.isclose(float(np.sum(widths)), thickness):
            msg = f"R9 {case_id} slice widths do not span sample A."
            raise ValueError(msg)
        parsed[case_id] = (shape, dx_m, dz_m, waist_m)
    if (
        fov.shape != (2,)
        or not np.allclose(fov, registered_fov, rtol=1e-12, atol=0.0)
    ):
        msg = "R9 sample-A FOV must equal the registered lateral FOV."
        raise ValueError(msg)
    coarse = parsed["axial_coarse"]
    common = parsed["common_reference"]
    axial_fine = parsed["axial_fine_reference"]
    lateral_fine = parsed["lateral_fine_reference"]
    if (
        coarse[0] != common[0]
        or not np.isclose(coarse[1], common[1])
        or not np.isclose(coarse[2], 2.0 * common[2])
        or axial_fine[0] != common[0]
        or not np.isclose(axial_fine[1], common[1])
        or not np.isclose(axial_fine[2], common[2] / 2.0)
        or lateral_fine[0] != (2 * common[0][0], 2 * common[0][1])
        or not np.isclose(lateral_fine[1], common[1] / 2.0)
        or not np.isclose(lateral_fine[2], common[2])
        or any(
            not np.isclose(case[3], common[3])
            for case in (coarse, axial_fine, lateral_fine)
        )
        or not np.isclose(common[3], float(sample_a["d_waist_m"]))
    ):
        msg = "R9 axial/lateral case relationships are inconsistent."
        raise ValueError(msg)
    expected_registration = {
        "r8_axial_reproduction_pair": ["axial_coarse", "common_reference"],
        "r8_axial_reproduction_reference": "common_reference",
        "axial_refinement_pair": ["common_reference", "axial_fine_reference"],
        "axial_refinement_reference": "axial_fine_reference",
        "lateral_pair": ["common_reference", "lateral_fine_reference"],
        "lateral_reference": "lateral_fine_reference",
        "reuse_common_cases": True,
        "stream_slices": True,
        "retain_full_volumes": False,
    }
    if any(
        registration.get(key) != value
        for key, value in expected_registration.items()
    ):
        msg = "R9 case pairs/references do not match the pre-registration."
        raise ValueError(msg)

    passband = _section(diagnostics_r9, "physical_passband")
    wavelength = float(_section(config, "optics")["wavelength_m"])
    external_index = float(
        _section(config, "optics")["external_medium_index"]
    )
    cutoff = _positive(
        passband["cutoff_cycles_per_m"], "R9 passband cutoff"
    )
    expected_cutoff = external_index / wavelength
    expected_passband = {
        "medium_index_source": "optics.external_medium_index",
        "cutoff_definition": (
            "external_medium_index_over_vacuum_wavelength"
        ),
        "mask_geometry": "radial_disk_intersect_native_nyquist_rectangle",
        "boundary_inclusive": True,
        "frequency_axis": "numpy_fft_fftfreq_cycles_per_m",
        "apply_to": "total_complex_u_a_exit",
    }
    if (
        any(passband.get(key) != value for key, value in expected_passband.items())
        or not np.isclose(cutoff, expected_cutoff, rtol=1e-12, atol=0.0)
        or cutoff >= 0.5 / common[1]
    ):
        msg = "R9 external propagating passband is inconsistent."
        raise ValueError(msg)

    restrictions = _section(diagnostics_r9, "lateral_restrictions")
    expected_restrictions = {
        "methods": [
            "centered_bilinear_complex_field",
            "aligned_2x2_complex_cell_average",
        ],
        "refinement_ratio": 2,
        "target_shape": list(common[0]),
        "target_dx_m": common[1],
        "cell_average_weights": [0.25, 0.25, 0.25, 0.25],
        "compare_both_without_selection": True,
    }
    if dict(restrictions) != expected_restrictions:
        msg = "R9 lateral restrictions do not match the pre-registration."
        raise ValueError(msg)

    comparisons = _section(diagnostics_r9, "comparisons")
    expected_comparisons = {
        "raw_and_passband_outputs": ["U_A_exit"],
        "r8_axial_denominator": "common_reference",
        "axial_refinement_denominator": "axial_fine_reference",
        "lateral_denominator": "restricted_lateral_fine_reference",
        "passband_before_lateral_restriction": True,
        "phase_scale_spatial_alignment": False,
        "r8_reproduction_absolute_tolerance_source": (
            "acceptance.algebra_relative_l2_max"
        ),
        "restriction_equivalence_tolerance_source": (
            "acceptance.algebra_relative_l2_max"
        ),
        "difference_energy_fraction_is_report_only": True,
    }
    if dict(comparisons) != expected_comparisons:
        msg = "R9 comparisons do not match the pre-registration."
        raise ValueError(msg)
    if dict(_section(diagnostics_r9, "determinism")) != {
        "scope": "repeated_passband_and_restriction_postprocessing",
        "case_id": "lateral_fine_reference",
    }:
        msg = "R9 determinism scope does not match the pre-registration."
        raise ValueError(msg)
    thresholds = _section(diagnostics_r9, "thresholds")
    if not all(
        thresholds.get(name) is True
        for name in (
            "reuse_acceptance_convergence_relative_l2_max",
            "reuse_acceptance_algebra_relative_l2_max",
            "reuse_acceptance_determinism_relative_l2_max",
        )
    ):
        msg = "R9 must reuse all registered thresholds."
        raise ValueError(msg)
    for stage in range(1, 9):
        stage_config = config.get(f"diagnostics_r{stage}")
        if not isinstance(stage_config, Mapping) or (
            stage_config.get("enabled") is not False
        ):
            msg = "R9 must not recompute earlier diagnostics."
            raise ValueError(msg)
    output = _section(config, "output")
    if output.get("save_r9_figures") is not True:
        msg = "R9 output must enable the three pre-registered figures."
        raise ValueError(msg)
    if any(
        output.get(f"save_r{stage}_figures") is not False
        for stage in range(1, 9)
    ):
        msg = "R9 must not recompute earlier diagnostic figures."
        raise ValueError(msg)


def build_exp040_hdf5_payload(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_yaml: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build arguments for ``save_ptycho_hdf5`` without fake groups."""

    baseline = _section(result, "baseline")
    sweep = _section(result, "sweep")
    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    sample_b = _section(config, "sample_b")
    detector = _section(optics, "detector")
    truth = {
        "n_volume": baseline["n_volume"],
        "z_m": baseline["z_m"],
        "slice_thickness_m": baseline["slice_thickness_m"],
        "diameter_z_m": baseline["diameter_z_m"],
        "incident_field_true": baseline["incident_field"],
        "U_A_exit_true": baseline["U_A_exit"],
        "P_B_true": baseline["P_B"],
        "B_true": baseline["B"],
        "projected_phase_product_true": result["controls"][
            "phase_screen_product"
        ],
        "parameter_sweep": {
            "case_ids": sweep["case_ids"],
            "d_waist_m": sweep["d_waist_m"],
            "U_A_exit_true": sweep["U_A_exit"],
            "P_B_true": sweep["P_B"],
            "I_stack_true": sweep["I_stack"],
        },
    }
    return {
        "I_stack": baseline["I_stack"],
        "scan_positions": result["scan_positions"],
        "instrument": {
            "wavelength": float(optics["wavelength_m"]),
            "dx": np.asarray(optics["baseline_dx_m"], dtype=np.float64),
            "z_AB": float(optics["z_AB_m"]),
            "z_BC": float(optics["z_BC_m"]),
            "detector_pixel_size": float(detector["pixel_size_m"]),
            "internal_reference_index": float(
                optics["internal_reference_index"]
            ),
            "external_medium_index": float(optics["external_medium_index"]),
            "angular_spectrum_bandlimit": bool(
                optics["angular_spectrum_bandlimit"]
            ),
        },
        "sample": {
            "sample_A_type": str(sample_a["type"]),
            "tgv_parameters": dict(sample_a),
            "sample_B_type": str(sample_b["type"]),
            "sample_B_parameters": dict(sample_b),
        },
        "truth": truth,
        "config_yaml": config_yaml,
        "metadata": dict(metadata),
        "metrics": result["metrics"],
    }


def _simulate_case(
    config: Mapping[str, Any],
    *,
    shape: tuple[int, int],
    dx_m: float,
    dz_m: float,
    d_waist_m: float,
    sample_b: NDArray[np.complex128],
    scan_positions: NDArray[np.float64],
    keep_volume: bool,
) -> dict[str, Any]:
    optics = _section(config, "optics")
    illumination = _section(config, "illumination")
    sample_a = _section(config, "sample_a")
    bandlimit = bool(optics["angular_spectrum_bandlimit"])
    z_m, widths = midpoint_z_grid(float(sample_a["thickness_m"]), dz_m)
    n_volume, volume_metadata = make_tgv_refractive_index_volume(
        shape_xyz=(len(z_m), *shape),
        dx=dx_m,
        dz=dz_m,
        thickness=float(sample_a["thickness_m"]),
        d_top=float(sample_a["d_top_m"]),
        d_waist=d_waist_m,
        d_bottom=float(sample_a["d_bottom_m"]),
        z_waist=float(sample_a["z_waist_m"]),
        n_glass=float(sample_a["n_glass"]),
        n_air=float(sample_a["n_air"]),
        center_xy_m=tuple(float(value) for value in sample_a["center_xy_m"]),
    )
    metadata_widths = np.asarray(
        volume_metadata["slice_thickness_m"], dtype=np.float64
    )
    if not np.array_equal(metadata_widths, widths):
        msg = "volume slice widths disagree with midpoint_z_grid."
        raise RuntimeError(msg)
    incident = make_plane_wave(
        shape,
        dx_m,
        float(optics["wavelength_m"]),
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    exit_field = multislice_propagate_A(
        incident,
        n_volume,
        dx_m,
        widths,
        float(optics["wavelength_m"]),
        n_ref=float(optics["internal_reference_index"]),
        bandlimit=bandlimit,
    )
    intensity, probe, returned_b, forward_metadata = (
        simulate_exit_field_B_forward(
            exit_field,
            sample_b,
            scan_positions,
            dx_m,
            float(optics["wavelength_m"]),
            float(optics["z_AB_m"]),
            float(optics["z_BC_m"]),
            external_medium_index=float(optics["external_medium_index"]),
            bandlimit=bandlimit,
            object_boundary=str(_section(config, "sample_b")["object_boundary"]),
        )
    )
    output: dict[str, Any] = {
        "shape": shape,
        "dx_m": dx_m,
        "target_dz_m": dz_m,
        "d_waist_m": d_waist_m,
        "z_m": np.asarray(volume_metadata["z_m"], dtype=np.float64),
        "slice_thickness_m": metadata_widths,
        "diameter_z_m": np.asarray(
            volume_metadata["diameter_z_m"], dtype=np.float64
        ),
        "incident_field": incident,
        "U_A_exit": exit_field,
        "P_B": probe,
        "B": returned_b,
        "I_stack": intensity,
        "volume_metadata": volume_metadata,
        "forward_metadata": forward_metadata,
    }
    if keep_volume:
        output["n_volume"] = n_volume
    else:
        del n_volume
        gc.collect()
    return output


def _run_controls(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    positions: NDArray[np.float64],
    sweep_cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    acceptance = _section(config, "acceptance")
    wavelength = float(optics["wavelength_m"])
    n_reference = float(optics["internal_reference_index"])
    bandlimit = bool(optics["angular_spectrum_bandlimit"])
    widths = np.asarray(baseline["slice_thickness_m"], dtype=np.float64)
    incident = np.asarray(baseline["incident_field"], dtype=np.complex128)
    n_volume = np.asarray(baseline["n_volume"], dtype=np.float64)
    dx_m = float(baseline["dx_m"])
    thickness = float(sample_a["thickness_m"])

    geometry_tolerance = max(
        float(
            _section(
                acceptance, "geometry_thickness_absolute_tolerance_m"
            )["fixed_floor_m"]
        ),
        float(
            _section(
                acceptance, "geometry_thickness_absolute_tolerance_m"
            )["floating_point_factor"]
        )
        * np.finfo(np.float64).eps
        * thickness,
    )
    expected_diameter = diameter_profile(
        np.asarray(baseline["z_m"], dtype=np.float64),
        thickness,
        float(sample_a["d_top_m"]),
        float(sample_a["d_waist_m"]),
        float(sample_a["d_bottom_m"]),
        float(sample_a["z_waist_m"]),
    )
    unique_indices = np.unique(n_volume)
    geometry_metrics = {
        "volume_shape_valid": n_volume.shape
        == (len(widths), *tuple(baseline["shape"])),
        "volume_dtype_valid": n_volume.dtype == np.float64,
        "material_values_valid": np.array_equal(
            unique_indices,
            np.asarray(
                [float(sample_a["n_air"]), float(sample_a["n_glass"])],
                dtype=np.float64,
            ),
        ),
        "diameter_profile_max_abs_error_m": float(
            np.max(np.abs(expected_diameter - baseline["diameter_z_m"]))
        ),
        "slice_width_sum_m": float(np.sum(widths)),
        "slice_width_sum_abs_error_m": abs(float(np.sum(widths)) - thickness),
        "slice_width_tolerance_m": geometry_tolerance,
        "slice_widths_positive": bool(np.all(widths > 0.0)),
    }
    geometry_metrics["pass"] = bool(
        geometry_metrics["volume_shape_valid"]
        and geometry_metrics["volume_dtype_valid"]
        and geometry_metrics["material_values_valid"]
        and geometry_metrics["diameter_profile_max_abs_error_m"]
        <= geometry_tolerance
        and geometry_metrics["slice_width_sum_abs_error_m"]
        <= geometry_tolerance
        and geometry_metrics["slice_widths_positive"]
    )

    homogeneous = np.full_like(n_volume, n_reference)
    homogeneous_exit = multislice_propagate_A(
        incident,
        homogeneous,
        dx_m,
        widths,
        wavelength,
        n_ref=n_reference,
        bandlimit=bandlimit,
    )
    homogeneous_reference = angular_spectrum_propagate(
        incident,
        dx_m,
        wavelength,
        thickness,
        n=n_reference,
        bandlimit=bandlimit,
    )
    zero_error = relative_l2(homogeneous_exit, homogeneous_reference)

    single_volume = n_volume[len(widths) // 2 : len(widths) // 2 + 1]
    single_width = np.asarray([thickness], dtype=np.float64)
    single_exit = multislice_propagate_A(
        incident,
        single_volume,
        dx_m,
        single_width,
        wavelength,
        n_ref=n_reference,
        bandlimit=bandlimit,
    )
    half_propagated = angular_spectrum_propagate(
        incident,
        dx_m,
        wavelength,
        thickness / 2.0,
        n=n_reference,
        bandlimit=bandlimit,
    )
    single_transmission = np.exp(
        1j
        * (2.0 * np.pi / wavelength)
        * (single_volume[0] - n_reference)
        * thickness
    )
    single_reference = angular_spectrum_propagate(
        half_propagated * single_transmission,
        dx_m,
        wavelength,
        thickness / 2.0,
        n=n_reference,
        bandlimit=bandlimit,
    )
    single_error = relative_l2(single_exit, single_reference)

    product = multislice_phase_screen_product(
        incident,
        n_volume,
        widths,
        wavelength,
        n_ref=n_reference,
    )
    explicit_product = incident * np.exp(
        1j
        * (2.0 * np.pi / wavelength)
        * np.sum(
            (n_volume - n_reference) * widths[:, np.newaxis, np.newaxis],
            axis=0,
        )
    )
    no_propagation_error = relative_l2(product, explicit_product)

    projected_discrete = make_tgv_projected_phase(
        tuple(baseline["shape"]),
        dx_m,
        wavelength,
        thickness,
        float(sample_a["d_top_m"]),
        float(sample_a["d_waist_m"]),
        float(sample_a["d_bottom_m"]),
        float(baseline["target_dz_m"]),
        z_waist=float(sample_a["z_waist_m"]),
        n_glass=float(sample_a["n_glass"]),
        n_air=float(sample_a["n_air"]),
        center_xy_m=tuple(float(value) for value in sample_a["center_xy_m"]),
        lateral_supersampling=1,
        integration_method="midpoint",
    )
    projected_analytic = make_tgv_projected_phase(
        tuple(baseline["shape"]),
        dx_m,
        wavelength,
        thickness,
        float(sample_a["d_top_m"]),
        float(sample_a["d_waist_m"]),
        float(sample_a["d_bottom_m"]),
        float(baseline["target_dz_m"]),
        z_waist=float(sample_a["z_waist_m"]),
        n_glass=float(sample_a["n_glass"]),
        n_air=float(sample_a["n_air"]),
        center_xy_m=tuple(float(value) for value in sample_a["center_xy_m"]),
        lateral_supersampling=1,
        integration_method="analytic",
    )
    projected_field = np.asarray(
        projected_discrete["A_effective_true"], dtype=np.complex128
    )
    projected_error = relative_l2(product, incident * projected_field)
    reference_exit = homogeneous_reference
    if np.any(np.abs(reference_exit) <= np.finfo(np.float64).eps):
        msg = "reference exit field contains zeros; relative envelope undefined."
        raise RuntimeError(msg)
    relative_envelope = np.asarray(baseline["U_A_exit"]) / reference_exit

    repeated_canonical_b, repeated_canonical_dx = _make_canonical_b(
        _section(config, "sample_b")
    )
    repeated_positions = _make_common_scan(_section(config, "scan"))
    repeated_b = _sample_b_for_grid(
        repeated_canonical_b,
        repeated_canonical_dx,
        tuple(baseline["shape"]),
        dx_m,
    )
    repeated_baseline = _simulate_case(
        config,
        shape=tuple(baseline["shape"]),
        dx_m=dx_m,
        dz_m=float(baseline["target_dz_m"]),
        d_waist_m=float(baseline["d_waist_m"]),
        sample_b=repeated_b,
        scan_positions=repeated_positions,
        keep_volume=True,
    )
    repeated_product = multislice_phase_screen_product(
        np.asarray(repeated_baseline["incident_field"]),
        np.asarray(repeated_baseline["n_volume"]),
        np.asarray(repeated_baseline["slice_thickness_m"]),
        wavelength,
        n_ref=n_reference,
    )
    baseline_deterministic_errors = {
        "U_A_exit_relative_l2": relative_l2(
            repeated_baseline["U_A_exit"], baseline["U_A_exit"]
        ),
        "P_B_relative_l2": relative_l2(
            repeated_baseline["P_B"], baseline["P_B"]
        ),
        "I_stack_relative_l2": relative_l2(
            repeated_baseline["I_stack"], baseline["I_stack"]
        ),
        "n_volume_relative_l2": relative_l2(
            repeated_baseline["n_volume"], baseline["n_volume"]
        ),
        "z_m_relative_l2": relative_l2(
            repeated_baseline["z_m"], baseline["z_m"]
        ),
        "slice_thickness_m_relative_l2": relative_l2(
            repeated_baseline["slice_thickness_m"],
            baseline["slice_thickness_m"],
        ),
        "diameter_z_m_relative_l2": relative_l2(
            repeated_baseline["diameter_z_m"], baseline["diameter_z_m"]
        ),
        "incident_field_relative_l2": relative_l2(
            repeated_baseline["incident_field"], baseline["incident_field"]
        ),
        "B_relative_l2": relative_l2(repeated_baseline["B"], baseline["B"]),
        "scan_positions_relative_l2": relative_l2(
            repeated_positions, positions
        ),
        "projected_phase_product_relative_l2": relative_l2(
            repeated_product, product
        ),
    }
    case_ids = list(_section(config, "waist_perturbation")["case_ids"])
    sweep_deterministic_errors: dict[str, dict[str, float]] = {}
    determinism_values = list(baseline_deterministic_errors.values())
    for case_id, case in zip(case_ids, sweep_cases, strict=True):
        if case_id == "baseline":
            repeated_case = repeated_baseline
        else:
            repeated_case = _simulate_case(
                config,
                shape=tuple(case["shape"]),
                dx_m=float(case["dx_m"]),
                dz_m=float(case["target_dz_m"]),
                d_waist_m=float(case["d_waist_m"]),
                sample_b=np.asarray(repeated_baseline["B"], dtype=np.complex128),
                scan_positions=repeated_positions,
                keep_volume=False,
            )
        case_errors = {
            "U_A_exit_relative_l2": relative_l2(
                repeated_case["U_A_exit"], case["U_A_exit"]
            ),
            "P_B_relative_l2": relative_l2(
                repeated_case["P_B"], case["P_B"]
            ),
            "I_stack_relative_l2": relative_l2(
                repeated_case["I_stack"], case["I_stack"]
            ),
        }
        sweep_deterministic_errors[str(case_id)] = case_errors
        determinism_values.extend(case_errors.values())
    determinism_max = max(determinism_values)

    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    metrics = {
        "geometry": geometry_metrics,
        "zero_contrast_relative_l2": zero_error,
        "single_slice_relative_l2": single_error,
        "no_propagation_product_relative_l2": no_propagation_error,
        "projected_phase_product_relative_l2": projected_error,
        "projected_discrete_to_analytic_relative_l2": relative_l2(
            projected_field,
            projected_analytic["A_effective_true"],
        ),
        "determinism": {
            **baseline_deterministic_errors,
            "baseline": baseline_deterministic_errors,
            "waist_sweep": sweep_deterministic_errors,
            "scope": "all_saved_baseline_truth_data_and_waist_sweep_outputs",
            "convergence_arrays_not_saved_to_hdf5": True,
            "max_relative_l2": determinism_max,
            "pass": determinism_max
            <= float(acceptance["determinism_relative_l2_max"]),
        },
    }
    metrics["pass"] = bool(
        geometry_metrics["pass"]
        and zero_error <= float(acceptance["zero_contrast_relative_l2_max"])
        and single_error <= float(acceptance["single_slice_relative_l2_max"])
        and no_propagation_error <= algebra_threshold
        and projected_error
        <= float(acceptance["projected_phase_product_relative_l2_max"])
        and metrics["determinism"]["pass"]
    )
    arrays = {
        "phase_screen_product": product,
        "projected_phase": projected_field,
        "projected_phase_analytic": projected_analytic["A_effective_true"],
        "projected_difference": product - incident * projected_field,
        "multislice_relative_envelope": relative_envelope,
        "homogeneous_exit": homogeneous_exit,
        "homogeneous_reference": homogeneous_reference,
        "single_slice_exit": single_exit,
        "single_slice_reference": single_reference,
    }
    return arrays, metrics


def _run_convergence_cases(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    canonical_b: NDArray[np.complex128],
    canonical_dx: float,
    positions: NDArray[np.float64],
) -> dict[str, Any]:
    sample_a = _section(config, "sample_a")
    convergence_cfg = _section(config, "convergence")
    baseline_shape = tuple(baseline["shape"])
    baseline_dx = float(baseline["dx_m"])
    baseline_dz = float(baseline["target_dz_m"])
    waist = float(sample_a["d_waist_m"])

    axial_cfg = _section(convergence_cfg, "axial")
    axial_cases: list[dict[str, Any]] = []
    for dz_value in axial_cfg["dz_cases_m"]:
        dz_float = float(dz_value)
        if dz_float == baseline_dz:
            case = baseline
        else:
            case = _simulate_case(
                config,
                shape=baseline_shape,
                dx_m=baseline_dx,
                dz_m=dz_float,
                d_waist_m=waist,
                sample_b=baseline["B"],
                scan_positions=positions,
                keep_volume=False,
            )
        axial_cases.append(case)
    axial_errors = _errors_to_finest_same_grid(axial_cases)
    axial_pair = [float(value) for value in axial_cfg["acceptance_pair_m"]]
    axial_test = _find_case(axial_cases, "target_dz_m", axial_pair[0])
    axial_reference = _find_case(axial_cases, "target_dz_m", axial_pair[1])
    axial_acceptance = _three_output_errors(axial_test, axial_reference)

    lateral_cfg = _section(convergence_cfg, "lateral_fixed_fov")
    lateral_cases: list[dict[str, Any]] = []
    for case_cfg in lateral_cfg["cases"]:
        shape = _shape(case_cfg["shape"], "lateral shape")
        dx_value = _isotropic_dx(case_cfg["dx_m"], "lateral dx")
        if shape == baseline_shape and dx_value == baseline_dx:
            case = baseline
        else:
            sample_b = _sample_b_for_grid(
                canonical_b, canonical_dx, shape, dx_value
            )
            case = _simulate_case(
                config,
                shape=shape,
                dx_m=dx_value,
                dz_m=baseline_dz,
                d_waist_m=waist,
                sample_b=sample_b,
                scan_positions=positions,
                keep_volume=False,
            )
        lateral_cases.append(case)
    lateral_reference = min(lateral_cases, key=lambda case: case["dx_m"])
    lateral_errors = {
        "U_A_exit": [],
        "P_B": [],
        "I_stack": [],
    }
    for case in lateral_cases:
        mapped_reference = _map_case_to_grid(
            lateral_reference, tuple(case["shape"]), float(case["dx_m"])
        )
        errors = _three_output_errors(case, mapped_reference)
        for name in lateral_errors:
            lateral_errors[name].append(errors[name])
    lateral_pair = [
        float(value) for value in lateral_cfg["acceptance_pair_dx_m"]
    ]
    lateral_test = _find_case(lateral_cases, "dx_m", lateral_pair[0])
    lateral_fine = _find_case(lateral_cases, "dx_m", lateral_pair[1])
    lateral_mapped = _map_case_to_grid(
        lateral_fine,
        tuple(lateral_test["shape"]),
        float(lateral_test["dx_m"]),
    )
    lateral_acceptance = _three_output_errors(lateral_test, lateral_mapped)

    fov_cfg = _section(convergence_cfg, "fov")
    fov_dx = float(fov_cfg["fixed_dx_m"])
    fov_cases: list[dict[str, Any]] = []
    for shape_cfg in fov_cfg["shapes"]:
        shape = _shape(shape_cfg, "fov shape")
        if shape == baseline_shape and fov_dx == baseline_dx:
            case = baseline
        else:
            sample_b = _sample_b_for_grid(
                canonical_b, canonical_dx, shape, fov_dx
            )
            case = _simulate_case(
                config,
                shape=shape,
                dx_m=fov_dx,
                dz_m=baseline_dz,
                d_waist_m=waist,
                sample_b=sample_b,
                scan_positions=positions,
                keep_volume=False,
            )
        fov_cases.append(case)
    common_shape = _shape(fov_cfg["common_center_roi_shape"], "common shape")
    largest = max(fov_cases, key=lambda case: np.prod(case["shape"]))
    largest_common = _crop_case(largest, common_shape)
    fov_errors = {"U_A_exit": [], "P_B": [], "I_stack": []}
    for case in fov_cases:
        common = _crop_case(case, common_shape)
        errors = _three_output_errors(common, largest_common)
        for name in fov_errors:
            fov_errors[name].append(errors[name])
    fov_pair_shapes = [
        _shape(value, "fov acceptance shape")
        for value in fov_cfg["acceptance_pair_shapes"]
    ]
    fov_test = _find_case(fov_cases, "shape", fov_pair_shapes[0])
    fov_reference = _find_case(fov_cases, "shape", fov_pair_shapes[1])
    fov_acceptance = _three_output_errors(
        _crop_case(fov_test, common_shape),
        _crop_case(fov_reference, common_shape),
    )

    return {
        "axial": {
            "case_values_m": np.asarray(
                [case["target_dz_m"] for case in axial_cases],
                dtype=np.float64,
            ),
            "x_values": np.asarray(
                [case["target_dz_m"] for case in axial_cases],
                dtype=np.float64,
            ),
            "cases": axial_cases,
            "errors_to_finest": axial_errors,
            **axial_errors,
            "acceptance_pair_m": np.asarray(axial_pair, dtype=np.float64),
            "acceptance": axial_acceptance,
        },
        "lateral": {
            "case_values_m": np.asarray(
                [case["dx_m"] for case in lateral_cases], dtype=np.float64
            ),
            "x_values": np.asarray(
                [case["dx_m"] for case in lateral_cases], dtype=np.float64
            ),
            "cases": lateral_cases,
            "errors_to_finest": lateral_errors,
            **lateral_errors,
            "acceptance_pair_m": np.asarray(lateral_pair, dtype=np.float64),
            "acceptance": lateral_acceptance,
        },
        "fov": {
            "case_values_m": np.asarray(
                [case["shape"][1] * case["dx_m"] for case in fov_cases],
                dtype=np.float64,
            ),
            "x_values": np.asarray(
                [case["shape"][1] * case["dx_m"] for case in fov_cases],
                dtype=np.float64,
            ),
            "cases": fov_cases,
            "errors_to_largest": fov_errors,
            **fov_errors,
            "acceptance_pair_shapes": np.asarray(
                fov_pair_shapes, dtype=np.int64
            ),
            "acceptance": fov_acceptance,
            "common_shape": common_shape,
        },
    }


def _assemble_metrics(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    sweep_cases: Sequence[Mapping[str, Any]],
    control_metrics: Mapping[str, Any],
    convergence: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = _section(config, "acceptance")
    convergence_threshold = float(acceptance["convergence_relative_l2_max"])
    convergence_metrics: dict[str, Any] = {}
    floors = {"U_A_exit": 0.0, "P_B": 0.0, "I_stack": 0.0}
    convergence_pass = True
    for group_name in ("axial", "lateral", "fov"):
        acceptance_values = dict(convergence[group_name]["acceptance"])
        group_pass = all(
            float(value) <= convergence_threshold
            for value in acceptance_values.values()
        )
        convergence_pass = convergence_pass and group_pass
        convergence_metrics[group_name] = {
            "acceptance": acceptance_values,
            "pass": group_pass,
            "case_values_m": convergence[group_name]["case_values_m"],
        }
        if group_name == "fov":
            convergence_metrics[group_name]["acceptance_pair_shapes"] = (
                convergence[group_name]["acceptance_pair_shapes"]
            )
        else:
            convergence_metrics[group_name]["acceptance_pair_m"] = convergence[
                group_name
            ]["acceptance_pair_m"]
        for output_name, value in acceptance_values.items():
            floors[output_name] = max(floors[output_name], float(value))

    baseline_case = sweep_cases[1]
    minus_case = sweep_cases[0]
    plus_case = sweep_cases[2]
    signals: dict[str, dict[str, float]] = {}
    ratios: dict[str, float] = {}
    for output_name in ("U_A_exit", "P_B", "I_stack"):
        minus_signal = relative_l2(
            minus_case[output_name], baseline_case[output_name]
        )
        plus_signal = relative_l2(
            plus_case[output_name], baseline_case[output_name]
        )
        floor = floors[output_name]
        signals[output_name] = {
            "waist_minus_relative_l2": minus_signal,
            "waist_plus_relative_l2": plus_signal,
            "discretization_floor": floor,
        }
        ratios[output_name] = min(minus_signal, plus_signal) / max(
            floor, np.finfo(np.float64).eps
        )
        signals[output_name]["signal_to_floor_min"] = ratios[output_name]

    per_frame_minus = _per_frame_errors(
        minus_case["I_stack"], baseline_case["I_stack"]
    )
    per_frame_plus = _per_frame_errors(
        plus_case["I_stack"], baseline_case["I_stack"]
    )
    visibility_threshold = float(
        acceptance["detector_visibility_signal_to_floor_min"]
    )
    visibility_pass = ratios["I_stack"] >= visibility_threshold
    visibility = {
        "signals": signals,
        "floor": floors["I_stack"],
        "detector_signal_to_floor_min": ratios["I_stack"],
        "detector_signal_to_floor_threshold": visibility_threshold,
        "detector_per_frame_relative_change_minus": per_frame_minus,
        "detector_per_frame_relative_change_plus": per_frame_plus,
        "per_frame_minus": per_frame_minus,
        "per_frame_plus": per_frame_plus,
        "detector_per_frame_minus_min": float(np.min(per_frame_minus)),
        "detector_per_frame_minus_median": float(np.median(per_frame_minus)),
        "detector_per_frame_minus_max": float(np.max(per_frame_minus)),
        "detector_per_frame_plus_min": float(np.min(per_frame_plus)),
        "detector_per_frame_plus_median": float(np.median(per_frame_plus)),
        "detector_per_frame_plus_max": float(np.max(per_frame_plus)),
        "most_sensitive_frame_minus": int(np.argmax(per_frame_minus)),
        "most_sensitive_frame_plus": int(np.argmax(per_frame_plus)),
        "most_sensitive_frame": int(
            np.argmax(0.5 * (per_frame_minus + per_frame_plus))
        ),
        "pass": visibility_pass,
    }

    arrays_to_check: list[NDArray[Any]] = [
        np.asarray(baseline["n_volume"]),
        np.asarray(baseline["U_A_exit"]),
        np.asarray(baseline["P_B"]),
        np.asarray(baseline["I_stack"]),
    ]
    for case in sweep_cases:
        arrays_to_check.extend(
            [
                np.asarray(case["U_A_exit"]),
                np.asarray(case["P_B"]),
                np.asarray(case["I_stack"]),
            ]
        )
    for group_name in ("axial", "lateral", "fov"):
        for case in convergence[group_name]["cases"]:
            arrays_to_check.extend(
                [
                    np.asarray(case["U_A_exit"]),
                    np.asarray(case["P_B"]),
                    np.asarray(case["I_stack"]),
                ]
            )
    all_finite = all(np.all(np.isfinite(array)) for array in arrays_to_check)
    intensity_arrays = [np.asarray(baseline["I_stack"])]
    intensity_arrays.extend(np.asarray(case["I_stack"]) for case in sweep_cases)
    for group_name in ("axial", "lateral", "fov"):
        intensity_arrays.extend(
            np.asarray(case["I_stack"])
            for case in convergence[group_name]["cases"]
        )
    intensity_nonnegative = all(
        np.all(intensity >= 0.0) for intensity in intensity_arrays
    )
    finite_pass = bool(all_finite and intensity_nonnegative)
    hard_controls_pass = bool(control_metrics["pass"] and finite_pass)
    if not hard_controls_pass:
        status = "Failed"
    elif not convergence_pass or not visibility_pass:
        status = "Inconclusive"
    else:
        status = "Passed"

    return {
        "experiment_status": status,
        "thresholds": {
            "algebra_relative_l2_max": float(
                acceptance["algebra_relative_l2_max"]
            ),
            "convergence_relative_l2_max": convergence_threshold,
            "determinism_relative_l2_max": float(
                acceptance["determinism_relative_l2_max"]
            ),
            "detector_visibility_signal_to_floor_min": visibility_threshold,
        },
        "geometry_control": control_metrics["geometry"],
        "algebraic_controls": {
            key: value
            for key, value in control_metrics.items()
            if key not in {"geometry", "determinism"}
        },
        "determinism_control": control_metrics["determinism"],
        "finite_control": {
            "all_outputs_finite": bool(all_finite),
            "all_intensity_nonnegative": bool(intensity_nonnegative),
            "pass": finite_pass,
        },
        "convergence": convergence_metrics,
        "visibility": visibility,
        "stage_status": {
            "hard_controls_pass": hard_controls_pass,
            "convergence_pass": bool(convergence_pass),
            "detector_visibility_pass": bool(visibility_pass),
        },
    }


def _make_r1_canonical_b(
    sample_b: Mapping[str, Any],
    diagnostics_r1: Mapping[str, Any],
    legacy_canonical: NDArray[np.complex128],
    legacy_dx_m: float,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    float,
    dict[str, Any],
]:
    """Build and validate the pre-registered fine-grid sample-B realization."""

    refinement = _section(diagnostics_r1, "sample_b_refinement")
    base_grid = _section(refinement, "base_grid")
    working_grid = _section(refinement, "working_grid")
    base_shape = _shape(base_grid["shape"], "R1 B base shape")
    working_shape = _shape(working_grid["shape"], "R1 B working shape")
    dx_m = _isotropic_dx(base_grid["dx_m"], "R1 B base dx")
    feature_pixels = _require_integer_ratio(
        float(sample_b["physical_feature_size_m"]),
        dx_m,
        "R1 B physical feature size",
    )
    base = make_random_phase_object(
        base_shape,
        phase_range=float(sample_b["phase_range_rad"]),
        seed=int(sample_b["seed"]),
        feature_size_px=feature_pixels,
    )
    base = np.asarray(base, dtype=np.complex128)
    padding = _shape(
        refinement["extension_each_side_px"], "R1 B extension padding"
    )
    working = np.pad(
        base,
        ((padding[0], padding[0]), (padding[1], padding[1])),
        mode="wrap",
    )
    working = np.asarray(working, dtype=np.complex128)
    if working.shape != working_shape:
        msg = "R1 periodic extension produced an unexpected working shape."
        raise RuntimeError(msg)

    mapped_to_r0 = resample_centered_grid(
        base,
        dx_m,
        tuple(legacy_canonical.shape),
        legacy_dx_m,
    )
    mapping_error = float(
        np.max(np.abs(mapped_to_r0 - np.asarray(legacy_canonical)))
    )
    center_error = float(np.max(np.abs(center_crop(working, base_shape) - base)))
    modulus_error = float(
        max(
            np.max(np.abs(np.abs(base) - 1.0)),
            np.max(np.abs(np.abs(working) - 1.0)),
        )
    )
    mapping_tolerance = float(refinement["r0_mapping_max_complex_error"])
    validation = {
        "base_shape": np.asarray(base_shape, dtype=np.int64),
        "working_shape": np.asarray(working_shape, dtype=np.int64),
        "dx_m": dx_m,
        "feature_size_px": feature_pixels,
        "max_complex_error": mapping_error,
        "r0_mapping_max_complex_error": mapping_tolerance,
        "periodic_center_max_complex_error": center_error,
        "unit_modulus_max_abs_error": modulus_error,
        "complex128": bool(
            base.dtype == np.complex128 and working.dtype == np.complex128
        ),
        "all_finite": bool(
            np.all(np.isfinite(base)) and np.all(np.isfinite(working))
        ),
    }
    validation["pass"] = bool(
        validation["complex128"]
        and validation["all_finite"]
        and mapping_error <= mapping_tolerance
        and center_error == 0.0
        and modulus_error <= 32.0 * np.finfo(np.float64).eps
    )
    return base, working, dx_m, validation


def _sample_r1_b(
    working_canonical: NDArray[np.complex128],
    canonical_dx_m: float,
    shape: tuple[int, int],
    dx_m: float,
) -> NDArray[np.complex128]:
    """Sample R1 B, using an exact crop when source sampling is unchanged."""

    target_shape = _shape(shape, "R1 sample-B target shape")
    if np.isclose(dx_m, canonical_dx_m, rtol=0.0, atol=0.0):
        sampled = center_crop(working_canonical, target_shape).copy()
    else:
        sampled = resample_centered_grid(
            working_canonical,
            canonical_dx_m,
            target_shape,
            dx_m,
        )
    return np.asarray(sampled, dtype=np.complex128)


def _center_pad(
    values: NDArray[np.generic], target_shape: tuple[int, int]
) -> NDArray[Any]:
    """Embed the last two axes in a centered, zero-valued larger array."""

    array = np.asarray(values)
    target_y, target_x = _shape(target_shape, "center-pad target shape")
    source_y, source_x = array.shape[-2:]
    if target_y < source_y or target_x < source_x:
        msg = "center-pad target shape must contain the source shape."
        raise ValueError(msg)
    delta_y = target_y - source_y
    delta_x = target_x - source_x
    if delta_y % 2 or delta_x % 2:
        msg = "center-pad source and target grids must have aligned centers."
        raise ValueError(msg)
    before_y = delta_y // 2
    before_x = delta_x // 2
    pad_width = [(0, 0)] * array.ndim
    pad_width[-2] = (before_y, before_y)
    pad_width[-1] = (before_x, before_x)
    return np.pad(array, pad_width, mode="constant", constant_values=0)


def _r1_case_arrays_finite(case: Mapping[str, Any]) -> bool:
    return all(
        np.all(np.isfinite(np.asarray(case[name])))
        for name in ("U_A_exit", "P_B", "I_stack")
    )


def _r1_case_intensity_nonnegative(case: Mapping[str, Any]) -> bool:
    intensity = np.asarray(case["I_stack"])
    return bool(np.all(np.isfinite(intensity)) and np.all(intensity >= 0.0))


def _r1_status(
    *,
    hard_checks_pass: bool,
    refinement_pass: bool,
    external_pass: bool,
    visibility_pass: bool,
) -> str:
    """Apply the pre-registered independent R1 three-state status logic."""

    if not hard_checks_pass:
        return "Failed"
    if not refinement_pass or not external_pass or not visibility_pass:
        return "Inconclusive"
    return "Passed"


def _run_r1_diagnostics(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    convergence: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
    legacy_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the append-only exp040-R1 numerical diagnostics."""

    diagnostics = _section(config, "diagnostics_r1")
    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    sample_b_cfg = _section(config, "sample_b")
    acceptance = _section(config, "acceptance")
    baseline_shape = tuple(baseline["shape"])
    baseline_dx = float(baseline["dx_m"])
    baseline_dz = float(baseline["target_dz_m"])
    waist = float(sample_a["d_waist_m"])
    convergence_threshold = float(acceptance["convergence_relative_l2_max"])

    fine_base_b, working_b, working_dx, canonical_validation = (
        _make_r1_canonical_b(
            sample_b_cfg,
            diagnostics,
            legacy_canonical_b,
            legacy_canonical_dx_m,
        )
    )
    all_finite = bool(canonical_validation["all_finite"])
    all_intensity_nonnegative = True

    axial_cfg = _section(diagnostics, "refined_axial")
    new_axial = _simulate_case(
        config,
        shape=baseline_shape,
        dx_m=baseline_dx,
        dz_m=float(axial_cfg["new_dz_m"]),
        d_waist_m=waist,
        sample_b=np.asarray(baseline["B"], dtype=np.complex128),
        scan_positions=positions,
        keep_volume=False,
    )
    all_finite = all_finite and _r1_case_arrays_finite(new_axial)
    all_intensity_nonnegative = (
        all_intensity_nonnegative
        and _r1_case_intensity_nonnegative(new_axial)
    )
    axial_cases = [*convergence["axial"]["cases"], new_axial]
    axial_errors = _errors_to_finest_same_grid(axial_cases)
    axial_pair = [float(value) for value in axial_cfg["acceptance_pair_m"]]
    axial_test = _find_case(axial_cases, "target_dz_m", axial_pair[0])
    axial_reference = _find_case(axial_cases, "target_dz_m", axial_pair[1])
    axial_acceptance = _three_output_errors(axial_test, axial_reference)
    axial_result = {
        "x_values": np.asarray(
            [case["target_dz_m"] for case in axial_cases], dtype=np.float64
        ),
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in axial_errors.items()
        },
    }
    del axial_cases, new_axial
    gc.collect()

    lateral_cfg = _section(diagnostics, "refined_lateral")
    lateral_new_cfg = _section(lateral_cfg, "new_case")
    lateral_shape = _shape(lateral_new_cfg["shape"], "R1 lateral new shape")
    lateral_dx = _isotropic_dx(lateral_new_cfg["dx_m"], "R1 lateral new dx")
    _validate_scan_on_grid(positions, lateral_dx)
    lateral_b = _sample_r1_b(working_b, working_dx, lateral_shape, lateral_dx)
    new_lateral = _simulate_case(
        config,
        shape=lateral_shape,
        dx_m=lateral_dx,
        dz_m=baseline_dz,
        d_waist_m=waist,
        sample_b=lateral_b,
        scan_positions=positions,
        keep_volume=False,
    )
    all_finite = all_finite and _r1_case_arrays_finite(new_lateral)
    all_intensity_nonnegative = (
        all_intensity_nonnegative
        and _r1_case_intensity_nonnegative(new_lateral)
    )
    lateral_cases = [*convergence["lateral"]["cases"], new_lateral]
    lateral_errors = {name: [] for name in ("U_A_exit", "P_B", "I_stack")}
    for case in lateral_cases:
        mapped_reference = _map_case_to_grid(
            new_lateral, tuple(case["shape"]), float(case["dx_m"])
        )
        errors = _three_output_errors(case, mapped_reference)
        for name in lateral_errors:
            lateral_errors[name].append(errors[name])
        del mapped_reference
    lateral_pair = [
        float(value) for value in lateral_cfg["acceptance_pair_dx_m"]
    ]
    lateral_test = _find_case(lateral_cases, "dx_m", lateral_pair[0])
    lateral_test_reference = _map_case_to_grid(
        new_lateral,
        tuple(lateral_test["shape"]),
        float(lateral_test["dx_m"]),
    )
    lateral_acceptance = _three_output_errors(
        lateral_test, lateral_test_reference
    )
    lateral_result = {
        "x_values": np.asarray(
            [case["dx_m"] for case in lateral_cases], dtype=np.float64
        ),
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in lateral_errors.items()
        },
    }
    del (
        lateral_b,
        lateral_cases,
        lateral_test_reference,
        new_lateral,
    )
    gc.collect()

    fov_cfg = _section(diagnostics, "refined_fov")
    fov_dx = _isotropic_dx(fov_cfg["fixed_dx_m"], "R1 FOV dx")
    common_shape = _shape(
        fov_cfg["common_center_roi_shape"], "R1 FOV common ROI"
    )
    fov_common_cases: list[dict[str, Any]] = []
    for case in convergence["fov"]["cases"]:
        fov_common_cases.append(
            {"shape": tuple(case["shape"]), **_crop_case(case, common_shape)}
        )
    for shape_value in fov_cfg["new_shapes"]:
        shape = _shape(shape_value, "R1 FOV new shape")
        _validate_scan_on_grid(positions, fov_dx)
        sample_b = _sample_r1_b(working_b, working_dx, shape, fov_dx)
        case = _simulate_case(
            config,
            shape=shape,
            dx_m=fov_dx,
            dz_m=baseline_dz,
            d_waist_m=waist,
            sample_b=sample_b,
            scan_positions=positions,
            keep_volume=False,
        )
        all_finite = all_finite and _r1_case_arrays_finite(case)
        all_intensity_nonnegative = (
            all_intensity_nonnegative
            and _r1_case_intensity_nonnegative(case)
        )
        fov_common_cases.append(
            {"shape": shape, **_crop_case(case, common_shape)}
        )
        del case, sample_b
        gc.collect()
    fov_reference = fov_common_cases[-1]
    fov_errors = {name: [] for name in ("U_A_exit", "P_B", "I_stack")}
    for case in fov_common_cases:
        errors = _three_output_errors(case, fov_reference)
        for name in fov_errors:
            fov_errors[name].append(errors[name])
    fov_pair = [
        _shape(value, "R1 FOV acceptance shape")
        for value in fov_cfg["acceptance_pair_shapes"]
    ]
    fov_test = _find_case(fov_common_cases, "shape", fov_pair[0])
    fov_pair_reference = _find_case(fov_common_cases, "shape", fov_pair[1])
    fov_acceptance = _three_output_errors(fov_test, fov_pair_reference)
    fov_result = {
        "x_values": np.asarray(
            [case["shape"][1] * fov_dx for case in fov_common_cases],
            dtype=np.float64,
        ),
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in fov_errors.items()
        },
    }
    del fov_common_cases
    gc.collect()

    external_cfg = _section(diagnostics, "external_padding")
    external_dx = _isotropic_dx(external_cfg["fixed_dx_m"], "R1 external dx")
    external_common_shape = _shape(
        external_cfg["common_center_roi_shape"], "R1 external common ROI"
    )
    homogeneous_source = np.asarray(
        controls["homogeneous_reference"], dtype=np.complex128
    )
    residual = (
        np.asarray(baseline["U_A_exit"], dtype=np.complex128)
        - homogeneous_source
    )
    edge_pixels = _require_integer_ratio(
        float(external_cfg["edge_ring_width_m"]),
        external_dx,
        "R1 external edge-ring width",
    )
    if 2 * edge_pixels >= min(residual.shape):
        msg = "R1 external edge ring must leave a non-empty interior."
        raise ValueError(msg)
    edge_mask = np.ones(residual.shape, dtype=bool)
    edge_mask[
        edge_pixels:-edge_pixels,
        edge_pixels:-edge_pixels,
    ] = False
    residual_energy = float(np.sum(np.abs(residual) ** 2))
    edge_energy_fraction = float(
        np.sum(np.abs(residual[edge_mask]) ** 2)
        / max(residual_energy, np.finfo(np.float64).eps)
    )
    external_cases: list[dict[str, Any]] = []
    invariance_values: list[float] = []
    wavelength = float(optics["wavelength_m"])
    bandlimit = bool(optics["angular_spectrum_bandlimit"])
    for shape_value in external_cfg["padded_shapes"]:
        shape = _shape(shape_value, "R1 external padded shape")
        incident = make_plane_wave(
            shape,
            external_dx,
            wavelength,
            theta_x=float(_section(config, "illumination")["theta_x_rad"]),
            theta_y=float(_section(config, "illumination")["theta_y_rad"]),
            amplitude=float(_section(config, "illumination")["amplitude"]),
        )
        homogeneous_full = angular_spectrum_propagate(
            incident,
            external_dx,
            wavelength,
            float(sample_a["thickness_m"]),
            n=float(optics["internal_reference_index"]),
            bandlimit=bandlimit,
        )
        padded_exit = homogeneous_full + _center_pad(residual, shape)
        invariance_values.append(
            relative_l2(
                center_crop(padded_exit, baseline_shape),
                baseline["U_A_exit"],
            )
        )
        sample_b = _sample_r1_b(working_b, working_dx, shape, external_dx)
        intensity, probe, _, _ = simulate_exit_field_B_forward(
            padded_exit,
            sample_b,
            positions,
            external_dx,
            wavelength,
            float(optics["z_AB_m"]),
            float(optics["z_BC_m"]),
            external_medium_index=float(optics["external_medium_index"]),
            bandlimit=bandlimit,
            object_boundary=str(sample_b_cfg["object_boundary"]),
        )
        all_finite = bool(
            all_finite
            and np.all(np.isfinite(padded_exit))
            and np.all(np.isfinite(probe))
            and np.all(np.isfinite(intensity))
            and np.all(np.isfinite(sample_b))
        )
        all_intensity_nonnegative = bool(
            all_intensity_nonnegative and np.all(intensity >= 0.0)
        )
        external_cases.append(
            {
                "shape": shape,
                "P_B": center_crop(probe, external_common_shape).copy(),
                "I_stack": center_crop(
                    intensity, external_common_shape
                ).copy(),
            }
        )
        del homogeneous_full, incident, intensity, padded_exit, probe, sample_b
        gc.collect()
    external_reference = external_cases[-1]
    external_errors = {"P_B": [], "I_stack": []}
    for case in external_cases:
        for name in external_errors:
            external_errors[name].append(
                relative_l2(case[name], external_reference[name])
            )
    external_pair = [
        _shape(value, "R1 external acceptance shape")
        for value in external_cfg["acceptance_pair_shapes"]
    ]
    external_test = _find_case(external_cases, "shape", external_pair[0])
    external_pair_reference = _find_case(
        external_cases, "shape", external_pair[1]
    )
    external_acceptance = {
        name: relative_l2(external_test[name], external_pair_reference[name])
        for name in external_errors
    }
    padded_shapes = [tuple(case["shape"]) for case in external_cases]
    external_result = {
        "x_values": np.asarray(
            [shape[1] * external_dx for shape in padded_shapes],
            dtype=np.float64,
        ),
        "P_B": np.asarray(external_errors["P_B"], dtype=np.float64),
        "I_stack": np.asarray(
            external_errors["I_stack"], dtype=np.float64
        ),
        "U_A_exit_center_invariance": np.asarray(
            invariance_values, dtype=np.float64
        ),
    }
    del external_cases, fine_base_b, working_b
    gc.collect()

    refined_acceptance = {
        "axial": axial_acceptance,
        "lateral": lateral_acceptance,
        "fov": fov_acceptance,
    }
    refined_passes = {
        name: all(
            float(value) <= convergence_threshold
            for value in group_acceptance.values()
        )
        for name, group_acceptance in refined_acceptance.items()
    }
    refined_floor_components = {
        output_name: {
            domain_name: float(group_acceptance[output_name])
            for domain_name, group_acceptance in refined_acceptance.items()
        }
        for output_name in ("U_A_exit", "P_B", "I_stack")
    }
    refined_floors = {
        output_name: max(domain_values.values())
        for output_name, domain_values in refined_floor_components.items()
    }
    refined_detector_floor = refined_floors["I_stack"]
    legacy_detector_signal = _section(
        _section(legacy_metrics, "visibility")["signals"], "I_stack"
    )
    detector_minus_signal = float(
        legacy_detector_signal["waist_minus_relative_l2"]
    )
    detector_plus_signal = float(
        legacy_detector_signal["waist_plus_relative_l2"]
    )
    visibility_ratio = min(detector_minus_signal, detector_plus_signal) / max(
        refined_detector_floor, np.finfo(np.float64).eps
    )
    visibility_threshold = float(
        acceptance["detector_visibility_signal_to_floor_min"]
    )
    visibility_pass = bool(visibility_ratio >= visibility_threshold)
    invariance_tolerance = float(
        external_cfg["require_a_exit_center_invariance_max"]
    )
    external_propagation_pass = all(
        float(value) <= convergence_threshold
        for value in external_acceptance.values()
    )
    external_invariance_pass = bool(
        max(invariance_values) <= invariance_tolerance
    )
    external_pass = bool(
        external_propagation_pass and external_invariance_pass
    )
    legacy_hard_checks_pass = bool(
        _section(legacy_metrics, "stage_status")["hard_controls_pass"]
    )
    hard_checks_pass = bool(
        legacy_hard_checks_pass
        and canonical_validation["pass"]
        and all_finite
        and all_intensity_nonnegative
    )
    refinement_pass = all(refined_passes.values())
    status = _r1_status(
        hard_checks_pass=hard_checks_pass,
        refinement_pass=refinement_pass,
        external_pass=external_pass,
        visibility_pass=visibility_pass,
    )

    thresholds = {
        "convergence_relative_l2_max": convergence_threshold,
        "determinism_relative_l2_max": float(
            acceptance["determinism_relative_l2_max"]
        ),
        "detector_visibility_signal_to_floor_min": visibility_threshold,
        "a_exit_center_invariance_max": invariance_tolerance,
        "canonical_b_r0_mapping_max_complex_error": float(
            canonical_validation["r0_mapping_max_complex_error"]
        ),
    }
    refined_metrics = {
        "axial": {
            "acceptance_pair_m": np.asarray(axial_pair, dtype=np.float64),
            "acceptance": axial_acceptance,
            "pass": bool(refined_passes["axial"]),
            **axial_result,
        },
        "lateral": {
            "acceptance_pair_dx_m": np.asarray(
                lateral_pair, dtype=np.float64
            ),
            "acceptance": lateral_acceptance,
            "pass": bool(refined_passes["lateral"]),
            **lateral_result,
        },
        "fov": {
            "acceptance_pair_shapes": np.asarray(fov_pair, dtype=np.int64),
            "acceptance": fov_acceptance,
            "pass": bool(refined_passes["fov"]),
            **fov_result,
        },
    }
    external_metrics = {
        "acceptance_pair_shapes": np.asarray(
            external_pair, dtype=np.int64
        ),
        "acceptance": external_acceptance,
        "a_exit_center_invariance_max": float(max(invariance_values)),
        "a_exit_center_invariance_pass": external_invariance_pass,
        "residual_edge_energy_fraction": edge_energy_fraction,
        "propagation_pass": bool(external_propagation_pass),
        "pass": external_pass,
        **external_result,
    }
    visibility_report = {
        "waist_minus_detector_signal": detector_minus_signal,
        "waist_plus_detector_signal": detector_plus_signal,
        "refined_detector_floor": refined_detector_floor,
        "detector_signal_to_floor_min": visibility_ratio,
        "detector_signal_to_floor_threshold": visibility_threshold,
        "pass": visibility_pass,
    }
    result = {
        "refined_convergence": {
            "axial": axial_result,
            "lateral": lateral_result,
            "fov": fov_result,
        },
        "external_padding": external_result,
    }
    metrics = {
        "version": str(diagnostics["version"]),
        "methods": dict(_section(diagnostics, "methods")),
        "canonical_b_validation": canonical_validation,
        "refined_convergence": refined_metrics,
        "external_padding": external_metrics,
        "refined_floor": {
            "components": refined_floor_components,
            **refined_floors,
        },
        "visibility_report": visibility_report,
        "thresholds": thresholds,
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_intensity_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    return result, metrics


def _center_periodic_extend(
    base_period: NDArray[np.complex128],
    target_shape: tuple[int, int],
) -> NDArray[np.complex128]:
    """Extend one centered periodic realization without new random draws."""

    base = np.asarray(base_period, dtype=np.complex128)
    target_y, target_x = _shape(target_shape, "periodic target shape")
    delta_y = target_y - base.shape[0]
    delta_x = target_x - base.shape[1]
    if delta_y < 0 or delta_x < 0 or delta_y % 2 or delta_x % 2:
        msg = "periodic target must be a center-aligned superset of base."
        raise ValueError(msg)
    extended = np.pad(
        base,
        ((delta_y // 2, delta_y // 2), (delta_x // 2, delta_x // 2)),
        mode="wrap",
    )
    return np.asarray(extended, dtype=np.complex128)


def _r2_padded_exit(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    shape: tuple[int, int],
    dx_m: float,
) -> NDArray[np.complex128]:
    """Build the registered homogeneous-reference plus residual A-exit."""

    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    illumination = _section(config, "illumination")
    wavelength = float(optics["wavelength_m"])
    residual = np.asarray(baseline["U_A_exit"], dtype=np.complex128) - np.asarray(
        controls["homogeneous_reference"], dtype=np.complex128
    )
    incident = make_plane_wave(
        shape,
        dx_m,
        wavelength,
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    homogeneous = angular_spectrum_propagate(
        incident,
        dx_m,
        wavelength,
        float(sample_a["thickness_m"]),
        n=float(optics["internal_reference_index"]),
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=False,
    )
    return np.asarray(
        homogeneous + _center_pad(residual, shape), dtype=np.complex128
    )


def _r2_external_forward(
    config: Mapping[str, Any],
    padded_exit: NDArray[np.complex128],
    sample_b: NDArray[np.complex128],
    positions: NDArray[np.float64],
    dx_m: float,
    common_shape: tuple[int, int],
    *,
    alias_control: bool,
) -> tuple[dict[str, Any], bool, bool]:
    """Run one R2 external case and retain only its registered common ROI."""

    optics = _section(config, "optics")
    sample_b_cfg = _section(config, "sample_b")
    intensity, probe, _, _ = simulate_exit_field_B_forward(
        padded_exit,
        sample_b,
        positions,
        dx_m,
        float(optics["wavelength_m"]),
        float(optics["z_AB_m"]),
        float(optics["z_BC_m"]),
        external_medium_index=float(optics["external_medium_index"]),
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=alias_control,
        object_boundary=str(sample_b_cfg["object_boundary"]),
    )
    finite = bool(
        np.all(np.isfinite(padded_exit))
        and np.all(np.isfinite(sample_b))
        and np.all(np.isfinite(probe))
        and np.all(np.isfinite(intensity))
    )
    nonnegative = bool(np.all(intensity >= 0.0))
    case = {
        "shape": tuple(padded_exit.shape),
        "P_B": center_crop(probe, common_shape).copy(),
        "I_stack": center_crop(intensity, common_shape).copy(),
    }
    del intensity, probe
    return case, finite, nonnegative


def _r2_method_convergence(
    cases: Sequence[Mapping[str, Any]],
    pair: Sequence[tuple[int, int]],
    threshold: float,
) -> dict[str, Any]:
    """Assemble relative-to-largest series and final-pair metrics."""

    reference = cases[-1]
    errors = {"P_B": [], "I_stack": []}
    for case in cases:
        for name in errors:
            errors[name].append(relative_l2(case[name], reference[name]))
    test = _find_case(cases, "shape", pair[0])
    pair_reference = _find_case(cases, "shape", pair[1])
    acceptance = {
        name: relative_l2(test[name], pair_reference[name]) for name in errors
    }
    passed = all(float(value) <= threshold for value in acceptance.values())
    return {
        "P_B": np.asarray(errors["P_B"], dtype=np.float64),
        "I_stack": np.asarray(errors["I_stack"], dtype=np.float64),
        "acceptance": acceptance,
        "pass": bool(passed),
    }


def _r2_mask_controls(
    shape: tuple[int, int],
    dx_m: float,
    wavelength: float,
    z_m: float,
    refractive_index: float,
) -> dict[str, Any]:
    """Measure the registered exact-mask transfer invariants."""

    mask = make_transfer_sampling_alias_mask(
        shape, dx_m, wavelength, z_m, n=refractive_index
    )
    current = make_angular_spectrum_transfer(
        shape,
        dx_m,
        wavelength,
        z_m,
        n=refractive_index,
        bandlimit=True,
        alias_control=False,
    )
    controlled = make_angular_spectrum_transfer(
        shape,
        dx_m,
        wavelength,
        z_m,
        n=refractive_index,
        bandlimit=True,
        alias_control=True,
    )
    inside_error = float(
        np.max(np.abs(controlled[mask] - current[mask]), initial=0.0)
    )
    outside_nonzero = int(np.count_nonzero(controlled[~mask]))
    result = {
        "kept_bin_fraction": float(np.count_nonzero(mask) / mask.size),
        "inside_transfer_max_complex_error": inside_error,
        "outside_nonzero_bins": outside_nonzero,
        "dc_preserved": bool(mask[0, 0] and controlled[0, 0] == current[0, 0]),
    }
    result["pass"] = bool(
        result["dc_preserved"]
        and inside_error == 0.0
        and outside_nonzero == 0
        and 0.0 < result["kept_bin_fraction"] <= 1.0
    )
    return result


def _r2_interpretation_code(
    current_pass: bool,
    alias_pass: bool,
    method_material: bool,
) -> str:
    if current_pass and alias_pass:
        return (
            "period_aligned_but_method_dependent"
            if method_material
            else "period_aligned_fov_supported"
        )
    if not current_pass and alias_pass and method_material:
        return "transfer_sampling_alias_supported"
    if not current_pass and not alias_pass:
        return "remaining_downstream_floor"
    if current_pass and not alias_pass:
        return "alias_control_method_conflict"
    return "ambiguous_method_effect"


def _run_r2_diagnostics(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
    r1_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the append-only exp040-R2 boundary/alias diagnostics."""

    diagnostics = _section(config, "diagnostics_r2")
    diagnostics_r1 = _section(config, "diagnostics_r1")
    period_cfg = _section(diagnostics, "period_commensurate")
    optics = _section(config, "optics")
    acceptance_cfg = _section(config, "acceptance")
    dx_m = _isotropic_dx(period_cfg["fixed_dx_m"], "R2 external dx")
    shapes = [_shape(value, "R2 period shape") for value in period_cfg["shapes"]]
    pair = [
        _shape(value, "R2 acceptance shape")
        for value in period_cfg["acceptance_pair_shapes"]
    ]
    common_shape = _shape(
        period_cfg["common_center_roi_shape"], "R2 common ROI"
    )
    convergence_threshold = float(
        acceptance_cfg["convergence_relative_l2_max"]
    )
    determinism_threshold = float(
        acceptance_cfg["determinism_relative_l2_max"]
    )
    invariance_threshold = float(
        _section(diagnostics_r1, "external_padding")[
            "require_a_exit_center_invariance_max"
        ]
    )
    mapping_threshold = float(
        _section(diagnostics_r1, "sample_b_refinement")[
            "r0_mapping_max_complex_error"
        ]
    )

    fine_base, _, fine_dx, r1_canonical_validation = _make_r1_canonical_b(
        _section(config, "sample_b"),
        diagnostics_r1,
        legacy_canonical_b,
        legacy_canonical_dx_m,
    )
    base_period_shape = _shape(
        period_cfg["base_period_shape"], "R2 base-period shape"
    )
    base_period = resample_centered_grid(
        fine_base, fine_dx, base_period_shape, dx_m
    )
    legacy_period = resample_centered_grid(
        legacy_canonical_b,
        legacy_canonical_dx_m,
        base_period_shape,
        dx_m,
    )
    base_mapping_error = float(np.max(np.abs(base_period - legacy_period)))
    extended_sample_b: list[NDArray[np.complex128]] = []
    center_errors: list[float] = []
    modulus_errors: list[float] = []
    for shape in shapes:
        extended = _center_periodic_extend(base_period, shape)
        extended_sample_b.append(extended)
        center_errors.append(
            float(
                np.max(
                    np.abs(
                        center_crop(extended, base_period_shape) - base_period
                    )
                )
            )
        )
        modulus_errors.append(float(np.max(np.abs(np.abs(extended) - 1.0))))
    canonical_validation = {
        "base_period_shape": np.asarray(base_period_shape, dtype=np.int64),
        "target_shapes": np.asarray(shapes, dtype=np.int64),
        "dx_m": dx_m,
        "base_to_r0_max_complex_error": base_mapping_error,
        "mapping_max_complex_error_threshold": mapping_threshold,
        "center_period_max_complex_error": float(max(center_errors)),
        "unit_modulus_max_abs_error": float(max(modulus_errors)),
        "same_r1_canonical_validation_pass": bool(
            r1_canonical_validation["pass"]
        ),
        "all_finite": bool(
            np.all(np.isfinite(base_period))
            and all(np.all(np.isfinite(value)) for value in extended_sample_b)
        ),
    }
    canonical_validation["pass"] = bool(
        canonical_validation["same_r1_canonical_validation_pass"]
        and canonical_validation["all_finite"]
        and base_mapping_error <= mapping_threshold
        and canonical_validation["center_period_max_complex_error"] == 0.0
        and canonical_validation["unit_modulus_max_abs_error"]
        <= 32.0 * np.finfo(np.float64).eps
    )
    del fine_base, legacy_period

    cases: dict[str, list[dict[str, Any]]] = {
        "current_asm": [],
        "alias_controlled": [],
    }
    invariance_values: list[float] = []
    method_differences = {"P_B": [], "I_stack": []}
    alias_mask_metrics = {"AB": [], "BC": []}
    all_finite = bool(canonical_validation["all_finite"])
    all_intensity_nonnegative = True
    wavelength = float(optics["wavelength_m"])
    external_index = float(optics["external_medium_index"])
    for shape, sample_b in zip(shapes, extended_sample_b, strict=True):
        padded_exit = _r2_padded_exit(
            config, baseline, controls, shape, dx_m
        )
        invariance_values.append(
            relative_l2(
                center_crop(padded_exit, tuple(baseline["shape"])),
                baseline["U_A_exit"],
            )
        )
        current, current_finite, current_nonnegative = _r2_external_forward(
            config,
            padded_exit,
            sample_b,
            positions,
            dx_m,
            common_shape,
            alias_control=False,
        )
        controlled, controlled_finite, controlled_nonnegative = (
            _r2_external_forward(
                config,
                padded_exit,
                sample_b,
                positions,
                dx_m,
                common_shape,
                alias_control=True,
            )
        )
        cases["current_asm"].append(current)
        cases["alias_controlled"].append(controlled)
        for name in method_differences:
            method_differences[name].append(
                relative_l2(current[name], controlled[name])
            )
        all_finite = bool(
            all_finite and current_finite and controlled_finite
        )
        all_intensity_nonnegative = bool(
            all_intensity_nonnegative
            and current_nonnegative
            and controlled_nonnegative
        )
        alias_mask_metrics["AB"].append(
            _r2_mask_controls(
                shape,
                dx_m,
                wavelength,
                float(optics["z_AB_m"]),
                external_index,
            )
        )
        alias_mask_metrics["BC"].append(
            _r2_mask_controls(
                shape,
                dx_m,
                wavelength,
                float(optics["z_BC_m"]),
                external_index,
            )
        )
        del padded_exit
        gc.collect()

    current_metrics = _r2_method_convergence(
        cases["current_asm"], pair, convergence_threshold
    )
    controlled_metrics = _r2_method_convergence(
        cases["alias_controlled"], pair, convergence_threshold
    )
    largest_method_difference = {
        name: float(values[-1]) for name, values in method_differences.items()
    }
    method_material = any(
        value > convergence_threshold
        for value in largest_method_difference.values()
    )

    largest_shape = shapes[-1]
    repeated_exit = _r2_padded_exit(
        config, baseline, controls, largest_shape, dx_m
    )
    repeated_case, repeated_finite, repeated_nonnegative = _r2_external_forward(
        config,
        repeated_exit,
        extended_sample_b[-1],
        positions,
        dx_m,
        common_shape,
        alias_control=True,
    )
    deterministic_reference = cases["alias_controlled"][-1]
    determinism_errors = {
        name: relative_l2(repeated_case[name], deterministic_reference[name])
        for name in ("P_B", "I_stack")
    }
    determinism_pass = all(
        value <= determinism_threshold for value in determinism_errors.values()
    )
    all_finite = bool(all_finite and repeated_finite)
    all_intensity_nonnegative = bool(
        all_intensity_nonnegative and repeated_nonnegative
    )
    invariance_max = float(max(invariance_values))
    invariance_pass = bool(invariance_max <= invariance_threshold)
    mask_controls_pass = all(
        bool(item["pass"])
        for stage_values in alias_mask_metrics.values()
        for item in stage_values
    )
    hard_checks_pass = bool(
        canonical_validation["pass"]
        and invariance_pass
        and mask_controls_pass
        and determinism_pass
        and all_finite
        and all_intensity_nonnegative
        and _section(r1_metrics, "canonical_b_validation")["pass"]
    )
    alias_convergence_pass = bool(controlled_metrics["pass"])
    if not hard_checks_pass:
        status = "Failed"
    elif alias_convergence_pass:
        status = "Passed"
    else:
        status = "Inconclusive"
    current_pass = bool(current_metrics["pass"])
    interpretation_code = _r2_interpretation_code(
        current_pass, alias_convergence_pass, method_material
    )

    x_values = np.asarray(period_cfg["fov_m"], dtype=np.float64)
    common = {
        "x_values_m": x_values,
        "shapes": np.asarray(shapes, dtype=np.int64),
        "acceptance_pair_shapes": np.asarray(pair, dtype=np.int64),
    }
    current_metrics = {**common, **current_metrics}
    controlled_metrics = {**common, **controlled_metrics}
    method_difference_metrics = {
        "x_values_m": x_values,
        "P_B": np.asarray(method_differences["P_B"], dtype=np.float64),
        "I_stack": np.asarray(
            method_differences["I_stack"], dtype=np.float64
        ),
        "largest_fov": largest_method_difference,
        "material": bool(method_material),
        "reference_method": "alias_controlled",
    }
    mask_metrics = {
        "x_values_m": x_values,
        "AB_kept_bin_fraction": np.asarray(
            [value["kept_bin_fraction"] for value in alias_mask_metrics["AB"]],
            dtype=np.float64,
        ),
        "BC_kept_bin_fraction": np.asarray(
            [value["kept_bin_fraction"] for value in alias_mask_metrics["BC"]],
            dtype=np.float64,
        ),
        "inside_transfer_max_complex_error": float(
            max(
                value["inside_transfer_max_complex_error"]
                for stage_values in alias_mask_metrics.values()
                for value in stage_values
            )
        ),
        "outside_nonzero_bins": int(
            sum(
                value["outside_nonzero_bins"]
                for stage_values in alias_mask_metrics.values()
                for value in stage_values
            )
        ),
        "all_dc_preserved": bool(
            all(
                value["dc_preserved"]
                for stage_values in alias_mask_metrics.values()
                for value in stage_values
            )
        ),
        "pass": bool(mask_controls_pass),
    }
    thresholds = {
        "convergence_relative_l2_max": convergence_threshold,
        "method_difference_material_relative_l2_min": convergence_threshold,
        "a_exit_center_invariance_max": invariance_threshold,
        "canonical_b_mapping_max_complex_error": mapping_threshold,
        "determinism_relative_l2_max": determinism_threshold,
    }
    outcome_flags = {
        "r1_external_padding_pass": bool(
            _section(r1_metrics, "external_padding")["pass"]
        ),
        "period_aligned_current_asm_pass": current_pass,
        "period_aligned_alias_controlled_pass": alias_convergence_pass,
        "alias_method_difference_material_at_largest_fov": bool(
            method_material
        ),
        "interpretation_code": interpretation_code,
    }
    r1_comparator = {
        "acceptance_pair_shapes": np.asarray(
            _section(r1_metrics, "external_padding")[
                "acceptance_pair_shapes"
            ],
            dtype=np.int64,
        ),
        "acceptance": dict(
            _section(_section(r1_metrics, "external_padding"), "acceptance")
        ),
    }
    result = {
        "period_aligned": {
            "current_asm": current_metrics,
            "alias_controlled": controlled_metrics,
        },
        "method_difference": method_difference_metrics,
        "alias_masks": mask_metrics,
    }
    metrics = {
        "version": str(diagnostics["version"]),
        "methods": dict(_section(diagnostics, "methods")),
        "canonical_b_validation": canonical_validation,
        "a_exit_center_invariance": {
            "x_values_m": x_values,
            "relative_l2": np.asarray(
                invariance_values, dtype=np.float64
            ),
            "max": invariance_max,
            "pass": invariance_pass,
        },
        "period_aligned": result["period_aligned"],
        "method_difference": method_difference_metrics,
        "alias_masks": mask_metrics,
        "determinism": {
            **determinism_errors,
            "pass": bool(determinism_pass),
        },
        "r1_external_comparator": r1_comparator,
        "thresholds": thresholds,
        "outcome_flags": outcome_flags,
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_intensity_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    del (
        base_period,
        cases,
        extended_sample_b,
        repeated_case,
        repeated_exit,
    )
    gc.collect()
    return result, metrics


def _r3_native_indices(length: int, factor: int, offset: int) -> NDArray[np.int64]:
    """Return the R3.8 integer indices for one native detector axis."""

    if factor <= 0 or offset < 0 or offset >= factor or length % factor:
        msg = "Invalid R3 native-sampling length, factor, or offset."
        raise ValueError(msg)
    indices = offset + factor * np.arange(length // factor, dtype=np.int64)
    if int(indices[-1]) >= length:
        msg = "R3 native-sampling indices exceed the fine grid."
        raise ValueError(msg)
    return indices


def _r3_native_sample(
    values: NDArray[np.generic], factor: int, offset: int
) -> NDArray[Any]:
    """Sample the last two axes at the registered physical native centers."""

    array = np.asarray(values)
    if array.ndim < 2:
        msg = "R3 native sampling requires at least two dimensions."
        raise ValueError(msg)
    y = _r3_native_indices(array.shape[-2], factor, offset)
    x = _r3_native_indices(array.shape[-1], factor, offset)
    return np.take(np.take(array, y, axis=-2), x, axis=-1)


def _r3_aligned_bilinear_upsample(
    values: NDArray[np.generic], factor: int, offset: int
) -> NDArray[Any]:
    """Bilinearly refine a 2D field while preserving registered native samples."""

    array = np.asarray(values)
    if array.ndim != 2:
        msg = "R3 aligned upsampling requires a 2D array."
        raise ValueError(msg)
    target_shape = (array.shape[0] * factor, array.shape[1] * factor)
    y = (np.arange(target_shape[0], dtype=np.float64) - offset) / factor
    x = (np.arange(target_shape[1], dtype=np.float64) - offset) / factor
    yy, xx = np.meshgrid(y, x, indexing="ij")
    coordinates = np.asarray([yy, xx], dtype=np.float64)

    if np.iscomplexobj(array):
        refined = map_coordinates(
            np.asarray(array.real, dtype=np.float64),
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        ) + 1j * map_coordinates(
            np.asarray(array.imag, dtype=np.float64),
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        return np.asarray(refined, dtype=np.complex128)
    return np.asarray(
        map_coordinates(
            np.asarray(array, dtype=np.float64),
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        ),
        dtype=np.float64,
    )


def _r3_energy_fraction(
    spectrum: NDArray[np.complexfloating], mask: NDArray[np.bool_]
) -> float:
    """Return the Parseval-equivalent spectrum energy outside ``mask``."""

    values = np.asarray(spectrum, dtype=np.complex128)
    keep = np.asarray(mask, dtype=np.bool_)
    if values.shape != keep.shape:
        msg = "R3 spectrum and mask shapes must match."
        raise ValueError(msg)
    energy = np.abs(values) ** 2
    return float(
        np.sum(energy[~keep])
        / max(float(np.sum(energy)), np.finfo(float).eps)
    )


def _r3_native_nyquist_mask(
    shape: tuple[int, int], dx_m: float, detector_pixel_m: float
) -> NDArray[np.bool_]:
    """Return frequencies observable within the native detector Nyquist box."""

    fy = np.fft.fftfreq(shape[0], d=dx_m)
    fx = np.fft.fftfreq(shape[1], d=dx_m)
    limit = 1.0 / (2.0 * detector_pixel_m)
    tolerance = 32.0 * np.finfo(np.float64).eps * limit
    return np.asarray(
        (np.abs(fy)[:, None] <= limit + tolerance)
        & (np.abs(fx)[None, :] <= limit + tolerance),
        dtype=np.bool_,
    )


def _r3_pixel_average_from_spectrum(
    intensity: NDArray[np.float64],
    spectrum: NDArray[np.complex128],
    mtf: NDArray[np.float64],
    tolerance: float,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Apply the registered pixel MTF and expose all numerical controls."""

    filtered = np.fft.ifft2(spectrum * mtf)
    real = np.asarray(filtered.real, dtype=np.float64)
    scale = max(float(np.max(np.abs(intensity))), np.finfo(np.float64).eps)
    imaginary_leak = float(np.max(np.abs(filtered.imag)) / scale)
    negative_relative = float(max(0.0, -float(np.min(real))) / scale)
    sum_relative_error = float(
        abs(float(np.sum(real)) - float(np.sum(intensity)))
        / max(abs(float(np.sum(intensity))), np.finfo(np.float64).eps)
    )
    if negative_relative <= tolerance:
        real = np.maximum(real, 0.0)
    return real, {
        "imaginary_relative_leak": imaginary_leak,
        "negative_relative_scale": negative_relative,
        "sum_relative_error": sum_relative_error,
    }


def _r3_synthetic_pixel_controls(
    dx_m: float, detector_pixel_m: float
) -> dict[str, float]:
    """Check constant preservation and FFT-origin shift equivariance."""

    shape = (32, 32)
    mtf = make_square_pixel_mtf(shape, dx_m, detector_pixel_m)
    constant = np.ones(shape, dtype=np.float64)
    constant_filtered = np.fft.ifft2(np.fft.fft2(constant) * mtf)
    constant_error = float(np.max(np.abs(constant_filtered - 1.0)))

    impulse = np.zeros(shape, dtype=np.float64)
    impulse[7, 9] = 1.0
    filtered = np.fft.ifft2(np.fft.fft2(impulse) * mtf)
    shift = (5, 3)
    shifted = np.roll(impulse, shift, axis=(0, 1))
    shifted_filtered = np.fft.ifft2(np.fft.fft2(shifted) * mtf)
    alignment_error = relative_l2(
        shifted_filtered,
        np.roll(filtered, shift, axis=(0, 1)),
    )
    sum_error = float(
        abs(float(np.sum(filtered.real)) - 1.0)
    )
    imaginary_leak = float(np.max(np.abs(filtered.imag)))
    return {
        "constant_max_abs_error": constant_error,
        "center_alignment_relative_l2": alignment_error,
        "impulse_sum_absolute_error": sum_error,
        "imaginary_absolute_leak": imaginary_leak,
    }


def _r3_padded_exit(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    shape: tuple[int, int],
    dx_m: float,
    factor: int,
    offset: int,
) -> NDArray[np.complex128]:
    """Map the fixed A-exit residual onto one R3 external grid."""

    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    illumination = _section(config, "illumination")
    residual = np.asarray(baseline["U_A_exit"], dtype=np.complex128) - np.asarray(
        controls["homogeneous_reference"], dtype=np.complex128
    )
    refined_residual = _r3_aligned_bilinear_upsample(residual, factor, offset)
    incident = make_plane_wave(
        shape,
        dx_m,
        float(optics["wavelength_m"]),
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    homogeneous = angular_spectrum_propagate(
        incident,
        dx_m,
        float(optics["wavelength_m"]),
        float(sample_a["thickness_m"]),
        n=float(optics["internal_reference_index"]),
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=False,
    )
    return np.asarray(
        homogeneous + _center_pad(refined_residual, shape), dtype=np.complex128
    )


def _r3_convergence_metrics(
    cases: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    """Assemble R3 relative-to-factor-4 and factor-2-to-4 errors."""

    reference = cases[-1]
    p_errors = np.asarray(
        [relative_l2(case["P_B"], reference["P_B"]) for case in cases],
        dtype=np.float64,
    )
    detector_errors: dict[str, dict[str, NDArray[np.float64]]] = {}
    detector_acceptance: dict[str, dict[str, float]] = {}
    for method in ("current_asm", "alias_controlled"):
        detector_errors[method] = {}
        detector_acceptance[method] = {}
        for branch in ("point_sample", "pixel_box_average"):
            detector_errors[method][branch] = np.asarray(
                [
                    relative_l2(
                        _section(_section(case, "detector"), method)[branch],
                        _section(_section(reference, "detector"), method)[branch],
                    )
                    for case in cases
                ],
                dtype=np.float64,
            )
            detector_acceptance[method][branch] = relative_l2(
                _section(_section(cases[-2], "detector"), method)[branch],
                _section(_section(cases[-1], "detector"), method)[branch],
            )
    p_acceptance = relative_l2(cases[-2]["P_B"], cases[-1]["P_B"])
    primary_acceptance = detector_acceptance["alias_controlled"][
        "pixel_box_average"
    ]
    return {
        "factors": np.asarray([case["factor"] for case in cases], dtype=np.int64),
        "dx_m": np.asarray([case["dx_m"] for case in cases], dtype=np.float64),
        "shapes": np.asarray([case["shape"] for case in cases], dtype=np.int64),
        "acceptance_pair_factors": np.asarray([2, 4], dtype=np.int64),
        "relative_to_factor4": {
            "P_B": p_errors,
            "detector": detector_errors,
        },
        "acceptance": {
            "P_B": p_acceptance,
            "detector": detector_acceptance,
        },
        "P_B_pass": bool(p_acceptance <= threshold),
        "primary_detector_pass": bool(primary_acceptance <= threshold),
    }


def _r3_outcome_code(
    *,
    hard_checks_pass: bool,
    probe_pass: bool,
    primary_pass: bool,
    point_pass: bool,
    point_vs_pixel_material: bool,
    b_exit_or_bc_material: bool,
) -> str:
    """Apply the frozen R3.6 interpretation table."""

    if not hard_checks_pass:
        return "hard_control_failure"
    if not probe_pass:
        return "upstream_sampling_not_converged"
    if primary_pass:
        if not point_pass and point_vs_pixel_material:
            return "point_detector_model_defect_supported"
        return "detector_path_sampling_converged"
    if b_exit_or_bc_material:
        return "finite_pixel_does_not_resolve_b_bc_floor"
    return "boundary_or_higher_physics_priority"


def _run_r3_diagnostics(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
    r2_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the append-only exp040-R3 streaming detector-path diagnostic."""

    diagnostics = _section(config, "diagnostics_r3")
    diagnostics_r1 = _section(config, "diagnostics_r1")
    sampling = _section(diagnostics, "sampling")
    optics = _section(config, "optics")
    detector_cfg = _section(optics, "detector")
    acceptance = _section(config, "acceptance")
    factors = [int(value) for value in sampling["factors"]]
    dx_values = [float(value) for value in sampling["dx_m"]]
    shapes = [_shape(value, "R3 shape") for value in sampling["shapes"]]
    offsets = [int(value) for value in sampling["native_sample_offsets_px"]]
    compensation = np.asarray(
        sampling["physical_origin_compensation_m"], dtype=np.float64
    )
    native_roi = _shape(sampling["native_roi_shape"], "R3 native ROI")
    base_period_native_shape = (
        shapes[0][0] // int(sampling["canonical_period_count"]),
        shapes[0][1] // int(sampling["canonical_period_count"]),
    )
    detector_pixel = float(detector_cfg["pixel_size_m"])
    wavelength = float(optics["wavelength_m"])
    external_index = float(optics["external_medium_index"])
    z_ab = float(optics["z_AB_m"])
    z_bc = float(optics["z_BC_m"])
    bandlimit = bool(optics["angular_spectrum_bandlimit"])
    convergence_threshold = float(acceptance["convergence_relative_l2_max"])
    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    determinism_threshold = float(acceptance["determinism_relative_l2_max"])
    selected_factor = int(
        _section(diagnostics, "detector_sampling")["selected_factor"]
    )
    selected_scan = int(
        _section(diagnostics, "detector_sampling")["selected_scan_index"]
    )

    fine_base, _, fine_dx, fine_validation = _make_r1_canonical_b(
        _section(config, "sample_b"),
        diagnostics_r1,
        legacy_canonical_b,
        legacy_canonical_dx_m,
    )
    max_factor = max(factors)
    base_periods: list[NDArray[np.complex128]] = []
    for factor in factors:
        stride = max_factor // factor
        source_offset = (stride - 1) // 2
        base = np.asarray(
            fine_base[source_offset::stride, source_offset::stride],
            dtype=np.complex128,
        )
        expected = (
            base_period_native_shape[0] * factor,
            base_period_native_shape[1] * factor,
        )
        if base.shape != expected:
            msg = "R3 canonical-B refinement produced an unexpected shape."
            raise RuntimeError(msg)
        base_periods.append(base)
    legacy_period = resample_centered_grid(
        legacy_canonical_b,
        legacy_canonical_dx_m,
        base_period_native_shape,
        dx_values[0],
    )
    base_to_r0_error = float(np.max(np.abs(base_periods[0] - legacy_period)))
    fine_reference_error = float(np.max(np.abs(base_periods[-1] - fine_base)))

    cases: list[dict[str, Any]] = []
    a_exit_errors: list[float] = []
    b_mapping_errors: list[float] = []
    b_exit_outside_mask_mean: list[float] = []
    b_exit_outside_mask_max: list[float] = []
    b_exit_outside_native_mean: list[float] = []
    b_exit_outside_native_max: list[float] = []
    detector_spectrum: dict[str, dict[str, list[float]]] = {
        method: {"mean": [], "max": []}
        for method in ("current_asm", "alias_controlled")
    }
    bc_field_difference: list[float] = []
    bc_intensity_difference: list[float] = []
    point_vs_pixel: dict[str, list[float]] = {
        "current_asm": [],
        "alias_controlled": [],
    }
    pixel_actual_controls = {
        "sum_relative_error": [],
        "imaginary_relative_leak": [],
        "negative_relative_scale": [],
    }
    synthetic_controls: list[dict[str, float]] = []
    mask_controls = {"AB": [], "BC": []}
    all_finite = bool(fine_validation["all_finite"])
    all_final_nonnegative = True
    determinism_error = float("nan")
    selected_images: dict[str, NDArray[np.float64]] = {}

    for factor, dx_m, shape, offset, base_period in zip(
        factors, dx_values, shapes, offsets, base_periods, strict=True
    ):
        padded_exit = _r3_padded_exit(
            config, baseline, controls, shape, dx_m, factor, offset
        )
        native_exit = _r3_native_sample(padded_exit, factor, offset)
        a_exit_errors.append(
            relative_l2(
                center_crop(native_exit, tuple(baseline["shape"])),
                baseline["U_A_exit"],
            )
        )
        sample_b = _center_periodic_extend(base_period, shape)
        native_b_period = center_crop(
            _r3_native_sample(sample_b, factor, offset),
            base_period_native_shape,
        )
        b_mapping_errors.append(
            float(np.max(np.abs(native_b_period - base_periods[0])))
        )

        transfer_ab = make_angular_spectrum_transfer(
            shape,
            dx_m,
            wavelength,
            z_ab,
            n=external_index,
            bandlimit=bandlimit,
            alias_control=True,
        )
        probe = apply_angular_spectrum_transfer(padded_exit, transfer_ab)
        del transfer_ab
        probe_roi = center_crop(
            _r3_native_sample(probe, factor, offset), native_roi
        ).copy()
        transfer_current = make_angular_spectrum_transfer(
            shape,
            dx_m,
            wavelength,
            z_bc,
            n=external_index,
            bandlimit=bandlimit,
            alias_control=False,
        )
        transfer_alias = make_angular_spectrum_transfer(
            shape,
            dx_m,
            wavelength,
            z_bc,
            n=external_index,
            bandlimit=bandlimit,
            alias_control=True,
        )
        bc_mask = make_transfer_sampling_alias_mask(
            shape, dx_m, wavelength, z_bc, n=external_index
        )
        native_nyquist = _r3_native_nyquist_mask(
            shape, dx_m, detector_pixel
        )
        mtf = make_square_pixel_mtf(shape, dx_m, detector_pixel)
        synthetic_controls.append(
            _r3_synthetic_pixel_controls(dx_m, detector_pixel)
        )
        mask_controls["AB"].append(
            _r2_mask_controls(
                shape, dx_m, wavelength, z_ab, external_index
            )
        )
        mask_controls["BC"].append(
            _r2_mask_controls(
                shape, dx_m, wavelength, z_bc, external_index
            )
        )

        detector_stacks = {
            method: {
                branch: np.empty(
                    (len(positions), *native_roi), dtype=np.float64
                )
                for branch in ("point_sample", "pixel_box_average")
            }
            for method in ("current_asm", "alias_controlled")
        }
        exit_mask_fractions: list[float] = []
        exit_native_fractions: list[float] = []
        detector_native_fractions = {
            "current_asm": [],
            "alias_controlled": [],
        }
        field_numerator = 0.0
        field_denominator = 0.0
        intensity_numerator = 0.0
        intensity_denominator = 0.0

        for scan_index, position_xy in enumerate(positions):
            shifted_b = shift_field_integer_pixels(
                sample_b,
                position_xy,
                dx_m,
                boundary=str(_section(config, "sample_b")["object_boundary"]),
            )
            exit_wave = probe * shifted_b
            exit_spectrum = np.fft.fft2(exit_wave)
            exit_mask_fractions.append(
                _r3_energy_fraction(exit_spectrum, bc_mask)
            )
            exit_native_fractions.append(
                _r3_energy_fraction(exit_spectrum, native_nyquist)
            )
            current_field = apply_angular_spectrum_transfer(
                exit_wave, transfer_current
            )
            alias_field = apply_angular_spectrum_transfer(exit_wave, transfer_alias)
            field_numerator += float(np.sum(np.abs(current_field - alias_field) ** 2))
            field_denominator += float(np.sum(np.abs(alias_field) ** 2))
            current_intensity = np.asarray(
                np.abs(current_field) ** 2, dtype=np.float64
            )
            alias_intensity = np.asarray(np.abs(alias_field) ** 2, dtype=np.float64)
            intensity_numerator += float(
                np.sum((current_intensity - alias_intensity) ** 2)
            )
            intensity_denominator += float(np.sum(alias_intensity**2))

            for method, intensity in (
                ("current_asm", current_intensity),
                ("alias_controlled", alias_intensity),
            ):
                spectrum = np.fft.fft2(intensity)
                detector_native_fractions[method].append(
                    _r3_energy_fraction(spectrum, native_nyquist)
                )
                averaged, frame_controls = _r3_pixel_average_from_spectrum(
                    intensity, spectrum, mtf, algebra_threshold
                )
                for name in pixel_actual_controls:
                    pixel_actual_controls[name].append(frame_controls[name])
                point_native = _r3_native_sample(intensity, factor, offset)
                pixel_native = _r3_native_sample(averaged, factor, offset)
                detector_stacks[method]["point_sample"][scan_index] = center_crop(
                    point_native, native_roi
                )
                detector_stacks[method]["pixel_box_average"][scan_index] = (
                    center_crop(pixel_native, native_roi)
                )
                all_finite = bool(
                    all_finite
                    and np.all(np.isfinite(intensity))
                    and np.all(np.isfinite(averaged))
                )
                all_final_nonnegative = bool(
                    all_final_nonnegative and np.all(pixel_native >= 0.0)
                )

            if factor == selected_factor and scan_index == selected_scan:
                selected_images = {
                    "point_sample": detector_stacks["alias_controlled"][
                        "point_sample"
                    ][scan_index].copy(),
                    "pixel_box_average": detector_stacks["alias_controlled"][
                        "pixel_box_average"
                    ][scan_index].copy(),
                }
                selected_images["relative_difference"] = (
                    selected_images["point_sample"]
                    - selected_images["pixel_box_average"]
                ) / max(
                    float(np.max(selected_images["pixel_box_average"])),
                    np.finfo(np.float64).eps,
                )
                repeated_field = apply_angular_spectrum_transfer(
                    exit_wave, transfer_alias
                )
                repeated_intensity = np.asarray(
                    np.abs(repeated_field) ** 2, dtype=np.float64
                )
                repeated_spectrum = np.fft.fft2(repeated_intensity)
                repeated_average, _ = _r3_pixel_average_from_spectrum(
                    repeated_intensity,
                    repeated_spectrum,
                    mtf,
                    algebra_threshold,
                )
                repeated_roi = center_crop(
                    _r3_native_sample(repeated_average, factor, offset),
                    native_roi,
                )
                determinism_error = relative_l2(
                    repeated_roi,
                    detector_stacks["alias_controlled"]["pixel_box_average"][
                        scan_index
                    ],
                )

            del (
                alias_field,
                alias_intensity,
                current_field,
                current_intensity,
                exit_spectrum,
                exit_wave,
                shifted_b,
            )

        b_exit_outside_mask_mean.append(float(np.mean(exit_mask_fractions)))
        b_exit_outside_mask_max.append(float(np.max(exit_mask_fractions)))
        b_exit_outside_native_mean.append(float(np.mean(exit_native_fractions)))
        b_exit_outside_native_max.append(float(np.max(exit_native_fractions)))
        for method in detector_native_fractions:
            detector_spectrum[method]["mean"].append(
                float(np.mean(detector_native_fractions[method]))
            )
            detector_spectrum[method]["max"].append(
                float(np.max(detector_native_fractions[method]))
            )
        bc_field_difference.append(
            float(
                np.sqrt(field_numerator)
                / max(np.sqrt(field_denominator), np.finfo(float).eps)
            )
        )
        bc_intensity_difference.append(
            float(
                np.sqrt(intensity_numerator)
                / max(np.sqrt(intensity_denominator), np.finfo(float).eps)
            )
        )
        for method in point_vs_pixel:
            point_vs_pixel[method].append(
                relative_l2(
                    detector_stacks[method]["point_sample"],
                    detector_stacks[method]["pixel_box_average"],
                )
            )
        cases.append(
            {
                "factor": factor,
                "dx_m": dx_m,
                "shape": shape,
                "P_B": probe_roi,
                "detector": detector_stacks,
            }
        )
        all_finite = bool(
            all_finite
            and np.all(np.isfinite(padded_exit))
            and np.all(np.isfinite(sample_b))
            and np.all(np.isfinite(probe))
        )
        del (
            bc_mask,
            mtf,
            native_b_period,
            native_exit,
            native_nyquist,
            padded_exit,
            probe,
            sample_b,
            transfer_alias,
            transfer_current,
        )
        gc.collect()

    convergence = _r3_convergence_metrics(cases, convergence_threshold)
    canonical_max = float(
        max([base_to_r0_error, fine_reference_error, *b_mapping_errors])
    )
    canonical_validation = {
        "base_period_shape_factor1": np.asarray(
            base_period_native_shape, dtype=np.int64
        ),
        "base_to_r0_max_complex_error": base_to_r0_error,
        "factor4_to_r1_fine_max_complex_error": fine_reference_error,
        "native_mapping_max_complex_error_by_factor": np.asarray(
            b_mapping_errors, dtype=np.float64
        ),
        "max_complex_error": canonical_max,
        "threshold": algebra_threshold,
        "same_r1_canonical_validation_pass": bool(fine_validation["pass"]),
        "pass": bool(
            fine_validation["pass"] and canonical_max <= algebra_threshold
        ),
    }
    a_exit_max = float(max(a_exit_errors))
    a_exit_mapping = {
        "relative_l2_by_factor": np.asarray(a_exit_errors, dtype=np.float64),
        "max": a_exit_max,
        "threshold": algebra_threshold,
        "pass": bool(a_exit_max <= algebra_threshold),
    }
    alias_mask_pass = all(
        bool(value["pass"])
        for stage in mask_controls.values()
        for value in stage
    )
    alias_masks = {
        "AB_kept_bin_fraction": np.asarray(
            [value["kept_bin_fraction"] for value in mask_controls["AB"]],
            dtype=np.float64,
        ),
        "BC_kept_bin_fraction": np.asarray(
            [value["kept_bin_fraction"] for value in mask_controls["BC"]],
            dtype=np.float64,
        ),
        "pass": bool(alias_mask_pass),
    }
    spectrum_metrics = {
        "factors": np.asarray(factors, dtype=np.int64),
        "B_exit": {
            "outside_BC_alias_mask_energy_fraction_mean": np.asarray(
                b_exit_outside_mask_mean, dtype=np.float64
            ),
            "outside_BC_alias_mask_energy_fraction_max": np.asarray(
                b_exit_outside_mask_max, dtype=np.float64
            ),
            "outside_native_detector_nyquist_energy_fraction_mean": np.asarray(
                b_exit_outside_native_mean, dtype=np.float64
            ),
            "outside_native_detector_nyquist_energy_fraction_max": np.asarray(
                b_exit_outside_native_max, dtype=np.float64
            ),
        },
        "detector_intensity": {
            method: {
                "outside_native_detector_nyquist_energy_fraction_mean": np.asarray(
                    values["mean"], dtype=np.float64
                ),
                "outside_native_detector_nyquist_energy_fraction_max": np.asarray(
                    values["max"], dtype=np.float64
                ),
            }
            for method, values in detector_spectrum.items()
        },
    }
    bc_metrics = {
        "factors": np.asarray(factors, dtype=np.int64),
        "detector_field_current_vs_alias_relative_l2": np.asarray(
            bc_field_difference, dtype=np.float64
        ),
        "full_intensity_current_vs_alias_relative_l2": np.asarray(
            bc_intensity_difference, dtype=np.float64
        ),
        "native_detector": {
            branch: np.asarray(
                [
                    relative_l2(
                        _section(_section(case, "detector"), "current_asm")[
                            branch
                        ],
                        _section(
                            _section(case, "detector"), "alias_controlled"
                        )[branch],
                    )
                    for case in cases
                ],
                dtype=np.float64,
            )
            for branch in ("point_sample", "pixel_box_average")
        },
    }
    operator_metrics = {
        "factors": np.asarray(factors, dtype=np.int64),
        "point_vs_pixel_relative_l2": {
            method: np.asarray(values, dtype=np.float64)
            for method, values in point_vs_pixel.items()
        },
        "selected_factor": selected_factor,
        "selected_scan_index": selected_scan,
    }
    synthetic_max = {
        key: float(max(value[key] for value in synthetic_controls))
        for key in synthetic_controls[0]
    }
    actual_max = {
        key: float(max(values)) for key, values in pixel_actual_controls.items()
    }
    pixel_controls_pass = bool(
        all(value <= algebra_threshold for value in synthetic_max.values())
        and all(value <= algebra_threshold for value in actual_max.values())
        and all_final_nonnegative
    )
    pixel_controls = {
        "synthetic_max": synthetic_max,
        "actual_max": actual_max,
        "threshold": algebra_threshold,
        "all_final_detector_averages_nonnegative": bool(
            all_final_nonnegative
        ),
        "pass": pixel_controls_pass,
    }
    determinism_pass = bool(
        np.isfinite(determinism_error)
        and determinism_error <= determinism_threshold
    )
    determinism = {
        "selected_factor": selected_factor,
        "selected_scan_index": selected_scan,
        "primary_relative_l2": determinism_error,
        "threshold": determinism_threshold,
        "pass": determinism_pass,
    }
    hard_checks_pass = bool(
        canonical_validation["pass"]
        and a_exit_mapping["pass"]
        and alias_mask_pass
        and pixel_controls_pass
        and determinism_pass
        and all_finite
        and bool(r2_metrics["hard_checks_pass"])
    )
    probe_pass = bool(convergence["P_B_pass"])
    primary_pass = bool(convergence["primary_detector_pass"])
    if not hard_checks_pass:
        status = "Failed"
    elif probe_pass and primary_pass:
        status = "Passed"
    else:
        status = "Inconclusive"
    point_acceptance = float(
        _section(_section(convergence["acceptance"], "detector"), "alias_controlled")[
            "point_sample"
        ]
    )
    point_pass = bool(point_acceptance <= convergence_threshold)
    point_material = bool(
        point_vs_pixel["alias_controlled"][-1] > convergence_threshold
    )
    b_exit_material = bool(
        b_exit_outside_mask_mean[-1] > convergence_threshold
        or b_exit_outside_mask_max[-1] > convergence_threshold
    )
    bc_material = bool(
        bc_field_difference[-1] > convergence_threshold
        or bc_intensity_difference[-1] > convergence_threshold
    )
    outcome_flags = {
        "P_B_convergence_pass": probe_pass,
        "primary_detector_convergence_pass": primary_pass,
        "alias_point_detector_convergence_pass": point_pass,
        "point_vs_pixel_material_at_factor4": point_material,
        "B_exit_alias_sensitive_energy_material_at_factor4": b_exit_material,
        "BC_method_difference_material_at_factor4": bc_material,
        "interpretation_code": _r3_outcome_code(
            hard_checks_pass=hard_checks_pass,
            probe_pass=probe_pass,
            primary_pass=primary_pass,
            point_pass=point_pass,
            point_vs_pixel_material=point_material,
            b_exit_or_bc_material=bool(b_exit_material or bc_material),
        ),
    }
    thresholds = {
        "convergence_relative_l2_max": convergence_threshold,
        "spectral_and_method_material_relative_l2_min": convergence_threshold,
        "mapping_and_pixel_relative_max": algebra_threshold,
        "determinism_relative_l2_max": determinism_threshold,
    }
    sampling_metrics = {
        "factors": np.asarray(factors, dtype=np.int64),
        "dx_m": np.asarray(dx_values, dtype=np.float64),
        "shapes": np.asarray(shapes, dtype=np.int64),
        "external_fov_m": np.asarray(
            sampling["external_fov_m"], dtype=np.float64
        ),
        "native_sample_offset_px": np.asarray(offsets, dtype=np.int64),
        "physical_origin_compensation_m": compensation,
        "native_roi_shape": np.asarray(native_roi, dtype=np.int64),
        "scan_count": int(len(positions)),
        "full_detector_stacks_retained": False,
    }
    metrics = {
        "version": str(diagnostics["version"]),
        "methods": dict(_section(diagnostics, "methods")),
        "sampling": sampling_metrics,
        "canonical_b_validation": canonical_validation,
        "a_exit_native_recovery": a_exit_mapping,
        "alias_masks": alias_masks,
        "spectra": spectrum_metrics,
        "bc_propagation": bc_metrics,
        "detector_sampling": convergence,
        "detector_operator_difference": operator_metrics,
        "pixel_operator_controls": pixel_controls,
        "determinism": determinism,
        "thresholds": thresholds,
        "outcome_flags": outcome_flags,
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_final_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    result = {
        "selected_scan": selected_images,
        "metrics": {
            "spectra": spectrum_metrics,
            "bc_propagation": bc_metrics,
            "detector_sampling": convergence,
            "detector_operator_difference": operator_metrics,
            "thresholds": thresholds,
        },
    }
    del base_periods, cases, fine_base, legacy_period
    gc.collect()
    return result, metrics


def _r4_block_mean(values: NDArray[np.generic], factor: int) -> NDArray[Any]:
    """Average q-by-q staggered nodes, including complex probe diagnostics."""

    array = np.asarray(values)
    q = int(factor)
    ny, nx = array.shape[-2:]
    reshaped = array.reshape(*array.shape[:-2], ny // q, q, nx // q, q)
    return reshaped.mean(axis=(-3, -1))


def _r4_padded_exit(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    shape: tuple[int, int],
    dx_m: float,
    factor: int,
) -> NDArray[np.complex128]:
    """Map the fixed A-exit residual to staggered midpoint nodes."""

    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    illumination = _section(config, "illumination")
    baseline_dx = _isotropic_dx(optics["baseline_dx_m"], "R4 baseline dx")
    residual = np.asarray(baseline["U_A_exit"], dtype=np.complex128) - np.asarray(
        controls["homogeneous_reference"], dtype=np.complex128
    )
    mapped_shape = (residual.shape[0] * factor, residual.shape[1] * factor)
    residual_with_ghost = np.pad(residual, ((1, 1), (1, 1)), mode="edge")
    mapped = resample_centered_grid(
        residual_with_ghost, baseline_dx, mapped_shape, dx_m
    )
    incident = make_plane_wave(
        shape,
        dx_m,
        float(optics["wavelength_m"]),
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    homogeneous = angular_spectrum_propagate(
        incident,
        dx_m,
        float(optics["wavelength_m"]),
        float(sample_a["thickness_m"]),
        n=float(optics["internal_reference_index"]),
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=False,
    )
    return np.asarray(homogeneous + _center_pad(mapped, shape), dtype=np.complex128)


def _r4_node_geometry_error(native_count: int, pixel_m: float, factor: int) -> float:
    """Return normalized block-center error for staggered midpoint nodes."""

    q = int(factor)
    nodes = (
        np.arange(native_count * q, dtype=np.float64)
        - (native_count * q - 1) / 2.0
    ) * (pixel_m / q)
    block_centers = nodes.reshape(native_count, q).mean(axis=1)
    native = (
        np.arange(native_count, dtype=np.float64) - (native_count - 1) / 2.0
    ) * pixel_m
    return float(np.max(np.abs(block_centers - native)) / pixel_m)


def _run_r4_diagnostics(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run R4 positive staggered detector quadrature in streaming mode."""

    diagnostics = _section(config, "diagnostics_r4")
    sampling = _section(diagnostics, "sampling")
    optics = _section(config, "optics")
    acceptance = _section(config, "acceptance")
    factors = [int(value) for value in sampling["factors"]]
    dx_values = [float(value) for value in sampling["node_dx_m"]]
    shapes = [_shape(value, "R4 node shape") for value in sampling["node_shapes"]]
    native_roi = _shape(sampling["native_roi_shape"], "R4 native ROI")
    pixel_m = float(_section(optics, "detector")["pixel_size_m"])
    wavelength = float(optics["wavelength_m"])
    external_index = float(optics["external_medium_index"])
    threshold = float(acceptance["convergence_relative_l2_max"])
    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    determinism_threshold = float(acceptance["determinism_relative_l2_max"])

    fine_base, _, fine_dx, fine_validation = _make_r1_canonical_b(
        _section(config, "sample_b"),
        _section(config, "diagnostics_r1"),
        legacy_canonical_b,
        legacy_canonical_dx_m,
    )
    feature_pixels_fine = _require_integer_ratio(
        float(_section(config, "sample_b")["physical_feature_size_m"]),
        fine_dx,
        "R4 fine B feature size",
    )
    coarse_y = fine_base.shape[0] // feature_pixels_fine
    coarse_x = fine_base.shape[1] // feature_pixels_fine
    coarse_cells = fine_base.reshape(
        coarse_y, feature_pixels_fine, coarse_x, feature_pixels_fine
    )[:, feature_pixels_fine // 2, :, feature_pixels_fine // 2]

    cases: list[dict[str, Any]] = []
    mapping_errors: list[float] = []
    geometry_errors: list[float] = []
    constant_errors: list[float] = []
    sum_errors: list[float] = []
    all_finite = bool(fine_validation["pass"])
    all_nonnegative = True
    selected_image: NDArray[np.float64] | None = None
    determinism_error = float("nan")
    z_ab = float(optics["z_AB_m"])
    z_bc = float(optics["z_BC_m"])

    for factor, dx_m, shape in zip(factors, dx_values, shapes, strict=True):
        feature_pixels = _require_integer_ratio(
            float(_section(config, "sample_b")["physical_feature_size_m"]),
            dx_m,
            "R4 B feature size",
        )
        base_period = np.repeat(
            np.repeat(coarse_cells, feature_pixels, axis=0),
            feature_pixels,
            axis=1,
        ).astype(np.complex128, copy=False)
        sample_b = _center_periodic_extend(base_period, shape)
        native_b = _r4_block_mean(sample_b, factor)
        native_reference = resample_centered_grid(
            legacy_canonical_b,
            legacy_canonical_dx_m,
            (base_period.shape[0] // factor, base_period.shape[1] // factor),
            pixel_m,
        )
        mapping_errors.append(
            float(
                np.max(
                    np.abs(
                        center_crop(native_b, native_reference.shape)
                        - native_reference
                    )
                )
            )
        )
        geometry_errors.append(
            max(
                _r4_node_geometry_error(shape[0] // factor, pixel_m, factor),
                _r4_node_geometry_error(shape[1] // factor, pixel_m, factor),
            )
        )
        constant = positive_midpoint_pixel_average(
            np.ones((16 * factor, 16 * factor), dtype=np.float64), factor
        )
        constant_errors.append(float(np.max(np.abs(constant - 1.0))))

        padded_exit = _r4_padded_exit(
            config, baseline, controls, shape, dx_m, factor
        )
        transfer_ab = make_angular_spectrum_transfer(
            shape,
            dx_m,
            wavelength,
            z_ab,
            n=external_index,
            bandlimit=True,
            alias_control=True,
        )
        probe = apply_angular_spectrum_transfer(padded_exit, transfer_ab)
        probe_roi = center_crop(_r4_block_mean(probe, factor), native_roi).copy()
        del padded_exit, transfer_ab
        gc.collect()
        transfer_bc = make_angular_spectrum_transfer(
            shape,
            dx_m,
            wavelength,
            z_bc,
            n=external_index,
            bandlimit=True,
            alias_control=True,
        )
        intensity_stack = np.empty(
            (len(positions), *native_roi), dtype=np.float64
        )
        factor_sum_errors: list[float] = []
        for scan_index, position_xy in enumerate(positions):
            exit_wave = shift_field_integer_pixels(
                sample_b, position_xy, dx_m, boundary="periodic"
            )
            exit_wave *= probe
            detector_field = apply_angular_spectrum_transfer(
                exit_wave, transfer_bc
            )
            intensity = np.asarray(np.abs(detector_field) ** 2, dtype=np.float64)
            pixels = positive_midpoint_pixel_average(intensity, factor)
            factor_sum_errors.append(
                float(
                    abs(float(np.sum(pixels)) * factor**2 - float(np.sum(intensity)))
                    / max(float(np.sum(intensity)), np.finfo(float).eps)
                )
            )
            intensity_stack[scan_index] = center_crop(pixels, native_roi)
            all_finite = bool(
                all_finite
                and np.all(np.isfinite(detector_field))
                and np.all(np.isfinite(pixels))
            )
            all_nonnegative = bool(
                all_nonnegative
                and np.all(intensity >= 0.0)
                and np.all(pixels >= 0.0)
            )
            if factor == 8 and scan_index == 0:
                selected_image = center_crop(pixels, native_roi).copy()
                repeated_field = apply_angular_spectrum_transfer(
                    exit_wave, transfer_bc
                )
                repeated_pixels = positive_midpoint_pixel_average(
                    np.abs(repeated_field) ** 2, factor
                )
                determinism_error = relative_l2(
                    center_crop(repeated_pixels, native_roi), selected_image
                )
                del repeated_field, repeated_pixels
            del detector_field, exit_wave, intensity, pixels
        sum_errors.append(float(max(factor_sum_errors)))
        cases.append(
            {
                "factor": factor,
                "dx_m": dx_m,
                "shape": shape,
                "P_B": probe_roi,
                "I_stack": intensity_stack,
            }
        )
        del base_period, probe, sample_b, transfer_bc
        gc.collect()

    reference = cases[-1]
    series = {
        name: np.asarray(
            [relative_l2(case[name], reference[name]) for case in cases],
            dtype=np.float64,
        )
        for name in ("P_B", "I_stack")
    }
    final_pair = {
        name: relative_l2(cases[-2][name], cases[-1][name])
        for name in ("P_B", "I_stack")
    }
    convergence_pass = {
        name: bool(value <= threshold) for name, value in final_pair.items()
    }
    mapping_max = float(max(mapping_errors))
    geometry_max = float(max(geometry_errors))
    constant_max = float(max(constant_errors))
    sum_max = float(max(sum_errors))
    weights = {
        "minimum": float(min(1.0 / factor**2 for factor in factors)),
        "maximum": float(max(1.0 / factor**2 for factor in factors)),
        "all_finite": True,
        "all_nonnegative": True,
        "sum_one_max_abs_error": 0.0,
    }
    determinism_pass = bool(
        np.isfinite(determinism_error)
        and determinism_error <= determinism_threshold
    )
    hard_checks_pass = bool(
        fine_validation["pass"]
        and mapping_max <= algebra_threshold
        and geometry_max <= algebra_threshold
        and constant_max <= algebra_threshold
        and sum_max <= algebra_threshold
        and determinism_pass
        and all_finite
        and all_nonnegative
    )
    if not hard_checks_pass:
        status = "Failed"
    elif all(convergence_pass.values()):
        status = "Passed"
    else:
        status = "Inconclusive"
    metrics = {
        "version": "R4",
        "methods": dict(_section(diagnostics, "methods")),
        "r3_provenance": dict(_section(diagnostics, "r3_provenance")),
        "sampling": {
            "factors": np.asarray(factors, dtype=np.int64),
            "node_dx_m": np.asarray(dx_values, dtype=np.float64),
            "node_shapes": np.asarray(shapes, dtype=np.int64),
            "native_roi_shape": np.asarray(native_roi, dtype=np.int64),
            "scan_count": int(len(positions)),
            "full_node_stacks_retained": False,
        },
        "canonical_b_validation": {
            "max_complex_error_by_factor": np.asarray(
                mapping_errors, dtype=np.float64
            ),
            "max_complex_error": mapping_max,
            "pass": bool(mapping_max <= algebra_threshold),
        },
        "quadrature_controls": {
            "node_geometry_normalized_error_by_factor": np.asarray(
                geometry_errors, dtype=np.float64
            ),
            "constant_max_abs_error_by_factor": np.asarray(
                constant_errors, dtype=np.float64
            ),
            "sum_relative_error_by_factor": np.asarray(
                sum_errors, dtype=np.float64
            ),
            "max_node_geometry_normalized_error": geometry_max,
            "max_constant_abs_error": constant_max,
            "max_sum_relative_error": sum_max,
            "weights": weights,
            "all_outputs_nonnegative": bool(all_nonnegative),
            "pass": bool(
                geometry_max <= algebra_threshold
                and constant_max <= algebra_threshold
                and sum_max <= algebra_threshold
                and all_nonnegative
            ),
        },
        "convergence": {
            "factors": np.asarray(factors, dtype=np.int64),
            "relative_to_q8": series,
            "acceptance_pair_factors": np.asarray([4, 8], dtype=np.int64),
            "acceptance": final_pair,
            "pass": convergence_pass,
        },
        "determinism": {
            "factor": 8,
            "scan_index": 0,
            "I_stack_relative_l2": determinism_error,
            "pass": determinism_pass,
        },
        "thresholds": {
            "convergence_relative_l2_max": threshold,
            "algebra_relative_l2_max": algebra_threshold,
            "determinism_relative_l2_max": determinism_threshold,
        },
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    result = {
        "selected_q8_scan0": selected_image,
        "metrics": {
            "convergence": metrics["convergence"],
            "quadrature_controls": metrics["quadrature_controls"],
            "thresholds": metrics["thresholds"],
        },
    }
    del cases, coarse_cells, fine_base
    gc.collect()
    return result, metrics


def _r5_homogeneous_a_exit(
    config: Mapping[str, Any], shape: tuple[int, int], dx_m: float
) -> NDArray[np.complex128]:
    """Return the homogeneous-reference A-exit field on one R5 node grid."""

    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    illumination = _section(config, "illumination")
    incident = make_plane_wave(
        shape,
        dx_m,
        float(optics["wavelength_m"]),
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    return angular_spectrum_propagate(
        incident,
        dx_m,
        float(optics["wavelength_m"]),
        float(sample_a["thickness_m"]),
        n=float(optics["internal_reference_index"]),
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=False,
    )


def _r5_boundary_ring_energy_fraction(
    values: NDArray[np.complexfloating], width_px: int
) -> float:
    """Return field energy in a rectangular outer ring."""

    field = np.asarray(values, dtype=np.complex128)
    width = int(width_px)
    if field.ndim != 2 or width <= 0 or 2 * width >= min(field.shape):
        msg = "R5 boundary ring width is invalid for the field shape."
        raise ValueError(msg)
    energy = np.abs(field) ** 2
    total = float(np.sum(energy))
    interior = float(np.sum(energy[width:-width, width:-width]))
    return float((total - interior) / max(total, np.finfo(float).eps))


def _r5_outcome_code(
    *, status: str, support_material: bool, boundary_material: bool
) -> str:
    """Return the frozen R5 support/boundary interpretation code."""

    if status != "Passed":
        return "attribution_blocked"
    if support_material and boundary_material:
        return "finite_support_and_circular_wrap_material"
    if support_material:
        return "finite_support_material"
    if boundary_material:
        return "circular_wrap_material"
    return "finite_support_and_circular_wrap_nonmaterial"


def _run_r5_diagnostics(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run R5 finite-B and zero-padded residual BC diagnostics."""

    diagnostics = _section(config, "diagnostics_r5")
    sampling = _section(diagnostics, "sampling")
    support = _section(diagnostics, "finite_support")
    optics = _section(config, "optics")
    acceptance = _section(config, "acceptance")
    factor = int(sampling["quadrature_factor"])
    dx_m = float(sampling["node_dx_m"])
    base_shape = _shape(sampling["base_node_shape"], "R5 base shape")
    padding_fov = [float(value) for value in sampling["padding_fov_m"]]
    padding_shapes = [
        _shape(value, "R5 padding shape")
        for value in sampling["padding_node_shapes"]
    ]
    native_roi = _shape(sampling["native_roi_shape"], "R5 native ROI")
    ring_width_px = _require_integer_ratio(
        float(sampling["boundary_ring_width_m"]),
        dx_m,
        "R5 boundary ring width",
    )
    threshold = float(acceptance["convergence_relative_l2_max"])
    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    determinism_threshold = float(acceptance["determinism_relative_l2_max"])
    wavelength = float(optics["wavelength_m"])
    external_index = float(optics["external_medium_index"])
    z_ab = float(optics["z_AB_m"])
    z_bc = float(optics["z_BC_m"])

    fine_base, _, fine_dx, fine_validation = _make_r1_canonical_b(
        _section(config, "sample_b"),
        _section(config, "diagnostics_r1"),
        legacy_canonical_b,
        legacy_canonical_dx_m,
    )
    feature_pixels_fine = _require_integer_ratio(
        float(_section(config, "sample_b")["physical_feature_size_m"]),
        fine_dx,
        "R5 fine B feature size",
    )
    coarse_y = fine_base.shape[0] // feature_pixels_fine
    coarse_x = fine_base.shape[1] // feature_pixels_fine
    coarse_cells = fine_base.reshape(
        coarse_y, feature_pixels_fine, coarse_x, feature_pixels_fine
    )[:, feature_pixels_fine // 2, :, feature_pixels_fine // 2]
    feature_pixels = _require_integer_ratio(
        float(_section(config, "sample_b")["physical_feature_size_m"]),
        dx_m,
        "R5 B feature size",
    )
    base_period = np.repeat(
        np.repeat(coarse_cells, feature_pixels, axis=0),
        feature_pixels,
        axis=1,
    ).astype(np.complex128, copy=False)
    registered_support_shape = tuple(
        _require_integer_ratio(float(value), dx_m, "R5 finite support")
        for value in support["physical_shape_m"]
    )
    if base_period.shape != registered_support_shape:
        msg = "R5 canonical phase cells do not fill the finite support."
        raise RuntimeError(msg)

    periodic_b = _center_periodic_extend(base_period, base_shape)
    finite_modulation = _center_pad(base_period - 1.0, base_shape)
    finite_b = 1.0 + finite_modulation
    support_mask = _center_pad(
        np.ones(base_period.shape, dtype=np.float64), base_shape
    ).astype(bool)
    native_period = _r4_block_mean(base_period, factor)
    native_reference = resample_centered_grid(
        legacy_canonical_b,
        legacy_canonical_dx_m,
        native_period.shape,
        float(_section(optics, "detector")["pixel_size_m"]),
    )
    canonical_mapping_error = float(
        np.max(np.abs(native_period - native_reference))
    )
    support_mapping_error = float(
        np.max(np.abs(center_crop(finite_b, base_period.shape) - base_period))
    )
    exterior_error = float(np.max(np.abs(finite_b[~support_mask] - 1.0)))
    unit_modulus_error = float(np.max(np.abs(np.abs(finite_b) - 1.0)))
    max_scan_xy = np.max(np.abs(positions), axis=0)
    support_half_xy = np.asarray(support["physical_shape_m"], dtype=float)[::-1]
    support_half_xy /= 2.0
    base_half_xy = np.asarray(sampling["base_fov_m"], dtype=float)[::-1]
    base_half_xy /= 2.0
    support_margin_m = float(np.min(base_half_xy - support_half_xy - max_scan_xy))

    padded_exit = _r4_padded_exit(
        config, baseline, controls, base_shape, dx_m, factor
    )
    homogeneous_exit = _r5_homogeneous_a_exit(config, base_shape, dx_m)
    transfer_ab = make_angular_spectrum_transfer(
        base_shape,
        dx_m,
        wavelength,
        z_ab,
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    probe_base = apply_angular_spectrum_transfer(padded_exit, transfer_ab)
    homogeneous_probe_base = apply_angular_spectrum_transfer(
        homogeneous_exit, transfer_ab
    )
    delta_probe = probe_base - homogeneous_probe_base
    decomposition_error = relative_l2(
        homogeneous_probe_base + delta_probe, probe_base
    )
    boundary_energy_fraction = _r5_boundary_ring_energy_fraction(
        delta_probe, ring_width_px
    )
    del padded_exit, homogeneous_exit, transfer_ab
    gc.collect()

    transfer_bc_base = make_angular_spectrum_transfer(
        base_shape,
        dx_m,
        wavelength,
        z_bc,
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    periodic_stack = np.empty((len(positions), *native_roi), dtype=np.float64)
    finite_circular_stack = np.empty_like(periodic_stack)
    sum_errors: list[float] = []
    all_finite = bool(fine_validation["pass"])
    all_nonnegative = True
    for scan_index, position_xy in enumerate(positions):
        shifted_periodic = shift_field_integer_pixels(
            periodic_b, position_xy, dx_m, boundary="periodic"
        )
        detector_periodic = apply_angular_spectrum_transfer(
            probe_base * shifted_periodic, transfer_bc_base
        )
        intensity_periodic = np.abs(detector_periodic) ** 2
        pixels_periodic = positive_midpoint_pixel_average(
            intensity_periodic, factor
        )
        periodic_stack[scan_index] = center_crop(pixels_periodic, native_roi)

        shifted_modulation = shift_field_integer_pixels(
            finite_modulation,
            position_xy,
            dx_m,
            boundary="constant",
            fill_value=0.0j,
        )
        detector_finite = apply_angular_spectrum_transfer(
            probe_base * (1.0 + shifted_modulation), transfer_bc_base
        )
        intensity_finite = np.abs(detector_finite) ** 2
        pixels_finite = positive_midpoint_pixel_average(intensity_finite, factor)
        finite_circular_stack[scan_index] = center_crop(
            pixels_finite, native_roi
        )
        for intensity, pixels in (
            (intensity_periodic, pixels_periodic),
            (intensity_finite, pixels_finite),
        ):
            sum_errors.append(
                float(
                    abs(
                        float(np.sum(pixels)) * factor**2
                        - float(np.sum(intensity))
                    )
                    / max(float(np.sum(intensity)), np.finfo(float).eps)
                )
            )
        all_finite = bool(
            all_finite
            and np.all(np.isfinite(detector_periodic))
            and np.all(np.isfinite(detector_finite))
            and np.all(np.isfinite(pixels_periodic))
            and np.all(np.isfinite(pixels_finite))
        )
        all_nonnegative = bool(
            all_nonnegative
            and np.all(intensity_periodic >= 0.0)
            and np.all(intensity_finite >= 0.0)
            and np.all(pixels_periodic >= 0.0)
            and np.all(pixels_finite >= 0.0)
        )
        del (
            shifted_periodic,
            detector_periodic,
            intensity_periodic,
            pixels_periodic,
            shifted_modulation,
            detector_finite,
            intensity_finite,
            pixels_finite,
        )
    del periodic_b, finite_b, support_mask, transfer_bc_base
    gc.collect()

    open_cases: list[dict[str, Any]] = []
    background_rois: list[NDArray[np.float64]] = []
    selected_open: NDArray[np.float64] | None = None
    determinism_error = float("nan")
    for fov_m, shape in zip(padding_fov, padding_shapes, strict=True):
        homogeneous_exit_full = _r5_homogeneous_a_exit(config, shape, dx_m)
        transfer_ab_full = make_angular_spectrum_transfer(
            shape,
            dx_m,
            wavelength,
            z_ab,
            n=external_index,
            bandlimit=True,
            alias_control=True,
        )
        homogeneous_probe = apply_angular_spectrum_transfer(
            homogeneous_exit_full, transfer_ab_full
        )
        delta_probe_full = _center_pad(delta_probe, shape)
        probe_full = homogeneous_probe + delta_probe_full
        modulation_full = _center_pad(base_period - 1.0, shape)
        del homogeneous_exit_full, transfer_ab_full
        gc.collect()

        transfer_bc = make_angular_spectrum_transfer(
            shape,
            dx_m,
            wavelength,
            z_bc,
            n=external_index,
            bandlimit=True,
            alias_control=True,
        )
        homogeneous_detector = apply_angular_spectrum_transfer(
            homogeneous_probe, transfer_bc
        )
        background_pixels = positive_midpoint_pixel_average(
            np.abs(homogeneous_detector) ** 2, factor
        )
        background_rois.append(center_crop(background_pixels, native_roi).copy())
        intensity_stack = np.empty(
            (len(positions), *native_roi), dtype=np.float64
        )
        for scan_index, position_xy in enumerate(positions):
            shifted_modulation = shift_field_integer_pixels(
                modulation_full,
                position_xy,
                dx_m,
                boundary="constant",
                fill_value=0.0j,
            )
            residual_exit = delta_probe_full + probe_full * shifted_modulation
            residual_detector = apply_angular_spectrum_transfer(
                residual_exit, transfer_bc
            )
            detector_field = homogeneous_detector + residual_detector
            intensity = np.abs(detector_field) ** 2
            pixels = positive_midpoint_pixel_average(intensity, factor)
            intensity_stack[scan_index] = center_crop(pixels, native_roi)
            sum_errors.append(
                float(
                    abs(
                        float(np.sum(pixels)) * factor**2
                        - float(np.sum(intensity))
                    )
                    / max(float(np.sum(intensity)), np.finfo(float).eps)
                )
            )
            all_finite = bool(
                all_finite
                and np.all(np.isfinite(detector_field))
                and np.all(np.isfinite(pixels))
            )
            all_nonnegative = bool(
                all_nonnegative
                and np.all(intensity >= 0.0)
                and np.all(pixels >= 0.0)
            )
            if fov_m == padding_fov[-1] and scan_index == 0:
                selected_open = center_crop(pixels, native_roi).copy()
                repeated_residual = apply_angular_spectrum_transfer(
                    residual_exit, transfer_bc
                )
                repeated_detector = homogeneous_detector + repeated_residual
                repeated_pixels = positive_midpoint_pixel_average(
                    np.abs(repeated_detector) ** 2, factor
                )
                determinism_error = relative_l2(
                    center_crop(repeated_pixels, native_roi), selected_open
                )
                del repeated_residual, repeated_detector, repeated_pixels
            del (
                shifted_modulation,
                residual_exit,
                residual_detector,
                detector_field,
                intensity,
                pixels,
            )
        open_cases.append(
            {
                "fov_m": fov_m,
                "shape": shape,
                "I_stack": intensity_stack,
            }
        )
        del (
            homogeneous_probe,
            delta_probe_full,
            probe_full,
            modulation_full,
            transfer_bc,
            homogeneous_detector,
            background_pixels,
        )
        gc.collect()

    reference_open = open_cases[-1]["I_stack"]
    open_series = np.asarray(
        [relative_l2(case["I_stack"], reference_open) for case in open_cases],
        dtype=np.float64,
    )
    open_acceptance = float(open_series[-2])
    padding_pass = bool(open_acceptance <= threshold)
    containment_pass = bool(boundary_energy_fraction <= threshold)
    base_branch_equivalence = relative_l2(
        open_cases[0]["I_stack"], finite_circular_stack
    )
    background_reference = background_rois[-1]
    background_errors = np.asarray(
        [relative_l2(roi, background_reference) for roi in background_rois],
        dtype=np.float64,
    )
    background_error_max = float(np.max(background_errors))
    constant = positive_midpoint_pixel_average(
        np.ones((16 * factor, 16 * factor), dtype=np.float64), factor
    )
    constant_error = float(np.max(np.abs(constant - 1.0)))
    sum_error_max = float(max(sum_errors))
    determinism_pass = bool(
        np.isfinite(determinism_error)
        and determinism_error <= determinism_threshold
    )

    effects = {
        "support_relative_l2": relative_l2(
            periodic_stack, finite_circular_stack
        ),
        "boundary_relative_l2": relative_l2(
            finite_circular_stack, reference_open
        ),
        "combined_relative_l2": relative_l2(periodic_stack, reference_open),
    }
    materiality = {
        name.replace("_relative_l2", "_material"): bool(value > threshold)
        for name, value in effects.items()
    }
    control_errors = (
        canonical_mapping_error,
        support_mapping_error,
        exterior_error,
        unit_modulus_error,
        decomposition_error,
        base_branch_equivalence,
        background_error_max,
        constant_error,
        sum_error_max,
    )
    hard_checks_pass = bool(
        fine_validation["pass"]
        and max(control_errors) <= algebra_threshold
        and support_margin_m > 0.0
        and determinism_pass
        and all_finite
        and all_nonnegative
    )
    if not hard_checks_pass:
        status = "Failed"
    elif padding_pass and containment_pass:
        status = "Passed"
    else:
        status = "Inconclusive"
    outcome_code = _r5_outcome_code(
        status=status,
        support_material=materiality["support_material"],
        boundary_material=materiality["boundary_material"],
    )
    weights = {
        "minimum": float(1.0 / factor**2),
        "maximum": float(1.0 / factor**2),
        "all_finite": True,
        "all_nonnegative": True,
        "sum_one_max_abs_error": 0.0,
    }
    metrics = {
        "version": "R5",
        "methods": dict(_section(diagnostics, "methods")),
        "r4_provenance": dict(_section(diagnostics, "r4_provenance")),
        "sampling": {
            "quadrature_factor": factor,
            "node_dx_m": dx_m,
            "base_fov_m": np.asarray(sampling["base_fov_m"], dtype=np.float64),
            "base_node_shape": np.asarray(base_shape, dtype=np.int64),
            "padding_fov_m": np.asarray(padding_fov, dtype=np.float64),
            "padding_node_shapes": np.asarray(padding_shapes, dtype=np.int64),
            "native_roi_shape": np.asarray(native_roi, dtype=np.int64),
            "boundary_ring_width_m": float(sampling["boundary_ring_width_m"]),
            "scan_count": int(len(positions)),
            "full_node_stacks_retained": False,
        },
        "finite_support": {
            "physical_shape_m": np.asarray(
                support["physical_shape_m"], dtype=np.float64
            ),
            "canonical_phase_cells": np.asarray(
                support["canonical_phase_cells"], dtype=np.int64
            ),
            "exterior_transmission": np.asarray([1.0, 0.0], dtype=np.float64),
            "canonical_mapping_max_complex_error": canonical_mapping_error,
            "support_mapping_max_complex_error": support_mapping_error,
            "exterior_max_complex_error": exterior_error,
            "unit_modulus_max_abs_error": unit_modulus_error,
            "minimum_shifted_support_margin_m": support_margin_m,
            "pass": bool(
                max(
                    canonical_mapping_error,
                    support_mapping_error,
                    exterior_error,
                    unit_modulus_error,
                )
                <= algebra_threshold
                and support_margin_m > 0.0
            ),
        },
        "source_containment": {
            "boundary_ring_width_m": float(sampling["boundary_ring_width_m"]),
            "delta_P_B_boundary_energy_fraction": boundary_energy_fraction,
            "pass": containment_pass,
        },
        "open_boundary_convergence": {
            "padding_fov_m": np.asarray(padding_fov, dtype=np.float64),
            "relative_to_384": open_series,
            "acceptance_pair_fov_m": np.asarray(
                padding_fov[-2:], dtype=np.float64
            ),
            "acceptance": open_acceptance,
            "pass": padding_pass,
        },
        "effects": {**effects, **materiality},
        "controls": {
            "probe_decomposition_relative_l2": decomposition_error,
            "base_open_vs_finite_circular_relative_l2": base_branch_equivalence,
            "homogeneous_background_relative_l2_by_padding": background_errors,
            "homogeneous_background_max_relative_l2": background_error_max,
            "constant_max_abs_error": constant_error,
            "max_sum_relative_error": sum_error_max,
            "weights": weights,
            "all_outputs_nonnegative": bool(all_nonnegative),
            "pass": bool(max(control_errors) <= algebra_threshold),
        },
        "determinism": {
            "padding_fov_m": padding_fov[-1],
            "scan_index": 0,
            "I_stack_relative_l2": determinism_error,
            "pass": determinism_pass,
        },
        "thresholds": {
            "convergence_and_materiality_relative_l2": threshold,
            "algebra_relative_l2_max": algebra_threshold,
            "determinism_relative_l2_max": determinism_threshold,
        },
        "outcome_flags": {"interpretation_code": outcome_code},
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    result = {
        "selected_scan0": {
            "periodic_circular_192": periodic_stack[0].copy(),
            "finite_circular_192": finite_circular_stack[0].copy(),
            "finite_open_384": selected_open,
        },
        "metrics": {
            "open_boundary_convergence": metrics["open_boundary_convergence"],
            "effects": metrics["effects"],
            "controls": metrics["controls"],
            "thresholds": metrics["thresholds"],
        },
    }
    del (
        base_period,
        coarse_cells,
        fine_base,
        finite_modulation,
        probe_base,
        homogeneous_probe_base,
        delta_probe,
        periodic_stack,
        finite_circular_stack,
        open_cases,
        background_rois,
    )
    gc.collect()
    return result, metrics


def _r6_periodic_support_pattern(
    base_period: NDArray[np.complex128], target_shape: tuple[int, int]
) -> NDArray[np.complex128]:
    """Center-crop or periodically extend one canonical B realization."""

    base = np.asarray(base_period, dtype=np.complex128)
    target = _shape(target_shape, "R6 support target shape")
    if target[0] <= base.shape[0] and target[1] <= base.shape[1]:
        return np.asarray(center_crop(base, target), dtype=np.complex128)
    return _center_periodic_extend(base, target)


def _r6_phase_tapered_transmission(
    pattern: NDArray[np.complex128], taper_width_px: int
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Apply a separable raised-cosine phase taper with unit modulus."""

    values = np.asarray(pattern, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) <= 0:
        msg = "R6 support pattern must be a nonempty 2D array."
        raise ValueError(msg)
    width = int(taper_width_px)
    if width < 0 or 2 * width >= min(values.shape):
        msg = "R6 taper width is invalid for the support shape."
        raise ValueError(msg)
    if width == 0:
        weights = np.ones(values.shape, dtype=np.float64)
    else:
        y_distance = np.minimum(
            np.arange(values.shape[0]),
            np.arange(values.shape[0])[::-1],
        ).astype(np.float64)
        x_distance = np.minimum(
            np.arange(values.shape[1]),
            np.arange(values.shape[1])[::-1],
        ).astype(np.float64)
        wy = 0.5 * (
            1.0 - np.cos(np.pi * np.clip(y_distance / width, 0.0, 1.0))
        )
        wx = 0.5 * (
            1.0 - np.cos(np.pi * np.clip(x_distance / width, 0.0, 1.0))
        )
        weights = wy[:, None] * wx[None, :]
    phase = np.angle(values)
    transmission = np.exp(1j * weights * phase)
    return (
        np.asarray(transmission, dtype=np.complex128),
        np.asarray(weights, dtype=np.float64),
    )


def _r6_outcome_code(*, status: str) -> str:
    """Return the frozen R6 robustness interpretation code."""

    if status == "Passed":
        return "periodic_b_materiality_robust_over_support_envelope"
    if status == "Inconclusive":
        return "periodic_b_materiality_support_sensitive"
    return "support_envelope_attribution_blocked"


def _run_r6_diagnostics(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the R6 virtual finite-B support/taper sensitivity envelope."""

    diagnostics = _section(config, "diagnostics_r6")
    sampling = _section(diagnostics, "sampling")
    family = _section(diagnostics, "support_family")
    optics = _section(config, "optics")
    acceptance = _section(config, "acceptance")
    factor = int(sampling["quadrature_factor"])
    dx_m = float(sampling["node_dx_m"])
    shape = _shape(sampling["node_shape"], "R6 node shape")
    native_roi = _shape(sampling["native_roi_shape"], "R6 native ROI")
    widths = [float(value) for value in family["support_width_m"]]
    tapers = [float(value) for value in family["edge_taper_width_m"]]
    nominal_width = float(family["nominal_support_width_m"])
    nominal_taper = float(family["nominal_edge_taper_width_m"])
    materiality_threshold = float(acceptance["convergence_relative_l2_max"])
    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    determinism_threshold = float(acceptance["determinism_relative_l2_max"])
    wavelength = float(optics["wavelength_m"])
    external_index = float(optics["external_medium_index"])

    fine_base, _, fine_dx, fine_validation = _make_r1_canonical_b(
        _section(config, "sample_b"),
        _section(config, "diagnostics_r1"),
        legacy_canonical_b,
        legacy_canonical_dx_m,
    )
    feature_pixels_fine = _require_integer_ratio(
        float(_section(config, "sample_b")["physical_feature_size_m"]),
        fine_dx,
        "R6 fine B feature size",
    )
    coarse_y = fine_base.shape[0] // feature_pixels_fine
    coarse_x = fine_base.shape[1] // feature_pixels_fine
    coarse_cells = fine_base.reshape(
        coarse_y, feature_pixels_fine, coarse_x, feature_pixels_fine
    )[:, feature_pixels_fine // 2, :, feature_pixels_fine // 2]
    feature_pixels = _require_integer_ratio(
        float(_section(config, "sample_b")["physical_feature_size_m"]),
        dx_m,
        "R6 B feature size",
    )
    base_period = np.repeat(
        np.repeat(coarse_cells, feature_pixels, axis=0),
        feature_pixels,
        axis=1,
    ).astype(np.complex128, copy=False)
    periodic_b = _center_periodic_extend(base_period, shape)
    padded_exit = _r4_padded_exit(
        config, baseline, controls, shape, dx_m, factor
    )
    transfer_ab = make_angular_spectrum_transfer(
        shape,
        dx_m,
        wavelength,
        float(optics["z_AB_m"]),
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    probe = apply_angular_spectrum_transfer(padded_exit, transfer_ab)
    transfer_bc = make_angular_spectrum_transfer(
        shape,
        dx_m,
        wavelength,
        float(optics["z_BC_m"]),
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    del padded_exit, transfer_ab
    gc.collect()

    periodic_stack = np.empty((len(positions), *native_roi), dtype=np.float64)
    sum_errors: list[float] = []
    all_finite = bool(
        fine_validation["pass"]
        and np.all(np.isfinite(periodic_b))
        and np.all(np.isfinite(probe))
    )
    all_nonnegative = True
    for scan_index, position_xy in enumerate(positions):
        shifted = shift_field_integer_pixels(
            periodic_b, position_xy, dx_m, boundary="periodic"
        )
        detector = apply_angular_spectrum_transfer(probe * shifted, transfer_bc)
        intensity = np.abs(detector) ** 2
        pixels = positive_midpoint_pixel_average(intensity, factor)
        periodic_stack[scan_index] = center_crop(pixels, native_roi)
        sum_errors.append(
            float(
                abs(
                    float(np.sum(pixels)) * factor**2
                    - float(np.sum(intensity))
                )
                / max(float(np.sum(intensity)), np.finfo(float).eps)
            )
        )
        all_finite = bool(
            all_finite
            and np.all(np.isfinite(detector))
            and np.all(np.isfinite(pixels))
        )
        all_nonnegative = bool(
            all_nonnegative
            and np.all(intensity >= 0.0)
            and np.all(pixels >= 0.0)
        )
        del shifted, detector, intensity, pixels

    cases: list[dict[str, Any]] = []
    support_mapping_errors: list[float] = []
    exterior_errors: list[float] = []
    unit_modulus_errors: list[float] = []
    taper_range_errors: list[float] = []
    taper_endpoint_errors: list[float] = []
    support_margins: list[float] = []
    determinism_error = float("nan")
    max_scan = float(np.max(np.abs(positions)))
    fov_half = float(np.asarray(sampling["fov_m"], dtype=float)[0] / 2.0)
    for width_m in widths:
        support_pixels = _require_integer_ratio(
            width_m, dx_m, "R6 support width"
        )
        raw_pattern = _r6_periodic_support_pattern(
            base_period, (support_pixels, support_pixels)
        )
        for taper_m in tapers:
            taper_pixels = (
                0
                if taper_m == 0.0
                else _require_integer_ratio(taper_m, dx_m, "R6 taper width")
            )
            transmission, taper_weights = _r6_phase_tapered_transmission(
                raw_pattern, taper_pixels
            )
            modulation = _center_pad(transmission - 1.0, shape)
            finite_b = 1.0 + modulation
            support_mask = _center_pad(
                np.ones(transmission.shape, dtype=np.float64), shape
            ).astype(bool)
            support_mapping_errors.append(
                float(
                    np.max(
                        np.abs(
                            center_crop(finite_b, transmission.shape)
                            - transmission
                        )
                    )
                )
            )
            exterior_errors.append(
                float(np.max(np.abs(finite_b[~support_mask] - 1.0)))
            )
            unit_modulus_errors.append(
                float(np.max(np.abs(np.abs(finite_b) - 1.0)))
            )
            taper_range_errors.append(
                float(
                    max(
                        0.0,
                        -float(np.min(taper_weights)),
                        float(np.max(taper_weights)) - 1.0,
                    )
                )
            )
            if taper_pixels == 0:
                endpoint_error = 0.0
            else:
                edge_values = np.concatenate(
                    (
                        taper_weights[0],
                        taper_weights[-1],
                        taper_weights[:, 0],
                        taper_weights[:, -1],
                    )
                )
                interior = taper_weights[
                    taper_pixels:-taper_pixels,
                    taper_pixels:-taper_pixels,
                ]
                endpoint_error = max(
                    float(np.max(np.abs(edge_values))),
                    float(np.max(np.abs(interior - 1.0))),
                )
            taper_endpoint_errors.append(endpoint_error)
            support_margins.append(fov_half - width_m / 2.0 - max_scan)

            intensity_stack = np.empty_like(periodic_stack)
            for scan_index, position_xy in enumerate(positions):
                shifted_modulation = shift_field_integer_pixels(
                    modulation,
                    position_xy,
                    dx_m,
                    boundary="constant",
                    fill_value=0.0j,
                )
                detector = apply_angular_spectrum_transfer(
                    probe * (1.0 + shifted_modulation), transfer_bc
                )
                intensity = np.abs(detector) ** 2
                pixels = positive_midpoint_pixel_average(intensity, factor)
                intensity_stack[scan_index] = center_crop(pixels, native_roi)
                sum_errors.append(
                    float(
                        abs(
                            float(np.sum(pixels)) * factor**2
                            - float(np.sum(intensity))
                        )
                        / max(float(np.sum(intensity)), np.finfo(float).eps)
                    )
                )
                all_finite = bool(
                    all_finite
                    and np.all(np.isfinite(detector))
                    and np.all(np.isfinite(pixels))
                )
                all_nonnegative = bool(
                    all_nonnegative
                    and np.all(intensity >= 0.0)
                    and np.all(pixels >= 0.0)
                )
                if (
                    width_m == nominal_width
                    and taper_m == nominal_taper
                    and scan_index == 0
                ):
                    repeated_detector = apply_angular_spectrum_transfer(
                        probe * (1.0 + shifted_modulation), transfer_bc
                    )
                    repeated_pixels = positive_midpoint_pixel_average(
                        np.abs(repeated_detector) ** 2, factor
                    )
                    determinism_error = relative_l2(
                        center_crop(repeated_pixels, native_roi),
                        intensity_stack[0],
                    )
                    del repeated_detector, repeated_pixels
                del shifted_modulation, detector, intensity, pixels
            cases.append(
                {
                    "support_width_m": width_m,
                    "edge_taper_width_m": taper_m,
                    "I_stack": intensity_stack,
                }
            )
            del modulation, finite_b, support_mask, transmission, taper_weights
        del raw_pattern
        gc.collect()

    nominal_case = next(
        case
        for case in cases
        if case["support_width_m"] == nominal_width
        and case["edge_taper_width_m"] == nominal_taper
    )
    support_effects = np.asarray(
        [relative_l2(periodic_stack, case["I_stack"]) for case in cases],
        dtype=np.float64,
    )
    nominal_differences = np.asarray(
        [
            relative_l2(case["I_stack"], nominal_case["I_stack"])
            for case in cases
        ],
        dtype=np.float64,
    )
    effect_matrix = support_effects.reshape(len(widths), len(tapers))
    nominal_matrix = nominal_differences.reshape(len(widths), len(tapers))
    material_matrix = effect_matrix > materiality_threshold
    robust_materiality = bool(np.all(material_matrix))
    nominal_flat_index = widths.index(nominal_width) * len(tapers) + tapers.index(
        nominal_taper
    )
    frozen_nominal_effect = float(
        _section(diagnostics, "r5_provenance")["support_relative_l2"]
    )
    nominal_provenance_error = float(
        abs(float(support_effects[nominal_flat_index]) - frozen_nominal_effect)
        / max(abs(frozen_nominal_effect), np.finfo(float).eps)
    )
    nominal_provenance_applicable = bool(
        shape == (1536, 1536)
        and np.isclose(dx_m, 1.25e-7, rtol=0.0, atol=0.0)
        and len(positions) == 25
    )
    nominal_provenance_control_error = (
        nominal_provenance_error if nominal_provenance_applicable else 0.0
    )
    min_index = int(np.argmin(support_effects))
    max_index = int(np.argmax(support_effects))
    constant = positive_midpoint_pixel_average(
        np.ones((16 * factor, 16 * factor), dtype=np.float64), factor
    )
    constant_error = float(np.max(np.abs(constant - 1.0)))
    sum_error_max = float(max(sum_errors))
    determinism_pass = bool(
        np.isfinite(determinism_error)
        and determinism_error <= determinism_threshold
    )
    controls_max = float(
        max(
            max(support_mapping_errors),
            max(exterior_errors),
            max(unit_modulus_errors),
            max(taper_range_errors),
            max(taper_endpoint_errors),
            constant_error,
            sum_error_max,
            nominal_provenance_control_error,
        )
    )
    hard_checks_pass = bool(
        fine_validation["pass"]
        and controls_max <= algebra_threshold
        and min(support_margins) > 0.0
        and determinism_pass
        and all_finite
        and all_nonnegative
    )
    if not hard_checks_pass:
        status = "Failed"
    elif robust_materiality:
        status = "Passed"
    else:
        status = "Inconclusive"
    case_widths = np.asarray(
        [case["support_width_m"] for case in cases], dtype=np.float64
    )
    case_tapers = np.asarray(
        [case["edge_taper_width_m"] for case in cases], dtype=np.float64
    )
    metrics = {
        "version": "R6",
        "methods": dict(_section(diagnostics, "methods")),
        "r5_provenance": dict(_section(diagnostics, "r5_provenance")),
        "sampling": {
            "quadrature_factor": factor,
            "node_dx_m": dx_m,
            "fov_m": np.asarray(sampling["fov_m"], dtype=np.float64),
            "node_shape": np.asarray(shape, dtype=np.int64),
            "native_roi_shape": np.asarray(native_roi, dtype=np.int64),
            "scan_count": int(len(positions)),
            "case_count": int(len(cases)),
            "full_node_stacks_retained": False,
        },
        "support_family": {
            "support_width_m": np.asarray(widths, dtype=np.float64),
            "edge_taper_width_m": np.asarray(tapers, dtype=np.float64),
            "nominal_support_width_m": nominal_width,
            "nominal_edge_taper_width_m": nominal_taper,
            "case_support_width_m": case_widths,
            "case_edge_taper_width_m": case_tapers,
            "exterior_transmission": np.asarray([1.0, 0.0]),
        },
        "support_effects": {
            "relative_l2_matrix": effect_matrix,
            "material_matrix": material_matrix,
            "minimum": float(np.min(support_effects)),
            "maximum": float(np.max(support_effects)),
            "span": float(np.max(support_effects) - np.min(support_effects)),
            "all_cases_material": robust_materiality,
        },
        "nominal_sensitivity": {
            "relative_l2_matrix": nominal_matrix,
            "maximum": float(np.max(nominal_differences)),
        },
        "selected_cases": {
            "minimum_effect": {
                "support_width_m": float(case_widths[min_index]),
                "edge_taper_width_m": float(case_tapers[min_index]),
                "support_effect_relative_l2": float(support_effects[min_index]),
            },
            "maximum_effect": {
                "support_width_m": float(case_widths[max_index]),
                "edge_taper_width_m": float(case_tapers[max_index]),
                "support_effect_relative_l2": float(support_effects[max_index]),
            },
        },
        "controls": {
            "support_mapping_max_complex_error": float(
                max(support_mapping_errors)
            ),
            "exterior_max_complex_error": float(max(exterior_errors)),
            "unit_modulus_max_abs_error": float(max(unit_modulus_errors)),
            "taper_range_max_error": float(max(taper_range_errors)),
            "taper_endpoint_max_error": float(max(taper_endpoint_errors)),
            "minimum_shifted_support_margin_m": float(min(support_margins)),
            "constant_max_abs_error": constant_error,
            "max_sum_relative_error": sum_error_max,
            "nominal_r5_provenance_relative_error": nominal_provenance_error,
            "nominal_r5_provenance_applicable": nominal_provenance_applicable,
            "all_outputs_nonnegative": bool(all_nonnegative),
            "pass": bool(
                controls_max <= algebra_threshold and min(support_margins) > 0.0
            ),
        },
        "determinism": {
            "support_width_m": nominal_width,
            "edge_taper_width_m": nominal_taper,
            "scan_index": 0,
            "I_stack_relative_l2": determinism_error,
            "pass": determinism_pass,
        },
        "thresholds": {
            "materiality_relative_l2": materiality_threshold,
            "algebra_relative_l2_max": algebra_threshold,
            "determinism_relative_l2_max": determinism_threshold,
        },
        "outcome_flags": {
            "interpretation_code": _r6_outcome_code(status=status)
        },
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    result = {
        "selected_scan0": {
            "periodic": periodic_stack[0].copy(),
            "nominal": nominal_case["I_stack"][0].copy(),
            "minimum_effect": cases[min_index]["I_stack"][0].copy(),
            "maximum_effect": cases[max_index]["I_stack"][0].copy(),
        },
        "metrics": {
            "support_family": metrics["support_family"],
            "support_effects": metrics["support_effects"],
            "nominal_sensitivity": metrics["nominal_sensitivity"],
            "selected_cases": metrics["selected_cases"],
            "thresholds": metrics["thresholds"],
        },
    }
    del (
        base_period,
        cases,
        coarse_cells,
        fine_base,
        periodic_b,
        periodic_stack,
        probe,
        transfer_bc,
    )
    gc.collect()
    return result, metrics


def _r7_outcome_code(
    *, status: str, binary_material_by_output: Mapping[str, bool]
) -> str:
    """Return the frozen R7 interface interpretation code."""

    if status == "Failed":
        return "subvoxel_interface_attribution_blocked"
    if status == "Inconclusive":
        return "subvoxel_interface_quadrature_not_converged"
    if any(binary_material_by_output.values()):
        return "binary_interface_material_for_at_least_one_output"
    return "binary_interface_nonmaterial_on_registered_outputs"


def _r7_streamed_tgv_exit(
    *,
    incident: NDArray[np.complex128],
    shape: tuple[int, int],
    dx_m: float,
    widths: NDArray[np.float64],
    diameters: NDArray[np.float64],
    interface_factor: int,
    center_xy_m: tuple[float, float],
    n_glass: float,
    n_air: float,
    wavelength: float,
    n_ref: float,
    bandlimit: bool,
    selected_slice_index: int,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], dict[str, Any]]:
    """Propagate one R7 interface case and return compact controls."""

    x_grid, y_grid = coordinate_grid(shape, dx_m)
    radius_squared_grid = (
        (x_grid - center_xy_m[0]) ** 2 + (y_grid - center_xy_m[1]) ** 2
    )
    selected_fraction: NDArray[np.float64] | None = None
    discrete_volume = 0.0
    fraction_bound_error = 0.0
    index_bound_error = 0.0
    count_identity_error = 0.0
    q1_identity_error = 0.0
    all_finite = True

    def slices() -> Any:
        nonlocal selected_fraction
        nonlocal discrete_volume
        nonlocal fraction_bound_error
        nonlocal index_bound_error
        nonlocal count_identity_error
        nonlocal q1_identity_error
        nonlocal all_finite
        for slice_index, (diameter, width) in enumerate(
            zip(diameters, widths, strict=True)
        ):
            fraction = make_tgv_air_fraction_slice(
                shape,
                dx_m,
                float(diameter),
                interface_factor,
                center_xy_m,
            )
            if slice_index == selected_slice_index:
                selected_fraction = fraction.copy()
            fraction_bound_error = max(
                fraction_bound_error,
                max(
                    0.0,
                    -float(np.min(fraction)),
                    float(np.max(fraction)) - 1.0,
                ),
            )
            scaled_counts = fraction * interface_factor**2
            count_identity_error = max(
                count_identity_error,
                float(np.max(np.abs(scaled_counts - np.rint(scaled_counts)))),
            )
            if interface_factor == 1:
                expected = radius_squared_grid <= (float(diameter) / 2.0) ** 2
                q1_identity_error = max(
                    q1_identity_error,
                    float(np.max(np.abs(fraction - expected.astype(float)))),
                )
            n_slice = n_glass + fraction * (n_air - n_glass)
            lower, upper = sorted((n_air, n_glass))
            index_bound_error = max(
                index_bound_error,
                max(
                    0.0,
                    lower - float(np.min(n_slice)),
                    float(np.max(n_slice)) - upper,
                ),
            )
            discrete_volume += float(np.sum(fraction)) * dx_m**2 * float(width)
            all_finite = bool(
                all_finite
                and np.all(np.isfinite(fraction))
                and np.all(np.isfinite(n_slice))
            )
            yield n_slice

    a_exit = multislice_propagate_streamed_A(
        incident,
        slices(),
        dx_m,
        widths,
        wavelength,
        n_ref=n_ref,
        bandlimit=bandlimit,
    )
    if selected_fraction is None:
        msg = "R7 selected interface slice was not generated."
        raise RuntimeError(msg)
    return a_exit, selected_fraction, {
        "discrete_air_volume_m3": discrete_volume,
        "fraction_bound_error": fraction_bound_error,
        "index_bound_error": index_bound_error,
        "count_identity_error": count_identity_error,
        "q1_identity_error": q1_identity_error,
        "all_finite": all_finite,
    }


def _run_r7_diagnostics(
    config: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the R7 streamed subvoxel-interface/open-detector comparison."""

    diagnostics = _section(config, "diagnostics_r7")
    interface = _section(diagnostics, "interface")
    a_sampling = _section(diagnostics, "sample_a_sampling")
    detector_sampling = _section(diagnostics, "detector_sampling")
    finite_b = _section(diagnostics, "finite_b")
    optics = _section(config, "optics")
    illumination = _section(config, "illumination")
    sample_a = _section(config, "sample_a")
    acceptance = _section(config, "acceptance")

    factors = [int(value) for value in interface["factors"]]
    a_shape = _shape(a_sampling["shape"], "R7 sample-A shape")
    a_dx = float(a_sampling["dx_m"])
    dz_m = float(a_sampling["dz_m"])
    q_detector = int(detector_sampling["quadrature_factor"])
    node_dx = float(detector_sampling["node_dx_m"])
    base_shape = _shape(detector_sampling["base_node_shape"], "R7 base shape")
    open_shape = _shape(detector_sampling["open_node_shape"], "R7 open shape")
    native_roi = _shape(
        detector_sampling["native_roi_shape"], "R7 native ROI"
    )
    selected_scan_index = int(detector_sampling["selected_scan_index"])
    threshold = float(acceptance["convergence_relative_l2_max"])
    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    determinism_threshold = float(acceptance["determinism_relative_l2_max"])
    wavelength = float(optics["wavelength_m"])
    n_ref = float(optics["internal_reference_index"])
    external_index = float(optics["external_medium_index"])
    z_ab = float(optics["z_AB_m"])
    z_bc = float(optics["z_BC_m"])
    n_glass = float(sample_a["n_glass"])
    n_air = float(sample_a["n_air"])
    center_xy = tuple(float(value) for value in sample_a["center_xy_m"])

    z_m, widths = midpoint_z_grid(float(sample_a["thickness_m"]), dz_m)
    waist_depth = validate_tgv_geometry(
        float(sample_a["thickness_m"]),
        float(sample_a["d_top_m"]),
        float(sample_a["d_waist_m"]),
        float(sample_a["d_bottom_m"]),
        float(sample_a["z_waist_m"]),
    )
    diameters = diameter_profile(
        z_m,
        float(sample_a["thickness_m"]),
        float(sample_a["d_top_m"]),
        float(sample_a["d_waist_m"]),
        float(sample_a["d_bottom_m"]),
        waist_depth,
    )
    selected_slice_index = int(np.argmin(np.abs(z_m - waist_depth)))
    incident = make_plane_wave(
        a_shape,
        a_dx,
        wavelength,
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    homogeneous_a_exit = _r5_homogeneous_a_exit(config, a_shape, a_dx)

    fine_base, _, fine_dx, fine_validation = _make_r1_canonical_b(
        _section(config, "sample_b"),
        _section(config, "diagnostics_r1"),
        legacy_canonical_b,
        legacy_canonical_dx_m,
    )
    feature_m = float(_section(config, "sample_b")["physical_feature_size_m"])
    feature_pixels_fine = _require_integer_ratio(
        feature_m, fine_dx, "R7 fine B feature size"
    )
    coarse_y = fine_base.shape[0] // feature_pixels_fine
    coarse_x = fine_base.shape[1] // feature_pixels_fine
    coarse_cells = fine_base.reshape(
        coarse_y, feature_pixels_fine, coarse_x, feature_pixels_fine
    )[:, feature_pixels_fine // 2, :, feature_pixels_fine // 2]
    feature_pixels = _require_integer_ratio(
        feature_m, node_dx, "R7 B feature size"
    )
    base_period = np.repeat(
        np.repeat(coarse_cells, feature_pixels, axis=0),
        feature_pixels,
        axis=1,
    ).astype(np.complex128, copy=False)
    support_shape = tuple(
        _require_integer_ratio(float(value), node_dx, "R7 B support")
        for value in finite_b["physical_shape_m"]
    )
    if base_period.shape != support_shape:
        msg = "R7 canonical phase cells do not fill the finite support."
        raise RuntimeError(msg)
    modulation_open = _center_pad(base_period - 1.0, open_shape)
    support_mask = _center_pad(
        np.ones(base_period.shape, dtype=np.float64), open_shape
    ).astype(bool)
    b_mapping_error = float(
        np.max(
            np.abs(
                center_crop(1.0 + modulation_open, base_period.shape)
                - base_period
            )
        )
    )
    b_exterior_error = float(
        np.max(np.abs((1.0 + modulation_open)[~support_mask] - 1.0))
    )
    b_modulus_error = float(np.max(np.abs(np.abs(base_period) - 1.0)))

    homogeneous_exit_base = _r5_homogeneous_a_exit(config, base_shape, node_dx)
    transfer_ab_base = make_angular_spectrum_transfer(
        base_shape,
        node_dx,
        wavelength,
        z_ab,
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_probe_base = apply_angular_spectrum_transfer(
        homogeneous_exit_base, transfer_ab_base
    )
    homogeneous_exit_open = _r5_homogeneous_a_exit(config, open_shape, node_dx)
    transfer_ab_open = make_angular_spectrum_transfer(
        open_shape,
        node_dx,
        wavelength,
        z_ab,
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_probe_open = apply_angular_spectrum_transfer(
        homogeneous_exit_open, transfer_ab_open
    )
    del homogeneous_exit_base, homogeneous_exit_open, transfer_ab_open
    transfer_bc_open = make_angular_spectrum_transfer(
        open_shape,
        node_dx,
        wavelength,
        z_bc,
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_detector_open = apply_angular_spectrum_transfer(
        homogeneous_probe_open, transfer_bc_open
    )

    mapped_shape = tuple(
        _require_integer_ratio(
            a_shape[index] * a_dx, node_dx, "R7 A mapping"
        )
        for index in range(2)
    )
    continuous_midpoint_volume = float(
        np.sum(np.pi * (diameters / 2.0) ** 2 * widths)
    )
    thickness_error = float(
        abs(float(np.sum(widths)) - float(sample_a["thickness_m"]))
    )
    geometry_rule = _section(
        acceptance, "geometry_thickness_absolute_tolerance_m"
    )
    geometry_tolerance = max(
        float(geometry_rule["fixed_floor_m"]),
        float(geometry_rule["floating_point_factor"])
        * np.finfo(np.float64).eps
        * float(sample_a["thickness_m"]),
    )

    cases: list[dict[str, Any]] = []
    selected_fractions: dict[str, NDArray[np.float64]] = {}
    fraction_bound_errors: list[float] = []
    index_bound_errors: list[float] = []
    count_identity_errors: list[float] = []
    volume_relative_errors: list[float] = []
    q1_identity_error = 0.0
    homogeneous_consistency_error = relative_l2(
        multislice_propagate_streamed_A(
            incident,
            (np.full(a_shape, n_ref, dtype=np.float64) for _ in widths),
            a_dx,
            widths,
            wavelength,
            n_ref=n_ref,
            bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        ),
        homogeneous_a_exit,
    )
    sum_errors: list[float] = []
    all_finite = bool(fine_validation["pass"])
    all_nonnegative = True
    determinism_error = float("nan")
    selected_detectors: dict[str, NDArray[np.float64]] = {}

    for factor in factors:
        a_exit, selected_fraction, interface_controls = _r7_streamed_tgv_exit(
            incident=incident,
            shape=a_shape,
            dx_m=a_dx,
            widths=widths,
            diameters=diameters,
            interface_factor=factor,
            center_xy_m=center_xy,
            n_glass=n_glass,
            n_air=n_air,
            wavelength=wavelength,
            n_ref=n_ref,
            bandlimit=bool(optics["angular_spectrum_bandlimit"]),
            selected_slice_index=selected_slice_index,
        )
        selected_fractions[f"q{factor}"] = selected_fraction
        fraction_bound_errors.append(
            float(interface_controls["fraction_bound_error"])
        )
        index_bound_errors.append(float(interface_controls["index_bound_error"]))
        count_identity_errors.append(
            float(interface_controls["count_identity_error"])
        )
        q1_identity_error = max(
            q1_identity_error, float(interface_controls["q1_identity_error"])
        )
        all_finite = bool(all_finite and interface_controls["all_finite"])
        discrete_volume = float(interface_controls["discrete_air_volume_m3"])
        volume_relative_errors.append(
            float(
                abs(discrete_volume - continuous_midpoint_volume)
                / max(continuous_midpoint_volume, np.finfo(float).eps)
            )
        )
        residual = a_exit - homogeneous_a_exit
        residual_with_ghost = np.pad(residual, ((1, 1), (1, 1)), mode="edge")
        mapped_residual = resample_centered_grid(
            residual_with_ghost, a_dx, mapped_shape, node_dx
        )
        residual_exit_base = _center_pad(mapped_residual, base_shape)
        delta_probe_base = apply_angular_spectrum_transfer(
            residual_exit_base, transfer_ab_base
        )
        probe_base = homogeneous_probe_base + delta_probe_base
        probe_roi = center_crop(
            _r4_block_mean(probe_base, q_detector), native_roi
        ).copy()
        delta_probe_open = _center_pad(delta_probe_base, open_shape)
        probe_open = homogeneous_probe_open + delta_probe_open
        intensity_stack = np.empty(
            (len(positions), *native_roi), dtype=np.float64
        )
        for scan_index, position_xy in enumerate(positions):
            shifted_modulation = shift_field_integer_pixels(
                modulation_open,
                position_xy,
                node_dx,
                boundary="constant",
                fill_value=0.0j,
            )
            residual_exit = delta_probe_open + probe_open * shifted_modulation
            residual_detector = apply_angular_spectrum_transfer(
                residual_exit, transfer_bc_open
            )
            detector_field = homogeneous_detector_open + residual_detector
            intensity = np.abs(detector_field) ** 2
            pixels = positive_midpoint_pixel_average(intensity, q_detector)
            intensity_stack[scan_index] = center_crop(pixels, native_roi)
            sum_errors.append(
                float(
                    abs(
                        float(np.sum(pixels)) * q_detector**2
                        - float(np.sum(intensity))
                    )
                    / max(float(np.sum(intensity)), np.finfo(float).eps)
                )
            )
            all_finite = bool(
                all_finite
                and np.all(np.isfinite(detector_field))
                and np.all(np.isfinite(pixels))
            )
            all_nonnegative = bool(
                all_nonnegative
                and np.all(intensity >= 0.0)
                and np.all(pixels >= 0.0)
            )
            if scan_index == selected_scan_index:
                selected_detectors[f"q{factor}"] = center_crop(
                    pixels, native_roi
                ).copy()
            if factor == 8 and scan_index == selected_scan_index:
                repeated_residual = apply_angular_spectrum_transfer(
                    residual_exit, transfer_bc_open
                )
                repeated_detector = homogeneous_detector_open + repeated_residual
                repeated_pixels = positive_midpoint_pixel_average(
                    np.abs(repeated_detector) ** 2, q_detector
                )
                determinism_error = relative_l2(
                    center_crop(repeated_pixels, native_roi),
                    intensity_stack[scan_index],
                )
                del repeated_residual, repeated_detector, repeated_pixels
            del (
                shifted_modulation,
                residual_exit,
                residual_detector,
                detector_field,
                intensity,
                pixels,
            )
        cases.append(
            {
                "factor": factor,
                "U_A_exit": a_exit,
                "P_B": probe_roi,
                "I_stack": intensity_stack,
            }
        )
        del (
            residual,
            residual_with_ghost,
            mapped_residual,
            residual_exit_base,
            delta_probe_base,
            probe_base,
            delta_probe_open,
            probe_open,
        )
        gc.collect()

    reference = cases[-1]
    series = {
        name: np.asarray(
            [relative_l2(case[name], reference[name]) for case in cases],
            dtype=np.float64,
        )
        for name in ("U_A_exit", "P_B", "I_stack")
    }
    final_pair = {
        name: relative_l2(cases[-2][name], reference[name])
        for name in ("U_A_exit", "P_B", "I_stack")
    }
    binary_effect = {
        name: relative_l2(cases[0][name], reference[name])
        for name in ("U_A_exit", "P_B", "I_stack")
    }
    convergence_pass = {
        name: bool(value <= threshold) for name, value in final_pair.items()
    }
    binary_material = {
        name: bool(value > threshold) for name, value in binary_effect.items()
    }
    constant = positive_midpoint_pixel_average(
        np.ones((16 * q_detector, 16 * q_detector), dtype=np.float64),
        q_detector,
    )
    constant_error = float(np.max(np.abs(constant - 1.0)))
    detector_pixel_m = float(_section(optics, "detector")["pixel_size_m"])
    geometry_error = max(
        _r4_node_geometry_error(
            open_shape[0] // q_detector, detector_pixel_m, q_detector
        ),
        _r4_node_geometry_error(
            open_shape[1] // q_detector, detector_pixel_m, q_detector
        ),
    )
    sum_error_max = float(max(sum_errors))
    determinism_pass = bool(
        np.isfinite(determinism_error)
        and determinism_error <= determinism_threshold
    )
    algebra_errors = [
        q1_identity_error,
        homogeneous_consistency_error,
        *fraction_bound_errors,
        *index_bound_errors,
        *count_identity_errors,
        b_mapping_error,
        b_exterior_error,
        b_modulus_error,
        constant_error,
        geometry_error,
        sum_error_max,
    ]
    hard_checks_pass = bool(
        fine_validation["pass"]
        and max(algebra_errors) <= algebra_threshold
        and thickness_error <= geometry_tolerance
        and determinism_pass
        and all_finite
        and all_nonnegative
    )
    if not hard_checks_pass:
        status = "Failed"
    elif all(convergence_pass.values()):
        status = "Passed"
    else:
        status = "Inconclusive"
    metrics = {
        "version": "R7",
        "methods": dict(_section(diagnostics, "methods")),
        "r6_provenance": dict(_section(diagnostics, "r6_provenance")),
        "sampling": {
            "interface_factors": np.asarray(factors, dtype=np.int64),
            "sample_a_shape": np.asarray(a_shape, dtype=np.int64),
            "sample_a_dx_m": a_dx,
            "sample_a_dz_m": dz_m,
            "slice_count": int(len(widths)),
            "detector_quadrature_factor": q_detector,
            "detector_node_dx_m": node_dx,
            "base_node_shape": np.asarray(base_shape, dtype=np.int64),
            "open_node_shape": np.asarray(open_shape, dtype=np.int64),
            "native_roi_shape": np.asarray(native_roi, dtype=np.int64),
            "scan_count": int(len(positions)),
            "full_volumes_retained": False,
            "full_node_stacks_retained": False,
        },
        "interface_controls": {
            "q1_binary_identity_max_abs_error": q1_identity_error,
            "homogeneous_streamed_relative_l2": homogeneous_consistency_error,
            "fraction_bound_error_by_factor": np.asarray(
                fraction_bound_errors
            ),
            "index_bound_error_by_factor": np.asarray(index_bound_errors),
            "subnode_count_identity_error_by_factor": np.asarray(
                count_identity_errors
            ),
            "continuous_midpoint_air_volume_m3": continuous_midpoint_volume,
            "discrete_air_volume_relative_error_by_factor": np.asarray(
                volume_relative_errors
            ),
            "slice_width_sum_absolute_error_m": thickness_error,
            "slice_width_sum_tolerance_m": geometry_tolerance,
        },
        "finite_b_controls": {
            "support_shape": np.asarray(support_shape, dtype=np.int64),
            "mapping_max_complex_error": b_mapping_error,
            "exterior_max_complex_error": b_exterior_error,
            "unit_modulus_max_abs_error": b_modulus_error,
        },
        "detector_controls": {
            "node_geometry_normalized_error": geometry_error,
            "constant_max_abs_error": constant_error,
            "max_sum_relative_error": sum_error_max,
            "weights": {
                "minimum": 1.0 / q_detector**2,
                "maximum": 1.0 / q_detector**2,
                "all_nonnegative": True,
                "sum_one_max_abs_error": 0.0,
            },
            "all_outputs_nonnegative": bool(all_nonnegative),
        },
        "convergence": {
            "relative_to_q8": series,
            "acceptance_pair_factors": np.asarray([4, 8], dtype=np.int64),
            "acceptance": final_pair,
            "pass": convergence_pass,
        },
        "binary_effect": {
            "relative_l2_q1_to_q8": binary_effect,
            "material_by_output": binary_material,
        },
        "determinism": {
            "interface_factor": 8,
            "scan_index": selected_scan_index,
            "I_stack_relative_l2": determinism_error,
            "pass": determinism_pass,
        },
        "model_uncertainty_context": {
            "r6_maximum_nominal_b_variation": float(
                _section(diagnostics, "r6_provenance")[
                    "maximum_nominal_b_variation"
                ]
            ),
            "combined_with_r7_metrics": False,
        },
        "thresholds": {
            "convergence_and_materiality_relative_l2": threshold,
            "algebra_relative_l2_max": algebra_threshold,
            "determinism_relative_l2_max": determinism_threshold,
        },
        "outcome_flags": {
            "interpretation_code": _r7_outcome_code(
                status=status, binary_material_by_output=binary_material
            )
        },
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    result = {
        "selected_slice_index": selected_slice_index,
        "selected_z_m": float(z_m[selected_slice_index]),
        "selected_fractions": selected_fractions,
        "selected_scan0": selected_detectors,
        "metrics": {
            "sampling": metrics["sampling"],
            "interface_controls": metrics["interface_controls"],
            "convergence": metrics["convergence"],
            "binary_effect": metrics["binary_effect"],
            "thresholds": metrics["thresholds"],
        },
    }
    del (
        base_period,
        cases,
        coarse_cells,
        fine_base,
        homogeneous_a_exit,
        homogeneous_detector_open,
        homogeneous_probe_base,
        homogeneous_probe_open,
        incident,
        modulation_open,
        support_mask,
        transfer_ab_base,
        transfer_bc_open,
    )
    gc.collect()
    return result, metrics


def _r8_outcome_code(
    *, status: str, convergence_pass: bool, visibility_pass: bool
) -> str:
    """Return the frozen R8 unified-forward interpretation code."""

    if status == "Failed":
        return "unified_forward_attribution_blocked"
    if not convergence_pass:
        return "unified_numerical_floor_not_closed"
    if not visibility_pass:
        return "waist_signal_not_above_registered_floor"
    return "waist_signal_resolved_within_registered_working_model"


def _r8_open_context(
    config: Mapping[str, Any],
    *,
    open_shape: tuple[int, int],
    node_dx_m: float,
    base_period: NDArray[np.complex128],
) -> dict[str, Any]:
    """Build one finite-B residual open-propagation context for R8."""

    optics = _section(config, "optics")
    wavelength = float(optics["wavelength_m"])
    external_index = float(optics["external_medium_index"])
    homogeneous_exit = _r5_homogeneous_a_exit(config, open_shape, node_dx_m)
    transfer_ab = make_angular_spectrum_transfer(
        open_shape,
        node_dx_m,
        wavelength,
        float(optics["z_AB_m"]),
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_probe = apply_angular_spectrum_transfer(
        homogeneous_exit, transfer_ab
    )
    transfer_bc = make_angular_spectrum_transfer(
        open_shape,
        node_dx_m,
        wavelength,
        float(optics["z_BC_m"]),
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_detector = apply_angular_spectrum_transfer(
        homogeneous_probe, transfer_bc
    )
    modulation = _center_pad(base_period - 1.0, open_shape)
    del homogeneous_exit, transfer_ab
    return {
        "shape": open_shape,
        "homogeneous_probe": homogeneous_probe,
        "homogeneous_detector": homogeneous_detector,
        "transfer_bc": transfer_bc,
        "modulation": modulation,
    }


def _r8_detector_stack(
    *,
    delta_probe_base: NDArray[np.complex128],
    positions: NDArray[np.float64],
    context: Mapping[str, Any],
    node_dx_m: float,
    quadrature_factor: int,
    native_roi: tuple[int, int],
    selected_scan_index: int,
    repeat_selected_scan: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Propagate one R8 probe residual through one registered open grid."""

    open_shape = _shape(context["shape"], "R8 open context shape")
    homogeneous_probe = np.asarray(
        context["homogeneous_probe"], dtype=np.complex128
    )
    homogeneous_detector = np.asarray(
        context["homogeneous_detector"], dtype=np.complex128
    )
    transfer_bc = np.asarray(context["transfer_bc"], dtype=np.complex128)
    modulation = np.asarray(context["modulation"], dtype=np.complex128)
    delta_probe_open = _center_pad(delta_probe_base, open_shape)
    probe_open = homogeneous_probe + delta_probe_open
    intensity_stack = np.empty(
        (len(positions), *native_roi), dtype=np.float64
    )
    selected: NDArray[np.float64] | None = None
    sum_errors: list[float] = []
    all_finite = True
    all_nonnegative = True
    determinism_error = float("nan")

    for scan_index, position_xy in enumerate(positions):
        shifted_modulation = shift_field_integer_pixels(
            modulation,
            position_xy,
            node_dx_m,
            boundary="constant",
            fill_value=0.0j,
        )
        residual_exit = delta_probe_open + probe_open * shifted_modulation
        residual_detector = apply_angular_spectrum_transfer(
            residual_exit, transfer_bc
        )
        detector_field = homogeneous_detector + residual_detector
        intensity = np.abs(detector_field) ** 2
        pixels = positive_midpoint_pixel_average(
            intensity, quadrature_factor
        )
        cropped = center_crop(pixels, native_roi)
        intensity_stack[scan_index] = cropped
        if scan_index == selected_scan_index:
            selected = cropped.copy()
        sum_errors.append(
            float(
                abs(
                    float(np.sum(pixels)) * quadrature_factor**2
                    - float(np.sum(intensity))
                )
                / max(float(np.sum(intensity)), np.finfo(float).eps)
            )
        )
        all_finite = bool(
            all_finite
            and np.all(np.isfinite(detector_field))
            and np.all(np.isfinite(pixels))
        )
        all_nonnegative = bool(
            all_nonnegative
            and np.all(intensity >= 0.0)
            and np.all(pixels >= 0.0)
        )
        if repeat_selected_scan and scan_index == selected_scan_index:
            repeated_residual = apply_angular_spectrum_transfer(
                residual_exit, transfer_bc
            )
            repeated_detector = homogeneous_detector + repeated_residual
            repeated_pixels = positive_midpoint_pixel_average(
                np.abs(repeated_detector) ** 2, quadrature_factor
            )
            determinism_error = relative_l2(
                center_crop(repeated_pixels, native_roi), cropped
            )
            del repeated_residual, repeated_detector, repeated_pixels
        del (
            shifted_modulation,
            residual_exit,
            residual_detector,
            detector_field,
            intensity,
            pixels,
            cropped,
        )
    if selected is None:
        msg = "R8 selected detector scan was not generated."
        raise RuntimeError(msg)
    del delta_probe_open, probe_open
    return intensity_stack, selected, {
        "max_sum_relative_error": float(max(sum_errors)),
        "all_finite": all_finite,
        "all_nonnegative": all_nonnegative,
        "determinism_relative_l2": determinism_error,
    }


def _run_r8_diagnostics(
    config: Mapping[str, Any],
    positions: NDArray[np.float64],
    legacy_canonical_b: NDArray[np.complex128],
    legacy_canonical_dx_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the R8 unified q8 convergence and waist-visibility study."""

    diagnostics = _section(config, "diagnostics_r8")
    case_registration = _section(diagnostics, "sample_a_cases")
    detector_sampling = _section(diagnostics, "detector_sampling")
    finite_b = _section(diagnostics, "finite_b")
    open_control = _section(diagnostics, "open_control")
    comparisons = _section(diagnostics, "comparisons")
    optics = _section(config, "optics")
    illumination = _section(config, "illumination")
    sample_a = _section(config, "sample_a")
    acceptance = _section(config, "acceptance")

    interface_factor = int(_section(diagnostics, "interface")["factor"])
    q_detector = int(detector_sampling["quadrature_factor"])
    node_dx = float(detector_sampling["node_dx_m"])
    base_shape = _shape(detector_sampling["base_node_shape"], "R8 base")
    primary_open_shape = _shape(
        detector_sampling["primary_open_node_shape"], "R8 primary open"
    )
    native_roi = _shape(
        detector_sampling["native_roi_shape"], "R8 native ROI"
    )
    selected_scan_index = int(detector_sampling["selected_scan_index"])
    convergence_threshold = float(acceptance["convergence_relative_l2_max"])
    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    determinism_threshold = float(acceptance["determinism_relative_l2_max"])
    visibility_threshold = float(
        acceptance["detector_visibility_signal_to_floor_min"]
    )
    wavelength = float(optics["wavelength_m"])
    n_ref = float(optics["internal_reference_index"])
    external_index = float(optics["external_medium_index"])
    n_glass = float(sample_a["n_glass"])
    n_air = float(sample_a["n_air"])
    center_xy = tuple(float(value) for value in sample_a["center_xy_m"])
    thickness = float(sample_a["thickness_m"])
    waist_depth = float(sample_a["z_waist_m"])

    fine_base, _, fine_dx, fine_validation = _make_r1_canonical_b(
        _section(config, "sample_b"),
        _section(config, "diagnostics_r1"),
        legacy_canonical_b,
        legacy_canonical_dx_m,
    )
    feature_m = float(_section(config, "sample_b")["physical_feature_size_m"])
    feature_pixels_fine = _require_integer_ratio(
        feature_m, fine_dx, "R8 fine B feature size"
    )
    coarse_y = fine_base.shape[0] // feature_pixels_fine
    coarse_x = fine_base.shape[1] // feature_pixels_fine
    coarse_cells = fine_base.reshape(
        coarse_y, feature_pixels_fine, coarse_x, feature_pixels_fine
    )[:, feature_pixels_fine // 2, :, feature_pixels_fine // 2]
    feature_pixels = _require_integer_ratio(
        feature_m, node_dx, "R8 B feature size"
    )
    base_period = np.repeat(
        np.repeat(coarse_cells, feature_pixels, axis=0),
        feature_pixels,
        axis=1,
    ).astype(np.complex128, copy=False)
    support_shape = tuple(
        _require_integer_ratio(float(value), node_dx, "R8 B support")
        for value in finite_b["physical_shape_m"]
    )
    if base_period.shape != support_shape:
        msg = "R8 canonical phase cells do not fill the finite support."
        raise RuntimeError(msg)

    primary_context = _r8_open_context(
        config,
        open_shape=primary_open_shape,
        node_dx_m=node_dx,
        base_period=base_period,
    )
    primary_modulation = np.asarray(
        primary_context["modulation"], dtype=np.complex128
    )
    support_mask = _center_pad(
        np.ones(base_period.shape, dtype=np.float64), primary_open_shape
    ).astype(bool)
    b_mapping_error = float(
        np.max(
            np.abs(
                center_crop(1.0 + primary_modulation, base_period.shape)
                - base_period
            )
        )
    )
    b_exterior_error = float(
        np.max(np.abs((1.0 + primary_modulation)[~support_mask] - 1.0))
    )
    b_modulus_error = float(np.max(np.abs(np.abs(base_period) - 1.0)))

    homogeneous_exit_base = _r5_homogeneous_a_exit(config, base_shape, node_dx)
    transfer_ab_base = make_angular_spectrum_transfer(
        base_shape,
        node_dx,
        wavelength,
        float(optics["z_AB_m"]),
        n=external_index,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_probe_base = apply_angular_spectrum_transfer(
        homogeneous_exit_base, transfer_ab_base
    )
    del homogeneous_exit_base

    geometry_rule = _section(
        acceptance, "geometry_thickness_absolute_tolerance_m"
    )
    geometry_tolerance = max(
        float(geometry_rule["fixed_floor_m"]),
        float(geometry_rule["floating_point_factor"])
        * np.finfo(np.float64).eps
        * thickness,
    )
    raw_cases = case_registration["cases"]
    cases: dict[str, dict[str, Any]] = {}
    selected_detectors: dict[str, NDArray[np.float64]] = {}
    case_shapes: list[tuple[int, int]] = []
    case_dx: list[float] = []
    case_dz: list[float] = []
    case_waists: list[float] = []
    slice_counts: list[int] = []
    fraction_errors: list[float] = []
    index_errors: list[float] = []
    count_errors: list[float] = []
    volume_errors: list[float] = []
    thickness_errors: list[float] = []
    detector_sum_errors: list[float] = []
    fine_mapping_identity_errors: list[float] = []
    all_finite = bool(fine_validation["pass"])
    all_nonnegative = True
    determinism_error = float("nan")
    homogeneous_consistency_error = float("nan")
    finest_delta_probe_base: NDArray[np.complex128] | None = None

    for case_cfg in raw_cases:
        case_id = str(case_cfg["id"])
        shape = _shape(case_cfg["shape"], f"R8 {case_id} shape")
        dx_m = float(case_cfg["dx_m"])
        dz_m = float(case_cfg["dz_m"])
        d_waist_m = float(case_cfg["d_waist_m"])
        z_m, widths = midpoint_z_grid(thickness, dz_m)
        diameters = diameter_profile(
            z_m,
            thickness,
            float(sample_a["d_top_m"]),
            d_waist_m,
            float(sample_a["d_bottom_m"]),
            waist_depth,
        )
        selected_slice_index = int(np.argmin(np.abs(z_m - waist_depth)))
        incident = make_plane_wave(
            shape,
            dx_m,
            wavelength,
            theta_x=float(illumination["theta_x_rad"]),
            theta_y=float(illumination["theta_y_rad"]),
            amplitude=float(illumination["amplitude"]),
        )
        homogeneous_a_exit = _r5_homogeneous_a_exit(config, shape, dx_m)
        a_exit, _, interface_controls = _r7_streamed_tgv_exit(
            incident=incident,
            shape=shape,
            dx_m=dx_m,
            widths=widths,
            diameters=diameters,
            interface_factor=interface_factor,
            center_xy_m=center_xy,
            n_glass=n_glass,
            n_air=n_air,
            wavelength=wavelength,
            n_ref=n_ref,
            bandlimit=bool(optics["angular_spectrum_bandlimit"]),
            selected_slice_index=selected_slice_index,
        )
        if case_id == "finest_baseline":
            homogeneous_consistency_error = relative_l2(
                multislice_propagate_streamed_A(
                    incident,
                    (
                        np.full(shape, n_ref, dtype=np.float64)
                        for _ in widths
                    ),
                    dx_m,
                    widths,
                    wavelength,
                    n_ref=n_ref,
                    bandlimit=bool(optics["angular_spectrum_bandlimit"]),
                ),
                homogeneous_a_exit,
            )
        continuous_volume = float(
            np.sum(np.pi * (diameters / 2.0) ** 2 * widths)
        )
        discrete_volume = float(interface_controls["discrete_air_volume_m3"])
        volume_errors.append(
            float(
                abs(discrete_volume - continuous_volume)
                / max(continuous_volume, np.finfo(float).eps)
            )
        )
        thickness_errors.append(abs(float(np.sum(widths)) - thickness))
        fraction_errors.append(float(interface_controls["fraction_bound_error"]))
        index_errors.append(float(interface_controls["index_bound_error"]))
        count_errors.append(float(interface_controls["count_identity_error"]))
        all_finite = bool(all_finite and interface_controls["all_finite"])

        residual = a_exit - homogeneous_a_exit
        residual_with_ghost = np.pad(residual, ((1, 1), (1, 1)), mode="edge")
        mapped_shape = tuple(
            _require_integer_ratio(
                shape[index] * dx_m, node_dx, f"R8 {case_id} A mapping"
            )
            for index in range(2)
        )
        mapped_residual = resample_centered_grid(
            residual_with_ghost, dx_m, mapped_shape, node_dx
        )
        if np.isclose(dx_m, node_dx, rtol=1e-12, atol=0.0):
            fine_mapping_identity_errors.append(
                relative_l2(mapped_residual, residual)
            )
        residual_exit_base = _center_pad(mapped_residual, base_shape)
        delta_probe_base = apply_angular_spectrum_transfer(
            residual_exit_base, transfer_ab_base
        )
        probe_base = homogeneous_probe_base + delta_probe_base
        probe_roi = center_crop(
            _r4_block_mean(probe_base, q_detector), native_roi
        ).copy()
        stack, selected, detector_controls = _r8_detector_stack(
            delta_probe_base=delta_probe_base,
            positions=positions,
            context=primary_context,
            node_dx_m=node_dx,
            quadrature_factor=q_detector,
            native_roi=native_roi,
            selected_scan_index=selected_scan_index,
            repeat_selected_scan=case_id == "finest_baseline",
        )
        detector_sum_errors.append(
            float(detector_controls["max_sum_relative_error"])
        )
        all_finite = bool(all_finite and detector_controls["all_finite"])
        all_nonnegative = bool(
            all_nonnegative and detector_controls["all_nonnegative"]
        )
        if case_id == "finest_baseline":
            determinism_error = float(
                detector_controls["determinism_relative_l2"]
            )
            finest_delta_probe_base = delta_probe_base.copy()
        selected_detectors[case_id] = selected
        cases[case_id] = {
            "U_A_exit": a_exit,
            "P_B": probe_roi,
            "I_stack": stack,
        }
        case_shapes.append(shape)
        case_dx.append(dx_m)
        case_dz.append(dz_m)
        case_waists.append(d_waist_m)
        slice_counts.append(len(widths))
        del (
            incident,
            homogeneous_a_exit,
            residual,
            residual_with_ghost,
            mapped_residual,
            residual_exit_base,
            delta_probe_base,
            probe_base,
        )
        gc.collect()

    if finest_delta_probe_base is None:
        msg = "R8 finest baseline probe residual was not retained."
        raise RuntimeError(msg)
    del primary_context, primary_modulation, support_mask
    gc.collect()

    open_shapes = [
        _shape(value, "R8 open control shape")
        for value in open_control["node_shapes"]
    ]
    open_288_context = _r8_open_context(
        config,
        open_shape=open_shapes[0],
        node_dx_m=node_dx,
        base_period=base_period,
    )
    open_288_stack, open_288_selected, open_288_controls = _r8_detector_stack(
        delta_probe_base=finest_delta_probe_base,
        positions=positions,
        context=open_288_context,
        node_dx_m=node_dx,
        quadrature_factor=q_detector,
        native_roi=native_roi,
        selected_scan_index=selected_scan_index,
        repeat_selected_scan=False,
    )
    detector_sum_errors.append(
        float(open_288_controls["max_sum_relative_error"])
    )
    all_finite = bool(all_finite and open_288_controls["all_finite"])
    all_nonnegative = bool(
        all_nonnegative and open_288_controls["all_nonnegative"]
    )
    del open_288_context, finest_delta_probe_base
    gc.collect()

    axial_test = cases["axial_coarse"]
    axial_reference = cases["common_reference"]
    finest = cases["finest_baseline"]
    axial_acceptance = {
        name: relative_l2(axial_test[name], axial_reference[name])
        for name in ("U_A_exit", "P_B", "I_stack")
    }
    finest_exit_common = resample_centered_grid(
        finest["U_A_exit"],
        float(case_dx[2]),
        _shape(
            comparisons["lateral_u_a_exit_common_grid"],
            "R8 lateral common grid",
        ),
        float(comparisons["lateral_u_a_exit_common_dx_m"]),
    )
    lateral_acceptance = {
        "U_A_exit": relative_l2(
            axial_reference["U_A_exit"], finest_exit_common
        ),
        "P_B": relative_l2(axial_reference["P_B"], finest["P_B"]),
        "I_stack": relative_l2(
            axial_reference["I_stack"], finest["I_stack"]
        ),
    }
    open_acceptance = relative_l2(open_288_stack, finest["I_stack"])
    axial_pass = {
        name: bool(value <= convergence_threshold)
        for name, value in axial_acceptance.items()
    }
    lateral_pass = {
        name: bool(value <= convergence_threshold)
        for name, value in lateral_acceptance.items()
    }
    open_pass = bool(open_acceptance <= convergence_threshold)
    convergence_pass = bool(
        all(axial_pass.values()) and all(lateral_pass.values()) and open_pass
    )

    waist_signals = {
        label: {
            name: relative_l2(cases[case_id][name], finest[name])
            for name in ("U_A_exit", "P_B", "I_stack")
        }
        for label, case_id in (
            ("waist_minus", "waist_minus"),
            ("waist_plus", "waist_plus"),
        )
    }
    waist_per_frame = {
        label: _per_frame_errors(cases[case_id]["I_stack"], finest["I_stack"])
        for label, case_id in (
            ("waist_minus", "waist_minus"),
            ("waist_plus", "waist_plus"),
        )
    }
    numerical_floor = {
        "U_A_exit": max(
            axial_acceptance["U_A_exit"], lateral_acceptance["U_A_exit"]
        ),
        "P_B": max(axial_acceptance["P_B"], lateral_acceptance["P_B"]),
        "I_stack": max(
            axial_acceptance["I_stack"],
            lateral_acceptance["I_stack"],
            open_acceptance,
        ),
    }
    signal_to_floor = {
        label: {
            name: float(
                signals[name]
                / max(numerical_floor[name], np.finfo(np.float64).eps)
            )
            for name in ("U_A_exit", "P_B", "I_stack")
        }
        for label, signals in waist_signals.items()
    }
    detector_signal_to_floor_min = min(
        signal_to_floor["waist_minus"]["I_stack"],
        signal_to_floor["waist_plus"]["I_stack"],
    )
    visibility_pass = bool(
        detector_signal_to_floor_min >= visibility_threshold
    )

    constant = positive_midpoint_pixel_average(
        np.ones((16 * q_detector, 16 * q_detector), dtype=np.float64),
        q_detector,
    )
    constant_error = float(np.max(np.abs(constant - 1.0)))
    detector_pixel_m = float(_section(optics, "detector")["pixel_size_m"])
    geometry_error = max(
        _r4_node_geometry_error(
            primary_open_shape[0] // q_detector,
            detector_pixel_m,
            q_detector,
        ),
        _r4_node_geometry_error(
            primary_open_shape[1] // q_detector,
            detector_pixel_m,
            q_detector,
        ),
    )
    sum_error_max = float(max(detector_sum_errors))
    fine_mapping_identity_error = float(max(fine_mapping_identity_errors))
    determinism_pass = bool(
        np.isfinite(determinism_error)
        and determinism_error <= determinism_threshold
    )
    algebra_errors = [
        *fraction_errors,
        *index_errors,
        *count_errors,
        b_mapping_error,
        b_exterior_error,
        b_modulus_error,
        homogeneous_consistency_error,
        fine_mapping_identity_error,
        constant_error,
        geometry_error,
        sum_error_max,
    ]
    hard_checks_pass = bool(
        fine_validation["pass"]
        and max(algebra_errors) <= algebra_threshold
        and max(thickness_errors) <= geometry_tolerance
        and determinism_pass
        and all_finite
        and all_nonnegative
    )
    if not hard_checks_pass:
        status = "Failed"
    elif convergence_pass and visibility_pass:
        status = "Passed"
    else:
        status = "Inconclusive"

    case_ids = [str(case["id"]) for case in raw_cases]
    metrics = {
        "version": "R8",
        "methods": dict(_section(diagnostics, "methods")),
        "r7_provenance": dict(_section(diagnostics, "r7_provenance")),
        "sampling": {
            "interface_factor": interface_factor,
            "case_ids": case_ids,
            "sample_a_shapes": np.asarray(case_shapes, dtype=np.int64),
            "sample_a_dx_m": np.asarray(case_dx, dtype=np.float64),
            "sample_a_dz_m": np.asarray(case_dz, dtype=np.float64),
            "d_waist_m": np.asarray(case_waists, dtype=np.float64),
            "slice_counts": np.asarray(slice_counts, dtype=np.int64),
            "detector_quadrature_factor": q_detector,
            "detector_node_dx_m": node_dx,
            "base_node_shape": np.asarray(base_shape, dtype=np.int64),
            "primary_open_node_shape": np.asarray(
                primary_open_shape, dtype=np.int64
            ),
            "open_control_node_shapes": np.asarray(
                open_shapes, dtype=np.int64
            ),
            "native_roi_shape": np.asarray(native_roi, dtype=np.int64),
            "scan_count": int(len(positions)),
            "full_volumes_retained": False,
            "full_node_stacks_retained": False,
        },
        "interface_controls": {
            "fraction_bound_error_by_case": np.asarray(fraction_errors),
            "index_bound_error_by_case": np.asarray(index_errors),
            "subnode_count_identity_error_by_case": np.asarray(count_errors),
            "air_volume_relative_error_by_case": np.asarray(volume_errors),
            "slice_width_sum_absolute_error_m_by_case": np.asarray(
                thickness_errors
            ),
            "slice_width_sum_tolerance_m": geometry_tolerance,
            "finest_homogeneous_streamed_relative_l2": (
                homogeneous_consistency_error
            ),
        },
        "mapping_controls": {
            "fine_a_to_node_identity_relative_l2": (
                fine_mapping_identity_error
            ),
            "finite_b_support_shape": np.asarray(
                support_shape, dtype=np.int64
            ),
            "finite_b_mapping_max_complex_error": b_mapping_error,
            "finite_b_exterior_max_complex_error": b_exterior_error,
            "finite_b_unit_modulus_max_abs_error": b_modulus_error,
        },
        "detector_controls": {
            "node_geometry_normalized_error": geometry_error,
            "constant_max_abs_error": constant_error,
            "max_sum_relative_error": sum_error_max,
            "weights": {
                "minimum": 1.0 / q_detector**2,
                "maximum": 1.0 / q_detector**2,
                "all_nonnegative": True,
                "sum_one_max_abs_error": 0.0,
            },
            "all_outputs_nonnegative": bool(all_nonnegative),
        },
        "convergence": {
            "axial": {
                "pair": ["axial_coarse", "common_reference"],
                "denominator": "common_reference",
                "acceptance": axial_acceptance,
                "pass": axial_pass,
            },
            "lateral": {
                "pair": ["common_reference", "finest_baseline"],
                "denominator": "finest_baseline",
                "U_A_exit_common_grid": np.asarray(
                    comparisons["lateral_u_a_exit_common_grid"],
                    dtype=np.int64,
                ),
                "acceptance": lateral_acceptance,
                "pass": lateral_pass,
            },
            "open": {
                "pair": ["open_288", "open_384"],
                "denominator": "open_384_finest_baseline_i_stack",
                "I_stack": open_acceptance,
                "pass": open_pass,
            },
            "all_pass": convergence_pass,
        },
        "visibility": {
            "signals": waist_signals,
            "per_frame_I_stack_relative_l2": waist_per_frame,
            "numerical_floor": numerical_floor,
            "signal_to_floor": signal_to_floor,
            "detector_signal_to_floor_min": detector_signal_to_floor_min,
            "detector_visibility_pass": visibility_pass,
        },
        "determinism": {
            "case_id": "finest_baseline",
            "scan_index": selected_scan_index,
            "open_reference": "open_384",
            "I_stack_relative_l2": determinism_error,
            "pass": determinism_pass,
        },
        "model_uncertainty_context": dict(
            _section(diagnostics, "r6_context")
        ),
        "thresholds": {
            "convergence_relative_l2_max": convergence_threshold,
            "algebra_relative_l2_max": algebra_threshold,
            "determinism_relative_l2_max": determinism_threshold,
            "detector_visibility_signal_to_floor_min": visibility_threshold,
        },
        "outcome_flags": {
            "interpretation_code": _r8_outcome_code(
                status=status,
                convergence_pass=convergence_pass,
                visibility_pass=visibility_pass,
            )
        },
        "legacy_experiment_status_preserved": True,
        "all_finite": bool(all_finite),
        "all_intensity_nonnegative": bool(all_nonnegative),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }
    result = {
        "selected_scan0": {
            key: selected_detectors[key]
            for key in ("waist_minus", "finest_baseline", "waist_plus")
        },
        "open_selected_scan0": {
            "open_288": open_288_selected,
            "open_384": selected_detectors["finest_baseline"],
        },
        "metrics": {
            "sampling": metrics["sampling"],
            "convergence": metrics["convergence"],
            "visibility": metrics["visibility"],
            "thresholds": metrics["thresholds"],
        },
    }
    del (
        base_period,
        cases,
        coarse_cells,
        fine_base,
        finest_exit_common,
        homogeneous_probe_base,
        open_288_stack,
        transfer_ab_base,
    )
    gc.collect()
    return result, metrics


def _r9_outcome_code(
    *, status: str, passband_pass: bool, raw_pass: bool
) -> str:
    """Return the frozen R9 A-exit attribution interpretation code."""

    if status == "Failed":
        return "a_exit_attribution_blocked"
    if not passband_pass:
        return "external_propagating_band_discrepancy_remains"
    if raw_pass:
        return "raw_and_external_passband_a_exit_converged"
    return "raw_discrepancy_attributed_outside_external_propagating_gate"


def _r9_project_with_controls(
    field: NDArray[np.complex128],
    dx_m: float,
    cutoff_cycles_per_m: float,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Project one R9 native field and return algebra controls."""

    values = np.asarray(field, dtype=np.complex128)
    mask = make_physical_passband_mask(
        values.shape, dx_m, cutoff_cycles_per_m
    )
    projected = project_field_to_passband(
        values, dx_m, cutoff_cycles_per_m
    )
    repeated = project_field_to_passband(
        values, dx_m, cutoff_cycles_per_m
    )
    idempotent = project_field_to_passband(
        projected, dx_m, cutoff_cycles_per_m
    )
    constant = project_field_to_passband(
        np.ones(values.shape, dtype=np.complex128),
        dx_m,
        cutoff_cycles_per_m,
    )
    total_energy = float(np.sum(np.abs(values) ** 2))
    retained_energy = float(np.sum(np.abs(projected) ** 2))
    controls = {
        "shape": np.asarray(values.shape, dtype=np.int64),
        "dx_m": float(dx_m),
        "frequency_spacing_cycles_per_m": np.asarray(
            [
                1.0 / (values.shape[0] * dx_m),
                1.0 / (values.shape[1] * dx_m),
            ],
            dtype=np.float64,
        ),
        "nyquist_cycles_per_m": 0.5 / float(dx_m),
        "mask_true_count": int(np.count_nonzero(mask)),
        "mask_total_count": int(mask.size),
        "mask_fraction": float(np.mean(mask)),
        "retained_reference_energy_fraction": float(
            retained_energy / max(total_energy, np.finfo(float).eps)
        ),
        "repeat_relative_l2": relative_l2(repeated, projected),
        "idempotence_relative_l2": relative_l2(idempotent, projected),
        "constant_max_abs_error": float(np.max(np.abs(constant - 1.0))),
        "all_finite": bool(
            np.all(np.isfinite(projected))
            and np.all(np.isfinite(repeated))
            and np.all(np.isfinite(idempotent))
            and np.all(np.isfinite(constant))
        ),
    }
    return projected, controls


def _r9_comparison_metrics(
    raw_test: NDArray[np.complex128],
    raw_reference: NDArray[np.complex128],
    passband_test: NDArray[np.complex128],
    passband_reference: NDArray[np.complex128],
    *,
    dx_m: float,
    cutoff_cycles_per_m: float,
) -> dict[str, Any]:
    """Measure raw, passband, and Parseval-attributed R9 field errors."""

    raw_test_values = np.asarray(raw_test, dtype=np.complex128)
    raw_reference_values = np.asarray(raw_reference, dtype=np.complex128)
    pass_test_values = np.asarray(passband_test, dtype=np.complex128)
    pass_reference_values = np.asarray(
        passband_reference, dtype=np.complex128
    )
    raw_difference = raw_test_values - raw_reference_values
    inside_difference = project_field_to_passband(
        raw_difference, dx_m, cutoff_cycles_per_m
    )
    outside_difference = raw_difference - inside_difference
    total_energy = float(np.sum(np.abs(raw_difference) ** 2))
    inside_energy = float(np.sum(np.abs(inside_difference) ** 2))
    outside_energy = float(np.sum(np.abs(outside_difference) ** 2))
    denominator = max(total_energy, np.finfo(float).eps)
    reference_energy = float(np.sum(np.abs(raw_reference_values) ** 2))
    pass_reference_energy = float(
        np.sum(np.abs(pass_reference_values) ** 2)
    )
    return {
        "raw_relative_l2": relative_l2(
            raw_test_values, raw_reference_values
        ),
        "external_passband_relative_l2": relative_l2(
            pass_test_values, pass_reference_values
        ),
        "reference_passband_retained_energy_fraction": float(
            pass_reference_energy
            / max(reference_energy, np.finfo(float).eps)
        ),
        "difference_energy": {
            "total": total_energy,
            "inside_external_passband": inside_energy,
            "outside_external_passband": outside_energy,
            "inside_fraction": inside_energy / denominator,
            "outside_fraction": outside_energy / denominator,
            "parseval_closure_relative_error": float(
                abs(inside_energy + outside_energy - total_energy)
                / denominator
            ),
            "inside_outside_orthogonality_relative_error": float(
                abs(
                    _r9_explicit_complex_inner_product(
                        inside_difference, outside_difference
                    )
                )
                / denominator
            ),
        },
    }


def _r9_explicit_complex_inner_product(
    left: NDArray[np.complex128], right: NDArray[np.complex128]
) -> complex:
    """Return ``sum(conj(left) * right)`` without BLAS ``vdot`` dispatch."""

    left_values = np.asarray(left, dtype=np.complex128)
    right_values = np.asarray(right, dtype=np.complex128)
    if left_values.shape != right_values.shape:
        msg = "R9 complex inner-product arrays must have the same shape."
        raise ValueError(msg)
    return complex(
        np.sum(
            np.conjugate(left_values) * right_values,
            dtype=np.complex128,
        )
    )


def _r9_normalized_error_map(
    test: NDArray[np.complex128], reference: NDArray[np.complex128]
) -> NDArray[np.float64]:
    denominator = max(float(np.max(np.abs(reference))), np.finfo(float).eps)
    return np.asarray(np.abs(test - reference) / denominator, dtype=np.float64)


def _r9_log_difference_spectrum(
    test: NDArray[np.complex128], reference: NDArray[np.complex128]
) -> NDArray[np.float64]:
    power = np.abs(np.fft.fftshift(np.fft.fft2(test - reference))) ** 2
    normalized = power / max(float(np.max(power)), np.finfo(float).eps)
    return np.asarray(
        np.log10(np.maximum(normalized, np.finfo(float).eps)),
        dtype=np.float64,
    )


def _run_r9_diagnostics(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the R9 A-exit raw/passband and restriction attribution study."""

    diagnostics = _section(config, "diagnostics_r9")
    registration = _section(diagnostics, "sample_a_cases")
    passband_registration = _section(diagnostics, "physical_passband")
    restrictions = _section(diagnostics, "lateral_restrictions")
    optics = _section(config, "optics")
    illumination = _section(config, "illumination")
    sample_a = _section(config, "sample_a")
    acceptance = _section(config, "acceptance")
    interface_factor = int(_section(diagnostics, "interface")["factor"])
    cutoff = float(passband_registration["cutoff_cycles_per_m"])
    wavelength = float(optics["wavelength_m"])
    n_ref = float(optics["internal_reference_index"])
    n_glass = float(sample_a["n_glass"])
    n_air = float(sample_a["n_air"])
    center_xy = tuple(float(value) for value in sample_a["center_xy_m"])
    thickness = float(sample_a["thickness_m"])
    waist_depth = float(sample_a["z_waist_m"])
    convergence_threshold = float(acceptance["convergence_relative_l2_max"])
    algebra_threshold = float(acceptance["algebra_relative_l2_max"])
    determinism_threshold = float(acceptance["determinism_relative_l2_max"])
    geometry_rule = _section(
        acceptance, "geometry_thickness_absolute_tolerance_m"
    )
    geometry_tolerance = max(
        float(geometry_rule["fixed_floor_m"]),
        float(geometry_rule["floating_point_factor"])
        * np.finfo(np.float64).eps
        * thickness,
    )

    cases: dict[str, dict[str, Any]] = {}
    case_shapes: list[tuple[int, int]] = []
    case_dx: list[float] = []
    case_dz: list[float] = []
    slice_counts: list[int] = []
    fraction_errors: list[float] = []
    index_errors: list[float] = []
    count_errors: list[float] = []
    volume_errors: list[float] = []
    thickness_errors: list[float] = []
    all_finite = True
    homogeneous_consistency_error = float("nan")

    registered_cases = registration["cases"]
    for case_index, case_cfg in enumerate(registered_cases, start=1):
        case_id = str(case_cfg["id"])
        shape = _shape(case_cfg["shape"], f"R9 {case_id} shape")
        dx_m = float(case_cfg["dx_m"])
        dz_m = float(case_cfg["dz_m"])
        d_waist_m = float(case_cfg["d_waist_m"])
        _emit_runtime_progress(
            progress_callback,
            "r9_case_started",
            case_id=case_id,
            case_index=case_index,
            case_count=len(registered_cases),
            shape=list(shape),
            dx_m=dx_m,
            dz_m=dz_m,
        )
        z_m, widths = midpoint_z_grid(thickness, dz_m)
        diameters = diameter_profile(
            z_m,
            thickness,
            float(sample_a["d_top_m"]),
            d_waist_m,
            float(sample_a["d_bottom_m"]),
            waist_depth,
        )
        selected_slice_index = int(np.argmin(np.abs(z_m - waist_depth)))
        incident = make_plane_wave(
            shape,
            dx_m,
            wavelength,
            theta_x=float(illumination["theta_x_rad"]),
            theta_y=float(illumination["theta_y_rad"]),
            amplitude=float(illumination["amplitude"]),
        )
        a_exit, _, interface_controls = _r7_streamed_tgv_exit(
            incident=incident,
            shape=shape,
            dx_m=dx_m,
            widths=widths,
            diameters=diameters,
            interface_factor=interface_factor,
            center_xy_m=center_xy,
            n_glass=n_glass,
            n_air=n_air,
            wavelength=wavelength,
            n_ref=n_ref,
            bandlimit=bool(optics["angular_spectrum_bandlimit"]),
            selected_slice_index=selected_slice_index,
        )
        if case_id == "axial_fine_reference":
            homogeneous_reference = _r5_homogeneous_a_exit(
                config, shape, dx_m
            )
            homogeneous_streamed = multislice_propagate_streamed_A(
                incident,
                (np.full(shape, n_ref, dtype=np.float64) for _ in widths),
                dx_m,
                widths,
                wavelength,
                n_ref=n_ref,
                bandlimit=bool(optics["angular_spectrum_bandlimit"]),
            )
            homogeneous_consistency_error = relative_l2(
                homogeneous_streamed, homogeneous_reference
            )
            del homogeneous_reference, homogeneous_streamed
        continuous_volume = float(
            np.sum(np.pi * (diameters / 2.0) ** 2 * widths)
        )
        discrete_volume = float(interface_controls["discrete_air_volume_m3"])
        volume_errors.append(
            abs(discrete_volume - continuous_volume)
            / max(continuous_volume, np.finfo(float).eps)
        )
        fraction_errors.append(float(interface_controls["fraction_bound_error"]))
        index_errors.append(float(interface_controls["index_bound_error"]))
        count_errors.append(float(interface_controls["count_identity_error"]))
        thickness_errors.append(abs(float(np.sum(widths)) - thickness))
        all_finite = bool(
            all_finite
            and interface_controls["all_finite"]
            and np.all(np.isfinite(a_exit))
        )
        cases[case_id] = {
            "U_A_exit": a_exit,
            "shape": shape,
            "dx_m": dx_m,
            "dz_m": dz_m,
        }
        case_shapes.append(shape)
        case_dx.append(dx_m)
        case_dz.append(dz_m)
        slice_counts.append(len(widths))
        del incident, z_m, widths, diameters
        gc.collect()
        _emit_runtime_progress(
            progress_callback,
            "r9_case_completed",
            case_id=case_id,
            case_index=case_index,
            case_count=len(registered_cases),
            slice_count=slice_counts[-1],
        )

    _emit_runtime_progress(progress_callback, "r9_postprocessing_started")
    projected: dict[str, NDArray[np.complex128]] = {}
    projection_controls: dict[str, dict[str, Any]] = {}
    for case_id, case in cases.items():
        projected_case, controls = _r9_project_with_controls(
            case["U_A_exit"], float(case["dx_m"]), cutoff
        )
        projected[case_id] = projected_case
        projection_controls[case_id] = controls
        all_finite = bool(all_finite and controls["all_finite"])

    axial_coarse = cases["axial_coarse"]["U_A_exit"]
    common = cases["common_reference"]["U_A_exit"]
    axial_fine = cases["axial_fine_reference"]["U_A_exit"]
    lateral_fine = cases["lateral_fine_reference"]["U_A_exit"]
    common_shape = _shape(restrictions["target_shape"], "R9 common shape")
    common_dx = float(restrictions["target_dx_m"])
    fine_dx = float(cases["lateral_fine_reference"]["dx_m"])
    ratio = int(restrictions["refinement_ratio"])

    lateral_bilinear = resample_centered_grid(
        lateral_fine, fine_dx, common_shape, common_dx
    )
    lateral_average = restrict_aligned_cell_average(lateral_fine, ratio)
    lateral_pass_bilinear = resample_centered_grid(
        projected["lateral_fine_reference"],
        fine_dx,
        common_shape,
        common_dx,
    )
    lateral_pass_average = restrict_aligned_cell_average(
        projected["lateral_fine_reference"], ratio
    )

    comparisons = {
        "r8_axial_reproduction": _r9_comparison_metrics(
            axial_coarse,
            common,
            projected["axial_coarse"],
            projected["common_reference"],
            dx_m=common_dx,
            cutoff_cycles_per_m=cutoff,
        ),
        "axial_refinement": _r9_comparison_metrics(
            common,
            axial_fine,
            projected["common_reference"],
            projected["axial_fine_reference"],
            dx_m=common_dx,
            cutoff_cycles_per_m=cutoff,
        ),
        "lateral_bilinear": _r9_comparison_metrics(
            common,
            lateral_bilinear,
            projected["common_reference"],
            lateral_pass_bilinear,
            dx_m=common_dx,
            cutoff_cycles_per_m=cutoff,
        ),
        "lateral_cell_average": _r9_comparison_metrics(
            common,
            lateral_average,
            projected["common_reference"],
            lateral_pass_average,
            dx_m=common_dx,
            cutoff_cycles_per_m=cutoff,
        ),
    }
    comparisons["r8_axial_reproduction"].update(
        pair=["axial_coarse", "common_reference"],
        denominator="common_reference",
        restriction="direct_same_grid",
    )
    comparisons["axial_refinement"].update(
        pair=["common_reference", "axial_fine_reference"],
        denominator="axial_fine_reference",
        restriction="direct_same_grid",
    )
    comparisons["lateral_bilinear"].update(
        pair=["common_reference", "lateral_fine_reference"],
        denominator="restricted_lateral_fine_reference",
        restriction="centered_bilinear_complex_field",
    )
    comparisons["lateral_cell_average"].update(
        pair=["common_reference", "lateral_fine_reference"],
        denominator="restricted_lateral_fine_reference",
        restriction="aligned_2x2_complex_cell_average",
    )
    provenance = _section(diagnostics, "r8_provenance")
    reproduction_errors = {
        "raw_axial_absolute_error": abs(
            float(comparisons["r8_axial_reproduction"]["raw_relative_l2"])
            - float(provenance["raw_axial_u_a_exit"])
        ),
        "raw_lateral_bilinear_absolute_error": abs(
            float(comparisons["lateral_bilinear"]["raw_relative_l2"])
            - float(provenance["raw_lateral_u_a_exit"])
        ),
    }
    raw_restriction_error = relative_l2(
        lateral_bilinear, lateral_average
    )
    pass_restriction_error = relative_l2(
        lateral_pass_bilinear, lateral_pass_average
    )
    restriction_error = max(raw_restriction_error, pass_restriction_error)

    repeated_lateral_projection = project_field_to_passband(
        lateral_fine, fine_dx, cutoff
    )
    repeated_bilinear = resample_centered_grid(
        repeated_lateral_projection, fine_dx, common_shape, common_dx
    )
    repeated_average = restrict_aligned_cell_average(
        repeated_lateral_projection, ratio
    )
    postprocessing_determinism = max(
        relative_l2(
            repeated_lateral_projection,
            projected["lateral_fine_reference"],
        ),
        relative_l2(repeated_bilinear, lateral_pass_bilinear),
        relative_l2(repeated_average, lateral_pass_average),
    )
    determinism_pass = bool(
        np.isfinite(postprocessing_determinism)
        and postprocessing_determinism <= determinism_threshold
    )

    comparison_pass = {
        name: {
            "raw": bool(values["raw_relative_l2"] <= convergence_threshold),
            "external_passband": bool(
                values["external_passband_relative_l2"]
                <= convergence_threshold
            ),
        }
        for name, values in comparisons.items()
    }
    passband_pass = bool(
        comparison_pass["axial_refinement"]["external_passband"]
        and comparison_pass["lateral_bilinear"]["external_passband"]
        and comparison_pass["lateral_cell_average"]["external_passband"]
    )
    raw_pass = bool(
        comparison_pass["axial_refinement"]["raw"]
        and comparison_pass["lateral_bilinear"]["raw"]
        and comparison_pass["lateral_cell_average"]["raw"]
    )
    parseval_errors = [
        float(values["difference_energy"]["parseval_closure_relative_error"])
        for values in comparisons.values()
    ]
    orthogonality_errors = [
        float(
            values["difference_energy"][
                "inside_outside_orthogonality_relative_error"
            ]
        )
        for values in comparisons.values()
    ]
    projection_errors = [
        float(controls[name])
        for controls in projection_controls.values()
        for name in (
            "repeat_relative_l2",
            "idempotence_relative_l2",
            "constant_max_abs_error",
        )
    ]
    algebra_errors = [
        *fraction_errors,
        *index_errors,
        *count_errors,
        homogeneous_consistency_error,
        *reproduction_errors.values(),
        restriction_error,
        *parseval_errors,
        *orthogonality_errors,
        *projection_errors,
    ]
    hard_checks_pass = bool(
        max(algebra_errors) <= algebra_threshold
        and max(thickness_errors) <= geometry_tolerance
        and determinism_pass
        and all_finite
    )
    if not hard_checks_pass:
        status = "Failed"
    elif passband_pass:
        status = "Passed"
    else:
        status = "Inconclusive"

    for name, values in comparisons.items():
        values["pass"] = comparison_pass[name]
    metrics = {
        "version": "R9",
        "methods": dict(_section(diagnostics, "methods")),
        "r8_provenance": dict(provenance),
        "sampling": {
            "interface_factor": interface_factor,
            "case_ids": [str(case["id"]) for case in registration["cases"]],
            "sample_a_shapes": np.asarray(case_shapes, dtype=np.int64),
            "sample_a_dx_m": np.asarray(case_dx, dtype=np.float64),
            "sample_a_dz_m": np.asarray(case_dz, dtype=np.float64),
            "slice_counts": np.asarray(slice_counts, dtype=np.int64),
            "full_volumes_retained": False,
            "detector_path_recomputed": False,
        },
        "passband": {
            "cutoff_cycles_per_m": cutoff,
            "medium_index": float(optics["external_medium_index"]),
            "vacuum_wavelength_m": wavelength,
            "native_projection_controls": projection_controls,
        },
        "interface_controls": {
            "fraction_bound_error_by_case": np.asarray(fraction_errors),
            "index_bound_error_by_case": np.asarray(index_errors),
            "subnode_count_identity_error_by_case": np.asarray(count_errors),
            "air_volume_relative_error_by_case": np.asarray(volume_errors),
            "slice_width_sum_absolute_error_m_by_case": np.asarray(
                thickness_errors
            ),
            "slice_width_sum_tolerance_m": geometry_tolerance,
            "axial_fine_homogeneous_streamed_relative_l2": (
                homogeneous_consistency_error
            ),
        },
        "r8_reproduction": {
            **reproduction_errors,
            "pass": bool(max(reproduction_errors.values()) <= algebra_threshold),
        },
        "comparisons": comparisons,
        "restriction_controls": {
            "methods": list(restrictions["methods"]),
            "raw_bilinear_to_cell_average_relative_l2": (
                raw_restriction_error
            ),
            "passband_bilinear_to_cell_average_relative_l2": (
                pass_restriction_error
            ),
            "maximum_relative_l2": restriction_error,
            "pass": bool(restriction_error <= algebra_threshold),
        },
        "spectral_controls": {
            "maximum_parseval_closure_relative_error": max(parseval_errors),
            "maximum_inside_outside_orthogonality_relative_error": max(
                orthogonality_errors
            ),
            "difference_energy_fraction_is_report_only": True,
        },
        "determinism": {
            "scope": "repeated_passband_and_restriction_postprocessing",
            "relative_l2": postprocessing_determinism,
            "pass": determinism_pass,
        },
        "thresholds": {
            "convergence_relative_l2_max": convergence_threshold,
            "algebra_relative_l2_max": algebra_threshold,
            "determinism_relative_l2_max": determinism_threshold,
        },
        "outcome_flags": {
            "passband_convergence_pass": passband_pass,
            "raw_convergence_pass": raw_pass,
            "interpretation_code": _r9_outcome_code(
                status=status,
                passband_pass=passband_pass,
                raw_pass=raw_pass,
            ),
        },
        "legacy_experiment_status_preserved": True,
        "all_finite": bool(all_finite),
        "hard_checks_pass": hard_checks_pass,
        "status": status,
    }

    mask_common = make_physical_passband_mask(common_shape, common_dx, cutoff)
    result = {
        "selected_maps": {
            "lateral_raw_bilinear": _r9_normalized_error_map(
                common, lateral_bilinear
            ),
            "lateral_raw_cell_average": _r9_normalized_error_map(
                common, lateral_average
            ),
            "lateral_passband_bilinear": _r9_normalized_error_map(
                projected["common_reference"], lateral_pass_bilinear
            ),
            "lateral_passband_cell_average": _r9_normalized_error_map(
                projected["common_reference"], lateral_pass_average
            ),
            "restriction_disagreement": _r9_normalized_error_map(
                lateral_bilinear, lateral_average
            ),
        },
        "difference_spectra": {
            "axial_refinement": _r9_log_difference_spectrum(
                common, axial_fine
            ),
            "lateral_cell_average": _r9_log_difference_spectrum(
                common, lateral_average
            ),
            "external_passband_mask": np.fft.fftshift(mask_common).astype(
                np.float64
            ),
        },
        "metrics": {
            "comparisons": metrics["comparisons"],
            "restriction_controls": metrics["restriction_controls"],
            "thresholds": metrics["thresholds"],
            "outcome_flags": metrics["outcome_flags"],
        },
    }
    del (
        axial_coarse,
        axial_fine,
        cases,
        common,
        lateral_average,
        lateral_bilinear,
        lateral_fine,
        lateral_pass_average,
        lateral_pass_bilinear,
        projected,
        repeated_average,
        repeated_bilinear,
        repeated_lateral_projection,
    )
    gc.collect()
    _emit_runtime_progress(progress_callback, "r9_postprocessing_completed")
    return result, metrics


def _make_canonical_b(
    sample_b: Mapping[str, Any],
) -> tuple[NDArray[np.complex128], float]:
    canonical = _section(sample_b, "canonical_grid")
    shape = _shape(canonical["shape"], "canonical shape")
    dx_m = float(canonical["dx_m"])
    feature_pixels = _require_integer_ratio(
        float(sample_b["physical_feature_size_m"]),
        dx_m,
        "sample B feature size",
    )
    values = make_random_phase_object(
        shape,
        phase_range=float(sample_b["phase_range_rad"]),
        seed=int(sample_b["seed"]),
        feature_size_px=feature_pixels,
    )
    return values, dx_m


def _sample_b_for_grid(
    canonical: NDArray[np.complex128],
    canonical_dx_m: float,
    shape: tuple[int, int],
    dx_m: float,
) -> NDArray[np.complex128]:
    sampled = resample_centered_grid(
        canonical,
        canonical_dx_m,
        shape,
        dx_m,
    )
    return np.asarray(sampled, dtype=np.complex128)


def _make_common_scan(scan: Mapping[str, Any]) -> NDArray[np.float64]:
    regular = make_grid_scan(
        int(scan["num_x"]),
        int(scan["num_y"]),
        float(scan["step_m"]),
        center=bool(scan["center"]),
    )
    return add_integer_pixel_jitter(
        regular,
        float(scan["jitter_quantum_m"]),
        int(scan["max_jitter_px"]),
        seed=int(scan["jitter_seed"]),
    )


def _validate_scan_on_grid(
    positions: NDArray[np.float64], dx_m: float
) -> None:
    pixel_values = np.asarray(positions, dtype=np.float64) / dx_m
    residual = np.abs(pixel_values - np.rint(pixel_values))
    if np.max(residual, initial=0.0) > 1e-9:
        msg = f"scan positions are not integer pixels for dx={dx_m:.6e} m."
        raise ValueError(msg)


def _all_grid_specs(config: Mapping[str, Any]) -> list[tuple[tuple[int, int], float]]:
    optics = _section(config, "optics")
    convergence = _section(config, "convergence")
    specs = [
        (
            _shape(optics["baseline_shape"], "baseline shape"),
            _isotropic_dx(optics["baseline_dx_m"], "baseline dx"),
        )
    ]
    for case in _section(convergence, "lateral_fixed_fov")["cases"]:
        specs.append(
            (
                _shape(case["shape"], "lateral shape"),
                _isotropic_dx(case["dx_m"], "lateral dx"),
            )
        )
    fov = _section(convergence, "fov")
    fov_dx = _isotropic_dx(fov["fixed_dx_m"], "fov dx")
    specs.extend((_shape(shape, "fov shape"), fov_dx) for shape in fov["shapes"])
    unique: list[tuple[tuple[int, int], float]] = []
    for item in specs:
        if item not in unique:
            unique.append(item)
    return unique


def _errors_to_finest_same_grid(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    reference = min(cases, key=lambda case: case["target_dz_m"])
    errors = {"U_A_exit": [], "P_B": [], "I_stack": []}
    for case in cases:
        values = _three_output_errors(case, reference)
        for name in errors:
            errors[name].append(values[name])
    return errors


def _map_case_to_grid(
    case: Mapping[str, Any], target_shape: tuple[int, int], target_dx_m: float
) -> dict[str, Any]:
    return {
        "U_A_exit": resample_centered_grid(
            case["U_A_exit"], case["dx_m"], target_shape, target_dx_m
        ),
        "P_B": resample_centered_grid(
            case["P_B"], case["dx_m"], target_shape, target_dx_m
        ),
        "I_stack": resample_centered_grid(
            case["I_stack"], case["dx_m"], target_shape, target_dx_m
        ),
    }


def _crop_case(
    case: Mapping[str, Any], target_shape: tuple[int, int]
) -> dict[str, Any]:
    return {
        "U_A_exit": center_crop(case["U_A_exit"], target_shape),
        "P_B": center_crop(case["P_B"], target_shape),
        "I_stack": center_crop(case["I_stack"], target_shape),
    }


def _three_output_errors(
    test: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    return {
        name: relative_l2(test[name], reference[name])
        for name in ("U_A_exit", "P_B", "I_stack")
    }


def _per_frame_errors(
    test: NDArray[np.floating], reference: NDArray[np.floating]
) -> NDArray[np.float64]:
    test_values = np.asarray(test, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if test_values.shape != reference_values.shape or test_values.ndim != 3:
        msg = "per-frame inputs must be same-shaped (scan, y, x) arrays."
        raise ValueError(msg)
    numerator = np.sqrt(
        np.sum((test_values - reference_values) ** 2, axis=(1, 2))
    )
    denominator = np.sqrt(np.sum(reference_values**2, axis=(1, 2)))
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps)


def _find_case(
    cases: Sequence[Mapping[str, Any]], key: str, value: Any
) -> Mapping[str, Any]:
    for case in cases:
        candidate = case[key]
        if isinstance(value, tuple):
            if tuple(candidate) == value:
                return case
        elif np.isclose(float(candidate), float(value), rtol=0.0, atol=1e-18):
            return case
    msg = f"No convergence case has {key}={value!r}."
    raise RuntimeError(msg)


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping."
        raise ValueError(msg)
    return value


def _shape(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        msg = f"{name} must contain [ny, nx]."
        raise ValueError(msg)
    if len(value) != 2:
        msg = f"{name} must contain [ny, nx]."
        raise ValueError(msg)
    shape = tuple(int(item) for item in value)
    if any(float(item) != integer for item, integer in zip(value, shape, strict=True)):
        msg = f"{name} entries must be integers."
        raise ValueError(msg)
    if min(shape) <= 0:
        msg = f"{name} entries must be positive."
        raise ValueError(msg)
    return shape


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not np.isfinite(number) or (number < 0.0 if allow_zero else number <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        msg = f"{name} must be finite and {qualifier}."
        raise ValueError(msg)
    return number


def _isotropic_dx(value: Any, name: str) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            msg = f"{name} tuple must be (dy, dx)."
            raise ValueError(msg)
        dy_m = _positive(value[0], f"{name}[0]")
        dx_m = _positive(value[1], f"{name}[1]")
        if not np.isclose(dy_m, dx_m, rtol=0.0, atol=0.0):
            msg = f"{name} must be isotropic for registered exp040 mappings."
            raise ValueError(msg)
        return dx_m
    return _positive(value, name)


def _require_integer_ratio(numerator: float, denominator: float, name: str) -> int:
    ratio = float(numerator) / float(denominator)
    rounded = int(np.rint(ratio))
    if rounded <= 0 or not np.isclose(ratio, rounded, rtol=0.0, atol=1e-9):
        msg = f"{name} must be an integer multiple of the grid sampling."
        raise ValueError(msg)
    return rounded
