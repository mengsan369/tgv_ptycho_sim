"""Run the formal exp040 R10 Stage-A 512-square to 1024-square comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.exp040 import (  # noqa: E402
    _r7_streamed_tgv_exit,
    _r9_comparison_metrics,
    _r9_log_difference_spectrum,
    _r9_normalized_error_map,
    _r9_project_with_controls,
    make_physical_passband_mask,
    project_field_to_passband,
    relative_l2,
    restrict_aligned_cell_average,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import (  # noqa: E402
    save_json,
    save_ptycho_hdf5,
)
from tgv_ptycho.objects.tgv_geometry import (  # noqa: E402
    diameter_profile,
    midpoint_z_grid,
)
from tgv_ptycho.optics.fields import make_plane_wave  # noqa: E402
from tgv_ptycho.viz.plot_exp040_r10 import (  # noqa: E402
    EXP040_R10_STAGE_A_FIGURE_FILENAMES,
    save_exp040_r10_stage_a_figures,
)

REGISTERED_CONFIG_SHA256 = (
    "3585AB185DD2E71A2D3872B310343A5F04F8F3B64B5D57BF9B640105A344D654"
)

ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require_exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        msg = f"R10 Stage-A {name} differs from its frozen registration."
        raise ValueError(msg)


def validate_stage_a_config(config: Mapping[str, Any]) -> None:
    """Validate all science-controlling values frozen in section 18.8."""

    _require_exact(
        set(config),
        {
            "run",
            "experiment",
            "provenance",
            "physics",
            "stage_a",
            "thresholds",
            "output",
        },
        "top-level sections",
    )
    _require_exact(config["experiment"]["id"], "exp040", "experiment id")
    _require_exact(config["experiment"]["stage"], "R10_stage_a", "stage")
    _require_exact(
        config["experiment"]["scientific_result"], True, "scientific role"
    )
    physics = config["physics"]
    expected_physics = {
        "wavelength_m": 5.32e-7,
        "internal_reference_index": 1.5,
        "external_medium_index": 1.0,
        "angular_spectrum_bandlimit": True,
        "illumination_amplitude": 1.0,
        "illumination_theta_x_rad": 0.0,
        "illumination_theta_y_rad": 0.0,
        "sample_thickness_m": 1.0e-4,
        "d_top_m": 3.0e-5,
        "d_waist_m": 2.0e-5,
        "d_bottom_m": 3.0e-5,
        "z_waist_m": 5.0e-5,
        "center_xy_m": [0.0, 0.0],
        "n_glass": 1.5,
        "n_air": 1.0,
    }
    _require_exact(dict(physics), expected_physics, "physics")

    stage = config["stage_a"]
    _require_exact(
        stage["fixed_case_order"],
        ["current_512", "fine_1024"],
        "case order",
    )
    _require_exact(
        stage["cases"],
        [
            {
                "id": "current_512",
                "shape": [512, 512],
                "dx_m": 1.25e-7,
                "dz_m": 2.5e-7,
                "expected_slice_count": 400,
            },
            {
                "id": "fine_1024",
                "shape": [1024, 1024],
                "dx_m": 6.25e-8,
                "dz_m": 2.5e-7,
                "expected_slice_count": 400,
            },
        ],
        "cases",
    )
    _require_exact(stage["common_fov_m"], [6.4e-5, 6.4e-5], "FOV")
    _require_exact(
        dict(stage["interface"]),
        {
            "factor": 8,
            "node_rule": "pixel_center_plus_a_half_over_q",
            "weights": "uniform_nonnegative",
            "effective_index": "linear_cell_average_of_indicator",
        },
        "interface",
    )
    _require_exact(
        dict(stage["propagation"]),
        {
            "operator": "centered_symmetric_split_step",
            "stream_slices": True,
            "retain_full_volumes": False,
            "detector_path_recomputed": False,
        },
        "propagation",
    )
    _require_exact(
        dict(stage["physical_passband"]),
        {
            "definition": "external_medium_index_over_vacuum_wavelength",
            "cutoff_cycles_per_m": 1879699.2481203007,
            "boundary_inclusive": True,
            "apply_on_each_native_grid_before_restriction": True,
        },
        "passband",
    )
    _require_exact(
        dict(stage["restriction"]),
        {
            "method": "aligned_2x2_complex_cell_average",
            "refinement_ratio": 2,
            "denominator": "restricted_fine_1024_reference",
            "phase_scale_spatial_alignment": False,
        },
        "restriction",
    )
    _require_exact(
        dict(config["thresholds"]),
        {
            "convergence_relative_l2_max": 5.0e-2,
            "algebra_relative_l2_max": 1.0e-12,
            "determinism_relative_l2_max": 1.0e-14,
            "geometry_thickness_absolute_tolerance_m": {
                "fixed_floor_m": 1.0e-15,
                "floating_point_factor": 16.0,
            },
            "require_all_finite": True,
        },
        "thresholds",
    )
    _require_exact(
        config["output"]["hdf5_filename"],
        "exp040_r10_stage_a.h5",
        "HDF5 filename",
    )
    _require_exact(
        config["output"]["figure_filenames"],
        list(EXP040_R10_STAGE_A_FIGURE_FILENAMES),
        "figure filenames",
    )
    if config["output"]["save_native_complex_fields"] is not False:
        raise ValueError("R10 Stage-A native fields must not be persisted.")
    if config["output"]["save_slice_volumes"] is not False:
        raise ValueError("R10 Stage-A slice volumes must not be persisted.")


def _validate_source_config(config_path: Path) -> None:
    actual = _sha256(config_path)
    if actual != REGISTERED_CONFIG_SHA256:
        msg = (
            "R10 Stage-A source config SHA256 differs from its frozen "
            f"registration: {actual}."
        )
        raise ValueError(msg)


def _emit(
    callback: ProgressCallback | None, event: str, **details: Any
) -> None:
    if callback is not None:
        callback(event, details)


def _case_field_and_controls(
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    case_index: int,
    progress_callback: ProgressCallback | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    physics = config["physics"]
    stage = config["stage_a"]
    shape = tuple(int(value) for value in case["shape"])
    dx_m = float(case["dx_m"])
    dz_m = float(case["dz_m"])
    case_id = str(case["id"])
    _emit(
        progress_callback,
        "case_started",
        case_id=case_id,
        case_index=case_index,
        case_count=2,
        shape=list(shape),
        dx_m=dx_m,
        dz_m=dz_m,
    )
    started = time.perf_counter()
    z_m, widths = midpoint_z_grid(float(physics["sample_thickness_m"]), dz_m)
    if len(widths) != int(case["expected_slice_count"]):
        msg = f"R10 Stage-A {case_id} slice count differs from registration."
        raise RuntimeError(msg)
    diameters = diameter_profile(
        z_m,
        float(physics["sample_thickness_m"]),
        float(physics["d_top_m"]),
        float(physics["d_waist_m"]),
        float(physics["d_bottom_m"]),
        float(physics["z_waist_m"]),
    )
    incident = make_plane_wave(
        shape,
        dx_m,
        float(physics["wavelength_m"]),
        theta_x=float(physics["illumination_theta_x_rad"]),
        theta_y=float(physics["illumination_theta_y_rad"]),
        amplitude=float(physics["illumination_amplitude"]),
    )
    selected_index = int(np.argmin(np.abs(z_m - float(physics["z_waist_m"]))))
    field, selected_fraction, interface = _r7_streamed_tgv_exit(
        incident=np.asarray(incident, dtype=np.complex128),
        shape=shape,
        dx_m=dx_m,
        widths=widths,
        diameters=diameters,
        interface_factor=int(stage["interface"]["factor"]),
        center_xy_m=tuple(float(value) for value in physics["center_xy_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        wavelength=float(physics["wavelength_m"]),
        n_ref=float(physics["internal_reference_index"]),
        bandlimit=bool(physics["angular_spectrum_bandlimit"]),
        selected_slice_index=selected_index,
    )
    elapsed_s = time.perf_counter() - started
    continuous_volume = float(
        np.sum(np.pi * (diameters / 2.0) ** 2 * widths)
    )
    discrete_volume = float(interface["discrete_air_volume_m3"])
    thickness_error = abs(
        float(np.sum(widths)) - float(physics["sample_thickness_m"])
    )
    all_finite = bool(
        interface["all_finite"]
        and np.all(np.isfinite(field))
        and np.all(np.isfinite(selected_fraction))
    )
    controls = {
        "id": case_id,
        "shape": list(shape),
        "dx_m": dx_m,
        "dz_m": dz_m,
        "fov_m": [shape[0] * dx_m, shape[1] * dx_m],
        "slice_count": int(len(widths)),
        "elapsed_s": float(elapsed_s),
        "fraction_bound_error": float(interface["fraction_bound_error"]),
        "index_bound_error": float(interface["index_bound_error"]),
        "subnode_count_identity_error": float(
            interface["count_identity_error"]
        ),
        "discrete_air_volume_m3": discrete_volume,
        "continuous_midpoint_air_volume_m3": continuous_volume,
        "air_volume_relative_error": float(
            abs(discrete_volume - continuous_volume)
            / max(continuous_volume, np.finfo(float).eps)
        ),
        "slice_width_sum_absolute_error_m": float(thickness_error),
        "selected_fraction_min": float(np.min(selected_fraction)),
        "selected_fraction_max": float(np.max(selected_fraction)),
        "all_finite": all_finite,
    }
    del incident, selected_fraction, z_m, widths, diameters
    gc.collect()
    _emit(
        progress_callback,
        "case_completed",
        case_id=case_id,
        case_index=case_index,
        case_count=2,
        slice_count=controls["slice_count"],
        elapsed_s=controls["elapsed_s"],
    )
    return np.asarray(field, dtype=np.complex128), controls


def _restriction_controls(
    fine_field: np.ndarray,
    restricted_field: np.ndarray,
    refinement_ratio: int,
) -> dict[str, Any]:
    expected_shape = (
        fine_field.shape[0] // refinement_ratio,
        fine_field.shape[1] // refinement_ratio,
    )
    fine_mean = complex(np.mean(fine_field, dtype=np.complex128))
    restricted_mean = complex(np.mean(restricted_field, dtype=np.complex128))
    mean_error = abs(restricted_mean - fine_mean) / max(
        abs(fine_mean), np.finfo(float).eps
    )

    constant = np.full((4, 4), 2.0 - 3.0j, dtype=np.complex128)
    constant_restricted = restrict_aligned_cell_average(constant, 2)
    constant_error = float(np.max(np.abs(constant_restricted - (2.0 - 3.0j))))
    alignment_error = 0.0
    for offset_y in range(2):
        for offset_x in range(2):
            impulse = np.zeros((4, 4), dtype=np.complex128)
            impulse[2 + offset_y, offset_x] = 1.0
            actual = restrict_aligned_cell_average(impulse, 2)
            expected = np.zeros((2, 2), dtype=np.complex128)
            expected[1, 0] = 0.25
            alignment_error = max(
                alignment_error, float(np.max(np.abs(actual - expected)))
            )
    return {
        "method": "aligned_2x2_complex_cell_average",
        "refinement_ratio": refinement_ratio,
        "fine_shape": list(fine_field.shape),
        "restricted_shape": list(restricted_field.shape),
        "expected_restricted_shape": list(expected_shape),
        "shape_matches": bool(restricted_field.shape == expected_shape),
        "constant_max_abs_error": constant_error,
        "area_weighted_complex_mean_relative_error": float(mean_error),
        "four_subpixel_alignment_max_abs_error": float(alignment_error),
        "registered_weights": [0.25, 0.25, 0.25, 0.25],
    }


def _postprocessing_repeat_error(
    *,
    current: np.ndarray,
    fine: np.ndarray,
    projected_current: np.ndarray,
    projected_fine: np.ndarray,
    restricted_fine: np.ndarray,
    restricted_projected_fine: np.ndarray,
    comparison: Mapping[str, Any],
    current_dx_m: float,
    fine_dx_m: float,
    cutoff: float,
    ratio: int,
) -> float:
    repeat_current = project_field_to_passband(current, current_dx_m, cutoff)
    repeat_fine = project_field_to_passband(fine, fine_dx_m, cutoff)
    repeat_restricted_fine = restrict_aligned_cell_average(fine, ratio)
    repeat_restricted_projected = restrict_aligned_cell_average(repeat_fine, ratio)
    repeat_comparison = _r9_comparison_metrics(
        current,
        repeat_restricted_fine,
        repeat_current,
        repeat_restricted_projected,
        dx_m=current_dx_m,
        cutoff_cycles_per_m=cutoff,
    )
    errors = [
        relative_l2(repeat_current, projected_current),
        relative_l2(repeat_fine, projected_fine),
        relative_l2(repeat_restricted_fine, restricted_fine),
        relative_l2(repeat_restricted_projected, restricted_projected_fine),
    ]
    for name in ("raw_relative_l2", "external_passband_relative_l2"):
        original = float(comparison[name])
        repeated = float(repeat_comparison[name])
        errors.append(
            abs(repeated - original) / max(abs(original), np.finfo(float).eps)
        )
    return float(max(errors))


def _stage_a_outcome(
    *,
    hard_controls_pass: bool,
    passband_relative_l2: float,
    raw_relative_l2: float,
    convergence_threshold: float,
) -> dict[str, Any]:
    passband_pass = bool(passband_relative_l2 <= convergence_threshold)
    raw_pass = bool(raw_relative_l2 <= convergence_threshold)
    if not hard_controls_pass:
        status = "Failed"
        interpretation = "stage_a_numerical_controls_failed"
        stage_b_allowed = False
    elif passband_pass:
        status = "Passed"
        interpretation = "scalar_lateral_reference_closed"
        stage_b_allowed = True
    else:
        status = "Inconclusive"
        interpretation = "scalar_lateral_reference_not_closed"
        stage_b_allowed = False
    return {
        "status": status,
        "interpretation_code": interpretation,
        "external_passband_convergence_pass": passband_pass,
        "raw_convergence_pass_report_only": raw_pass,
        "stage_b_allowed": stage_b_allowed,
    }


def _run_stage_a(
    config: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    stage = config["stage_a"]
    physics = config["physics"]
    thresholds = config["thresholds"]
    cases = list(stage["cases"])
    fields: dict[str, np.ndarray] = {}
    case_controls: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        field, controls = _case_field_and_controls(
            config,
            case,
            case_index=case_index,
            progress_callback=progress_callback,
        )
        fields[str(case["id"])] = field
        case_controls.append(controls)

    _emit(progress_callback, "postprocessing_started")
    current = fields["current_512"]
    fine = fields["fine_1024"]
    current_dx_m = float(cases[0]["dx_m"])
    fine_dx_m = float(cases[1]["dx_m"])
    cutoff = float(stage["physical_passband"]["cutoff_cycles_per_m"])
    ratio = int(stage["restriction"]["refinement_ratio"])
    projected_current, current_projection_controls = _r9_project_with_controls(
        current, current_dx_m, cutoff
    )
    projected_fine, fine_projection_controls = _r9_project_with_controls(
        fine, fine_dx_m, cutoff
    )
    restricted_fine = restrict_aligned_cell_average(fine, ratio)
    restricted_projected_fine = restrict_aligned_cell_average(
        projected_fine, ratio
    )
    comparison = _r9_comparison_metrics(
        current,
        restricted_fine,
        projected_current,
        restricted_projected_fine,
        dx_m=current_dx_m,
        cutoff_cycles_per_m=cutoff,
    )
    comparison.update(
        pair=["current_512", "fine_1024"],
        denominator="restricted_fine_1024_reference",
        restriction="aligned_2x2_complex_cell_average",
        alignment="none",
    )
    restriction_controls = _restriction_controls(fine, restricted_fine, ratio)
    determinism_error = _postprocessing_repeat_error(
        current=current,
        fine=fine,
        projected_current=projected_current,
        projected_fine=projected_fine,
        restricted_fine=restricted_fine,
        restricted_projected_fine=restricted_projected_fine,
        comparison=comparison,
        current_dx_m=current_dx_m,
        fine_dx_m=fine_dx_m,
        cutoff=cutoff,
        ratio=ratio,
    )

    algebra_threshold = float(thresholds["algebra_relative_l2_max"])
    determinism_threshold = float(thresholds["determinism_relative_l2_max"])
    convergence_threshold = float(thresholds["convergence_relative_l2_max"])
    geometry_rule = thresholds["geometry_thickness_absolute_tolerance_m"]
    geometry_tolerance = max(
        float(geometry_rule["fixed_floor_m"]),
        float(geometry_rule["floating_point_factor"])
        * np.finfo(np.float64).eps
        * float(physics["sample_thickness_m"]),
    )
    projection_controls = {
        "current_512": current_projection_controls,
        "fine_1024": fine_projection_controls,
    }
    projection_errors = [
        float(controls[name])
        for controls in projection_controls.values()
        for name in (
            "repeat_relative_l2",
            "idempotence_relative_l2",
            "constant_max_abs_error",
        )
    ]
    interface_errors = [
        float(case[name])
        for case in case_controls
        for name in (
            "fraction_bound_error",
            "index_bound_error",
            "subnode_count_identity_error",
        )
    ]
    spectral = comparison["difference_energy"]
    algebra_errors = [
        *projection_errors,
        *interface_errors,
        float(spectral["parseval_closure_relative_error"]),
        float(spectral["inside_outside_orthogonality_relative_error"]),
        float(restriction_controls["constant_max_abs_error"]),
        float(restriction_controls["area_weighted_complex_mean_relative_error"]),
        float(restriction_controls["four_subpixel_alignment_max_abs_error"]),
    ]
    maximum_algebra_error = float(max(algebra_errors))
    maximum_thickness_error = float(
        max(case["slice_width_sum_absolute_error_m"] for case in case_controls)
    )
    all_finite = bool(
        all(bool(case["all_finite"]) for case in case_controls)
        and all(
            bool(controls["all_finite"])
            for controls in projection_controls.values()
        )
        and np.all(np.isfinite(restricted_fine))
        and np.all(np.isfinite(restricted_projected_fine))
        and np.isfinite(float(comparison["raw_relative_l2"]))
        and np.isfinite(float(comparison["external_passband_relative_l2"]))
    )
    hard_controls_pass = bool(
        maximum_algebra_error <= algebra_threshold
        and maximum_thickness_error <= geometry_tolerance
        and determinism_error <= determinism_threshold
        and restriction_controls["shape_matches"]
        and all_finite
    )
    outcome = _stage_a_outcome(
        hard_controls_pass=hard_controls_pass,
        passband_relative_l2=float(
            comparison["external_passband_relative_l2"]
        ),
        raw_relative_l2=float(comparison["raw_relative_l2"]),
        convergence_threshold=convergence_threshold,
    )

    r9 = config["provenance"]["r9_lateral_cell_average"]
    raw_ratio = float(r9["raw_relative_l2"]) / max(
        float(comparison["raw_relative_l2"]), np.finfo(float).eps
    )
    passband_ratio = float(r9["external_passband_relative_l2"]) / max(
        float(comparison["external_passband_relative_l2"]), np.finfo(float).eps
    )
    empirical = {
        "report_only": True,
        "pairwise_raw_error_ratio_r9_over_r10": raw_ratio,
        "pairwise_passband_error_ratio_r9_over_r10": passband_ratio,
        "apparent_raw_order_log2_ratio": float(np.log2(raw_ratio)),
        "apparent_passband_order_log2_ratio": float(np.log2(passband_ratio)),
        "mathematical_convergence_proof": False,
    }
    metrics = {
        "version": "R10_stage_a",
        "scientific_result": True,
        "provenance": dict(config["provenance"]),
        "methods": {
            "field": stage["comparison"]["field"],
            "interface": "q8_staggered_midpoint_air_area_fraction",
            "propagator": stage["propagation"]["operator"],
            "passband_order": "project_each_native_grid_then_restrict_fine",
            "restriction": stage["restriction"]["method"],
            "denominator": stage["restriction"]["denominator"],
            "phase_scale_spatial_alignment": False,
        },
        "sampling": {
            "case_ids": [str(case["id"]) for case in cases],
            "current_shape": list(current.shape),
            "fine_shape": list(fine.shape),
            "current_dx_m": current_dx_m,
            "fine_dx_m": fine_dx_m,
            "dz_m": float(cases[0]["dz_m"]),
            "common_fov_m": list(stage["common_fov_m"]),
            "interface_factor": int(stage["interface"]["factor"]),
            "streamed_slices": True,
            "full_volumes_retained": False,
            "detector_path_recomputed": False,
        },
        "case_controls": case_controls,
        "passband": {
            "cutoff_cycles_per_m": cutoff,
            "external_medium_index": float(physics["external_medium_index"]),
            "vacuum_wavelength_m": float(physics["wavelength_m"]),
            "native_projection_controls": projection_controls,
        },
        "comparison": comparison,
        "restriction_controls": restriction_controls,
        "empirical_pairwise_diagnostic": empirical,
        "hard_controls": {
            "maximum_algebra_error": maximum_algebra_error,
            "maximum_slice_width_sum_absolute_error_m": maximum_thickness_error,
            "slice_width_sum_tolerance_m": geometry_tolerance,
            "postprocessing_determinism_relative_l2": determinism_error,
            "all_finite": all_finite,
            "pass": hard_controls_pass,
        },
        "thresholds": {
            "convergence_relative_l2_max": convergence_threshold,
            "algebra_relative_l2_max": algebra_threshold,
            "determinism_relative_l2_max": determinism_threshold,
        },
        "outcome": outcome,
        "status": outcome["status"],
    }
    mask = make_physical_passband_mask(current.shape, current_dx_m, cutoff)
    selected_maps = {
        "raw_normalized_residual": _r9_normalized_error_map(
            current, restricted_fine
        ),
        "passband_normalized_residual": _r9_normalized_error_map(
            projected_current, restricted_projected_fine
        ),
        "raw_difference_spectrum": _r9_log_difference_spectrum(
            current, restricted_fine
        ),
        "external_passband_mask": np.fft.fftshift(mask).astype(np.float64),
    }
    if not all(np.all(np.isfinite(value)) for value in selected_maps.values()):
        raise RuntimeError("R10 Stage-A selected maps are non-finite.")
    result = {"metrics": metrics, "selected_maps": selected_maps}
    del (
        current,
        fine,
        fields,
        projected_current,
        projected_fine,
        restricted_fine,
        restricted_projected_fine,
    )
    gc.collect()
    _emit(
        progress_callback,
        "postprocessing_completed",
        raw_relative_l2=float(comparison["raw_relative_l2"]),
        external_passband_relative_l2=float(
            comparison["external_passband_relative_l2"]
        ),
        status=outcome["status"],
        stage_b_allowed=outcome["stage_b_allowed"],
    )
    return result


def _progress_writer(path: Path) -> ProgressCallback:
    payload: dict[str, Any] = {
        "purpose": "formal_scientific_r10_stage_a_comparison",
        "created_at": created_at_utc(),
        "events": [],
    }

    def write(event: str, details: Mapping[str, Any]) -> None:
        record = {"event": event, "at": created_at_utc(), **dict(details)}
        payload["events"].append(record)
        payload["latest_event"] = record
        save_json(path, payload)
        print(f"progress: {event} {dict(details)}", flush=True)

    return write


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    expected = set(str(value) for value in config["output"]["required_files"])
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        msg = f"R10 Stage-A artifact set differs: {sorted(actual)}"
        raise RuntimeError(msg)
    json_names = (
        "metadata.json",
        "metrics.json",
        "run_state.json",
        "run_progress.json",
    )
    for name in json_names:
        with (run_dir / name).open("r", encoding="utf-8") as handle:
            json.load(handle)
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as h5:
        entry = h5["entry"]
        required = {
            "config_yaml",
            "data",
            "instrument",
            "sample",
            "metadata",
            "metrics",
        }
        if set(entry) != required:
            raise RuntimeError(f"R10 Stage-A HDF5 entry keys differ: {sorted(entry)}")
        if len(entry["data"]) != 0:
            raise RuntimeError(
                "R10 Stage-A HDF5 must not persist detector or field data."
            )
        if "truth" in entry or "reconstruction" in entry:
            raise RuntimeError(
                "R10 Stage-A HDF5 must not contain truth/reconstruction."
            )
    for filename in EXP040_R10_STAGE_A_FIGURE_FILENAMES:
        image = np.asarray(iio.imread(run_dir / "figures" / filename))
        if image.ndim not in (2, 3) or image.size == 0:
            raise RuntimeError(f"R10 Stage-A figure is invalid: {filename}")


def run(config_path: Path) -> Path:
    """Execute the one formal Stage-A scientific comparison."""

    config_path = config_path.resolve()
    _validate_source_config(config_path)
    config = load_config(config_path)
    validate_stage_a_config(config)
    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    save_config(run_dir / "config.yaml", dict(config))
    save_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "created_at": created_at_utc(),
            "source_config": str(config_path),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
            "scientific_result": True,
        },
    )
    progress = _progress_writer(run_dir / "run_progress.json")
    progress("run_started", {"source_config_sha256": REGISTERED_CONFIG_SHA256})
    started = time.perf_counter()
    try:
        result = _run_stage_a(config, progress_callback=progress)
        metrics = result["metrics"]
        elapsed_s = time.perf_counter() - started
        metrics["total_execution_elapsed_s"] = float(elapsed_s)
        outcome = metrics["outcome"]
        metadata = {
            "experiment_id": "exp040",
            "diagnostic_stage": "R10_stage_a",
            "scientific_result": True,
            "run_path": str(run_dir.resolve()),
            "source_config": str(config_path),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
            "created_at": created_at_utc(),
            "git_commit": get_git_commit(PROJECT_ROOT) or "unavailable",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "status": outcome["status"],
            "interpretation_code": outcome["interpretation_code"],
            "stage_b_allowed": bool(outcome["stage_b_allowed"]),
        }
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        progress("artifacts_writing_started", {})
        save_exp040_r10_stage_a_figures(result, run_dir / "figures")
        save_ptycho_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            instrument={
                "wavelength_m": float(config["physics"]["wavelength_m"]),
                "internal_reference_index": float(
                    config["physics"]["internal_reference_index"]
                ),
                "external_medium_index": float(
                    config["physics"]["external_medium_index"]
                ),
                "angular_spectrum_bandlimit": bool(
                    config["physics"]["angular_spectrum_bandlimit"]
                ),
                "sampling": metrics["sampling"],
            },
            sample={
                "type": "single_axisymmetric_air_filled_tgv_in_glass",
                "geometry": dict(config["physics"]),
                "interface": dict(config["stage_a"]["interface"]),
            },
            config_yaml=config_to_yaml(dict(config)),
            metadata=metadata,
            metrics=metrics,
        )
        save_json(
            run_dir / "run_state.json",
            {
                "status": "validation_pending",
                "completed_at": created_at_utc(),
                "scientific_status": outcome["status"],
                "interpretation_code": outcome["interpretation_code"],
                "stage_b_allowed": bool(outcome["stage_b_allowed"]),
                "scientific_result": True,
            },
        )
        _validate_artifacts(run_dir, config)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "completed_at": created_at_utc(),
                "scientific_status": outcome["status"],
                "interpretation_code": outcome["interpretation_code"],
                "stage_b_allowed": bool(outcome["stage_b_allowed"]),
                "scientific_result": True,
                "artifacts_validated": True,
            },
        )
        progress(
            "artifacts_validated",
            {
                "scientific_status": outcome["status"],
                "stage_b_allowed": bool(outcome["stage_b_allowed"]),
            },
        )
        _validate_artifacts(run_dir, config)
    except Exception as error:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed_during_execution",
                "failed_at": created_at_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "scientific_result": True,
                "formal_attempt_retained": True,
            },
        )
        raise

    comparison = metrics["comparison"]
    print(f"run_dir: {run_dir.resolve()}", flush=True)
    print(f"stage_a_status: {outcome['status']}", flush=True)
    print(f"interpretation: {outcome['interpretation_code']}", flush=True)
    print(
        f"raw_relative_l2: {comparison['raw_relative_l2']:.17g}", flush=True
    )
    print(
        "external_passband_relative_l2: "
        f"{comparison['external_passband_relative_l2']:.17g}",
        flush=True,
    )
    print(f"stage_b_allowed: {outcome['stage_b_allowed']}", flush=True)
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
