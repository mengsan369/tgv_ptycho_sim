"""Run exp031 finite nonperiodic-B and illumination-spot sensitivity study."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tgv_ptycho.forward.exp031 import (  # noqa: E402
    estimate_case_peak_memory_bytes,
    finite_metrics,
    gaussian_reference_radial,
    make_asm_transfer_complex64,
    make_open_shape,
    make_tgv_radial_probe_result,
    plane_wave_reference,
    propagate_probe_patch_batch_to_roi,
    reference_plus_residual_identity_error,
    residual_edge_energy_fraction,
    sample_radial_field,
)
from tgv_ptycho.forward.scan import (  # noqa: E402
    add_integer_pixel_jitter,
    make_grid_scan,
)
from tgv_ptycho.forward.scan_windows import (  # noqa: E402
    ScanWindowPlan,
    extract_scan_window,
    make_scan_window_plan,
    minimum_large_canvas_shape,
    scan_window_adjoint_relative_error,
)
from tgv_ptycho.inverse.exp031 import (  # noqa: E402
    aligned_square_shape,
    change_class,
    detector_dynamic_range_metrics,
    detector_sensitivity_metrics,
    gaussian_amplitude_for_total_power,
    gaussian_square_captured_power_fraction,
    gaussian_square_side_for_omitted_power,
    gaussian_total_power,
    poisson_fisher_per_incident_photon,
    probe_sensitivity_metrics,
    relative_change,
)
from tgv_ptycho.io.config import config_to_yaml, load_config, save_config  # noqa: E402
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json  # noqa: E402
from tgv_ptycho.objects.sample_b import (  # noqa: E402
    PhysicalPhaseCellMap,
    make_physical_phase_cell_map,
    rasterize_physical_phase_values,
)
from tgv_ptycho.recon.exp031 import (  # noqa: E402
    measurement_scaled_reference_initialization,
    reconstruct_known_b_probe_open,
    simulation_evaluation_probe_error,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--plot-existing-run",
        type=Path,
        help="Read a completed numerical run and generate figures only.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Create a diagnostic run containing resource and geometry estimates.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}.")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=_json_default)


def _norm(values: np.ndarray, axis: tuple[int, ...] | None = None) -> Any:
    """BLAS-free Euclidean norm for this repository's Windows runtime."""

    squared = np.abs(np.asarray(values)) ** 2
    return np.sqrt(np.sum(squared, axis=axis, dtype=np.float64))


def _small_matrix_inverse_no_blas(values: np.ndarray) -> np.ndarray:
    """Invert the tiny affine matrices used by Matplotlib without BLAS."""

    array = np.asarray(values)
    if array.ndim > 2:
        return np.stack([_small_matrix_inverse_no_blas(item) for item in array], axis=0)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("inverse input must contain square matrices.")
    dtype = np.result_type(array.dtype, np.float64)
    size = array.shape[0]
    augmented = np.concatenate(
        [array.astype(dtype, copy=True), np.eye(size, dtype=dtype)], axis=1
    )
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot, column]) <= np.finfo(np.float64).tiny:
            raise np.linalg.LinAlgError("singular matrix")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row in range(size):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]
    return augmented[:, size:]


def _write_h5(group: h5py.Group, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        child = group.require_group(str(name))
        for key, item in value.items():
            _write_h5(child, str(key), item)
        return
    if isinstance(value, str):
        group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
        return
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        group.create_dataset(
            name,
            data=np.asarray(value, dtype=h5py.string_dtype("utf-8")),
        )
        return
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, dict) for item in value
    ):
        child = group.require_group(str(name))
        for index, item in enumerate(value):
            _write_h5(child, f"item_{index:03d}", item)
        return
    group.create_dataset(name, data=np.asarray(value))


def _make_scan(config: dict[str, Any]) -> np.ndarray:
    scan = config["scan"]
    if scan["type"] != "jittered_grid":
        raise ValueError("exp031 requires the preregistered jittered_grid scan.")
    regular = make_grid_scan(
        int(scan["num_x"]),
        int(scan["num_y"]),
        float(scan["step_m"]),
        center=bool(scan["center"]),
    )
    positions = add_integer_pixel_jitter(
        regular,
        float(scan["jitter_quantum_m"]),
        int(scan["max_jitter_px"]),
        seed=int(scan["jitter_seed"]),
    )
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError("scan positions contain duplicates.")
    return np.asarray(positions, dtype=np.float64)


def _case_id(spot_m: float, suffix: str) -> str:
    return f"gaussian_{int(round(spot_m * 1.0e6)):03d}um_{suffix}"


def _case_descriptors(config: dict[str, Any]) -> list[dict[str, Any]]:
    active = config["active_window"]
    illumination = config["illumination"]
    dx_m = float(config["optics"]["dx_m"])
    feature_m = float(config["sample_b"]["feature_size_m"])
    formal_omitted = float(active["formal_omitted_incident_power_max"])
    strict_omitted = float(active["strict_omitted_incident_power_max"])
    strict_spots = {float(value) for value in active["strict_control_spot_diameter_m"]}
    residual_support_spot = float(active["residual_support_spot_diameter_m"])
    residual_support_omitted = float(
        active["residual_support_omitted_incident_power_max"]
    )
    residual_support_control_omitted = float(
        active["residual_support_control_omitted_incident_power_max"]
    )
    cases: list[dict[str, Any]] = []
    for spot_m in (
        float(value) for value in illumination["spot_diameter_1e2_intensity_m"]
    ):
        case_formal_omitted = (
            residual_support_omitted
            if spot_m == residual_support_spot
            else formal_omitted
        )
        formal_side = gaussian_square_side_for_omitted_power(
            0.5 * spot_m, case_formal_omitted
        )
        cases.append(
            {
                "id": _case_id(spot_m, "formal"),
                "kind": "gaussian",
                "role": "formal",
                "spot_diameter_1e2_intensity_m": spot_m,
                "omitted_power_target": case_formal_omitted,
                "support_basis": (
                    "residual_edge_convergence"
                    if spot_m == residual_support_spot
                    else "formal_incident_power"
                ),
                "active_shape": aligned_square_shape(
                    formal_side, dx_m, feature_size_m=feature_m
                ),
                "padding_guard_m_per_side": float(
                    config["open_boundary"]["padding_guard_m_per_side"]
                ),
            }
        )
        if spot_m in strict_spots:
            case_strict_omitted = (
                residual_support_control_omitted
                if spot_m == residual_support_spot
                else strict_omitted
            )
            strict_side = gaussian_square_side_for_omitted_power(
                0.5 * spot_m, case_strict_omitted
            )
            cases.append(
                {
                    "id": _case_id(spot_m, "strict_fov"),
                    "kind": "gaussian",
                    "role": (
                        "residual_support_control"
                        if spot_m == residual_support_spot
                        else "strict_fov_control"
                    ),
                    "spot_diameter_1e2_intensity_m": spot_m,
                    "omitted_power_target": case_strict_omitted,
                    "support_basis": (
                        "residual_edge_convergence_control"
                        if spot_m == residual_support_spot
                        else "strict_incident_power_control"
                    ),
                    "active_shape": aligned_square_shape(
                        strict_side, dx_m, feature_size_m=feature_m
                    ),
                    "padding_guard_m_per_side": float(
                        config["open_boundary"]["padding_guard_m_per_side"]
                    ),
                }
            )
    padding_spot = float(config["open_boundary"]["padding_control_spot_diameter_m"])
    formal = next(
        item
        for item in cases
        if item["role"] == "formal"
        and item["spot_diameter_1e2_intensity_m"] == padding_spot
    )
    formal_padding = float(config["open_boundary"]["padding_guard_m_per_side"])
    padding_sequence = [
        float(value)
        for value in config["open_boundary"]["padding_sequence_guard_m_per_side"]
    ]
    if padding_sequence != sorted(set(padding_sequence)):
        raise ValueError("padding sequence must be strictly increasing and unique.")
    if formal_padding not in padding_sequence:
        raise ValueError("padding sequence must include the formal padding guard.")
    if float(config["open_boundary"]["padding_control_guard_m_per_side"]) != max(
        padding_sequence
    ):
        raise ValueError("padding control guard must equal the largest sequence guard.")
    for guard_m in padding_sequence:
        if guard_m == formal_padding:
            continue
        guard_um = int(round(guard_m * 1.0e6))
        cases.append(
            {
                **formal,
                "id": _case_id(
                    padding_spot, f"padding_{guard_um:03d}um_control"
                ),
                "role": "padding_control",
                "padding_guard_m_per_side": guard_m,
            }
        )
    if bool(illumination["include_plane_wave_fixed_center"]):
        cases.append(
            {
                "id": "plane_wave_legacy_roi",
                "kind": "plane_wave",
                "role": "formal_reference",
                "spot_diameter_1e2_intensity_m": None,
                "omitted_power_target": None,
                "active_shape": tuple(
                    int(value) for value in active["plane_wave_legacy_shape"]
                ),
                "padding_guard_m_per_side": float(
                    config["open_boundary"]["padding_guard_m_per_side"]
                ),
            }
        )
    for case in cases:
        case["open_shape"] = make_open_shape(
            case["active_shape"],
            dx_m,
            case["padding_guard_m_per_side"],
        )
        case["active_size_yx_m"] = [
            case["active_shape"][0] * dx_m,
            case["active_shape"][1] * dx_m,
        ]
    return cases


def _probe_envelope_comparison_shape(
    config: dict[str, Any], cases: list[dict[str, Any]]
) -> tuple[int, int]:
    """Return the fixed centered probe-envelope reporting domain."""

    raw_shape = config["active_window"]["probe_envelope_comparison_shape"]
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        raise ValueError("probe envelope comparison shape must contain (ny, nx).")
    shape = tuple(int(value) for value in raw_shape)
    if any(value <= 0 for value in shape):
        raise ValueError("probe envelope comparison shape must be positive.")
    for case in cases:
        if case["role"] not in {"formal", "formal_reference"}:
            continue
        active_shape = tuple(int(value) for value in case["active_shape"])
        shape_pairs = tuple(zip(shape, active_shape, strict=True))
        if any(target > active for target, active in shape_pairs):
            raise ValueError(
                f"probe envelope comparison shape does not fit case {case['id']}."
            )
        if any((active - target) % 2 for target, active in shape_pairs):
            raise ValueError(
                "probe envelope comparison shape is not center-aligned for "
                f"{case['id']}."
            )
    return shape


def _geometry(config: dict[str, Any], positions: np.ndarray) -> dict[str, Any]:
    dx_m = float(config["optics"]["dx_m"])
    feature_m = float(config["sample_b"]["feature_size_m"])
    pixels_per_cell = int(round(feature_m / dx_m))
    descriptors = _case_descriptors(config)
    probe_envelope_shape = _probe_envelope_comparison_shape(config, descriptors)
    largest_shape = max(
        (item["active_shape"] for item in descriptors if item["kind"] == "gaussian"),
        key=lambda shape: shape[0] * shape[1],
    )
    guard_m = int(config["sample_b"]["guard_feature_cells_per_side"]) * feature_m
    master_shape = minimum_large_canvas_shape(
        largest_shape,
        positions,
        dx_m,
        guard_m_per_side=guard_m,
        alignment_px_yx=pixels_per_cell,
        preserve_centered_parity=bool(
            config["active_window"]["preserve_centered_parity"]
        ),
    )
    legacy_shape = tuple(int(value) for value in config["optics"]["detector_roi_shape"])
    legacy_recommended_shape = minimum_large_canvas_shape(
        legacy_shape,
        positions,
        dx_m,
        guard_m_per_side=guard_m,
        alignment_px_yx=pixels_per_cell,
        preserve_centered_parity=True,
    )
    span_xy_m = np.ptp(positions, axis=0)
    legacy_size_xy_m = np.asarray(legacy_shape[::-1]) * dx_m
    continuous_required_xy_m = legacy_size_xy_m + span_xy_m
    guarded_lower_xy_m = continuous_required_xy_m + 2.0 * guard_m
    return {
        "case_descriptors": descriptors,
        "pixels_per_feature_cell": pixels_per_cell,
        "scan_min_xy_m": np.min(positions, axis=0).tolist(),
        "scan_max_xy_m": np.max(positions, axis=0).tolist(),
        "scan_span_xy_m": span_xy_m.tolist(),
        "legacy_active_shape": list(legacy_shape),
        "legacy_active_size_xy_m": legacy_size_xy_m.tolist(),
        "legacy_continuous_required_B_size_xy_m": continuous_required_xy_m.tolist(),
        "legacy_guarded_continuous_lower_bound_xy_m": guarded_lower_xy_m.tolist(),
        "legacy_recommended_B_shape": list(legacy_recommended_shape),
        "legacy_recommended_B_size_xy_m": (
            np.asarray(legacy_recommended_shape[::-1]) * dx_m
        ).tolist(),
        "largest_control_active_shape": list(largest_shape),
        "master_B_shape": list(master_shape),
        "master_B_size_xy_m": (np.asarray(master_shape[::-1]) * dx_m).tolist(),
        "probe_envelope_comparison_shape": list(probe_envelope_shape),
        "probe_envelope_comparison_size_yx_m": (
            np.asarray(probe_envelope_shape, dtype=np.float64) * dx_m
        ).tolist(),
        "probe_envelope_comparison_domain": "fixed_config_centered_B_plane",
        "guard_m_per_side": guard_m,
        "formula": (
            "L_B,axis >= L_window,axis + scan_span,axis + 2*guard; "
            "then cell/parity align"
        ),
    }


def _preflight_metrics(
    config: dict[str, Any], positions: np.ndarray, geometry: dict[str, Any]
) -> dict[str, Any]:
    n_frames = len(positions)
    roi_shape = tuple(int(value) for value in config["optics"]["detector_roi_shape"])
    case_metrics: dict[str, Any] = {}
    total_fft_work = 0.0
    maximum_peak = 0
    for case in geometry["case_descriptors"]:
        peak = estimate_case_peak_memory_bytes(
            tuple(case["active_shape"]), tuple(case["open_shape"]), batch_size=4
        )
        maximum_peak = max(maximum_peak, peak)
        opened = int(np.prod(case["open_shape"]))
        transforms = 8 * n_frames
        total_fft_work += transforms * opened * math.log2(max(opened, 2))
        case_metrics[case["id"]] = {
            "role": case["role"],
            "active_shape": list(case["active_shape"]),
            "active_size_yx_m": case["active_size_yx_m"],
            "open_shape": list(case["open_shape"]),
            "padding_guard_m_per_side": case["padding_guard_m_per_side"],
            "peak_memory_bytes": peak,
        }
    representative_count = len(config["reconstruction"]["representative_cases"])
    reconstruction_factor = 6 * n_frames * representative_count
    representative_opened = max(
        int(np.prod(case["open_shape"]))
        for case in geometry["case_descriptors"]
        if case["role"] in {"formal", "formal_reference"}
    )
    total_fft_work += (
        reconstruction_factor
        * representative_opened
        * math.log2(max(representative_opened, 2))
    )
    estimated_runtime_s = total_fft_work / 2.0e8
    formal_count = sum(
        case["role"] in {"formal", "formal_reference"}
        for case in geometry["case_descriptors"]
    )
    roi_stack_bytes = n_frames * int(np.prod(roi_shape)) * 4
    estimated_hdf5_bytes = int(
        formal_count * roi_stack_bytes
        + 6 * roi_stack_bytes
        + np.prod(geometry["master_B_shape"]) * 2
        + 250.0e6
    )
    memory_limit = float(config["preflight"]["maximum_peak_memory_bytes"])
    runtime_limit = float(config["preflight"]["maximum_estimated_runtime_s"])
    return {
        "status": "passed"
        if maximum_peak <= memory_limit and estimated_runtime_s <= runtime_limit
        else "failed",
        "cases": case_metrics,
        "maximum_peak_memory_bytes": maximum_peak,
        "maximum_peak_memory_GiB": maximum_peak / 2**30,
        "estimated_runtime_s": estimated_runtime_s,
        "estimated_hdf5_bytes": estimated_hdf5_bytes,
        "estimated_hdf5_GiB": estimated_hdf5_bytes / 2**30,
        "limits": {
            "maximum_peak_memory_bytes": memory_limit,
            "maximum_estimated_runtime_s": runtime_limit,
        },
        "estimation_model": {
            "fft_work_rate_pixel_log2_per_s": 2.0e8,
            "forward_batch_size": 4,
            "streaming": "one illumination case and one frame at a time",
            "interpretation": "planning estimate, not a measured benchmark",
        },
    }


def _metadata(config: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "experiment_id": "exp031",
        "document_version": config["experiment"]["document_version"],
        "change_id": config["experiment"]["change_id"],
        "created_at_utc": created_at_utc(),
        "git_commit": get_git_commit(REPO_ROOT),
        "mode": mode,
        "python": sys.version,
        "platform": sys.platform,
        "working_array_precision": config["optics"]["propagation_precision"],
        "position_columns": ["x", "y"],
        "position_units": "m",
        "dx_tuple_order": ["dy", "dx"],
        "spot_definition": (
            "spot_diameter_1e2_intensity_m=2w; "
            "A=A0 exp(-r^2/w^2); I=|A0|^2 exp(-2r^2/w^2)"
        ),
        "model_scope": "2D axisymmetric centered projected phase; not 3D detectability",
    }


def _initialize_run(
    config: dict[str, Any], config_path: Path, mode: str
) -> tuple[Path, dict[str, Any]]:
    run_name = str(config["run"]["name"])
    if mode == "preflight":
        run_name += "_preflight"
    output_root = (REPO_ROOT / str(config["run"]["output_root"])).resolve()
    run_dir = make_run_dir(output_root, run_name)
    save_config(run_dir / "config.yaml", config)
    metadata = _metadata(config, mode)
    metadata["source_config"] = str(config_path.resolve())
    save_json(run_dir / "metadata.json", metadata)
    save_json(
        run_dir / "run_state.json",
        {"status": "running", "mode": mode, "updated_at_utc": created_at_utc()},
    )
    return run_dir, metadata


def _save_preflight_hdf5(
    path: Path,
    config_yaml: str,
    metadata: dict[str, Any],
    positions: np.ndarray,
    geometry: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    with h5py.File(path, "x") as h5:
        entry = h5.require_group("entry")
        _write_h5(entry, "config_yaml", config_yaml)
        entry.require_group("data").create_dataset("scan_positions", data=positions)
        _write_h5(entry, "metadata", metadata)
        metrics = entry.require_group("metrics")
        _write_h5(metrics, "B_coverage", geometry)
        _write_h5(metrics, "numerical_convergence", {"preflight": preflight})


def _radial_cache(
    config: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[float | None, Any]:
    dx_m = float(config["optics"]["dx_m"])
    output_step = float(config["projected_phase"]["radial_output_step_m"])
    cache: dict[float | None, Any] = {}
    keys: list[float | None] = sorted(
        {
            None
            if case["kind"] == "plane_wave"
            else 0.5 * float(case["spot_diameter_1e2_intensity_m"])
            for case in cases
        },
        key=lambda value: math.inf if value is None else value,
    )
    for waist_radius in keys:
        matching = [
            case
            for case in cases
            if (case["kind"] == "plane_wave" and waist_radius is None)
            or (
                case["kind"] == "gaussian"
                and waist_radius == 0.5 * case["spot_diameter_1e2_intensity_m"]
            )
        ]
        max_radius = max(
            0.5 * dx_m * math.sqrt(shape[0] ** 2 + shape[1] ** 2)
            for shape in (case["active_shape"] for case in matching)
        )
        cache[waist_radius] = make_tgv_radial_probe_result(
            tgv={
                key: float(value) if key != "type" else value
                for key, value in config["tgv"].items()
                if key not in {"type", "center_xy_m"}
            },
            wavelength_m=float(config["optics"]["wavelength_m"]),
            z_AB_m=float(config["optics"]["z_AB_m"]),
            radial_output_max_m=max_radius + output_step,
            radial_output_step_m=output_step,
            finite_difference_step_m=float(config["sensitivity"]["delta_d_waist_m"]),
            core_step_m=float(config["projected_phase"]["radial_core_step_m"]),
            transition_step_m=float(
                config["projected_phase"]["radial_transition_step_m"]
            ),
            gaussian_waist_radius_m=waist_radius,
            amplitude=float(config["illumination"]["fixed_center_amplitude"]),
            refractive_index=float(config["optics"]["medium_index"]),
        )
    return cache


def _coordinates(shape: tuple[int, int], dx_m: float) -> tuple[np.ndarray, np.ndarray]:
    y = (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * dx_m
    x = (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * dx_m
    return y, x


def _case_fields(
    config: dict[str, Any], case: dict[str, Any], radial: Any
) -> dict[str, np.ndarray]:
    optics = config["optics"]
    shape = tuple(case["active_shape"])
    detector_shape = tuple(int(value) for value in optics["detector_roi_shape"])
    dx_m = float(optics["dx_m"])
    wavelength_m = float(optics["wavelength_m"])
    z_ab = float(optics["z_AB_m"])
    z_bc = float(optics["z_BC_m"])
    n_medium = float(optics["medium_index"])
    amplitude = float(config["illumination"]["fixed_center_amplitude"])
    y, x = _coordinates(shape, dx_m)
    radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    yd, xd = _coordinates(detector_shape, dx_m)
    detector_radius = np.sqrt(yd[:, None] ** 2 + xd[None, :] ** 2)
    if case["kind"] == "gaussian":
        waist = 0.5 * float(case["spot_diameter_1e2_intensity_m"])
        reference_b = gaussian_reference_radial(
            radius,
            waist_radius_m=waist,
            amplitude=amplitude,
            wavelength_m=wavelength_m,
            distance_m=z_ab,
            refractive_index=n_medium,
        )
        reference_detector = gaussian_reference_radial(
            detector_radius,
            waist_radius_m=waist,
            amplitude=amplitude,
            wavelength_m=wavelength_m,
            distance_m=z_ab + z_bc,
            refractive_index=n_medium,
        )
    else:
        reference_b = plane_wave_reference(
            shape,
            amplitude=amplitude,
            wavelength_m=wavelength_m,
            distance_m=z_ab,
            refractive_index=n_medium,
        )
        reference_detector = plane_wave_reference(
            detector_shape,
            amplitude=amplitude,
            wavelength_m=wavelength_m,
            distance_m=z_ab + z_bc,
            refractive_index=n_medium,
        )
    delta_minus = sample_radial_field(
        radial.output_radius_m, radial.probe_delta_minus, shape, dx_m
    )
    delta_base = sample_radial_field(
        radial.output_radius_m, radial.probe_delta_baseline, shape, dx_m
    )
    delta_plus = sample_radial_field(
        radial.output_radius_m, radial.probe_delta_plus, shape, dx_m
    )
    return {
        "reference_b": np.asarray(reference_b, dtype=np.complex64),
        "reference_detector": np.asarray(reference_detector, dtype=np.complex64),
        "probe_minus": np.asarray(reference_b + delta_minus, dtype=np.complex64),
        "probe_baseline": np.asarray(reference_b + delta_base, dtype=np.complex64),
        "probe_plus": np.asarray(reference_b + delta_plus, dtype=np.complex64),
    }


def _phase_to_transmission(phase: np.ndarray) -> np.ndarray:
    values = np.asarray(phase, dtype=np.float32)
    return np.asarray(np.cos(values) + 1j * np.sin(values), dtype=np.complex64)


def _patch_getter(
    phase_canvas: np.ndarray,
    plan: ScanWindowPlan,
    *,
    periodic: bool = False,
):
    if periodic:
        center_plan = make_scan_window_plan(
            plan.large_shape,
            plan.window_shape,
            np.zeros((1, 2), dtype=np.float64),
            plan.dx_yx_m,
        )
        center = _phase_to_transmission(
            extract_scan_window(phase_canvas, center_plan, 0)
        )

        def get_periodic(index: int) -> np.ndarray:
            shift_x, shift_y = plan.shifts_xy_px[index]
            return np.roll(
                center,
                shift=(int(shift_y), int(shift_x)),
                axis=(0, 1),
            )

        return get_periodic

    def get_aperiodic(index: int) -> np.ndarray:
        return _phase_to_transmission(extract_scan_window(phase_canvas, plan, index))

    return get_aperiodic


def _illumination_metrics(
    config: dict[str, Any], case: dict[str, Any], probe: np.ndarray
) -> dict[str, Any]:
    dx_m = float(config["optics"]["dx_m"])
    amplitude = float(config["illumination"]["fixed_center_amplitude"])
    shape = tuple(case["active_shape"])
    y, x = _coordinates(shape, dx_m)
    radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    tgv = config["tgv"]
    if case["kind"] == "gaussian":
        waist = 0.5 * float(case["spot_diameter_1e2_intensity_m"])
        incident = amplitude**2 * np.exp(-2.0 * radius**2 / waist**2)
        full_power = gaussian_total_power(amplitude, waist)
        waist_relative = float(np.exp(-2.0 * (0.5 * tgv["d_waist_m"]) ** 2 / waist**2))
        top_relative = float(np.exp(-2.0 * (0.5 * tgv["d_top_m"]) ** 2 / waist**2))
        top_power = float(1.0 - np.exp(-2.0 * (0.5 * tgv["d_top_m"]) ** 2 / waist**2))
        captured = gaussian_square_captured_power_fraction(shape[0] * dx_m, waist)
    else:
        incident = np.full(shape, amplitude**2, dtype=np.float64)
        full_power = amplitude**2 * shape[0] * shape[1] * dx_m**2
        waist_relative = 1.0
        top_relative = 1.0
        top_power = (
            math.pi
            * (0.5 * float(tgv["d_top_m"])) ** 2
            / (shape[0] * shape[1] * dx_m**2)
        )
        captured = 1.0
    ring_px = max(1, int(round(config["active_window"]["edge_ring_width_m"] / dx_m)))
    total_discrete = float(np.sum(incident, dtype=np.float64))
    interior = incident[ring_px:-ring_px, ring_px:-ring_px]
    edge_fraction = float(
        (total_discrete - np.sum(interior, dtype=np.float64))
        / max(total_discrete, np.finfo(float).tiny)
    )
    intensity_b = np.abs(probe) ** 2
    probe_power = float(np.sum(intensity_b, dtype=np.float64) * dx_m**2)
    mean_r2 = float(
        np.sum(radius**2 * intensity_b, dtype=np.float64)
        * dx_m**2
        / max(probe_power, np.finfo(float).tiny)
    )
    return {
        "center_intensity_fixed_center": amplitude**2,
        "waist_radius_relative_intensity": waist_relative,
        "top_radius_relative_intensity": top_relative,
        "tgv_top_aperture_power_fraction": top_power,
        "active_window_captured_power_fraction": captured,
        "active_window_omitted_power_fraction": 1.0 - captured,
        "incident_edge_ring_energy_fraction": edge_fraction,
        "B_plane_second_moment_spot_diameter_m": 2.0 * math.sqrt(2.0 * mean_r2),
        "incident_power_fixed_center": full_power,
        "illumination_nonuniformity_over_top_radius": 1.0 - top_relative,
    }


def _simulate_case(
    config: dict[str, Any],
    case: dict[str, Any],
    positions: np.ndarray,
    phase_canvas: np.ndarray,
    master_shape: tuple[int, int],
    radial: Any,
    *,
    periodic: bool = False,
) -> dict[str, Any]:
    optics = config["optics"]
    dx_m = float(optics["dx_m"])
    detector_shape = tuple(int(value) for value in optics["detector_roi_shape"])
    plan = make_scan_window_plan(
        master_shape, tuple(case["active_shape"]), positions, dx_m
    )
    get_patch = _patch_getter(phase_canvas, plan, periodic=periodic)
    fields = _case_fields(config, case, radial)
    transfer = make_asm_transfer_complex64(
        tuple(case["open_shape"]),
        dx_m=dx_m,
        wavelength_m=float(optics["wavelength_m"]),
        distance_m=float(optics["z_BC_m"]),
        refractive_index=float(optics["medium_index"]),
        bandlimit=bool(optics["bandlimit"]),
        alias_control=bool(optics["alias_control"]),
    )
    n_frames = len(positions)
    stacks = {
        name: np.empty((n_frames, *detector_shape), dtype=np.float32)
        for name in ("minus", "baseline", "plus", "background")
    }
    edge_fractions = np.empty(n_frames, dtype=np.float64)
    repeat_difference = math.nan
    ring_px = max(
        1,
        int(round(config["active_window"]["edge_ring_width_m"] / dx_m)),
    )
    probes = np.stack(
        [
            fields["probe_minus"],
            fields["probe_baseline"],
            fields["probe_plus"],
            fields["reference_b"],
        ]
    )
    start = time.perf_counter()
    for index in range(n_frames):
        patch = get_patch(index)
        detector_fields = propagate_probe_patch_batch_to_roi(
            probes,
            fields["reference_b"],
            patch,
            fields["reference_detector"],
            transfer,
            workers=int(config["open_boundary"]["fft_workers"]),
        )
        intensities = np.abs(detector_fields) ** 2
        for output_index, name in enumerate(stacks):
            stacks[name][index] = intensities[output_index]
        residual = fields["probe_baseline"] * patch - fields["reference_b"]
        edge_fractions[index] = residual_edge_energy_fraction(residual, ring_px)
        if index == 0 and not periodic:
            repeated = propagate_probe_patch_batch_to_roi(
                probes,
                fields["reference_b"],
                patch,
                fields["reference_detector"],
                transfer,
                workers=int(config["open_boundary"]["fft_workers"]),
            )
            repeat_difference = float(np.max(np.abs(repeated - detector_fields)))
    elapsed = time.perf_counter() - start
    step = float(config["sensitivity"]["delta_d_waist_m"])
    d_probe = (fields["probe_plus"] - fields["probe_minus"]) / (2.0 * step)
    d_intensity = (stacks["plus"] - stacks["minus"]) / (2.0 * step)
    illumination = _illumination_metrics(config, case, fields["probe_baseline"])
    if case["kind"] == "gaussian":
        waist = 0.5 * float(case["spot_diameter_1e2_intensity_m"])
        incident_fixed_center = gaussian_total_power(
            float(config["illumination"]["fixed_center_amplitude"]), waist
        )
        fixed_total_scale = gaussian_amplitude_for_total_power(
            float(config["illumination"]["fixed_total_incident_power"]), waist
        ) / float(config["illumination"]["fixed_center_amplitude"])
    else:
        incident_fixed_center = (
            float(config["illumination"]["fixed_center_amplitude"]) ** 2
            * case["active_shape"][0]
            * case["active_shape"][1]
            * dx_m**2
        )
        fixed_total_scale = None
    pixel_area = float(optics["detector_pixel_size_m"]) ** 2
    parameter_scale = float(config["tgv"]["d_waist_m"])
    fixed_center = {
        "probe_sensitivity": probe_sensitivity_metrics(
            fields["probe_baseline"],
            d_probe,
            parameter_scale,
            dx_m=dx_m,
            incident_power=incident_fixed_center,
        ),
        "detector_sensitivity": detector_sensitivity_metrics(
            stacks["baseline"],
            d_intensity,
            parameter_scale,
            pixel_area_m2=pixel_area,
            incident_power=incident_fixed_center,
        ),
        "poisson_fisher": poisson_fisher_per_incident_photon(
            stacks["baseline"],
            d_intensity,
            pixel_area_m2=pixel_area,
            incident_power_per_frame=incident_fixed_center,
            mu_epsilon=float(config["sensitivity"]["poisson_mu_epsilon"]),
        ),
        "detector_diagnostics": detector_dynamic_range_metrics(
            stacks["baseline"],
            pixel_area_m2=pixel_area,
            incident_power_per_frame=incident_fixed_center,
            low_signal_fraction_of_peak=float(
                config["reporting"]["low_signal_fraction_of_peak"]
            ),
            percentile_low=float(config["reporting"]["detector_percentile_low"]),
            percentile_high=float(config["reporting"]["detector_percentile_high"]),
        ),
        "incident_power_per_frame": incident_fixed_center,
        "mean_detector_absolute_signal": float(
            np.mean(np.sum(stacks["baseline"], axis=(1, 2), dtype=np.float64))
            * pixel_area
        ),
    }
    tgv_differential = float(
        _norm(stacks["baseline"] - stacks["background"])
        / max(_norm(stacks["background"]), np.finfo(float).tiny)
    )
    fixed_center["detector_diagnostics"][
        "TGV_differential_signal_relative_to_glass_B_background"
    ] = tgv_differential
    branches: dict[str, Any] = {"fixed_center_intensity": fixed_center}
    if fixed_total_scale is not None:
        scale2 = fixed_total_scale**2
        fixed_total_power = float(config["illumination"]["fixed_total_incident_power"])
        fixed_total = {
            "probe_sensitivity": probe_sensitivity_metrics(
                fixed_total_scale * fields["probe_baseline"],
                fixed_total_scale * d_probe,
                parameter_scale,
                dx_m=dx_m,
                incident_power=fixed_total_power,
            ),
            "detector_sensitivity": detector_sensitivity_metrics(
                scale2 * stacks["baseline"],
                scale2 * d_intensity,
                parameter_scale,
                pixel_area_m2=pixel_area,
                incident_power=fixed_total_power,
            ),
            "poisson_fisher": poisson_fisher_per_incident_photon(
                scale2 * stacks["baseline"],
                scale2 * d_intensity,
                pixel_area_m2=pixel_area,
                incident_power_per_frame=fixed_total_power,
                mu_epsilon=float(config["sensitivity"]["poisson_mu_epsilon"]),
            ),
            "detector_diagnostics": detector_dynamic_range_metrics(
                scale2 * stacks["baseline"],
                pixel_area_m2=pixel_area,
                incident_power_per_frame=fixed_total_power,
                low_signal_fraction_of_peak=float(
                    config["reporting"]["low_signal_fraction_of_peak"]
                ),
                percentile_low=float(config["reporting"]["detector_percentile_low"]),
                percentile_high=float(config["reporting"]["detector_percentile_high"]),
            ),
            "incident_power_per_frame": fixed_total_power,
            "fixed_center_to_total_field_amplitude_scale": fixed_total_scale,
            "mean_detector_absolute_signal": float(
                scale2
                * np.mean(np.sum(stacks["baseline"], axis=(1, 2), dtype=np.float64))
                * pixel_area
            ),
        }
        fixed_total["detector_diagnostics"][
            "TGV_differential_signal_relative_to_glass_B_background"
        ] = tgv_differential
        branches["fixed_total_power"] = fixed_total
    first_patch = get_patch(0)
    coverage = {
        "active_window_shape": list(case["active_shape"]),
        "active_window_physical_size_yx_m": case["active_size_yx_m"],
        "per_position_margins_px_left_right_top_bottom": plan.margins_px.tolist(),
        "minimum_coverage_margin_px": plan.minimum_margin_px,
        "minimum_coverage_margin_m": plan.minimum_margin_m,
        "patch_fully_inside": plan.fully_inside.tolist(),
        "all_patches_fully_inside": bool(np.all(plan.fully_inside)),
        "periodic_wrap_count": int(n_frames if periodic else 0),
        "visited_pixel_count": int(np.count_nonzero(plan.visited_region)),
        "unvisited_pixel_count": int(np.count_nonzero(plan.unvisited_region)),
        "energy_weighted_coded_illumination_fraction": float(
            illumination["active_window_captured_power_fraction"]
        ),
        "center_alignment_offset_xy_m": list(plan.center_alignment_offset_xy_m),
    }
    numerical = {
        "residual_edge_energy_fraction_per_frame": edge_fractions.tolist(),
        "residual_edge_energy_fraction_minimum": float(np.min(edge_fractions)),
        "residual_edge_energy_fraction_median": float(np.median(edge_fractions)),
        "residual_edge_energy_fraction_maximum": float(np.max(edge_fractions)),
        "reference_plus_residual_identity_relative_error": (
            reference_plus_residual_identity_error(
                fields["reference_b"],
                fields["probe_baseline"] - fields["reference_b"],
                first_patch,
            )
        ),
        "same_config_repeat_max_abs_field_difference": repeat_difference,
        "intensity_nonnegative": bool(
            all(np.all(stack >= 0) for stack in stacks.values())
        ),
        "finite": bool(finite_metrics(stacks)),
        "elapsed_s": elapsed,
    }
    return {
        "case": case,
        "plan": plan,
        "fields": fields,
        "transfer": transfer,
        "stacks": stacks,
        "d_probe": d_probe,
        "d_intensity": d_intensity,
        "illumination": illumination,
        "coverage": coverage,
        "branches": branches,
        "numerical": numerical,
    }


def _common_center_crop(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ny, nx = values.shape[-2:]
    sy = (ny - shape[0]) // 2
    sx = (nx - shape[1]) // 2
    return values[..., sy : sy + shape[0], sx : sx + shape[1]]


def _normalized_envelope_difference(first: np.ndarray, second: np.ndarray) -> float:
    a = np.abs(np.asarray(first, dtype=np.complex128))
    b = np.abs(np.asarray(second, dtype=np.complex128))
    a /= max(float(_norm(a)), np.finfo(float).tiny)
    b /= max(float(_norm(b)), np.finfo(float).tiny)
    return float(_norm(a - b))


def _save_formal_case_arrays(
    entry: h5py.Group,
    result: dict[str, Any],
    *,
    representative: bool,
) -> None:
    case_id = result["case"]["id"]
    case_group = entry.require_group("data").require_group("illumination_cases")
    group = case_group.require_group(case_id)
    group.create_dataset("I_stack_fixed_center", data=result["stacks"]["baseline"])
    if representative:
        group.create_dataset(
            "I_stack_minus_fixed_center", data=result["stacks"]["minus"]
        )
        group.create_dataset("I_stack_plus_fixed_center", data=result["stacks"]["plus"])


def _add_relative_ratios(metrics: dict[str, Any]) -> None:
    formal_ids = [_case_id(spot * 1.0e-6, "formal") for spot in (50, 100, 200, 400)]
    plane_id = "plane_wave_legacy_roi"
    for branch in ("fixed_center_intensity", "fixed_total_power"):
        if branch not in metrics["branches"]:
            continue
        cases = metrics["branches"][branch]
        reference_100 = cases[_case_id(100.0e-6, "formal")]
        for case_id in formal_ids:
            case = cases[case_id]
            for section, key in (
                ("detector_sensitivity", "normalized_sensitivity"),
                ("detector_sensitivity", "raw_derivative_l2_per_m"),
                (
                    "poisson_fisher",
                    "fisher_information_per_incident_photon_per_m2",
                ),
                ("detector_diagnostics", "maximum_to_median_ratio"),
            ):
                value = float(case[section][key])
                reference = float(reference_100[section][key])
                case[f"{section}_{key}_ratio_to_gaussian_100um"] = value / reference
                case[f"{section}_{key}_change_class_vs_gaussian_100um"] = change_class(
                    relative_change(value, reference)
                )
            if branch == "fixed_center_intensity":
                plane = cases[plane_id]
                for section, key in (
                    ("detector_sensitivity", "normalized_sensitivity"),
                    (
                        "poisson_fisher",
                        "fisher_information_per_incident_photon_per_m2",
                    ),
                ):
                    value = float(case[section][key])
                    reference = float(plane[section][key])
                    case[f"{section}_{key}_ratio_to_plane_wave"] = value / reference
                    case[f"{section}_{key}_change_class_vs_plane_wave"] = change_class(
                        relative_change(value, reference)
                    )


def _plot_outputs(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    cells: PhysicalPhaseCellMap,
    phase_canvas_shape: tuple[int, int],
    coverage_map: np.ndarray,
    representative_patches: list[np.ndarray],
) -> list[str]:
    figures = run_dir / "figures"
    names: list[str] = []

    def save(name: str) -> None:
        plt.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.88, wspace=0.30)
        plt.savefig(figures / name, dpi=150)
        plt.close()
        names.append(name)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(cells.phase_rad, cmap="twilight", origin="lower")
    axes[0].set_title("master physical phase cells")
    for axis, patch, title in zip(
        axes[1:],
        representative_patches,
        ["corner 1", "corner 2", "corner 3", "corner 4"],
        strict=True,
    ):
        axis.imshow(np.angle(patch), cmap="twilight", origin="lower")
        axis.set_title(title)
    fig.suptitle("Finite aperiodic B; phase [rad], coordinates sampled in µm")
    save("B_master_and_corner_patches.png")

    plt.figure(figsize=(6, 5))
    extent = (
        np.asarray(
            [
                -phase_canvas_shape[1],
                phase_canvas_shape[1],
                -phase_canvas_shape[0],
                phase_canvas_shape[0],
            ]
        )
        * float(config["optics"]["dx_m"])
        * 0.5e6
    )
    plot_stride = max(1, int(math.ceil(max(coverage_map.shape) / 768)))
    plt.imshow(
        coverage_map[::plot_stride, ::plot_stride],
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    plt.colorbar(label="number of visits")
    plt.xlabel("x [µm]")
    plt.ylabel("y [µm]")
    plt.title("Scan coverage on one finite nonperiodic B canvas")
    save("B_scan_coverage_map.png")

    spots = np.asarray([50.0, 100.0, 200.0, 400.0])
    formal_ids = [_case_id(value * 1.0e-6, "formal") for value in spots]
    illumination = metrics["illumination"]
    radius = np.linspace(0.0, 35.0, 400)
    plt.figure(figsize=(7, 5))
    for spot in spots:
        w = 0.5 * spot
        plt.plot(radius, np.exp(-2.0 * radius**2 / w**2), label=f"{spot:g} µm")
    plt.axvline(
        0.5 * config["tgv"]["d_waist_m"] * 1e6, color="k", ls="--", label="waist radius"
    )
    plt.axvline(
        0.5 * config["tgv"]["d_top_m"] * 1e6, color="0.4", ls=":", label="top radius"
    )
    plt.xlabel("A-plane radius [µm]")
    plt.ylabel("relative intensity")
    plt.title("A-plane Gaussian profiles; diameter is 1/e² intensity")
    plt.legend()
    save("illumination_profiles_with_TGV_overlay.png")

    plt.figure(figsize=(7, 5))
    plt.plot(
        spots,
        [
            illumination[case]["illumination_nonuniformity_over_top_radius"]
            for case in formal_ids
        ],
        "o-",
        label="top-radius nonuniformity",
    )
    plt.plot(
        spots,
        [
            illumination[case]["active_window_omitted_power_fraction"]
            for case in formal_ids
        ],
        "s-",
        label="omitted active-window power",
    )
    plt.yscale("log")
    plt.xlabel("spot diameter 1/e² intensity [µm]")
    plt.ylabel("fraction")
    plt.title("Spot power containment and TGV illumination uniformity")
    plt.legend()
    save("spot_power_and_uniformity.png")

    for filename, section, key, ylabel in (
        (
            "probe_sensitivity_vs_spot_size.png",
            "probe_sensitivity",
            "normalized_gauge_removed_sensitivity",
            "normalized probe sensitivity",
        ),
        (
            "detector_sensitivity_vs_spot_size.png",
            "detector_sensitivity",
            "normalized_sensitivity",
            "normalized detector sensitivity",
        ),
        (
            "poisson_information_vs_spot_size.png",
            "poisson_fisher",
            "fisher_information_per_incident_photon_per_m2",
            "FI per incident photon [m⁻²]",
        ),
        (
            "detector_dynamic_range_vs_spot_size.png",
            "detector_diagnostics",
            "maximum_to_median_ratio",
            "maximum / median",
        ),
    ):
        plt.figure(figsize=(7, 5))
        for branch, label in (
            ("fixed_center_intensity", "fixed center"),
            ("fixed_total_power", "fixed total power"),
        ):
            values = [
                metrics["branches"][branch][case][section][key] for case in formal_ids
            ]
            plt.plot(spots, values, "o-", label=label)
        plt.xlabel("spot diameter 1/e² intensity [µm]")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        save(filename)

    convergence = metrics["numerical_convergence"]
    plt.figure(figsize=(7, 5))
    labels = ["50 µm support", "400 µm FOV", "100 µm padding"]
    values = [
        convergence["fov_controls"][_case_id(50.0e-6, "formal")][
            "maximum_relative_change"
        ],
        convergence["fov_controls"][_case_id(400.0e-6, "formal")][
            "maximum_relative_change"
        ],
        convergence["padding_control"]["maximum_relative_change"],
    ]
    plt.bar(labels, 100.0 * np.asarray(values))
    plt.axhline(5.0, color="r", ls="--", label="5% numerical gate")
    plt.ylabel("maximum registered metric change [%]")
    plt.title("FOV and open-padding convergence")
    plt.legend()
    save("fov_convergence.png")
    return names


def _plot_reconstruction(run_dir: Path, metrics: dict[str, Any]) -> str:
    case_ids = list(metrics["reconstruction"]["cases"])
    labels = [case_id.replace("_formal", "") for case_id in case_ids]
    errors = [
        metrics["reconstruction"]["cases"][case_id]["baseline"][
            "simulation_evaluation_only"
        ]["complex_scalar_aligned_relative_l2_simulation_evaluation_only"]
        for case_id in case_ids
    ]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, errors)
    plt.ylabel("aligned probe relative L2 (simulation evaluation only)")
    plt.title("Known-B probe recovery; fixed one-epoch schedule")
    plt.xticks(rotation=20)
    plt.subplots_adjust(left=0.14, right=0.96, bottom=0.24, top=0.88)
    name = "known_B_recovery_vs_spot_size.png"
    plt.savefig(run_dir / "figures" / name, dpi=150)
    plt.close()
    return name


def _pillow_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _phase_image(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    phase = np.asarray(values, dtype=np.float64)
    red = 0.5 + 0.5 * np.sin(phase)
    green = 0.5 + 0.5 * np.sin(phase + 2.0 * np.pi / 3.0)
    blue = 0.5 + 0.5 * np.sin(phase + 4.0 * np.pi / 3.0)
    rgb = np.asarray(255.0 * np.stack([red, green, blue], axis=-1), dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.NEAREST)


def _heat_image(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    array = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(array))
    scale = max(float(np.max(array)) - minimum, np.finfo(float).tiny)
    normalized = (array - minimum) / scale
    red = np.clip(1.5 * normalized, 0.0, 1.0)
    green = np.clip(1.5 - 2.0 * np.abs(normalized - 0.5), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - normalized), 0.0, 1.0)
    rgb = np.asarray(255.0 * np.stack([red, green, blue], axis=-1), dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.NEAREST)


def _pillow_line_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x: np.ndarray,
    series: dict[str, np.ndarray],
    log_y: bool = False,
) -> None:
    width, height = 1000, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _pillow_font(20)
    small = _pillow_font(16)
    left, right, top, bottom = 125, 955, 80, 600
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    transformed: dict[str, np.ndarray] = {}
    for label, values in series.items():
        array = np.asarray(values, dtype=np.float64)
        if log_y:
            array = np.log10(np.maximum(array, np.finfo(float).tiny))
        transformed[label] = array
    all_y = np.concatenate(list(transformed.values()))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    x_min, x_max = float(np.min(x)), float(np.max(x))
    colors = ["#1261a0", "#d1495b", "#2a9d8f", "#6a4c93"]
    for color, (label, values) in zip(colors, transformed.items(), strict=False):
        points = []
        for x_value, y_value in zip(x, values, strict=True):
            px = left + (float(x_value) - x_min) / max(x_max - x_min, 1.0) * (
                right - left
            )
            py = bottom - (float(y_value) - y_min) / (y_max - y_min) * (bottom - top)
            points.append((px, py))
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse(
                (point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5),
                fill=color,
            )
        legend_y = top + 8 + 25 * list(transformed).index(label)
        draw.line((right - 230, legend_y, right - 190, legend_y), fill=color, width=4)
        draw.text((right - 180, legend_y - 10), label, fill="black", font=small)
    if len(x) <= 8:
        tick_values = np.asarray(x)
    else:
        tick_indices = np.linspace(0, len(x) - 1, 6).astype(int)
        tick_values = np.asarray(x)[tick_indices]
    for x_value in tick_values:
        px = left + (float(x_value) - x_min) / max(x_max - x_min, 1.0) * (right - left)
        draw.line((px, bottom, px, bottom + 7), fill="black", width=2)
        draw.text((px - 18, bottom + 12), f"{x_value:g}", fill="black", font=small)
    y_min_label = f"10^{y_min:.2f}" if log_y else f"{y_min:.3g}"
    y_max_label = f"10^{y_max:.2f}" if log_y else f"{y_max:.3g}"
    draw.text((10, bottom - 10), y_min_label, fill="black", font=small)
    draw.text((10, top - 10), y_max_label, fill="black", font=small)
    draw.text((left, 20), title, fill="black", font=font)
    draw.text((420, 650), x_label, fill="black", font=small)
    draw.text((10, 45), y_label, fill="black", font=small)
    image.save(path)


def _plot_outputs_pillow(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    cells: PhysicalPhaseCellMap,
    master_shape: tuple[int, int],
    coverage_map: np.ndarray,
    representative_patches: list[np.ndarray],
) -> list[str]:
    figures = run_dir / "figures"
    title_font = _pillow_font(18)
    small_font = _pillow_font(14)
    names: list[str] = []

    canvas = Image.new("RGB", (1500, 360), "white")
    panels = [cells.phase_rad] + [np.angle(item) for item in representative_patches]
    labels = ["master cells", "corner 1", "corner 2", "corner 3", "corner 4"]
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        canvas.paste(_phase_image(panel, (280, 280)), (10 + 298 * index, 50))
        ImageDraw.Draw(canvas).text(
            (20 + 298 * index, 20), label, fill="black", font=small_font
        )
    ImageDraw.Draw(canvas).text(
        (520, 335), "finite aperiodic B phase [rad]", fill="black", font=small_font
    )
    name = "B_master_and_corner_patches.png"
    canvas.save(figures / name)
    names.append(name)

    image = Image.new("RGB", (850, 760), "white")
    image.paste(_heat_image(coverage_map, (680, 680)), (120, 40))
    draw = ImageDraw.Draw(image)
    extent_um = 0.5 * master_shape[0] * float(config["optics"]["dx_m"]) * 1e6
    draw.text((150, 10), "B scan coverage map [visits]", fill="black", font=title_font)
    draw.text((330, 730), "x [µm]", fill="black", font=small_font)
    draw.text((10, 370), "y [µm]", fill="black", font=small_font)
    draw.text((95, 720), f"{-extent_um:.0f}", fill="black", font=small_font)
    draw.text((765, 720), f"{extent_um:.0f}", fill="black", font=small_font)
    name = "B_scan_coverage_map.png"
    image.save(figures / name)
    names.append(name)

    spots = np.asarray([50.0, 100.0, 200.0, 400.0])
    formal_ids = [_case_id(value * 1.0e-6, "formal") for value in spots]
    radius = np.linspace(0.0, 35.0, 200)
    profiles = {
        f"{spot:g} µm": np.exp(-2.0 * radius**2 / (0.5 * spot) ** 2) for spot in spots
    }
    _pillow_line_chart(
        figures / "illumination_profiles_with_TGV_overlay.png",
        title="A-plane Gaussian profiles; diameter = 1/e² intensity",
        x_label="A-plane radius [µm]",
        y_label="relative intensity; TGV top radius = 25 µm",
        x=radius,
        series=profiles,
    )
    names.append("illumination_profiles_with_TGV_overlay.png")

    _pillow_line_chart(
        figures / "spot_power_and_uniformity.png",
        title="Spot power containment and TGV illumination uniformity",
        x_label="spot diameter 1/e² intensity [µm]",
        y_label="fraction (log10 axis)",
        x=spots,
        series={
            "top-radius nonuniformity": np.asarray(
                [
                    metrics["illumination"][case][
                        "illumination_nonuniformity_over_top_radius"
                    ]
                    for case in formal_ids
                ]
            ),
            "omitted active power": np.asarray(
                [
                    metrics["illumination"][case][
                        "active_window_omitted_power_fraction"
                    ]
                    for case in formal_ids
                ]
            ),
        },
        log_y=True,
    )
    names.append("spot_power_and_uniformity.png")

    for filename, section, key, ylabel, log_y in (
        (
            "probe_sensitivity_vs_spot_size.png",
            "probe_sensitivity",
            "normalized_gauge_removed_sensitivity",
            "normalized probe sensitivity",
            True,
        ),
        (
            "detector_sensitivity_vs_spot_size.png",
            "detector_sensitivity",
            "normalized_sensitivity",
            "normalized detector sensitivity",
            True,
        ),
        (
            "poisson_information_vs_spot_size.png",
            "poisson_fisher",
            "fisher_information_per_incident_photon_per_m2",
            "FI per incident photon [m^-2]",
            True,
        ),
        (
            "detector_dynamic_range_vs_spot_size.png",
            "detector_diagnostics",
            "maximum_to_median_ratio",
            "maximum / median",
            True,
        ),
    ):
        series = {
            "fixed center": np.asarray(
                [
                    metrics["branches"]["fixed_center_intensity"][case][section][key]
                    for case in formal_ids
                ]
            ),
            "fixed total power": np.asarray(
                [
                    metrics["branches"]["fixed_total_power"][case][section][key]
                    for case in formal_ids
                ]
            ),
        }
        _pillow_line_chart(
            figures / filename,
            title=ylabel,
            x_label="spot diameter 1/e² intensity [µm]",
            y_label=ylabel + (" (log10 axis)" if log_y else ""),
            x=spots,
            series=series,
            log_y=log_y,
        )
        names.append(filename)

    convergence = metrics["numerical_convergence"]
    convergence_values = np.asarray(
        [
            convergence["fov_controls"][_case_id(50.0e-6, "formal")][
                "maximum_relative_change"
            ],
            convergence["fov_controls"][_case_id(400.0e-6, "formal")][
                "maximum_relative_change"
            ],
            convergence["padding_control"]["maximum_relative_change"],
        ]
    )
    _pillow_line_chart(
        figures / "fov_convergence.png",
        title="FOV and open-padding convergence; gate = 5%",
        x_label="control index: 1=50 support, 2=400 FOV, 3=100 padding",
        y_label="maximum registered relative change",
        x=np.asarray([1.0, 2.0, 3.0]),
        series={"relative change": convergence_values},
    )
    names.append("fov_convergence.png")
    return names


def _plot_existing_run(run_dir: Path) -> list[str]:
    """Generate figures in a fresh process from sealed numerical artifacts."""

    resolved = run_dir.resolve()
    config = load_config(resolved / "config.yaml")
    with (resolved / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    hdf5_path = resolved / "outputs" / config["output"]["hdf5_filename"]
    with h5py.File(hdf5_path, "r") as h5:
        phase_cells = np.asarray(
            h5["/entry/truth/sample_B_phase_cells"][...], dtype=np.float64
        )
        coverage_dataset = h5["/entry/metrics/B_coverage/coverage_map"]
        coverage_stride = max(1, int(math.ceil(max(coverage_dataset.shape) / 768)))
        coverage = coverage_dataset[::coverage_stride, ::coverage_stride]
        positions = np.asarray(h5["/entry/data/scan_positions"][...])
    geometry = metrics["B_coverage"]["geometry"]
    master_shape = tuple(int(value) for value in geometry["master_B_shape"])
    feature_m = float(config["sample_b"]["feature_size_m"])
    cells = PhysicalPhaseCellMap(
        phase_rad=phase_cells,
        feature_size_m=feature_m,
        origin_xy_m=(
            -0.5 * phase_cells.shape[1] * feature_m,
            -0.5 * phase_cells.shape[0] * feature_m,
        ),
        phase_range_rad=float(config["sample_b"]["phase_range_rad"]),
        seed=int(config["sample_b"]["seed"]),
    )
    formal_100_id = _case_id(100.0e-6, "formal")
    formal_100 = next(
        case for case in geometry["case_descriptors"] if case["id"] == formal_100_id
    )
    corner_indices = [
        int(np.argmin(np.sum(positions, axis=1))),
        int(np.argmax(positions[:, 0] - positions[:, 1])),
        int(np.argmax(positions[:, 1] - positions[:, 0])),
        int(np.argmax(np.sum(positions, axis=1))),
    ]
    representative_patches = []
    for index in corner_indices:
        center_xy_m = (-float(positions[index, 0]), -float(positions[index, 1]))
        phase = rasterize_physical_phase_values(
            cells,
            tuple(formal_100["active_shape"]),
            float(config["optics"]["dx_m"]),
            center_xy_m=center_xy_m,
        )
        representative_patches.append(_phase_to_transmission(phase)[::8, ::8])
    names = _plot_outputs_pillow(
        resolved,
        config,
        metrics,
        cells,
        master_shape,
        coverage,
        representative_patches,
    )
    if metrics["reconstruction"]["executed"]:
        case_ids = list(metrics["reconstruction"]["cases"])
        errors = np.asarray(
            [
                metrics["reconstruction"]["cases"][case_id]["baseline"][
                    "simulation_evaluation_only"
                ]["complex_scalar_aligned_relative_l2_simulation_evaluation_only"]
                for case_id in case_ids
            ]
        )
        name = "known_B_recovery_vs_spot_size.png"
        _pillow_line_chart(
            resolved / "figures" / name,
            title="Known-B probe recovery; fixed one-epoch schedule",
            x_label="representative case index",
            y_label="aligned relative L2 (simulation evaluation only)",
            x=np.arange(1, len(case_ids) + 1, dtype=np.float64),
            series={"probe error": errors},
            log_y=True,
        )
        names.append(name)
    return names


def _audit_hdf5(
    path: Path,
    config_yaml: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    expected_figures: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    required = [
        "/entry/config_yaml",
        "/entry/data/scan_positions",
        "/entry/data/I_stack",
        "/entry/instrument/illumination",
        "/entry/sample/sample_B_parameters",
        "/entry/truth/sample_B_phase_cells",
        "/entry/truth/illumination_cases/spot_diameter_1e2_intensity_m",
        "/entry/truth/P_B_true",
        "/entry/metrics/B_coverage",
        "/entry/metrics/illumination",
        "/entry/metrics/probe_sensitivity",
        "/entry/metrics/detector_sensitivity",
        "/entry/metrics/poisson_fisher",
        "/entry/metrics/numerical_convergence",
    ]
    numeric_finite = True
    with h5py.File(path, "r") as h5:
        missing = [name for name in required if name not in h5]

        def visitor(_: str, node: h5py.Group | h5py.Dataset) -> None:
            nonlocal numeric_finite
            if isinstance(node, h5py.Dataset) and node.dtype.kind in "biufc":
                numeric_finite &= bool(np.all(np.isfinite(node[...])))

        h5.visititems(visitor)
        config_match = h5["/entry/config_yaml"].asstr()[()] == config_yaml
        metadata_match = json.loads(
            h5["/entry/metadata/json"].asstr()[()]
        ) == json.loads(_json_text(metadata))
        metrics_match = json.loads(
            h5["/entry/metrics/json"].asstr()[()]
        ) == json.loads(_json_text(metrics))
        intensity_nonnegative = bool(np.all(h5["/entry/data/I_stack"][...] >= 0))
        reconstruction_present = "/entry/reconstruction" in h5
    missing_figures = [
        name for name in expected_figures if not (run_dir / "figures" / name).is_file()
    ]
    return {
        "required_paths_missing": missing,
        "config_yaml_matches_external": config_match,
        "metadata_json_matches_external": metadata_match,
        "metrics_json_matches_external": metrics_match,
        "all_numeric_hdf5_finite": numeric_finite,
        "I_stack_nonnegative": intensity_nonnegative,
        "reconstruction_group_present": reconstruction_present,
        "missing_figures": missing_figures,
        "passed": bool(
            not missing
            and config_match
            and metadata_match
            and metrics_match
            and numeric_finite
            and intensity_nonnegative
            and not missing_figures
        ),
    }


def _run_preflight(
    config: dict[str, Any], config_path: Path, run_dir: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    positions = _make_scan(config)
    geometry = _geometry(config, positions)
    preflight = _preflight_metrics(config, positions, geometry)
    metrics = {
        "experiment_id": "exp031",
        "run_mode": "preflight_only",
        "status": preflight["status"],
        "B_coverage": geometry,
        "numerical_convergence": {"preflight": preflight},
    }
    save_json(run_dir / "metrics.json", metrics)
    _save_preflight_hdf5(
        run_dir / "outputs" / config["output"]["hdf5_filename"],
        config_to_yaml(config),
        metadata,
        positions,
        geometry,
        preflight,
    )
    save_json(
        run_dir / "run_state.json",
        {
            "status": "completed" if preflight["status"] == "passed" else "failed",
            "mode": "preflight",
            "updated_at_utc": created_at_utc(),
        },
    )
    return metrics


def _run_formal(
    config: dict[str, Any], run_dir: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    positions = _make_scan(config)
    geometry = _geometry(config, positions)
    preflight = _preflight_metrics(config, positions, geometry)
    if preflight["status"] != "passed":
        raise RuntimeError("formal run blocked by preregistered preflight limits.")
    cases = geometry["case_descriptors"]
    master_shape = tuple(int(value) for value in geometry["master_B_shape"])
    feature_m = float(config["sample_b"]["feature_size_m"])
    pixels_per_cell = int(geometry["pixels_per_feature_cell"])
    cell_shape = (
        master_shape[0] // pixels_per_cell,
        master_shape[1] // pixels_per_cell,
    )
    cells = make_physical_phase_cell_map(
        cell_shape,
        feature_size_m=feature_m,
        phase_range_rad=float(config["sample_b"]["phase_range_rad"]),
        seed=int(config["sample_b"]["seed"]),
    )
    repeated_cells = make_physical_phase_cell_map(
        cell_shape,
        feature_size_m=feature_m,
        phase_range_rad=float(config["sample_b"]["phase_range_rad"]),
        seed=int(config["sample_b"]["seed"]),
    )
    cell_deterministic = bool(np.array_equal(cells.phase_rad, repeated_cells.phase_rad))
    phase_canvas = rasterize_physical_phase_values(
        cells,
        master_shape,
        float(config["optics"]["dx_m"]),
    ).astype(np.float32)
    radial = _radial_cache(config, cases)
    adjoint_plan = make_scan_window_plan(
        tuple(geometry["legacy_recommended_B_shape"]),
        tuple(geometry["legacy_active_shape"]),
        positions,
        float(config["optics"]["dx_m"]),
    )
    adjoint_error = scan_window_adjoint_relative_error(adjoint_plan, seed=31)
    del adjoint_plan
    gc.collect()
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    config_yaml = config_to_yaml(config)
    metrics: dict[str, Any] = {
        "experiment_id": "exp031",
        "run_mode": "formal",
        "status": "running",
        "B_coverage": {"geometry": geometry, "cases": {}},
        "illumination": {},
        "probe_sensitivity": {},
        "detector_sensitivity": {},
        "poisson_fisher": {},
        "detector_diagnostics": {},
        "branches": {"fixed_center_intensity": {}, "fixed_total_power": {}},
        "periodic_B_negative_control": {},
        "numerical_convergence": {"preflight": preflight},
        "reconstruction": {"executed": False},
        "scientific_scope": {
            "model": "2D projected phase, axisymmetric, centered, zero tilt",
            "poisson_bound": "shot-noise-only optimistic local bound",
            "excluded": [
                "read noise",
                "dark current",
                "gain",
                "full well",
                "saturation",
                "stage error",
                "model mismatch",
                "3D TGV inference",
            ],
        },
    }
    formal_results: dict[str, dict[str, Any]] = {}
    control_results: dict[str, dict[str, Any]] = {}
    coverage_data: np.ndarray | None = None
    formal_common_probe_shape = tuple(
        int(value) for value in geometry["probe_envelope_comparison_shape"]
    )
    with h5py.File(hdf5_path, "x") as h5:
        entry = h5.require_group("entry")
        _write_h5(entry, "config_yaml", config_yaml)
        data = entry.require_group("data")
        data.create_dataset("scan_positions", data=positions)
        instrument = entry.require_group("instrument")
        _write_h5(
            instrument,
            "illumination",
            {
                **config["illumination"],
                "definition": metadata["spot_definition"],
            },
        )
        sample = entry.require_group("sample")
        _write_h5(
            sample,
            "sample_B_parameters",
            {
                **config["sample_b"],
                "physical_origin_xy_m": cells.origin_xy_m,
                "physical_extent_xy_m": cells.physical_extent_xy_m,
                "raster_shape": master_shape,
                "raster_dx_yx_m": [
                    config["optics"]["dx_m"],
                    config["optics"]["dx_m"],
                ],
                "periodic_tiling": False,
            },
        )
        truth = entry.require_group("truth")
        truth.create_dataset("sample_B_phase_cells", data=cells.phase_rad)
        illumination_truth = truth.require_group("illumination_cases")
        illumination_truth.create_dataset(
            "spot_diameter_1e2_intensity_m",
            data=np.asarray(config["illumination"]["spot_diameter_1e2_intensity_m"]),
        )
        _write_h5(entry, "metadata", {**metadata, "json": _json_text(metadata)})

        print("exp031: starting formal forward matrix", flush=True)
        for case in cases:
            key = (
                None
                if case["kind"] == "plane_wave"
                else 0.5 * float(case["spot_diameter_1e2_intensity_m"])
            )
            print(
                f"exp031: case {case['id']} active={case['active_shape']} "
                f"open={case['open_shape']}",
                flush=True,
            )
            result = _simulate_case(
                config,
                case,
                positions,
                phase_canvas,
                master_shape,
                radial[key],
            )
            metrics["B_coverage"]["cases"][case["id"]] = result["coverage"]
            metrics["illumination"][case["id"]] = result["illumination"]
            for branch, values in result["branches"].items():
                metrics["branches"][branch][case["id"]] = values
                if case["role"] in {"formal", "formal_reference"}:
                    formal_results[case["id"]] = {
                        "case": case,
                        "slice_plan": replace(
                            result["plan"],
                            coverage_map=np.empty((0, 0), dtype=np.int32),
                        ),
                        "common_probe": _common_center_crop(
                            result["fields"]["probe_baseline"],
                            formal_common_probe_shape,
                        ).copy(),
                    }
            metrics["numerical_convergence"][case["id"]] = result["numerical"]
            if case["role"] in {"formal", "formal_reference"}:
                representative = case["id"] in {
                    _case_id(100.0e-6, "formal"),
                    _case_id(400.0e-6, "formal"),
                    "plane_wave_legacy_roi",
                }
                _save_formal_case_arrays(entry, result, representative=representative)
                if case["id"] == _case_id(100.0e-6, "formal"):
                    data["I_stack"] = data["illumination_cases"][case["id"]][
                        "I_stack_fixed_center"
                    ]
                    truth.create_dataset(
                        "P_B_true", data=result["fields"]["probe_baseline"]
                    )
                if case["id"] == _case_id(400.0e-6, "formal"):
                    coverage_data = result["plan"].coverage_map.astype(np.int16)
            else:
                control_results[case["id"]] = {
                    "case": case,
                    "branches": result["branches"],
                    "numerical": result["numerical"],
                }
            del result
            gc.collect()

        plane_probe = formal_results["plane_wave_legacy_roi"]["common_probe"]
        metrics["illumination"]["probe_envelope_comparison_shape"] = list(
            formal_common_probe_shape
        )
        metrics["illumination"]["probe_envelope_comparison_size_yx_m"] = (
            np.asarray(formal_common_probe_shape, dtype=np.float64)
            * float(config["optics"]["dx_m"])
        ).tolist()
        metrics["illumination"]["probe_envelope_comparison_domain"] = (
            "fixed_config_centered_B_plane"
        )
        for spot_um in (50, 100, 200, 400):
            case_id = _case_id(spot_um * 1.0e-6, "formal")
            metrics["illumination"][case_id][
                "Gaussian_relative_to_plane_wave_probe_envelope_l2"
            ] = _normalized_envelope_difference(
                formal_results[case_id]["common_probe"], plane_probe
            )

        formal_100_id = _case_id(100.0e-6, "formal")
        formal_100_case = next(case for case in cases if case["id"] == formal_100_id)
        periodic_result = _simulate_case(
            config,
            {**formal_100_case, "id": "gaussian_100um_periodic_B_control"},
            positions,
            phase_canvas,
            master_shape,
            radial[50.0e-6],
            periodic=True,
        )
        aperiodic_stack = data["illumination_cases"][formal_100_id][
            "I_stack_fixed_center"
        ][...]
        periodic_stack = periodic_result["stacks"]["baseline"]
        metrics["periodic_B_negative_control"] = {
            "case": "gaussian_100um_fixed_center",
            "same_center_realization": True,
            "legacy_periodic_to_aperiodic_I_stack_relative_l2": float(
                _norm(periodic_stack - aperiodic_stack)
                / max(_norm(aperiodic_stack), np.finfo(float).tiny)
            ),
            "mean_frame_relative_l2": float(
                np.mean(
                    _norm(periodic_stack - aperiodic_stack, axis=(1, 2))
                    / np.maximum(
                        _norm(aperiodic_stack, axis=(1, 2)),
                        np.finfo(float).tiny,
                    )
                )
            ),
            "periodic_wrap_count": len(positions),
            "formal_model_periodic_wrap_count": 0,
        }
        del periodic_result, periodic_stack, aperiodic_stack
        gc.collect()
        print("exp031: periodic-B negative control completed", flush=True)

        fov_controls: dict[str, Any] = {}
        for spot_um in (50, 400):
            formal_id = _case_id(spot_um * 1.0e-6, "formal")
            strict_id = _case_id(spot_um * 1.0e-6, "strict_fov")
            changes = {}
            for branch in ("fixed_center_intensity", "fixed_total_power"):
                formal_branch = metrics["branches"][branch][formal_id]
                strict_branch = control_results[strict_id]["branches"][branch]
                changes[branch] = {
                    "normalized_detector_sensitivity_relative_change": relative_change(
                        strict_branch["detector_sensitivity"]["normalized_sensitivity"],
                        formal_branch["detector_sensitivity"]["normalized_sensitivity"],
                    ),
                    "poisson_FI_relative_change": relative_change(
                        strict_branch["poisson_fisher"][
                            "fisher_information_per_incident_photon_per_m2"
                        ],
                        formal_branch["poisson_fisher"][
                            "fisher_information_per_incident_photon_per_m2"
                        ],
                    ),
                }
            maximum_change = max(
                value for branch in changes.values() for value in branch.values()
            )
            fov_controls[formal_id] = {
                "strict_case": strict_id,
                "changes": changes,
                "maximum_relative_change": maximum_change,
            }
        formal_padding = float(config["open_boundary"]["padding_guard_m_per_side"])
        padding_sequence = [
            float(value)
            for value in config["open_boundary"]["padding_sequence_guard_m_per_side"]
        ]
        padding_level_results: dict[float, dict[str, Any]] = {}
        for guard_m in padding_sequence:
            if guard_m == formal_padding:
                padding_level_results[guard_m] = {
                    "case": formal_100_id,
                    "branches": {
                        branch: metrics["branches"][branch][formal_100_id]
                        for branch in ("fixed_center_intensity", "fixed_total_power")
                    },
                }
            else:
                guard_um = int(round(guard_m * 1.0e6))
                case_id = _case_id(
                    100.0e-6, f"padding_{guard_um:03d}um_control"
                )
                padding_level_results[guard_m] = {
                    "case": case_id,
                    "branches": control_results[case_id]["branches"],
                }
        padding_levels = []
        for guard_m in padding_sequence:
            level = padding_level_results[guard_m]
            padding_levels.append(
                {
                    "guard_m_per_side": guard_m,
                    "case": level["case"],
                    "open_shape": next(
                        case["open_shape"]
                        for case in cases
                        if case["id"] == level["case"]
                    ),
                    "branches": {
                        branch: {
                            "normalized_detector_sensitivity": level["branches"][
                                branch
                            ]["detector_sensitivity"]["normalized_sensitivity"],
                            "fisher_information_per_incident_photon_per_m2": level[
                                "branches"
                            ][branch]["poisson_fisher"][
                                "fisher_information_per_incident_photon_per_m2"
                            ],
                        }
                        for branch in ("fixed_center_intensity", "fixed_total_power")
                    },
                }
            )
        padding_adjacent_changes: dict[str, Any] = {}
        for smaller_guard, larger_guard in zip(
            padding_sequence[:-1], padding_sequence[1:], strict=True
        ):
            smaller = padding_level_results[smaller_guard]
            larger = padding_level_results[larger_guard]
            label = (
                f"{int(round(smaller_guard * 1.0e6)):03d}um_to_"
                f"{int(round(larger_guard * 1.0e6)):03d}um"
            )
            padding_adjacent_changes[label] = {}
            for branch in ("fixed_center_intensity", "fixed_total_power"):
                smaller_branch = smaller["branches"][branch]
                larger_branch = larger["branches"][branch]
                padding_adjacent_changes[label][branch] = {
                    "normalized_detector_sensitivity_relative_change": relative_change(
                        larger_branch["detector_sensitivity"]["normalized_sensitivity"],
                        smaller_branch["detector_sensitivity"]["normalized_sensitivity"],
                    ),
                    "poisson_FI_relative_change": relative_change(
                        larger_branch["poisson_fisher"][
                            "fisher_information_per_incident_photon_per_m2"
                        ],
                        smaller_branch["poisson_fisher"][
                            "fisher_information_per_incident_photon_per_m2"
                        ],
                    ),
                }
        control_padding = float(
            config["open_boundary"]["padding_control_guard_m_per_side"]
        )
        gate_label = (
            f"{int(round(formal_padding * 1.0e6)):03d}um_to_"
            f"{int(round(control_padding * 1.0e6)):03d}um"
        )
        padding_changes = padding_adjacent_changes[gate_label]
        metrics["numerical_convergence"]["fov_controls"] = fov_controls
        metrics["numerical_convergence"]["padding_control"] = {
            "formal_case": formal_100_id,
            "formal_guard_m_per_side": formal_padding,
            "control_case": padding_level_results[control_padding]["case"],
            "control_guard_m_per_side": control_padding,
            "sequence": padding_levels,
            "adjacent_changes": padding_adjacent_changes,
            "gate_comparison": gate_label,
            "changes": padding_changes,
            "maximum_relative_change": max(
                value
                for branch in padding_changes.values()
                for value in branch.values()
            ),
        }
        _add_relative_ratios(metrics)
        print("exp031: convergence ratios completed", flush=True)
        for branch in metrics["branches"]:
            for case_id, values in metrics["branches"][branch].items():
                metrics["probe_sensitivity"].setdefault(branch, {})[case_id] = values[
                    "probe_sensitivity"
                ]
                metrics["detector_sensitivity"].setdefault(branch, {})[case_id] = (
                    values["detector_sensitivity"]
                )
                metrics["poisson_fisher"].setdefault(branch, {})[case_id] = values[
                    "poisson_fisher"
                ]
                metrics["detector_diagnostics"].setdefault(branch, {})[case_id] = (
                    values["detector_diagnostics"]
                )

        formal_case_ids = [
            _case_id(value * 1.0e-6, "formal") for value in (50, 100, 200, 400)
        ]
        max_edge = max(
            metrics["numerical_convergence"][case_id][
                "residual_edge_energy_fraction_maximum"
            ]
            for case_id in formal_case_ids
        )
        max_fov = max(item["maximum_relative_change"] for item in fov_controls.values())
        residual_support_change = fov_controls[
            _case_id(
                float(config["active_window"]["residual_support_spot_diameter_m"]),
                "formal",
            )
        ]["maximum_relative_change"]
        padding_change = metrics["numerical_convergence"]["padding_control"][
            "maximum_relative_change"
        ]
        all_inside = all(
            item["all_patches_fully_inside"]
            for item in metrics["B_coverage"]["cases"].values()
        )
        minimum_margin = min(
            item["minimum_coverage_margin_m"]
            for item in metrics["B_coverage"]["cases"].values()
        )
        repeat_max = max(
            metrics["numerical_convergence"][case_id][
                "same_config_repeat_max_abs_field_difference"
            ]
            for case_id in formal_case_ids
        )
        gates = {
            "all_B_patches_fully_inside": all_inside,
            "minimum_coverage_margin_strictly_positive": minimum_margin > 0,
            "formal_periodic_wrap_count_zero": all(
                metrics["B_coverage"]["cases"][case_id]["periodic_wrap_count"] == 0
                for case_id in formal_case_ids
            ),
            "scan_window_adjoint_relative_error": adjoint_error,
            "scan_window_adjoint_test_plan": (
                "actual 49 positions on 96 um active / 152 um B sanity geometry"
            ),
            "scan_window_adjoint_passed": adjoint_error
            <= float(config["gates"]["adjoint_relative_error_max"]),
            "same_seed_config_repeat_max_abs_difference": repeat_max,
            "same_seed_config_repeat_passed": repeat_max == 0.0 and cell_deterministic,
            "formal_strict_FOV_max_relative_change": max_fov,
            "formal_strict_FOV_passed": max_fov
            < float(config["gates"]["numerical_relative_change_max"]),
            "residual_support_max_relative_change": residual_support_change,
            "residual_support_convergence_passed": residual_support_change
            < float(config["gates"]["numerical_relative_change_max"]),
            "residual_edge_energy_fraction_maximum": max_edge,
            "residual_edge_energy_passed": max_edge
            <= float(config["gates"]["residual_edge_energy_fraction_max"]),
            "padding_max_relative_change": padding_change,
            "padding_passed": padding_change
            < float(config["gates"]["numerical_relative_change_max"]),
            "finite_metrics": finite_metrics(metrics),
            "intensity_nonnegative": all(
                metrics["numerical_convergence"][case_id]["intensity_nonnegative"]
                for case_id in formal_case_ids
            ),
            "B_phase_cell_determinism": cell_deterministic,
        }
        primary_pass = bool(
            all_inside
            and minimum_margin > 0
            and gates["formal_periodic_wrap_count_zero"]
            and gates["scan_window_adjoint_passed"]
            and gates["same_seed_config_repeat_passed"]
            and gates["formal_strict_FOV_passed"]
            and gates["residual_support_convergence_passed"]
            and gates["residual_edge_energy_passed"]
            and gates["padding_passed"]
            and gates["finite_metrics"]
            and gates["intensity_nonnegative"]
        )
        gates["primary_numerical_gates_passed_before_artifact_audit"] = primary_pass
        metrics["numerical_convergence"]["hard_gates"] = gates
        print(f"exp031: primary numerical gates passed={primary_pass}", flush=True)

        if primary_pass and bool(
            config["reconstruction"]["enabled_if_primary_gates_pass"]
        ):
            print(
                "exp031: primary gates passed; running secondary reconstruction",
                flush=True,
            )
            reconstruction_group = entry.require_group("reconstruction")
            reconstruction_metrics: dict[str, Any] = {
                "executed": True,
                "truth_used_by_optimizer": False,
                "selection": config["reconstruction"]["stopping_and_selection"],
                "cases": {},
            }
            representative_map = {
                "gaussian_100um": formal_100_id,
                "gaussian_400um": _case_id(400.0e-6, "formal"),
                "plane_wave": "plane_wave_legacy_roi",
            }
            for label in config["reconstruction"]["representative_cases"]:
                case_id = representative_map[str(label)]
                case = next(item for item in cases if item["id"] == case_id)
                key = (
                    None
                    if case["kind"] == "plane_wave"
                    else 0.5 * case["spot_diameter_1e2_intensity_m"]
                )
                fields = _case_fields(config, case, radial[key])
                plan = formal_results[case_id]["slice_plan"]
                get_patch = _patch_getter(phase_canvas, plan)
                transfer = make_asm_transfer_complex64(
                    tuple(case["open_shape"]),
                    dx_m=float(config["optics"]["dx_m"]),
                    wavelength_m=float(config["optics"]["wavelength_m"]),
                    distance_m=float(config["optics"]["z_BC_m"]),
                    refractive_index=float(config["optics"]["medium_index"]),
                    bandlimit=bool(config["optics"]["bandlimit"]),
                    alias_control=bool(config["optics"]["alias_control"]),
                )
                data_group = data["illumination_cases"][case_id]
                baseline_stack = data_group["I_stack_fixed_center"][...]
                init = measurement_scaled_reference_initialization(
                    fields["reference_b"],
                    fields["reference_detector"],
                    baseline_stack,
                )
                rec_fields: dict[str, np.ndarray] = {}
                rec_metrics: dict[str, Any] = {}
                case_h5 = reconstruction_group.require_group(case_id)
                for parameter_case, dataset_name, truth_name in (
                    ("minus", "I_stack_minus_fixed_center", "probe_minus"),
                    ("baseline", "I_stack_fixed_center", "probe_baseline"),
                    ("plus", "I_stack_plus_fixed_center", "probe_plus"),
                ):
                    reconstruction = reconstruct_known_b_probe_open(
                        data_group[dataset_name][...],
                        get_patch,
                        fields["reference_b"],
                        fields["reference_detector"],
                        transfer,
                        init,
                        num_iterations=int(config["reconstruction"]["num_iterations"]),
                        beta_probe=float(config["reconstruction"]["beta_probe"]),
                        workers=int(config["open_boundary"]["fft_workers"]),
                    )
                    recovered = reconstruction["P_B_rec_raw"]
                    rec_fields[parameter_case] = recovered
                    evaluation = simulation_evaluation_probe_error(
                        recovered, fields[truth_name]
                    )
                    rec_metrics[parameter_case] = {
                        "loss_curve_measurement_only": reconstruction[
                            "loss_curve"
                        ].tolist(),
                        "simulation_evaluation_only": evaluation,
                    }
                    sub = case_h5.require_group(parameter_case)
                    sub.create_dataset("P_B_rec_raw", data=recovered)
                    _write_h5(sub, "metadata", reconstruction["metadata"])
                    _write_h5(sub, "simulation_evaluation_only", evaluation)
                recovered_derivative = (rec_fields["plus"] - rec_fields["minus"]) / (
                    2.0 * float(config["sensitivity"]["delta_d_waist_m"])
                )
                recovered_sensitivity = probe_sensitivity_metrics(
                    rec_fields["baseline"],
                    recovered_derivative,
                    float(config["tgv"]["d_waist_m"]),
                    dx_m=float(config["optics"]["dx_m"]),
                    incident_power=metrics["branches"]["fixed_center_intensity"][
                        case_id
                    ]["incident_power_per_frame"],
                )
                reconstruction_metrics["cases"][case_id] = {
                    **rec_metrics,
                    "recovered_probe_sensitivity": recovered_sensitivity,
                }
            truth_order = sorted(
                reconstruction_metrics["cases"],
                key=lambda case_id: metrics["probe_sensitivity"][
                    "fixed_center_intensity"
                ][case_id]["normalized_gauge_removed_sensitivity"],
            )
            recovered_order = sorted(
                reconstruction_metrics["cases"],
                key=lambda case_id: reconstruction_metrics["cases"][case_id][
                    "recovered_probe_sensitivity"
                ]["normalized_gauge_removed_sensitivity"],
            )
            reconstruction_metrics["truth_sensitivity_ascending_order"] = truth_order
            reconstruction_metrics["recovered_sensitivity_ascending_order"] = (
                recovered_order
            )
            reconstruction_metrics["spot_size_ordering_preserved"] = (
                truth_order == recovered_order
            )
            metrics["reconstruction"] = reconstruction_metrics

        if coverage_data is None:
            raise RuntimeError("formal 400 um coverage map was not retained.")
        metrics["B_coverage"]["coverage_map_case"] = _case_id(400.0e-6, "formal")
        metrics["B_coverage"]["coverage_map_shape"] = list(coverage_data.shape)
        metrics["B_coverage"]["B_phase_cell_determinism"] = cell_deterministic
        metrics["B_coverage"]["periodic_wrap_count"] = 0

        del phase_canvas, radial, formal_results, control_results, cells
        gc.collect()
        print("exp031: released forward-only working state", flush=True)
        expected_figures = [
            "B_master_and_corner_patches.png",
            "B_scan_coverage_map.png",
            "illumination_profiles_with_TGV_overlay.png",
            "spot_power_and_uniformity.png",
            "probe_sensitivity_vs_spot_size.png",
            "detector_sensitivity_vs_spot_size.png",
            "poisson_information_vs_spot_size.png",
            "detector_dynamic_range_vs_spot_size.png",
            "fov_convergence.png",
        ]
        if metrics["reconstruction"]["executed"]:
            expected_figures.append("known_B_recovery_vs_spot_size.png")

        metrics["figure_files"] = expected_figures
        metrics["status"] = (
            "passed_2D_preregistered_numerical_problem"
            if primary_pass
            else "failed_primary_numerical_gates"
        )
        metrics_group = entry.require_group("metrics")
        _write_h5(metrics_group, "B_coverage", metrics["B_coverage"])
        metrics_group["B_coverage"].create_dataset("coverage_map", data=coverage_data)
        _write_h5(metrics_group, "illumination", metrics["illumination"])
        _write_h5(metrics_group, "probe_sensitivity", metrics["probe_sensitivity"])
        _write_h5(
            metrics_group,
            "detector_sensitivity",
            metrics["detector_sensitivity"],
        )
        _write_h5(metrics_group, "poisson_fisher", metrics["poisson_fisher"])
        _write_h5(
            metrics_group,
            "numerical_convergence",
            metrics["numerical_convergence"],
        )
        _write_h5(metrics_group, "json", _json_text(metrics))

    save_json(run_dir / "metrics.json", metrics)
    plot_process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plot-existing-run",
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if plot_process.returncode != 0:
        metrics["plot_subprocess_failure"] = {
            "returncode": plot_process.returncode,
            "stdout": plot_process.stdout,
            "stderr": plot_process.stderr,
        }
        save_json(run_dir / "metrics.json", metrics)
        with h5py.File(hdf5_path, "a") as h5:
            del h5["/entry/metrics/json"]
            _write_h5(h5["/entry/metrics"], "json", _json_text(metrics))
    else:
        print("exp031: isolated figure subprocess completed", flush=True)
    audit = _audit_hdf5(
        hdf5_path,
        config_yaml,
        metadata,
        metrics,
        metrics["figure_files"],
        run_dir,
    )
    metrics["artifact_audit"] = audit
    metrics["numerical_convergence"]["hard_gates"]["artifact_semantics_passed"] = audit[
        "passed"
    ]
    final_pass = bool(
        metrics["numerical_convergence"]["hard_gates"][
            "primary_numerical_gates_passed_before_artifact_audit"
        ]
        and audit["passed"]
    )
    metrics["status"] = (
        "passed_2D_preregistered_numerical_problem"
        if final_pass
        else "failed_or_inconclusive"
    )
    save_json(run_dir / "metrics.json", metrics)
    with h5py.File(hdf5_path, "a") as h5:
        del h5["/entry/metrics"]
        metrics_group = h5["/entry"].require_group("metrics")
        for section in (
            "B_coverage",
            "illumination",
            "probe_sensitivity",
            "detector_sensitivity",
            "poisson_fisher",
            "numerical_convergence",
        ):
            _write_h5(metrics_group, section, metrics[section])
        metrics_group["B_coverage"].create_dataset("coverage_map", data=coverage_data)
        _write_h5(metrics_group, "json", _json_text(metrics))
    final_audit = _audit_hdf5(
        hdf5_path,
        config_yaml,
        metadata,
        metrics,
        metrics["figure_files"],
        run_dir,
    )
    metrics["artifact_audit"] = final_audit
    save_json(run_dir / "metrics.json", metrics)
    with h5py.File(hdf5_path, "a") as h5:
        del h5["/entry/metrics/json"]
        _write_h5(h5["/entry/metrics"], "json", _json_text(metrics))
    return metrics


def main() -> int:
    args = _parse_args()
    if args.plot_existing_run is not None:
        try:
            names = _plot_existing_run(args.plot_existing_run)
            print(f"exp031: generated {len(names)} isolated figures", flush=True)
            return 0
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return 1
    if args.config is None:
        raise SystemExit("--config is required unless --plot-existing-run is used.")
    config_path = args.config.resolve()
    config = load_config(config_path)
    mode = "preflight" if args.preflight_only else "formal"
    run_dir, metadata = _initialize_run(config, config_path, mode)
    print(f"exp031 run directory: {run_dir}", flush=True)
    try:
        if args.preflight_only:
            metrics = _run_preflight(config, config_path, run_dir, metadata)
        else:
            metrics = _run_formal(config, run_dir, metadata)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "completed",
                "scientific_status": metrics["status"],
                "mode": mode,
                "updated_at_utc": created_at_utc(),
            },
        )
        print(f"exp031 status: {metrics['status']}", flush=True)
        return 0
    except Exception as exc:
        failure = {
            "experiment_id": "exp031",
            "status": "failed_with_exception",
            "mode": mode,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(run_dir / "metrics.json", failure)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "mode": mode,
                "updated_at_utc": created_at_utc(),
                "exception": str(exc),
            },
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
