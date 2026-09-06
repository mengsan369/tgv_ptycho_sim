"""Source contract and estimator for the exp050 projected-probe oracle fit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.exp030 import (
    build_exp030_radial_operator,
    make_exp030_projected_probe,
)
from tgv_ptycho.inverse.waist_fit import (
    extract_profile_minimum,
    finite_difference_controls,
    raw_complex_probe_loss,
    replay_error_metrics,
)
from tgv_ptycho.io.config import load_config


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array(values: NDArray[np.generic]) -> str:
    """Hash the C-contiguous bytes of one numeric array."""

    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest().upper()


def _decode_scalar(dataset: h5py.Dataset) -> Any:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if isinstance(value, np.generic) else value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        matches = bool(np.isclose(float(actual), expected, rtol=0.0, atol=0.0))
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"Source identity mismatch for {label}: {actual!r} != {expected!r}."
        )


def validate_exp050_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate the registered exp050 role, source, and fit design."""

    experiment = config.get("experiment", {})
    if experiment.get("id") != "exp050" or experiment.get("role") != (
        "2d_projected_true_probe_single_parameter_waist_fit"
    ):
        raise ValueError("Unexpected exp050 experiment identity or role.")
    if mode not in {"preflight", "formal"}:
        raise ValueError("mode must be 'preflight' or 'formal'.")
    source = config.get("source", {})
    target = str(source.get("target_dataset", ""))
    if target != "/entry/truth/P_B_true":
        raise ValueError("exp050 target must be exactly /entry/truth/P_B_true.")
    if "reconstruction" in target.lower() or target.lower().endswith("p_b_rec"):
        raise ValueError("P_B_rec and reconstruction datasets are forbidden inputs.")
    if source.get("experiment") != "exp030":
        raise ValueError("exp050 source experiment must be exp030.")
    design = config.get(mode, {})
    required = {
        "bounds_m",
        "global_step_m",
        "fine_half_width_m",
        "fine_step_m",
        "finite_difference_steps_m",
        "golden_iterations",
        "thresholds",
    }
    missing = sorted(required - set(design))
    if missing:
        raise ValueError(f"exp050 {mode} design is missing: {missing}.")
    lower, upper = (float(value) for value in design["bounds_m"])
    true_waist = float(config["sample_a"]["d_waist_true_m"])
    if not np.all(np.isfinite([lower, upper, true_waist])):
        raise ValueError("bounds and true waist must be finite.")
    if not 0.0 < lower < true_waist < upper:
        raise ValueError("bounds must strictly contain the positive true waist.")
    for key in ("global_step_m", "fine_half_width_m", "fine_step_m"):
        if not np.isfinite(float(design[key])) or float(design[key]) <= 0.0:
            raise ValueError(f"{key} must be finite and positive.")
    h1, h2 = (float(value) for value in design["finite_difference_steps_m"])
    if not lower < true_waist - h1 < true_waist + h1 < upper or not h1 > h2 > 0:
        raise ValueError("finite-difference steps must be ordered and inside bounds.")
    if int(design["golden_iterations"]) <= 0:
        raise ValueError("golden_iterations must be positive.")


def load_exp030_source(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    """Load and validate the authoritative exp030 raw true probe handoff."""

    source_cfg = config["source"]
    run_dir = (project_root / str(source_cfg["run"])).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"exp030 source run does not exist: {run_dir}.")
    paths = {
        "config": run_dir / "config.yaml",
        "metadata": run_dir / "metadata.json",
        "metrics": run_dir / "metrics.json",
        "run_state": run_dir / "run_state.json",
        "hdf5": run_dir / str(source_cfg["hdf5_relative_path"]),
    }
    required_hashes = source_cfg["sha256"]
    actual_hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing exp030 source artifact: {path}.")
        actual_hashes[name] = sha256_file(path)
        _require_equal(actual_hashes[name], str(required_hashes[name]).upper(), name)

    state = json.loads(paths["run_state"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    source_config = load_config(paths["config"])
    _require_equal(state.get("status"), "complete", "run_state.status")
    _require_equal(state.get("experiment_status"), "Passed", "run_state experiment")
    _require_equal(metrics.get("experiment_status"), "Passed", "metrics status")
    _require_equal(
        metrics["stage_status"].get("stage_A_to_C"), "Passed", "Stage A-C status"
    )
    _require_equal(
        metadata.get("experiment"),
        "exp030_TGV_2d_effective_phase",
        "experiment",
    )
    _require_equal(metadata.get("git_commit"), source_cfg["git_commit"], "git commit")
    _require_equal(
        metadata.get("a_to_b_forward_solver"),
        "continuous_axisymmetric_fresnel_hankel_on_compact_T_minus_1",
        "A-to-B solver",
    )
    _require_equal(
        metadata.get("a_to_b_reference_field"),
        "infinite_plane_wave_propagated_analytically",
        "reference field",
    )

    target_path = str(source_cfg["target_dataset"])
    arrays: dict[str, NDArray[np.generic]] = {}
    with h5py.File(paths["hdf5"], "r") as h5:
        if target_path not in h5:
            raise KeyError(f"Missing exp050 target dataset: {target_path}.")
        target_dataset = h5[target_path]
        target = np.asarray(target_dataset[()])
        if target.dtype != np.complex128 or target.ndim != 2:
            raise ValueError("exp050 target must be a 2D complex128 dataset.")
        if not np.all(np.isfinite(target)) or np.linalg.norm(target) <= 0.0:
            raise ValueError("exp050 target must be finite and have positive energy.")
        expected_shape = tuple(int(value) for value in source_cfg["shape_ny_nx"])
        if target.shape != expected_shape:
            raise ValueError("exp050 target shape does not match the source contract.")
        target_hash = sha256_array(target)
        _require_equal(target_hash, source_cfg["target_bytes_sha256"], "target bytes")

        radial_paths = {
            "source_radius_m": "/entry/truth/effective_forward/radial_source_r_m",
            "source_weights_m": "/entry/truth/effective_forward/radial_source_weight_m",
            "source_transmission_true": (
                "/entry/truth/effective_forward/A_effective_radial_true"
            ),
            "output_radius_m": "/entry/truth/effective_forward/P_B_radial_r_m",
            "target_radial_probe": "/entry/truth/effective_forward/P_B_radial_true",
        }
        for name, path in radial_paths.items():
            if path not in h5:
                raise KeyError(f"Missing exp030 operator dataset: {path}.")
            arrays[name] = np.asarray(h5[path][()])
            _require_equal(
                sha256_array(arrays[name]),
                source_cfg["operator_array_sha256"][name],
                name,
            )
        instrument = {
            name: float(h5[f"/entry/instrument/{dataset}"][()])
            for name, dataset in {
                "wavelength_m": "wavelength",
                "dx_m": "dx",
                "z_AB_m": "z_AB",
                "z_BC_m": "z_BC",
                "detector_pixel_size_m": "detector_pixel_size",
                "medium_index": "medium_index",
            }.items()
        }
        sample_a = {
            name: float(h5[f"/entry/sample/tgv_parameters/{dataset}"][()])
            for name, dataset in {
                "thickness_m": "thickness_m",
                "d_top_m": "d_top_m",
                "d_waist_true_m": "d_waist_m",
                "d_bottom_m": "d_bottom_m",
                "z_waist_m": "z_waist_m",
                "n_glass": "n_glass",
                "n_air": "n_air",
            }.items()
        }
        center = tuple(
            float(value) for value in h5["/entry/sample/tgv_parameters/center_xy_m"][()]
        )

    for key, value in config["instrument"].items():
        if key in instrument:
            _require_equal(instrument[key], float(value), f"instrument.{key}")
    for key, value in config["sample_a"].items():
        if key in sample_a:
            _require_equal(sample_a[key], float(value), f"sample_a.{key}")
    _require_equal(
        source_config["optics"]["shape"],
        list(target.shape),
        "source config shape",
    )
    _require_equal(
        source_config["optics"]["dx_m"],
        instrument["dx_m"],
        "source config dx",
    )

    fov_y = target.shape[0] * instrument["dx_m"]
    fov_x = target.shape[1] * instrument["dx_m"]
    endpoints = (
        -0.5 * (target.shape[1] - 1) * instrument["dx_m"],
        0.5 * (target.shape[1] - 1) * instrument["dx_m"],
    )
    return {
        "run_dir": run_dir,
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": actual_hashes,
        "state": state,
        "metadata": metadata,
        "metrics": metrics,
        "source_config": source_config,
        "target": target.astype(np.complex128, copy=False),
        "target_bytes_sha256": target_hash,
        "target_dataset_attrs": {},
        "target_dataset_units_attribute_present": False,
        "source_artifacts_validated_field_present": "artifacts_validated" in state,
        "instrument": instrument,
        "sample_a": sample_a,
        "center_xy_m": center,
        "shape_ny_nx": target.shape,
        "axis_order": "y_x",
        "coordinate_convention": "(index-(N-1)/2)*dx",
        "coordinate_endpoints_x_m": endpoints,
        "coordinate_endpoints_y_m": endpoints,
        "fov_yx_m": (fov_y, fov_x),
        **arrays,
    }


def make_exp050_candidate_generator(
    source: Mapping[str, Any],
) -> Callable[[float], dict[str, Any]]:
    """Create the matched fixed-operator exp050 candidate generator."""

    instrument = source["instrument"]
    sample = source["sample_a"]
    operator = build_exp030_radial_operator(
        source["source_radius_m"],
        source["source_weights_m"],
        source["output_radius_m"],
        shape=tuple(source["shape_ny_nx"]),
        dx_m=float(instrument["dx_m"]),
        center_xy_m=tuple(source["center_xy_m"]),
        wavelength_m=float(instrument["wavelength_m"]),
        propagation_distance_m=float(instrument["z_AB_m"]),
        medium_index=float(instrument["medium_index"]),
        incident_amplitude=1.0,
    )

    def generate(d_waist_m: float) -> dict[str, Any]:
        return make_exp030_projected_probe(
            operator,
            thickness_m=float(sample["thickness_m"]),
            d_top_m=float(sample["d_top_m"]),
            d_waist_m=float(d_waist_m),
            d_bottom_m=float(sample["d_bottom_m"]),
            z_waist_m=float(sample["z_waist_m"]),
            n_glass=float(sample["n_glass"]),
            n_air=float(sample["n_air"]),
            wavelength_m=float(instrument["wavelength_m"]),
            phase_scale=1.0,
        )

    return generate


@dataclass
class ProbeCache:
    """Deterministic float64-keyed cache for exp050 candidate fields."""

    generator: Callable[[float], Mapping[str, Any]]
    target_shape: tuple[int, int]
    index_by_key: dict[float, int] = field(default_factory=dict)
    diameters_m: list[float] = field(default_factory=list)
    probes: list[NDArray[np.complex128]] = field(default_factory=list)
    radial_probes: list[NDArray[np.complex128]] = field(default_factory=list)

    def get(self, diameter_m: float) -> tuple[NDArray[np.complex128], int]:
        key = float(np.float64(diameter_m))
        if not np.isfinite(key):
            raise ValueError("candidate diameter must be finite.")
        existing = self.index_by_key.get(key)
        if existing is not None:
            return self.probes[existing], existing
        result = self.generator(key)
        probe = np.asarray(result["P_B"])
        radial = np.asarray(result["P_B_radial"])
        if probe.dtype != np.complex128 or probe.shape != self.target_shape:
            raise ValueError("candidate generator returned invalid P_B shape or dtype.")
        if radial.dtype != np.complex128 or radial.ndim != 1:
            raise ValueError("candidate generator returned invalid radial probe.")
        if not np.all(np.isfinite(probe)) or not np.all(np.isfinite(radial)):
            raise ValueError("candidate generator returned non-finite values.")
        index = len(self.probes)
        self.index_by_key[key] = index
        self.diameters_m.append(key)
        self.probes.append(probe.copy())
        self.radial_probes.append(radial.copy())
        return self.probes[index], index


def _regular_grid(lower: float, upper: float, step: float) -> NDArray[np.float64]:
    count = int(np.rint((upper - lower) / step))
    if count <= 0 or not np.isclose(
        lower + count * step,
        upper,
        rtol=0.0,
        atol=32.0 * np.finfo(np.float64).eps * max(1.0, abs(upper)),
    ):
        raise ValueError("profile interval must be exactly divisible by its step.")
    grid = (lower + np.arange(count + 1, dtype=np.float64) * step).astype(
        np.float64
    )
    grid[-1] = upper
    return grid


def fixed_budget_golden_section_search(
    objective: Callable[[float], tuple[float, int]],
    *,
    bracket_m: tuple[float, float],
    iterations: int,
) -> dict[str, Any]:
    """Run a deterministic fixed-iteration golden-section cross-check."""

    lower, upper = (float(value) for value in bracket_m)
    if not np.all(np.isfinite([lower, upper])) or lower >= upper:
        raise ValueError("golden-section bracket must be finite and increasing.")
    if iterations <= 0:
        raise ValueError("golden-section iterations must be positive.")
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss, left_cache = objective(left)
    right_loss, right_cache = objective(right)
    evaluated_d = [left, right]
    evaluated_loss = [left_loss, right_loss]
    evaluated_cache = [left_cache, right_cache]
    lower_track = [lower]
    upper_track = [upper]
    for _ in range(iterations):
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            right_cache = left_cache
            left = upper - ratio * (upper - lower)
            left_loss, left_cache = objective(left)
            evaluated_d.append(left)
            evaluated_loss.append(left_loss)
            evaluated_cache.append(left_cache)
        else:
            lower = left
            left = right
            left_loss = right_loss
            left_cache = right_cache
            right = lower + ratio * (upper - lower)
            right_loss, right_cache = objective(right)
            evaluated_d.append(right)
            evaluated_loss.append(right_loss)
            evaluated_cache.append(right_cache)
        lower_track.append(lower)
        upper_track.append(upper)
    estimate = 0.5 * (lower + upper)
    estimate_loss, estimate_cache = objective(estimate)
    evaluated_d.append(estimate)
    evaluated_loss.append(estimate_loss)
    evaluated_cache.append(estimate_cache)
    return {
        "algorithm": "fixed_budget_golden_section_search",
        "iterations": int(iterations),
        "evaluation_count": len(evaluated_d),
        "initial_bracket_m": np.asarray(bracket_m, dtype=np.float64),
        "lower_m": np.asarray(lower_track, dtype=np.float64),
        "upper_m": np.asarray(upper_track, dtype=np.float64),
        "evaluated_d_waist_m": np.asarray(evaluated_d, dtype=np.float64),
        "evaluated_loss": np.asarray(evaluated_loss, dtype=np.float64),
        "evaluated_cache_index": np.asarray(evaluated_cache, dtype=np.int64),
        "final_bracket_m": np.asarray([lower, upper], dtype=np.float64),
        "final_bracket_width_m": float(upper - lower),
        "estimate_m": float(estimate),
        "estimate_loss": float(estimate_loss),
        "estimate_cache_index": int(estimate_cache),
        "stopping_reason": "fixed_iteration_budget",
    }


def run_exp050_estimator(
    target_probe: NDArray[np.complex128],
    target_radial_probe: NDArray[np.complex128],
    generator: Callable[[float], Mapping[str, Any]],
    *,
    true_waist_m: float,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the registered profile-first exp050 oracle estimator."""

    target = np.asarray(target_probe)
    target_radial = np.asarray(target_radial_probe)
    if target.dtype != np.complex128 or target.ndim != 2:
        raise ValueError("target_probe must be 2D complex128.")
    if target_radial.dtype != np.complex128 or target_radial.ndim != 1:
        raise ValueError("target_radial_probe must be 1D complex128.")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(target_radial)):
        raise ValueError("targets must contain only finite values.")
    lower, upper = (float(value) for value in design["bounds_m"])
    thresholds = design["thresholds"]
    cache = ProbeCache(generator, tuple(target.shape))

    replay_result = generator(float(true_waist_m))
    repeat_result = generator(float(true_waist_m))
    replay = np.asarray(replay_result["P_B"])
    repeat = np.asarray(repeat_result["P_B"])
    radial_replay = np.asarray(replay_result["P_B_radial"])
    replay_metrics = replay_error_metrics(replay, target)
    replay_metrics.update(
        {
            "deterministic_repeat_relative_l2": float(
                np.sqrt(raw_complex_probe_loss(repeat, replay))
            ),
            "radial_raw_relative_l2": float(
                np.linalg.norm(radial_replay - target_radial)
                / np.linalg.norm(target_radial)
            ),
        }
    )
    replay_pass = bool(
        replay_metrics["raw_complex_relative_l2"]
        <= float(thresholds["replay_raw_relative_l2_max"])
        and replay_metrics["amplitude_relative_l2"]
        <= float(thresholds["replay_amplitude_relative_l2_max"])
        and replay_metrics["amplitude_weighted_phase_sensitive_relative_l2"]
        <= float(thresholds["replay_phase_sensitive_relative_l2_max"])
        and replay_metrics["deterministic_repeat_relative_l2"]
        <= float(thresholds["deterministic_repeat_relative_l2_max"])
        and replay_metrics["radial_raw_relative_l2"]
        <= float(thresholds["radial_replay_relative_l2_max"])
    )
    cache.index_by_key[float(np.float64(true_waist_m))] = 0
    cache.diameters_m.append(float(true_waist_m))
    cache.probes.append(replay.copy())
    cache.radial_probes.append(radial_replay.copy())
    result: dict[str, Any] = {
        "status": "Inconclusive",
        "interpretation": "artifact_operator_handoff_not_closed",
        "replay": replay_metrics,
        "replay_pass": replay_pass,
        "P_B_replay_raw": replay,
        "P_B_replay_repeat_raw": repeat,
        "P_B_radial_replay_raw": radial_replay,
    }
    if not replay_pass:
        result["cache"] = {
            "D_waist_m": np.asarray(cache.diameters_m, dtype=np.float64),
            "P_B_candidate": np.stack(cache.probes),
            "P_B_radial_candidate": np.stack(cache.radial_probes),
        }
        return result

    def objective(diameter_m: float) -> tuple[float, int]:
        if not lower <= diameter_m <= upper:
            raise ValueError("candidate diameter lies outside registered bounds.")
        probe, cache_index = cache.get(diameter_m)
        return raw_complex_probe_loss(probe, target), cache_index

    global_grid = _regular_grid(lower, upper, float(design["global_step_m"]))
    global_losses: list[float] = []
    global_cache: list[int] = []
    for diameter in global_grid:
        loss, cache_index = objective(float(diameter))
        global_losses.append(loss)
        global_cache.append(cache_index)
    global_loss = np.asarray(global_losses, dtype=np.float64)
    global_argmin = int(np.argmin(global_loss))
    global_center = float(global_grid[global_argmin])
    fine_lower = global_center - float(design["fine_half_width_m"])
    fine_upper = global_center + float(design["fine_half_width_m"])
    if fine_lower < lower or fine_upper > upper:
        raise ValueError("profile-derived fine interval crosses registered bounds.")
    fine_grid = _regular_grid(fine_lower, fine_upper, float(design["fine_step_m"]))
    fine_losses: list[float] = []
    fine_cache: list[int] = []
    for diameter in fine_grid:
        loss, cache_index = objective(float(diameter))
        fine_losses.append(loss)
        fine_cache.append(cache_index)
    fine_loss = np.asarray(fine_losses, dtype=np.float64)

    floor_relative = max(
        replay_metrics["raw_complex_relative_l2"],
        replay_metrics["deterministic_repeat_relative_l2"],
        np.finfo(np.float64).eps,
    )
    floor_loss = floor_relative**2
    uniqueness_tolerance = max(
        float(thresholds["profile_uniqueness_loss_floor"]), 100.0 * floor_loss
    )
    global_minimum = extract_profile_minimum(
        global_grid, global_loss, uniqueness_tolerance=uniqueness_tolerance
    )
    fine_minimum = extract_profile_minimum(
        fine_grid, fine_loss, uniqueness_tolerance=uniqueness_tolerance
    )
    fine_index = int(fine_minimum["minimum_index"])
    if fine_index == 0 or fine_index == len(fine_grid) - 1:
        resolution_interval = np.asarray([fine_grid[fine_index]] * 2)
    else:
        resolution_interval = np.asarray(
            [
                0.5 * (fine_grid[fine_index - 1] + fine_grid[fine_index]),
                0.5 * (fine_grid[fine_index] + fine_grid[fine_index + 1]),
            ],
            dtype=np.float64,
        )
    second_best_to_floor = float(
        fine_minimum["second_best_loss"]
        / max(floor_loss, np.finfo(np.float64).eps)
    )
    profile_error = abs(float(fine_minimum["minimum_d_waist_m"]) - true_waist_m)
    profile_pass = bool(
        global_minimum["unique"]
        and fine_minimum["unique"]
        and 0 < fine_index < len(fine_grid) - 1
        and profile_error <= float(thresholds["profile_absolute_error_m_max"])
        and float(fine_minimum["minimum_loss"])
        <= float(thresholds["profile_minimum_loss_max"])
        and second_best_to_floor
        >= float(thresholds["profile_second_best_to_floor_ratio_min"])
    )

    h1, h2 = (float(value) for value in design["finite_difference_steps_m"])
    fd_fields: dict[str, NDArray[np.complex128]] = {}
    fd_cache: dict[str, int] = {}
    for name, diameter in {
        "minus_h1": true_waist_m - h1,
        "plus_h1": true_waist_m + h1,
        "minus_h2": true_waist_m - h2,
        "plus_h2": true_waist_m + h2,
    }.items():
        fd_fields[name], fd_cache[name] = cache.get(diameter)
    fd = finite_difference_controls(
        target,
        replay,
        fd_fields["minus_h1"],
        fd_fields["plus_h1"],
        fd_fields["minus_h2"],
        fd_fields["plus_h2"],
        h1_m=h1,
        h2_m=h2,
    )
    signature = min(
        np.sqrt(float(fd["minus_h2_loss"])),
        np.sqrt(float(fd["plus_h2_loss"])),
    )
    signature_to_floor = float(signature / floor_relative)
    fd_pass = bool(
        float(fd["normalized_jacobian_h1_per_m"])
        >= float(thresholds["normalized_jacobian_per_m_min"])
        and np.isfinite(float(fd["loss_curvature_per_m2"]))
        and float(fd["loss_curvature_per_m2"]) > 0.0
        and float(fd["jacobian_step_relative_l2"])
        <= float(thresholds["jacobian_step_relative_l2_max"])
        and signature_to_floor >= float(thresholds["signature_to_floor_ratio_min"])
    )
    fd.update(
        {
            "cache_indices": fd_cache,
            "one_sided_signature_relative_l2": signature,
            "signature_to_replay_floor_ratio": signature_to_floor,
            "pass": fd_pass,
        }
    )

    if global_argmin == 0 or global_argmin == len(global_grid) - 1:
        golden_bracket = (lower, upper)
    else:
        golden_bracket = (
            float(global_grid[global_argmin - 1]),
            float(global_grid[global_argmin + 1]),
        )
    golden = fixed_budget_golden_section_search(
        objective,
        bracket_m=golden_bracket,
        iterations=int(design["golden_iterations"]),
    )
    profile_estimate = float(fine_minimum["minimum_d_waist_m"])
    golden_error = abs(float(golden["estimate_m"]) - true_waist_m)
    method_agreement = abs(float(golden["estimate_m"]) - profile_estimate)
    golden_pass = bool(
        golden["stopping_reason"] == "fixed_iteration_budget"
        and golden["final_bracket_width_m"]
        <= float(thresholds["golden_final_bracket_width_m_max"])
        and golden_error <= float(thresholds["golden_absolute_error_m_max"])
        and method_agreement <= float(thresholds["method_agreement_m_max"])
        and float(golden["final_bracket_m"][0])
        > lower + float(thresholds["boundary_margin_m"])
        and float(golden["final_bracket_m"][1])
        < upper - float(thresholds["boundary_margin_m"])
    )
    golden.update(
        {
            "absolute_error_m": golden_error,
            "profile_agreement_m": method_agreement,
            "pass": golden_pass,
        }
    )
    estimate_m = profile_estimate
    best_probe, best_cache_index = cache.get(estimate_m)
    residual = np.asarray(best_probe - target, dtype=np.complex128)
    if not fd_pass:
        status = "Inconclusive"
        interpretation = "local_numerical_control_not_closed"
    elif not profile_pass or not golden_pass:
        status = "Failed"
        interpretation = "registered_profile_or_estimator_stability_gate_failed"
    else:
        status = "Passed"
        interpretation = (
            "exp030_projected_true_probe_single_parameter_oracle_fit_passed"
        )
    result.update(
        {
            "status": status,
            "interpretation": interpretation,
            "profile": {
                "global_d_waist_m": global_grid,
                "global_loss": global_loss,
                "global_cache_index": np.asarray(global_cache, dtype=np.int64),
                "global_minimum": global_minimum,
                "fine_d_waist_m": fine_grid,
                "fine_loss": fine_loss,
                "fine_cache_index": np.asarray(fine_cache, dtype=np.int64),
                "fine_minimum": fine_minimum,
                "reported_resolution_interval_m": resolution_interval,
                "reported_resolution_interval_width_m": float(
                    resolution_interval[1] - resolution_interval[0]
                ),
                "reported_resolution_role": (
                    "registered_profile_grid_cell_not_physical_uncertainty"
                ),
                "absolute_error_m": profile_error,
                "second_best_to_replay_floor_loss_ratio": second_best_to_floor,
                "replay_floor_relative_l2": floor_relative,
                "replay_floor_loss": floor_loss,
                "uniqueness_tolerance": uniqueness_tolerance,
                "pass": profile_pass,
            },
            "finite_difference": fd,
            "estimator_crosscheck": golden,
            "estimate_m": estimate_m,
            "best_cache_index": best_cache_index,
            "P_B_best_raw": best_probe,
            "residual_field_raw": residual,
            "cache": {
                "D_waist_m": np.asarray(cache.diameters_m, dtype=np.float64),
                "P_B_candidate": np.stack(cache.probes),
                "P_B_radial_candidate": np.stack(cache.radial_probes),
            },
        }
    )
    return result
