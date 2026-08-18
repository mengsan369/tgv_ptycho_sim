"""Run the non-scientific exp040 R10 Stage-A cost preflight."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import platform
import statistics
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.exp040 import (  # noqa: E402
    _r7_streamed_tgv_exit,
    relative_l2,
)
from tgv_ptycho.io.config import load_config, save_config  # noqa: E402
from tgv_ptycho.io.metadata import (  # noqa: E402
    created_at_utc,
    get_git_commit,
)
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json  # noqa: E402
from tgv_ptycho.objects.tgv_geometry import diameter_profile  # noqa: E402
from tgv_ptycho.optics.fields import make_plane_wave  # noqa: E402


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load_percent", ctypes.c_ulong),
        ("total_physical_bytes", ctypes.c_ulonglong),
        ("available_physical_bytes", ctypes.c_ulonglong),
        ("total_pagefile_bytes", ctypes.c_ulonglong),
        ("available_pagefile_bytes", ctypes.c_ulonglong),
        ("total_virtual_bytes", ctypes.c_ulonglong),
        ("available_virtual_bytes", ctypes.c_ulonglong),
        ("available_extended_virtual_bytes", ctypes.c_ulonglong),
    ]


class _RssSampler:
    """Sample current-process RSS on the registered Windows host."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._samples: list[int] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples.append(_current_rss_bytes())
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._samples.append(_current_rss_bytes())
        return max(self._samples)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._samples.append(_current_rss_bytes())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _expected_config() -> dict[str, Any]:
    return {
        "run": {
            "name": "exp040_TGV_3d_multislice_r10_stage_a_preflight",
            "output_root": "runs",
        },
        "experiment": {
            "id": "exp040",
            "stage": "R10_stage_a_preflight",
            "scientific_result": False,
            "description": (
                "Non-scientific runtime and memory feasibility preflight for "
                "the pre-registered R10 Stage A workload."
            ),
        },
        "provenance": {
            "r9_run": (
                "runs/exp040_TGV_3d_multislice_r9_a_exit_attribution_"
                "20260814_184026"
            ),
            "r9_progress_sha256": (
                "F083A10481EF028D55DD53635D9A503442898753A3852487E7081AB929CDF476"
            ),
            "observed_512_full_case_elapsed_s": 81.738972,
        },
        "physics": {
            "wavelength_m": 5.32e-7,
            "internal_reference_index": 1.5,
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
        },
        "benchmark": {
            "kernel": "r7_streamed_tgv_exit",
            "interface_factor": 8,
            "dz_m": 2.5e-7,
            "representative_z_rule": (
                "equal_stratum_midpoints_full_thickness"
            ),
            "timed_slice_count": 16,
            "warmup_slice_count": 1,
            "timed_repeats": 3,
            "fixed_case_order": ["current_512", "fine_1024"],
            "cases": [
                {
                    "id": "current_512",
                    "shape": [512, 512],
                    "dx_m": 1.25e-7,
                    "formal_slice_count": 400,
                },
                {
                    "id": "fine_1024",
                    "shape": [1024, 1024],
                    "dx_m": 6.25e-8,
                    "formal_slice_count": 400,
                },
            ],
            "stream_slices": True,
            "retain_full_volumes": False,
            "recompute_detector_path": False,
            "save_fields": False,
            "rss_sampling_interval_s": 0.02,
            "runtime_statistic": "median_seconds_per_slice",
            "full_case_projection": (
                "median_seconds_per_slice_times_formal_slice_count"
            ),
        },
        "feasibility": {
            "determinism_relative_l2_max": 1.0e-14,
            "calibration_relative_error_max": 0.25,
            "safety_factor": 1.5,
            "safety_adjusted_stage_a_wall_s_max": 900.0,
            "sampled_peak_rss_over_available_max": 0.5,
            "require_zero_interface_bound_errors": True,
            "require_all_finite": True,
        },
        "output": {
            "save_hdf5": False,
            "save_png": False,
            "save_fields": False,
            "allowed_files": [
                "config.yaml",
                "metadata.json",
                "metrics.json",
                "run_state.json",
            ],
        },
    }


def validate_preflight_config(config: Mapping[str, Any]) -> None:
    """Require exact agreement with the append-only pre-registration."""

    if dict(config) != _expected_config():
        msg = "R10 Stage-A preflight config differs from its frozen registration."
        raise ValueError(msg)


def _require_windows() -> None:
    if sys.platform != "win32":
        msg = "The registered R10 performance preflight targets the Windows host."
        raise RuntimeError(msg)


def _current_rss_bytes() -> int:
    _require_windows()
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    success = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        ctypes.sizeof(counters),
    )
    if success == 0:
        msg = "GetProcessMemoryInfo failed during R10 preflight."
        raise OSError(msg)
    return int(counters.working_set_size)


def _available_memory_bytes() -> int:
    _require_windows()
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalMemoryStatusEx.argtypes = [
        ctypes.POINTER(_MemoryStatus)
    ]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    success = kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if success == 0:
        msg = "GlobalMemoryStatusEx failed during R10 preflight."
        raise OSError(msg)
    return int(status.available_physical_bytes)


def _representative_z_m(thickness_m: float, count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    return (indices + 0.5) * float(thickness_m) / float(count)


def _run_kernel(
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    slice_count: int,
    diameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    physics = config["physics"]
    benchmark = config["benchmark"]
    shape = tuple(int(value) for value in case["shape"])
    dx_m = float(case["dx_m"])
    incident = make_plane_wave(
        shape,
        dx_m,
        float(physics["wavelength_m"]),
        theta_x=float(physics["illumination_theta_x_rad"]),
        theta_y=float(physics["illumination_theta_y_rad"]),
        amplitude=float(physics["illumination_amplitude"]),
    )
    widths = np.full(
        slice_count, float(benchmark["dz_m"]), dtype=np.float64
    )
    output, selected_fraction, controls = _r7_streamed_tgv_exit(
        incident=np.asarray(incident, dtype=np.complex128),
        shape=shape,
        dx_m=dx_m,
        widths=widths,
        diameters=np.asarray(diameters, dtype=np.float64),
        interface_factor=int(benchmark["interface_factor"]),
        center_xy_m=tuple(float(value) for value in physics["center_xy_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        wavelength=float(physics["wavelength_m"]),
        n_ref=float(physics["internal_reference_index"]),
        bandlimit=bool(physics["angular_spectrum_bandlimit"]),
        selected_slice_index=slice_count // 2,
    )
    return output, selected_fraction, controls


def _benchmark_case(
    config: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    physics = config["physics"]
    benchmark = config["benchmark"]
    timed_slice_count = int(benchmark["timed_slice_count"])
    thickness_m = float(physics["sample_thickness_m"])
    representative_z = _representative_z_m(thickness_m, timed_slice_count)
    representative_diameters = diameter_profile(
        representative_z,
        thickness_m,
        float(physics["d_top_m"]),
        float(physics["d_waist_m"]),
        float(physics["d_bottom_m"]),
        float(physics["z_waist_m"]),
    )
    warmup_diameter = diameter_profile(
        np.asarray([float(physics["z_waist_m"])]),
        thickness_m,
        float(physics["d_top_m"]),
        float(physics["d_waist_m"]),
        float(physics["d_bottom_m"]),
        float(physics["z_waist_m"]),
    )

    warmup_started = time.perf_counter()
    warmup_output, warmup_fraction, warmup_controls = _run_kernel(
        config,
        case,
        slice_count=int(benchmark["warmup_slice_count"]),
        diameters=warmup_diameter,
    )
    warmup_elapsed_s = time.perf_counter() - warmup_started
    if not (
        np.all(np.isfinite(warmup_output))
        and np.all(np.isfinite(warmup_fraction))
        and bool(warmup_controls["all_finite"])
    ):
        msg = f"R10 preflight warm-up is non-finite for {case['id']}."
        raise RuntimeError(msg)
    del warmup_output, warmup_fraction
    gc.collect()

    repeat_records: list[dict[str, Any]] = []
    reference_output: np.ndarray | None = None
    determinism_errors: list[float] = []
    for repeat_index in range(int(benchmark["timed_repeats"])):
        sampler = _RssSampler(float(benchmark["rss_sampling_interval_s"]))
        sampler.start()
        started = time.perf_counter()
        output, selected_fraction, controls = _run_kernel(
            config,
            case,
            slice_count=timed_slice_count,
            diameters=representative_diameters,
        )
        elapsed_s = time.perf_counter() - started
        peak_rss_bytes = sampler.stop()
        all_finite = bool(
            np.all(np.isfinite(output))
            and np.all(np.isfinite(selected_fraction))
            and controls["all_finite"]
        )
        if reference_output is None:
            determinism_error = 0.0
            reference_output = output.copy()
        else:
            determinism_error = relative_l2(output, reference_output)
        determinism_errors.append(float(determinism_error))
        repeat_records.append(
            {
                "repeat_index": repeat_index,
                "elapsed_s": float(elapsed_s),
                "seconds_per_slice": float(elapsed_s / timed_slice_count),
                "sampled_peak_rss_bytes": int(peak_rss_bytes),
                "all_finite": all_finite,
                "fraction_bound_error": float(
                    controls["fraction_bound_error"]
                ),
                "index_bound_error": float(controls["index_bound_error"]),
                "count_identity_error": float(
                    controls["count_identity_error"]
                ),
                "determinism_relative_l2": float(determinism_error),
            }
        )
        del output, selected_fraction
        gc.collect()

    elapsed_values = [record["elapsed_s"] for record in repeat_records]
    median_elapsed_s = float(statistics.median(elapsed_values))
    median_seconds_per_slice = median_elapsed_s / timed_slice_count
    formal_slice_count = int(case["formal_slice_count"])
    projected_full_case_s = median_seconds_per_slice * formal_slice_count
    shape = tuple(int(value) for value in case["shape"])
    return {
        "id": str(case["id"]),
        "shape": list(shape),
        "dx_m": float(case["dx_m"]),
        "interface_factor": int(benchmark["interface_factor"]),
        "timed_slice_count": timed_slice_count,
        "formal_slice_count": formal_slice_count,
        "q8_subnode_tests_timed": int(
            shape[0]
            * shape[1]
            * timed_slice_count
            * int(benchmark["interface_factor"]) ** 2
        ),
        "q8_subnode_tests_formal": int(
            shape[0]
            * shape[1]
            * formal_slice_count
            * int(benchmark["interface_factor"]) ** 2
        ),
        "warmup_elapsed_s": float(warmup_elapsed_s),
        "repeat_records": repeat_records,
        "elapsed_s_min": float(min(elapsed_values)),
        "elapsed_s_median": median_elapsed_s,
        "elapsed_s_max": float(max(elapsed_values)),
        "median_seconds_per_slice": float(median_seconds_per_slice),
        "projected_full_case_elapsed_s": float(projected_full_case_s),
        "sampled_peak_rss_bytes": int(
            max(
                int(record["sampled_peak_rss_bytes"])
                for record in repeat_records
            )
        ),
        "maximum_determinism_relative_l2": float(max(determinism_errors)),
        "all_finite": bool(
            all(bool(record["all_finite"]) for record in repeat_records)
        ),
        "maximum_interface_bound_error": float(
            max(
                max(
                    float(record[name])
                    for name in (
                        "fraction_bound_error",
                        "index_bound_error",
                        "count_identity_error",
                    )
                )
                for record in repeat_records
            )
        ),
    }


def _preflight_outcome(
    config: Mapping[str, Any],
    case_results: list[Mapping[str, Any]],
    *,
    available_memory_bytes: int,
) -> dict[str, Any]:
    feasibility = config["feasibility"]
    provenance = config["provenance"]
    by_id = {str(case["id"]): case for case in case_results}
    current = by_id["current_512"]
    observed_s = float(provenance["observed_512_full_case_elapsed_s"])
    projected_current_s = float(current["projected_full_case_elapsed_s"])
    calibration_relative_error = abs(projected_current_s - observed_s) / observed_s
    projected_stage_a_s = float(
        sum(float(case["projected_full_case_elapsed_s"]) for case in case_results)
    )
    safety_adjusted_s = (
        float(feasibility["safety_factor"]) * projected_stage_a_s
    )
    sampled_peak_rss_bytes = max(
        int(case["sampled_peak_rss_bytes"]) for case in case_results
    )
    peak_rss_over_available = (
        sampled_peak_rss_bytes / float(available_memory_bytes)
    )
    kernel_controls_pass = bool(
        all(bool(case["all_finite"]) for case in case_results)
        and all(
            float(case["maximum_interface_bound_error"]) == 0.0
            for case in case_results
        )
        and all(
            float(case["maximum_determinism_relative_l2"])
            <= float(feasibility["determinism_relative_l2_max"])
            for case in case_results
        )
    )
    calibration_pass = bool(
        calibration_relative_error
        <= float(feasibility["calibration_relative_error_max"])
    )
    wall_time_pass = bool(
        safety_adjusted_s
        <= float(feasibility["safety_adjusted_stage_a_wall_s_max"])
    )
    memory_pass = bool(
        peak_rss_over_available
        <= float(feasibility["sampled_peak_rss_over_available_max"])
    )

    if not kernel_controls_pass:
        status = "Failed"
        interpretation = "preflight_kernel_control_failed"
    elif not calibration_pass:
        status = "Inconclusive"
        interpretation = "short_kernel_extrapolation_not_calibrated"
    elif not (wall_time_pass and memory_pass):
        status = "Inconclusive"
        interpretation = "stage_a_cost_not_feasible_on_current_host"
    else:
        status = "Passed"
        interpretation = "stage_a_formal_run_feasible"
    return {
        "status": status,
        "interpretation_code": interpretation,
        "kernel_controls_pass": kernel_controls_pass,
        "calibration_pass": calibration_pass,
        "wall_time_pass": wall_time_pass,
        "memory_pass": memory_pass,
        "observed_512_full_case_elapsed_s": observed_s,
        "projected_512_full_case_elapsed_s": projected_current_s,
        "calibration_relative_error": float(calibration_relative_error),
        "projected_stage_a_elapsed_s": projected_stage_a_s,
        "safety_factor": float(feasibility["safety_factor"]),
        "safety_adjusted_stage_a_elapsed_s": float(safety_adjusted_s),
        "available_memory_bytes_at_start": int(available_memory_bytes),
        "maximum_sampled_peak_rss_bytes": int(sampled_peak_rss_bytes),
        "maximum_sampled_peak_rss_over_available": float(
            peak_rss_over_available
        ),
        "performance_only": True,
        "scientific_conclusion_allowed": False,
        "stage_b_allowed": False,
    }


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    expected = set(config["output"]["allowed_files"])
    actual = {
        path.name for path in run_dir.rglob("*") if path.is_file()
    }
    if actual != expected:
        msg = f"R10 preflight artifact names differ: {sorted(actual)}"
        raise RuntimeError(msg)
    for name in ("metadata.json", "metrics.json", "run_state.json"):
        with (run_dir / name).open("r", encoding="utf-8") as handle:
            json.load(handle)
    if any((run_dir / "figures").iterdir()):
        msg = "R10 preflight must not write figures."
        raise RuntimeError(msg)
    if any((run_dir / "outputs").iterdir()):
        msg = "R10 preflight must not write scientific outputs."
        raise RuntimeError(msg)


def run(config_path: Path) -> Path:
    """Execute the frozen non-scientific performance preflight."""

    config = load_config(config_path)
    validate_preflight_config(config)
    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "created_at": created_at_utc(),
            "source_config": str(config_path.resolve()),
            "scientific_result": False,
        },
    )

    started = time.perf_counter()
    try:
        available_memory = _available_memory_bytes()
        case_results = [
            _benchmark_case(config, case)
            for case in config["benchmark"]["cases"]
        ]
        outcome = _preflight_outcome(
            config,
            case_results,
            available_memory_bytes=available_memory,
        )
        elapsed_s = time.perf_counter() - started
        metrics = {
            "version": "R10_stage_a_preflight",
            "scientific_result": False,
            "benchmark": {
                "representative_z_rule": config["benchmark"][
                    "representative_z_rule"
                ],
                "timed_slice_count": int(
                    config["benchmark"]["timed_slice_count"]
                ),
                "timed_repeats": int(
                    config["benchmark"]["timed_repeats"]
                ),
                "rss_sampling_interval_s": float(
                    config["benchmark"]["rss_sampling_interval_s"]
                ),
                "case_results": case_results,
            },
            "outcome": outcome,
            "total_preflight_elapsed_s": float(elapsed_s),
            "no_hdf5": True,
            "no_figures": True,
            "no_fields_persisted": True,
        }
        metadata = {
            "experiment_id": "exp040",
            "diagnostic_stage": "R10_stage_a_preflight",
            "scientific_result": False,
            "run_path": str(run_dir.resolve()),
            "source_config": str(config_path.resolve()),
            "created_at": created_at_utc(),
            "git_commit": get_git_commit(PROJECT_ROOT) or "unavailable",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "status": outcome["status"],
            "interpretation_code": outcome["interpretation_code"],
        }
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "validation_pending",
                "completed_at": created_at_utc(),
                "preflight_status": outcome["status"],
                "scientific_result": False,
            },
        )
        _validate_artifacts(run_dir, config)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "completed_at": created_at_utc(),
                "preflight_status": outcome["status"],
                "interpretation_code": outcome["interpretation_code"],
                "scientific_result": False,
                "artifacts_validated": True,
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
                "scientific_result": False,
            },
        )
        raise

    print(f"run_dir: {run_dir.resolve()}")
    print(f"preflight_status: {outcome['status']}")
    print(f"interpretation: {outcome['interpretation_code']}")
    print(
        "projected_stage_a_elapsed_s: "
        f"{outcome['projected_stage_a_elapsed_s']:.6f}"
    )
    print(
        "safety_adjusted_stage_a_elapsed_s: "
        f"{outcome['safety_adjusted_stage_a_elapsed_s']:.6f}"
    )
    print(
        "maximum_sampled_peak_rss_over_available: "
        f"{outcome['maximum_sampled_peak_rss_over_available']:.6f}"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
