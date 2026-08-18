"""Run the non-scientific exp040 R11 resource and algebra preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil
from scipy.special import j0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.helmholtz_axisymmetric import (  # noqa: E402
    adc5_shifted_wavenumber_squared,
    assemble_cylindrical_helmholtz,
    cartesian_polar_angular_diagnostics,
    make_axisymmetric_grid,
    make_background_n2,
    make_cylindrical_pml,
    make_manufactured_vector,
    solve_sparse_direct,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.objects.tgv3d import (  # noqa: E402
    make_tgv_air_fraction_slice_chord_quadrature,
)

REGISTERED_CONFIG_SHA256 = (
    "2FEAA121E7B6EA4F2B3F3BC0AC3C2843891AC31214FBF2156FD369D072252CF4"
)


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


def _relative_l2(test: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(test - reference) ** 2))
    denominator = float(np.sum(np.abs(reference) ** 2))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).eps)))


def _hdf5_safe(value: Any) -> Any:
    """Encode mapping-valued sequences for the project's HDF5 writer."""

    if isinstance(value, Mapping):
        return {str(key): _hdf5_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)) and any(
        isinstance(child, Mapping) for child in value
    ):
        return {
            "__sequence_encoding__": "indexed_mapping_v1",
            "length": len(value),
            "items": {
                f"{index:06d}": _hdf5_safe(child)
                for index, child in enumerate(value)
            },
        }
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _hdf5_to_plain(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _hdf5_to_plain(node[key]) for key in node}
    return _plain(node[()])


def _decode_hdf5_sequences(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__sequence_encoding__", "items", "length"}:
            if value["__sequence_encoding__"] != "indexed_mapping_v1":
                raise ValueError("unknown R11 preflight HDF5 sequence encoding")
            length = int(value["length"])
            items = value["items"]
            if not isinstance(items, Mapping):
                raise ValueError("encoded HDF5 sequence items must be a mapping")
            expected = [f"{index:06d}" for index in range(length)]
            if list(sorted(items)) != expected:
                raise ValueError("encoded HDF5 sequence indices differ")
            return [_decode_hdf5_sequences(items[key]) for key in expected]
        return {
            str(key): _decode_hdf5_sequences(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_decode_hdf5_sequences(child) for child in value]
    return value


def _require_exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(f"R11 preflight {name} differs from registration.")


def validate_preflight_config(config: Mapping[str, Any]) -> None:
    """Validate all registered preflight controls."""

    _require_exact(
        set(config),
        {
            "run",
            "experiment",
            "provenance",
            "physics",
            "formal_grids",
            "resource_model",
            "probe_grid",
            "geometry_control",
            "polar_control",
            "thresholds",
            "output",
        },
        "top-level sections",
    )
    _require_exact(config["experiment"]["id"], "exp040", "experiment id")
    _require_exact(config["experiment"]["stage"], "R11_preflight", "stage")
    _require_exact(
        config["experiment"]["scientific_result"], False, "scientific role"
    )
    _require_exact(
        dict(config["physics"]),
        {
            "wavelength_m": 5.32e-7,
            "n_glass": 1.5,
            "n_air": 1.0,
            "interface_z_m": 1.0e-4,
        },
        "physics",
    )
    expected_cases = [
        {
            "id": "core24_fine",
            "radial_core_max_m": 2.4e-5,
            "h_m": 8.333333333333333e-8,
            "expected_nr": 312,
            "expected_nz": 1296,
            "expected_unknowns": 404352,
        },
        {
            "id": "core36_fine",
            "radial_core_max_m": 3.6e-5,
            "h_m": 8.333333333333333e-8,
            "expected_nr": 456,
            "expected_nz": 1296,
            "expected_unknowns": 590976,
        },
        {
            "id": "core48_coarse",
            "radial_core_max_m": 4.8e-5,
            "h_m": 1.25e-7,
            "expected_nr": 400,
            "expected_nz": 864,
            "expected_unknowns": 345600,
        },
        {
            "id": "core48_fine",
            "radial_core_max_m": 4.8e-5,
            "h_m": 8.333333333333333e-8,
            "expected_nr": 600,
            "expected_nz": 1296,
            "expected_unknowns": 777600,
        },
    ]
    _require_exact(
        list(config["formal_grids"]["cases"]),
        expected_cases,
        "formal grids",
    )
    _require_exact(
        dict(config["geometry_control"]),
        {
            "shapes": [[512, 512], [1024, 1024]],
            "dx_m": [1.25e-7, 6.25e-8],
            "diameters_m": [2.0e-5, 2.5e-5, 3.0e-5],
            "formal_order": 64,
            "validation_order": 128,
        },
        "geometry control",
    )
    _require_exact(
        dict(config["polar_control"]),
        {
            "radius_count": 160,
            "radius_spacing_m": 1.25e-7,
            "theta_count": 720,
            "interpolation_order": 3,
            "radial_test_frequency_cycles_per_m": 1879699.2481203007,
            "shapes": [[512, 512], [1024, 1024]],
            "dx_m": [1.25e-7, 6.25e-8],
        },
        "polar control",
    )
    _require_exact(
        config["output"]["hdf5_filename"],
        "exp040_r11_preflight.h5",
        "HDF5 filename",
    )


def _progress_writer(path: Path):
    payload: dict[str, Any] = {
        "purpose": "r11_non_scientific_resource_and_algebra_preflight",
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


def _validate_provenance(config: Mapping[str, Any]) -> None:
    provenance = config["provenance"]
    run_dir = PROJECT_ROOT / str(provenance["r10_stage_b_run"])
    paths = {
        "r10_metrics_sha256": run_dir / "metrics.json",
        "r10_q8_checkpoint_sha256": (
            run_dir / "checkpoints" / "multislice_fine_1024.npz"
        ),
        "r10_repaired_hdf5_sha256": (
            run_dir / "outputs" / "exp040_r10_stage_b_repaired.h5"
        ),
    }
    for key, path in paths.items():
        if not path.is_file() or _sha256(path) != str(provenance[key]):
            raise RuntimeError(f"locked provenance differs: {key}")


def _formal_grid_controls(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    formal = config["formal_grids"]
    controls: list[dict[str, Any]] = []
    for case in formal["cases"]:
        grid = make_axisymmetric_grid(
            dr_m=float(case["h_m"]),
            dz_m=float(case["h_m"]),
            radial_core_max_m=float(case["radial_core_max_m"]),
            z_core_min_m=float(formal["z_core_min_m"]),
            z_core_max_m=float(formal["z_core_max_m"]),
            pml_thickness_m=float(formal["pml_thickness_m"]),
        )
        expected = [
            int(case["expected_nr"]),
            int(case["expected_nz"]),
            int(case["expected_unknowns"]),
        ]
        actual = [grid.nr, grid.nz, grid.unknown_count]
        controls.append(
            {
                "id": str(case["id"]),
                "nr": grid.nr,
                "nz": grid.nz,
                "unknown_count": grid.unknown_count,
                "expected_match": bool(actual == expected),
            }
        )
    return controls


def _resource_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["resource_model"]
    cases = config["formal_grids"]["cases"]
    maximum_unknowns = max(int(case["expected_unknowns"]) for case in cases)
    scale = (
        maximum_unknowns / int(model["reference_unknowns"])
    ) ** float(model["scaling_exponent"])
    estimated_peak = float(model["reference_peak_rss_bytes"]) * scale
    estimated_time = float(model["reference_factor_solve_s"]) * scale
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(PROJECT_ROOT)
    total_fraction = estimated_peak / memory.total
    return {
        "maximum_unknowns": maximum_unknowns,
        "estimated_peak_rss_bytes": int(np.ceil(estimated_peak)),
        "estimated_peak_rss_gib": float(estimated_peak / 2**30),
        "estimated_factor_solve_s": estimated_time,
        "total_physical_memory_bytes": int(memory.total),
        "available_physical_memory_bytes_report_only": int(memory.available),
        "estimated_peak_fraction_of_total": float(total_fraction),
        "disk_free_bytes": int(disk.free),
        "total_memory_pass": bool(
            memory.total
            >= float(model["total_physical_memory_gib_min"]) * 2**30
        ),
        "estimated_peak_fraction_pass": bool(
            total_fraction
            <= float(model["estimated_peak_fraction_of_total_max"])
        ),
        "disk_free_pass": bool(
            disk.free >= float(model["disk_free_gib_min"]) * 2**30
        ),
    }


def _adc_algebra_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    physics = config["physics"]
    k0 = 2.0 * np.pi / float(physics["wavelength_m"])
    errors: list[float] = []
    ratios: list[float] = []
    for refractive_index in (float(physics["n_air"]), float(physics["n_glass"])):
        k = k0 * refractive_index
        for spacing in (1.25e-7, 8.333333333333333e-8):
            shifted = float(adc5_shifted_wavenumber_squared(k, spacing))
            axis = float((2.0 / spacing * np.sin(0.5 * k * spacing)) ** 2)
            diagonal = float(
                (
                    2.0
                    * np.sqrt(2.0)
                    / spacing
                    * np.sin(k * spacing / (2.0 * np.sqrt(2.0)))
                )
                ** 2
            )
            errors.append(abs(2.0 * shifted - axis - diagonal) / k**2)
            ratios.append(shifted / k**2)
    small_spacing = 1.0e-12
    small_k = k0 * float(physics["n_glass"])
    continuum_error = abs(
        float(adc5_shifted_wavenumber_squared(small_k, small_spacing))
        / small_k**2
        - 1.0
    )
    return {
        "midpoint_identity_max_relative_error": float(max(errors)),
        "continuum_limit_relative_error": float(continuum_error),
        "shifted_to_physical_k2_ratio_min": float(min(ratios)),
        "shifted_to_physical_k2_ratio_max": float(max(ratios)),
        "positive": bool(min(ratios) > 0.0),
        "all_finite": bool(np.all(np.isfinite([*errors, *ratios, continuum_error]))),
    }


def _probe_solve_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    probe = config["probe_grid"]
    physics = config["physics"]
    grid = make_axisymmetric_grid(
        dr_m=float(probe["dr_m"]),
        dz_m=float(probe["dz_m"]),
        radial_core_max_m=float(probe["radial_core_max_m"]),
        z_core_min_m=float(probe["z_core_min_m"]),
        z_core_max_m=float(probe["z_core_max_m"]),
        pml_thickness_m=float(probe["pml_thickness_m"]),
    )
    expected = [
        int(probe["expected_nr"]),
        int(probe["expected_nz"]),
        int(probe["expected_unknowns"]),
    ]
    if [grid.nr, grid.nz, grid.unknown_count] != expected:
        raise RuntimeError("R11 probe grid differs from registration")
    pml = make_cylindrical_pml(
        grid,
        wavelength_m=float(physics["wavelength_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        polynomial_order=int(probe["pml_polynomial_order"]),
        target_one_way_amplitude=float(probe["pml_target_one_way_amplitude"]),
    )
    n2 = make_background_n2(
        grid,
        interface_z_m=float(probe["interface_z_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
    )
    k0 = 2.0 * np.pi / float(physics["wavelength_m"])
    corrected_n2 = adc5_shifted_wavenumber_squared(
        k0 * np.sqrt(n2), grid.dr_m
    ) / k0**2
    matrix, matrix_controls = assemble_cylindrical_helmholtz(
        grid,
        pml,
        corrected_n2,
        wavelength_m=float(physics["wavelength_m"]),
    )
    manufactured = make_manufactured_vector(grid)
    rhs = matrix @ manufactured
    solution, solver_controls = solve_sparse_direct(
        matrix, rhs, permc_spec=str(probe["solver_permc_spec"])
    )
    recovery_error = _relative_l2(solution, manufactured)
    return {
        "grid": {
            "nr": grid.nr,
            "nz": grid.nz,
            "unknown_count": grid.unknown_count,
        },
        "matrix_controls": matrix_controls,
        "solver_controls": solver_controls,
        "manufactured_recovery_relative_l2": recovery_error,
        "all_finite": bool(
            matrix_controls["finite_data"]
            and solver_controls["all_finite"]
            and np.isfinite(recovery_error)
        ),
    }


def _geometry_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    registered = config["geometry_control"]
    rows: list[dict[str, Any]] = []
    for shape_values, spacing in zip(
        registered["shapes"], registered["dx_m"], strict=True
    ):
        shape = tuple(int(value) for value in shape_values)
        dx_m = float(spacing)
        for diameter in registered["diameters_m"]:
            formal = make_tgv_air_fraction_slice_chord_quadrature(
                shape,
                dx_m,
                float(diameter),
                int(registered["formal_order"]),
            )
            validation = make_tgv_air_fraction_slice_chord_quadrature(
                shape,
                dx_m,
                float(diameter),
                int(registered["validation_order"]),
            )
            expected_area = np.pi * (0.5 * float(diameter)) ** 2
            area = float(np.sum(formal) * dx_m**2)
            rows.append(
                {
                    "shape": list(shape),
                    "dx_m": dx_m,
                    "diameter_m": float(diameter),
                    "order_relative_l2": _relative_l2(formal, validation),
                    "area_relative_error": float(
                        abs(area - expected_area) / expected_area
                    ),
                    "fraction_min": float(np.min(formal)),
                    "fraction_max": float(np.max(formal)),
                    "all_finite": bool(np.all(np.isfinite(formal))),
                }
            )
    return {
        "cases": rows,
        "maximum_order_relative_l2": float(
            max(row["order_relative_l2"] for row in rows)
        ),
        "maximum_area_relative_error": float(
            max(row["area_relative_error"] for row in rows)
        ),
        "fraction_bound_error": float(
            max(
                0.0,
                max(-row["fraction_min"] for row in rows),
                max(row["fraction_max"] - 1.0 for row in rows),
            )
        ),
        "all_finite": bool(all(row["all_finite"] for row in rows)),
    }


def _polar_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    registered = config["polar_control"]
    radius = (
        np.arange(int(registered["radius_count"]), dtype=np.float64) + 0.5
    ) * float(registered["radius_spacing_m"])
    theta = (
        2.0
        * np.pi
        * np.arange(int(registered["theta_count"]), dtype=np.float64)
        / int(registered["theta_count"])
    )
    frequency = float(registered["radial_test_frequency_cycles_per_m"])
    rows: list[dict[str, Any]] = []
    for shape_values, spacing in zip(
        registered["shapes"], registered["dx_m"], strict=True
    ):
        shape = tuple(int(value) for value in shape_values)
        dx_m = float(spacing)
        y = (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * dx_m
        x = (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * dx_m
        radial = np.hypot(y[:, None], x[None, :])
        field = j0(2.0 * np.pi * frequency * radial).astype(np.complex128)
        controls, _ = cartesian_polar_angular_diagnostics(
            field,
            dx_m=dx_m,
            radius_m=radius,
            theta_rad=theta,
            interpolation_order=int(registered["interpolation_order"]),
        )
        rows.append({"shape": list(shape), "dx_m": dx_m, **controls})
    return {
        "cases": rows,
        "maximum_angular_relative_l2": float(
            max(row["angular_relative_l2"] for row in rows)
        ),
        "all_finite": bool(all(row["all_finite"] for row in rows)),
    }


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    expected = set(str(value) for value in config["output"]["required_files"])
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"R11 preflight artifact set differs: {sorted(actual)}")
    for filename in (
        "metadata.json",
        "metrics.json",
        "run_state.json",
        "run_progress.json",
    ):
        with (run_dir / filename).open("r", encoding="utf-8") as handle:
            json.load(handle)
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as h5:
        if set(h5["entry"]) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
        }:
            raise RuntimeError("R11 preflight HDF5 layout differs")
        if len(h5["entry/data"]) != 0:
            raise RuntimeError("R11 preflight data group must be empty")
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            external_metrics = json.load(handle)
        with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            external_metadata = json.load(handle)
        stored_metrics = _decode_hdf5_sequences(
            _hdf5_to_plain(h5["entry/metrics"])
        )
        stored_metadata = _decode_hdf5_sequences(
            _hdf5_to_plain(h5["entry/metadata"])
        )
        if stored_metrics != external_metrics:
            raise RuntimeError("R11 preflight HDF5 metrics round-trip differs")
        if stored_metadata != external_metadata:
            raise RuntimeError("R11 preflight HDF5 metadata round-trip differs")


def run(config_path: Path) -> Path:
    """Execute the registered non-scientific R11 preflight."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R11 preflight source config hash differs")
    config = load_config(source)
    validate_preflight_config(config)
    _validate_provenance(config)
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
            "scientific_result": False,
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
        },
    )
    progress = _progress_writer(run_dir / "run_progress.json")
    progress("preflight_started", {"source_config_sha256": REGISTERED_CONFIG_SHA256})
    started = time.perf_counter()
    try:
        grid_controls = _formal_grid_controls(config)
        resource = _resource_controls(config)
        progress("resource_controls_completed", resource)
        adc = _adc_algebra_controls(config)
        probe = _probe_solve_controls(config)
        progress(
            "probe_solve_completed",
            {
                "relative_residual": probe["solver_controls"][
                    "relative_residual"
                ],
                "recovery_relative_l2": probe[
                    "manufactured_recovery_relative_l2"
                ],
            },
        )
        geometry = _geometry_controls(config)
        progress(
            "geometry_controls_completed",
            {
                "maximum_order_relative_l2": geometry[
                    "maximum_order_relative_l2"
                ],
                "maximum_area_relative_error": geometry[
                    "maximum_area_relative_error"
                ],
            },
        )
        polar = _polar_controls(config)
        progress(
            "polar_controls_completed",
            {
                "maximum_angular_relative_l2": polar[
                    "maximum_angular_relative_l2"
                ]
            },
        )
        thresholds = config["thresholds"]
        algebra_error = max(
            float(adc["midpoint_identity_max_relative_error"]),
            float(probe["matrix_controls"]["complex_symmetric_max_abs_error"]),
            float(geometry["fraction_bound_error"]),
        )
        control_pass = {
            "formal_grid_counts": bool(
                all(row["expected_match"] for row in grid_controls)
            ),
            "resource_total_memory": bool(resource["total_memory_pass"]),
            "resource_peak_fraction": bool(
                resource["estimated_peak_fraction_pass"]
            ),
            "resource_disk": bool(resource["disk_free_pass"]),
            "adc_algebra": bool(
                adc["positive"]
                and adc["all_finite"]
                and algebra_error
                <= float(thresholds["algebra_absolute_or_relative_max"])
            ),
            "probe_residual": bool(
                float(probe["solver_controls"]["relative_residual"])
                <= float(thresholds["solve_relative_residual_max"])
            ),
            "probe_recovery": bool(
                float(probe["manufactured_recovery_relative_l2"])
                <= float(thresholds["manufactured_recovery_relative_l2_max"])
            ),
            "geometry_order": bool(
                float(geometry["maximum_order_relative_l2"])
                <= float(thresholds["geometry_order_relative_l2_max"])
            ),
            "geometry_area": bool(
                float(geometry["maximum_area_relative_error"])
                <= float(thresholds["geometry_area_relative_error_max"])
            ),
            "polar_manufactured": bool(
                float(polar["maximum_angular_relative_l2"])
                <= float(
                    thresholds["polar_manufactured_angular_relative_l2_max"]
                )
            ),
            "all_finite": bool(
                adc["all_finite"]
                and probe["all_finite"]
                and geometry["all_finite"]
                and polar["all_finite"]
            ),
        }
        hard_pass = bool(all(control_pass.values()))
        status = "Passed" if hard_pass else "Blocked"
        interpretation = (
            "r11_formal_preflight_passed"
            if hard_pass
            else "r11_formal_preflight_failed"
        )
        metrics = {
            "version": "R11_preflight",
            "scientific_result": False,
            "provenance": dict(config["provenance"]),
            "formal_grid_controls": grid_controls,
            "resource_controls": resource,
            "adc5_algebra_controls": adc,
            "probe_solve_controls": probe,
            "geometry_controls": geometry,
            "polar_controls": polar,
            "maximum_algebra_error": float(algebra_error),
            "control_pass": control_pass,
            "hard_controls_pass": hard_pass,
            "formal_r11_allowed": hard_pass,
            "thresholds": dict(thresholds),
            "total_execution_elapsed_s": float(time.perf_counter() - started),
            "status": status,
            "interpretation_code": interpretation,
        }
        metadata = {
            "experiment_id": "exp040",
            "diagnostic_stage": "R11_preflight",
            "scientific_result": False,
            "run_path": str(run_dir.resolve()),
            "source_config": str(source),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
            "created_at": created_at_utc(),
            "git_commit": get_git_commit(PROJECT_ROOT) or "unavailable",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "status": status,
            "interpretation_code": interpretation,
            "formal_r11_allowed": hard_pass,
        }
        save_json(run_dir / "metrics.json", metrics)
        save_json(run_dir / "metadata.json", metadata)
        save_ptycho_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            instrument=_hdf5_safe(
                {
                    "wavelength_m": float(config["physics"]["wavelength_m"]),
                    "formal_grid_controls": grid_controls,
                    "resource_controls": resource,
                }
            ),
            config_yaml=config_to_yaml(dict(config)),
            metadata=_hdf5_safe(metadata),
            metrics=_hdf5_safe(metrics),
        )
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "completed_at": created_at_utc(),
                "preflight_status": status,
                "interpretation_code": interpretation,
                "formal_r11_allowed": hard_pass,
                "scientific_result": False,
                "artifacts_validated": False,
            },
        )
        progress(
            "artifacts_written",
            {"preflight_status": status, "formal_r11_allowed": hard_pass},
        )
        _validate_artifacts(run_dir, config)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "completed_at": created_at_utc(),
                "preflight_status": status,
                "interpretation_code": interpretation,
                "formal_r11_allowed": hard_pass,
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
                "formal_r11_allowed": False,
            },
        )
        raise

    print(f"run_dir: {run_dir.resolve()}", flush=True)
    print(f"preflight_status: {status}", flush=True)
    print(f"interpretation: {interpretation}", flush=True)
    print(f"formal_r11_allowed: {hard_pass}", flush=True)
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
