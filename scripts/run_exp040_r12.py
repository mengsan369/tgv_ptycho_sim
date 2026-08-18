"""Run the pre-registered exp040 R12 numerical reference-closure experiment."""

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
    project_field_to_passband,
)
from tgv_ptycho.forward.helmholtz_axisymmetric import (  # noqa: E402
    adc5_shifted_wavenumber_squared,
    annular_mean_from_cartesian,
    assemble_cylindrical_helmholtz,
    cartesian_polar_angular_diagnostics,
    make_axisymmetric_grid,
    make_background_n2,
    make_contrast_source,
    make_cylindrical_pml,
    make_tgv_n2_cell_average,
    observation_trace,
    outer_guard_rms_ratio,
    radial_trace_to_cartesian,
    radial_weighted_relative_l2,
    scalar_interface_background,
    solve_sparse_direct,
)
from tgv_ptycho.forward.helmholtz_axisymmetric_fem import (  # noqa: E402
    assemble_axisymmetric_weak_form,
    evaluate_fem_field,
    make_axisymmetric_fem_grid,
    make_tgv_scattered_fem_evaluator,
)
from tgv_ptycho.forward.multislice_A import (  # noqa: E402
    multislice_propagate_streamed_A,
)
from tgv_ptycho.forward.multislice_radial import (  # noqa: E402
    radial_multislice_contrast_propagate,
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
from tgv_ptycho.objects.tgv_geometry import (  # noqa: E402
    diameter_profile,
    midpoint_z_grid,
)
from tgv_ptycho.optics.angular_spectrum import (  # noqa: E402
    angular_spectrum_propagate,
)
from tgv_ptycho.optics.hankel import (  # noqa: E402
    make_qdht_plan,
    qdht_plan_controls,
)
from tgv_ptycho.viz.plot_exp040_r12 import (  # noqa: E402
    EXP040_R12_FIGURE_FILENAMES,
    save_exp040_r12_figures,
)

REGISTERED_CONFIG_SHA256 = (
    "6BE3E84867E65F92AC3D74AE7D3CC02C9C062150CB2C66E0A6D74AD127B54523"
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


def _relative_l2(test: np.ndarray, reference: np.ndarray) -> float:
    left = np.asarray(test)
    right = np.asarray(reference)
    if left.shape != right.shape:
        raise ValueError("relative-L2 arrays must have matching shapes.")
    denominator = max(float(np.linalg.norm(right)), np.finfo(float).eps)
    return float(np.linalg.norm(left - right) / denominator)


def _require_exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(f"R12 {name} differs from registration.")


def validate_r12_config(config: Mapping[str, Any]) -> None:
    """Validate all result-controlling R12 settings."""

    _require_exact(config["experiment"]["id"], "exp040", "experiment id")
    _require_exact(config["experiment"]["stage"], "R12", "stage")
    _require_exact(
        config["experiment"]["scientific_result"], True, "scientific role"
    )
    _require_exact(
        config["domain"]["fixed_case_order"],
        ["adc_fine_core60"],
        "domain case order",
    )
    _require_exact(
        config["fem"]["fixed_case_order"],
        ["fem_p2_core60", "fem_p3_core60"],
        "FEM case order",
    )
    _require_exact(
        config["multislice"]["fixed_case_order"],
        [
            "chord_fov96_standard",
            "chord_fov128_standard",
            "chord_fov128_alias",
        ],
        "Cartesian case order",
    )
    _require_exact(
        dict(config["conditional_execution"]),
        {
            "cross_model_requires": [
                "domain_gate_pass",
                "fem_mesh_gate_pass",
                "cartesian_reference_gate_pass",
            ],
            "vector_model_enabled": False,
        },
        "conditional execution",
    )
    _require_exact(
        list(config["output"]["figure_filenames"]),
        list(EXP040_R12_FIGURE_FILENAMES),
        "figure filenames",
    )
    _require_exact(
        float(config["thresholds"]["domain_passband_relative_l2_max"]),
        0.05,
        "domain threshold",
    )
    _require_exact(
        float(config["thresholds"]["mesh_passband_relative_l2_max"]),
        0.05,
        "mesh threshold",
    )
    _require_exact(
        float(config["thresholds"]["polar_angular_relative_l2_max"]),
        0.05,
        "angular threshold",
    )


def _emit(
    callback: ProgressCallback | None, event: str, **details: Any
) -> None:
    if callback is not None:
        callback(event, details)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _save_radial_checkpoint(path: Path, result: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        radius_m=np.asarray(result["radius_m"]),
        normalized_total_trace=np.asarray(result["normalized_total_trace"]),
        normalized_scattered_trace=np.asarray(
            result["normalized_scattered_trace"]
        ),
        controls_json=json.dumps(
            result["controls"], sort_keys=True, default=_json_default
        ),
    )


def _save_cartesian_checkpoint(path: Path, result: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        normalized_native_field=np.asarray(result["normalized_native_field"]),
        controls_json=json.dumps(
            result["controls"], sort_keys=True, default=_json_default
        ),
    )


def _load_r11_checkpoint(
    path: Path, expected_hash: str, kind: str
) -> dict[str, Any]:
    if _sha256(path) != expected_hash:
        raise RuntimeError(f"R11 {kind} checkpoint hash differs.")
    with np.load(path) as data:
        controls = json.loads(str(data["controls_json"]))
        if kind == "radial":
            return {
                "radius_m": np.asarray(data["radius_m"]),
                "normalized_total_trace": np.asarray(
                    data["normalized_total_trace"]
                ),
                "normalized_scattered_trace": np.asarray(
                    data["normalized_scattered_trace"]
                ),
                "controls": controls,
            }
        return {
            "normalized_native_field": np.asarray(
                data["normalized_native_field"]
            ),
            "controls": controls,
        }


def _effective_adc5_n2(
    physical_n2: np.ndarray, *, spacing_m: float, wavelength_m: float
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(physical_n2, dtype=np.float64)
    k0 = 2.0 * np.pi / float(wavelength_m)
    effective = adc5_shifted_wavenumber_squared(
        k0 * np.sqrt(values), spacing_m
    ) / k0**2
    ratio = effective / values
    return effective, {
        "method": "adc5",
        "effective_to_physical_n2_ratio_min": float(np.min(ratio)),
        "effective_to_physical_n2_ratio_max": float(np.max(ratio)),
        "all_finite": bool(np.all(np.isfinite(effective))),
        "positive": bool(np.all(effective > 0.0)),
    }


def _solve_adc_core60(
    config: Mapping[str, Any], callback: ProgressCallback | None
) -> dict[str, Any]:
    physics = config["physics"]
    registered = config["domain"]
    case = registered["cases"]["adc_fine_core60"]
    _emit(
        callback,
        "domain_case_started",
        case_id="adc_fine_core60",
        expected_unknowns=int(case["expected_unknowns"]),
    )
    started = time.perf_counter()
    grid = make_axisymmetric_grid(
        dr_m=float(case["dr_m"]),
        dz_m=float(case["dz_m"]),
        radial_core_max_m=float(case["radial_core_max_m"]),
        z_core_min_m=float(registered["z_core_min_m"]),
        z_core_max_m=float(registered["z_core_max_m"]),
        pml_thickness_m=float(registered["pml_thickness_m"]),
    )
    actual = [grid.nr, grid.nz, grid.unknown_count]
    expected = [
        int(case["expected_nr"]),
        int(case["expected_nz"]),
        int(case["expected_unknowns"]),
    ]
    if actual != expected:
        raise RuntimeError("R12 core60 finite-volume grid differs.")
    pml_config = registered["pml"]
    pml = make_cylindrical_pml(
        grid,
        wavelength_m=float(physics["wavelength_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        polynomial_order=int(pml_config["polynomial_order"]),
        target_one_way_amplitude=float(
            pml_config["target_one_way_amplitude"]
        ),
    )
    background_physical = make_background_n2(
        grid,
        interface_z_m=float(physics["background_interface_z_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
    )
    tgv_physical, material_controls = make_tgv_n2_cell_average(
        grid,
        thickness_m=float(physics["sample_thickness_m"]),
        d_top_m=float(physics["d_top_m"]),
        d_waist_m=float(physics["d_waist_m"]),
        d_bottom_m=float(physics["d_bottom_m"]),
        z_waist_m=float(physics["z_waist_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        axial_subnodes=int(registered["axial_material_subnodes"]),
        background_interface_z_m=float(physics["background_interface_z_m"]),
    )
    tgv_effective, tgv_mass = _effective_adc5_n2(
        tgv_physical,
        spacing_m=grid.dr_m,
        wavelength_m=float(physics["wavelength_m"]),
    )
    background_effective, background_mass = _effective_adc5_n2(
        background_physical,
        spacing_m=grid.dr_m,
        wavelength_m=float(physics["wavelength_m"]),
    )
    background_z = scalar_interface_background(
        pml.z_tilde_centers_m,
        physical_z_m=grid.z_centers_m,
        wavelength_m=float(physics["wavelength_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        interface_z_m=float(physics["background_interface_z_m"]),
        incident_amplitude=float(physics["incident_amplitude"]),
    )
    matrix, matrix_controls = assemble_cylindrical_helmholtz(
        grid,
        pml,
        tgv_effective,
        wavelength_m=float(physics["wavelength_m"]),
    )
    source = make_contrast_source(
        grid,
        pml,
        tgv_effective,
        background_effective,
        wavelength_m=float(physics["wavelength_m"]),
        background_field_z=background_z,
    )
    _emit(
        callback,
        "domain_factor_solve_started",
        case_id="adc_fine_core60",
        matrix_nnz=int(matrix.nnz),
    )
    scattered, solver_controls = solve_sparse_direct(
        matrix, source, permc_spec=str(registered["solver"]["permc_spec"])
    )
    observation_z = float(config["observation"]["z_m"])
    scattered_trace, observation_controls = observation_trace(
        scattered, grid, observation_z_m=observation_z
    )
    background_observation = complex(
        scalar_interface_background(
            np.asarray([observation_z], dtype=np.complex128),
            physical_z_m=np.asarray([observation_z]),
            wavelength_m=float(physics["wavelength_m"]),
            n_glass=float(physics["n_glass"]),
            n_air=float(physics["n_air"]),
            interface_z_m=float(physics["background_interface_z_m"]),
            incident_amplitude=float(physics["incident_amplitude"]),
        )[0]
    )
    normalized_scattered = scattered_trace / background_observation
    normalized_total = 1.0 + normalized_scattered
    comparison = config["comparison"]
    guard = outer_guard_rms_ratio(
        normalized_scattered,
        grid.r_centers_m,
        inner_max_radius_m=float(comparison["comparison_max_radius_m"]),
        guard_min_radius_m=float(case["radial_core_max_m"])
        - float(comparison["guard_width_m"]),
        guard_max_radius_m=float(case["radial_core_max_m"]),
    )
    controls = {
        "id": "adc_fine_core60",
        "method": "adc5",
        "nr": grid.nr,
        "nz": grid.nz,
        "unknown_count": grid.unknown_count,
        "dr_m": grid.dr_m,
        "dz_m": grid.dz_m,
        "radial_core_max_m": float(case["radial_core_max_m"]),
        "pml_thickness_m": grid.pml_thickness_m,
        "matrix_controls": matrix_controls,
        "material_controls": material_controls,
        "tgv_mass_controls": tgv_mass,
        "background_mass_controls": background_mass,
        "solver_controls": solver_controls,
        "observation_controls": observation_controls,
        "outer_guard_rms_ratio": float(guard),
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": bool(
            material_controls["all_finite"]
            and tgv_mass["all_finite"]
            and background_mass["all_finite"]
            and solver_controls["all_finite"]
            and np.all(np.isfinite(normalized_total))
        ),
    }
    del (
        background_effective,
        background_physical,
        background_z,
        matrix,
        pml,
        scattered,
        source,
        tgv_effective,
        tgv_physical,
    )
    gc.collect()
    _emit(
        callback,
        "domain_case_completed",
        case_id="adc_fine_core60",
        outer_guard_rms_ratio=float(guard),
        elapsed_s=controls["total_elapsed_s"],
    )
    return {
        "radius_m": grid.r_centers_m.copy(),
        "normalized_total_trace": np.asarray(normalized_total),
        "normalized_scattered_trace": np.asarray(normalized_scattered),
        "controls": controls,
    }


def _solve_fem_case(
    config: Mapping[str, Any], case_id: str, callback: ProgressCallback | None
) -> dict[str, Any]:
    physics = config["physics"]
    registered = config["fem"]
    case = registered["cases"][case_id]
    degree = int(case["degree"])
    _emit(
        callback,
        "fem_case_started",
        case_id=case_id,
        degree=degree,
        expected_unknowns=int(case["expected_unknowns"]),
    )
    started = time.perf_counter()
    grid = make_axisymmetric_fem_grid(
        degree=degree,
        radial_extent_m=float(registered["radial_extent_m"]),
        z_min_m=float(registered["z_min_m"]),
        z_max_m=float(registered["z_max_m"]),
        radial_element_size_m=float(registered["radial_element_size_m"]),
        axial_element_size_m=float(registered["axial_element_size_m"]),
    )
    if grid.active_unknown_count != int(case["expected_unknowns"]):
        raise RuntimeError(f"R12 FEM grid differs for {case_id}.")
    evaluator = make_tgv_scattered_fem_evaluator(
        wavelength_m=float(physics["wavelength_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        incident_amplitude=float(physics["incident_amplitude"]),
        background_interface_z_m=float(physics["background_interface_z_m"]),
        sample_thickness_m=float(physics["sample_thickness_m"]),
        d_top_m=float(physics["d_top_m"]),
        d_waist_m=float(physics["d_waist_m"]),
        d_bottom_m=float(physics["d_bottom_m"]),
        z_waist_m=float(physics["z_waist_m"]),
        radial_core_max_m=float(registered["radial_core_max_m"]),
        z_core_min_m=float(registered["z_core_min_m"]),
        z_core_max_m=float(registered["z_core_max_m"]),
        pml_thickness_m=float(registered["pml_thickness_m"]),
        pml_polynomial_order=int(registered["pml"]["polynomial_order"]),
        pml_target_one_way_amplitude=float(
            registered["pml"]["target_one_way_amplitude"]
        ),
    )
    matrix, rhs, assembly_controls = assemble_axisymmetric_weak_form(
        grid,
        evaluator,
        quadrature_order=int(registered["quadrature_order"]),
    )
    _emit(
        callback,
        "fem_factor_solve_started",
        case_id=case_id,
        matrix_nnz=int(matrix.nnz),
    )
    scattered, solver_controls = solve_sparse_direct(
        matrix, rhs, permc_spec=str(registered["solver"]["permc_spec"])
    )
    comparison = config["comparison"]
    trace_spacing = float(comparison["trace_sampling_m"])
    trace_count = int(
        np.rint(float(registered["radial_core_max_m"]) / trace_spacing)
    )
    radius = (np.arange(trace_count, dtype=np.float64) + 0.5) * trace_spacing
    observation_z = float(config["observation"]["z_m"])
    scattered_trace = evaluate_fem_field(
        scattered, grid, radius, observation_z
    )
    background_observation = complex(
        scalar_interface_background(
            np.asarray([observation_z], dtype=np.complex128),
            physical_z_m=np.asarray([observation_z]),
            wavelength_m=float(physics["wavelength_m"]),
            n_glass=float(physics["n_glass"]),
            n_air=float(physics["n_air"]),
            interface_z_m=float(physics["background_interface_z_m"]),
            incident_amplitude=float(physics["incident_amplitude"]),
        )[0]
    )
    normalized_scattered = scattered_trace / background_observation
    normalized_total = 1.0 + normalized_scattered
    guard = outer_guard_rms_ratio(
        normalized_scattered,
        radius,
        inner_max_radius_m=float(comparison["comparison_max_radius_m"]),
        guard_min_radius_m=float(registered["radial_core_max_m"])
        - float(comparison["guard_width_m"]),
        guard_max_radius_m=float(registered["radial_core_max_m"]),
    )
    controls = {
        "id": case_id,
        "method": "continuous_tensor_product_weak_form_fem",
        "degree": degree,
        "active_unknown_count": grid.active_unknown_count,
        "radial_element_count": grid.radial_element_count,
        "axial_element_count": grid.axial_element_count,
        "quadrature_order": int(registered["quadrature_order"]),
        "assembly_controls": assembly_controls,
        "solver_controls": solver_controls,
        "outer_guard_rms_ratio": float(guard),
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": bool(
            assembly_controls["matrix_rhs_all_finite"]
            and solver_controls["all_finite"]
            and np.all(np.isfinite(normalized_total))
        ),
    }
    del matrix, rhs, scattered
    gc.collect()
    _emit(
        callback,
        "fem_case_completed",
        case_id=case_id,
        outer_guard_rms_ratio=float(guard),
        elapsed_s=controls["total_elapsed_s"],
    )
    return {
        "radius_m": radius,
        "normalized_total_trace": np.asarray(normalized_total),
        "normalized_scattered_trace": np.asarray(normalized_scattered),
        "controls": controls,
    }


def _run_cartesian_case(
    config: Mapping[str, Any], case_id: str, callback: ProgressCallback | None
) -> dict[str, Any]:
    physics = config["physics"]
    registered = config["multislice"]
    case = registered["cases"][case_id]
    shape = tuple(int(value) for value in case["shape"])
    dx_m = float(case["dx_m"])
    alias_control = bool(case["alias_control"])
    _emit(
        callback,
        "cartesian_case_started",
        case_id=case_id,
        shape=list(shape),
        alias_control=alias_control,
    )
    started = time.perf_counter()
    z_m, widths = midpoint_z_grid(
        float(physics["sample_thickness_m"]), float(registered["dz_m"])
    )
    if widths.size != int(case["expected_slice_count"]):
        raise RuntimeError(f"R12 slice count differs for {case_id}.")
    diameters = diameter_profile(
        z_m,
        float(physics["sample_thickness_m"]),
        float(physics["d_top_m"]),
        float(physics["d_waist_m"]),
        float(physics["d_bottom_m"]),
        float(physics["z_waist_m"]),
    )
    fraction_bound_error = 0.0
    index_bound_error = 0.0
    discrete_volume = 0.0

    def slices():
        nonlocal fraction_bound_error, index_bound_error, discrete_volume
        for index, (diameter, width) in enumerate(
            zip(diameters, widths, strict=True)
        ):
            fraction = make_tgv_air_fraction_slice_chord_quadrature(
                shape,
                dx_m,
                float(diameter),
                int(registered["interface_order"]),
                (0.0, 0.0),
            )
            n_slice = float(physics["n_glass"]) + fraction * (
                float(physics["n_air"]) - float(physics["n_glass"])
            )
            fraction_bound_error = max(
                fraction_bound_error,
                0.0,
                -float(np.min(fraction)),
                float(np.max(fraction)) - 1.0,
            )
            index_bound_error = max(
                index_bound_error,
                0.0,
                float(physics["n_air"]) - float(np.min(n_slice)),
                float(np.max(n_slice)) - float(physics["n_glass"]),
            )
            discrete_volume += float(np.sum(fraction)) * dx_m**2 * float(width)
            if index % 50 == 0 or index + 1 == widths.size:
                _emit(
                    callback,
                    "cartesian_slice_progress",
                    case_id=case_id,
                    completed_slices=index + 1,
                    total_slices=int(widths.size),
                )
            yield n_slice

    incident = np.ones(shape, dtype=np.complex128) * float(
        physics["incident_amplitude"]
    )
    sample_exit = multislice_propagate_streamed_A(
        incident,
        slices(),
        dx_m,
        widths,
        float(physics["wavelength_m"]),
        n_ref=float(registered["internal_reference_index"]),
        bandlimit=True,
        alias_control=alias_control,
    )
    air_exit = angular_spectrum_propagate(
        sample_exit,
        dx_m,
        float(physics["wavelength_m"]),
        float(registered["post_exit_air_distance_m"]),
        n=float(physics["n_air"]),
        bandlimit=True,
        alias_control=alias_control,
    )
    k0 = 2.0 * np.pi / float(physics["wavelength_m"])
    analytic_homogeneous = float(physics["incident_amplitude"]) * np.exp(
        1j
        * k0
        * (
            float(physics["n_glass"])
            * float(physics["sample_thickness_m"])
            + float(physics["n_air"])
            * float(registered["post_exit_air_distance_m"])
        )
    )
    normalized = air_exit / analytic_homogeneous
    continuous_volume = float(
        np.sum(np.pi * (diameters / 2.0) ** 2 * widths)
    )
    controls = {
        "id": case_id,
        "shape": list(shape),
        "dx_m": dx_m,
        "fov_m": [shape[0] * dx_m, shape[1] * dx_m],
        "dz_m": float(registered["dz_m"]),
        "slice_count": int(widths.size),
        "interface_rule": str(registered["interface_rule"]),
        "interface_order": int(registered["interface_order"]),
        "alias_control": alias_control,
        "fraction_bound_error": fraction_bound_error,
        "index_bound_error": index_bound_error,
        "air_volume_relative_error": float(
            abs(discrete_volume - continuous_volume)
            / max(continuous_volume, np.finfo(float).eps)
        ),
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": bool(np.all(np.isfinite(normalized))),
    }
    del air_exit, incident, sample_exit
    gc.collect()
    _emit(
        callback,
        "cartesian_case_completed",
        case_id=case_id,
        elapsed_s=controls["total_elapsed_s"],
    )
    return {"normalized_native_field": np.asarray(normalized), "controls": controls}


def _run_qdht_case(
    config: Mapping[str, Any], callback: ProgressCallback | None
) -> dict[str, Any]:
    physics = config["physics"]
    registered = config["qdht"]
    _emit(callback, "qdht_case_started", case_id="qdht_r64_n512")
    started = time.perf_counter()
    plan = make_qdht_plan(
        int(registered["sample_count"]),
        float(registered["radial_max_m"]),
        order=int(registered["order"]),
    )
    z_m, widths = midpoint_z_grid(
        float(physics["sample_thickness_m"]), float(registered["dz_m"])
    )
    diameters = diameter_profile(
        z_m,
        float(physics["sample_thickness_m"]),
        float(physics["d_top_m"]),
        float(physics["d_waist_m"]),
        float(physics["d_bottom_m"]),
        float(physics["z_waist_m"]),
    )

    def progress(completed: int, total: int) -> None:
        if completed % 50 == 0 or completed == total:
            _emit(
                callback,
                "qdht_slice_progress",
                completed_slices=completed,
                total_slices=total,
            )

    normalized, propagation_controls = radial_multislice_contrast_propagate(
        plan,
        diameters,
        widths,
        wavelength_m=float(physics["wavelength_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        post_exit_air_distance_m=float(registered["post_exit_air_distance_m"]),
        bandlimit=True,
        progress_callback=progress,
    )
    controls = {
        "id": "qdht_r64_n512",
        "plan_controls": qdht_plan_controls(plan),
        "propagation_controls": propagation_controls,
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": bool(
            propagation_controls["all_finite"]
            and np.all(np.isfinite(normalized))
        ),
    }
    _emit(
        callback,
        "qdht_case_completed",
        elapsed_s=controls["total_elapsed_s"],
    )
    return {
        "radius_m": plan.radial_nodes_m.copy(),
        "normalized_total_trace": normalized,
        "normalized_scattered_trace": normalized - 1.0,
        "controls": controls,
    }


def _center_crop(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(field)
    if values.ndim != 2 or any(
        target > current for target, current in zip(shape, values.shape, strict=True)
    ):
        raise ValueError("center crop must fit inside a 2D field.")
    pairs = list(zip(values.shape, shape, strict=True))
    starts = [(current - target) // 2 for current, target in pairs]
    if any((current - target) % 2 for current, target in pairs):
        raise ValueError("center crop requires even symmetric margins.")
    return values[
        starts[0] : starts[0] + shape[0],
        starts[1] : starts[1] + shape[1],
    ]


def _project_with_controls(
    field: np.ndarray, dx_m: float, cutoff: float
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(field, dtype=np.complex128)
    projected = project_field_to_passband(values, dx_m, cutoff)
    repeated = project_field_to_passband(values, dx_m, cutoff)
    idempotent = project_field_to_passband(projected, dx_m, cutoff)
    return projected, {
        "shape": list(values.shape),
        "dx_m": float(dx_m),
        "frequency_spacing_cycles_per_m": [
            1.0 / (values.shape[0] * dx_m),
            1.0 / (values.shape[1] * dx_m),
        ],
        "repeat_relative_l2": _relative_l2(repeated, projected),
        "idempotence_relative_l2": _relative_l2(idempotent, projected),
        "all_finite": bool(np.all(np.isfinite(projected))),
    }


def _radial_profile(
    trace: np.ndarray,
    radius: np.ndarray,
    support: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    comparison = config["comparison"]
    shape = tuple(int(value) for value in comparison["cartesian_shape"])
    dx_m = float(comparison["cartesian_dx_m"])
    cartesian = radial_trace_to_cartesian(
        trace,
        radius,
        shape=shape,
        dx_m=dx_m,
        trace_support_radius_m=support,
        outer_value=1.0 + 0.0j,
    )
    projected, controls = _project_with_controls(
        cartesian,
        dx_m,
        float(comparison["physical_passband_cutoff_cycles_per_m"]),
    )
    radial_radius, radial, counts = annular_mean_from_cartesian(
        projected,
        dx_m=dx_m,
        bin_width_m=float(comparison["annular_bin_width_m"]),
        maximum_radius_m=float(comparison["annular_maximum_radius_m"]),
    )
    return projected, radial_radius, radial, controls | {
        "annular_bin_count_min": int(np.min(counts)),
        "annular_bin_count_max": int(np.max(counts)),
    }


def _postprocess(
    config: Mapping[str, Any],
    core48: Mapping[str, Any],
    core60: Mapping[str, Any],
    fem_results: Mapping[str, Mapping[str, Any]],
    cartesian_results: Mapping[str, Mapping[str, Any]],
    qdht: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    comparison = config["comparison"]
    cutoff = float(comparison["physical_passband_cutoff_cycles_per_m"])
    dx_m = float(comparison["cartesian_dx_m"])
    shape = tuple(int(value) for value in comparison["cartesian_shape"])
    radial_fields: dict[str, np.ndarray] = {}
    radial_profiles: dict[str, np.ndarray] = {}
    radial_projection_controls: dict[str, Any] = {}
    radial_radius: np.ndarray | None = None
    radial_cases = {
        "adc_core48": (core48, 48.0e-6),
        "adc_core60": (core60, 60.0e-6),
        "fem_p2": (fem_results["fem_p2_core60"], 60.0e-6),
        "fem_p3": (fem_results["fem_p3_core60"], 60.0e-6),
        "qdht": (qdht, float(config["qdht"]["radial_max_m"])),
    }
    for name, (result, support) in radial_cases.items():
        cart, radius, profile, controls = _radial_profile(
            np.asarray(result["normalized_total_trace"]),
            np.asarray(result["radius_m"]),
            support,
            config,
        )
        if radial_radius is None:
            radial_radius = radius
        elif not np.array_equal(radial_radius, radius):
            raise RuntimeError("R12 radial bins differ.")
        radial_fields[name] = cart
        radial_profiles[name] = profile
        radial_projection_controls[name] = controls
    assert radial_radius is not None

    baseline = np.asarray(
        cartesian_results["chord_fov64_standard"]["normalized_native_field"]
    )
    cropped = {
        "chord_fov64_standard": baseline,
        "chord_fov96_standard": _center_crop(
            np.asarray(
                cartesian_results["chord_fov96_standard"][
                    "normalized_native_field"
                ]
            ),
            shape,
        ),
        "chord_fov128_standard": _center_crop(
            np.asarray(
                cartesian_results["chord_fov128_standard"][
                    "normalized_native_field"
                ]
            ),
            shape,
        ),
        "chord_fov128_alias": _center_crop(
            np.asarray(
                cartesian_results["chord_fov128_alias"][
                    "normalized_native_field"
                ]
            ),
            shape,
        ),
    }
    cartesian_pass: dict[str, np.ndarray] = {}
    cartesian_projection_controls: dict[str, Any] = {}
    for name, field in cropped.items():
        projected, controls = _project_with_controls(field, dx_m, cutoff)
        cartesian_pass[name] = projected
        cartesian_projection_controls[name] = controls
    polar_config = config["anisotropy"]
    polar_radius = (
        np.arange(int(polar_config["radius_count"]), dtype=np.float64) + 0.5
    ) * float(polar_config["radius_spacing_m"])
    theta = (
        2.0
        * np.pi
        * np.arange(int(polar_config["theta_count"]), dtype=np.float64)
        / int(polar_config["theta_count"])
    )
    polar_controls: dict[str, Any] = {}
    polar_means: dict[str, np.ndarray] = {}
    for name, field in cartesian_pass.items():
        controls, mean = cartesian_polar_angular_diagnostics(
            field,
            dx_m=dx_m,
            radius_m=polar_radius,
            theta_rad=theta,
            interpolation_order=int(polar_config["interpolation_order"]),
            harmonics=tuple(int(value) for value in polar_config["harmonics"]),
        )
        polar_controls[name] = controls
        polar_means[name] = mean
    cartesian_alias_radius, cartesian_alias_radial, _ = annular_mean_from_cartesian(
        cartesian_pass["chord_fov128_alias"],
        dx_m=dx_m,
        bin_width_m=float(comparison["annular_bin_width_m"]),
        maximum_radius_m=float(comparison["annular_maximum_radius_m"]),
    )
    if not np.array_equal(cartesian_alias_radius, radial_radius):
        raise RuntimeError("R12 Cartesian and radial bins differ.")

    domain_pair = radial_weighted_relative_l2(
        radial_profiles["adc_core48"],
        radial_profiles["adc_core60"],
        radial_radius,
    )
    fem_pair = radial_weighted_relative_l2(
        radial_profiles["fem_p2"], radial_profiles["fem_p3"], radial_radius
    )
    adc_to_fem = radial_weighted_relative_l2(
        radial_profiles["adc_core60"], radial_profiles["fem_p3"], radial_radius
    )
    qdht_to_cartesian = radial_weighted_relative_l2(
        radial_profiles["qdht"], cartesian_alias_radial, radial_radius
    )
    fov64_to_96 = _relative_l2(
        cartesian_pass["chord_fov64_standard"],
        cartesian_pass["chord_fov96_standard"],
    )
    fov96_to_128 = _relative_l2(
        cartesian_pass["chord_fov96_standard"],
        cartesian_pass["chord_fov128_standard"],
    )
    alias_difference = _relative_l2(
        cartesian_pass["chord_fov128_standard"],
        cartesian_pass["chord_fov128_alias"],
    )
    projection_max_repeat = max(
        float(value["repeat_relative_l2"])
        for value in (
            list(radial_projection_controls.values())
            + list(cartesian_projection_controls.values())
        )
    )
    projection_max_idempotence = max(
        float(value["idempotence_relative_l2"])
        for value in (
            list(radial_projection_controls.values())
            + list(cartesian_projection_controls.values())
        )
    )
    metrics = {
        "domain": {
            "core48_to_core60_passband_radial_l2": domain_pair,
            "core60_outer_guard_rms_ratio": float(
                core60["controls"]["outer_guard_rms_ratio"]
            ),
        },
        "fem": {
            "p2_to_p3_passband_radial_l2": fem_pair,
            "p3_outer_guard_rms_ratio": float(
                fem_results["fem_p3_core60"]["controls"][
                    "outer_guard_rms_ratio"
                ]
            ),
            "adc5_to_p3_passband_radial_l2_report_only": adc_to_fem,
        },
        "cartesian": {
            "fov64_to_fov96_passband_relative_l2_report_only": fov64_to_96,
            "fov96_to_fov128_passband_relative_l2": fov96_to_128,
            "fov128_standard_to_alias_passband_relative_l2": alias_difference,
            "qdht_to_cartesian_alias_passband_radial_l2_report_only": qdht_to_cartesian,
            "polar_controls": polar_controls,
        },
        "projection_controls": {
            "radial": radial_projection_controls,
            "cartesian": cartesian_projection_controls,
            "maximum_repeat_relative_l2": projection_max_repeat,
            "maximum_idempotence_relative_l2": projection_max_idempotence,
        },
    }
    arrays = {
        "radial_radius_m": radial_radius,
        "cartesian_alias_radial": cartesian_alias_radial,
        "qdht_radial": radial_profiles["qdht"],
        "fem_p3_radial": radial_profiles["fem_p3"],
        "adc_core60_radial": radial_profiles["adc_core60"],
        "fem_p3_passband_cartesian": radial_fields["fem_p3"],
        "cartesian_alias_passband": cartesian_pass["chord_fov128_alias"],
        "polar_radius_m": polar_radius,
        **{f"polar_mean_{key}": value for key, value in polar_means.items()},
    }
    return metrics, arrays


def _hdf5_safe(value: Any) -> Any:
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


def _load_and_validate_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    r11 = PROJECT_ROOT / str(provenance["r11_run"])
    preflight = PROJECT_ROOT / str(provenance["preflight_run"])
    paths = {
        "r11_metrics": r11 / "metrics.json",
        "r11_core48": r11 / "checkpoints" / "adc_fine_core48.npz",
        "r11_chord512": r11 / "checkpoints" / "chord512.npz",
        "preflight_config": preflight / "config.yaml",
        "preflight_metrics": preflight / "metrics.json",
        "preflight_hdf5": preflight
        / "outputs"
        / "exp040_r12_preflight.h5",
    }
    expected = {
        "r11_metrics": str(provenance["r11_metrics_sha256"]),
        "r11_core48": str(provenance["r11_core48_checkpoint_sha256"]),
        "r11_chord512": str(provenance["r11_chord512_checkpoint_sha256"]),
        "preflight_config": str(provenance["preflight_config_sha256"]),
        "preflight_metrics": str(provenance["preflight_metrics_sha256"]),
        "preflight_hdf5": str(provenance["preflight_hdf5_sha256"]),
    }
    hashes = {key: _sha256(path) for key, path in paths.items()}
    if hashes != expected:
        raise RuntimeError("R12 formal provenance hashes differ.")
    with paths["preflight_metrics"].open("r", encoding="utf-8") as handle:
        preflight_metrics = json.load(handle)
    if (
        preflight_metrics.get("version") != "R12_preflight"
        or preflight_metrics.get("status") != "Passed"
        or preflight_metrics.get("formal_r12_allowed") is not True
        or not all(preflight_metrics.get("gates", {}).values())
    ):
        raise RuntimeError("R12 formal preflight did not pass.")
    return {
        "paths": paths,
        "hashes": hashes,
        "preflight_metrics": preflight_metrics,
    }


def _write_hdf5(
    path: Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    qdht: Mapping[str, Any],
) -> None:
    save_ptycho_hdf5(
        path,
        instrument=_hdf5_safe(
            {
                "wavelength_m": float(config["physics"]["wavelength_m"]),
                "sampling": metrics["sampling"],
            }
        ),
        sample=_hdf5_safe(
            {
                "model": "canonical_TGV_A",
                "geometry": dict(config["physics"]),
            }
        ),
        config_yaml=config_to_yaml(dict(config)),
        metadata=_hdf5_safe(dict(metadata)),
        metrics=_hdf5_safe(dict(metrics)),
    )
    with h5py.File(path, "a") as handle:
        data = handle["entry/data"]
        selected = data.require_group("selected_complex_fields")
        selected.create_dataset(
            "fem_p3_passband_cartesian",
            data=np.asarray(arrays["fem_p3_passband_cartesian"]),
        )
        selected.create_dataset(
            "cartesian_alias_passband",
            data=np.asarray(arrays["cartesian_alias_passband"]),
        )
        radial = data.require_group("radial_profiles")
        for key in (
            "radial_radius_m",
            "cartesian_alias_radial",
            "qdht_radial",
            "fem_p3_radial",
            "adc_core60_radial",
        ):
            radial.create_dataset(key, data=np.asarray(arrays[key]))
        polar = data.require_group("polar_means")
        for key, value in arrays.items():
            if key.startswith("polar_mean_"):
                polar.create_dataset(key, data=np.asarray(value))
        qdht_group = data.require_group("qdht_native")
        qdht_group.create_dataset("radius_m", data=np.asarray(qdht["radius_m"]))
        qdht_group.create_dataset(
            "normalized_total_trace",
            data=np.asarray(qdht["normalized_total_trace"]),
        )


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    expected = set(config["output"]["required_files"])
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"R12 artifact set differs: {sorted(actual)}")
    for checkpoint in (
        "adc_fine_core60",
        "fem_p2_core60",
        "fem_p3_core60",
        "chord_fov96_standard",
        "chord_fov128_standard",
        "chord_fov128_alias",
        "qdht_r64_n512",
    ):
        with np.load(run_dir / "checkpoints" / f"{checkpoint}.npz") as data:
            if "controls_json" not in data.files:
                raise RuntimeError(f"R12 checkpoint is incomplete: {checkpoint}")
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as handle:
        entry = handle["entry"]
        if set(entry) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
        }:
            raise RuntimeError("R12 HDF5 entry layout differs.")
        if "truth" in entry or "reconstruction" in entry:
            raise RuntimeError("R12 HDF5 must not claim truth/reconstruction.")
        data = entry["data"]
        if set(data) != {
            "polar_means",
            "qdht_native",
            "radial_profiles",
            "selected_complex_fields",
        }:
            raise RuntimeError("R12 HDF5 data layout differs.")
        for dataset in data["selected_complex_fields"].values():
            values = np.asarray(dataset)
            if values.shape != (512, 512) or not np.all(np.isfinite(values)):
                raise RuntimeError("R12 selected complex field is invalid.")
    for filename in EXP040_R12_FIGURE_FILENAMES:
        image = iio.imread(run_dir / "figures" / filename)
        if image.ndim < 2 or min(image.shape[:2]) <= 10:
            raise RuntimeError(f"R12 figure is invalid: {filename}")


def _progress_writer(path: Path) -> ProgressCallback:
    payload: dict[str, Any] = {
        "purpose": "r12_formal_scientific_execution",
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


def _execute(
    config: Mapping[str, Any], run_dir: Path, callback: ProgressCallback
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    provenance = config["provenance"]
    r11_run = PROJECT_ROOT / str(provenance["r11_run"])
    core48 = _load_r11_checkpoint(
        r11_run / "checkpoints" / "adc_fine_core48.npz",
        str(provenance["r11_core48_checkpoint_sha256"]),
        "radial",
    )
    chord512 = _load_r11_checkpoint(
        r11_run / "checkpoints" / "chord512.npz",
        str(provenance["r11_chord512_checkpoint_sha256"]),
        "cartesian",
    )
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    core60 = _solve_adc_core60(config, callback)
    _save_radial_checkpoint(checkpoints / "adc_fine_core60.npz", core60)
    _emit(callback, "checkpoint_saved", case_id="adc_fine_core60")

    fem_results: dict[str, dict[str, Any]] = {}
    for case_id in config["fem"]["fixed_case_order"]:
        result = _solve_fem_case(config, str(case_id), callback)
        fem_results[str(case_id)] = result
        _save_radial_checkpoint(checkpoints / f"{case_id}.npz", result)
        _emit(callback, "checkpoint_saved", case_id=str(case_id))

    cartesian_results: dict[str, dict[str, Any]] = {
        "chord_fov64_standard": chord512
    }
    for case_id in config["multislice"]["fixed_case_order"]:
        result = _run_cartesian_case(config, str(case_id), callback)
        cartesian_results[str(case_id)] = result
        _save_cartesian_checkpoint(checkpoints / f"{case_id}.npz", result)
        _emit(callback, "checkpoint_saved", case_id=str(case_id))

    qdht = _run_qdht_case(config, callback)
    _save_radial_checkpoint(checkpoints / "qdht_r64_n512.npz", qdht)
    _emit(callback, "checkpoint_saved", case_id="qdht_r64_n512")

    post, arrays = _postprocess(
        config,
        core48,
        core60,
        fem_results,
        cartesian_results,
        qdht,
    )
    thresholds = config["thresholds"]
    case_controls = {
        "adc_fine_core60": core60["controls"],
        **{key: value["controls"] for key, value in fem_results.items()},
        **{
            key: value["controls"]
            for key, value in cartesian_results.items()
        },
        "qdht_r64_n512": qdht["controls"],
    }
    solver_controls = [
        core60["controls"]["solver_controls"],
        *[
            fem_results[key]["controls"]["solver_controls"]
            for key in config["fem"]["fixed_case_order"]
        ],
    ]
    maximum_solver_residual = max(
        float(value["relative_residual"]) for value in solver_controls
    )
    all_finite = bool(
        all(value["all_finite"] for value in case_controls.values())
        and all(np.all(np.isfinite(value)) for value in arrays.values())
    )
    projection_algebra = max(
        float(post["projection_controls"]["maximum_repeat_relative_l2"]),
        float(post["projection_controls"]["maximum_idempotence_relative_l2"]),
    )
    hard_controls = {
        "preflight_pass": True,
        "maximum_solver_relative_residual": maximum_solver_residual,
        "solver_pass": bool(
            maximum_solver_residual
            <= float(thresholds["solve_relative_residual_max"])
        ),
        "projection_algebra_max_relative_error": projection_algebra,
        "projection_algebra_pass": bool(
            projection_algebra
            <= float(thresholds["algebra_absolute_or_relative_max"])
        ),
        "all_finite": all_finite,
    }
    hard_pass = bool(
        hard_controls["solver_pass"]
        and hard_controls["projection_algebra_pass"]
        and hard_controls["all_finite"]
    )
    gates = {
        "domain_gate_pass": bool(
            hard_pass
            and post["domain"]["core48_to_core60_passband_radial_l2"]
            <= float(thresholds["domain_passband_relative_l2_max"])
            and post["domain"]["core60_outer_guard_rms_ratio"]
            <= float(thresholds["outer_guard_rms_ratio_max"])
        ),
        "fem_mesh_gate_pass": bool(
            hard_pass
            and post["fem"]["p2_to_p3_passband_radial_l2"]
            <= float(thresholds["mesh_passband_relative_l2_max"])
            and post["fem"]["p3_outer_guard_rms_ratio"]
            <= float(thresholds["outer_guard_rms_ratio_max"])
        ),
        "cartesian_reference_gate_pass": bool(
            hard_pass
            and post["cartesian"]["fov96_to_fov128_passband_relative_l2"]
            <= float(thresholds["fov_passband_relative_l2_max"])
            and post["cartesian"]["polar_controls"][
                "chord_fov128_alias"
            ]["angular_relative_l2"]
            <= float(thresholds["polar_angular_relative_l2_max"])
        ),
    }
    reference_validated = all(gates.values())
    if reference_validated:
        cross_l2 = radial_weighted_relative_l2(
            arrays["cartesian_alias_radial"],
            arrays["fem_p3_radial"],
            arrays["radial_radius_m"],
        )
        cross = {
            "executed": True,
            "passband_radial_l2": cross_l2,
            "failed_gates": [],
            "material": bool(
                cross_l2
                > float(thresholds["cross_model_materiality_relative_l2"])
            ),
        }
        arrays["cross_fem_radial"] = arrays["fem_p3_radial"]
        arrays["cross_multislice_radial"] = arrays[
            "cartesian_alias_radial"
        ]
    else:
        cross = {
            "executed": False,
            "numeric_comparison_present": False,
            "failed_gates": [key for key, value in gates.items() if not value],
        }
        arrays["cross_fem_radial"] = np.asarray([], dtype=np.complex128)
        arrays["cross_multislice_radial"] = np.asarray([], dtype=np.complex128)
    if not hard_pass:
        status = "Failed"
        interpretation = "r12_hard_controls_failed"
    elif not reference_validated:
        labels = [
            label
            for key, label in (
                ("domain_gate_pass", "domain"),
                ("fem_mesh_gate_pass", "fem_mesh"),
                ("cartesian_reference_gate_pass", "cartesian"),
            )
            if not gates[key]
        ]
        status = "Failed"
        interpretation = "r12_reference_not_closed__" + "_".join(labels)
    elif bool(cross["material"]):
        status = "Passed"
        interpretation = "r12_scalar_cross_model_difference_resolved"
    else:
        status = "Passed"
        interpretation = "r12_scalar_cross_model_difference_not_resolved"
    metrics = {
        "version": "R12",
        "scientific_result": True,
        "status": status,
        "interpretation_code": interpretation,
        "reference_validated": reference_validated,
        "provenance": dict(config["provenance"]),
        "sampling": {
            "domain_case_order": list(config["domain"]["fixed_case_order"]),
            "fem_case_order": list(config["fem"]["fixed_case_order"]),
            "cartesian_case_order": list(
                config["multislice"]["fixed_case_order"]
            ),
            "cartesian_comparison_shape": list(
                config["comparison"]["cartesian_shape"]
            ),
            "cartesian_dx_m": float(config["comparison"]["cartesian_dx_m"]),
            "comparison_max_radius_m": float(
                config["comparison"]["comparison_max_radius_m"]
            ),
        },
        "case_controls": case_controls,
        "hard_controls": hard_controls,
        "gates": gates,
        "domain": post["domain"],
        "fem": post["fem"],
        "cartesian": post["cartesian"],
        "projection_controls": post["projection_controls"],
        "conditional_cross_model": cross,
        "thresholds": dict(thresholds),
    }
    return metrics, arrays, qdht


def run(config_path: Path) -> Path:
    """Execute the single formal R12 scientific run."""

    source = config_path.resolve()
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R12 source config hash differs.")
    config = load_config(source)
    validate_r12_config(config)
    provenance = _load_and_validate_provenance(config)
    run_dir = make_run_dir(
        PROJECT_ROOT / str(config["run"]["output_root"]),
        str(config["run"]["name"]),
    )
    state_path = run_dir / "run_state.json"
    progress = _progress_writer(run_dir / "run_progress.json")
    started = time.perf_counter()
    try:
        save_json(
            state_path,
            {
                "stage": "R12",
                "state": "running",
                "scientific_result": True,
                "formal_execution_count": 1,
                "created_at": created_at_utc(),
            },
        )
        progress(
            "formal_execution_started",
            {"config_sha256": REGISTERED_CONFIG_SHA256},
        )
        metrics, arrays, qdht = _execute(config, run_dir, progress)
        metrics["total_execution_elapsed_s"] = float(
            time.perf_counter() - started
        )
        metrics["formal_provenance_hashes"] = provenance["hashes"]
        metadata = {
            "created_at": created_at_utc(),
            "experiment_id": "exp040",
            "diagnostic_stage": "R12",
            "scientific_result": True,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit(PROJECT_ROOT),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
        }
        save_config(run_dir / "config.yaml", config)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        _write_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            config,
            metadata,
            metrics,
            arrays,
            qdht,
        )
        save_exp040_r12_figures(run_dir / "figures", metrics, arrays)
        save_json(
            state_path,
            {
                "stage": "R12",
                "state": "completed",
                "scientific_result": True,
                "formal_execution_count": 1,
                "status": metrics["status"],
                "interpretation_code": metrics["interpretation_code"],
                "reference_validated": metrics["reference_validated"],
                "cross_model_executed": metrics["conditional_cross_model"][
                    "executed"
                ],
                "completed_at": created_at_utc(),
            },
        )
        progress(
            "formal_execution_completed",
            {
                "status": metrics["status"],
                "interpretation_code": metrics["interpretation_code"],
            },
        )
        _validate_artifacts(run_dir, config)
    except Exception:
        save_json(
            state_path,
            {
                "stage": "R12",
                "state": "failed_during_execution",
                "scientific_result": True,
                "formal_execution_count": 1,
                "failed_at": created_at_utc(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"run_dir: {run_dir}", flush=True)
    print(f"status: {metrics['status']}", flush=True)
    print(f"interpretation: {metrics['interpretation_code']}", flush=True)
    print(f"reference_validated: {metrics['reference_validated']}", flush=True)
    print(f"gates: {metrics['gates']}", flush=True)
    print(
        "cross_model_executed: "
        f"{metrics['conditional_cross_model']['executed']}",
        flush=True,
    )
    return run_dir


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
