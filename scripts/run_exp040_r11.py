"""Run the pre-registered exp040 R11 reference-closure experiment."""

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
    make_physical_passband_mask,
    project_field_to_passband,
    restrict_aligned_cell_average,
)
from tgv_ptycho.forward.helmholtz_axisymmetric import (  # noqa: E402
    adc5_shifted_wavenumber_squared,
    annular_anisotropy_relative_l2,
    annular_mean_from_cartesian,
    assemble_cylindrical_helmholtz,
    background_interface_controls,
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
from tgv_ptycho.forward.multislice_A import (  # noqa: E402
    multislice_propagate_streamed_A,
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
from tgv_ptycho.viz.plot_exp040_r11 import (  # noqa: E402
    EXP040_R11_FIGURE_FILENAMES,
    save_exp040_r11_figures,
)

REGISTERED_CONFIG_SHA256 = (
    "89B531E75749274F6226BE33A424B0ED7398C920C03B298EB665B2117D90772B"
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
        raise ValueError("relative-L2 arrays must have matching shapes")
    numerator = float(np.sum(np.abs(left - right) ** 2))
    denominator = float(np.sum(np.abs(right) ** 2))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).eps)))


def _require_exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(f"R11 {name} differs from registration.")


def validate_r11_config(config: Mapping[str, Any]) -> None:
    """Validate the result-controlling R11 configuration."""

    _require_exact(
        set(config),
        {
            "run",
            "experiment",
            "provenance",
            "physics",
            "helmholtz",
            "observation",
            "multislice",
            "comparison",
            "anisotropy",
            "thresholds",
            "conditional_execution",
            "output",
        },
        "top-level sections",
    )
    _require_exact(config["experiment"]["id"], "exp040", "experiment id")
    _require_exact(config["experiment"]["stage"], "R11", "stage")
    _require_exact(
        config["experiment"]["scientific_result"], True, "scientific role"
    )
    helmholtz = config["helmholtz"]
    expected_order = [
        "adc_fine_core24",
        "adc_fine_core36",
        "adc_fine_core48",
        "adc_coarse_core48",
        "standard_coarse_core48",
        "standard_fine_core48",
    ]
    _require_exact(
        list(helmholtz["fixed_case_order"]), expected_order, "case order"
    )
    expected_cases = {
        "adc_fine_core24": ("adc5", 2.4e-5, 8.333333333333333e-8, 312, 1296, 404352),
        "adc_fine_core36": ("adc5", 3.6e-5, 8.333333333333333e-8, 456, 1296, 590976),
        "adc_fine_core48": ("adc5", 4.8e-5, 8.333333333333333e-8, 600, 1296, 777600),
        "adc_coarse_core48": ("adc5", 4.8e-5, 1.25e-7, 400, 864, 345600),
        "standard_coarse_core48": ("standard", 4.8e-5, 1.25e-7, 400, 864, 345600),
        "standard_fine_core48": (
            "standard",
            4.8e-5,
            8.333333333333333e-8,
            600,
            1296,
            777600,
        ),
    }
    for case_id, expected in expected_cases.items():
        case = helmholtz["cases"][case_id]
        actual = (
            str(case["method"]),
            float(case["radial_core_max_m"]),
            float(case["dr_m"]),
            int(case["expected_nr"]),
            int(case["expected_nz"]),
            int(case["expected_unknowns"]),
        )
        _require_exact(actual, expected, f"case {case_id}")
        _require_exact(
            float(case["dr_m"]), float(case["dz_m"]), f"square mesh {case_id}"
        )
    _require_exact(
        dict(helmholtz["adc5"]),
        {
            "formula": "midpoint_of_axis_and_diagonal_five_point_symbols",
            "apply_to": "mass_and_contrast_source",
            "posthoc_method_selection": False,
        },
        "ADC5 rule",
    )
    _require_exact(
        list(config["multislice"]["fixed_case_order"]),
        ["chord512", "chord1024"],
        "multislice case order",
    )
    _require_exact(
        int(config["multislice"]["interface_order"]), 64, "interface order"
    )
    _require_exact(
        dict(config["anisotropy"]),
        {
            "radius_count": 160,
            "radius_spacing_m": 1.25e-7,
            "theta_count": 720,
            "interpolation_order": 3,
            "harmonics": [4, 8],
            "legacy_annular_reproduction_target": 0.09549961270718764,
            "legacy_reproduction_absolute_tolerance": 1.0e-12,
        },
        "anisotropy controls",
    )
    _require_exact(
        dict(config["conditional_execution"]),
        {
            "cross_model_requires": [
                "domain_gate_pass",
                "adc5_mesh_gate_pass",
                "cartesian_anisotropy_gate_pass",
            ],
            "vector_model_enabled": False,
        },
        "conditional execution",
    )
    _require_exact(
        list(config["output"]["figure_filenames"]),
        list(EXP040_R11_FIGURE_FILENAMES),
        "figure filenames",
    )
    if any(
        bool(config["output"][name])
        for name in (
            "save_sparse_matrices",
            "save_full_rz_fields",
            "save_slice_volumes",
        )
    ):
        raise ValueError("R11 full matrices/volumes must not be persisted")


def _emit(
    callback: ProgressCallback | None, event: str, **details: Any
) -> None:
    if callback is not None:
        callback(event, details)


def _pml_identity_error(grid, pml) -> float:
    physical_r = grid.r_centers_m < grid.radial_core_max_m
    physical_z = (grid.z_centers_m > grid.z_core_min_m) & (
        grid.z_centers_m < grid.z_core_max_m
    )
    return float(
        max(
            np.max(np.abs(pml.r_stretch_centers[physical_r] - 1.0)),
            np.max(
                np.abs(
                    pml.r_tilde_centers_m[physical_r]
                    - grid.r_centers_m[physical_r]
                )
            ),
            np.max(np.abs(pml.z_stretch_centers[physical_z] - 1.0)),
            np.max(
                np.abs(
                    pml.z_tilde_centers_m[physical_z]
                    - grid.z_centers_m[physical_z]
                )
            ),
            abs(pml.r_tilde_faces_m[0]),
        )
    )


def _effective_n2(
    physical_n2: np.ndarray,
    *,
    method: str,
    spacing_m: float,
    wavelength_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(physical_n2, dtype=np.float64)
    if method == "standard":
        effective = values.copy()
    elif method == "adc5":
        k0 = 2.0 * np.pi / float(wavelength_m)
        effective = (
            adc5_shifted_wavenumber_squared(
                k0 * np.sqrt(values), float(spacing_m)
            )
            / k0**2
        )
    else:
        raise ValueError(f"unsupported R11 Helmholtz method: {method}")
    ratio = effective / values
    return effective, {
        "method": method,
        "effective_to_physical_n2_ratio_min": float(np.min(ratio)),
        "effective_to_physical_n2_ratio_max": float(np.max(ratio)),
        "positive": bool(np.all(effective > 0.0)),
        "all_finite": bool(np.all(np.isfinite(effective))),
    }


def _solve_helmholtz_case(
    config: Mapping[str, Any],
    case_id: str,
    *,
    case_index: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    physics = config["physics"]
    registered = config["helmholtz"]
    case = registered["cases"][case_id]
    _emit(
        progress_callback,
        "helmholtz_case_started",
        case_id=case_id,
        case_index=case_index,
        case_count=len(registered["fixed_case_order"]),
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
    if [grid.nr, grid.nz, grid.unknown_count] != [
        int(case["expected_nr"]),
        int(case["expected_nz"]),
        int(case["expected_unknowns"]),
    ]:
        raise RuntimeError(f"R11 grid differs for {case_id}")
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
    pml_error = _pml_identity_error(grid, pml)
    background_physical_n2 = make_background_n2(
        grid,
        interface_z_m=float(physics["background_interface_z_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
    )
    tgv_physical_n2, material_controls = make_tgv_n2_cell_average(
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
    method = str(case["method"])
    tgv_effective_n2, tgv_mass_controls = _effective_n2(
        tgv_physical_n2,
        method=method,
        spacing_m=grid.dr_m,
        wavelength_m=float(physics["wavelength_m"]),
    )
    background_effective_n2, background_mass_controls = _effective_n2(
        background_physical_n2,
        method=method,
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
    _emit(
        progress_callback,
        "helmholtz_matrix_assembly_started",
        case_id=case_id,
        unknown_count=grid.unknown_count,
    )
    matrix, matrix_controls = assemble_cylindrical_helmholtz(
        grid,
        pml,
        tgv_effective_n2,
        wavelength_m=float(physics["wavelength_m"]),
    )
    source = make_contrast_source(
        grid,
        pml,
        tgv_effective_n2,
        background_effective_n2,
        wavelength_m=float(physics["wavelength_m"]),
        background_field_z=background_z,
    )
    _emit(
        progress_callback,
        "helmholtz_factor_solve_started",
        case_id=case_id,
        matrix_nnz=int(matrix.nnz),
    )
    scattered, solver_controls = solve_sparse_direct(
        matrix,
        source,
        permc_spec=str(registered["solver"]["permc_spec"]),
    )
    _emit(
        progress_callback,
        "helmholtz_factor_solve_completed",
        case_id=case_id,
        relative_residual=float(solver_controls["relative_residual"]),
        factor_and_solve_elapsed_s=float(
            solver_controls["factor_and_solve_elapsed_s"]
        ),
        peak_rss_bytes=int(solver_controls["peak_rss_bytes"]),
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
    core = float(case["radial_core_max_m"])
    guard_ratio = outer_guard_rms_ratio(
        normalized_scattered,
        grid.r_centers_m,
        inner_max_radius_m=float(comparison["comparison_max_radius_m"]),
        guard_min_radius_m=core - float(comparison["guard_width_m"]),
        guard_max_radius_m=core,
    )
    all_finite = bool(
        matrix_controls["finite_data"]
        and material_controls["all_finite"]
        and solver_controls["all_finite"]
        and tgv_mass_controls["all_finite"]
        and background_mass_controls["all_finite"]
        and np.all(np.isfinite(normalized_total))
        and np.all(np.isfinite(normalized_scattered))
        and np.isfinite(guard_ratio)
    )
    controls = {
        "id": case_id,
        "method": method,
        "nr": grid.nr,
        "nz": grid.nz,
        "unknown_count": grid.unknown_count,
        "dr_m": grid.dr_m,
        "dz_m": grid.dz_m,
        "radial_core_max_m": core,
        "pml_thickness_m": grid.pml_thickness_m,
        "pml_physical_core_identity_max_abs_error": pml_error,
        "matrix_controls": matrix_controls,
        "material_controls": material_controls,
        "tgv_mass_controls": tgv_mass_controls,
        "background_mass_controls": background_mass_controls,
        "solver_controls": solver_controls,
        "observation_controls": observation_controls,
        "background_observation_abs": float(abs(background_observation)),
        "source_l2": float(np.sqrt(np.sum(np.abs(source) ** 2))),
        "guard_min_radius_m": core - float(comparison["guard_width_m"]),
        "guard_max_radius_m": core,
        "outer_guard_rms_ratio": guard_ratio,
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": all_finite,
    }
    result = {
        "radius_m": grid.r_centers_m.copy(),
        "normalized_total_trace": np.asarray(normalized_total),
        "normalized_scattered_trace": np.asarray(normalized_scattered),
        "controls": controls,
    }
    del (
        background_effective_n2,
        background_physical_n2,
        background_z,
        matrix,
        pml,
        scattered,
        source,
        tgv_effective_n2,
        tgv_physical_n2,
    )
    gc.collect()
    _emit(
        progress_callback,
        "helmholtz_case_completed",
        case_id=case_id,
        outer_guard_rms_ratio=guard_ratio,
        total_elapsed_s=controls["total_elapsed_s"],
    )
    return result


def _multislice_chord_case(
    config: Mapping[str, Any],
    case_id: str,
    *,
    case_index: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    physics = config["physics"]
    registered = config["multislice"]
    case = registered["cases"][case_id]
    shape = tuple(int(value) for value in case["shape"])
    dx_m = float(case["dx_m"])
    dz_m = float(registered["dz_m"])
    order = int(registered["interface_order"])
    _emit(
        progress_callback,
        "multislice_case_started",
        case_id=case_id,
        case_index=case_index,
        case_count=len(registered["fixed_case_order"]),
        shape=list(shape),
        dx_m=dx_m,
    )
    started = time.perf_counter()
    z_m, widths = midpoint_z_grid(
        float(physics["sample_thickness_m"]), dz_m
    )
    if len(widths) != int(case["expected_slice_count"]):
        raise RuntimeError(f"R11 slice count differs for {case_id}")
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
    all_finite = True

    def slices():
        nonlocal fraction_bound_error
        nonlocal index_bound_error
        nonlocal discrete_volume
        nonlocal all_finite
        for index, (diameter, width) in enumerate(
            zip(diameters, widths, strict=True)
        ):
            fraction = make_tgv_air_fraction_slice_chord_quadrature(
                shape,
                dx_m,
                float(diameter),
                order,
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
            all_finite = bool(
                all_finite
                and np.all(np.isfinite(fraction))
                and np.all(np.isfinite(n_slice))
            )
            if index % 25 == 0 or index + 1 == len(widths):
                _emit(
                    progress_callback,
                    "multislice_slice_progress",
                    case_id=case_id,
                    completed_slices=index + 1,
                    total_slices=len(widths),
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
        bandlimit=bool(registered["internal_bandlimit"]),
    )
    air_exit = angular_spectrum_propagate(
        sample_exit,
        dx_m,
        float(physics["wavelength_m"]),
        float(registered["post_exit_air_distance_m"]),
        n=float(physics["n_air"]),
        bandlimit=bool(registered["post_exit_air_bandlimit"]),
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
    volume_error = float(
        abs(discrete_volume - continuous_volume)
        / max(continuous_volume, np.finfo(float).eps)
    )
    homogeneous_test = angular_spectrum_propagate(
        np.full(
            shape,
            float(physics["incident_amplitude"])
            * np.exp(
                1j
                * k0
                * float(physics["n_glass"])
                * float(physics["sample_thickness_m"])
            ),
            dtype=np.complex128,
        ),
        dx_m,
        float(physics["wavelength_m"]),
        float(registered["post_exit_air_distance_m"]),
        n=float(physics["n_air"]),
        bandlimit=bool(registered["post_exit_air_bandlimit"]),
    )
    homogeneous_error = float(
        np.sqrt(np.sum(np.abs(homogeneous_test - analytic_homogeneous) ** 2))
        / max(
            float(np.sqrt(np.sum(np.abs(homogeneous_test) ** 2))),
            np.finfo(float).eps,
        )
    )
    all_finite = bool(
        all_finite
        and np.all(np.isfinite(normalized))
        and np.all(np.isfinite(homogeneous_test))
    )
    controls = {
        "id": case_id,
        "shape": list(shape),
        "dx_m": dx_m,
        "dz_m": dz_m,
        "slice_count": int(len(widths)),
        "interface_rule": str(registered["interface_rule"]),
        "interface_order": order,
        "fraction_bound_error": fraction_bound_error,
        "index_bound_error": index_bound_error,
        "discrete_air_volume_m3": discrete_volume,
        "continuous_midpoint_air_volume_m3": continuous_volume,
        "air_volume_relative_error": volume_error,
        "homogeneous_control_relative_l2": homogeneous_error,
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": all_finite,
    }
    del air_exit, homogeneous_test, incident, sample_exit
    gc.collect()
    _emit(
        progress_callback,
        "multislice_case_completed",
        case_id=case_id,
        elapsed_s=controls["total_elapsed_s"],
        air_volume_relative_error=volume_error,
    )
    return {"normalized_native_field": np.asarray(normalized), "controls": controls}


def _project_with_controls(
    field: np.ndarray, dx_m: float, cutoff: float
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(field, dtype=np.complex128)
    mask = make_physical_passband_mask(values.shape, dx_m, cutoff)
    projected = project_field_to_passband(values, dx_m, cutoff)
    repeated = project_field_to_passband(values, dx_m, cutoff)
    idempotent = project_field_to_passband(projected, dx_m, cutoff)
    constant = project_field_to_passband(
        np.ones(values.shape, dtype=np.complex128), dx_m, cutoff
    )
    total_energy = float(np.sum(np.abs(values) ** 2))
    retained_energy = float(np.sum(np.abs(projected) ** 2))
    controls = {
        "shape": list(values.shape),
        "dx_m": float(dx_m),
        "frequency_spacing_cycles_per_m": [
            1.0 / (values.shape[0] * dx_m),
            1.0 / (values.shape[1] * dx_m),
        ],
        "nyquist_cycles_per_m": 0.5 / dx_m,
        "mask_true_count": int(np.count_nonzero(mask)),
        "mask_total_count": int(mask.size),
        "mask_fraction": float(np.mean(mask)),
        "retained_energy_fraction": float(
            retained_energy / max(total_energy, np.finfo(float).eps)
        ),
        "repeat_relative_l2": _relative_l2(repeated, projected),
        "idempotence_relative_l2": _relative_l2(idempotent, projected),
        "constant_max_abs_error": float(np.max(np.abs(constant - 1.0))),
        "all_finite": bool(
            np.all(np.isfinite(projected))
            and np.all(np.isfinite(repeated))
            and np.all(np.isfinite(idempotent))
            and np.all(np.isfinite(constant))
        ),
    }
    return projected, controls


def _restriction_controls(
    fine_field: np.ndarray, restricted_field: np.ndarray
) -> dict[str, Any]:
    expected_shape = (fine_field.shape[0] // 2, fine_field.shape[1] // 2)
    fine_mean = complex(np.mean(fine_field, dtype=np.complex128))
    restricted_mean = complex(np.mean(restricted_field, dtype=np.complex128))
    constant = np.full((4, 4), 2.0 - 3.0j, dtype=np.complex128)
    constant_restricted = restrict_aligned_cell_average(constant, 2)
    return {
        "method": "aligned_2x2_complex_cell_average",
        "fine_shape": list(fine_field.shape),
        "restricted_shape": list(restricted_field.shape),
        "expected_restricted_shape": list(expected_shape),
        "shape_matches": bool(restricted_field.shape == expected_shape),
        "constant_max_abs_error": float(
            np.max(np.abs(constant_restricted - (2.0 - 3.0j)))
        ),
        "area_weighted_complex_mean_relative_error": float(
            abs(restricted_mean - fine_mean)
            / max(abs(fine_mean), np.finfo(float).eps)
        ),
        "weights": [0.25, 0.25, 0.25, 0.25],
    }


def _helmholtz_postprocess(
    config: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    comparison = config["comparison"]
    shape = tuple(int(value) for value in comparison["cartesian_shape"])
    dx_m = float(comparison["cartesian_dx_m"])
    cutoff = float(comparison["physical_passband_cutoff_cycles_per_m"])
    bin_width = float(comparison["annular_bin_width_m"])
    maximum = float(comparison["annular_maximum_radius_m"])
    cartesian_raw: dict[str, np.ndarray] = {}
    cartesian_pass: dict[str, np.ndarray] = {}
    radial_raw: dict[str, np.ndarray] = {}
    radial_pass: dict[str, np.ndarray] = {}
    projection_controls: dict[str, dict[str, Any]] = {}
    radius_reference: np.ndarray | None = None
    counts_reference: np.ndarray | None = None
    for case_id in config["helmholtz"]["fixed_case_order"]:
        result = results[str(case_id)]
        core = float(result["controls"]["radial_core_max_m"])
        mapped = radial_trace_to_cartesian(
            result["normalized_total_trace"],
            result["radius_m"],
            shape=shape,
            dx_m=dx_m,
            trace_support_radius_m=core,
            outer_value=complex(comparison["outer_fill_normalized_value"]),
        )
        projected, controls = _project_with_controls(mapped, dx_m, cutoff)
        radii, raw_mean, counts = annular_mean_from_cartesian(
            mapped,
            dx_m=dx_m,
            bin_width_m=bin_width,
            maximum_radius_m=maximum,
        )
        pass_radii, pass_mean, pass_counts = annular_mean_from_cartesian(
            projected,
            dx_m=dx_m,
            bin_width_m=bin_width,
            maximum_radius_m=maximum,
        )
        if not np.array_equal(radii, pass_radii) or not np.array_equal(
            counts, pass_counts
        ):
            raise RuntimeError("R11 Helmholtz annular mapping differs")
        if radius_reference is None:
            radius_reference = radii
            counts_reference = counts
        elif not np.array_equal(radius_reference, radii) or not np.array_equal(
            counts_reference, counts
        ):
            raise RuntimeError("R11 Helmholtz annular bins differ")
        cartesian_raw[str(case_id)] = mapped
        cartesian_pass[str(case_id)] = projected
        radial_raw[str(case_id)] = raw_mean
        radial_pass[str(case_id)] = pass_mean
        projection_controls[str(case_id)] = controls
    if radius_reference is None or counts_reference is None:
        raise RuntimeError("R11 Helmholtz postprocessing is empty")

    def pair(test_id: str, reference_id: str) -> dict[str, Any]:
        return {
            "pair": [test_id, reference_id],
            "denominator": reference_id,
            "raw_radial_l2": radial_weighted_relative_l2(
                radial_raw[test_id], radial_raw[reference_id], radius_reference
            ),
            "passband_radial_l2": radial_weighted_relative_l2(
                radial_pass[test_id], radial_pass[reference_id], radius_reference
            ),
            "raw_cartesian_l2_report_only": _relative_l2(
                cartesian_raw[test_id], cartesian_raw[reference_id]
            ),
            "passband_cartesian_l2_report_only": _relative_l2(
                cartesian_pass[test_id], cartesian_pass[reference_id]
            ),
        }

    comparisons = {
        "domain": {
            "core24_to_core36": pair(
                "adc_fine_core24", "adc_fine_core36"
            ),
            "core36_to_core48": pair(
                "adc_fine_core36", "adc_fine_core48"
            ),
            "core48_outer_guard_rms_ratio": float(
                results["adc_fine_core48"]["controls"][
                    "outer_guard_rms_ratio"
                ]
            ),
        },
        "mesh": {
            "adc5": pair("adc_coarse_core48", "adc_fine_core48"),
            "standard_report_only": pair(
                "standard_coarse_core48", "standard_fine_core48"
            ),
        },
        "method_report_only": pair(
            "standard_fine_core48", "adc_fine_core48"
        ),
    }
    return {
        "cartesian_raw": cartesian_raw,
        "cartesian_pass": cartesian_pass,
        "radial_raw": radial_raw,
        "radial_pass": radial_pass,
        "radial_radius_m": radius_reference,
        "annular_bin_counts": counts_reference,
        "projection_controls": projection_controls,
        "comparisons": comparisons,
    }


def _multislice_postprocess(
    config: Mapping[str, Any],
    chord_results: Mapping[str, Mapping[str, Any]],
    q8_field: np.ndarray,
) -> dict[str, Any]:
    comparison = config["comparison"]
    anisotropy_config = config["anisotropy"]
    cutoff = float(comparison["physical_passband_cutoff_cycles_per_m"])
    chord512 = np.asarray(
        chord_results["chord512"]["normalized_native_field"]
    )
    chord1024 = np.asarray(
        chord_results["chord1024"]["normalized_native_field"]
    )
    q8 = np.asarray(q8_field, dtype=np.complex128)
    q8_pass, q8_projection = _project_with_controls(q8, 6.25e-8, cutoff)
    chord512_pass, chord512_projection = _project_with_controls(
        chord512, 1.25e-7, cutoff
    )
    chord1024_pass, chord1024_projection = _project_with_controls(
        chord1024, 6.25e-8, cutoff
    )
    q8_restricted = restrict_aligned_cell_average(q8_pass, 2)
    chord1024_restricted = restrict_aligned_cell_average(chord1024_pass, 2)
    restriction = _restriction_controls(chord1024_pass, chord1024_restricted)
    legacy, _ = annular_anisotropy_relative_l2(
        q8_restricted,
        dx_m=1.25e-7,
        bin_width_m=float(comparison["annular_bin_width_m"]),
        maximum_radius_m=float(comparison["annular_maximum_radius_m"]),
    )
    legacy_error = abs(
        legacy
        - float(anisotropy_config["legacy_annular_reproduction_target"])
    )
    radius = (
        np.arange(int(anisotropy_config["radius_count"]), dtype=np.float64)
        + 0.5
    ) * float(anisotropy_config["radius_spacing_m"])
    theta = (
        2.0
        * np.pi
        * np.arange(int(anisotropy_config["theta_count"]), dtype=np.float64)
        / int(anisotropy_config["theta_count"])
    )
    fields = {
        "q8_native_1024": (q8_pass, 6.25e-8),
        "q8_restricted_512": (q8_restricted, 1.25e-7),
        "chord512": (chord512_pass, 1.25e-7),
        "chord1024_native": (chord1024_pass, 6.25e-8),
        "chord1024_restricted": (chord1024_restricted, 1.25e-7),
    }
    polar_controls: dict[str, dict[str, Any]] = {}
    polar_means: dict[str, np.ndarray] = {}
    for name, (field, spacing) in fields.items():
        controls, angular_mean = cartesian_polar_angular_diagnostics(
            field,
            dx_m=spacing,
            radius_m=radius,
            theta_rad=theta,
            interpolation_order=int(anisotropy_config["interpolation_order"]),
            harmonics=tuple(int(value) for value in anisotropy_config["harmonics"]),
        )
        polar_controls[name] = controls
        polar_means[name] = angular_mean
    formal_names = (
        "chord512",
        "chord1024_native",
        "chord1024_restricted",
    )
    maximum_formal = float(
        max(
            polar_controls[name]["angular_relative_l2"]
            for name in formal_names
        )
    )
    restriction_increase = float(
        max(
            0.0,
            polar_controls["chord1024_restricted"]["angular_relative_l2"]
            - polar_controls["chord1024_native"]["angular_relative_l2"],
        )
    )
    q8_delta = _relative_l2(q8_pass, chord1024_pass)
    lateral = _relative_l2(chord512_pass, chord1024_restricted)
    normalized_q8_residual = np.abs(q8_pass - chord1024_pass) / max(
        float(np.max(np.abs(chord1024_pass))), np.finfo(float).eps
    )
    return {
        "projection_controls": {
            "q8_1024": q8_projection,
            "chord512": chord512_projection,
            "chord1024": chord1024_projection,
        },
        "restriction_controls": restriction,
        "legacy_annular_residual": float(legacy),
        "legacy_reproduction_absolute_error": float(legacy_error),
        "q8_vs_chord1024_native_passband_relative_l2": q8_delta,
        "chord_lateral_passband_relative_l2": lateral,
        "restriction_angular_increase": restriction_increase,
        "maximum_formal_polar_angular_relative_l2": maximum_formal,
        "polar_controls": polar_controls,
        "polar_radius_m": radius,
        "polar_means": polar_means,
        "fields": {
            "q8_native_passband": q8_pass,
            "q8_restricted_passband": q8_restricted,
            "chord512_passband": chord512_pass,
            "chord1024_native_passband": chord1024_pass,
            "chord1024_restricted_passband": chord1024_restricted,
        },
        "selected_maps": {
            "q8_vs_chord_normalized_residual": normalized_q8_residual,
            "chord1024_passband_amplitude": np.abs(chord1024_pass),
        },
    }


def _conditional_cross_model(
    config: Mapping[str, Any],
    helmholtz: Mapping[str, Any],
    multislice: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute the cross comparator only after all registered gates pass."""

    reference_id = str(config["comparison"]["final_helmholtz_reference"])
    reference = helmholtz["cartesian_pass"][reference_id]
    test = multislice["fields"]["chord1024_restricted_passband"]
    radius, test_radial, _ = annular_mean_from_cartesian(
        test,
        dx_m=float(config["comparison"]["cartesian_dx_m"]),
        bin_width_m=float(config["comparison"]["annular_bin_width_m"]),
        maximum_radius_m=float(config["comparison"]["annular_maximum_radius_m"]),
    )
    reference_radial = helmholtz["radial_pass"][reference_id]
    if not np.array_equal(radius, helmholtz["radial_radius_m"]):
        raise RuntimeError("R11 cross-model radial bins differ")
    metrics = {
        "executed": True,
        "pair": ["chord1024", reference_id],
        "denominator": reference_id,
        "passband_radial_l2": radial_weighted_relative_l2(
            test_radial, reference_radial, radius
        ),
        "passband_cartesian_l2_report_only": _relative_l2(test, reference),
        "alignment": "none",
    }
    selected = {
        "helmholtz_passband_amplitude": np.abs(reference),
        "multislice_passband_amplitude": np.abs(test),
        "normalized_cross_residual": np.abs(test - reference)
        / max(float(np.max(np.abs(reference))), np.finfo(float).eps),
        "cross_phase_difference_rad": np.angle(test * np.conjugate(reference)),
    }
    radial_profiles = {
        "chord1024_passband": test_radial,
        f"{reference_id}_cross_passband": reference_radial,
    }
    return metrics, selected, radial_profiles


def _postprocess_once(
    config: Mapping[str, Any],
    helmholtz_results: Mapping[str, Mapping[str, Any]],
    chord_results: Mapping[str, Mapping[str, Any]],
    q8_field: np.ndarray,
    *,
    hard_controls_prepass: bool,
) -> dict[str, Any]:
    helmholtz = _helmholtz_postprocess(config, helmholtz_results)
    multislice = _multislice_postprocess(config, chord_results, q8_field)
    thresholds = config["thresholds"]
    domain_comparison = helmholtz["comparisons"]["domain"][
        "core36_to_core48"
    ]
    domain_metric_pass = bool(
        float(domain_comparison["passband_radial_l2"])
        <= float(thresholds["domain_passband_relative_l2_max"])
    )
    guard_pass = bool(
        float(
            helmholtz["comparisons"]["domain"][
                "core48_outer_guard_rms_ratio"
            ]
        )
        <= float(thresholds["outer_guard_rms_ratio_max"])
    )
    mesh_metric_pass = bool(
        float(
            helmholtz["comparisons"]["mesh"]["adc5"][
                "passband_radial_l2"
            ]
        )
        <= float(thresholds["mesh_passband_relative_l2_max"])
    )
    legacy_pass = bool(
        float(multislice["legacy_reproduction_absolute_error"])
        <= float(config["anisotropy"]["legacy_reproduction_absolute_tolerance"])
    )
    polar_pass = bool(
        float(multislice["maximum_formal_polar_angular_relative_l2"])
        <= float(thresholds["polar_angular_relative_l2_max"])
    )
    restriction_pass = bool(
        float(multislice["restriction_angular_increase"])
        <= float(thresholds["restriction_angular_increase_max"])
        and bool(multislice["restriction_controls"]["shape_matches"])
    )
    lateral_pass = bool(
        float(multislice["chord_lateral_passband_relative_l2"])
        <= float(thresholds["lateral_passband_relative_l2_max"])
    )
    gates = {
        "domain_metric_pass": domain_metric_pass,
        "outer_guard_pass": guard_pass,
        "domain_gate_pass": bool(
            hard_controls_prepass and domain_metric_pass and guard_pass
        ),
        "adc5_mesh_metric_pass": mesh_metric_pass,
        "adc5_mesh_gate_pass": bool(hard_controls_prepass and mesh_metric_pass),
        "legacy_reproduction_pass": legacy_pass,
        "polar_angular_pass": polar_pass,
        "restriction_pass": restriction_pass,
        "lateral_grid_pass": lateral_pass,
        "cartesian_anisotropy_gate_pass": bool(
            hard_controls_prepass
            and legacy_pass
            and polar_pass
            and restriction_pass
            and lateral_pass
        ),
    }
    cross_allowed = bool(
        gates["domain_gate_pass"]
        and gates["adc5_mesh_gate_pass"]
        and gates["cartesian_anisotropy_gate_pass"]
    )
    selected_maps = dict(multislice["selected_maps"])
    radial_profiles = {
        "radius_m": helmholtz["radial_radius_m"],
        **{
            f"{case_id}_raw": values
            for case_id, values in helmholtz["radial_raw"].items()
        },
        **{
            f"{case_id}_passband": values
            for case_id, values in helmholtz["radial_pass"].items()
        },
    }
    if cross_allowed:
        cross, cross_maps, cross_radial = _conditional_cross_model(
            config, helmholtz, multislice
        )
        selected_maps.update(cross_maps)
        radial_profiles.update(cross_radial)
    else:
        failed = [
            name
            for name in (
                "domain_gate_pass",
                "adc5_mesh_gate_pass",
                "cartesian_anisotropy_gate_pass",
            )
            if not gates[name]
        ]
        cross = {
            "executed": False,
            "reason": "registered_reference_gates_failed",
            "failed_gates": failed,
            "numeric_comparison_present": False,
        }
    return {
        "helmholtz": helmholtz,
        "multislice": multislice,
        "gates": gates,
        "conditional_cross_model": cross,
        "selected_maps": selected_maps,
        "radial_profiles": radial_profiles,
    }


def _postprocessing_repeat_error(
    original: Mapping[str, Any], repeated: Mapping[str, Any]
) -> float:
    errors: list[float] = []
    for key, values in original["selected_maps"].items():
        errors.append(_relative_l2(repeated["selected_maps"][key], values))
    for section, names in (
        ("domain", ("core24_to_core36", "core36_to_core48")),
        ("mesh", ("adc5", "standard_report_only")),
    ):
        left = original["helmholtz"]["comparisons"][section]
        right = repeated["helmholtz"]["comparisons"][section]
        for name in names:
            for metric in ("raw_radial_l2", "passband_radial_l2"):
                denominator = max(abs(float(left[name][metric])), np.finfo(float).eps)
                errors.append(
                    abs(float(right[name][metric]) - float(left[name][metric]))
                    / denominator
                )
    for name, controls in original["multislice"]["polar_controls"].items():
        left = float(controls["angular_relative_l2"])
        right = float(
            repeated["multislice"]["polar_controls"][name][
                "angular_relative_l2"
            ]
        )
        errors.append(abs(right - left) / max(abs(left), np.finfo(float).eps))
    if bool(original["conditional_cross_model"]["executed"]):
        left = float(original["conditional_cross_model"]["passband_radial_l2"])
        right = float(repeated["conditional_cross_model"]["passband_radial_l2"])
        errors.append(abs(right - left) / max(abs(left), np.finfo(float).eps))
    return float(max(errors, default=0.0))


def _checkpoint_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot checkpoint {type(value).__name__}")


def _save_helmholtz_checkpoint(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        radius_m=np.asarray(result["radius_m"]),
        normalized_total_trace=np.asarray(result["normalized_total_trace"]),
        normalized_scattered_trace=np.asarray(
            result["normalized_scattered_trace"]
        ),
        controls_json=json.dumps(
            result["controls"], sort_keys=True, default=_checkpoint_default
        ),
    )


def _save_multislice_checkpoint(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        normalized_native_field=np.asarray(result["normalized_native_field"]),
        controls_json=json.dumps(
            result["controls"], sort_keys=True, default=_checkpoint_default
        ),
    )


def _load_q8_checkpoint(config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    provenance = config["provenance"]
    path = (
        PROJECT_ROOT
        / str(provenance["r10_stage_b_run"])
        / "checkpoints"
        / "multislice_fine_1024.npz"
    )
    if _sha256(path) != str(provenance["r10_q8_checkpoint_sha256"]):
        raise RuntimeError("R10 q8 checkpoint provenance differs")
    with np.load(path) as data:
        field = np.asarray(data["normalized_native_field"], dtype=np.complex128)
        controls = json.loads(str(data["controls_json"]))
    if field.shape != (1024, 1024) or controls.get("interface_factor") != 8:
        raise RuntimeError("R10 q8 checkpoint content differs")
    return field, controls


def _run_r11(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    preflight_metrics: Mapping[str, Any],
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    helmholtz_results: dict[str, dict[str, Any]] = {}
    for index, case_id in enumerate(
        config["helmholtz"]["fixed_case_order"], start=1
    ):
        result = _solve_helmholtz_case(
            config,
            str(case_id),
            case_index=index,
            progress_callback=progress_callback,
        )
        helmholtz_results[str(case_id)] = result
        checkpoint = run_dir / "checkpoints" / f"{case_id}.npz"
        _save_helmholtz_checkpoint(checkpoint, result)
        _emit(
            progress_callback,
            "checkpoint_saved",
            case_id=str(case_id),
            path=checkpoint.relative_to(run_dir).as_posix(),
            sha256=_sha256(checkpoint),
        )
    chord_results: dict[str, dict[str, Any]] = {}
    for index, case_id in enumerate(
        config["multislice"]["fixed_case_order"], start=1
    ):
        result = _multislice_chord_case(
            config,
            str(case_id),
            case_index=index,
            progress_callback=progress_callback,
        )
        chord_results[str(case_id)] = result
        checkpoint = run_dir / "checkpoints" / f"{case_id}.npz"
        _save_multislice_checkpoint(checkpoint, result)
        _emit(
            progress_callback,
            "checkpoint_saved",
            case_id=str(case_id),
            path=checkpoint.relative_to(run_dir).as_posix(),
            sha256=_sha256(checkpoint),
        )
    q8_field, q8_controls = _load_q8_checkpoint(config)
    _emit(
        progress_callback,
        "q8_checkpoint_loaded",
        sha256=str(config["provenance"]["r10_q8_checkpoint_sha256"]),
    )
    thresholds = config["thresholds"]
    case_controls = {
        case_id: result["controls"]
        for case_id, result in helmholtz_results.items()
    }
    case_controls.update(
        {
            case_id: result["controls"]
            for case_id, result in chord_results.items()
        }
    )
    solver_residuals = {
        case_id: float(controls["solver_controls"]["relative_residual"])
        for case_id, controls in case_controls.items()
        if "solver_controls" in controls
    }
    solver_pass = bool(
        max(solver_residuals.values())
        <= float(thresholds["solve_relative_residual_max"])
    )
    algebra_values: list[float] = []
    for case_id in config["helmholtz"]["fixed_case_order"]:
        controls = case_controls[str(case_id)]
        material = controls["material_controls"]
        algebra_values.extend(
            [
                float(controls["pml_physical_core_identity_max_abs_error"]),
                float(controls["matrix_controls"]["complex_symmetric_max_abs_error"]),
                float(material["fraction_bound_error"]),
                float(material["annular_to_subnode_volume_relative_error"]),
                abs(float(controls["observation_controls"]["upper_weight"]) - 0.5),
            ]
        )
    for case_id in config["multislice"]["fixed_case_order"]:
        controls = case_controls[str(case_id)]
        algebra_values.extend(
            [
                float(controls["fraction_bound_error"]),
                float(controls["index_bound_error"]),
                float(controls["homogeneous_control_relative_l2"]),
            ]
        )
    interface_controls = background_interface_controls(
        wavelength_m=float(config["physics"]["wavelength_m"]),
        n_glass=float(config["physics"]["n_glass"]),
        n_air=float(config["physics"]["n_air"]),
        interface_z_m=float(config["physics"]["background_interface_z_m"]),
        incident_amplitude=float(config["physics"]["incident_amplitude"]),
    )
    algebra_values.extend(
        [
            float(interface_controls["value_continuity_relative_error"]),
            float(interface_controls["derivative_continuity_relative_error"]),
            float(preflight_metrics["maximum_algebra_error"]),
        ]
    )
    preliminary_algebra_error = float(max(algebra_values))
    preflight_pass = bool(preflight_metrics["formal_r11_allowed"])
    all_finite_prepass = bool(
        all(bool(controls["all_finite"]) for controls in case_controls.values())
        and np.all(np.isfinite(q8_field))
    )
    _emit(progress_callback, "postprocessing_started")
    preliminary_post = _postprocess_once(
        config,
        helmholtz_results,
        chord_results,
        q8_field,
        hard_controls_prepass=False,
    )
    repeated = _postprocess_once(
        config,
        helmholtz_results,
        chord_results,
        q8_field,
        hard_controls_prepass=False,
    )
    determinism_error = _postprocessing_repeat_error(preliminary_post, repeated)
    del repeated
    projection_errors = [
        float(controls[name])
        for controls in (
            *preliminary_post["helmholtz"]["projection_controls"].values(),
            *preliminary_post["multislice"]["projection_controls"].values(),
        )
        for name in (
            "repeat_relative_l2",
            "idempotence_relative_l2",
            "constant_max_abs_error",
        )
    ]
    restriction = preliminary_post["multislice"]["restriction_controls"]
    maximum_algebra_error = float(
        max(
            preliminary_algebra_error,
            *projection_errors,
            float(restriction["constant_max_abs_error"]),
            float(restriction["area_weighted_complex_mean_relative_error"]),
        )
    )
    algebra_pass = bool(
        maximum_algebra_error
        <= float(thresholds["algebra_absolute_or_relative_max"])
    )
    determinism_pass = bool(
        determinism_error
        <= float(thresholds["postprocessing_determinism_relative_l2_max"])
    )
    all_finite = bool(
        all_finite_prepass
        and all(
            bool(controls["all_finite"])
            for controls in preliminary_post["helmholtz"][
                "projection_controls"
            ].values()
        )
        and all(
            bool(controls["all_finite"])
            for controls in preliminary_post["multislice"][
                "projection_controls"
            ].values()
        )
        and all(
            bool(controls["all_finite"])
            for controls in preliminary_post["multislice"][
                "polar_controls"
            ].values()
        )
    )
    hard_controls_pass = bool(
        solver_pass
        and algebra_pass
        and determinism_pass
        and preflight_pass
        and all_finite
    )
    # Fail closed: the two passes used to establish projection algebra and
    # determinism above are never allowed to call the cross-model comparator.
    # Only this final pass receives the fully evaluated hard-control state.
    post = _postprocess_once(
        config,
        helmholtz_results,
        chord_results,
        q8_field,
        hard_controls_prepass=hard_controls_pass,
    )
    gates = post["gates"]
    reference_validated = bool(
        gates["domain_gate_pass"]
        and gates["adc5_mesh_gate_pass"]
        and gates["cartesian_anisotropy_gate_pass"]
    )
    cross = post["conditional_cross_model"]
    if not hard_controls_pass:
        status = "Failed"
        interpretation = "r11_hard_controls_failed"
    elif not reference_validated:
        failed_components = [
            label
            for key, label in (
                ("domain_gate_pass", "domain"),
                ("adc5_mesh_gate_pass", "mesh"),
                ("cartesian_anisotropy_gate_pass", "anisotropy"),
            )
            if not gates[key]
        ]
        status = "Failed"
        interpretation = "r11_reference_not_closed__" + "_".join(
            failed_components
        )
    elif float(cross["passband_radial_l2"]) <= float(
        thresholds["cross_model_materiality_relative_l2"]
    ):
        status = "Passed"
        interpretation = "r11_scalar_cross_model_difference_not_resolved"
    else:
        status = "Passed"
        interpretation = "r11_scalar_cross_model_difference_resolved"
    metrics = {
        "version": "R11",
        "scientific_result": True,
        "provenance": dict(config["provenance"]),
        "methods": {
            "helmholtz_formulation": str(config["helmholtz"]["formulation"]),
            "formal_helmholtz_discretization": str(
                config["helmholtz"]["formal_discretization"]
            ),
            "report_helmholtz_discretization": str(
                config["helmholtz"]["report_discretization"]
            ),
            "multislice_interface": str(config["multislice"]["interface_rule"]),
            "multislice_propagator": str(
                config["multislice"]["propagation_operator"]
            ),
            "angular_diagnostic": "fixed_radius_cubic_polar_sampling",
            "alignment": "none",
        },
        "sampling": {
            "helmholtz_case_order": list(
                config["helmholtz"]["fixed_case_order"]
            ),
            "multislice_case_order": list(
                config["multislice"]["fixed_case_order"]
            ),
            "comparison_max_radius_m": float(
                config["comparison"]["comparison_max_radius_m"]
            ),
            "cartesian_shape": list(config["comparison"]["cartesian_shape"]),
            "cartesian_dx_m": float(config["comparison"]["cartesian_dx_m"]),
            "polar_radius_count": int(config["anisotropy"]["radius_count"]),
            "polar_theta_count": int(config["anisotropy"]["theta_count"]),
        },
        "case_controls": case_controls,
        "q8_checkpoint_controls": q8_controls,
        "background_interface_controls": interface_controls,
        "projection_controls": {
            "helmholtz": post["helmholtz"]["projection_controls"],
            "multislice": post["multislice"]["projection_controls"],
        },
        "restriction_controls": restriction,
        "comparisons": post["helmholtz"]["comparisons"],
        "anisotropy": {
            key: value
            for key, value in post["multislice"].items()
            if key
            not in {
                "fields",
                "polar_means",
                "polar_radius_m",
                "projection_controls",
                "restriction_controls",
                "selected_maps",
            }
        },
        "gates": gates,
        "conditional_cross_model": cross,
        "hard_controls": {
            "preflight_pass": preflight_pass,
            "solver_residuals": solver_residuals,
            "maximum_solver_relative_residual": float(
                max(solver_residuals.values())
            ),
            "solver_pass": solver_pass,
            "maximum_algebra_error": maximum_algebra_error,
            "algebra_pass": algebra_pass,
            "postprocessing_determinism_relative_l2": determinism_error,
            "determinism_pass": determinism_pass,
            "all_finite": all_finite,
            "pass": hard_controls_pass,
        },
        "reference_validated": reference_validated,
        "thresholds": dict(thresholds),
        "status": status,
        "interpretation_code": interpretation,
    }
    selected_fields = {
        "chord1024_restricted_passband": post["multislice"]["fields"][
            "chord1024_restricted_passband"
        ],
        "adc_fine_core48_passband": post["helmholtz"]["cartesian_pass"][
            "adc_fine_core48"
        ],
    }
    _emit(
        progress_callback,
        "postprocessing_completed",
        domain_gate_pass=bool(gates["domain_gate_pass"]),
        adc5_mesh_gate_pass=bool(gates["adc5_mesh_gate_pass"]),
        cartesian_anisotropy_gate_pass=bool(
            gates["cartesian_anisotropy_gate_pass"]
        ),
        cross_model_executed=bool(cross["executed"]),
        status=status,
        interpretation_code=interpretation,
    )
    return {
        "metrics": metrics,
        "selected_maps": post["selected_maps"],
        "radial_profiles": post["radial_profiles"],
        "polar_profiles": {
            "radius_m": post["multislice"]["polar_radius_m"],
            **post["multislice"]["polar_means"],
        },
        "native_traces": {
            case_id: {
                "radius_m": result["radius_m"],
                "normalized_total_trace": result["normalized_total_trace"],
                "normalized_scattered_trace": result[
                    "normalized_scattered_trace"
                ],
            }
            for case_id, result in helmholtz_results.items()
        },
        "selected_complex_fields": selected_fields,
    }


def _progress_writer(path: Path) -> ProgressCallback:
    payload: dict[str, Any] = {
        "purpose": "formal_scientific_r11_reference_closure",
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


def _load_and_validate_provenance(
    config: Mapping[str, Any]
) -> dict[str, Any]:
    provenance = config["provenance"]
    if any(
        "__LOCK_AFTER_PREFLIGHT__" in str(value)
        for value in provenance.values()
    ):
        raise RuntimeError("R11 preflight provenance is not locked")
    r10_run = PROJECT_ROOT / str(provenance["r10_stage_b_run"])
    r10_paths = {
        "r10_metrics_sha256": r10_run / "metrics.json",
        "r10_q8_checkpoint_sha256": (
            r10_run / "checkpoints" / "multislice_fine_1024.npz"
        ),
        "r10_repaired_hdf5_sha256": (
            r10_run / "outputs" / "exp040_r10_stage_b_repaired.h5"
        ),
    }
    for key, path in r10_paths.items():
        if _sha256(path) != str(provenance[key]):
            raise RuntimeError(f"R10 provenance differs: {key}")
    preflight_run = PROJECT_ROOT / str(provenance["preflight_run"])
    metrics_path = preflight_run / "metrics.json"
    state_path = preflight_run / "run_state.json"
    hdf5_path = preflight_run / str(
        provenance["preflight_hdf5_relative_path"]
    )
    repair_path = preflight_run / "artifact_repair.json"
    if _sha256(metrics_path) != str(provenance["preflight_metrics_sha256"]):
        raise RuntimeError("R11 preflight metrics provenance differs")
    if _sha256(state_path) != str(
        provenance["preflight_original_run_state_sha256"]
    ):
        raise RuntimeError("R11 original preflight run state differs")
    if _sha256(hdf5_path) != str(provenance["preflight_hdf5_sha256"]):
        raise RuntimeError("R11 preflight HDF5 provenance differs")
    if _sha256(repair_path) != str(
        provenance["preflight_repair_manifest_sha256"]
    ):
        raise RuntimeError("R11 preflight repair manifest provenance differs")
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with state_path.open("r", encoding="utf-8") as handle:
        original_state = json.load(handle)
    with repair_path.open("r", encoding="utf-8") as handle:
        repair = json.load(handle)
    if (
        metrics.get("status") != "Passed"
        or metrics.get("interpretation_code") != "r11_formal_preflight_passed"
        or metrics.get("formal_r11_allowed") is not True
        or metrics.get("hard_controls_pass") is not True
        or metrics.get("scientific_result") is not False
        or not all(metrics.get("control_pass", {}).values())
    ):
        raise RuntimeError("R11 preflight did not pass")
    if (
        original_state.get("status") != "failed_during_execution"
        or original_state.get("formal_r11_allowed") is not False
        or original_state.get("scientific_result") is not False
    ):
        raise RuntimeError("R11 original preflight failure was not preserved")
    repaired_relative = str(provenance["preflight_hdf5_relative_path"])
    if (
        repair.get("version") != "R11_preflight_artifact_repair_v1"
        or repair.get("original_run_state_preserved") is not True
        or repair.get("preflight_control_recomputation") is not False
        or repair.get("scientific_forward_recomputation") is not False
        or repair.get("external_metrics_formal_r11_allowed") is not True
        or repair.get("registered_source_config_sha256")
        != str(provenance["preflight_source_config_sha256"])
        or repair.get("output_sha256", {}).get(repaired_relative)
        != str(provenance["preflight_hdf5_sha256"])
        or not all(repair.get("validation", {}).values())
    ):
        raise RuntimeError("R11 preflight artifact repair did not validate")
    return metrics


def _write_hdf5(
    path: Path,
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    save_ptycho_hdf5(
        path,
        instrument={
            "wavelength_m": float(config["physics"]["wavelength_m"]),
            "n_glass": float(config["physics"]["n_glass"]),
            "n_air": float(config["physics"]["n_air"]),
            "sampling": result["metrics"]["sampling"],
        },
        sample={
            "type": "single_axisymmetric_air_filled_tgv_in_glass",
            "geometry": dict(config["physics"]),
            "interface_rule": str(config["multislice"]["interface_rule"]),
        },
        config_yaml=config_to_yaml(dict(config)),
        metadata=dict(metadata),
        metrics=result["metrics"],
    )
    with h5py.File(path, "a") as h5:
        data = h5["entry/data"]
        radial = data.require_group("radial_profiles")
        for name, values in result["radial_profiles"].items():
            radial.create_dataset(name, data=np.asarray(values))
        polar = data.require_group("polar_profiles")
        for name, values in result["polar_profiles"].items():
            polar.create_dataset(name, data=np.asarray(values))
        traces = data.require_group("native_helmholtz_traces")
        for case_id, values in result["native_traces"].items():
            group = traces.require_group(case_id)
            for name, array in values.items():
                group.create_dataset(name, data=np.asarray(array))
        fields = data.require_group("selected_complex_fields")
        for name, values in result["selected_complex_fields"].items():
            fields.create_dataset(name, data=np.asarray(values))


def _validate_artifacts(run_dir: Path, config: Mapping[str, Any]) -> None:
    expected = set(str(value) for value in config["output"]["required_files"])
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"R11 artifact set differs: {sorted(actual)}")
    for filename in (
        "metadata.json",
        "metrics.json",
        "run_state.json",
        "run_progress.json",
    ):
        with (run_dir / filename).open("r", encoding="utf-8") as handle:
            json.load(handle)
    for checkpoint in (
        *config["helmholtz"]["fixed_case_order"],
        *config["multislice"]["fixed_case_order"],
    ):
        with np.load(run_dir / "checkpoints" / f"{checkpoint}.npz") as data:
            if "controls_json" not in data:
                raise RuntimeError(f"R11 checkpoint is incomplete: {checkpoint}")
            json.loads(str(data["controls_json"]))
    hdf5_path = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    with h5py.File(hdf5_path, "r") as h5:
        entry = h5["entry"]
        if set(entry) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
        }:
            raise RuntimeError("R11 HDF5 entry layout differs")
        if set(entry["data"]) != {
            "native_helmholtz_traces",
            "polar_profiles",
            "radial_profiles",
            "selected_complex_fields",
        }:
            raise RuntimeError("R11 HDF5 data layout differs")
        if "truth" in entry or "reconstruction" in entry:
            raise RuntimeError("R11 HDF5 must not claim truth/reconstruction")
    for filename in EXP040_R11_FIGURE_FILENAMES:
        image = np.asarray(iio.imread(run_dir / "figures" / filename))
        if image.ndim not in (2, 3) or image.size == 0:
            raise RuntimeError(f"R11 figure is invalid: {filename}")


def run(config_path: Path) -> Path:
    """Execute the one formal R11 scientific experiment."""

    source = config_path.resolve()
    if REGISTERED_CONFIG_SHA256 == "__LOCK_AFTER_PREFLIGHT__":
        raise RuntimeError("R11 source config has not been locked")
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R11 source config hash differs")
    config = load_config(source)
    validate_r11_config(config)
    preflight_metrics = _load_and_validate_provenance(config)
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
            "source_config": str(source),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
            "scientific_result": True,
        },
    )
    progress = _progress_writer(run_dir / "run_progress.json")
    progress("run_started", {"source_config_sha256": REGISTERED_CONFIG_SHA256})
    started = time.perf_counter()
    try:
        result = _run_r11(
            config,
            run_dir=run_dir,
            preflight_metrics=preflight_metrics,
            progress_callback=progress,
        )
        metrics = result["metrics"]
        metrics["total_execution_elapsed_s"] = float(
            time.perf_counter() - started
        )
        status = str(metrics["status"])
        interpretation = str(metrics["interpretation_code"])
        metadata = {
            "experiment_id": "exp040",
            "diagnostic_stage": "R11",
            "scientific_result": True,
            "run_path": str(run_dir.resolve()),
            "source_config": str(source),
            "source_config_sha256": REGISTERED_CONFIG_SHA256,
            "created_at": created_at_utc(),
            "git_commit": get_git_commit(PROJECT_ROOT) or "unavailable",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "status": status,
            "interpretation_code": interpretation,
            "reference_validated": bool(metrics["reference_validated"]),
            "cross_model_executed": bool(
                metrics["conditional_cross_model"]["executed"]
            ),
        }
        save_json(run_dir / "metrics.json", metrics)
        save_json(run_dir / "metadata.json", metadata)
        progress("artifacts_writing_started", {})
        save_exp040_r11_figures(result, run_dir / "figures")
        _write_hdf5(
            run_dir / "outputs" / str(config["output"]["hdf5_filename"]),
            config=config,
            metadata=metadata,
            result=result,
        )
        save_json(
            run_dir / "run_state.json",
            {
                "status": "validation_pending",
                "completed_at": created_at_utc(),
                "scientific_status": status,
                "interpretation_code": interpretation,
                "scientific_result": True,
            },
        )
        _validate_artifacts(run_dir, config)
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "completed_at": created_at_utc(),
                "scientific_status": status,
                "interpretation_code": interpretation,
                "reference_validated": bool(metrics["reference_validated"]),
                "cross_model_executed": bool(
                    metrics["conditional_cross_model"]["executed"]
                ),
                "scientific_result": True,
                "artifacts_validated": True,
            },
        )
        progress(
            "artifacts_validated",
            {
                "scientific_status": status,
                "interpretation_code": interpretation,
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

    print(f"run_dir: {run_dir.resolve()}", flush=True)
    print(f"r11_status: {status}", flush=True)
    print(f"interpretation: {interpretation}", flush=True)
    print(f"domain_gate_pass: {metrics['gates']['domain_gate_pass']}", flush=True)
    print(
        f"adc5_mesh_gate_pass: {metrics['gates']['adc5_mesh_gate_pass']}",
        flush=True,
    )
    print(
        "cartesian_anisotropy_gate_pass: "
        f"{metrics['gates']['cartesian_anisotropy_gate_pass']}",
        flush=True,
    )
    print(
        "cross_model_executed: "
        f"{metrics['conditional_cross_model']['executed']}",
        flush=True,
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
