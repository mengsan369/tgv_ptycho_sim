"""Run the formal exp040 R10 Stage-B bidirectional scalar comparator."""

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
    _r9_project_with_controls,
    relative_l2,
    restrict_aligned_cell_average,
)
from tgv_ptycho.forward.helmholtz_axisymmetric import (  # noqa: E402
    annular_anisotropy_relative_l2,
    annular_mean_from_cartesian,
    assemble_cylindrical_helmholtz,
    background_interface_controls,
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
from tgv_ptycho.objects.tgv3d import make_tgv_air_fraction_slice  # noqa: E402
from tgv_ptycho.objects.tgv_geometry import (  # noqa: E402
    diameter_profile,
    midpoint_z_grid,
)
from tgv_ptycho.optics.angular_spectrum import (  # noqa: E402
    angular_spectrum_propagate,
)
from tgv_ptycho.viz.plot_exp040_r10_stage_b import (  # noqa: E402
    EXP040_R10_STAGE_B_FIGURE_FILENAMES,
    save_exp040_r10_stage_b_figures,
)

REGISTERED_CONFIG_SHA256 = (
    "AF718435D573DA13DD5D01D9AAC13A47B51281D7B2B1745379EC6ADCBD023652"
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
        raise ValueError(f"R10 Stage-B {name} differs from registration.")


def validate_stage_b_config(config: Mapping[str, Any]) -> None:
    """Validate all Stage-B model, sampling, and threshold controls."""

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
            "thresholds",
            "output",
        },
        "top-level sections",
    )
    _require_exact(config["experiment"]["id"], "exp040", "experiment id")
    _require_exact(config["experiment"]["stage"], "R10_stage_b", "stage")
    _require_exact(
        config["experiment"]["scientific_result"], True, "scientific role"
    )
    _require_exact(
        dict(config["physics"]),
        {
            "wavelength_m": 5.32e-7,
            "n_glass": 1.5,
            "n_air": 1.0,
            "background_interface_z_m": 1.0e-4,
            "incident_amplitude": 1.0,
            "sample_thickness_m": 1.0e-4,
            "d_top_m": 3.0e-5,
            "d_waist_m": 2.0e-5,
            "d_bottom_m": 3.0e-5,
            "z_waist_m": 5.0e-5,
        },
        "physics",
    )
    helmholtz = config["helmholtz"]
    _require_exact(
        helmholtz["fixed_case_order"],
        ["coarse_nominal", "fine_nominal", "fine_enlarged_pml"],
        "Helmholtz case order",
    )
    _require_exact(
        dict(helmholtz["cases"]),
        {
            "coarse_nominal": {
                "dr_m": 1.25e-7,
                "dz_m": 1.25e-7,
                "pml_thickness_m": 2.0e-6,
                "expected_nr": 208,
                "expected_nz": 864,
                "expected_unknowns": 179712,
            },
            "fine_nominal": {
                "dr_m": 8.333333333333333e-8,
                "dz_m": 8.333333333333333e-8,
                "pml_thickness_m": 2.0e-6,
                "expected_nr": 312,
                "expected_nz": 1296,
                "expected_unknowns": 404352,
            },
            "fine_enlarged_pml": {
                "dr_m": 8.333333333333333e-8,
                "dz_m": 8.333333333333333e-8,
                "pml_thickness_m": 3.0e-6,
                "expected_nr": 324,
                "expected_nz": 1320,
                "expected_unknowns": 427680,
            },
        },
        "Helmholtz cases",
    )
    _require_exact(
        {
            key: helmholtz[key]
            for key in (
                "formulation",
                "discretization",
                "radial_core_max_m",
                "z_core_min_m",
                "z_core_max_m",
                "axial_material_subnodes",
                "radial_material_rule",
                "mass_material_rule",
            )
        },
        {
            "formulation": "axisymmetric_scattered_field_contrast_source",
            "discretization": "cell_centered_conservative_five_point",
            "radial_core_max_m": 2.4e-5,
            "z_core_min_m": -2.0e-6,
            "z_core_max_m": 1.02e-4,
            "axial_material_subnodes": 8,
            "radial_material_rule": "exact_annular_disk_intersection",
            "mass_material_rule": "cell_average_n_squared",
        },
        "Helmholtz discretization",
    )
    _require_exact(
        dict(helmholtz["pml"]),
        {
            "coordinate_stretch": "cylindrical_complex_coordinate",
            "polynomial_order": 3,
            "target_one_way_amplitude": 1.0e-8,
            "radial_reference_medium": "air",
            "lower_z_reference_medium": "glass",
            "upper_z_reference_medium": "air",
            "outer_boundary": (
                "scattered_field_zero_dirichlet_at_half_cell_distance"
            ),
        },
        "PML",
    )
    _require_exact(
        dict(helmholtz["solver"]),
        {
            "package": "scipy_splu",
            "sparse_format": "csc",
            "permc_spec": "COLAMD",
            "retain_one_matrix_and_lu_at_a_time": True,
        },
        "solver",
    )
    _require_exact(
        dict(config["observation"]),
        {
            "z_m": 1.01e-4,
            "axial_extraction": "linear_between_bracketing_cell_centers",
            "normalization": "total_field_divided_by_analytic_background",
            "phase_scale_shift_tilt_alignment": False,
        },
        "observation",
    )
    _require_exact(
        dict(config["multislice"]),
        {
            "shape": [1024, 1024],
            "dx_m": 6.25e-8,
            "dz_m": 2.5e-7,
            "expected_slice_count": 400,
            "interface_factor": 8,
            "interface_rule": "staggered_midpoint_cartesian_air_fraction",
            "effective_index_rule": "linear_cell_average_of_index",
            "propagation_operator": "centered_symmetric_split_step",
            "internal_reference_index": 1.5,
            "internal_bandlimit": True,
            "post_exit_air_distance_m": 1.0e-6,
            "post_exit_air_bandlimit": True,
            "homogeneous_control": "analytic_glass_then_air_plane_wave",
        },
        "multislice",
    )
    _require_exact(
        dict(config["comparison"]),
        {
            "cartesian_shape": [512, 512],
            "cartesian_dx_m": 1.25e-7,
            "trace_support_radius_m": 2.4e-5,
            "outer_fill_normalized_value": 1.0,
            "physical_passband_cutoff_cycles_per_m": 1879699.2481203007,
            "passband_boundary_inclusive": True,
            "multislice_order": (
                "native_1024_passband_then_aligned_2x2_cell_average"
            ),
            "annular_bin_width_m": 1.25e-7,
            "annular_maximum_radius_m": 2.0e-5,
            "weighted_norm": "two_pi_r_relative_l2",
            "mesh_pair": ["coarse_nominal", "fine_nominal"],
            "mesh_denominator": "fine_nominal",
            "pml_pair": ["fine_nominal", "fine_enlarged_pml"],
            "pml_denominator": "fine_enlarged_pml",
            "cross_model_pair": [
                "multislice_fine_1024",
                "fine_enlarged_pml",
            ],
            "cross_model_denominator": "fine_enlarged_pml",
            "guard_inner_max_radius_m": 2.0e-5,
            "guard_min_radius_m": 2.2e-5,
            "guard_max_radius_m": 2.4e-5,
            "phase_scale_shift_tilt_alignment": False,
            "postprocessing_determinism": (
                "repeat_complete_mapping_projection_mean_metrics"
            ),
        },
        "comparison",
    )
    _require_exact(
        dict(config["thresholds"]),
        {
            "solve_relative_residual_max": 1.0e-9,
            "algebra_absolute_or_relative_max": 1.0e-12,
            "postprocessing_determinism_relative_l2_max": 1.0e-14,
            "reference_passband_relative_l2_max": 5.0e-2,
            "homogeneous_field_relative_l2_max": 5.0e-2,
            "multislice_azimuthal_anisotropy_relative_l2_max": 5.0e-2,
            "outer_guard_rms_ratio_max": 5.0e-2,
            "cross_model_materiality_relative_l2": 5.0e-2,
            "require_all_finite": True,
        },
        "thresholds",
    )
    _require_exact(
        config["output"]["figure_filenames"],
        list(EXP040_R10_STAGE_B_FIGURE_FILENAMES),
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
        raise ValueError("R10 Stage-B full matrices/volumes must not be persisted.")


def _emit(callback: ProgressCallback | None, event: str, **details: Any) -> None:
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


def _solve_helmholtz_case(
    config: Mapping[str, Any],
    case_id: str,
    *,
    case_index: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    physics = config["physics"]
    helmholtz = config["helmholtz"]
    case = helmholtz["cases"][case_id]
    _emit(
        progress_callback,
        "helmholtz_case_started",
        case_id=case_id,
        case_index=case_index,
        case_count=3,
        expected_unknowns=int(case["expected_unknowns"]),
    )
    started = time.perf_counter()
    grid = make_axisymmetric_grid(
        dr_m=float(case["dr_m"]),
        dz_m=float(case["dz_m"]),
        radial_core_max_m=float(helmholtz["radial_core_max_m"]),
        z_core_min_m=float(helmholtz["z_core_min_m"]),
        z_core_max_m=float(helmholtz["z_core_max_m"]),
        pml_thickness_m=float(case["pml_thickness_m"]),
    )
    actual_shape = [grid.nr, grid.nz, grid.unknown_count]
    expected_shape = [
        int(case["expected_nr"]),
        int(case["expected_nz"]),
        int(case["expected_unknowns"]),
    ]
    if actual_shape != expected_shape:
        raise RuntimeError(f"{case_id} grid differs from registration")
    pml_config = helmholtz["pml"]
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
    pml_identity_error = _pml_identity_error(grid, pml)
    background_n2 = make_background_n2(
        grid,
        interface_z_m=float(physics["background_interface_z_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
    )
    tgv_n2, material_controls = make_tgv_n2_cell_average(
        grid,
        thickness_m=float(physics["sample_thickness_m"]),
        d_top_m=float(physics["d_top_m"]),
        d_waist_m=float(physics["d_waist_m"]),
        d_bottom_m=float(physics["d_bottom_m"]),
        z_waist_m=float(physics["z_waist_m"]),
        n_glass=float(physics["n_glass"]),
        n_air=float(physics["n_air"]),
        axial_subnodes=int(helmholtz["axial_material_subnodes"]),
        background_interface_z_m=float(
            physics["background_interface_z_m"]
        ),
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
        tgv_n2,
        wavelength_m=float(physics["wavelength_m"]),
    )
    source = make_contrast_source(
        grid,
        pml,
        tgv_n2,
        background_n2,
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
        permc_spec=str(helmholtz["solver"]["permc_spec"]),
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
    guard_ratio = outer_guard_rms_ratio(
        normalized_scattered,
        grid.r_centers_m,
        inner_max_radius_m=float(comparison["guard_inner_max_radius_m"]),
        guard_min_radius_m=float(comparison["guard_min_radius_m"]),
        guard_max_radius_m=float(comparison["guard_max_radius_m"]),
    )
    all_finite = bool(
        matrix_controls["finite_data"]
        and material_controls["all_finite"]
        and solver_controls["all_finite"]
        and np.all(np.isfinite(background_z))
        and np.all(np.isfinite(normalized_total))
        and np.all(np.isfinite(normalized_scattered))
        and np.isfinite(guard_ratio)
    )
    controls = {
        "id": case_id,
        "nr": grid.nr,
        "nz": grid.nz,
        "unknown_count": grid.unknown_count,
        "dr_m": grid.dr_m,
        "dz_m": grid.dz_m,
        "pml_thickness_m": grid.pml_thickness_m,
        "pml_radial_peak_alpha": pml.radial_peak_alpha,
        "pml_lower_z_peak_alpha": pml.lower_z_peak_alpha,
        "pml_upper_z_peak_alpha": pml.upper_z_peak_alpha,
        "pml_physical_core_identity_max_abs_error": pml_identity_error,
        "matrix_controls": matrix_controls,
        "material_controls": material_controls,
        "solver_controls": solver_controls,
        "observation_controls": observation_controls,
        "background_observation_abs": float(abs(background_observation)),
        "source_l2": float(np.linalg.norm(source)),
        "outer_guard_rms_ratio": guard_ratio,
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": all_finite,
    }
    result = {
        "radius_m": grid.r_centers_m.copy(),
        "normalized_total_trace": normalized_total.astype(
            np.complex128, copy=True
        ),
        "normalized_scattered_trace": normalized_scattered.astype(
            np.complex128, copy=True
        ),
        "controls": controls,
    }
    del (
        background_n2,
        background_z,
        matrix,
        normalized_scattered,
        normalized_total,
        pml,
        scattered,
        scattered_trace,
        source,
        tgv_n2,
    )
    gc.collect()
    _emit(
        progress_callback,
        "helmholtz_case_completed",
        case_id=case_id,
        case_index=case_index,
        case_count=3,
        outer_guard_rms_ratio=guard_ratio,
        total_elapsed_s=controls["total_elapsed_s"],
    )
    return result


def _multislice_reference(
    config: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    physics = config["physics"]
    registered = config["multislice"]
    shape = tuple(int(value) for value in registered["shape"])
    dx_m = float(registered["dx_m"])
    dz_m = float(registered["dz_m"])
    q = int(registered["interface_factor"])
    _emit(
        progress_callback,
        "multislice_case_started",
        shape=list(shape),
        dx_m=dx_m,
        dz_m=dz_m,
    )
    started = time.perf_counter()
    z_m, widths = midpoint_z_grid(
        float(physics["sample_thickness_m"]), dz_m
    )
    if len(widths) != int(registered["expected_slice_count"]):
        raise RuntimeError("multislice slice count differs from registration")
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
    count_identity_error = 0.0
    discrete_volume = 0.0
    all_finite = True

    def slices():
        nonlocal fraction_bound_error
        nonlocal index_bound_error
        nonlocal count_identity_error
        nonlocal discrete_volume
        nonlocal all_finite
        for index, (diameter, width) in enumerate(
            zip(diameters, widths, strict=True)
        ):
            fraction = make_tgv_air_fraction_slice(
                shape, dx_m, float(diameter), q, (0.0, 0.0)
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
            scaled = fraction * q**2
            count_identity_error = max(
                count_identity_error,
                float(np.max(np.abs(scaled - np.rint(scaled)))),
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
    analytic_homogeneous_value = float(physics["incident_amplitude"]) * np.exp(
        1j
        * k0
        * (
            float(physics["n_glass"])
            * float(physics["sample_thickness_m"])
            + float(physics["n_air"])
            * float(registered["post_exit_air_distance_m"])
        )
    )
    numerical_homogeneous = angular_spectrum_propagate(
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
        np.linalg.norm(numerical_homogeneous - analytic_homogeneous_value)
        / max(np.linalg.norm(numerical_homogeneous), np.finfo(float).eps)
    )
    normalized = air_exit / analytic_homogeneous_value
    continuous_midpoint_volume = float(
        np.sum(np.pi * (diameters / 2.0) ** 2 * widths)
    )
    volume_error = float(
        abs(discrete_volume - continuous_midpoint_volume)
        / max(continuous_midpoint_volume, np.finfo(float).eps)
    )
    all_finite = bool(
        all_finite
        and np.all(np.isfinite(sample_exit))
        and np.all(np.isfinite(air_exit))
        and np.all(np.isfinite(normalized))
        and np.all(np.isfinite(numerical_homogeneous))
    )
    controls = {
        "id": "multislice_fine_1024",
        "shape": list(shape),
        "dx_m": dx_m,
        "dz_m": dz_m,
        "slice_count": int(len(widths)),
        "interface_factor": q,
        "fraction_bound_error": fraction_bound_error,
        "index_bound_error": index_bound_error,
        "subnode_count_identity_error": count_identity_error,
        "discrete_air_volume_m3": discrete_volume,
        "continuous_midpoint_air_volume_m3": continuous_midpoint_volume,
        "air_volume_relative_error_report_only": volume_error,
        "homogeneous_control_relative_l2": homogeneous_error,
        "total_elapsed_s": float(time.perf_counter() - started),
        "all_finite": all_finite,
    }
    del (
        air_exit,
        incident,
        numerical_homogeneous,
        sample_exit,
    )
    gc.collect()
    _emit(
        progress_callback,
        "multislice_case_completed",
        slice_count=controls["slice_count"],
        homogeneous_control_relative_l2=homogeneous_error,
        total_elapsed_s=controls["total_elapsed_s"],
    )
    return {
        "normalized_native_field": normalized.astype(np.complex128, copy=False),
        "controls": controls,
    }


def _restriction_controls(
    fine: np.ndarray, restricted: np.ndarray
) -> dict[str, Any]:
    constant = np.full((4, 4), 2.0 - 3.0j, dtype=np.complex128)
    constant_error = float(
        np.max(
            np.abs(restrict_aligned_cell_average(constant, 2) - (2.0 - 3.0j))
        )
    )
    mean_error = float(
        abs(np.mean(fine) - np.mean(restricted))
        / max(abs(np.mean(fine)), np.finfo(float).eps)
    )
    return {
        "method": "aligned_2x2_complex_cell_average",
        "fine_shape": list(fine.shape),
        "restricted_shape": list(restricted.shape),
        "constant_max_abs_error": constant_error,
        "area_weighted_complex_mean_relative_error": mean_error,
        "shape_matches": bool(
            restricted.shape == (fine.shape[0] // 2, fine.shape[1] // 2)
        ),
    }


def _postprocess_once(
    config: Mapping[str, Any],
    helmholtz_cases: Mapping[str, Mapping[str, Any]],
    multislice: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = config["comparison"]
    shape = tuple(int(value) for value in comparison["cartesian_shape"])
    dx_m = float(comparison["cartesian_dx_m"])
    cutoff = float(comparison["physical_passband_cutoff_cycles_per_m"])
    bin_width = float(comparison["annular_bin_width_m"])
    max_radius = float(comparison["annular_maximum_radius_m"])
    support_radius = float(comparison["trace_support_radius_m"])
    cartesian_raw: dict[str, np.ndarray] = {}
    cartesian_pass: dict[str, np.ndarray] = {}
    radial_raw: dict[str, np.ndarray] = {}
    radial_pass: dict[str, np.ndarray] = {}
    projection_controls: dict[str, Any] = {}
    annular_counts: np.ndarray | None = None
    radial_radius: np.ndarray | None = None
    for case_id in config["helmholtz"]["fixed_case_order"]:
        case = helmholtz_cases[str(case_id)]
        mapped = radial_trace_to_cartesian(
            case["normalized_total_trace"],
            case["radius_m"],
            shape=shape,
            dx_m=dx_m,
            trace_support_radius_m=support_radius,
            outer_value=complex(comparison["outer_fill_normalized_value"]),
        )
        projected, controls = _r9_project_with_controls(mapped, dx_m, cutoff)
        radii, raw_mean, counts = annular_mean_from_cartesian(
            mapped,
            dx_m=dx_m,
            bin_width_m=bin_width,
            maximum_radius_m=max_radius,
        )
        pass_radii, pass_mean, pass_counts = annular_mean_from_cartesian(
            projected,
            dx_m=dx_m,
            bin_width_m=bin_width,
            maximum_radius_m=max_radius,
        )
        if not np.array_equal(radii, pass_radii) or not np.array_equal(
            counts, pass_counts
        ):
            raise RuntimeError("registered annular bins changed after projection")
        if radial_radius is None:
            radial_radius = radii
            annular_counts = counts
        elif not np.array_equal(radial_radius, radii) or not np.array_equal(
            annular_counts, counts
        ):
            raise RuntimeError("Helmholtz annular bins differ between cases")
        cartesian_raw[str(case_id)] = mapped
        cartesian_pass[str(case_id)] = projected
        radial_raw[str(case_id)] = raw_mean
        radial_pass[str(case_id)] = pass_mean
        projection_controls[str(case_id)] = controls

    native_ms = np.asarray(multislice["normalized_native_field"])
    native_ms_pass, ms_projection_controls = _r9_project_with_controls(
        native_ms,
        float(config["multislice"]["dx_m"]),
        cutoff,
    )
    restricted_ms = restrict_aligned_cell_average(native_ms, 2)
    restricted_ms_pass = restrict_aligned_cell_average(native_ms_pass, 2)
    if restricted_ms.shape != shape or restricted_ms_pass.shape != shape:
        raise RuntimeError("multislice restriction shape differs from registration")
    ms_radii, ms_raw_mean, ms_counts = annular_mean_from_cartesian(
        restricted_ms,
        dx_m=dx_m,
        bin_width_m=bin_width,
        maximum_radius_m=max_radius,
    )
    ms_pass_radii, ms_pass_mean, ms_pass_counts = annular_mean_from_cartesian(
        restricted_ms_pass,
        dx_m=dx_m,
        bin_width_m=bin_width,
        maximum_radius_m=max_radius,
    )
    if (
        radial_radius is None
        or annular_counts is None
        or not np.array_equal(radial_radius, ms_radii)
        or not np.array_equal(radial_radius, ms_pass_radii)
        or not np.array_equal(annular_counts, ms_counts)
        or not np.array_equal(annular_counts, ms_pass_counts)
    ):
        raise RuntimeError("multislice annular bins differ from Helmholtz bins")
    cartesian_raw["multislice_fine_1024"] = restricted_ms
    cartesian_pass["multislice_fine_1024"] = restricted_ms_pass
    radial_raw["multislice_fine_1024"] = ms_raw_mean
    radial_pass["multislice_fine_1024"] = ms_pass_mean
    projection_controls["multislice_fine_1024_native"] = ms_projection_controls
    anisotropy, annular_projection = annular_anisotropy_relative_l2(
        restricted_ms_pass,
        dx_m=dx_m,
        bin_width_m=bin_width,
        maximum_radius_m=max_radius,
    )
    restriction_controls = _restriction_controls(native_ms, restricted_ms)
    constant_radii, constant_mean, _ = annular_mean_from_cartesian(
        np.ones(shape, dtype=np.complex128),
        dx_m=dx_m,
        bin_width_m=bin_width,
        maximum_radius_m=max_radius,
    )
    if not np.array_equal(constant_radii, radial_radius):
        raise RuntimeError("constant annular control radii differ")
    annular_constant_error = float(np.max(np.abs(constant_mean - 1.0)))

    def pair_metrics(test_id: str, reference_id: str) -> dict[str, float]:
        return {
            "raw_radial_l2": radial_weighted_relative_l2(
                radial_raw[test_id], radial_raw[reference_id], radial_radius
            ),
            "passband_radial_l2": radial_weighted_relative_l2(
                radial_pass[test_id], radial_pass[reference_id], radial_radius
            ),
            "raw_cartesian_l2_report_only": relative_l2(
                cartesian_raw[test_id], cartesian_raw[reference_id]
            ),
            "passband_cartesian_l2_report_only": relative_l2(
                cartesian_pass[test_id], cartesian_pass[reference_id]
            ),
        }

    comparisons = {
        "mesh": {
            "pair": list(comparison["mesh_pair"]),
            "denominator": str(comparison["mesh_denominator"]),
            **pair_metrics("coarse_nominal", "fine_nominal"),
        },
        "pml": {
            "pair": list(comparison["pml_pair"]),
            "denominator": str(comparison["pml_denominator"]),
            **pair_metrics("fine_nominal", "fine_enlarged_pml"),
        },
        "cross_model": {
            "pair": list(comparison["cross_model_pair"]),
            "denominator": str(comparison["cross_model_denominator"]),
            **pair_metrics("multislice_fine_1024", "fine_enlarged_pml"),
            "alignment": "none",
        },
    }
    reference = cartesian_pass["fine_enlarged_pml"]
    test = cartesian_pass["multislice_fine_1024"]
    selected_maps = {
        "helmholtz_passband": reference,
        "multislice_passband": test,
        "normalized_cross_residual": np.abs(test - reference)
        / max(float(np.max(np.abs(reference))), np.finfo(float).eps),
        "cross_phase_difference_rad": np.angle(test * np.conjugate(reference)),
        "multislice_annular_projection": annular_projection,
    }
    return {
        "cartesian_raw": cartesian_raw,
        "cartesian_pass": cartesian_pass,
        "radial_raw": radial_raw,
        "radial_pass": radial_pass,
        "radial_radius_m": radial_radius,
        "annular_bin_counts": annular_counts,
        "projection_controls": projection_controls,
        "restriction_controls": restriction_controls,
        "annular_constant_max_abs_error": annular_constant_error,
        "multislice_azimuthal_anisotropy_relative_l2": anisotropy,
        "comparisons": comparisons,
        "selected_maps": selected_maps,
    }


def _postprocessing_repeat_error(
    original: Mapping[str, Any], repeated: Mapping[str, Any]
) -> float:
    errors: list[float] = []
    for category in ("cartesian_raw", "cartesian_pass", "radial_raw", "radial_pass"):
        left = original[category]
        right = repeated[category]
        for key in left:
            errors.append(relative_l2(np.asarray(right[key]), np.asarray(left[key])))
    for comparison_name in ("mesh", "pml", "cross_model"):
        left = original["comparisons"][comparison_name]
        right = repeated["comparisons"][comparison_name]
        for metric in (
            "raw_radial_l2",
            "passband_radial_l2",
            "raw_cartesian_l2_report_only",
            "passband_cartesian_l2_report_only",
        ):
            denominator = max(abs(float(left[metric])), np.finfo(float).eps)
            errors.append(abs(float(right[metric]) - float(left[metric])) / denominator)
    errors.append(
        abs(
            float(repeated["multislice_azimuthal_anisotropy_relative_l2"])
            - float(original["multislice_azimuthal_anisotropy_relative_l2"])
        )
        / max(
            abs(float(original["multislice_azimuthal_anisotropy_relative_l2"])),
            np.finfo(float).eps,
        )
    )
    return float(max(errors))


def _checkpoint_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} in checkpoint.")


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


def _run_stage_b(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    preflight_metrics: Mapping[str, Any],
    progress_callback: ProgressCallback | None = None,
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
    multislice = _multislice_reference(
        config, progress_callback=progress_callback
    )
    ms_checkpoint = run_dir / "checkpoints" / "multislice_fine_1024.npz"
    _save_multislice_checkpoint(ms_checkpoint, multislice)
    _emit(
        progress_callback,
        "checkpoint_saved",
        case_id="multislice_fine_1024",
        path=ms_checkpoint.relative_to(run_dir).as_posix(),
        sha256=_sha256(ms_checkpoint),
    )
    _emit(progress_callback, "postprocessing_started")
    post = _postprocess_once(config, helmholtz_results, multislice)
    repeated = _postprocess_once(config, helmholtz_results, multislice)
    determinism_error = _postprocessing_repeat_error(post, repeated)
    del repeated
    gc.collect()

    thresholds = config["thresholds"]
    interface_controls = background_interface_controls(
        wavelength_m=float(config["physics"]["wavelength_m"]),
        n_glass=float(config["physics"]["n_glass"]),
        n_air=float(config["physics"]["n_air"]),
        interface_z_m=float(config["physics"]["background_interface_z_m"]),
        incident_amplitude=float(config["physics"]["incident_amplitude"]),
    )
    case_controls = {
        case_id: result["controls"]
        for case_id, result in helmholtz_results.items()
    }
    case_controls["multislice_fine_1024"] = multislice["controls"]
    projection_errors = [
        float(controls[name])
        for controls in post["projection_controls"].values()
        for name in (
            "repeat_relative_l2",
            "idempotence_relative_l2",
            "constant_max_abs_error",
        )
    ]
    helmholtz_algebra_errors: list[float] = []
    for case_id in config["helmholtz"]["fixed_case_order"]:
        controls = case_controls[str(case_id)]
        material = controls["material_controls"]
        helmholtz_algebra_errors.extend(
            [
                float(controls["pml_physical_core_identity_max_abs_error"]),
                float(controls["matrix_controls"]["complex_symmetric_max_abs_error"]),
                float(material["fraction_bound_error"]),
                float(material["annular_to_subnode_volume_relative_error"]),
                abs(float(controls["observation_controls"]["upper_weight"]) - 0.5),
            ]
        )
    multislice_controls = case_controls["multislice_fine_1024"]
    restriction = post["restriction_controls"]
    algebra_errors = [
        *projection_errors,
        *helmholtz_algebra_errors,
        float(interface_controls["value_continuity_relative_error"]),
        float(interface_controls["derivative_continuity_relative_error"]),
        float(preflight_metrics["maximum_algebra_error"]),
        float(multislice_controls["fraction_bound_error"]),
        float(multislice_controls["index_bound_error"]),
        float(multislice_controls["subnode_count_identity_error"]),
        float(restriction["constant_max_abs_error"]),
        float(restriction["area_weighted_complex_mean_relative_error"]),
        float(post["annular_constant_max_abs_error"]),
    ]
    maximum_algebra_error = float(max(algebra_errors))
    solver_residuals = {
        str(case_id): float(
            case_controls[str(case_id)]["solver_controls"]["relative_residual"]
        )
        for case_id in config["helmholtz"]["fixed_case_order"]
    }
    maximum_solver_residual = float(max(solver_residuals.values()))
    guard_ratios = {
        str(case_id): float(case_controls[str(case_id)]["outer_guard_rms_ratio"])
        for case_id in config["helmholtz"]["fixed_case_order"]
    }
    maximum_guard_ratio = float(max(guard_ratios.values()))
    homogeneous_error = max(
        float(preflight_metrics["zero_contrast_normalization_max_abs_error"]),
        float(multislice_controls["homogeneous_control_relative_l2"]),
    )
    comparisons = post["comparisons"]
    mesh_pass = bool(
        float(comparisons["mesh"]["passband_radial_l2"])
        <= float(thresholds["reference_passband_relative_l2_max"])
    )
    pml_pass = bool(
        float(comparisons["pml"]["passband_radial_l2"])
        <= float(thresholds["reference_passband_relative_l2_max"])
    )
    solver_pass = bool(
        maximum_solver_residual
        <= float(thresholds["solve_relative_residual_max"])
    )
    algebra_pass = bool(
        maximum_algebra_error
        <= float(thresholds["algebra_absolute_or_relative_max"])
    )
    determinism_pass = bool(
        determinism_error
        <= float(thresholds["postprocessing_determinism_relative_l2_max"])
    )
    homogeneous_pass = bool(
        homogeneous_error
        <= float(thresholds["homogeneous_field_relative_l2_max"])
    )
    anisotropy_pass = bool(
        float(post["multislice_azimuthal_anisotropy_relative_l2"])
        <= float(
            thresholds["multislice_azimuthal_anisotropy_relative_l2_max"]
        )
    )
    guard_pass = bool(
        maximum_guard_ratio <= float(thresholds["outer_guard_rms_ratio_max"])
    )
    all_finite = bool(
        all(bool(controls["all_finite"]) for controls in case_controls.values())
        and np.all(np.isfinite(post["radial_radius_m"]))
        and all(
            np.all(np.isfinite(values))
            for category in (
                post["cartesian_raw"],
                post["cartesian_pass"],
                post["radial_raw"],
                post["radial_pass"],
            )
            for values in category.values()
        )
    )
    reference_validated = bool(
        solver_pass
        and algebra_pass
        and determinism_pass
        and homogeneous_pass
        and mesh_pass
        and pml_pass
        and anisotropy_pass
        and guard_pass
        and all_finite
        and bool(preflight_metrics["formal_stage_b_allowed"])
        and bool(restriction["shape_matches"])
    )
    cross_passband = float(comparisons["cross_model"]["passband_radial_l2"])
    if not reference_validated:
        status = "Failed"
        interpretation = "helmholtz_reference_not_validated"
    elif cross_passband <= float(
        thresholds["cross_model_materiality_relative_l2"]
    ):
        status = "Passed"
        interpretation = "bidirectional_scalar_effect_not_resolved_at_registered_gate"
    else:
        status = "Passed"
        interpretation = "bidirectional_scalar_model_difference_resolved"
    reference_controls = {
        "preflight_pass": bool(preflight_metrics["formal_stage_b_allowed"]),
        "solver_residuals": solver_residuals,
        "maximum_solver_relative_residual": maximum_solver_residual,
        "solver_pass": solver_pass,
        "maximum_algebra_error": maximum_algebra_error,
        "algebra_pass": algebra_pass,
        "postprocessing_determinism_relative_l2": determinism_error,
        "determinism_pass": determinism_pass,
        "homogeneous_field_relative_l2": homogeneous_error,
        "homogeneous_pass": homogeneous_pass,
        "mesh_pass": mesh_pass,
        "pml_pass": pml_pass,
        "multislice_azimuthal_anisotropy_relative_l2": float(
            post["multislice_azimuthal_anisotropy_relative_l2"]
        ),
        "anisotropy_pass": anisotropy_pass,
        "outer_guard_rms_ratios": guard_ratios,
        "maximum_outer_guard_rms_ratio": maximum_guard_ratio,
        "outer_guard_pass": guard_pass,
        "all_finite": all_finite,
        "reference_validated": reference_validated,
    }
    metrics = {
        "version": "R10_stage_b",
        "scientific_result": True,
        "provenance": dict(config["provenance"]),
        "methods": {
            "helmholtz": str(config["helmholtz"]["formulation"]),
            "helmholtz_discretization": str(
                config["helmholtz"]["discretization"]
            ),
            "multislice": str(config["multislice"]["propagation_operator"]),
            "mapping": "radial_to_cartesian_then_external_passband_then_annular_mean",
            "multislice_mapping": str(config["comparison"]["multislice_order"]),
            "weighted_norm": str(config["comparison"]["weighted_norm"]),
            "alignment": "none",
        },
        "sampling": {
            "helmholtz_case_order": list(
                config["helmholtz"]["fixed_case_order"]
            ),
            "multislice_shape": list(config["multislice"]["shape"]),
            "multislice_dx_m": float(config["multislice"]["dx_m"]),
            "multislice_dz_m": float(config["multislice"]["dz_m"]),
            "cartesian_shape": list(config["comparison"]["cartesian_shape"]),
            "cartesian_dx_m": float(config["comparison"]["cartesian_dx_m"]),
            "annular_bin_count": int(post["radial_radius_m"].size),
            "annular_bin_width_m": float(
                config["comparison"]["annular_bin_width_m"]
            ),
            "annular_bin_counts": post["annular_bin_counts"],
            "observation_z_m": float(config["observation"]["z_m"]),
        },
        "case_controls": case_controls,
        "background_interface_controls": interface_controls,
        "projection_controls": post["projection_controls"],
        "restriction_controls": restriction,
        "annular_constant_max_abs_error": float(
            post["annular_constant_max_abs_error"]
        ),
        "comparisons": comparisons,
        "reference_controls": reference_controls,
        "thresholds": dict(thresholds),
        "status": status,
        "interpretation_code": interpretation,
    }
    selected_maps = {
        "helmholtz_passband_amplitude": np.abs(
            post["selected_maps"]["helmholtz_passband"]
        ),
        "multislice_passband_amplitude": np.abs(
            post["selected_maps"]["multislice_passband"]
        ),
        "normalized_cross_residual": post["selected_maps"][
            "normalized_cross_residual"
        ],
        "cross_phase_difference_rad": post["selected_maps"][
            "cross_phase_difference_rad"
        ],
    }
    result = {
        "metrics": metrics,
        "selected_maps": selected_maps,
        "radial_profiles": {
            "radius_m": post["radial_radius_m"],
            **{
                f"{case_id}_raw": values
                for case_id, values in post["radial_raw"].items()
            },
            **{
                f"{case_id}_passband": values
                for case_id, values in post["radial_pass"].items()
            },
        },
        "native_traces": {
            case_id: {
                "radius_m": values["radius_m"],
                "normalized_total_trace": values["normalized_total_trace"],
                "normalized_scattered_trace": values[
                    "normalized_scattered_trace"
                ],
            }
            for case_id, values in helmholtz_results.items()
        },
        "selected_complex_fields": {
            "helmholtz_reference_passband": post["selected_maps"][
                "helmholtz_passband"
            ],
            "multislice_passband": post["selected_maps"][
                "multislice_passband"
            ],
        },
    }
    _emit(
        progress_callback,
        "postprocessing_completed",
        mesh_passband_radial_l2=float(
            comparisons["mesh"]["passband_radial_l2"]
        ),
        pml_passband_radial_l2=float(
            comparisons["pml"]["passband_radial_l2"]
        ),
        cross_model_passband_radial_l2=cross_passband,
        reference_validated=reference_validated,
        status=status,
        interpretation_code=interpretation,
    )
    return result


def _progress_writer(path: Path) -> ProgressCallback:
    payload: dict[str, Any] = {
        "purpose": "formal_scientific_r10_stage_b_comparison",
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


def _load_and_validate_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    if "__LOCK_AFTER_PREFLIGHT__" in {
        str(provenance["preflight_run"]),
        str(provenance["preflight_metrics_sha256"]),
        str(provenance["preflight_hdf5_sha256"]),
    }:
        raise RuntimeError("formal Stage-B preflight provenance is not locked")
    stage_a_run = PROJECT_ROOT / str(provenance["stage_a_run"])
    stage_a_metrics = stage_a_run / "metrics.json"
    stage_a_h5 = stage_a_run / "outputs" / "exp040_r10_stage_a_repaired.h5"
    if _sha256(stage_a_metrics) != str(provenance["stage_a_metrics_sha256"]):
        raise RuntimeError("Stage-A metrics provenance differs")
    if _sha256(stage_a_h5) != str(
        provenance["stage_a_repaired_hdf5_sha256"]
    ):
        raise RuntimeError("Stage-A repaired HDF5 provenance differs")
    preflight_run = PROJECT_ROOT / str(provenance["preflight_run"])
    preflight_metrics_path = preflight_run / "metrics.json"
    preflight_h5 = (
        preflight_run / "outputs" / "exp040_r10_stage_b_preflight.h5"
    )
    if _sha256(preflight_metrics_path) != str(
        provenance["preflight_metrics_sha256"]
    ):
        raise RuntimeError("Stage-B preflight metrics provenance differs")
    if _sha256(preflight_h5) != str(provenance["preflight_hdf5_sha256"]):
        raise RuntimeError("Stage-B preflight HDF5 provenance differs")
    with preflight_metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if (
        metrics.get("status") != "Passed"
        or metrics.get("interpretation_code")
        != "formal_grid_preflight_passed"
        or metrics.get("formal_stage_b_allowed") is not True
    ):
        raise RuntimeError("Stage-B preflight did not pass the registered gate")
    return metrics


def _write_hdf5(
    path: Path,
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    metrics = result["metrics"]
    save_ptycho_hdf5(
        path,
        instrument={
            "wavelength_m": float(config["physics"]["wavelength_m"]),
            "n_glass": float(config["physics"]["n_glass"]),
            "n_air": float(config["physics"]["n_air"]),
            "sampling": metrics["sampling"],
        },
        sample={
            "type": "single_axisymmetric_air_filled_tgv_in_glass",
            "geometry": dict(config["physics"]),
            "helmholtz_material": {
                "radial_rule": config["helmholtz"]["radial_material_rule"],
                "axial_subnodes": config["helmholtz"][
                    "axial_material_subnodes"
                ],
                "mass_rule": config["helmholtz"]["mass_material_rule"],
            },
        },
        config_yaml=config_to_yaml(dict(config)),
        metadata=dict(metadata),
        metrics=metrics,
    )
    with h5py.File(path, "a") as h5:
        data = h5["entry/data"]
        radial = data.require_group("radial_profiles")
        for name, values in result["radial_profiles"].items():
            radial.create_dataset(name, data=np.asarray(values))
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
        raise RuntimeError(f"Stage-B artifact set differs: {sorted(actual)}")
    for name in (
        "metadata.json",
        "metrics.json",
        "run_state.json",
        "run_progress.json",
    ):
        with (run_dir / name).open("r", encoding="utf-8") as handle:
            json.load(handle)
    for checkpoint in (
        "coarse_nominal",
        "fine_nominal",
        "fine_enlarged_pml",
        "multislice_fine_1024",
    ):
        with np.load(run_dir / "checkpoints" / f"{checkpoint}.npz") as data:
            if "controls_json" not in data:
                raise RuntimeError(f"Checkpoint is incomplete: {checkpoint}")
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
            raise RuntimeError("Stage-B HDF5 entry layout differs")
        if set(entry["data"]) != {
            "native_helmholtz_traces",
            "radial_profiles",
            "selected_complex_fields",
        }:
            raise RuntimeError("Stage-B compact HDF5 data layout differs")
        if "truth" in entry or "reconstruction" in entry:
            raise RuntimeError("Stage-B HDF5 must not claim truth/reconstruction")
    for filename in EXP040_R10_STAGE_B_FIGURE_FILENAMES:
        image = np.asarray(iio.imread(run_dir / "figures" / filename))
        if image.ndim not in (2, 3) or image.size == 0:
            raise RuntimeError(f"Stage-B figure is invalid: {filename}")


def run(config_path: Path) -> Path:
    """Execute the one formal Stage-B scientific comparator."""

    source = config_path.resolve()
    if REGISTERED_CONFIG_SHA256 == "__LOCK_AFTER_PREFLIGHT__":
        raise RuntimeError("formal Stage-B source config has not been locked")
    if _sha256(source) != REGISTERED_CONFIG_SHA256:
        raise ValueError("R10 Stage-B source config hash differs")
    config = load_config(source)
    validate_stage_b_config(config)
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
        result = _run_stage_b(
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
            "diagnostic_stage": "R10_stage_b",
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
            "reference_validated": bool(
                metrics["reference_controls"]["reference_validated"]
            ),
        }
        save_json(run_dir / "metrics.json", metrics)
        save_json(run_dir / "metadata.json", metadata)
        progress("artifacts_writing_started", {})
        save_exp040_r10_stage_b_figures(result, run_dir / "figures")
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
                "reference_validated": bool(
                    metrics["reference_controls"]["reference_validated"]
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

    comparisons = metrics["comparisons"]
    print(f"run_dir: {run_dir.resolve()}", flush=True)
    print(f"stage_b_status: {status}", flush=True)
    print(f"interpretation: {interpretation}", flush=True)
    print(
        "mesh_passband_radial_l2: "
        f"{comparisons['mesh']['passband_radial_l2']:.17g}",
        flush=True,
    )
    print(
        "pml_passband_radial_l2: "
        f"{comparisons['pml']['passband_radial_l2']:.17g}",
        flush=True,
    )
    print(
        "cross_model_passband_radial_l2: "
        f"{comparisons['cross_model']['passband_radial_l2']:.17g}",
        flush=True,
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
