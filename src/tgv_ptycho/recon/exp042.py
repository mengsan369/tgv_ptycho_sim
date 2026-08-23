"""Matched exp040-R8 measurement operator for exp042 known-B recovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.camera import positive_midpoint_pixel_average
from tgv_ptycho.forward.exp040 import center_crop, relative_l2
from tgv_ptycho.forward.integer_shift import (
    shift_field_integer_pixels,
    unshift_field_delta_integer_pixels,
)
from tgv_ptycho.forward.multislice_A import multislice_propagate_streamed_A
from tgv_ptycho.forward.scan import add_integer_pixel_jitter, make_grid_scan
from tgv_ptycho.objects.sample_b import make_random_phase_object
from tgv_ptycho.objects.tgv3d import make_tgv_air_fraction_slice
from tgv_ptycho.objects.tgv_geometry import diameter_profile, midpoint_z_grid
from tgv_ptycho.optics.angular_spectrum import (
    angular_spectrum_propagate,
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)
from tgv_ptycho.optics.fields import make_plane_wave

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]

MATCHED_PIXEL_AVERAGE_READOUT = "positive_pixel_average"
Q1_POINT_MISMATCH_READOUT = "bilinear_midpoint_point_from_q4_nodes"


def _complex_inner_product(
    left: NDArray[np.generic], right: NDArray[np.generic]
) -> np.complex128:
    """Return the Euclidean inner product without a platform BLAS dispatch."""

    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        raise ValueError("inner-product operands must have the same shape.")
    return np.complex128(
        np.sum(
            np.conj(left_values) * right_values,
            dtype=np.complex128,
        )
    )


def _l2_norm(values: NDArray[np.generic]) -> float:
    """Return an L2 norm through the deterministic explicit reduction path."""

    squared = float(_complex_inner_product(values, values).real)
    return float(np.sqrt(max(squared, 0.0)))


def _real_inner_product(
    left: NDArray[np.generic], right: NDArray[np.generic]
) -> float:
    """Return the real Hilbert-space product for complex probe parameters."""

    return float(np.real(_complex_inner_product(left, right)))


def _shape(value: Any, name: str) -> tuple[int, int]:
    values = tuple(int(item) for item in value)
    if len(values) != 2 or min(values) <= 0:
        msg = f"{name} must be a positive (ny, nx) pair."
        raise ValueError(msg)
    return values


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        msg = f"Missing mapping section: {name}."
        raise ValueError(msg)
    return value


def _center_embed(
    values: NDArray[np.generic], target_shape: tuple[int, int]
) -> NDArray[Any]:
    """Apply centered zero embedding on the last two axes."""

    array = np.asarray(values)
    if array.ndim < 2:
        msg = "values must have at least two dimensions."
        raise ValueError(msg)
    target_y, target_x = _shape(target_shape, "target_shape")
    source_y, source_x = array.shape[-2:]
    delta_y = target_y - source_y
    delta_x = target_x - source_x
    if delta_y < 0 or delta_x < 0 or delta_y % 2 or delta_x % 2:
        msg = "target_shape must contain values with aligned centers."
        raise ValueError(msg)
    pad = [(0, 0)] * array.ndim
    pad[-2] = (delta_y // 2, delta_y // 2)
    pad[-1] = (delta_x // 2, delta_x // 2)
    return np.pad(array, pad, mode="constant", constant_values=0)


def _pixel_average_adjoint_euclidean(
    pixel_values: NDArray[np.floating], factor: int
) -> NDArray[np.float64]:
    """Apply the Euclidean transpose of q-by-q positive block averaging."""

    values = np.asarray(pixel_values, dtype=np.float64)
    q = int(factor)
    if values.ndim < 2 or q <= 0:
        msg = "pixel_values must be at least 2D and factor must be positive."
        raise ValueError(msg)
    repeated = np.repeat(np.repeat(values, q, axis=-2), q, axis=-1)
    return np.asarray(repeated / float(q * q), dtype=np.float64)


def _pixel_average_adjoint_physical(
    pixel_values: NDArray[np.floating], factor: int
) -> NDArray[np.float64]:
    """Apply the block-average adjoint under pixel/node area inner products."""

    values = np.asarray(pixel_values, dtype=np.float64)
    q = int(factor)
    if values.ndim < 2 or q <= 0:
        msg = "pixel_values must be at least 2D and factor must be positive."
        raise ValueError(msg)
    return np.repeat(np.repeat(values, q, axis=-2), q, axis=-1).astype(
        np.float64, copy=False
    )


def _bilinear_midpoint_point_field(
    node_field: NDArray[np.generic], factor: int
) -> NDArray[Any]:
    """Interpolate an even q-node block to its complex pixel midpoint.

    For q4 nodes the pixel midpoint lies between the central four nodes.  The
    bilinear interpolant there is their equal-weight complex-field average.
    This is an explicit point-detector diagnostic, not a pixel integral.
    """

    values = np.asarray(node_field)
    q = int(factor)
    if values.ndim < 2 or q <= 0 or q % 2:
        raise ValueError("Point readout requires a positive even factor.")
    if values.shape[-2] % q or values.shape[-1] % q:
        raise ValueError("Node field must contain complete q-by-q blocks.")
    pixel_y = values.shape[-2] // q
    pixel_x = values.shape[-1] // q
    blocked = values.reshape(*values.shape[:-2], pixel_y, q, pixel_x, q)
    center = q // 2
    return np.mean(
        blocked[..., :, center - 1 : center + 1, :, center - 1 : center + 1],
        axis=(-3, -1),
    )


def _bilinear_midpoint_point_field_adjoint(
    pixel_field: NDArray[np.generic], factor: int
) -> NDArray[Any]:
    """Apply the Euclidean transpose of midpoint complex-field interpolation."""

    values = np.asarray(pixel_field)
    q = int(factor)
    if values.ndim < 2 or q <= 0 or q % 2:
        raise ValueError("Point-readout adjoint requires a positive even factor.")
    pixel_y, pixel_x = values.shape[-2:]
    blocked = np.zeros(
        (*values.shape[:-2], pixel_y, q, pixel_x, q), dtype=values.dtype
    )
    center = q // 2
    blocked[
        ..., :, center - 1 : center + 1, :, center - 1 : center + 1
    ] = values[..., :, None, :, None] / 4.0
    return blocked.reshape(
        *values.shape[:-2], pixel_y * q, pixel_x * q
    )


def _validate_initialization_ablation_config(
    reconstruction: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    execution_mode: str,
) -> None:
    settings = _section(verification, "initialization_ablation")
    if settings.get("enabled") is not True:
        raise ValueError("The initialization ablation must be enabled.")
    family = str(settings.get("initialization_family"))
    supported_designs = {
        "homogeneous_plus_zero_mean_complex_perturbations": (
            "initialization_ablation_matched",
            "truth_free_equal_budget_initialization_ablation",
        ),
        "fixed_direction_magnitude_sweep": (
            "initialization_magnitude_ablation_matched",
            "truth_free_equal_budget_initialization_magnitude_ablation",
        ),
    }
    if family not in supported_designs:
        raise ValueError("The initialization family is invalid.")
    expected_mode, expected_role = supported_designs[family]
    if execution_mode != expected_mode or settings.get("role") != expected_role:
        raise ValueError("The initialization ablation mode or role is invalid.")
    if settings.get("equal_optimizer_settings") is not True:
        raise ValueError("All initialization branches must reuse optimizer settings.")
    iteration_budget = int(settings["equal_iteration_budget"])
    if (
        iteration_budget <= 0
        or iteration_budget != float(settings["equal_iteration_budget"])
        or iteration_budget != int(reconstruction["iterations"])
    ):
        raise ValueError("The registered ablation budget must match reconstruction.")
    if float(reconstruction.get("gradient_norm_stop", -1.0)) != 0.0:
        raise ValueError("The ablation cannot use a tuned gradient stopping rule.")
    if reconstruction.get("algorithm") != (
        "batch_complex_gradient_descent_gn_scaled_armijo"
    ):
        raise ValueError("The ablation must retain the registered GN-scaled optimizer.")
    if int(reconstruction.get("curvature_recompute_interval", 0)) != 1:
        raise ValueError("The ablation must recompute GN curvature every iteration.")

    branches = settings.get("branches")
    if not isinstance(branches, list) or not 2 <= len(branches) <= 8:
        raise ValueError("The initialization ablation requires 2 to 8 branches.")
    names: list[str] = []
    perturbation_seeds: list[int] = []
    magnitudes: list[float] = []
    homogeneous_count = 0
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ValueError("Every initialization branch must be a mapping.")
        name = branch.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Initialization branch names must be identifiers.")
        names.append(name)
        kind = branch.get("kind")
        if kind == "homogeneous_reference_probe":
            homogeneous_count += 1
        elif kind == "deterministic_zero_mean_complex_perturbation":
            pass
        else:
            raise ValueError("Unsupported initialization ablation branch kind.")
        if family == "homogeneous_plus_zero_mean_complex_perturbations":
            if kind == "homogeneous_reference_probe":
                if "seed" in branch:
                    raise ValueError("The homogeneous branch cannot define a seed.")
            else:
                seed = int(branch["seed"])
                if seed < 0 or seed != float(branch["seed"]):
                    raise ValueError(
                        "A perturbation seed must be a nonnegative integer."
                    )
                perturbation_seeds.append(seed)
        else:
            magnitude = float(branch["relative_l2_to_homogeneous"])
            if not np.isfinite(magnitude) or not 0.0 <= magnitude < 1.0:
                raise ValueError("A magnitude level must lie in [0, 1).")
            if kind == "homogeneous_reference_probe" and magnitude != 0.0:
                raise ValueError("The homogeneous magnitude must be zero.")
            if kind != "homogeneous_reference_probe" and magnitude <= 0.0:
                raise ValueError("Perturbed magnitudes must be positive.")
            if "seed" in branch:
                raise ValueError("Magnitude branches must reuse the fixed seed.")
            magnitudes.append(magnitude)
    if len(set(names)) != len(names):
        raise ValueError("Initialization branch names must be unique.")
    if perturbation_seeds and len(set(perturbation_seeds)) != len(
        perturbation_seeds
    ):
        raise ValueError("Initialization perturbation seeds must be unique.")
    if homogeneous_count != 1 or settings.get("reference_branch") not in names:
        raise ValueError("The ablation requires one registered reference branch.")
    reference_index = names.index(str(settings["reference_branch"]))
    if branches[reference_index].get("kind") != "homogeneous_reference_probe":
        raise ValueError("The reference branch must use the homogeneous probe.")
    if settings.get("pairwise_alignment") != (
        "global_phase_only_diagnostic_convention"
    ):
        raise ValueError("The pairwise alignment convention is invalid.")
    checkpoint_metrics: list[str] = []
    if family == "homogeneous_plus_zero_mean_complex_perturbations":
        relative_l2_value = float(
            settings["perturbation_relative_l2_to_homogeneous"]
        )
        if not np.isfinite(relative_l2_value) or not 0.0 < relative_l2_value < 1.0:
            raise ValueError("The registered perturbation magnitude is invalid.")
    else:
        if settings.get("sweep_axis") != (
            "perturbation_relative_l2_to_homogeneous"
        ):
            raise ValueError("The magnitude-sweep axis is invalid.")
        registered_levels = [float(value) for value in settings["magnitude_levels"]]
        if (
            registered_levels != magnitudes
            or registered_levels != sorted(set(registered_levels))
            or registered_levels[0] != 0.0
        ):
            raise ValueError("Magnitude levels must be exact, unique, and increasing.")
        fixed_seed = int(settings["fixed_perturbation_seed"])
        if fixed_seed < 0 or fixed_seed != float(settings["fixed_perturbation_seed"]):
            raise ValueError("The fixed perturbation seed is invalid.")
        if settings.get("same_perturbation_direction") is not True:
            raise ValueError("Magnitude branches must reuse one direction.")
        expected_panels = [
            "loss_curve",
            "detector_relative_residual_curve",
            "raw_probe_error_simulation_evaluation_only",
            "aligned_probe_error_simulation_evaluation_only",
            "checkpoint_raw_probe_distance_to_reference",
            "checkpoint_aligned_probe_distance_to_reference",
            "checkpoint_prediction_distance_to_reference",
            "final_aligned_probe_distance_matrix",
        ]
        if settings.get("figure_panels") != expected_panels:
            raise ValueError("The magnitude-ablation figure panels are invalid.")
        direction_tolerance = float(
            settings["same_direction_max_unit_l2_error_tolerance"]
        )
        if not np.isfinite(direction_tolerance) or not (
            0.0 < direction_tolerance <= 1.0e-6
        ):
            raise ValueError("The same-direction tolerance is invalid.")
        checkpoint_interval = int(settings["checkpoint_interval"])
        if (
            checkpoint_interval <= 0
            or checkpoint_interval != float(settings["checkpoint_interval"])
            or iteration_budget % checkpoint_interval
        ):
            raise ValueError("Checkpoint interval must divide the iteration budget.")
        checkpoint_metrics = [
            "checkpoint_pairwise_raw_probe_relative_l2_curve",
            "checkpoint_pairwise_global_phase_aligned_probe_relative_l2_curve",
            "checkpoint_pairwise_prediction_relative_l2_curve",
        ]
    expected_metrics = [
        "loss_curve",
        "detector_relative_residual_curve",
        "probe_raw_relative_l2_curve_simulation_evaluation_only",
        "probe_global_phase_aligned_relative_l2_curve_simulation_evaluation_only",
        "initial_pairwise_raw_probe_relative_l2",
        "initial_pairwise_global_phase_aligned_probe_relative_l2",
        "final_pairwise_raw_probe_relative_l2",
        "final_pairwise_global_phase_aligned_probe_relative_l2",
        "final_pairwise_prediction_relative_l2",
        *checkpoint_metrics,
        "iterations_completed",
        "stopping_reason",
    ]
    if settings.get("common_metrics") != expected_metrics:
        raise ValueError("The registered initialization metrics are invalid.")
    required_false = (
        "truth_used_by_initialization",
        "truth_used_by_optimizer",
        "truth_used_by_branch_selection",
        "truth_used_by_stopping",
        "scientific_thresholds_preregistered",
    )
    if any(settings.get(key) is not False for key in required_false):
        raise ValueError("Truth and scientific thresholds are forbidden here.")
    if settings.get("simulation_truth_evaluation_only") is not True:
        raise ValueError("Truth must remain post-hoc simulation evaluation only.")


def _validate_detector_quadrature_ablation_config(
    reconstruction: Mapping[str, Any],
    verification: Mapping[str, Any],
    detector: Mapping[str, Any],
    *,
    execution_mode: str,
) -> None:
    """Validate the registered q4/q1 data-model pairing control."""

    settings = _section(verification, "detector_quadrature_ablation")
    if execution_mode != "detector_quadrature_three_branch_control":
        raise ValueError("The detector-quadrature ablation mode is invalid.")
    if settings.get("enabled") is not True or settings.get("role") != (
        "truth_free_equal_budget_detector_data_model_pairing_control"
    ):
        raise ValueError("The detector-quadrature ablation role is invalid.")
    if detector.get("model") != "q4_positive_staggered_midpoint_pixel_average":
        raise ValueError("The primary detector model must remain positive q4.")
    if int(detector["quadrature_factor"]) != 4:
        raise ValueError("The q1 diagnostic requires the frozen q4 node grid.")
    if settings.get("primary_data_model") != detector.get("model"):
        raise ValueError("The primary data model must use configured q4.")
    if settings.get("control_data_model") != Q1_POINT_MISMATCH_READOUT:
        raise ValueError("The control data model must use q1 point readout.")
    if settings.get("sweep_axis") != (
        "detector_data_and_reconstruction_readout_pairing"
    ):
        raise ValueError("The detector-ablation sweep axis is invalid.")
    if settings.get("point_interpolation") != (
        "bilinear_average_of_central_2x2_complex_q4_nodes"
    ):
        raise ValueError("The q1 point interpolation rule is invalid.")
    expected_branches = [
        {
            "name": "matched_q4",
            "data_source": "primary_q4_data",
            "data_readout": MATCHED_PIXEL_AVERAGE_READOUT,
            "detector_readout": MATCHED_PIXEL_AVERAGE_READOUT,
            "model_role": "exp040_matched_reference",
        },
        {
            "name": "mismatch_q1_point",
            "data_source": "primary_q4_data",
            "data_readout": MATCHED_PIXEL_AVERAGE_READOUT,
            "detector_readout": Q1_POINT_MISMATCH_READOUT,
            "model_role": "explicit_mismatch_diagnostic",
        },
        {
            "name": "matched_q1_point",
            "data_source": "q1_point_control_data",
            "data_readout": Q1_POINT_MISMATCH_READOUT,
            "detector_readout": Q1_POINT_MISMATCH_READOUT,
            "model_role": "simplified_matched_control_not_exp040_matched",
        },
    ]
    if settings.get("branches") != expected_branches:
        raise ValueError("The registered q4/q1 pairing branches changed.")
    if settings.get("reference_branch") != "matched_q4":
        raise ValueError("The matched q4 branch must remain the reference.")
    if settings.get("primary_q4_data_branches") != [
        "matched_q4",
        "mismatch_q1_point",
    ]:
        raise ValueError("The two primary-q4-data branches changed.")
    expected_comparisons = {
        "matched_q4_vs_mismatch_q1_point": (
            "reconstruction_readout_mismatch_on_shared_q4_data"
        ),
        "mismatch_q1_point_vs_matched_q1_point": (
            "q4_vs_q1_data_with_shared_q1_reconstruction_model"
        ),
        "matched_q4_vs_matched_q1_point": (
            "matched_detector_model_family_effect"
        ),
    }
    if settings.get("comparison_contract") != expected_comparisons:
        raise ValueError("The three-branch comparison contract changed.")
    if settings.get("same_truth_b_scan_initialization") is not True:
        raise ValueError("Non-readout ablation inputs must remain identical.")
    if settings.get("equal_optimizer_settings") is not True:
        raise ValueError("All detector branches must reuse optimizer settings.")
    iteration_budget = int(settings["equal_iteration_budget"])
    if (
        iteration_budget <= 0
        or iteration_budget != float(settings["equal_iteration_budget"])
        or iteration_budget != int(reconstruction["iterations"])
    ):
        raise ValueError("The detector-ablation budget must match reconstruction.")
    checkpoint_interval = int(settings["checkpoint_interval"])
    if (
        checkpoint_interval <= 0
        or checkpoint_interval != float(settings["checkpoint_interval"])
        or iteration_budget % checkpoint_interval
    ):
        raise ValueError("Detector-ablation checkpoint interval is invalid.")
    if reconstruction.get("algorithm") != (
        "batch_complex_gradient_descent_gn_scaled_armijo"
    ):
        raise ValueError("The detector ablation must retain the GN optimizer.")
    if int(reconstruction.get("curvature_recompute_interval", 0)) != 1:
        raise ValueError("GN curvature must be recomputed every iteration.")
    if float(reconstruction.get("gradient_norm_stop", -1.0)) != 0.0:
        raise ValueError("The detector ablation uses the fixed iteration budget.")
    expected_panels = [
        "loss_curve",
        "detector_relative_residual_curve",
        "raw_probe_error_simulation_evaluation_only",
        "aligned_probe_error_simulation_evaluation_only",
        "checkpoint_raw_probe_distance_to_reference",
        "checkpoint_aligned_probe_distance_to_reference",
        "checkpoint_prediction_distance_to_reference",
        "final_aligned_probe_distance_matrix",
    ]
    if settings.get("figure_panels") != expected_panels:
        raise ValueError("The detector-ablation figure panels are invalid.")
    tolerances = _section(settings, "numerical_control_tolerances")
    for name, upper_bound in (
        ("point_readout_adjoint_relative_error", 1.0e-10),
        ("intensity_jacobian_adjoint_relative_error", 1.0e-10),
        ("full_loss_directional_gradient_relative_error", 1.0e-4),
    ):
        value = float(tolerances[name])
        if not np.isfinite(value) or not 0.0 < value <= upper_bound:
            raise ValueError(f"Invalid detector-ablation tolerance: {name}.")
    required_false = (
        "truth_used_by_initialization",
        "truth_used_by_optimizer",
        "truth_used_by_branch_selection",
        "truth_used_by_stopping",
        "scientific_thresholds_preregistered",
        "q1_branch_is_exp040_matched",
    )
    if any(settings.get(key) is not False for key in required_false):
        raise ValueError("Truth and scientific gates are forbidden here.")
    if settings.get("q1_q1_matched_control_included") is not True:
        raise ValueError("The registered q1/q1 matched control is required.")
    if settings.get("simulation_truth_evaluation_only") is not True:
        raise ValueError("Truth must remain post-hoc simulation evaluation only.")


def validate_exp042_config(config: Mapping[str, Any]) -> None:
    """Validate the matched deterministic exp042 development configuration."""

    experiment = _section(config, "experiment")
    provenance = _section(config, "provenance")
    optics = _section(config, "optics")
    sample_a = _section(config, "sample_a")
    probe_grid = _section(config, "probe_grid")
    sample_b = _section(config, "sample_b")
    scan = _section(config, "scan")
    detector = _section(config, "detector")
    loss = _section(config, "loss")
    reconstruction = _section(config, "reconstruction")
    verification = _section(config, "verification")
    execution = _section(config, "execution")

    if experiment.get("id") != "exp042":
        raise ValueError("experiment.id must be exp042.")
    if provenance.get("reference_validated") is not False:
        raise ValueError("exp042 must preserve reference_validated=false.")
    if provenance.get("full_tgv_reference_authorized") is not False:
        raise ValueError(
            "exp042 must preserve full_tgv_reference_authorized=false."
        )
    for name in (
        "wavelength_m",
        "internal_reference_index",
        "external_medium_index",
        "z_AB_m",
        "z_BC_m",
    ):
        value = float(optics[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"optics.{name} must be finite and positive.")
    if optics.get("angular_spectrum_bandlimit") is not True:
        raise ValueError("The matched branch requires bandlimit=true.")
    if optics.get("alias_control_external") is not True:
        raise ValueError("The matched external branch requires alias control.")

    sample_shape = _shape(sample_a["shape"], "sample_a.shape")
    native_shape = _shape(probe_grid["native_shape"], "native_shape")
    open_shape = _shape(probe_grid["open_shape"], "open_shape")
    if sample_shape != native_shape:
        raise ValueError("The development case uses identity A-to-B base mapping.")
    if any(
        (outer - inner) < 0 or (outer - inner) % 2
        for inner, outer in zip(native_shape, open_shape, strict=True)
    ):
        raise ValueError("native_shape must center-align inside open_shape.")
    node_dx = float(probe_grid["node_dx_m"])
    if node_dx <= 0.0 or float(sample_a["dx_m"]) != node_dx:
        raise ValueError("sample-A dx and probe node dx must match exactly.")
    support_shape = _shape(sample_b["support_shape"], "support_shape")
    if any(
        (outer - inner) < 0 or (outer - inner) % 2
        for inner, outer in zip(support_shape, open_shape, strict=True)
    ):
        raise ValueError("finite B support must center-align inside open_shape.")
    if sample_b.get("shifted_quantity") != "B_minus_1":
        raise ValueError("The finite shift must act on B_minus_1.")
    if sample_b.get("shift_boundary") != "constant_zero":
        raise ValueError("The finite-B shift boundary must be constant_zero.")
    if list(sample_b.get("exterior_transmission", [])) != [1.0, 0.0]:
        raise ValueError("The finite-B exterior must be transparent 1+0j.")
    if int(sample_b["feature_size_px"]) <= 0:
        raise ValueError("sample_b.feature_size_px must be positive.")

    q = int(detector["quadrature_factor"])
    roi = _shape(detector["native_roi_shape"], "native_roi_shape")
    if q <= 0 or any(length % q for length in open_shape):
        raise ValueError("open_shape must contain complete q-by-q detector blocks.")
    full_pixel_shape = tuple(length // q for length in open_shape)
    if any(
        (outer - inner) < 0 or (outer - inner) % 2
        for inner, outer in zip(roi, full_pixel_shape, strict=True)
    ):
        raise ValueError("detector ROI must center-align inside the pixel grid.")
    if float(detector["node_dx_m"]) != node_dx:
        raise ValueError("detector and probe node dx must match exactly.")
    if float(detector["pixel_size_m"]) != q * node_dx:
        raise ValueError("detector pixel size must equal q times node dx.")
    if scan.get("integer_pixel_shifts_only") is not True:
        raise ValueError("The development scan must use integer node shifts.")
    if loss.get("amplitude_replacement") is not False:
        raise ValueError("q4 data cannot use node-amplitude replacement.")
    if reconstruction.get("known_sample_b") is not True:
        raise ValueError("The first exp042 baseline requires known sample B.")
    if reconstruction.get("update_sample_b") is not False:
        raise ValueError("The first exp042 baseline must not update sample B.")
    if reconstruction.get("truth_used_by_optimizer") is not False:
        raise ValueError("Simulation truth must not enter the optimizer.")
    if reconstruction.get("initialization") != "homogeneous_reference_probe":
        raise ValueError("The primary initialization must remain homogeneous.")
    execution_mode = str(execution.get("mode"))
    if execution_mode in {
        "initialization_ablation_matched",
        "initialization_magnitude_ablation_matched",
    }:
        _validate_initialization_ablation_config(
            reconstruction,
            verification,
            execution_mode=execution_mode,
        )
        return
    if execution_mode == "detector_quadrature_three_branch_control":
        _validate_detector_quadrature_ablation_config(
            reconstruction,
            verification,
            detector,
            execution_mode=execution_mode,
        )
        return
    stability = _section(reconstruction, "initialization_stability_control")
    if stability.get("enabled") is not True:
        raise ValueError("The initialization stability control must be enabled.")
    if stability.get("role") != "truth_free_local_basin_diagnostic":
        raise ValueError("The stability control role is invalid.")
    if stability.get("initialization") != (
        "deterministic_zero_mean_complex_perturbation_of_primary"
    ):
        raise ValueError("The stability-control initialization is invalid.")
    if stability.get("zero_mean_perturbation") is not True:
        raise ValueError("The stability perturbation must have zero mean.")
    if stability.get("reuse_primary_optimizer_settings") is not True:
        raise ValueError("The stability control must reuse optimizer settings.")
    if stability.get("pairwise_alignment") != "global_phase_only":
        raise ValueError("Only pairwise global-phase alignment is allowed.")
    if stability.get("truth_used_by_initialization") is not False:
        raise ValueError("Truth is forbidden from the stability initialization.")
    stability_seed = int(stability["seed"])
    if stability_seed < 0 or stability_seed != float(stability["seed"]):
        raise ValueError("The stability seed must be a nonnegative integer.")
    stability_relative_l2 = float(stability["relative_l2_to_primary"])
    if not np.isfinite(stability_relative_l2) or not (
        0.0 < stability_relative_l2 < 1.0
    ):
        raise ValueError("The stability perturbation must be finite and local.")
    continuation = _section(reconstruction, "paired_continuation_control")
    if continuation.get("enabled") is not True:
        raise ValueError("The paired continuation control must be enabled.")
    if continuation.get("role") != (
        "truth_free_equal_budget_paired_continuation"
    ):
        raise ValueError("The paired continuation role is invalid.")
    if continuation.get("start_fields") != (
        "same_run_primary_and_control_raw_final"
    ):
        raise ValueError("The continuation must start from same-run raw fields.")
    if continuation.get("reuse_optimizer_settings") is not True:
        raise ValueError("The continuation must reuse optimizer settings.")
    if continuation.get("pairwise_alignment") != "global_phase_only":
        raise ValueError("The continuation pairwise alignment is invalid.")
    if continuation.get("truth_used_by_continuation") is not False:
        raise ValueError("Truth is forbidden from the continuation optimizer.")
    if continuation.get("truth_used_by_diagnostic") is not False:
        raise ValueError("Truth is forbidden from the continuation diagnostic.")
    if continuation.get("scientific_thresholds_preregistered") is not False:
        raise ValueError("No continuation pass/fail threshold is registered.")
    additional_iterations = int(continuation["additional_iterations"])
    if additional_iterations != int(reconstruction["iterations"]):
        raise ValueError("The continuation must use an equal iteration budget.")
    checkpoint_interval = int(continuation["checkpoint_interval"])
    if (
        checkpoint_interval <= 0
        or checkpoint_interval != float(continuation["checkpoint_interval"])
        or additional_iterations % checkpoint_interval
    ):
        raise ValueError(
            "The continuation checkpoint interval must divide its budget."
        )
    algorithm = reconstruction.get("algorithm")
    supported_algorithms = {
        "batch_complex_gradient_descent_armijo",
        "batch_complex_gradient_descent_gn_scaled_armijo",
    }
    if algorithm not in supported_algorithms:
        raise ValueError("Unsupported exp042 reconstruction algorithm.")
    if (
        algorithm == "batch_complex_gradient_descent_gn_scaled_armijo"
        and int(reconstruction.get("curvature_recompute_interval", 0)) != 1
    ):
        raise ValueError(
            "The development GN scale must be recomputed every iteration."
        )
    conditioning = _section(verification, "pairwise_conditioning")
    if conditioning.get("enabled") is not True:
        raise ValueError("The pairwise conditioning diagnostic must be enabled.")
    if conditioning.get("role") != "truth_free_local_jacobian_conditioning":
        raise ValueError("The pairwise conditioning role is invalid.")
    if conditioning.get("base_probe") != "primary_final_reconstruction":
        raise ValueError("The conditioning base probe is invalid.")
    if conditioning.get("direction") != (
        "global_phase_aligned_control_minus_primary"
    ):
        raise ValueError("The pairwise conditioning direction is invalid.")
    if conditioning.get("direction_normalization") != "unit_probe_l2":
        raise ValueError("The conditioning direction normalization is invalid.")
    if conditioning.get("jacobian_output_norm") != (
        "rms_over_detector_stack"
    ):
        raise ValueError("The conditioning Jacobian norm is invalid.")
    if conditioning.get("random_direction_distribution") != (
        "zero_mean_complex_gaussian"
    ):
        raise ValueError("The conditioning random-direction family is invalid.")
    if conditioning.get("include_primary_gradient_direction") is not True:
        raise ValueError("The primary gradient comparator must be enabled.")
    if (
        conditioning.get("include_native_global_phase_rotation_comparator")
        is not True
    ):
        raise ValueError(
            "The native global-phase rotation comparator must be enabled."
        )
    if conditioning.get("truth_used_by_diagnostic") is not False:
        raise ValueError("Truth is forbidden from the conditioning diagnostic.")
    if conditioning.get("scientific_thresholds_preregistered") is not False:
        raise ValueError("No conditioning pass/fail threshold is registered.")
    conditioning_seed = int(conditioning["random_seed"])
    if conditioning_seed < 0 or conditioning_seed != float(
        conditioning["random_seed"]
    ):
        raise ValueError("The conditioning seed must be a nonnegative integer.")
    conditioning_count = int(conditioning["random_direction_count"])
    if conditioning_count < 1 or conditioning_count > 64:
        raise ValueError("The conditioning random count must lie in [1, 64].")

    if execution.get("mode") != "local_spectral_diagnostic_from_prior_run":
        raise ValueError("The current exp042 config requires spectral-only execution.")
    spectral = _section(verification, "local_spectral_diagnostic")
    if spectral.get("enabled") is not True:
        raise ValueError("The local spectral diagnostic must be enabled.")
    if spectral.get("role") != "truth_free_matrix_free_local_jtj_lanczos":
        raise ValueError("The local spectral diagnostic role is invalid.")
    if spectral.get("base_probe") != "source_primary_continuation_final_raw":
        raise ValueError("The local spectral base probe is invalid.")
    if spectral.get("direction") != (
        "global_phase_aligned_source_control_continuation_final_minus_primary"
    ):
        raise ValueError("The local spectral pairwise direction is invalid.")
    if spectral.get("matrix_operator") != (
        "real_intensity_jacobian_transpose_times_jacobian"
    ):
        raise ValueError("The local spectral matrix operator is invalid.")
    if spectral.get("matrix_normalization") != "mean_over_detector_stack":
        raise ValueError("The local spectral normalization is invalid.")
    if spectral.get("start_vectors") != [
        "pairwise_direction",
        "deterministic_zero_mean_complex_gaussian",
    ]:
        raise ValueError("The local spectral start vectors are invalid.")
    if spectral.get("reorthogonalization") != "full_two_pass":
        raise ValueError("The local spectral reorthogonalization is invalid.")
    if spectral.get("truth_used_by_diagnostic") is not False:
        raise ValueError("Truth is forbidden from the spectral diagnostic.")
    if spectral.get("scientific_thresholds_preregistered") is not False:
        raise ValueError("No spectral scientific threshold is registered.")
    krylov_dimension = int(spectral["krylov_dimension"])
    if (
        krylov_dimension < 2
        or krylov_dimension > 64
        or krylov_dimension != float(spectral["krylov_dimension"])
    ):
        raise ValueError("The Krylov dimension must lie in [2, 64].")
    reorthogonalization_passes = int(spectral["reorthogonalization_passes"])
    if reorthogonalization_passes != 2:
        raise ValueError("The spectral diagnostic requires two reorthogonalizations.")
    breakdown_tolerance = float(spectral["breakdown_relative_tolerance"])
    if not np.isfinite(breakdown_tolerance) or not (
        0.0 < breakdown_tolerance < 1.0
    ):
        raise ValueError("The Krylov breakdown tolerance must lie in (0, 1).")
    low_ritz_count = int(spectral["reported_low_ritz_count"])
    if low_ritz_count < 1 or low_ritz_count > krylov_dimension:
        raise ValueError("reported_low_ritz_count exceeds the Krylov dimension.")
    spectral_seed = int(spectral["random_seed"])
    if spectral_seed < 0 or spectral_seed != float(spectral["random_seed"]):
        raise ValueError("The spectral seed must be a nonnegative integer.")
    for key in (
        "source_run",
        "source_hdf5_relative_path",
        "source_primary_probe_hdf5_path",
        "source_control_probe_hdf5_path",
    ):
        if not isinstance(spectral.get(key), str) or not spectral[key]:
            raise ValueError(f"local_spectral_diagnostic.{key} is required.")
    for key in (
        "source_config_sha256",
        "source_metrics_sha256",
        "source_hdf5_sha256",
    ):
        digest = str(spectral.get(key, ""))
        if len(digest) != 64 or any(
            character not in "0123456789ABCDEF" for character in digest
        ):
            raise ValueError(f"local_spectral_diagnostic.{key} is invalid.")
    convergence = _section(spectral, "dimension_convergence_control")
    if convergence.get("enabled") is not True:
        raise ValueError("The dimension convergence control must be enabled.")
    if convergence.get("role") != (
        "bounded_12_to_24_krylov_dimension_extension"
    ):
        raise ValueError("The dimension convergence role is invalid.")
    reference_dimension = int(convergence["reference_krylov_dimension"])
    extended_dimension = int(convergence["extended_krylov_dimension"])
    prefix_dimension = int(convergence["recurrence_prefix_dimension"])
    if (
        reference_dimension != 12
        or extended_dimension != 24
        or prefix_dimension != reference_dimension
        or krylov_dimension != extended_dimension
    ):
        raise ValueError("The registered Krylov extension must be 12 to 24.")
    expected_quantities = [
        "recurrence_prefix",
        "lowest_ritz_value",
        "lowest_ritz_residual",
        "lowest_ritz_residual_to_value",
        "lowest_ritz_spectral_weight",
        "lowest_three_ritz_spectral_weight",
        "spectral_weighted_median_ritz_value",
    ]
    if convergence.get("comparison_quantities") != expected_quantities:
        raise ValueError("The dimension convergence quantities are invalid.")
    if convergence.get("same_source_operator_starts_and_seed") is not True:
        raise ValueError("The dimension extension must reuse source and starts.")
    if convergence.get("single_extension_only") is not True:
        raise ValueError("Only one bounded dimension extension is allowed.")
    if convergence.get("truth_used_by_control") is not False:
        raise ValueError("Truth is forbidden from the convergence control.")
    if convergence.get("scientific_thresholds_preregistered") is not False:
        raise ValueError("No convergence scientific threshold is registered.")
    for key in ("reference_run", "reference_hdf5_relative_path"):
        if not isinstance(convergence.get(key), str) or not convergence[key]:
            raise ValueError(f"dimension_convergence_control.{key} is required.")
    for key in (
        "reference_config_sha256",
        "reference_metrics_sha256",
        "reference_hdf5_sha256",
    ):
        digest = str(convergence.get(key, ""))
        if len(digest) != 64 or any(
            character not in "0123456789ABCDEF" for character in digest
        ):
            raise ValueError(f"dimension_convergence_control.{key} is invalid.")


@dataclass(frozen=True)
class MatchedKnownBProbeOperator:
    """Exp040-R8 open operator with exact matched or diagnostic readout adjoint."""

    positions_m: NDArray[np.float64]
    node_dx_m: float
    quadrature_factor: int
    native_shape: tuple[int, int]
    open_shape: tuple[int, int]
    detector_roi_shape: tuple[int, int]
    homogeneous_probe_native: NDArray[np.complex128]
    homogeneous_probe_open: NDArray[np.complex128]
    homogeneous_detector_open: NDArray[np.complex128]
    finite_b_modulation_open: NDArray[np.complex128]
    transfer_bc: NDArray[np.complex128]
    detector_readout: str = MATCHED_PIXEL_AVERAGE_READOUT

    def __post_init__(self) -> None:
        supported = {
            MATCHED_PIXEL_AVERAGE_READOUT,
            Q1_POINT_MISMATCH_READOUT,
        }
        if self.detector_readout not in supported:
            raise ValueError("Unsupported exp042 detector readout.")
        if (
            self.detector_readout == Q1_POINT_MISMATCH_READOUT
            and self.quadrature_factor % 2
        ):
            raise ValueError("The q1 midpoint proxy requires an even node factor.")

    def _readout_intensity(
        self, detector_field: ComplexArray
    ) -> NDArray[np.float64]:
        field = np.asarray(detector_field, dtype=np.complex128)
        if self.detector_readout == MATCHED_PIXEL_AVERAGE_READOUT:
            values = positive_midpoint_pixel_average(
                np.abs(field) ** 2, self.quadrature_factor
            )
        else:
            point_field = _bilinear_midpoint_point_field(
                field, self.quadrature_factor
            )
            values = np.abs(point_field) ** 2
        return np.asarray(values, dtype=np.float64)

    def _readout_intensity_direction(
        self,
        detector_field: ComplexArray,
        detector_direction: ComplexArray,
    ) -> NDArray[np.float64]:
        field = np.asarray(detector_field, dtype=np.complex128)
        direction = np.asarray(detector_direction, dtype=np.complex128)
        if field.shape != direction.shape:
            raise ValueError("Detector field and direction shapes must match.")
        if self.detector_readout == MATCHED_PIXEL_AVERAGE_READOUT:
            node_derivative = 2.0 * np.real(np.conj(field) * direction)
            values = positive_midpoint_pixel_average(
                node_derivative, self.quadrature_factor
            )
        else:
            point_field = _bilinear_midpoint_point_field(
                field, self.quadrature_factor
            )
            point_direction = _bilinear_midpoint_point_field(
                direction, self.quadrature_factor
            )
            values = 2.0 * np.real(np.conj(point_field) * point_direction)
        return np.asarray(values, dtype=np.float64)

    def _readout_intensity_adjoint_to_nodes(
        self,
        detector_field: ComplexArray,
        pixel_values: FloatArray,
    ) -> NDArray[np.complex128]:
        field = np.asarray(detector_field, dtype=np.complex128)
        values = np.asarray(pixel_values, dtype=np.float64)
        expected_pixel_shape = tuple(
            length // self.quadrature_factor for length in self.open_shape
        )
        if field.shape != self.open_shape or values.shape != expected_pixel_shape:
            raise ValueError("Detector field or pixel values have the wrong shape.")
        if self.detector_readout == MATCHED_PIXEL_AVERAGE_READOUT:
            residual_nodes = _pixel_average_adjoint_euclidean(
                values, self.quadrature_factor
            )
            return np.asarray(
                2.0 * residual_nodes * field, dtype=np.complex128
            )
        point_field = _bilinear_midpoint_point_field(
            field, self.quadrature_factor
        )
        point_gradient = 2.0 * values * point_field
        return np.asarray(
            _bilinear_midpoint_point_field_adjoint(
                point_gradient, self.quadrature_factor
            ),
            dtype=np.complex128,
        )

    def _shifted_modulation(self, scan_index: int) -> NDArray[np.complex128]:
        return shift_field_integer_pixels(
            self.finite_b_modulation_open,
            self.positions_m[scan_index],
            self.node_dx_m,
            boundary="constant",
            fill_value=0.0j,
        )

    def detector_field(
        self, probe_native: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        """Return q-node detector field for one scan, including reference."""

        probe = np.asarray(probe_native, dtype=np.complex128)
        if probe.shape != self.native_shape:
            raise ValueError("probe_native has the wrong shape.")
        delta_open = _center_embed(
            probe - self.homogeneous_probe_native, self.open_shape
        )
        probe_open = self.homogeneous_probe_open + delta_open
        modulation = self._shifted_modulation(scan_index)
        residual_exit = delta_open + probe_open * modulation
        residual_detector = apply_angular_spectrum_transfer(
            residual_exit, self.transfer_bc
        )
        return self.homogeneous_detector_open + residual_detector

    def predict_frame(
        self, probe_native: ComplexArray, scan_index: int
    ) -> NDArray[np.float64]:
        """Return one native detector pixel-intensity frame."""

        detector_field = self.detector_field(probe_native, scan_index)
        pixels = self._readout_intensity(detector_field)
        return np.asarray(
            center_crop(pixels, self.detector_roi_shape), dtype=np.float64
        )

    def predict_stack(self, probe_native: ComplexArray) -> NDArray[np.float64]:
        """Return all detector pixel-intensity frames for this readout."""

        return np.stack(
            [
                self.predict_frame(probe_native, scan_index)
                for scan_index in range(len(self.positions_m))
            ]
        )

    def intensity_jacobian_direction(
        self, probe_native: ComplexArray, direction_native: ComplexArray
    ) -> NDArray[np.float64]:
        """Apply the real pixel-intensity Jacobian to a probe direction."""

        probe = np.asarray(probe_native, dtype=np.complex128)
        direction = np.asarray(direction_native, dtype=np.complex128)
        if (
            probe.shape != self.native_shape
            or direction.shape != self.native_shape
        ):
            raise ValueError(
                "probe_native or direction_native has the wrong shape."
            )
        frames: list[NDArray[np.float64]] = []
        for scan_index in range(len(self.positions_m)):
            detector_field = self.detector_field(probe, scan_index)
            detector_direction = self.linear_detector_field(
                direction, scan_index
            )
            full_pixels = self._readout_intensity_direction(
                detector_field, detector_direction
            )
            frames.append(
                np.asarray(
                    center_crop(full_pixels, self.detector_roi_shape),
                    dtype=np.float64,
                )
            )
        return np.stack(frames)

    def intensity_jacobian_adjoint(
        self, probe_native: ComplexArray, detector_residual: FloatArray
    ) -> NDArray[np.complex128]:
        """Apply the real transpose of the pixel-intensity Jacobian.

        The paired products are ``sum((J d) * r)`` in detector space and
        ``Re(sum(conj(d) * J^T r))`` in the complex probe parameterization.
        """

        probe = np.asarray(probe_native, dtype=np.complex128)
        residual = np.asarray(detector_residual, dtype=np.float64)
        expected_shape = (len(self.positions_m), *self.detector_roi_shape)
        if probe.shape != self.native_shape or residual.shape != expected_shape:
            raise ValueError("probe_native or detector_residual has the wrong shape.")
        if not np.all(np.isfinite(probe)) or not np.all(np.isfinite(residual)):
            raise ValueError("probe_native and detector_residual must be finite.")
        full_pixel_shape = tuple(
            length // self.quadrature_factor for length in self.open_shape
        )
        adjoint = np.zeros(self.native_shape, dtype=np.complex128)
        for scan_index in range(len(self.positions_m)):
            detector_field = self.detector_field(probe, scan_index)
            residual_full = _center_embed(
                residual[scan_index], full_pixel_shape
            )
            adjoint += self.linear_detector_field_adjoint(
                self._readout_intensity_adjoint_to_nodes(
                    detector_field, residual_full
                ),
                scan_index,
            )
        return adjoint

    def linear_detector_field(
        self, delta_probe_native: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        """Apply the linear probe-to-detector-field Jacobian for one scan."""

        delta = np.asarray(delta_probe_native, dtype=np.complex128)
        if delta.shape != self.native_shape:
            raise ValueError("delta_probe_native has the wrong shape.")
        embedded = _center_embed(delta, self.open_shape)
        transmission = 1.0 + self._shifted_modulation(scan_index)
        return apply_angular_spectrum_transfer(
            transmission * embedded, self.transfer_bc
        )

    def linear_detector_field_adjoint(
        self, detector_delta: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        """Apply the exact conjugate transpose of the field Jacobian."""

        delta = np.asarray(detector_delta, dtype=np.complex128)
        if delta.shape != self.open_shape:
            raise ValueError("detector_delta has the wrong shape.")
        backpropagated = apply_angular_spectrum_transfer(
            delta, np.conj(self.transfer_bc)
        )
        transmission = 1.0 + self._shifted_modulation(scan_index)
        native = center_crop(
            np.conj(transmission) * backpropagated, self.native_shape
        )
        return np.asarray(native, dtype=np.complex128)

    def loss_and_gradient(
        self, probe_native: ComplexArray, measured: FloatArray
    ) -> tuple[float, NDArray[np.complex128], NDArray[np.float64]]:
        """Return mean pixel-intensity loss, real-complex gradient, prediction."""

        probe = np.asarray(probe_native, dtype=np.complex128)
        data = np.asarray(measured, dtype=np.float64)
        expected_shape = (len(self.positions_m), *self.detector_roi_shape)
        if probe.shape != self.native_shape or data.shape != expected_shape:
            raise ValueError("probe or measured data has the wrong shape.")
        prediction = np.empty(expected_shape, dtype=np.float64)
        gradient = np.zeros(self.native_shape, dtype=np.complex128)
        loss_sum = 0.0
        full_pixel_shape = tuple(
            length // self.quadrature_factor for length in self.open_shape
        )
        normalization = float(data.size)
        for scan_index in range(len(self.positions_m)):
            detector_field = self.detector_field(probe, scan_index)
            full_pixels = self._readout_intensity(detector_field)
            frame = np.asarray(
                center_crop(full_pixels, self.detector_roi_shape),
                dtype=np.float64,
            )
            prediction[scan_index] = frame
            residual = frame - data[scan_index]
            loss_sum += 0.5 * float(np.sum(residual**2))
            residual_full = _center_embed(residual, full_pixel_shape)
            detector_gradient = self._readout_intensity_adjoint_to_nodes(
                detector_field, residual_full
            )
            gradient += self.linear_detector_field_adjoint(
                detector_gradient, scan_index
            )
        return loss_sum / normalization, gradient / normalization, prediction


def make_detector_readout_operator(
    reference: MatchedKnownBProbeOperator, detector_readout: str
) -> MatchedKnownBProbeOperator:
    """Return an operator sharing every linear component but the readout rule."""

    if not isinstance(reference, MatchedKnownBProbeOperator):
        raise TypeError("reference must be a MatchedKnownBProbeOperator.")
    return replace(reference, detector_readout=str(detector_readout))


def _make_positions(config: Mapping[str, Any], node_dx_m: float) -> NDArray[np.float64]:
    scan = _section(config, "scan")
    positions = make_grid_scan(
        int(scan["num_x"]),
        int(scan["num_y"]),
        float(scan["step_m"]),
        center=bool(scan["center"]),
    )
    positions = add_integer_pixel_jitter(
        positions,
        float(scan["jitter_quantum_m"]),
        int(scan["max_jitter_px"]),
        seed=int(scan["jitter_seed"]),
    )
    shifts = positions / node_dx_m
    if not np.allclose(shifts, np.rint(shifts), rtol=0.0, atol=1.0e-9):
        raise ValueError("All exp042 positions must be integer node shifts.")
    return positions


def build_matched_development_case(config: Mapping[str, Any]) -> dict[str, Any]:
    """Generate one deterministic 3D-TGV truth/data/operator matched case."""

    validate_exp042_config(config)
    optics = _section(config, "optics")
    illumination = _section(config, "illumination")
    sample_a = _section(config, "sample_a")
    probe_grid = _section(config, "probe_grid")
    sample_b = _section(config, "sample_b")
    detector = _section(config, "detector")

    native_shape = _shape(probe_grid["native_shape"], "native_shape")
    open_shape = _shape(probe_grid["open_shape"], "open_shape")
    node_dx = float(probe_grid["node_dx_m"])
    wavelength = float(optics["wavelength_m"])
    n_ref = float(optics["internal_reference_index"])
    n_external = float(optics["external_medium_index"])
    thickness = float(sample_a["thickness_m"])
    z_m, slice_widths = midpoint_z_grid(
        thickness, float(sample_a["target_dz_m"])
    )
    diameters = diameter_profile(
        z_m,
        thickness,
        float(sample_a["d_top_m"]),
        float(sample_a["d_waist_m"]),
        float(sample_a["d_bottom_m"]),
        float(sample_a["z_waist_m"]),
    )

    incident_native = make_plane_wave(
        native_shape,
        node_dx,
        wavelength,
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    homogeneous_exit_native = angular_spectrum_propagate(
        incident_native,
        node_dx,
        wavelength,
        thickness,
        n=n_ref,
        bandlimit=True,
        alias_control=False,
    )
    n_glass = float(sample_a["n_glass"])
    n_air = float(sample_a["n_air"])
    interface_factor = int(sample_a["interface_factor"])
    center_xy = tuple(float(value) for value in sample_a["center_xy_m"])

    def n_slices() -> Any:
        for diameter in diameters:
            fraction = make_tgv_air_fraction_slice(
                native_shape,
                node_dx,
                float(diameter),
                interface_factor,
                center_xy,
            )
            yield n_glass + fraction * (n_air - n_glass)

    a_exit_true = multislice_propagate_streamed_A(
        incident_native,
        n_slices(),
        node_dx,
        slice_widths,
        wavelength,
        n_ref=n_ref,
        bandlimit=True,
        alias_control=False,
    )
    transfer_ab_native = make_angular_spectrum_transfer(
        native_shape,
        node_dx,
        wavelength,
        float(optics["z_AB_m"]),
        n=n_external,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_probe_native = apply_angular_spectrum_transfer(
        homogeneous_exit_native, transfer_ab_native
    )
    p_b_true = homogeneous_probe_native + apply_angular_spectrum_transfer(
        a_exit_true - homogeneous_exit_native, transfer_ab_native
    )

    incident_open = make_plane_wave(
        open_shape,
        node_dx,
        wavelength,
        theta_x=float(illumination["theta_x_rad"]),
        theta_y=float(illumination["theta_y_rad"]),
        amplitude=float(illumination["amplitude"]),
    )
    homogeneous_exit_open = angular_spectrum_propagate(
        incident_open,
        node_dx,
        wavelength,
        thickness,
        n=n_ref,
        bandlimit=True,
        alias_control=False,
    )
    transfer_ab_open = make_angular_spectrum_transfer(
        open_shape,
        node_dx,
        wavelength,
        float(optics["z_AB_m"]),
        n=n_external,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_probe_open = apply_angular_spectrum_transfer(
        homogeneous_exit_open, transfer_ab_open
    )
    transfer_bc = make_angular_spectrum_transfer(
        open_shape,
        node_dx,
        wavelength,
        float(optics["z_BC_m"]),
        n=n_external,
        bandlimit=True,
        alias_control=True,
    )
    homogeneous_detector_open = apply_angular_spectrum_transfer(
        homogeneous_probe_open, transfer_bc
    )

    support_shape = _shape(sample_b["support_shape"], "support_shape")
    b_support = make_random_phase_object(
        support_shape,
        phase_range=float(sample_b["phase_range_rad"]),
        seed=int(sample_b["seed"]),
        feature_size_px=int(sample_b["feature_size_px"]),
    )
    modulation_open = np.asarray(
        _center_embed(b_support - 1.0, open_shape), dtype=np.complex128
    )
    positions = _make_positions(config, node_dx)
    operator = MatchedKnownBProbeOperator(
        positions_m=positions,
        node_dx_m=node_dx,
        quadrature_factor=int(detector["quadrature_factor"]),
        native_shape=native_shape,
        open_shape=open_shape,
        detector_roi_shape=_shape(
            detector["native_roi_shape"], "native_roi_shape"
        ),
        homogeneous_probe_native=np.asarray(
            homogeneous_probe_native, dtype=np.complex128
        ),
        homogeneous_probe_open=np.asarray(
            homogeneous_probe_open, dtype=np.complex128
        ),
        homogeneous_detector_open=np.asarray(
            homogeneous_detector_open, dtype=np.complex128
        ),
        finite_b_modulation_open=modulation_open,
        transfer_bc=np.asarray(transfer_bc, dtype=np.complex128),
    )
    intensity = operator.predict_stack(p_b_true)
    return {
        "operator": operator,
        "I_stack": intensity,
        "scan_positions": positions,
        "P_B_true": np.asarray(p_b_true, dtype=np.complex128),
        "B_true": np.asarray(1.0 + modulation_open, dtype=np.complex128),
        "B_support_true": np.asarray(b_support, dtype=np.complex128),
        "U_A_exit_true": np.asarray(a_exit_true, dtype=np.complex128),
        "homogeneous_probe_native": np.asarray(
            homogeneous_probe_native, dtype=np.complex128
        ),
        "z_m": z_m,
        "slice_widths_m": slice_widths,
        "D_z_m": diameters,
    }


def linear_adjoint_relative_error(
    operator: MatchedKnownBProbeOperator, seed: int
) -> float:
    """Return the weighted complex dot-test error of the full field Jacobian."""

    rng = np.random.default_rng(seed)
    probe_delta = rng.normal(size=operator.native_shape) + 1j * rng.normal(
        size=operator.native_shape
    )
    detector_delta = rng.normal(
        size=(len(operator.positions_m), *operator.open_shape)
    ) + 1j * rng.normal(size=(len(operator.positions_m), *operator.open_shape))
    forward = np.stack(
        [
            operator.linear_detector_field(probe_delta, scan_index)
            for scan_index in range(len(operator.positions_m))
        ]
    )
    adjoint = sum(
        (
            operator.linear_detector_field_adjoint(
                detector_delta[scan_index], scan_index
            )
            for scan_index in range(len(operator.positions_m))
        ),
        start=np.zeros(operator.native_shape, dtype=np.complex128),
    )
    node_area = operator.node_dx_m**2
    left = _complex_inner_product(forward, detector_delta) * node_area
    right = _complex_inner_product(probe_delta, adjoint) * node_area
    return float(
        abs(left - right)
        / max(abs(left), abs(right), np.finfo(np.float64).eps)
    )


def intensity_jacobian_adjoint_relative_error(
    operator: MatchedKnownBProbeOperator,
    probe: ComplexArray,
    seed: int,
) -> float:
    """Return the real-inner-product dot-test error for ``J`` and ``J^T``."""

    probe_values = np.asarray(probe, dtype=np.complex128)
    if probe_values.shape != operator.native_shape:
        raise ValueError("probe has the wrong shape.")
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=operator.native_shape) + 1j * rng.normal(
        size=operator.native_shape
    )
    residual = rng.normal(
        size=(len(operator.positions_m), *operator.detector_roi_shape)
    )
    jacobian_direction = operator.intensity_jacobian_direction(
        probe_values, direction
    )
    jacobian_adjoint = operator.intensity_jacobian_adjoint(
        probe_values, residual
    )
    left = float(np.sum(jacobian_direction * residual, dtype=np.float64))
    right = _real_inner_product(direction, jacobian_adjoint)
    return float(
        abs(left - right)
        / max(abs(left), abs(right), np.finfo(np.float64).eps)
    )


def shift_adjoint_relative_error(
    operator: MatchedKnownBProbeOperator, seed: int
) -> float:
    """Return a constant-zero shift/adjoint complex dot-test error."""

    rng = np.random.default_rng(seed)
    source = rng.normal(size=operator.open_shape) + 1j * rng.normal(
        size=operator.open_shape
    )
    target = rng.normal(size=operator.open_shape) + 1j * rng.normal(
        size=operator.open_shape
    )
    position = operator.positions_m[-1]
    shifted = shift_field_integer_pixels(
        source,
        position,
        operator.node_dx_m,
        boundary="constant",
        fill_value=0.0j,
    )
    unshifted = unshift_field_delta_integer_pixels(
        target, position, operator.node_dx_m, boundary="constant"
    )
    left = _complex_inner_product(shifted, target)
    right = _complex_inner_product(source, unshifted)
    return float(
        abs(left - right)
        / max(abs(left), abs(right), np.finfo(np.float64).eps)
    )


def quadrature_adjoint_relative_error(
    operator: MatchedKnownBProbeOperator, seed: int
) -> float:
    """Return q4 dot-test error using physical pixel/node area weights."""

    rng = np.random.default_rng(seed)
    nodes = rng.normal(size=operator.open_shape)
    pixel_shape = tuple(
        length // operator.quadrature_factor for length in operator.open_shape
    )
    pixels = rng.normal(size=pixel_shape)
    averaged = positive_midpoint_pixel_average(
        nodes, operator.quadrature_factor
    )
    physical_adjoint = _pixel_average_adjoint_physical(
        pixels, operator.quadrature_factor
    )
    node_area = operator.node_dx_m**2
    pixel_area = (operator.node_dx_m * operator.quadrature_factor) ** 2
    left = float(_complex_inner_product(averaged, pixels).real * pixel_area)
    right = float(
        _complex_inner_product(nodes, physical_adjoint).real * node_area
    )
    return abs(left - right) / max(
        abs(left), abs(right), np.finfo(np.float64).eps
    )


def point_readout_adjoint_relative_error(
    operator: MatchedKnownBProbeOperator, seed: int
) -> float:
    """Return the complex Euclidean dot-test error for q1 point interpolation."""

    if operator.detector_readout != Q1_POINT_MISMATCH_READOUT:
        raise ValueError("The point-readout dot test requires the q1 diagnostic.")
    rng = np.random.default_rng(seed)
    nodes = rng.normal(size=operator.open_shape) + 1j * rng.normal(
        size=operator.open_shape
    )
    pixel_shape = tuple(
        length // operator.quadrature_factor for length in operator.open_shape
    )
    pixels = rng.normal(size=pixel_shape) + 1j * rng.normal(size=pixel_shape)
    sampled = _bilinear_midpoint_point_field(
        nodes, operator.quadrature_factor
    )
    adjoint = _bilinear_midpoint_point_field_adjoint(
        pixels, operator.quadrature_factor
    )
    left = _complex_inner_product(sampled, pixels)
    right = _complex_inner_product(nodes, adjoint)
    return float(
        abs(left - right)
        / max(abs(left), abs(right), np.finfo(np.float64).eps)
    )


def loss_directional_gradient_relative_error(
    operator: MatchedKnownBProbeOperator,
    probe: ComplexArray,
    measured: FloatArray,
    *,
    seed: int,
    relative_step: float,
) -> float:
    """Check the full pixel-intensity loss gradient by central difference."""

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=operator.native_shape) + 1j * rng.normal(
        size=operator.native_shape
    )
    direction /= _l2_norm(direction)
    probe_values = np.asarray(probe, dtype=np.complex128)
    step = float(relative_step) * max(_l2_norm(probe_values), 1.0)
    loss_plus = operator.loss_and_gradient(
        probe_values + step * direction, measured
    )[0]
    loss_minus = operator.loss_and_gradient(
        probe_values - step * direction, measured
    )[0]
    _, gradient, _ = operator.loss_and_gradient(probe_values, measured)
    finite_difference = (loss_plus - loss_minus) / (2.0 * step)
    analytic = float(np.real(_complex_inner_product(gradient, direction)))
    return float(
        abs(finite_difference - analytic)
        / max(
            abs(finite_difference),
            abs(analytic),
            np.finfo(np.float64).eps,
        )
    )


def intensity_jacobian_directional_relative_error(
    operator: MatchedKnownBProbeOperator,
    probe: ComplexArray,
    direction: ComplexArray,
    *,
    relative_step: float,
) -> float:
    """Check the full q4 intensity Jacobian action by central difference."""

    probe_values = np.asarray(probe, dtype=np.complex128)
    direction_values = np.asarray(direction, dtype=np.complex128)
    direction_norm = _l2_norm(direction_values)
    if direction_norm <= np.finfo(np.float64).eps:
        raise ValueError("direction must have a nonzero finite norm.")
    direction_values = direction_values / direction_norm
    step = float(relative_step) * max(_l2_norm(probe_values), 1.0)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("relative_step must produce a finite positive step.")
    finite_difference = (
        operator.predict_stack(probe_values + step * direction_values)
        - operator.predict_stack(probe_values - step * direction_values)
    ) / (2.0 * step)
    analytic = operator.intensity_jacobian_direction(
        probe_values, direction_values
    )
    difference_norm = _l2_norm(finite_difference - analytic)
    return float(
        difference_norm
        / max(
            _l2_norm(finite_difference),
            _l2_norm(analytic),
            np.finfo(np.float64).eps,
        )
    )


def gauss_newton_directional_curvature(
    operator: MatchedKnownBProbeOperator,
    probe: ComplexArray,
    direction: ComplexArray,
) -> float:
    """Return mean ``|J direction|^2`` for the measured intensity model."""

    jacobian_direction = operator.intensity_jacobian_direction(
        probe, direction
    )
    return float(np.mean(np.square(jacobian_direction), dtype=np.float64))


def gauss_newton_step_diagnostic(
    operator: MatchedKnownBProbeOperator,
    probe: ComplexArray,
    gradient: ComplexArray,
    *,
    fallback_step: float,
) -> dict[str, float | bool]:
    """Derive a truth-free GN step scale from the measurement gradient."""

    fallback = float(fallback_step)
    if not np.isfinite(fallback) or fallback <= 0.0:
        raise ValueError("fallback_step must be finite and positive.")
    gradient_values = np.asarray(gradient, dtype=np.complex128)
    gradient_squared_norm = _l2_norm(gradient_values) ** 2
    curvature = gauss_newton_directional_curvature(
        operator, probe, gradient_values
    )
    use_fallback = bool(
        not np.isfinite(gradient_squared_norm)
        or gradient_squared_norm <= 0.0
        or not np.isfinite(curvature)
        or curvature <= 0.0
    )
    if use_fallback:
        proposed_step = fallback
    else:
        proposed_step = gradient_squared_norm / curvature
        if not np.isfinite(proposed_step) or proposed_step <= 0.0:
            proposed_step = fallback
            use_fallback = True
    return {
        "gradient_squared_norm": float(gradient_squared_norm),
        "gauss_newton_directional_curvature": float(curvature),
        "proposed_step": float(proposed_step),
        "used_fallback": use_fallback,
    }


def operator_consistency_metrics(
    case: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate Stage-A replay, adjoint, gradient, boundary, and fixed point."""

    operator = case["operator"]
    if not isinstance(operator, MatchedKnownBProbeOperator):
        raise TypeError("case.operator has the wrong type.")
    measured = np.asarray(case["I_stack"], dtype=np.float64)
    truth = np.asarray(case["P_B_true"], dtype=np.complex128)
    verification = _section(config, "verification")
    seed = int(verification["random_seed"])
    replay = operator.predict_stack(truth)
    replay_error = relative_l2(replay, measured)
    deterministic_error = relative_l2(operator.predict_stack(truth), replay)
    loss, gradient, _ = operator.loss_and_gradient(truth, measured)
    gradient_step = truth - gradient

    rng = np.random.default_rng(seed + 10)
    gradient_probe = operator.homogeneous_probe_native * (
        1.0
        + 0.02
        * (
            rng.normal(size=operator.native_shape)
            + 1j * rng.normal(size=operator.native_shape)
        )
    )
    q = operator.quadrature_factor
    constant_nodes = np.ones(operator.open_shape, dtype=np.float64)
    constant_pixels = positive_midpoint_pixel_average(constant_nodes, q)
    quadrature_sum_error = abs(
        float(np.sum(constant_pixels)) * q * q
        - float(np.sum(constant_nodes))
    ) / float(np.sum(constant_nodes))
    pixel_size = operator.node_dx_m * q
    node_count = operator.open_shape[0]
    nodes = (
        np.arange(node_count, dtype=np.float64) - (node_count - 1) / 2.0
    ) * operator.node_dx_m
    block_centers = nodes.reshape(node_count // q, q).mean(axis=1)
    pixels = (
        np.arange(node_count // q, dtype=np.float64)
        - (node_count // q - 1) / 2.0
    ) * pixel_size
    geometry_error = float(np.max(np.abs(block_centers - pixels)) / pixel_size)
    boundary_edge_max = 0.0
    for scan_index in range(len(operator.positions_m)):
        shifted = operator._shifted_modulation(scan_index)
        boundary_edge_max = max(
            boundary_edge_max,
            float(
                np.max(
                    np.abs(
                        np.concatenate(
                            [
                                shifted[0],
                                shifted[-1],
                                shifted[:, 0],
                                shifted[:, -1],
                            ]
                        )
                    )
                )
            ),
        )
    jacobian_direction = rng.normal(
        size=operator.native_shape
    ) + 1j * rng.normal(size=operator.native_shape)
    return {
        "truth_replay_relative_l2": replay_error,
        "linear_adjoint_relative_error": linear_adjoint_relative_error(
            operator, seed
        ),
        "intensity_jacobian_adjoint_relative_error": (
            intensity_jacobian_adjoint_relative_error(
                operator, gradient_probe, seed + 4
            )
        ),
        "constant_zero_shift_adjoint_relative_error": (
            shift_adjoint_relative_error(operator, seed + 1)
        ),
        "quadrature_physical_weighted_adjoint_relative_error": (
            quadrature_adjoint_relative_error(operator, seed + 2)
        ),
        "full_loss_directional_gradient_relative_error": (
            loss_directional_gradient_relative_error(
                operator,
                gradient_probe,
                measured,
                seed=seed + 3,
                relative_step=float(
                    verification["finite_difference_relative_step"]
                ),
            )
        ),
        "intensity_jacobian_directional_relative_error": (
            intensity_jacobian_directional_relative_error(
                operator,
                gradient_probe,
                jacobian_direction,
                relative_step=float(
                    verification["finite_difference_relative_step"]
                ),
            )
        ),
        "truth_loss": loss,
        "truth_gradient_l2_norm": _l2_norm(gradient),
        "truth_one_step_relative_change": relative_l2(gradient_step, truth),
        "zero_residual_update_l2_norm": _l2_norm(gradient),
        "detector_constant_max_abs_error": float(
            np.max(np.abs(constant_pixels - 1.0))
        ),
        "detector_sum_relative_error": quadrature_sum_error,
        "detector_node_geometry_normalized_error": geometry_error,
        "detector_minimum_intensity": float(np.min(measured)),
        "detector_all_nonnegative": bool(np.all(measured >= 0.0)),
        "finite_b_shift_boundary_edge_max_abs_modulation": boundary_edge_max,
        "deterministic_repeat_relative_l2": deterministic_error,
        "all_arrays_finite": bool(
            np.all(np.isfinite(measured))
            and np.all(np.isfinite(truth))
            and np.all(np.isfinite(case["B_true"]))
        ),
        "interpolation_mapping": "identity_on_development_native_grid",
        "scientific_pass_fail_conclusion": False,
    }


def detector_quadrature_ablation_consistency_metrics(
    case: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Check q4/q4, q4/q1, and q1/q1 data-model pairing controls."""

    matched = case["operator"]
    if not isinstance(matched, MatchedKnownBProbeOperator):
        raise TypeError("case.operator has the wrong type.")
    if matched.detector_readout != MATCHED_PIXEL_AVERAGE_READOUT:
        raise ValueError("The development truth operator must remain matched q4.")
    point = make_detector_readout_operator(
        matched, Q1_POINT_MISMATCH_READOUT
    )
    measured_q4 = np.asarray(case["I_stack"], dtype=np.float64)
    truth = np.asarray(case["P_B_true"], dtype=np.complex128)
    verification = _section(config, "verification")
    seed = int(verification["random_seed"])
    relative_step = float(verification["finite_difference_relative_step"])
    q4_replay = matched.predict_stack(truth)
    q1_truth_prediction = point.predict_stack(truth)
    q1_self_replay = point.predict_stack(truth)
    q1_self_loss, q1_self_gradient, q1_self_prediction = (
        point.loss_and_gradient(truth, q1_truth_prediction)
    )
    q1_fixed_point = reconstruct_known_b_probe(
        point,
        q1_truth_prediction,
        truth,
        _section(config, "reconstruction"),
    )
    rng = np.random.default_rng(seed + 30)
    gradient_probe = matched.homogeneous_probe_native * (
        1.0
        + 0.02
        * (
            rng.normal(size=matched.native_shape)
            + 1j * rng.normal(size=matched.native_shape)
        )
    )
    jacobian_direction = rng.normal(
        size=matched.native_shape
    ) + 1j * rng.normal(size=matched.native_shape)
    shared_array_fields = (
        "positions_m",
        "homogeneous_probe_native",
        "homogeneous_probe_open",
        "homogeneous_detector_open",
        "finite_b_modulation_open",
        "transfer_bc",
    )
    shared_scalar_fields = (
        "node_dx_m",
        "quadrature_factor",
        "native_shape",
        "open_shape",
        "detector_roi_shape",
    )
    return {
        "matched_q4_truth_replay_exact": bool(
            np.array_equal(q4_replay, measured_q4)
        ),
        "matched_q4_truth_replay_relative_l2": relative_l2(
            q4_replay, measured_q4
        ),
        "matched_q4_quadrature_adjoint_relative_error": (
            quadrature_adjoint_relative_error(matched, seed + 1)
        ),
        "q1_point_readout_adjoint_relative_error": (
            point_readout_adjoint_relative_error(point, seed + 2)
        ),
        "q1_intensity_jacobian_adjoint_relative_error": (
            intensity_jacobian_adjoint_relative_error(
                point, gradient_probe, seed + 3
            )
        ),
        "q1_intensity_jacobian_directional_relative_error": (
            intensity_jacobian_directional_relative_error(
                point,
                gradient_probe,
                jacobian_direction,
                relative_step=relative_step,
            )
        ),
        "q1_full_loss_directional_gradient_relative_error_on_q4_data": (
            loss_directional_gradient_relative_error(
                point,
                gradient_probe,
                measured_q4,
                seed=seed + 4,
                relative_step=relative_step,
            )
        ),
        "q1_full_loss_directional_gradient_relative_error_on_q1_data": (
            loss_directional_gradient_relative_error(
                point,
                gradient_probe,
                q1_truth_prediction,
                seed=seed + 5,
                relative_step=relative_step,
            )
        ),
        "q1_self_truth_replay_exact": bool(
            np.array_equal(q1_self_replay, q1_truth_prediction)
        ),
        "q1_self_truth_replay_relative_l2": relative_l2(
            q1_self_replay, q1_truth_prediction
        ),
        "q1_self_truth_loss": q1_self_loss,
        "q1_self_zero_residual_gradient_l2_norm": _l2_norm(
            q1_self_gradient
        ),
        "q1_self_zero_residual_prediction_exact": bool(
            np.array_equal(q1_self_prediction, q1_truth_prediction)
        ),
        "q1_self_truth_fixed_point_exact": bool(
            q1_fixed_point["iterations_completed"] == 0
            and q1_fixed_point["stopping_reason"] == "gradient_norm"
            and np.array_equal(q1_fixed_point["P_B_rec"], truth)
        ),
        "q1_on_q4_truth_replay_exact": bool(
            np.array_equal(q1_truth_prediction, measured_q4)
        ),
        "q1_on_q4_truth_replay_relative_l2": relative_l2(
            q1_truth_prediction, measured_q4
        ),
        "q1_self_data_relative_l2_to_q4": relative_l2(
            q1_truth_prediction, measured_q4
        ),
        "q1_self_data_all_nonnegative": bool(
            np.all(q1_truth_prediction >= 0.0)
        ),
        "q1_deterministic_repeat_relative_l2": relative_l2(
            point.predict_stack(truth), q1_truth_prediction
        ),
        "prediction_shapes_equal": q1_truth_prediction.shape
        == measured_q4.shape,
        "shared_linear_array_components_identity": all(
            getattr(point, name) is getattr(matched, name)
            for name in shared_array_fields
        ),
        "shared_linear_scalar_components_exact": all(
            getattr(point, name) == getattr(matched, name)
            for name in shared_scalar_fields
        ),
        "only_registered_readout_differs": (
            matched.detector_readout == MATCHED_PIXEL_AVERAGE_READOUT
            and point.detector_readout == Q1_POINT_MISMATCH_READOUT
        ),
        "q1_branch_is_exp040_matched": False,
        "q1_q1_matched_control_included": True,
        "truth_used_by_controls": False,
        "scientific_pass_fail_conclusion": False,
    }


def detector_relative_residual(
    prediction: FloatArray, measured: FloatArray
) -> float:
    """Return the full-stack pixel-intensity residual relative L2."""

    predicted = np.asarray(prediction, dtype=np.float64)
    data = np.asarray(measured, dtype=np.float64)
    if predicted.shape != data.shape:
        raise ValueError("prediction and measured must have the same shape.")
    return relative_l2(predicted, data)


def make_deterministic_complex_probe_initialization(
    primary_probe: ComplexArray,
    *,
    seed: int,
    relative_l2_to_primary: float,
) -> NDArray[np.complex128]:
    """Return a truth-free zero-mean complex perturbation of a primary probe."""

    primary = np.asarray(primary_probe, dtype=np.complex128)
    if primary.ndim != 2 or not np.all(np.isfinite(primary)):
        raise ValueError("primary_probe must be a finite 2D complex field.")
    primary_norm = _l2_norm(primary)
    relative_l2_value = float(relative_l2_to_primary)
    if primary_norm <= np.finfo(np.float64).eps:
        raise ValueError("primary_probe must have a nonzero norm.")
    if not np.isfinite(relative_l2_value) or not (
        0.0 < relative_l2_value < 1.0
    ):
        raise ValueError("relative_l2_to_primary must lie strictly in (0, 1).")
    rng = np.random.default_rng(int(seed))
    perturbation = rng.normal(size=primary.shape) + 1j * rng.normal(
        size=primary.shape
    )
    perturbation -= np.mean(perturbation, dtype=np.complex128)
    perturbation_norm = _l2_norm(perturbation)
    if perturbation_norm <= np.finfo(np.float64).eps:
        raise RuntimeError("The deterministic perturbation has zero norm.")
    perturbation *= relative_l2_value * primary_norm / perturbation_norm
    return np.asarray(primary + perturbation, dtype=np.complex128)


def reconstruct_known_b_probe(
    operator: MatchedKnownBProbeOperator,
    measured: FloatArray,
    init_probe: ComplexArray,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover only ``P_B`` with batch complex gradient descent and Armijo.

    The function accepts no simulation truth.  Its objective uses the measured
    q4 pixel intensities directly, and every gradient is returned through the
    exact adjoint implemented by :class:`MatchedKnownBProbeOperator`.
    """

    algorithm = str(settings.get("algorithm"))
    supported_algorithms = {
        "batch_complex_gradient_descent_armijo",
        "batch_complex_gradient_descent_gn_scaled_armijo",
    }
    if algorithm not in supported_algorithms:
        raise ValueError("Unsupported exp042 reconstruction algorithm.")
    if settings.get("known_sample_b") is not True:
        raise ValueError("known_sample_b must remain true.")
    if settings.get("update_sample_b") is not False:
        raise ValueError("The first exp042 optimizer cannot update sample B.")
    if settings.get("truth_used_by_optimizer") is not False:
        raise ValueError("Truth is forbidden from the exp042 optimizer.")

    iterations = int(settings["iterations"])
    initial_step = float(settings["initial_step"])
    backtracking_factor = float(settings["backtracking_factor"])
    armijo_c = float(settings["armijo_c"])
    max_backtracking = int(settings["max_backtracking_steps"])
    minimum_step = float(settings["minimum_step"])
    gradient_stop = float(settings["gradient_norm_stop"])
    gn_scaled = algorithm == "batch_complex_gradient_descent_gn_scaled_armijo"
    if gn_scaled and int(settings.get("curvature_recompute_interval", 0)) != 1:
        raise ValueError("GN curvature must be recomputed every iteration.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step must be finite and positive.")
    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must lie strictly in (0, 1).")
    if not 0.0 < armijo_c < 1.0:
        raise ValueError("armijo_c must lie strictly in (0, 1).")
    if max_backtracking < 1 or minimum_step <= 0.0 or gradient_stop < 0.0:
        raise ValueError("Invalid backtracking or stopping configuration.")

    data = np.asarray(measured, dtype=np.float64)
    probe = np.asarray(init_probe, dtype=np.complex128).copy()
    if probe.shape != operator.native_shape or not np.all(np.isfinite(probe)):
        raise ValueError("init_probe must be finite and match the native grid.")
    expected_data_shape = (
        len(operator.positions_m),
        *operator.detector_roi_shape,
    )
    if data.shape != expected_data_shape or not np.all(np.isfinite(data)):
        raise ValueError("measured must be finite and match the detector stack.")

    loss, gradient, prediction = operator.loss_and_gradient(probe, data)
    losses = [loss]
    residuals = [detector_relative_residual(prediction, data)]
    gradient_norms = [_l2_norm(gradient)]
    accepted_steps = [0.0]
    backtracking_counts = [0]
    proposed_steps: list[float] = []
    directional_curvatures: list[float] = []
    step_scale_fallbacks: list[bool] = []
    probe_history = [probe.copy()]
    stopping_reason = "iteration_budget"

    for _ in range(iterations):
        gradient_norm = _l2_norm(gradient)
        if gradient_norm <= max(gradient_stop, np.finfo(np.float64).eps):
            stopping_reason = "gradient_norm"
            break
        slope = -(gradient_norm**2)
        if gn_scaled:
            scale = gauss_newton_step_diagnostic(
                operator,
                probe,
                gradient,
                fallback_step=initial_step,
            )
            step = float(scale["proposed_step"])
            directional_curvature = float(
                scale["gauss_newton_directional_curvature"]
            )
            used_fallback = bool(scale["used_fallback"])
        else:
            step = initial_step
            directional_curvature = 0.0
            used_fallback = False
        proposed_step = step
        accepted = False
        candidate_loss = float("nan")
        candidate_gradient = gradient
        candidate_prediction = prediction
        backtracking_count = 0
        for attempt in range(max_backtracking):
            backtracking_count = attempt
            candidate = probe - step * gradient
            (
                candidate_loss,
                candidate_gradient,
                candidate_prediction,
            ) = operator.loss_and_gradient(candidate, data)
            if candidate_loss <= loss + armijo_c * step * slope:
                accepted = True
                break
            step *= backtracking_factor
            if step < minimum_step:
                break
        if not accepted:
            stopping_reason = "line_search_failed"
            break
        probe = candidate
        loss = float(candidate_loss)
        gradient = np.asarray(candidate_gradient, dtype=np.complex128)
        prediction = np.asarray(candidate_prediction, dtype=np.float64)
        losses.append(loss)
        residuals.append(detector_relative_residual(prediction, data))
        gradient_norms.append(_l2_norm(gradient))
        accepted_steps.append(step)
        backtracking_counts.append(backtracking_count)
        proposed_steps.append(proposed_step)
        directional_curvatures.append(directional_curvature)
        step_scale_fallbacks.append(used_fallback)
        probe_history.append(probe.copy())

    return {
        "P_B_rec": probe,
        "P_B_init": np.asarray(init_probe, dtype=np.complex128).copy(),
        "prediction_final": prediction,
        "loss_curve": np.asarray(losses, dtype=np.float64),
        "detector_relative_residual_curve": np.asarray(
            residuals, dtype=np.float64
        ),
        "gradient_l2_norm_curve": np.asarray(
            gradient_norms, dtype=np.float64
        ),
        "accepted_step_curve": np.asarray(accepted_steps, dtype=np.float64),
        "backtracking_count_curve": np.asarray(
            backtracking_counts, dtype=np.int64
        ),
        "proposed_step_curve": np.asarray(proposed_steps, dtype=np.float64),
        "gauss_newton_directional_curvature_curve": np.asarray(
            directional_curvatures, dtype=np.float64
        ),
        "step_scale_fallback_curve": np.asarray(
            step_scale_fallbacks, dtype=np.bool_
        ),
        "step_scale_fallback_count": int(
            np.count_nonzero(step_scale_fallbacks)
        ),
        "total_backtracking_steps": int(np.sum(backtracking_counts)),
        "probe_history": np.stack(probe_history),
        "iterations_completed": len(losses) - 1,
        "stopping_reason": stopping_reason,
        "algorithm": algorithm,
        "truth_used_by_optimizer": False,
        "sample_b_updated": False,
    }


def _align_global_phase(
    probe: ComplexArray, reference: ComplexArray
) -> tuple[NDArray[np.complex128], complex]:
    """Return a global-phase-aligned copy and its unit factor."""

    estimate = np.asarray(probe, dtype=np.complex128)
    target = np.asarray(reference, dtype=np.complex128)
    if estimate.shape != target.shape:
        raise ValueError("probe and reference must have the same shape.")
    inner = _complex_inner_product(estimate, target)
    if abs(inner) <= np.finfo(np.float64).eps:
        factor = 1.0 + 0.0j
    else:
        factor = complex(inner / abs(inner))
    return np.asarray(factor * estimate, dtype=np.complex128), factor


def align_global_phase_simulation_only(
    probe: ComplexArray, truth: ComplexArray
) -> tuple[NDArray[np.complex128], complex]:
    """Return a truth-phase-aligned evaluation copy and its unit factor."""

    return _align_global_phase(probe, truth)


def truth_free_initialization_stability_diagnostic(
    primary_reconstruction: Mapping[str, Any],
    control_reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two reconstructions post hoc without simulation truth."""

    primary_init = np.asarray(
        primary_reconstruction["P_B_init"], dtype=np.complex128
    )
    control_init = np.asarray(
        control_reconstruction["P_B_init"], dtype=np.complex128
    )
    primary_final = np.asarray(
        primary_reconstruction["P_B_rec"], dtype=np.complex128
    )
    control_final = np.asarray(
        control_reconstruction["P_B_rec"], dtype=np.complex128
    )
    control_init_aligned, initial_factor = _align_global_phase(
        control_init, primary_init
    )
    control_final_aligned, final_factor = _align_global_phase(
        control_final, primary_final
    )
    primary_prediction = np.asarray(
        primary_reconstruction["prediction_final"], dtype=np.float64
    )
    control_prediction = np.asarray(
        control_reconstruction["prediction_final"], dtype=np.float64
    )
    primary_loss = float(primary_reconstruction["loss_curve"][-1])
    control_loss = float(control_reconstruction["loss_curve"][-1])
    primary_residual = float(
        primary_reconstruction["detector_relative_residual_curve"][-1]
    )
    control_residual = float(
        control_reconstruction["detector_relative_residual_curve"][-1]
    )
    return {
        "P_B_control_init_global_phase_aligned_to_primary_init": (
            control_init_aligned
        ),
        "P_B_control_rec_global_phase_aligned_to_primary_rec": (
            control_final_aligned
        ),
        "initial_global_phase_alignment_factor": initial_factor,
        "final_global_phase_alignment_factor": final_factor,
        "initial_probe_raw_relative_l2": relative_l2(
            control_init, primary_init
        ),
        "initial_probe_global_phase_aligned_relative_l2": relative_l2(
            control_init_aligned, primary_init
        ),
        "final_probe_raw_relative_l2": relative_l2(
            control_final, primary_final
        ),
        "final_probe_global_phase_aligned_relative_l2": relative_l2(
            control_final_aligned, primary_final
        ),
        "final_prediction_relative_l2": relative_l2(
            control_prediction, primary_prediction
        ),
        "final_loss_absolute_difference": abs(control_loss - primary_loss),
        "final_detector_residual_absolute_difference": abs(
            control_residual - primary_residual
        ),
        "truth_used_by_diagnostic": False,
        "diagnostic_role": "truth_free_posthoc_initialization_stability",
    }


def truth_free_initialization_ablation_diagnostic(
    reconstructions: Mapping[str, Mapping[str, Any]],
    *,
    reference_branch: str,
    operator: MatchedKnownBProbeOperator | None = None,
    checkpoint_interval: int | None = None,
) -> dict[str, Any]:
    """Compare equal-budget initialization branches without simulation truth."""

    branch_names = list(reconstructions)
    if len(branch_names) < 2 or reference_branch not in reconstructions:
        raise ValueError("At least two branches and the reference are required.")
    if len(set(branch_names)) != len(branch_names):
        raise ValueError("Initialization branch names must be unique.")

    initial_probes: list[NDArray[np.complex128]] = []
    final_probes: list[NDArray[np.complex128]] = []
    predictions: list[NDArray[np.float64]] = []
    iterations: list[int] = []
    stopping_reasons: list[str] = []
    algorithms: list[str] = []
    loss_nonincreasing: list[bool] = []
    residual_decreased: list[bool] = []
    curve_lengths: list[int] = []
    probe_histories: list[NDArray[np.complex128]] = []
    reference_shape: tuple[int, ...] | None = None
    prediction_shape: tuple[int, ...] | None = None
    for name in branch_names:
        result = reconstructions[name]
        if result.get("truth_used_by_optimizer") is not False:
            raise ValueError("Truth is forbidden from initialization branches.")
        initial = np.asarray(result["P_B_init"], dtype=np.complex128)
        final = np.asarray(result["P_B_rec"], dtype=np.complex128)
        prediction = np.asarray(result["prediction_final"], dtype=np.float64)
        loss_curve = np.asarray(result["loss_curve"], dtype=np.float64)
        residual_curve = np.asarray(
            result["detector_relative_residual_curve"], dtype=np.float64
        )
        probe_history = np.asarray(result["probe_history"], dtype=np.complex128)
        if reference_shape is None:
            reference_shape = initial.shape
            prediction_shape = prediction.shape
        if (
            initial.shape != reference_shape
            or final.shape != reference_shape
            or prediction.shape != prediction_shape
            or loss_curve.ndim != 1
            or residual_curve.shape != loss_curve.shape
            or probe_history.shape != (len(loss_curve), *initial.shape)
            or not all(
                np.all(np.isfinite(values))
                for values in (
                    initial,
                    final,
                    prediction,
                    loss_curve,
                    residual_curve,
                    probe_history,
                )
            )
            or not np.array_equal(probe_history[0], initial)
            or not np.array_equal(probe_history[-1], final)
        ):
            raise ValueError("Initialization branch arrays are inconsistent.")
        initial_probes.append(initial)
        final_probes.append(final)
        predictions.append(prediction)
        iterations.append(int(result["iterations_completed"]))
        stopping_reasons.append(str(result["stopping_reason"]))
        algorithms.append(str(result["algorithm"]))
        loss_nonincreasing.append(bool(np.all(np.diff(loss_curve) <= 0.0)))
        residual_decreased.append(bool(residual_curve[-1] < residual_curve[0]))
        curve_lengths.append(len(loss_curve))
        probe_histories.append(probe_history)

    def symmetric_relative_l2(left: np.ndarray, right: np.ndarray) -> float:
        denominator = 0.5 * (_l2_norm(left) + _l2_norm(right))
        return float(
            _l2_norm(left - right)
            / max(denominator, np.finfo(np.float64).eps)
        )

    def pairwise_matrices(
        probes: list[NDArray[np.complex128]],
        predicted: list[NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        raw = np.zeros((count, count), dtype=np.float64)
        aligned = np.zeros((count, count), dtype=np.float64)
        prediction_matrix = np.zeros((count, count), dtype=np.float64)
        for row in range(count):
            for column in range(row + 1, count):
                column_aligned, _ = _align_global_phase(
                    probes[column], probes[row]
                )
                values = (
                    symmetric_relative_l2(probes[row], probes[column]),
                    symmetric_relative_l2(probes[row], column_aligned),
                    symmetric_relative_l2(predicted[row], predicted[column]),
                )
                for matrix, value in zip(
                    (raw, aligned, prediction_matrix), values, strict=True
                ):
                    matrix[row, column] = value
                    matrix[column, row] = value
        return raw, aligned, prediction_matrix

    count = len(branch_names)
    initial_predictions = [
        np.zeros_like(prediction, dtype=np.float64) for prediction in predictions
    ]
    initial_raw, initial_aligned, _ = pairwise_matrices(
        initial_probes, initial_predictions
    )
    final_raw, final_aligned, prediction_relative = pairwise_matrices(
        final_probes, predictions
    )
    off_diagonal = np.triu_indices(count, k=1)
    reference_index = branch_names.index(reference_branch)
    reference_initial = initial_probes[reference_index]
    reference_norm = _l2_norm(reference_initial)
    perturbation_magnitudes = np.asarray(
        [
            _l2_norm(initial - reference_initial)
            / max(reference_norm, np.finfo(np.float64).eps)
            for initial in initial_probes
        ],
        dtype=np.float64,
    )
    nonzero_indices = [
        index
        for index, magnitude in enumerate(perturbation_magnitudes)
        if magnitude > np.finfo(np.float64).eps
    ]
    unit_directions = [
        (initial_probes[index] - reference_initial)
        / _l2_norm(initial_probes[index] - reference_initial)
        for index in nonzero_indices
    ]
    direction_error_matrix = np.zeros(
        (len(unit_directions), len(unit_directions)), dtype=np.float64
    )
    direction_correlation_matrix = np.eye(
        len(unit_directions), dtype=np.float64
    )
    for row in range(len(unit_directions)):
        for column in range(row + 1, len(unit_directions)):
            error = _l2_norm(unit_directions[row] - unit_directions[column])
            correlation = _real_inner_product(
                unit_directions[row], unit_directions[column]
            )
            direction_error_matrix[row, column] = error
            direction_error_matrix[column, row] = error
            direction_correlation_matrix[row, column] = correlation
            direction_correlation_matrix[column, row] = correlation
    diagnostic: dict[str, Any] = {
        "branch_names": branch_names,
        "reference_branch": reference_branch,
        "initial_perturbation_relative_l2_to_reference": (
            perturbation_magnitudes
        ),
        "nonzero_perturbation_branch_names": [
            branch_names[index] for index in nonzero_indices
        ],
        "initial_unit_direction_pairwise_l2_error": direction_error_matrix,
        "initial_unit_direction_real_correlation": direction_correlation_matrix,
        "maximum_initial_unit_direction_pairwise_l2_error": float(
            np.max(direction_error_matrix, initial=0.0)
        ),
        "initial_pairwise_raw_probe_relative_l2": initial_raw,
        "initial_pairwise_global_phase_aligned_probe_relative_l2": (
            initial_aligned
        ),
        "final_pairwise_raw_probe_relative_l2": final_raw,
        "final_pairwise_global_phase_aligned_probe_relative_l2": final_aligned,
        "final_pairwise_prediction_relative_l2": prediction_relative,
        "initial_raw_probe_relative_l2_to_reference": initial_raw[
            reference_index
        ],
        "final_raw_probe_relative_l2_to_reference": final_raw[reference_index],
        "final_global_phase_aligned_probe_relative_l2_to_reference": (
            final_aligned[reference_index]
        ),
        "final_prediction_relative_l2_to_reference": prediction_relative[
            reference_index
        ],
        "maximum_final_pairwise_raw_probe_relative_l2": float(
            np.max(final_raw[off_diagonal])
        ),
        "maximum_final_pairwise_global_phase_aligned_probe_relative_l2": float(
            np.max(final_aligned[off_diagonal])
        ),
        "maximum_final_pairwise_prediction_relative_l2": float(
            np.max(prediction_relative[off_diagonal])
        ),
        "iterations_completed": np.asarray(iterations, dtype=np.int64),
        "stopping_reasons": stopping_reasons,
        "equal_iterations_completed": len(set(iterations)) == 1,
        "all_registered_budgets_completed": all(
            reason == "iteration_budget" for reason in stopping_reasons
        ),
        "equal_optimizer_algorithm": len(set(algorithms)) == 1,
        "equal_curve_lengths": len(set(curve_lengths)) == 1,
        "all_loss_nonincreasing": all(loss_nonincreasing),
        "all_detector_residuals_decreased": all(residual_decreased),
        "truth_used_by_diagnostic": False,
        "diagnostic_role": "truth_free_equal_budget_initialization_ablation",
        "pairwise_alignment_role": "diagnostic_convention_not_optimizer_gauge",
    }
    if (operator is None) != (checkpoint_interval is None):
        raise ValueError("operator and checkpoint_interval must be provided together.")
    if operator is not None and checkpoint_interval is not None:
        interval = int(checkpoint_interval)
        completed_iterations = curve_lengths[0] - 1
        if (
            interval <= 0
            or interval != float(checkpoint_interval)
            or len(set(curve_lengths)) != 1
            or completed_iterations % interval
        ):
            raise ValueError("Checkpoint interval must divide the common history.")
        checkpoint_iterations = np.arange(
            0, completed_iterations + 1, interval, dtype=np.int64
        )
        raw_curve = np.empty(
            (len(checkpoint_iterations), count, count), dtype=np.float64
        )
        aligned_curve = np.empty_like(raw_curve)
        prediction_curve = np.empty_like(raw_curve)
        for checkpoint_index, iteration in enumerate(checkpoint_iterations):
            fields = [history[iteration] for history in probe_histories]
            checkpoint_predictions = [
                operator.predict_stack(field) for field in fields
            ]
            raw, aligned, prediction_matrix = pairwise_matrices(
                fields, checkpoint_predictions
            )
            raw_curve[checkpoint_index] = raw
            aligned_curve[checkpoint_index] = aligned
            prediction_curve[checkpoint_index] = prediction_matrix
        diagnostic.update(
            {
                "checkpoint_iterations": checkpoint_iterations,
                "checkpoint_pairwise_raw_probe_relative_l2_curve": raw_curve,
                "checkpoint_pairwise_global_phase_aligned_probe_relative_l2_curve": (
                    aligned_curve
                ),
                "checkpoint_pairwise_prediction_relative_l2_curve": (
                    prediction_curve
                ),
                "checkpoint_raw_probe_relative_l2_to_reference_curve": (
                    raw_curve[:, reference_index, :]
                ),
                (
                    "checkpoint_global_phase_aligned_probe_"
                    "relative_l2_to_reference_curve"
                ): (
                    aligned_curve[:, reference_index, :]
                ),
                "checkpoint_prediction_relative_l2_to_reference_curve": (
                    prediction_curve[:, reference_index, :]
                ),
                "checkpoint_interval": interval,
                "checkpoint_truth_used_by_diagnostic": False,
            }
        )
    return diagnostic


def truth_free_detector_quadrature_ablation_diagnostic(
    reconstructions: Mapping[str, Mapping[str, Any]],
    operators: Mapping[str, MatchedKnownBProbeOperator],
    *,
    reference_branch: str,
    checkpoint_interval: int,
) -> dict[str, Any]:
    """Compare detector data-model pairing branches without truth."""

    branch_names = list(reconstructions)
    if (
        len(branch_names) < 2
        or reference_branch not in reconstructions
        or set(operators) != set(branch_names)
    ):
        raise ValueError("At least two registered detector branches are required.")
    initial_probes: list[NDArray[np.complex128]] = []
    final_probes: list[NDArray[np.complex128]] = []
    final_predictions: list[NDArray[np.float64]] = []
    histories: list[NDArray[np.complex128]] = []
    curve_lengths: list[int] = []
    iterations: list[int] = []
    stopping_reasons: list[str] = []
    algorithms: list[str] = []
    loss_nonincreasing: list[bool] = []
    residual_decreased: list[bool] = []
    for name in branch_names:
        result = reconstructions[name]
        if result.get("truth_used_by_optimizer") is not False:
            raise ValueError("Truth is forbidden from detector branches.")
        initial = np.asarray(result["P_B_init"], dtype=np.complex128)
        final = np.asarray(result["P_B_rec"], dtype=np.complex128)
        prediction = np.asarray(result["prediction_final"], dtype=np.float64)
        history = np.asarray(result["probe_history"], dtype=np.complex128)
        loss_curve = np.asarray(result["loss_curve"], dtype=np.float64)
        residual_curve = np.asarray(
            result["detector_relative_residual_curve"], dtype=np.float64
        )
        expected_prediction_shape = (
            len(operators[name].positions_m),
            *operators[name].detector_roi_shape,
        )
        if (
            final.shape != initial.shape
            or prediction.shape != expected_prediction_shape
            or history.shape != (len(loss_curve), *initial.shape)
            or residual_curve.shape != loss_curve.shape
            or not np.array_equal(history[0], initial)
            or not np.array_equal(history[-1], final)
            or not all(
                np.all(np.isfinite(values))
                for values in (
                    initial,
                    final,
                    prediction,
                    history,
                    loss_curve,
                    residual_curve,
                )
            )
        ):
            raise ValueError("Detector-ablation branch arrays are inconsistent.")
        initial_probes.append(initial)
        final_probes.append(final)
        final_predictions.append(prediction)
        histories.append(history)
        curve_lengths.append(len(loss_curve))
        iterations.append(int(result["iterations_completed"]))
        stopping_reasons.append(str(result["stopping_reason"]))
        algorithms.append(str(result["algorithm"]))
        loss_nonincreasing.append(bool(np.all(np.diff(loss_curve) <= 0.0)))
        residual_decreased.append(bool(residual_curve[-1] < residual_curve[0]))
    reference_index = branch_names.index(reference_branch)
    reference_initial = initial_probes[reference_index]
    initial_exact = all(
        np.array_equal(initial, reference_initial) for initial in initial_probes
    )
    if not initial_exact:
        raise ValueError("Detector branches must use one exact initialization.")

    def symmetric_relative_l2(left: np.ndarray, right: np.ndarray) -> float:
        denominator = 0.5 * (_l2_norm(left) + _l2_norm(right))
        return float(
            _l2_norm(left - right)
            / max(denominator, np.finfo(np.float64).eps)
        )

    def pairwise_matrices(
        probes: list[NDArray[np.complex128]],
        predictions: list[NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        count = len(probes)
        raw = np.zeros((count, count), dtype=np.float64)
        aligned = np.zeros_like(raw)
        prediction = np.zeros_like(raw)
        for row in range(count):
            for column in range(row + 1, count):
                column_aligned, _ = _align_global_phase(
                    probes[column], probes[row]
                )
                values = (
                    symmetric_relative_l2(probes[row], probes[column]),
                    symmetric_relative_l2(probes[row], column_aligned),
                    symmetric_relative_l2(
                        predictions[row], predictions[column]
                    ),
                )
                for matrix, value in zip(
                    (raw, aligned, prediction), values, strict=True
                ):
                    matrix[row, column] = value
                    matrix[column, row] = value
        return raw, aligned, prediction

    initial_predictions = [
        operators[name].predict_stack(initial_probes[index])
        for index, name in enumerate(branch_names)
    ]
    initial_raw, initial_aligned, initial_prediction = pairwise_matrices(
        initial_probes, initial_predictions
    )
    final_raw, final_aligned, final_prediction = pairwise_matrices(
        final_probes, final_predictions
    )
    interval = int(checkpoint_interval)
    if (
        interval <= 0
        or interval != float(checkpoint_interval)
        or len(set(curve_lengths)) != 1
        or (curve_lengths[0] - 1) % interval
    ):
        raise ValueError("Checkpoint interval must divide the common history.")
    checkpoint_iterations = np.arange(
        0, curve_lengths[0], interval, dtype=np.int64
    )
    count = len(branch_names)
    raw_curve = np.empty(
        (len(checkpoint_iterations), count, count), dtype=np.float64
    )
    aligned_curve = np.empty_like(raw_curve)
    prediction_curve = np.empty_like(raw_curve)
    for checkpoint_index, iteration in enumerate(checkpoint_iterations):
        fields = [history[iteration] for history in histories]
        predictions = [
            operators[name].predict_stack(fields[index])
            for index, name in enumerate(branch_names)
        ]
        raw, aligned, prediction = pairwise_matrices(fields, predictions)
        raw_curve[checkpoint_index] = raw
        aligned_curve[checkpoint_index] = aligned
        prediction_curve[checkpoint_index] = prediction
    return {
        "branch_names": branch_names,
        "reference_branch": reference_branch,
        "initial_probe_exact_equal": initial_exact,
        "initial_pairwise_raw_probe_relative_l2": initial_raw,
        "initial_pairwise_global_phase_aligned_probe_relative_l2": (
            initial_aligned
        ),
        "initial_pairwise_prediction_relative_l2": initial_prediction,
        "final_pairwise_raw_probe_relative_l2": final_raw,
        "final_pairwise_global_phase_aligned_probe_relative_l2": final_aligned,
        "final_pairwise_prediction_relative_l2": final_prediction,
        "checkpoint_iterations": checkpoint_iterations,
        "checkpoint_pairwise_raw_probe_relative_l2_curve": raw_curve,
        "checkpoint_pairwise_global_phase_aligned_probe_relative_l2_curve": (
            aligned_curve
        ),
        "checkpoint_pairwise_prediction_relative_l2_curve": prediction_curve,
        "checkpoint_raw_probe_relative_l2_to_reference_curve": raw_curve[
            :, reference_index, :
        ],
        "checkpoint_global_phase_aligned_probe_relative_l2_to_reference_curve": (
            aligned_curve[:, reference_index, :]
        ),
        "checkpoint_prediction_relative_l2_to_reference_curve": (
            prediction_curve[:, reference_index, :]
        ),
        "final_raw_probe_relative_l2_to_reference": final_raw[reference_index],
        "final_global_phase_aligned_probe_relative_l2_to_reference": (
            final_aligned[reference_index]
        ),
        "final_prediction_relative_l2_to_reference": final_prediction[
            reference_index
        ],
        "iterations_completed": np.asarray(iterations, dtype=np.int64),
        "stopping_reasons": stopping_reasons,
        "equal_iterations_completed": len(set(iterations)) == 1,
        "all_registered_budgets_completed": all(
            reason == "iteration_budget" for reason in stopping_reasons
        ),
        "equal_optimizer_algorithm": len(set(algorithms)) == 1,
        "equal_curve_lengths": len(set(curve_lengths)) == 1,
        "all_loss_nonincreasing": all(loss_nonincreasing),
        "all_detector_residuals_decreased": all(residual_decreased),
        "branch_detector_readouts": [
            operators[name].detector_readout for name in branch_names
        ],
        "checkpoint_interval": interval,
        "truth_used_by_diagnostic": False,
        "diagnostic_role": (
            "truth_free_equal_budget_detector_data_model_pairing_control"
        ),
        "pairwise_alignment_role": "diagnostic_convention_not_optimizer_gauge",
    }


def normalized_intensity_jacobian_sensitivity(
    operator: MatchedKnownBProbeOperator,
    probe: ComplexArray,
    direction: ComplexArray,
) -> dict[str, Any]:
    """Return truth-free local intensity sensitivity for a unit-L2 direction."""

    probe_values = np.asarray(probe, dtype=np.complex128)
    direction_values = np.asarray(direction, dtype=np.complex128)
    if (
        probe_values.shape != operator.native_shape
        or direction_values.shape != operator.native_shape
        or not np.all(np.isfinite(probe_values))
        or not np.all(np.isfinite(direction_values))
    ):
        raise ValueError("probe and direction must be finite native-grid fields.")
    direction_norm = _l2_norm(direction_values)
    if direction_norm <= np.finfo(np.float64).eps:
        raise ValueError("direction must have a nonzero L2 norm.")
    unit_direction = np.asarray(
        direction_values / direction_norm, dtype=np.complex128
    )
    jacobian_direction = operator.intensity_jacobian_direction(
        probe_values, unit_direction
    )
    directional_curvature = float(
        np.mean(np.square(jacobian_direction), dtype=np.float64)
    )
    jacobian_rms_gain = float(np.sqrt(max(directional_curvature, 0.0)))
    return {
        "direction_unit_l2": unit_direction,
        "direction_l2_norm_before_normalization": direction_norm,
        "jacobian_rms_gain": jacobian_rms_gain,
        "directional_gauss_newton_curvature": directional_curvature,
    }


def truth_free_pairwise_conditioning_diagnostic(
    operator: MatchedKnownBProbeOperator,
    measured: FloatArray,
    primary_reconstruction: Mapping[str, Any],
    control_reconstruction: Mapping[str, Any],
    *,
    random_seed: int,
    random_direction_count: int,
) -> dict[str, Any]:
    """Compare pairwise, gradient, native-phase, and random directions."""

    seed_value = int(random_seed)
    count = int(random_direction_count)
    if seed_value < 0 or seed_value != float(random_seed):
        raise ValueError("random_seed must be a nonnegative integer.")
    if count < 1 or count > 64:
        raise ValueError("random_direction_count must lie in [1, 64].")
    primary = np.asarray(
        primary_reconstruction["P_B_rec"], dtype=np.complex128
    )
    control = np.asarray(
        control_reconstruction["P_B_rec"], dtype=np.complex128
    )
    control_aligned, alignment_factor = _align_global_phase(control, primary)
    pairwise = normalized_intensity_jacobian_sensitivity(
        operator, primary, control_aligned - primary
    )
    _, gradient, _ = operator.loss_and_gradient(primary, measured)
    gradient_sensitivity = normalized_intensity_jacobian_sensitivity(
        operator, primary, gradient
    )
    native_phase_sensitivity = normalized_intensity_jacobian_sensitivity(
        operator, primary, 1j * primary
    )
    rng = np.random.default_rng(seed_value)
    random_gains: list[float] = []
    random_curvatures: list[float] = []
    for _ in range(count):
        direction = rng.normal(size=operator.native_shape) + 1j * rng.normal(
            size=operator.native_shape
        )
        direction -= np.mean(direction, dtype=np.complex128)
        sensitivity = normalized_intensity_jacobian_sensitivity(
            operator, primary, direction
        )
        random_gains.append(float(sensitivity["jacobian_rms_gain"]))
        random_curvatures.append(
            float(sensitivity["directional_gauss_newton_curvature"])
        )
    random_gain_values = np.asarray(random_gains, dtype=np.float64)
    random_curvature_values = np.asarray(
        random_curvatures, dtype=np.float64
    )
    pairwise_gain = float(pairwise["jacobian_rms_gain"])
    random_median = float(np.median(random_gain_values))
    pairwise_unit = np.asarray(
        pairwise["direction_unit_l2"], dtype=np.complex128
    )
    gradient_unit = np.asarray(
        gradient_sensitivity["direction_unit_l2"], dtype=np.complex128
    )
    native_phase_unit = np.asarray(
        native_phase_sensitivity["direction_unit_l2"], dtype=np.complex128
    )
    return {
        "P_B_pairwise_direction_unit_l2": pairwise_unit,
        "pairwise_alignment_factor": alignment_factor,
        "pairwise_direction_l2_norm_before_normalization": pairwise[
            "direction_l2_norm_before_normalization"
        ],
        "pairwise_jacobian_rms_gain": pairwise_gain,
        "pairwise_directional_gauss_newton_curvature": pairwise[
            "directional_gauss_newton_curvature"
        ],
        "primary_gradient_l2_norm": gradient_sensitivity[
            "direction_l2_norm_before_normalization"
        ],
        "primary_gradient_jacobian_rms_gain": gradient_sensitivity[
            "jacobian_rms_gain"
        ],
        "primary_gradient_directional_gauss_newton_curvature": (
            gradient_sensitivity["directional_gauss_newton_curvature"]
        ),
        "native_global_phase_rotation_jacobian_rms_gain": (
            native_phase_sensitivity[
                "jacobian_rms_gain"
            ]
        ),
        "native_global_phase_rotation_directional_gauss_newton_curvature": (
            native_phase_sensitivity["directional_gauss_newton_curvature"]
        ),
        "pairwise_to_gradient_real_inner_product_abs": abs(
            float(np.real(_complex_inner_product(pairwise_unit, gradient_unit)))
        ),
        "pairwise_to_native_global_phase_real_inner_product_abs": abs(
            float(
                np.real(
                    _complex_inner_product(pairwise_unit, native_phase_unit)
                )
            )
        ),
        "random_direction_jacobian_rms_gains": random_gain_values,
        "random_directional_gauss_newton_curvatures": (
            random_curvature_values
        ),
        "random_jacobian_rms_gain_minimum": float(np.min(random_gain_values)),
        "random_jacobian_rms_gain_median": random_median,
        "random_jacobian_rms_gain_maximum": float(np.max(random_gain_values)),
        "pairwise_to_random_median_gain_ratio": pairwise_gain / random_median,
        "random_gain_fraction_at_or_below_pairwise": float(
            np.mean(random_gain_values <= pairwise_gain)
        ),
        "random_seed": seed_value,
        "random_direction_count": count,
        "random_direction_distribution": "zero_mean_complex_gaussian",
        "direction_normalization": "unit_probe_l2",
        "jacobian_output_norm": "rms_over_detector_stack",
        "truth_used_by_diagnostic": False,
        "diagnostic_role": "truth_free_local_jacobian_conditioning",
    }


def _weighted_ritz_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    """Return a discrete weighted quantile for sorted Ritz values."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    cumulative = np.cumsum(weights, dtype=np.float64)
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(values[min(index, len(values) - 1)])


def _matrix_free_lanczos(
    matvec: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    start: ComplexArray,
    *,
    krylov_dimension: int,
    reorthogonalization_passes: int,
    breakdown_relative_tolerance: float,
) -> tuple[
    dict[str, Any],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Run deterministic fully reorthogonalized real-Hilbert Lanczos."""

    requested = int(krylov_dimension)
    passes = int(reorthogonalization_passes)
    tolerance = float(breakdown_relative_tolerance)
    start_values = np.asarray(start, dtype=np.complex128)
    if start_values.ndim != 2 or not np.all(np.isfinite(start_values)):
        raise ValueError("Lanczos start must be a finite 2D complex field.")
    if requested < 1 or passes < 1:
        raise ValueError("Lanczos dimension and reorthogonalization must be positive.")
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("breakdown_relative_tolerance must lie in (0, 1).")
    start_norm = _l2_norm(start_values)
    if start_norm <= np.finfo(np.float64).eps:
        raise ValueError("Lanczos start must have a nonzero L2 norm.")

    basis: list[NDArray[np.complex128]] = []
    alphas: list[float] = []
    betas: list[float] = []
    q = np.asarray(start_values / start_norm, dtype=np.complex128)
    q_previous = np.zeros_like(q)
    beta_previous = 0.0
    terminal_beta = 0.0
    breakdown = False
    for index in range(requested):
        basis.append(q.copy())
        action = np.asarray(matvec(q), dtype=np.complex128)
        if action.shape != q.shape or not np.all(np.isfinite(action)):
            raise ValueError("Lanczos matvec returned an invalid field.")
        alpha = _real_inner_product(q, action)
        if not np.isfinite(alpha):
            raise ValueError("Lanczos alpha is non-finite.")
        alphas.append(alpha)
        residual = action - alpha * q
        if index:
            residual -= beta_previous * q_previous
        for _ in range(passes):
            for vector in basis:
                residual -= _real_inner_product(vector, residual) * vector
        terminal_beta = _l2_norm(residual)
        scale = max(
            _l2_norm(action),
            abs(alpha),
            abs(beta_previous),
            np.finfo(np.float64).eps,
        )
        if index + 1 == requested:
            break
        if terminal_beta <= tolerance * scale:
            breakdown = True
            break
        betas.append(terminal_beta)
        q_previous = q
        q = np.asarray(residual / terminal_beta, dtype=np.complex128)
        beta_previous = terminal_beta

    basis_array = np.stack(basis)
    completed = len(alphas)
    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if completed > 1:
        off_diagonal = np.asarray(betas[: completed - 1], dtype=np.float64)
        tridiagonal += np.diag(off_diagonal, 1)
        tridiagonal += np.diag(off_diagonal, -1)
    ritz_values, eigenvectors = np.linalg.eigh(tridiagonal)
    weight_sum_before = float(
        np.sum(np.square(eigenvectors[0]), dtype=np.float64)
    )
    spectral_weights = np.asarray(
        np.square(eigenvectors[0]) / weight_sum_before,
        dtype=np.float64,
    )
    ritz_residuals = np.asarray(
        terminal_beta * np.abs(eigenvectors[-1]), dtype=np.float64
    )
    spectral_radius = max(
        float(np.max(np.abs(ritz_values))), np.finfo(np.float64).eps
    )
    ritz_fields = np.asarray(
        np.tensordot(eigenvectors.T, basis_array, axes=(1, 0)),
        dtype=np.complex128,
    )
    orthogonality = np.empty((completed, completed), dtype=np.float64)
    for row in range(completed):
        for column in range(completed):
            orthogonality[row, column] = _real_inner_product(
                basis_array[row], basis_array[column]
            )
    orthogonality_error = float(
        np.max(np.abs(orthogonality - np.eye(completed)))
    )
    start_rayleigh = float(alphas[0])
    spectral_rayleigh = float(
        np.sum(spectral_weights * ritz_values, dtype=np.float64)
    )
    result = {
        "requested_krylov_dimension": requested,
        "completed_krylov_dimension": completed,
        "matvec_count": completed,
        "breakdown_detected": breakdown,
        "terminal_beta": terminal_beta,
        "breakdown_relative_tolerance": tolerance,
        "reorthogonalization_passes": passes,
        "basis_real_orthogonality_max_abs_error": orthogonality_error,
        "tridiagonal_diagonal": np.asarray(alphas, dtype=np.float64),
        "tridiagonal_off_diagonal": np.asarray(
            betas[: completed - 1], dtype=np.float64
        ),
        "ritz_values": np.asarray(ritz_values, dtype=np.float64),
        "ritz_singular_gain_estimates": np.sqrt(
            np.maximum(ritz_values, 0.0)
        ),
        "ritz_residual_l2_estimates": ritz_residuals,
        "ritz_residual_relative_to_spectral_radius": (
            ritz_residuals / spectral_radius
        ),
        "spectral_weights": spectral_weights,
        "spectral_weight_sum_before_normalization": weight_sum_before,
        "start_rayleigh_quotient": start_rayleigh,
        "spectral_reconstructed_rayleigh_quotient": spectral_rayleigh,
        "rayleigh_reconstruction_relative_error": abs(
            start_rayleigh - spectral_rayleigh
        )
        / max(
            abs(start_rayleigh),
            abs(spectral_rayleigh),
            np.finfo(np.float64).eps,
        ),
        "minimum_ritz_value": float(ritz_values[0]),
        "maximum_ritz_value": float(ritz_values[-1]),
        "negative_ritz_value_count": int(np.count_nonzero(ritz_values < 0.0)),
        "maximum_ritz_residual_l2_estimate": float(np.max(ritz_residuals)),
        "spectral_weighted_ritz_quantile_10": _weighted_ritz_quantile(
            ritz_values, spectral_weights, 0.10
        ),
        "spectral_weighted_ritz_quantile_25": _weighted_ritz_quantile(
            ritz_values, spectral_weights, 0.25
        ),
        "spectral_weighted_median_ritz_value": _weighted_ritz_quantile(
            ritz_values, spectral_weights, 0.50
        ),
        "spectral_weighted_ritz_quantile_75": _weighted_ritz_quantile(
            ritz_values, spectral_weights, 0.75
        ),
        "spectral_weighted_ritz_quantile_90": _weighted_ritz_quantile(
            ritz_values, spectral_weights, 0.90
        ),
    }
    return result, basis_array, ritz_fields


def matrix_free_lanczos_spectral_measure(
    matvec: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    start: ComplexArray,
    *,
    krylov_dimension: int,
    reorthogonalization_passes: int,
    breakdown_relative_tolerance: float,
) -> dict[str, Any]:
    """Return a bounded Ritz spectral measure without materializing a matrix."""

    result, _, _ = _matrix_free_lanczos(
        matvec,
        start,
        krylov_dimension=krylov_dimension,
        reorthogonalization_passes=reorthogonalization_passes,
        breakdown_relative_tolerance=breakdown_relative_tolerance,
    )
    return result


def truth_free_local_spectral_diagnostic(
    operator: MatchedKnownBProbeOperator,
    primary_probe: ComplexArray,
    control_probe: ComplexArray,
    *,
    krylov_dimension: int,
    reorthogonalization_passes: int,
    breakdown_relative_tolerance: float,
    random_seed: int,
    reported_low_ritz_count: int,
) -> dict[str, Any]:
    """Probe local ``J^T J`` spectral measures from pairwise/random starts."""

    primary = np.asarray(primary_probe, dtype=np.complex128)
    control = np.asarray(control_probe, dtype=np.complex128)
    if (
        primary.shape != operator.native_shape
        or control.shape != operator.native_shape
        or not np.all(np.isfinite(primary))
        or not np.all(np.isfinite(control))
    ):
        raise ValueError(
            "primary_probe and control_probe must be finite native fields."
        )
    control_aligned, alignment_factor = _align_global_phase(control, primary)
    pairwise_direction = control_aligned - primary
    pairwise_norm = _l2_norm(pairwise_direction)
    if pairwise_norm <= np.finfo(np.float64).eps:
        raise ValueError("The aligned pairwise spectral direction is zero.")
    pairwise_unit = np.asarray(
        pairwise_direction / pairwise_norm, dtype=np.complex128
    )
    seed_value = int(random_seed)
    if seed_value < 0 or seed_value != float(random_seed):
        raise ValueError("random_seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed_value)
    random_direction = rng.normal(size=operator.native_shape) + 1j * rng.normal(
        size=operator.native_shape
    )
    random_direction -= np.mean(random_direction, dtype=np.complex128)
    random_unit = np.asarray(
        random_direction / _l2_norm(random_direction), dtype=np.complex128
    )
    detector_count = int(
        len(operator.positions_m) * np.prod(operator.detector_roi_shape)
    )

    def normal_matvec(
        direction: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        jacobian_direction = operator.intensity_jacobian_direction(
            primary, direction
        )
        return np.asarray(
            operator.intensity_jacobian_adjoint(
                primary, jacobian_direction
            )
            / float(detector_count),
            dtype=np.complex128,
        )

    pairwise_result, _, _ = _matrix_free_lanczos(
        normal_matvec,
        pairwise_unit,
        krylov_dimension=krylov_dimension,
        reorthogonalization_passes=reorthogonalization_passes,
        breakdown_relative_tolerance=breakdown_relative_tolerance,
    )
    random_result, _, random_ritz_fields = _matrix_free_lanczos(
        normal_matvec,
        random_unit,
        krylov_dimension=krylov_dimension,
        reorthogonalization_passes=reorthogonalization_passes,
        breakdown_relative_tolerance=breakdown_relative_tolerance,
    )
    low_count = int(reported_low_ritz_count)
    if low_count < 1 or low_count > min(
        int(pairwise_result["completed_krylov_dimension"]),
        int(random_result["completed_krylov_dimension"]),
    ):
        raise ValueError("reported_low_ritz_count exceeds completed dimensions.")
    pairwise_to_random_ritz_overlaps = np.asarray(
        [
            _real_inner_product(pairwise_unit, field)
            for field in random_ritz_fields
        ],
        dtype=np.float64,
    )
    pairwise_to_random_ritz_weights = np.square(
        pairwise_to_random_ritz_overlaps
    )
    random_median = float(
        random_result["spectral_weighted_median_ritz_value"]
    )
    pairwise_ritz_values = np.asarray(
        pairwise_result["ritz_values"], dtype=np.float64
    )
    pairwise_weights = np.asarray(
        pairwise_result["spectral_weights"], dtype=np.float64
    )
    random_ritz_values = np.asarray(
        random_result["ritz_values"], dtype=np.float64
    )
    random_weights = np.asarray(
        random_result["spectral_weights"], dtype=np.float64
    )
    random_projection_weight = float(
        np.sum(pairwise_to_random_ritz_weights, dtype=np.float64)
    )
    random_low_projection_weight = float(
        np.sum(pairwise_to_random_ritz_weights[:low_count], dtype=np.float64)
    )
    pairwise_rayleigh = float(pairwise_result["start_rayleigh_quotient"])
    random_rayleigh = float(random_result["start_rayleigh_quotient"])
    return {
        "P_B_primary_probe_raw": primary,
        "P_B_control_probe_raw": control,
        "P_B_control_probe_global_phase_aligned_to_primary": control_aligned,
        "P_B_pairwise_direction_unit_l2": pairwise_unit,
        "P_B_random_start_direction_unit_l2": random_unit,
        "pairwise_alignment_factor": alignment_factor,
        "pairwise_direction_l2_norm_before_normalization": pairwise_norm,
        "matrix_operator": "real_intensity_jacobian_transpose_times_jacobian",
        "matrix_normalization": "mean_over_detector_stack",
        "detector_value_count": detector_count,
        "pairwise_start": pairwise_result,
        "random_start": random_result,
        "reported_low_ritz_count": low_count,
        "pairwise_lowest_reported_ritz_spectral_weight": float(
            np.sum(pairwise_weights[:low_count], dtype=np.float64)
        ),
        "random_lowest_reported_ritz_spectral_weight": float(
            np.sum(random_weights[:low_count], dtype=np.float64)
        ),
        "pairwise_spectral_weight_at_or_below_random_weighted_median": float(
            np.sum(pairwise_weights[pairwise_ritz_values <= random_median])
        ),
        "random_spectral_weight_at_or_below_own_weighted_median": float(
            np.sum(random_weights[random_ritz_values <= random_median])
        ),
        "pairwise_to_random_ritz_vector_real_overlaps": (
            pairwise_to_random_ritz_overlaps
        ),
        "pairwise_to_random_ritz_vector_squared_overlaps": (
            pairwise_to_random_ritz_weights
        ),
        "pairwise_projection_weight_onto_random_krylov_subspace": (
            random_projection_weight
        ),
        "pairwise_projection_weight_onto_random_low_ritz_subspace": (
            random_low_projection_weight
        ),
        "pairwise_random_low_projection_fraction_of_captured_weight": (
            random_low_projection_weight
            / max(random_projection_weight, np.finfo(np.float64).eps)
        ),
        "pairwise_to_random_start_rayleigh_quotient_ratio": (
            pairwise_rayleigh
            / max(random_rayleigh, np.finfo(np.float64).eps)
        ),
        "pairwise_to_random_weighted_median_ritz_value_ratio": (
            float(pairwise_result["spectral_weighted_median_ritz_value"])
            / max(random_median, np.finfo(np.float64).eps)
        ),
        "pairwise_direct_jacobian_rms_gain": float(
            np.sqrt(max(pairwise_rayleigh, 0.0))
        ),
        "random_direct_jacobian_rms_gain": float(
            np.sqrt(max(random_rayleigh, 0.0))
        ),
        "random_seed": seed_value,
        "start_vectors": [
            "pairwise_direction",
            "deterministic_zero_mean_complex_gaussian",
        ],
        "truth_used_by_diagnostic": False,
        "scientific_thresholds_preregistered": False,
        "diagnostic_role": "truth_free_matrix_free_local_jtj_lanczos",
        "interpretation_boundary": (
            "bounded start-dependent Ritz measures; not a complete Jacobian "
            "spectrum, rank, resolution, or detection-limit estimate"
        ),
    }


def lanczos_dimension_convergence_diagnostic(
    reference_diagnostic: Mapping[str, Any],
    extended_diagnostic: Mapping[str, Any],
    *,
    reference_krylov_dimension: int,
    reported_low_ritz_count: int,
) -> dict[str, Any]:
    """Compare one registered Lanczos dimension extension without thresholds."""

    reference_dimension = int(reference_krylov_dimension)
    low_count = int(reported_low_ritz_count)
    if reference_dimension < 2 or low_count < 1 or low_count > reference_dimension:
        raise ValueError("Invalid reference dimension or low Ritz count.")
    for key in (
        "matrix_operator",
        "matrix_normalization",
        "random_seed",
        "start_vectors",
    ):
        if reference_diagnostic.get(key) != extended_diagnostic.get(key):
            raise ValueError(f"Spectral diagnostics differ at {key}.")

    def compare_start(name: str) -> dict[str, Any]:
        reference = _section(reference_diagnostic, name)
        extended = _section(extended_diagnostic, name)
        reference_completed = int(reference["completed_krylov_dimension"])
        extended_completed = int(extended["completed_krylov_dimension"])
        if reference_completed != reference_dimension:
            raise ValueError(f"{name} reference dimension is invalid.")
        if extended_completed <= reference_dimension:
            raise ValueError(f"{name} did not complete the registered extension.")
        reference_diagonal = np.asarray(
            reference["tridiagonal_diagonal"], dtype=np.float64
        )
        extended_diagonal = np.asarray(
            extended["tridiagonal_diagonal"], dtype=np.float64
        )
        reference_off_diagonal = np.asarray(
            reference["tridiagonal_off_diagonal"], dtype=np.float64
        )
        extended_off_diagonal = np.asarray(
            extended["tridiagonal_off_diagonal"], dtype=np.float64
        )
        reference_ritz_values = np.asarray(
            reference["ritz_values"], dtype=np.float64
        )
        reference_weights = np.asarray(
            reference["spectral_weights"], dtype=np.float64
        )
        extended_ritz_values = np.asarray(
            extended["ritz_values"], dtype=np.float64
        )
        extended_residuals = np.asarray(
            extended["ritz_residual_l2_estimates"], dtype=np.float64
        )
        extended_weights = np.asarray(
            extended["spectral_weights"], dtype=np.float64
        )
        reference_residuals = np.asarray(
            reference["ritz_residual_l2_estimates"], dtype=np.float64
        )
        prefix_diagonal = extended_diagonal[:reference_dimension]
        prefix_off_diagonal = extended_off_diagonal[
            : reference_dimension - 1
        ]
        prefix_matrix = np.diag(prefix_diagonal)
        prefix_matrix += np.diag(prefix_off_diagonal, 1)
        prefix_matrix += np.diag(prefix_off_diagonal, -1)
        prefix_values, prefix_vectors = np.linalg.eigh(prefix_matrix)
        prefix_weights = np.square(prefix_vectors[0])
        prefix_weights /= np.sum(prefix_weights, dtype=np.float64)
        reference_terminal_beta = float(reference["terminal_beta"])
        extended_prefix_terminal_beta = float(
            extended_off_diagonal[reference_dimension - 1]
        )
        reference_lowest_value = float(reference_ritz_values[0])
        extended_lowest_value = float(extended_ritz_values[0])
        reference_lowest_residual = float(reference_residuals[0])
        extended_lowest_residual = float(extended_residuals[0])
        reference_residual_to_value = reference_lowest_residual / max(
            abs(reference_lowest_value), np.finfo(np.float64).eps
        )
        extended_residual_to_value = extended_lowest_residual / max(
            abs(extended_lowest_value), np.finfo(np.float64).eps
        )
        return {
            "reference_krylov_dimension": reference_completed,
            "extended_krylov_dimension": extended_completed,
            "diagonal_prefix_exact": bool(
                np.array_equal(reference_diagonal, prefix_diagonal)
            ),
            "off_diagonal_prefix_exact": bool(
                np.array_equal(reference_off_diagonal, prefix_off_diagonal)
            ),
            "terminal_beta_prefix_exact": bool(
                reference_terminal_beta == extended_prefix_terminal_beta
            ),
            "diagonal_prefix_max_abs_error": float(
                np.max(np.abs(reference_diagonal - prefix_diagonal))
            ),
            "off_diagonal_prefix_max_abs_error": float(
                np.max(
                    np.abs(reference_off_diagonal - prefix_off_diagonal)
                )
            ),
            "terminal_beta_prefix_abs_error": abs(
                reference_terminal_beta - extended_prefix_terminal_beta
            ),
            "prefix_ritz_values_exact": bool(
                np.array_equal(reference_ritz_values, prefix_values)
            ),
            "prefix_ritz_values_max_abs_error": float(
                np.max(np.abs(reference_ritz_values - prefix_values))
            ),
            "prefix_spectral_weights_exact": bool(
                np.array_equal(reference_weights, prefix_weights)
            ),
            "prefix_spectral_weights_max_abs_error": float(
                np.max(np.abs(reference_weights - prefix_weights))
            ),
            "reference_lowest_ritz_value": reference_lowest_value,
            "extended_lowest_ritz_value": extended_lowest_value,
            "extended_to_reference_lowest_ritz_value_ratio": (
                extended_lowest_value
                / max(abs(reference_lowest_value), np.finfo(np.float64).eps)
            ),
            "reference_lowest_ritz_residual_l2": reference_lowest_residual,
            "extended_lowest_ritz_residual_l2": extended_lowest_residual,
            "extended_to_reference_lowest_ritz_residual_ratio": (
                extended_lowest_residual
                / max(reference_lowest_residual, np.finfo(np.float64).eps)
            ),
            "reference_lowest_ritz_residual_to_value": (
                reference_residual_to_value
            ),
            "extended_lowest_ritz_residual_to_value": (
                extended_residual_to_value
            ),
            "extended_to_reference_residual_to_value_ratio": (
                extended_residual_to_value
                / max(reference_residual_to_value, np.finfo(np.float64).eps)
            ),
            "reference_lowest_ritz_spectral_weight": float(
                reference_weights[0]
            ),
            "extended_lowest_ritz_spectral_weight": float(
                extended_weights[0]
            ),
            "reference_lowest_reported_ritz_spectral_weight": float(
                np.sum(reference_weights[:low_count], dtype=np.float64)
            ),
            "extended_lowest_reported_ritz_spectral_weight": float(
                np.sum(extended_weights[:low_count], dtype=np.float64)
            ),
            "reference_spectral_weighted_median_ritz_value": float(
                reference["spectral_weighted_median_ritz_value"]
            ),
            "extended_spectral_weighted_median_ritz_value": float(
                extended["spectral_weighted_median_ritz_value"]
            ),
            "extended_to_reference_weighted_median_ritz_value_ratio": (
                float(extended["spectral_weighted_median_ritz_value"])
                / max(
                    abs(
                        float(
                            reference["spectral_weighted_median_ritz_value"]
                        )
                    ),
                    np.finfo(np.float64).eps,
                )
            ),
            "start_rayleigh_quotient_exact": bool(
                float(reference["start_rayleigh_quotient"])
                == float(extended["start_rayleigh_quotient"])
            ),
            "basis_real_orthogonality_max_abs_error_extended": float(
                extended["basis_real_orthogonality_max_abs_error"]
            ),
        }

    pairwise = compare_start("pairwise_start")
    random = compare_start("random_start")
    return {
        "reference_krylov_dimension": reference_dimension,
        "extended_krylov_dimension": int(
            extended_diagnostic["pairwise_start"][
                "completed_krylov_dimension"
            ]
        ),
        "recurrence_prefix_dimension": reference_dimension,
        "reported_low_ritz_count": low_count,
        "pairwise_start": pairwise,
        "random_start": random,
        "both_recurrence_prefixes_exact": bool(
            pairwise["diagonal_prefix_exact"]
            and pairwise["off_diagonal_prefix_exact"]
            and pairwise["terminal_beta_prefix_exact"]
            and random["diagonal_prefix_exact"]
            and random["off_diagonal_prefix_exact"]
            and random["terminal_beta_prefix_exact"]
        ),
        "same_source_operator_starts_and_seed": True,
        "single_extension_only": True,
        "truth_used_by_control": False,
        "scientific_thresholds_preregistered": False,
        "diagnostic_role": "bounded_12_to_24_krylov_dimension_extension",
        "interpretation_boundary": (
            "one bounded dimension extension; no complete spectrum, rank, "
            "resolution, or detection-limit claim"
        ),
    }


def truth_free_paired_continuation_diagnostic(
    operator: MatchedKnownBProbeOperator,
    primary_continuation: Mapping[str, Any],
    control_continuation: Mapping[str, Any],
    *,
    checkpoint_interval: int,
) -> dict[str, Any]:
    """Track a paired continuation without simulation-truth information."""

    interval = int(checkpoint_interval)
    if interval <= 0 or interval != float(checkpoint_interval):
        raise ValueError("checkpoint_interval must be a positive integer.")
    primary_history = np.asarray(
        primary_continuation["probe_history"], dtype=np.complex128
    )
    control_history = np.asarray(
        control_continuation["probe_history"], dtype=np.complex128
    )
    if (
        primary_history.ndim != 3
        or primary_history.shape != control_history.shape
        or primary_history.shape[1:] != operator.native_shape
        or not np.all(np.isfinite(primary_history))
        or not np.all(np.isfinite(control_history))
    ):
        raise ValueError("Continuation histories must be paired finite fields.")
    iterations = primary_history.shape[0] - 1
    if iterations <= 0 or iterations % interval:
        raise ValueError("checkpoint_interval must divide completed iterations.")
    if (
        int(primary_continuation["iterations_completed"]) != iterations
        or int(control_continuation["iterations_completed"]) != iterations
    ):
        raise ValueError("Continuation histories and iteration counts disagree.")
    if (
        primary_continuation.get("truth_used_by_optimizer") is not False
        or control_continuation.get("truth_used_by_optimizer") is not False
    ):
        raise ValueError("Truth is forbidden from paired continuation.")
    if not (
        np.array_equal(primary_history[0], primary_continuation["P_B_init"])
        and np.array_equal(primary_history[-1], primary_continuation["P_B_rec"])
        and np.array_equal(control_history[0], control_continuation["P_B_init"])
        and np.array_equal(control_history[-1], control_continuation["P_B_rec"])
    ):
        raise ValueError("Continuation history endpoints are inconsistent.")

    aligned_differences: list[NDArray[np.complex128]] = []
    alignment_factors: list[complex] = []
    raw_relative_l2_curve: list[float] = []
    aligned_relative_l2_curve: list[float] = []
    for primary, control in zip(
        primary_history, control_history, strict=True
    ):
        control_aligned, alignment_factor = _align_global_phase(
            control, primary
        )
        aligned_differences.append(control_aligned - primary)
        alignment_factors.append(alignment_factor)
        raw_relative_l2_curve.append(relative_l2(control, primary))
        aligned_relative_l2_curve.append(
            relative_l2(control_aligned, primary)
        )
    difference_history = np.stack(aligned_differences)
    base_norm = _l2_norm(difference_history[0])
    if base_norm <= np.finfo(np.float64).eps:
        raise ValueError("The initial pairwise continuation direction is zero.")
    base_direction = np.asarray(
        difference_history[0] / base_norm, dtype=np.complex128
    )
    projection_curve = np.asarray(
        [
            float(np.real(_complex_inner_product(base_direction, difference)))
            / base_norm
            for difference in difference_history
        ],
        dtype=np.float64,
    )
    direction_cosine_curve = np.asarray(
        [
            float(np.real(_complex_inner_product(base_direction, difference)))
            / max(_l2_norm(difference), np.finfo(np.float64).eps)
            for difference in difference_history
        ],
        dtype=np.float64,
    )
    checkpoint_iterations = np.arange(
        0, iterations + 1, interval, dtype=np.int64
    )
    prediction_relative_l2: list[float] = []
    pairwise_gains: list[float] = []
    for index in checkpoint_iterations:
        primary = primary_history[index]
        control = control_history[index]
        primary_prediction = operator.predict_stack(primary)
        control_prediction = operator.predict_stack(control)
        prediction_relative_l2.append(
            relative_l2(control_prediction, primary_prediction)
        )
        sensitivity = normalized_intensity_jacobian_sensitivity(
            operator, primary, difference_history[index]
        )
        pairwise_gains.append(float(sensitivity["jacobian_rms_gain"]))
    prediction_curve = np.asarray(prediction_relative_l2, dtype=np.float64)
    gain_curve = np.asarray(pairwise_gains, dtype=np.float64)
    aligned_curve = np.asarray(aligned_relative_l2_curve, dtype=np.float64)
    raw_curve = np.asarray(raw_relative_l2_curve, dtype=np.float64)
    return {
        "P_B_pairwise_start_direction_unit_l2": base_direction,
        "checkpoint_iterations": checkpoint_iterations,
        "pairwise_alignment_factor_curve": np.asarray(
            alignment_factors, dtype=np.complex128
        ),
        "pairwise_raw_relative_l2_curve": raw_curve,
        "pairwise_global_phase_aligned_relative_l2_curve": aligned_curve,
        "pairwise_start_direction_projection_fraction_curve": projection_curve,
        "pairwise_start_direction_real_cosine_curve": direction_cosine_curve,
        "checkpoint_prediction_relative_l2_curve": prediction_curve,
        "checkpoint_pairwise_jacobian_rms_gain_curve": gain_curve,
        "initial_pairwise_global_phase_aligned_relative_l2": aligned_curve[0],
        "final_pairwise_global_phase_aligned_relative_l2": aligned_curve[-1],
        "final_to_initial_pairwise_probe_difference_ratio": (
            aligned_curve[-1] / aligned_curve[0]
        ),
        "initial_prediction_relative_l2": prediction_curve[0],
        "final_prediction_relative_l2": prediction_curve[-1],
        "final_to_initial_prediction_difference_ratio": (
            prediction_curve[-1] / prediction_curve[0]
        ),
        "initial_pairwise_jacobian_rms_gain": gain_curve[0],
        "final_pairwise_jacobian_rms_gain": gain_curve[-1],
        "final_to_initial_pairwise_jacobian_gain_ratio": (
            gain_curve[-1] / gain_curve[0]
        ),
        "final_start_direction_projection_fraction": projection_curve[-1],
        "final_start_direction_real_cosine": direction_cosine_curve[-1],
        "additional_iterations": iterations,
        "checkpoint_interval": interval,
        "start_field_source": "same_run_primary_and_control_raw_final",
        "truth_used_by_continuation": False,
        "truth_used_by_diagnostic": False,
        "diagnostic_role": "truth_free_equal_budget_paired_continuation",
    }


def simulation_evaluation_only(
    reconstruction: Mapping[str, Any], truth: ComplexArray
) -> dict[str, Any]:
    """Compute post-hoc truth-aided probe metrics without optimizer feedback."""

    reference = np.asarray(truth, dtype=np.complex128)
    history = np.asarray(reconstruction["probe_history"], dtype=np.complex128)
    if history.ndim != 3 or history.shape[1:] != reference.shape:
        raise ValueError("probe_history and truth shapes disagree.")
    raw_curve = np.asarray(
        [relative_l2(probe, reference) for probe in history],
        dtype=np.float64,
    )
    aligned_fields: list[NDArray[np.complex128]] = []
    factors: list[complex] = []
    aligned_curve: list[float] = []
    for probe in history:
        aligned, factor = align_global_phase_simulation_only(probe, reference)
        aligned_fields.append(aligned)
        factors.append(factor)
        aligned_curve.append(relative_l2(aligned, reference))
    return {
        "P_B_rec_global_phase_aligned": aligned_fields[-1],
        "global_phase_alignment_factor": factors[-1],
        "probe_raw_relative_l2_curve": raw_curve,
        "probe_global_phase_aligned_relative_l2_curve": np.asarray(
            aligned_curve, dtype=np.float64
        ),
        "initial_probe_raw_relative_l2": float(raw_curve[0]),
        "final_probe_raw_relative_l2": float(raw_curve[-1]),
        "initial_probe_global_phase_aligned_relative_l2": float(
            aligned_curve[0]
        ),
        "final_probe_global_phase_aligned_relative_l2": float(
            aligned_curve[-1]
        ),
        "truth_used_by_optimizer": False,
        "evaluation_role": "simulation_evaluation_only",
    }
