"""Preregistered local-differentiability controls for exp051."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.inverse.exp051 import sha256_file, validate_exp051_config
from tgv_ptycho.inverse.waist_fit import raw_complex_probe_loss
from tgv_ptycho.objects.tgv3d import (
    make_tgv_air_fraction_slice,
    make_tgv_air_fraction_slice_chord_quadrature,
)
from tgv_ptycho.objects.tgv_geometry import diameter_profile, midpoint_z_grid

ComplexGenerator = Callable[[float], NDArray[np.complex128]]
FractionBuilder = Callable[
    [tuple[int, int], float, float, int, tuple[float, float]],
    NDArray[np.float64],
]


@dataclass(frozen=True)
class PriorExp051Artifact:
    """Validated identity of the initial exp051 formal run."""

    run_dir: Path
    file_sha256: dict[str, str]
    metrics: dict[str, Any]
    run_state: dict[str, Any]


def _relative_l2(
    test: NDArray[np.generic], reference: NDArray[np.generic]
) -> float:
    numerator = float(np.linalg.norm(np.asarray(test) - np.asarray(reference)))
    denominator = float(np.linalg.norm(np.asarray(reference)))
    return numerator / max(denominator, np.finfo(np.float64).eps)


def validate_exp051_local_control_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen 2026-08-24 q8 local-control contract."""

    validate_exp051_config(config)
    if config["experiment"].get("role") != (
        "3d_scalar_multislice_true_probe_q8_local_differentiability_control"
    ):
        raise ValueError("Unexpected exp051 local-control role.")
    control = config.get("local_control", {})
    expected_steps = np.asarray(
        [
            5.0e-7,
            2.5e-7,
            1.25e-7,
            6.25e-8,
            3.125e-8,
            1.5625e-8,
            7.8125e-9,
            3.90625e-9,
        ],
        dtype=np.float64,
    )
    steps = np.asarray(control.get("step_family_m"), dtype=np.float64)
    if not np.array_equal(steps, expected_steps):
        raise ValueError("exp051 local-control step family changed.")
    if (
        control.get("q8_interface_method") != "subpixel_midpoint_count"
        or int(control.get("q8_interface_factor", 0)) != 8
        or control.get("chord_interface_method")
        != "analytic_chord_gauss_legendre_cell_average"
        or int(control.get("chord_formal_order", 0)) != 64
        or int(control.get("chord_reference_order", 0)) != 128
        or float(control.get("inside_gap_fraction", 0.0)) != 0.5
        or float(control.get("cross_gap_fraction", 0.0)) != 1.5
    ):
        raise ValueError("exp051 local-control interface contract changed.")
    thresholds = control.get("thresholds", {})
    expected_thresholds = {
        "source_replay_relative_l2_max": 1.0e-12,
        "deterministic_repeat_relative_l2_max": 1.0e-14,
        "cross_probe_response_relative_l2_min": 1.0e-12,
        "chord_geometry_order_relative_l2_max": 1.0e-5,
        "chord_geometry_volume_relative_error_max": 1.0e-5,
        "chord_final_jacobian_relative_l2_max": 0.10,
        "chord_final_normalized_jacobian_per_m_min": 1.0e4,
    }
    if set(thresholds) != set(expected_thresholds) or any(
        float(thresholds[key]) != value
        for key, value in expected_thresholds.items()
    ):
        raise ValueError("exp051 local-control thresholds changed.")
    prior = config.get("prior_exp051", {})
    if prior.get("required_interpretation") != (
        "local_jacobian_numerical_control_not_closed"
    ):
        raise ValueError("Prior exp051 interpretation lock changed.")


def load_prior_exp051_formal(
    config: Mapping[str, Any], project_root: Path
) -> PriorExp051Artifact:
    """Load and hash-check the initial exp051 formal run."""

    validate_exp051_local_control_config(config)
    prior = config["prior_exp051"]
    run_dir = Path(prior["run"])
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    run_dir = run_dir.resolve()
    relative_paths = {
        "config": "config.yaml",
        "metadata": "metadata.json",
        "metrics": "metrics.json",
        "run_state": "run_state.json",
        "hdf5": str(prior["hdf5_relative_path"]),
    }
    actual: dict[str, str] = {}
    for key, relative in relative_paths.items():
        path = run_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Missing prior exp051 artifact: {path}")
        actual[key] = sha256_file(path)
        if actual[key] != str(prior["expected_sha256"][key]).upper():
            raise RuntimeError(f"Prior exp051 {key} hash mismatch.")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (run_dir / "run_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if (
        state.get("status") != "complete"
        or state.get("artifacts_validated") is not True
        or metrics.get("experiment_status") != "Inconclusive"
        or metrics.get("interpretation") != prior["required_interpretation"]
        or metrics.get("reference_validated") is not False
        or metrics.get("full_tgv_reference_authorized") is not False
        or metrics.get("p_b_rec_used_as_primary_input") is not False
    ):
        raise RuntimeError("Prior exp051 formal state/provenance differs.")
    return PriorExp051Artifact(run_dir, actual, metrics, state)


def q8_waist_breakpoint_map(
    source_config: Mapping[str, Any],
    *,
    bounds_m: tuple[float, float],
    true_waist_m: float,
    q: int = 8,
) -> dict[str, Any]:
    """Return analytic fixed-midpoint diameter breakpoints inside bounds."""

    sample = source_config["sample_a"]
    grid = source_config["probe_grid"]
    shape = tuple(int(value) for value in grid["native_shape"])
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("native_shape must be two positive integers.")
    dx_m = float(grid["node_dx_m"])
    if q <= 0 or not np.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("q and node spacing must be positive.")
    center_x, center_y = (
        float(value) for value in sample["center_xy_m"]
    )
    y_centers = (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * dx_m
    x_centers = (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * dx_m
    offsets = ((np.arange(q) + 0.5) / q - 0.5) * dx_m
    y_nodes = (y_centers[:, None] + offsets[None, :] - center_y).ravel()
    x_nodes = (x_centers[:, None] + offsets[None, :] - center_x).ravel()
    radii = np.hypot(y_nodes[:, None], x_nodes[None, :]).ravel()
    unique_radii, radial_multiplicity = np.unique(
        radii, return_counts=True
    )

    thickness = float(sample["thickness_m"])
    z_waist = float(sample["z_waist_m"])
    d_top = float(sample["d_top_m"])
    d_bottom = float(sample["d_bottom_m"])
    z_m, _ = midpoint_z_grid(thickness, float(sample["target_dz_m"]))
    before = z_m <= z_waist
    alpha = np.empty_like(z_m)
    constant = np.empty_like(z_m)
    alpha[before] = z_m[before] / z_waist
    constant[before] = d_top * (1.0 - alpha[before])
    after_fraction = (z_m[~before] - z_waist) / (thickness - z_waist)
    alpha[~before] = 1.0 - after_fraction
    constant[~before] = d_bottom * after_fraction
    if np.any(alpha <= 0.0):
        raise RuntimeError("Midpoint slice coefficients must be positive.")

    lower_bound, upper_bound = (float(value) for value in bounds_m)
    breakpoint_parts: list[NDArray[np.float64]] = []
    multiplicity_parts: list[NDArray[np.int64]] = []
    slice_parts: list[NDArray[np.int64]] = []
    for slice_index, (coefficient, offset) in enumerate(
        zip(alpha, constant, strict=True)
    ):
        values = (2.0 * unique_radii - offset) / coefficient
        selected = (
            np.isfinite(values)
            & (values >= lower_bound)
            & (values <= upper_bound)
        )
        breakpoint_parts.append(values[selected])
        multiplicity_parts.append(radial_multiplicity[selected].astype(np.int64))
        slice_parts.append(
            np.full(np.count_nonzero(selected), slice_index, dtype=np.int64)
        )
    raw_breakpoints = np.concatenate(breakpoint_parts)
    raw_multiplicity = np.concatenate(multiplicity_parts)
    raw_slice = np.concatenate(slice_parts)
    order = np.argsort(raw_breakpoints, kind="stable")
    raw_breakpoints = raw_breakpoints[order]
    raw_multiplicity = raw_multiplicity[order]
    raw_slice = raw_slice[order]
    unique_breakpoints, inverse = np.unique(
        raw_breakpoints, return_inverse=True
    )
    multiplicity = np.bincount(
        inverse, weights=raw_multiplicity
    ).astype(np.int64)
    below = unique_breakpoints < true_waist_m
    above = unique_breakpoints > true_waist_m
    if not np.any(below) or not np.any(above):
        raise RuntimeError("No q8 breakpoint brackets the true waist.")
    lower_index = int(np.flatnonzero(below)[-1])
    upper_index = int(np.flatnonzero(above)[0])
    lower = float(unique_breakpoints[lower_index])
    upper = float(unique_breakpoints[upper_index])
    lower_raw = raw_breakpoints == lower
    upper_raw = raw_breakpoints == upper
    equal = unique_breakpoints == true_waist_m
    return {
        "breakpoints_m": unique_breakpoints.astype(np.float64),
        "multiplicity": multiplicity,
        "lower_breakpoint_m": lower,
        "upper_breakpoint_m": upper,
        "lower_gap_m": float(true_waist_m - lower),
        "upper_gap_m": float(upper - true_waist_m),
        "lower_multiplicity": int(np.sum(raw_multiplicity[lower_raw])),
        "upper_multiplicity": int(np.sum(raw_multiplicity[upper_raw])),
        "lower_slice_indices": np.unique(raw_slice[lower_raw]).astype(np.int64),
        "upper_slice_indices": np.unique(raw_slice[upper_raw]).astype(np.int64),
        "equality_at_truth_multiplicity": int(np.sum(multiplicity[equal])),
        "q": int(q),
        "subnode_spacing_m": dx_m / q,
    }


def make_air_fraction_stack(
    source_config: Mapping[str, Any],
    d_waist_m: float,
    *,
    builder: FractionBuilder,
    resolution: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the registered 100-slice air-fraction stack for diagnostics."""

    sample = source_config["sample_a"]
    grid = source_config["probe_grid"]
    shape = tuple(int(value) for value in grid["native_shape"])
    dx_m = float(grid["node_dx_m"])
    thickness = float(sample["thickness_m"])
    z_m, widths = midpoint_z_grid(thickness, float(sample["target_dz_m"]))
    diameters = diameter_profile(
        z_m,
        thickness,
        float(sample["d_top_m"]),
        float(d_waist_m),
        float(sample["d_bottom_m"]),
        float(sample["z_waist_m"]),
    )
    center = tuple(float(value) for value in sample["center_xy_m"])
    stack = np.empty((len(z_m), *shape), dtype=np.float64)
    for index, diameter in enumerate(diameters):
        stack[index] = builder(
            shape, dx_m, float(diameter), int(resolution), center
        )
    return stack, z_m, widths


def air_fraction_change_metrics(
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    widths_m: NDArray[np.float64],
    dx_m: float,
    q: int,
) -> dict[str, Any]:
    """Measure discrete q-node changes without retaining another volume."""

    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if candidate_values.shape != reference_values.shape:
        raise ValueError("fraction stacks must have identical shapes.")
    if candidate_values.shape[0] != len(widths_m):
        raise ValueError("slice widths do not match fraction stacks.")
    difference = candidate_values - reference_values
    node_changes = np.abs(difference) * q**2
    return {
        "changed_fraction_voxel_count": int(np.count_nonzero(difference)),
        "changed_subpixel_node_count": int(np.rint(np.sum(node_changes))),
        "signed_subpixel_node_change": int(
            np.rint(np.sum(difference) * q**2)
        ),
        "signed_discrete_air_volume_change_m3": float(
            np.sum(
                difference
                * np.asarray(widths_m)[:, None, None]
                * dx_m**2,
                dtype=np.float64,
            )
        ),
        "fraction_stack_relative_l2": _relative_l2(
            candidate_values, reference_values
        ),
        "bitwise_equal": bool(
            np.array_equal(candidate_values, reference_values)
        ),
    }


def central_field_step_series(
    generator: ComplexGenerator,
    *,
    center_diameter_m: float,
    steps_m: NDArray[np.float64],
) -> dict[str, Any]:
    """Evaluate a complete central-difference field series and cache it."""

    steps = np.asarray(steps_m, dtype=np.float64)
    if (
        steps.ndim != 1
        or len(steps) < 2
        or np.any(steps <= 0.0)
        or np.any(np.diff(steps) >= 0.0)
    ):
        raise ValueError("steps must be a strictly decreasing positive vector.")
    center = np.asarray(generator(center_diameter_m), dtype=np.complex128)
    diameters = [float(center_diameter_m)]
    fields = [center]
    minus_indices: list[int] = []
    plus_indices: list[int] = []
    jacobians: list[NDArray[np.complex128]] = []
    one_sided: list[tuple[float, float]] = []
    asymmetry: list[float] = []
    center_norm = float(np.linalg.norm(center))
    for step in steps:
        minus = np.asarray(
            generator(float(center_diameter_m - step)), dtype=np.complex128
        )
        plus = np.asarray(
            generator(float(center_diameter_m + step)), dtype=np.complex128
        )
        if minus.shape != center.shape or plus.shape != center.shape:
            raise RuntimeError("candidate field shape changed in step series.")
        minus_indices.append(len(fields))
        fields.append(minus)
        diameters.append(float(center_diameter_m - step))
        plus_indices.append(len(fields))
        fields.append(plus)
        diameters.append(float(center_diameter_m + step))
        jacobian = (plus - minus) / (2.0 * float(step))
        jacobians.append(np.asarray(jacobian, dtype=np.complex128))
        delta_minus = minus - center
        delta_plus = plus - center
        one_sided.append(
            (
                float(np.linalg.norm(delta_minus) / center_norm),
                float(np.linalg.norm(delta_plus) / center_norm),
            )
        )
        asymmetry.append(
            float(
                np.linalg.norm(delta_plus + delta_minus)
                / max(
                    float(np.linalg.norm(delta_plus) + np.linalg.norm(delta_minus)),
                    np.finfo(np.float64).eps,
                )
            )
        )
    jacobian_stack = np.stack(jacobians).astype(np.complex128)
    jacobian_norm = np.linalg.norm(
        jacobian_stack.reshape(len(steps), -1), axis=1
    )
    pairwise = np.asarray(
        [
            np.linalg.norm(jacobian_stack[index] - jacobian_stack[index + 1])
            / max(jacobian_norm[index + 1], np.finfo(np.float64).eps)
            for index in range(len(steps) - 1)
        ],
        dtype=np.float64,
    )
    return {
        "steps_m": steps,
        "cache": {
            "D_waist_m": np.asarray(diameters, dtype=np.float64),
            "P_B_candidate": np.stack(fields).astype(np.complex128),
            "center_index": 0,
            "minus_indices": np.asarray(minus_indices, dtype=np.int64),
            "plus_indices": np.asarray(plus_indices, dtype=np.int64),
        },
        "jacobian": jacobian_stack,
        "normalized_jacobian_per_m": jacobian_norm / center_norm,
        "pairwise_jacobian_relative_l2": pairwise,
        "one_sided_probe_relative_l2": np.asarray(one_sided, dtype=np.float64),
        "second_difference_asymmetry": np.asarray(
            asymmetry, dtype=np.float64
        ),
        "zero_pair": np.asarray(
            [
                np.array_equal(fields[minus_indices[index]], center)
                and np.array_equal(fields[plus_indices[index]], center)
                for index in range(len(steps))
            ],
            dtype=np.bool_,
        ),
    }


def run_local_differentiability_control(
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    target: NDArray[np.complex128],
    q8_generator: ComplexGenerator,
    chord_generator: ComplexGenerator,
    *,
    memory_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run the complete preregistered q8/chord local-control diagnostic."""

    validate_exp051_local_control_config(config)
    control = config["local_control"]
    thresholds = control["thresholds"]
    true_waist = float(config["fit"]["true_d_waist_m"])
    bounds = tuple(float(value) for value in config["fit"]["bounds_m"])
    steps = np.asarray(control["step_family_m"], dtype=np.float64)
    target_values = np.asarray(target, dtype=np.complex128)
    q8_center = np.asarray(q8_generator(true_waist), dtype=np.complex128)
    q8_repeat = np.asarray(q8_generator(true_waist), dtype=np.complex128)
    source_replay = float(
        np.sqrt(raw_complex_probe_loss(q8_center, target_values))
    )
    q8_deterministic = _relative_l2(q8_repeat, q8_center)

    breakpoint = q8_waist_breakpoint_map(
        source_config,
        bounds_m=bounds,
        true_waist_m=true_waist,
        q=int(control["q8_interface_factor"]),
    )
    lower_gap = float(breakpoint["lower_gap_m"])
    upper_gap = float(breakpoint["upper_gap_m"])
    inside_fraction = float(control["inside_gap_fraction"])
    cross_fraction = float(control["cross_gap_fraction"])
    derived_diameters = {
        "inside_minus_m": true_waist - inside_fraction * lower_gap,
        "inside_plus_m": true_waist + inside_fraction * upper_gap,
        "cross_minus_m": true_waist - cross_fraction * lower_gap,
        "cross_plus_m": true_waist + cross_fraction * upper_gap,
    }
    if not bounds[0] <= min(derived_diameters.values()) or not max(
        derived_diameters.values()
    ) <= bounds[1]:
        raise RuntimeError("Derived breakpoint control lies outside bounds.")

    q = int(control["q8_interface_factor"])
    q8_truth_fraction, z_m, widths = make_air_fraction_stack(
        source_config,
        true_waist,
        builder=make_tgv_air_fraction_slice,
        resolution=q,
    )
    dx_m = float(source_config["probe_grid"]["node_dx_m"])
    plateau_probe: dict[str, NDArray[np.complex128]] = {}
    plateau_geometry: dict[str, Any] = {}
    plateau_determinism: dict[str, float] = {}
    for name, diameter in derived_diameters.items():
        probe = np.asarray(q8_generator(diameter), dtype=np.complex128)
        repeated = np.asarray(q8_generator(diameter), dtype=np.complex128)
        fraction, _, _ = make_air_fraction_stack(
            source_config,
            diameter,
            builder=make_tgv_air_fraction_slice,
            resolution=q,
        )
        plateau_probe[name] = probe
        plateau_geometry[name] = air_fraction_change_metrics(
            fraction,
            q8_truth_fraction,
            widths_m=widths,
            dx_m=dx_m,
            q=q,
        )
        plateau_geometry[name]["probe_relative_l2"] = _relative_l2(
            probe, q8_center
        )
        plateau_determinism[name] = _relative_l2(repeated, probe)
        if memory_callback is not None:
            memory_callback()

    q8_series = central_field_step_series(
        q8_generator, center_diameter_m=true_waist, steps_m=steps
    )
    q8_geometry_minus: list[dict[str, Any]] = []
    q8_geometry_plus: list[dict[str, Any]] = []
    for step in steps:
        minus, _, _ = make_air_fraction_stack(
            source_config,
            true_waist - float(step),
            builder=make_tgv_air_fraction_slice,
            resolution=q,
        )
        plus, _, _ = make_air_fraction_stack(
            source_config,
            true_waist + float(step),
            builder=make_tgv_air_fraction_slice,
            resolution=q,
        )
        q8_geometry_minus.append(
            air_fraction_change_metrics(
                minus,
                q8_truth_fraction,
                widths_m=widths,
                dx_m=dx_m,
                q=q,
            )
        )
        q8_geometry_plus.append(
            air_fraction_change_metrics(
                plus,
                q8_truth_fraction,
                widths_m=widths,
                dx_m=dx_m,
                q=q,
            )
        )

    smallest_step = float(steps[-1])
    smallest_repeats = []
    for diameter in (true_waist - smallest_step, true_waist + smallest_step):
        first = np.asarray(q8_generator(diameter), dtype=np.complex128)
        second = np.asarray(q8_generator(diameter), dtype=np.complex128)
        smallest_repeats.append(_relative_l2(second, first))
    q8_deterministic_max = max(
        [q8_deterministic, *plateau_determinism.values(), *smallest_repeats]
    )

    chord_order64, _, _ = make_air_fraction_stack(
        source_config,
        true_waist,
        builder=make_tgv_air_fraction_slice_chord_quadrature,
        resolution=int(control["chord_formal_order"]),
    )
    chord_order128, _, _ = make_air_fraction_stack(
        source_config,
        true_waist,
        builder=make_tgv_air_fraction_slice_chord_quadrature,
        resolution=int(control["chord_reference_order"]),
    )
    chord_geometry_relative = _relative_l2(chord_order64, chord_order128)
    volume64 = float(
        np.sum(chord_order64 * widths[:, None, None] * dx_m**2)
    )
    volume128 = float(
        np.sum(chord_order128 * widths[:, None, None] * dx_m**2)
    )
    chord_volume_relative = abs(volume64 - volume128) / abs(volume128)
    chord_geometry_pass = bool(
        chord_geometry_relative
        <= float(thresholds["chord_geometry_order_relative_l2_max"])
        and chord_volume_relative
        <= float(thresholds["chord_geometry_volume_relative_error_max"])
    )
    chord_center = np.asarray(chord_generator(true_waist), dtype=np.complex128)
    chord_repeat = np.asarray(chord_generator(true_waist), dtype=np.complex128)
    chord_deterministic = _relative_l2(chord_repeat, chord_center)
    chord_series: dict[str, Any] | None = None
    chord_derivative_pass = False
    if chord_geometry_pass:
        chord_series = central_field_step_series(
            chord_generator,
            center_diameter_m=true_waist,
            steps_m=steps,
        )
        final_pair = float(chord_series["pairwise_jacobian_relative_l2"][-1])
        final_norms = np.asarray(
            chord_series["normalized_jacobian_per_m"][-2:],
            dtype=np.float64,
        )
        chord_derivative_pass = bool(
            final_pair
            <= float(thresholds["chord_final_jacobian_relative_l2_max"])
            and float(np.min(final_norms))
            >= float(thresholds["chord_final_normalized_jacobian_per_m_min"])
            and chord_deterministic
            <= float(thresholds["deterministic_repeat_relative_l2_max"])
        )

    inside_names = ("inside_minus_m", "inside_plus_m")
    cross_names = ("cross_minus_m", "cross_plus_m")
    source_gate = bool(
        source_replay
        <= float(thresholds["source_replay_relative_l2_max"])
        and q8_deterministic_max
        <= float(thresholds["deterministic_repeat_relative_l2_max"])
    )
    breakpoint_order_gate = bool(
        lower_gap > 0.0
        and upper_gap > 0.0
        and breakpoint["equality_at_truth_multiplicity"] == 0
    )
    inside_gate = all(
        plateau_geometry[name]["bitwise_equal"]
        and plateau_geometry[name]["probe_relative_l2"] == 0.0
        for name in inside_names
    )
    crossing_gate = all(
        plateau_geometry[name]["changed_subpixel_node_count"] > 0
        and plateau_geometry[name]["probe_relative_l2"]
        >= float(thresholds["cross_probe_response_relative_l2_min"])
        for name in cross_names
    )
    q8_hard_controls_pass = bool(
        source_gate and breakpoint_order_gate and inside_gate and crossing_gate
    )
    chord_controls_pass = bool(chord_geometry_pass and chord_derivative_pass)
    if not q8_hard_controls_pass:
        status = "DiagnosticFailed"
        interpretation = "q8_breakpoint_or_artifact_hard_control_failed"
    elif breakpoint_order_gate and inside_gate and chord_controls_pass:
        status = "DiagnosticPassed"
        interpretation = (
            "q8_midpoint_piecewise_constant_at_truth__"
            "chord_control_locally_converged"
        )
    elif breakpoint_order_gate and inside_gate:
        status = "DiagnosticInconclusive"
        interpretation = "q8_plateau_attributed__smooth_control_not_closed"
    else:
        status = "DiagnosticInconclusive"
        interpretation = "prior_jacobian_failure_not_attributed"

    waist_depth = float(source_config["sample_a"]["z_waist_m"])
    selected_slice = int(np.argmin(np.abs(z_m - waist_depth)))
    return {
        "diagnostic_status": status,
        "interpretation": interpretation,
        "experiment_status": "Inconclusive",
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "source_replay_relative_l2": source_replay,
        "q8_deterministic_repeat_relative_l2_max": q8_deterministic_max,
        "derived_diameters": derived_diameters,
        "breakpoint": breakpoint,
        "q8": {
            "series": q8_series,
            "plateau_geometry": plateau_geometry,
            "geometry_step_minus": q8_geometry_minus,
            "geometry_step_plus": q8_geometry_plus,
            "plateau_probe": plateau_probe,
            "truth_fraction_selected_slice": q8_truth_fraction[selected_slice],
        },
        "chord_control": {
            "geometry_order_relative_l2": chord_geometry_relative,
            "geometry_volume_relative_error": chord_volume_relative,
            "geometry_order_pass": chord_geometry_pass,
            "deterministic_repeat_relative_l2": chord_deterministic,
            "q8_to_chord_center_relative_l2": _relative_l2(
                chord_center, q8_center
            ),
            "derivative_pass": chord_derivative_pass,
            "series": chord_series,
            "order64_fraction_selected_slice": chord_order64[selected_slice],
            "order128_fraction_selected_slice": chord_order128[selected_slice],
        },
        "selected_slice_index": selected_slice,
        "gates": {
            "source_replay_and_determinism_pass": source_gate,
            "breakpoint_order_pass": breakpoint_order_gate,
            "inside_plateau_exact_pass": inside_gate,
            "crossing_observed_pass": crossing_gate,
            "q8_hard_controls_pass": q8_hard_controls_pass,
            "chord_geometry_pass": chord_geometry_pass,
            "chord_derivative_pass": chord_derivative_pass,
            "chord_controls_pass": chord_controls_pass,
        },
    }
