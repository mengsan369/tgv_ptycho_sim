"""Minimal geometry-nuisance identifiability for the exp040 scalar model."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.exp040 import build_scalar_working_model_probe
from tgv_ptycho.inverse.exp051 import (
    Exp051SourceArtifact,
    load_exp051_source_true_probe,
    sha256_array_bytes,
)
from tgv_ptycho.inverse.exp051_plateau_interval import (
    build_q8_cell_partition,
    expand_connected_cell_component,
    fixed_membership_bisection,
    q8_cell_index,
    q8_cell_midpoint,
)
from tgv_ptycho.inverse.waist_fit import raw_complex_probe_loss

ParameterKey = tuple[float, float, float]
CandidateGenerator = Callable[[float, float, float], NDArray[np.complex128]]


@dataclass
class ProbeCache:
    """Cache raw fields and loss without writing the full stack to artifacts."""

    generator: CandidateGenerator
    target: NDArray[np.complex128]
    values: dict[ParameterKey, int] = field(default_factory=dict)
    parameters_m: list[ParameterKey] = field(default_factory=list)
    fields: list[NDArray[np.complex128]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    field_sha256: list[str] = field(default_factory=list)

    def get(
        self, d_waist_m: float, d_surface_m: float, z_waist_m: float
    ) -> tuple[NDArray[np.complex128], float, int]:
        """Return one cached raw candidate, normalized loss, and cache index."""

        key = tuple(
            float(np.float64(value))
            for value in (d_waist_m, d_surface_m, z_waist_m)
        )
        if not all(np.isfinite(key)) or min(key) <= 0.0:
            raise ValueError("candidate geometry parameters must be finite/positive.")
        cached = self.values.get(key)
        if cached is not None:
            return self.fields[cached], self.losses[cached], cached
        probe = np.asarray(self.generator(*key))
        if probe.shape != self.target.shape or probe.dtype != np.complex128:
            raise ValueError("candidate generator returned a wrong shape or dtype.")
        if not np.all(np.isfinite(probe)):
            raise ValueError("candidate generator returned non-finite values.")
        loss = float(raw_complex_probe_loss(probe, self.target))
        index = len(self.fields)
        self.values[key] = index
        self.parameters_m.append(key)
        self.fields.append(probe.copy())
        self.losses.append(loss)
        self.field_sha256.append(sha256_array_bytes(probe))
        return self.fields[index], loss, index

    def artifact_payload(self) -> dict[str, Any]:
        """Return compact cache provenance without the full candidate stack."""

        params = np.asarray(self.parameters_m, dtype=np.float64)
        if params.size == 0:
            params = np.empty((0, 3), dtype=np.float64)
        return {
            "parameters_m": params,
            "loss": np.asarray(self.losses, dtype=np.float64),
            "field_sha256": list(self.field_sha256),
            "entry_count": len(self.losses),
            "parameter_order": ["d_waist_m", "d_surface_m", "z_waist_m"],
        }


def validate_exp055_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen section-1 exp055 contract."""

    experiment = config.get("experiment", {})
    if (
        experiment.get("id") != "exp055"
        or experiment.get("stage") != "true_probe"
        or experiment.get("reference_validated") is not False
        or experiment.get("full_tgv_reference_authorized") is not False
    ):
        raise ValueError("exp055 experiment identity/provenance changed.")
    source = config.get("source", {})
    if source.get("target_hdf5_path") != "/entry/truth/P_B_true":
        raise ValueError("exp055 first stage must use raw /entry/truth/P_B_true.")
    lowered = str(source.get("target_hdf5_path", "")).lower()
    if any(
        str(token).lower() in lowered
        for token in source.get("forbidden_primary_input_tokens", [])
    ):
        raise ValueError("exp055 primary target contains a forbidden token.")
    fixed = config.get("fixed_model", {})
    if (
        fixed.get("shape") != [96, 96]
        or int(fixed.get("interface_factor", 0)) != 8
        or fixed.get("complex_dtype") != "complex128"
        or fixed.get("internal_alias_control") is not False
        or fixed.get("external_alias_control") is not True
    ):
        raise ValueError("exp055 fixed scalar working-model identity changed.")
    nuisance = config.get("parameters", {}).get("nuisance", [])
    if (
        len(nuisance) != 2
        or nuisance[0].get("name") != "d_surface_m"
        or nuisance[0].get("applies_to") != ["d_top_m", "d_bottom_m"]
        or nuisance[0].get("tied") is not True
        or nuisance[1].get("name") != "z_waist_m"
    ):
        raise ValueError("exp055 registered minimal parameter set changed.")
    fit = config.get("fit", {})
    if (
        fit.get("primary_loss") != "raw_complex_normalized_squared_l2"
        or fit.get("mask") != "full_native_field_all_true"
        or fit.get("phase_or_scale_alignment") != "none"
        or fit.get("connectivity") != "six_neighbor_cartesian_lattice"
    ):
        raise ValueError("exp055 primary loss/equivalence convention changed.")
    expected_axes = {
        "coarse_grid": (
            np.arange(16.0, 25.0) * 1.0e-6,
            np.arange(28.0, 33.0) * 1.0e-6,
            np.arange(45.0, 55.1, 2.5) * 1.0e-6,
        ),
        "local_grid": (
            np.arange(19.5, 20.5001, 0.125) * 1.0e-6,
            np.arange(29.5, 30.5001, 0.125) * 1.0e-6,
            np.arange(48.0, 52.0001, 0.5) * 1.0e-6,
        ),
    }
    for grid_name, expected in expected_axes.items():
        grid = fit.get(grid_name, {})
        actual = (
            np.asarray(grid.get("d_waist_m", []), dtype=np.float64),
            np.asarray(grid.get("d_surface_m", []), dtype=np.float64),
            np.asarray(grid.get("z_waist_m", []), dtype=np.float64),
        )
        if any(
            not np.allclose(left, right, rtol=0.0, atol=1.0e-20)
            for left, right in zip(actual, expected, strict=True)
        ):
            raise ValueError(f"exp055 frozen {grid_name} changed.")
    equivalence = fit.get("equivalence_relative_l2", {})
    fiber = fit.get("q8_fiber", {})
    paths = fit.get("multistart_paths", {})
    if (
        equivalence != {"primary": 1.0e-12, "low": 1.0e-13, "high": 1.0e-11}
        or int(fiber.get("max_adjacent_cells_per_direction", 0)) != 256
        or int(fiber.get("boundary_bisection_iterations", 0)) != 24
        or float(fiber.get("boundary_resolution_m_max", 0.0)) != 1.0e-12
        or paths.get("algorithm")
        != "deterministic_six_neighbor_steepest_descent_on_frozen_local_grid"
        or int(paths.get("maximum_moves_per_start", 0)) != 32
    ):
        raise ValueError("exp055 frozen threshold/search contract changed.")
    output = config.get("output", {})
    if (
        output.get("hdf5_filename") != "exp055_nuisance_identifiability.h5"
        or len(output.get("figure_filenames", [])) != 5
    ):
        raise ValueError("exp055 output contract changed.")


def _exp051_source_adapter(config: Mapping[str, Any]) -> dict[str, Any]:
    adapter = deepcopy(dict(config))
    adapter["experiment"] = deepcopy(dict(config["experiment"]))
    adapter["experiment"]["id"] = "exp051"
    adapter["fit"] = deepcopy(dict(config["fit"]))
    adapter["fit"].update(
        {
            "true_d_waist_m": 2.0e-5,
            "bounds_m": [1.6e-5, 2.4e-5],
            "coarse_grid_m": [value * 1.0e-6 for value in range(16, 25)],
            "optimizer": {
                "starts_m": [16.5e-6, 18.5e-6, 21.5e-6, 23.5e-6],
                "evaluation_budget_per_start": 41,
            },
        }
    )
    return adapter


def load_exp055_true_probe(
    config: Mapping[str, Any], project_root: Any
) -> Exp051SourceArtifact:
    """Reuse the strict exp051 raw-true-probe validator for exp055."""

    validate_exp055_config(config)
    return load_exp051_source_true_probe(
        _exp051_source_adapter(config), project_root
    )


def make_exp055_candidate_generator(
    source_config: Mapping[str, Any],
    *,
    memory_callback: Callable[[], None] | None = None,
) -> CandidateGenerator:
    """Return the matched q8 candidate generator for three geometry values."""

    frozen = deepcopy(dict(source_config))

    def generate(
        d_waist_m: float, d_surface_m: float, z_waist_m: float
    ) -> NDArray[np.complex128]:
        values = (d_waist_m, d_surface_m, z_waist_m)
        if not all(np.isfinite(values)) or min(values) <= 0.0:
            raise ValueError("candidate parameters must be finite and positive.")
        candidate = deepcopy(frozen)
        candidate["sample_a"]["d_waist_m"] = float(d_waist_m)
        candidate["sample_a"]["d_top_m"] = float(d_surface_m)
        candidate["sample_a"]["d_bottom_m"] = float(d_surface_m)
        candidate["sample_a"]["z_waist_m"] = float(z_waist_m)
        generated = build_scalar_working_model_probe(candidate)
        if memory_callback is not None:
            memory_callback()
        return np.asarray(generated["P_B"], dtype=np.complex128)

    return generate


def _grid_axes(
    fit: Mapping[str, Any], name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    grid = fit[name]
    return tuple(
        np.asarray(grid[key], dtype=np.float64)
        for key in ("d_waist_m", "d_surface_m", "z_waist_m")
    )  # type: ignore[return-value]


def _evaluate_grid(
    cache: ProbeCache,
    axes: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    shape = tuple(len(axis) for axis in axes)
    loss = np.empty(shape, dtype=np.float64)
    cache_index = np.empty(shape, dtype=np.int64)
    for index in np.ndindex(shape):
        params = tuple(float(axes[dim][index[dim]]) for dim in range(3))
        _, loss[index], cache_index[index] = cache.get(*params)
    return loss, cache_index


def connected_components_3d(mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Label six-neighbor connected components of a three-dimensional mask."""

    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 3:
        raise ValueError("connected component mask must be three-dimensional.")
    labels = np.full(values.shape, -1, dtype=np.int64)
    components: list[NDArray[np.int64]] = []
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for seed in np.argwhere(values):
        seed_key = tuple(int(value) for value in seed)
        if labels[seed_key] >= 0:
            continue
        label = len(components)
        queue: deque[tuple[int, int, int]] = deque([seed_key])
        labels[seed_key] = label
        members: list[tuple[int, int, int]] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for delta in neighbors:
                nxt = tuple(current[dim] + delta[dim] for dim in range(3))
                outside = any(
                    nxt[dim] < 0 or nxt[dim] >= values.shape[dim]
                    for dim in range(3)
                )
                if outside:
                    continue
                if values[nxt] and labels[nxt] < 0:
                    labels[nxt] = label
                    queue.append(nxt)
        components.append(np.asarray(members, dtype=np.int64))
    return {"labels": labels, "components": components, "count": len(components)}


def _multistart_paths(
    loss: NDArray[np.float64], starts: Sequence[Sequence[int]], max_moves: int
) -> dict[str, Any]:
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    branches: dict[str, Any] = {}
    for branch_index, start in enumerate(starts):
        current = tuple(int(value) for value in start)
        track = [current]
        losses = [float(loss[current])]
        reason = "maximum_moves"
        for _ in range(max_moves):
            candidates = [current]
            for delta in neighbors:
                nxt = tuple(current[dim] + delta[dim] for dim in range(3))
                if all(0 <= nxt[dim] < loss.shape[dim] for dim in range(3)):
                    candidates.append(nxt)
            best = min(candidates, key=lambda idx: (float(loss[idx]), idx))
            if not float(loss[best]) < float(loss[current]):
                reason = "local_minimum"
                break
            current = best
            track.append(current)
            losses.append(float(loss[current]))
        branches[f"start_{branch_index:02d}"] = {
            "start_index": np.asarray(start, dtype=np.int64),
            "visited_index": np.asarray(track, dtype=np.int64),
            "visited_loss": np.asarray(losses, dtype=np.float64),
            "terminal_index": np.asarray(current, dtype=np.int64),
            "move_count": len(track) - 1,
            "stopping_reason": reason,
        }
    return branches


def _fiber_component_signature(component: Mapping[str, Any]) -> tuple[int, int]:
    return int(component["left_cell_index"]), int(component["right_cell_index"])


def _q8_fiber(
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    cache: ProbeCache,
    *,
    seed_m: tuple[float, float, float],
) -> dict[str, Any]:
    fit = config["fit"]
    design = fit["q8_fiber"]
    tau_values = fit["equivalence_relative_l2"]
    modified = deepcopy(dict(source_config))
    modified["sample_a"]["d_top_m"] = seed_m[1]
    modified["sample_a"]["d_bottom_m"] = seed_m[1]
    modified["sample_a"]["z_waist_m"] = seed_m[2]
    partition = build_q8_cell_partition(
        modified,
        bounds_m=(1.6e-5, 2.4e-5),
        reference_diameter_m=seed_m[0],
        q=8,
    )
    edges = np.asarray(partition["cell_edges_m"], dtype=np.float64)
    seed_cell = q8_cell_index(seed_m[0], edges)
    components: dict[str, Any] = {}
    for name in ("low", "primary", "high"):
        tau = float(tau_values[name])

        def member(cell_index: int, tau: float = tau) -> bool:
            diameter = q8_cell_midpoint(cell_index, edges)
            return bool(cache.get(diameter, seed_m[1], seed_m[2])[1] <= tau**2)

        component = expand_connected_cell_component(
            member,
            cell_count=len(edges) - 1,
            seed_cell=seed_cell,
            max_cells_per_direction=int(design["max_adjacent_cells_per_direction"]),
        )
        components[name] = component
    primary = components["primary"]
    if not primary["seed_member"]:
        return {
            "seed_m": np.asarray(seed_m, dtype=np.float64),
            "seed_cell_index": seed_cell,
            "partition_breakpoint_count": len(edges) - 2,
            "partition_edges_sha256": sha256_array_bytes(edges),
            "components": components,
            "valid": False,
        }
    left = int(primary["left_cell_index"])
    right = int(primary["right_cell_index"])
    left_out = int(primary["left_outside_cell_index"])
    right_out = int(primary["right_outside_cell_index"])
    cap_hit = bool(primary["left_cap_hit"] or primary["right_cap_hit"])
    if left_out < 0 or right_out < 0:
        cap_hit = True
    lower = float(edges[left])
    upper = float(edges[right + 1])
    inside_left = q8_cell_midpoint(left, edges)
    inside_right = q8_cell_midpoint(right, edges)
    outside_left = q8_cell_midpoint(left_out, edges) if left_out >= 0 else np.nan
    outside_right = q8_cell_midpoint(right_out, edges) if right_out >= 0 else np.nan
    bisection: dict[str, Any] = {}
    outside_ratio = np.zeros(2, dtype=np.float64)
    if not cap_hit:
        def objective(diameter: float) -> tuple[float, int]:
            _, loss, cache_index = cache.get(
                diameter, seed_m[1], seed_m[2]
            )
            return loss, cache_index

        bisection = {
            "lower": fixed_membership_bisection(
                objective,
                inside_m=inside_left,
                outside_m=outside_left,
                tau_relative_l2=float(tau_values["primary"]),
                iterations=int(design["boundary_bisection_iterations"]),
            ),
            "upper": fixed_membership_bisection(
                objective,
                inside_m=inside_right,
                outside_m=outside_right,
                tau_relative_l2=float(tau_values["primary"]),
                iterations=int(design["boundary_bisection_iterations"]),
            ),
        }
        outside_ratio = np.sqrt(
            np.asarray(
                [
                    cache.get(outside_left, seed_m[1], seed_m[2])[1],
                    cache.get(outside_right, seed_m[1], seed_m[2])[1],
                ],
                dtype=np.float64,
            )
        ) / float(tau_values["primary"])
    signatures = {
        name: _fiber_component_signature(component)
        for name, component in components.items()
        if component["seed_member"]
    }
    threshold_agreement = bool(
        len(signatures) == 3 and len(set(signatures.values())) == 1
    )
    max_bracket = max(
        (float(item["bracket_width_m"]) for item in bisection.values()),
        default=np.inf,
    )
    tau_squared = float(tau_values["primary"]) ** 2
    lower_closed = bool(
        cache.get(lower, seed_m[1], seed_m[2])[1] <= tau_squared
    )
    upper_closed = bool(
        cache.get(upper, seed_m[1], seed_m[2])[1] <= tau_squared
    )
    return {
        "seed_m": np.asarray(seed_m, dtype=np.float64),
        "seed_cell_index": seed_cell,
        "partition_breakpoint_count": len(edges) - 2,
        "partition_edges_sha256": sha256_array_bytes(edges),
        "components": components,
        "reported_interval": {
            "lower_m": lower,
            "upper_m": upper,
            "width_m": upper - lower,
            "lower_closed": lower_closed,
            "upper_closed": upper_closed,
        },
        "outside_d_waist_m": np.asarray([outside_left, outside_right]),
        "outside_to_tau_ratio": outside_ratio,
        "bisection": bisection,
        "threshold_component_agreement": threshold_agreement,
        "cell_expansion_cap_hit": cap_hit,
        "maximum_bisection_width_m": max_bracket,
        "valid": bool(
            not cap_hit
            and threshold_agreement
            and max_bracket <= float(design["boundary_resolution_m_max"])
        ),
    }


def _local_svd(
    config: Mapping[str, Any], cache: ProbeCache
) -> dict[str, Any]:
    nominal = np.asarray([2.0e-5, 3.0e-5, 5.0e-5], dtype=np.float64)
    offsets_cfg = config["fit"]["local_diagnostic"]["offsets_m"]
    offsets = np.asarray(
        [
            offsets_cfg["d_waist_m"],
            offsets_cfg["d_surface_m"],
            offsets_cfg["z_waist_m"],
        ],
        dtype=np.float64,
    )
    columns: list[NDArray[np.float64]] = []
    raw_norm: list[float] = []
    for parameter_index, offset in enumerate(offsets):
        minus = nominal.copy()
        plus = nominal.copy()
        minus[parameter_index] -= offset
        plus[parameter_index] += offset
        field_minus = cache.get(*minus)[0]
        field_plus = cache.get(*plus)[0]
        derivative = (field_plus - field_minus) / (2.0 * offset)
        real_column = np.concatenate([derivative.real.ravel(), derivative.imag.ravel()])
        norm = float(np.linalg.norm(real_column))
        raw_norm.append(norm)
        columns.append(real_column / norm if norm > 0.0 else real_column)
    matrix = np.column_stack(columns)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    return {
        "parameter_order": ["d_waist_m", "d_surface_m", "z_waist_m"],
        "offsets_m": offsets,
        "raw_column_norm": np.asarray(raw_norm, dtype=np.float64),
        "normalized_singular_values": singular_values,
        "right_singular_vectors": vh,
        "condition_number_diagnostic": float(
            singular_values[0]
            / max(singular_values[-1], np.finfo(np.float64).tiny)
        ),
        "enters_status_gate": False,
    }


def run_exp055_preflight(
    config: Mapping[str, Any],
    target: NDArray[np.complex128],
    generator: CandidateGenerator,
) -> dict[str, Any]:
    """Run the frozen correctness-only seven-point preflight."""

    validate_exp055_config(config)
    cache = ProbeCache(generator, np.asarray(target, dtype=np.complex128))
    points = np.asarray(config["preflight"]["points"], dtype=np.float64)
    losses = np.empty(len(points), dtype=np.float64)
    indices = np.empty(len(points), dtype=np.int64)
    for index, point in enumerate(points):
        _, losses[index], indices[index] = cache.get(*point)
    repeat = np.asarray(generator(*points[0]), dtype=np.complex128)
    replay_relative = float(np.sqrt(losses[0]))
    repeat_relative = float(
        np.sqrt(raw_complex_probe_loss(repeat, cache.fields[indices[0]]))
    )
    passed = bool(
        replay_relative
        <= float(config["thresholds"]["source_replay_relative_l2_max"])
        and repeat_relative
        <= float(
            config["thresholds"]["deterministic_repeat_relative_l2_max"]
        )
        and np.all(np.isfinite(losses))
    )
    return {
        "mode": "preflight",
        "preflight_status": "PreflightPassed" if passed else "PreflightFailed",
        "scientific_status": "NotEvaluated",
        "interpretation": "correctness_only_preflight",
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "points_m": points,
        "loss": losses,
        "raw_relative_l2": np.sqrt(losses),
        "cache_index": indices,
        "replay_relative_l2": replay_relative,
        "deterministic_repeat_relative_l2": repeat_relative,
        "gates": {"correctness_preflight_pass": passed},
        "cache": cache.artifact_payload(),
        "P_B_best_raw": cache.fields[indices[0]],
        "residual_field_raw": cache.fields[indices[0]] - target,
    }


def run_exp055_formal(
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    target: NDArray[np.complex128],
    generator: CandidateGenerator,
) -> dict[str, Any]:
    """Execute the frozen joint-grid/profile/equivalence-set formal method."""

    validate_exp055_config(config)
    target = np.asarray(target, dtype=np.complex128)
    cache = ProbeCache(generator, target)
    fit = config["fit"]
    thresholds = config["thresholds"]
    coarse_axes = _grid_axes(fit, "coarse_grid")
    local_axes = _grid_axes(fit, "local_grid")
    coarse_loss, coarse_cache_index = _evaluate_grid(cache, coarse_axes)
    local_loss, local_cache_index = _evaluate_grid(cache, local_axes)

    nominal = (2.0e-5, 3.0e-5, 5.0e-5)
    nominal_field, nominal_loss, nominal_cache_index = cache.get(*nominal)
    repeated = np.asarray(generator(*nominal), dtype=np.complex128)
    replay_relative = float(np.sqrt(nominal_loss))
    repeat_relative = float(np.sqrt(raw_complex_probe_loss(repeated, nominal_field)))

    masks = {
        name: local_loss <= float(tau) ** 2
        for name, tau in fit["equivalence_relative_l2"].items()
    }
    component_sets = {
        name: connected_components_3d(mask) for name, mask in masks.items()
    }
    component_agreement = bool(
        np.array_equal(masks["low"], masks["primary"])
        and np.array_equal(masks["primary"], masks["high"])
    )
    primary_components = component_sets["primary"]
    best_local_index = tuple(
        int(value)
        for value in np.unravel_index(np.argmin(local_loss), local_loss.shape)
    )
    primary_label = int(primary_components["labels"][best_local_index])
    primary_members = (
        primary_components["components"][primary_label]
        if primary_label >= 0
        else np.empty((0, 3), dtype=np.int64)
    )

    local_qualifying_parameters = {
        tuple(float(local_axes[dim][index[dim]]) for dim in range(3))
        for index in np.argwhere(masks["primary"])
    }
    coarse_mask = coarse_loss <= float(fit["equivalence_relative_l2"]["primary"]) ** 2
    external_coarse_mask = coarse_mask.copy()
    for index in np.argwhere(coarse_mask):
        key = tuple(float(coarse_axes[dim][index[dim]]) for dim in range(3))
        if key in local_qualifying_parameters:
            external_coarse_mask[tuple(index)] = False
    external_coarse = connected_components_3d(external_coarse_mask)
    global_component_count = int(primary_components["count"] + external_coarse["count"])

    branches = _multistart_paths(
        local_loss,
        fit["multistart_paths"]["start_indices"],
        int(fit["multistart_paths"]["maximum_moves_per_start"]),
    )
    terminal_labels: list[int] = []
    for branch in branches.values():
        terminal = tuple(int(value) for value in branch["terminal_index"])
        label = int(primary_components["labels"][terminal])
        branch["terminal_component_label"] = label
        branch["terminal_qualifies"] = bool(label >= 0)
        terminal_labels.append(label)
    multistart_pass = bool(
        primary_label >= 0
        and all(label == primary_label for label in terminal_labels)
    )

    nuisance_pairs = sorted(
        {
            (float(local_axes[1][index[1]]), float(local_axes[2][index[2]]))
            for index in primary_members
        }
    )
    fibers: dict[str, Any] = {}
    for fiber_index, (surface, z_waist) in enumerate(nuisance_pairs):
        member_rows = [
            index
            for index in primary_members
            if float(local_axes[1][index[1]]) == surface
            and float(local_axes[2][index[2]]) == z_waist
        ]
        seed_index = min(member_rows, key=lambda idx: float(local_loss[tuple(idx)]))
        seed = (float(local_axes[0][seed_index[0]]), surface, z_waist)
        fibers[f"fiber_{fiber_index:03d}"] = _q8_fiber(
            config, source_config, cache, seed_m=seed
        )
    valid_fibers = [fiber for fiber in fibers.values() if fiber.get("valid")]
    interval_pairs = sorted(
        (
            float(fiber["reported_interval"]["lower_m"]),
            float(fiber["reported_interval"]["upper_m"]),
        )
        for fiber in valid_fibers
    )
    projection_connected = bool(interval_pairs)
    if interval_pairs:
        running_upper = interval_pairs[0][1]
        for lower, upper in interval_pairs[1:]:
            if lower > running_upper:
                projection_connected = False
            running_upper = max(running_upper, upper)
        projected_lower = min(item[0] for item in interval_pairs)
        projected_upper = max(item[1] for item in interval_pairs)
    else:
        projected_lower = 0.0
        projected_upper = 0.0
    projected_width = projected_upper - projected_lower
    fiber_numerical_pass = bool(
        fibers
        and len(valid_fibers) == len(fibers)
        and all(
            np.nanmin(fiber["outside_to_tau_ratio"])
            >= float(thresholds["outside_to_tau_ratio_min"])
            for fiber in valid_fibers
        )
    )

    boundary_touch = bool(
        any(
            np.any(member == 0) or np.any(member == np.asarray(local_loss.shape) - 1)
            for member in primary_members
        )
    )
    bound_margin = float(
        min(projected_lower - 1.6e-5, 2.4e-5 - projected_upper)
        if np.isfinite(projected_width)
        else -np.inf
    )
    profile_loss = np.min(local_loss, axis=(1, 2))
    profile_argflat = np.argmin(local_loss.reshape(len(local_axes[0]), -1), axis=1)
    profile_nuisance_index = np.column_stack(
        np.unravel_index(profile_argflat, local_loss.shape[1:])
    ).astype(np.int64)
    diagnostic = _local_svd(config, cache)

    replay_pass = bool(
        replay_relative <= float(thresholds["source_replay_relative_l2_max"])
    )
    repeat_pass = bool(
        repeat_relative <= float(thresholds["deterministic_repeat_relative_l2_max"])
    )
    numerical_pass = bool(
        replay_pass
        and repeat_pass
        and component_agreement
        and fiber_numerical_pass
        and np.all(np.isfinite(coarse_loss))
        and np.all(np.isfinite(local_loss))
    )
    uniqueness_pass = bool(
        global_component_count <= int(thresholds["maximum_equivalence_component_count"])
    )
    profile_pass = bool(
        projection_connected
        and not boundary_touch
        and 0.0 < projected_width <= float(thresholds["projected_d_waist_width_m_max"])
        and bound_margin >= float(thresholds["fitting_bound_margin_m"])
    )
    if not numerical_pass:
        status = "Inconclusive"
        interpretation = "nuisance_numerical_control_not_closed"
    elif not uniqueness_pass:
        status = "Failed"
        interpretation = "nuisance_equivalence_nonunique"
    elif not multistart_pass:
        status = "Failed"
        interpretation = "multistart_search_path_inconsistent"
    elif not profile_pass:
        status = "Failed"
        interpretation = "waist_profile_not_stably_identifiable"
    else:
        status = "Passed"
        interpretation = "minimal_geometry_nuisance_identifiability_passed"

    all_losses = np.asarray(cache.losses, dtype=np.float64)
    best_cache_index = int(np.argmin(all_losses))
    best_field = cache.fields[best_cache_index]
    best_parameters = np.asarray(
        cache.parameters_m[best_cache_index], dtype=np.float64
    )
    truth_d = float(
        config["parameters"]["target"][
            "truth_m_simulation_evaluation_only"
        ]
    )
    truth_distance = float(
        0.0
        if projected_lower <= truth_d < projected_upper
        else min(abs(truth_d - projected_lower), abs(truth_d - projected_upper))
    )
    return {
        "mode": "formal",
        "experiment_status": status,
        "interpretation": interpretation,
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "source_replay": {
            "raw_relative_l2": replay_relative,
            "deterministic_repeat_relative_l2": repeat_relative,
            "nominal_cache_index": nominal_cache_index,
        },
        "coarse_grid": {
            "d_waist_m": coarse_axes[0],
            "d_surface_m": coarse_axes[1],
            "z_waist_m": coarse_axes[2],
            "loss": coarse_loss,
            "cache_index": coarse_cache_index,
            "primary_qualifies": coarse_mask,
            "external_qualifies": external_coarse_mask,
        },
        "local_grid": {
            "d_waist_m": local_axes[0],
            "d_surface_m": local_axes[1],
            "z_waist_m": local_axes[2],
            "loss": local_loss,
            "cache_index": local_cache_index,
            "low_qualifies": masks["low"],
            "primary_qualifies": masks["primary"],
            "high_qualifies": masks["high"],
            "primary_component_labels": primary_components["labels"],
        },
        "profile": {
            "d_waist_m": local_axes[0],
            "minimum_loss": profile_loss,
            "argmin_nuisance_index": profile_nuisance_index,
        },
        "equivalence": {
            "local_component_count": primary_components["count"],
            "external_coarse_component_count": external_coarse["count"],
            "global_screened_component_count": global_component_count,
            "primary_component_label": primary_label,
            "primary_component_member_index": primary_members,
            "threshold_component_agreement": component_agreement,
            "touches_local_grid_boundary": boundary_touch,
            "projected_d_waist_interval": {
                "available": bool(interval_pairs),
                "lower_m": projected_lower,
                "upper_m": projected_upper,
                "lower_closed": True,
                "upper_closed": False,
                "width_m": projected_width,
                "connected": projection_connected,
                "fitting_bound_margin_m": bound_margin,
            },
            "q8_fibers": fibers,
        },
        "multistart": {
            "branches": branches,
            "all_terminal_common_component": multistart_pass,
        },
        "local_diagnostic": diagnostic,
        "best_candidate": {
            "parameters_m": best_parameters,
            "loss": float(all_losses[best_cache_index]),
            "cache_index": best_cache_index,
            "field_sha256": cache.field_sha256[best_cache_index],
        },
        "P_B_best_raw": best_field,
        "residual_field_raw": best_field - target,
        "simulation_evaluation_only": {
            "true_d_waist_m": truth_d,
            "truth_to_projected_interval_distance_m": truth_distance,
            "used_in_method_threshold_stop_or_status": False,
        },
        "gates": {
            "source_replay_pass": replay_pass,
            "deterministic_repeat_pass": repeat_pass,
            "threshold_component_agreement_pass": component_agreement,
            "q8_fiber_numerical_controls_pass": fiber_numerical_pass,
            "numerical_controls_pass": numerical_pass,
            "screened_component_uniqueness_pass": uniqueness_pass,
            "multistart_common_component_pass": multistart_pass,
            "profiled_waist_interval_pass": profile_pass,
        },
        "cache": cache.artifact_payload(),
    }
