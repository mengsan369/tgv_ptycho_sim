"""Non-smooth plateau-interval estimator for the fixed-q8 exp051 operator."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.inverse.exp051 import (
    sha256_array_bytes,
    sha256_file,
    validate_exp051_config,
)
from tgv_ptycho.inverse.exp051_local_control import (
    make_air_fraction_stack,
    q8_waist_breakpoint_map,
)
from tgv_ptycho.inverse.waist_fit import (
    equal_budget_bounded_pattern_search,
    raw_complex_probe_loss,
    replay_error_metrics,
)
from tgv_ptycho.objects.tgv3d import make_tgv_air_fraction_slice

ComplexGenerator = Callable[[float], NDArray[np.complex128]]


@dataclass(frozen=True)
class PriorQ8LocalControlArtifact:
    """Validated identity of the q8 differentiability-attribution run."""

    run_dir: Path
    file_sha256: dict[str, str]
    metrics: dict[str, Any]
    run_state: dict[str, Any]


@dataclass
class _ProbeCache:
    generator: ComplexGenerator
    target: NDArray[np.complex128]
    bounds_m: tuple[float, float]
    values: dict[float, int] = field(default_factory=dict)
    diameters_m: list[float] = field(default_factory=list)
    probes: list[NDArray[np.complex128]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def get(self, diameter_m: float) -> tuple[NDArray[np.complex128], float, int]:
        """Return one cached complex128 candidate and its raw primary loss."""

        key = float(np.float64(diameter_m))
        lower, upper = self.bounds_m
        if not np.isfinite(key) or not lower <= key <= upper:
            raise ValueError("candidate D_waist lies outside registered bounds.")
        index = self.values.get(key)
        if index is not None:
            return self.probes[index], self.losses[index], index
        probe = np.asarray(self.generator(key))
        if probe.shape != self.target.shape or probe.dtype != np.complex128:
            raise ValueError(
                "candidate generator must return target-shaped complex128."
            )
        if not np.all(np.isfinite(probe)):
            raise ValueError("candidate generator returned non-finite values.")
        loss = raw_complex_probe_loss(probe, self.target)
        index = len(self.probes)
        self.values[key] = index
        self.diameters_m.append(key)
        self.probes.append(probe.copy())
        self.losses.append(float(loss))
        return self.probes[index], float(loss), index


def validate_exp051_plateau_interval_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen section-26 q8 plateau-interval contract."""

    validate_exp051_config(config)
    if config["experiment"].get("role") != (
        "3d_scalar_multislice_true_probe_q8_plateau_interval_fit"
    ):
        raise ValueError("Unexpected exp051 plateau-interval role.")
    prior = config.get("prior_q8_local_control", {})
    if (
        prior.get("required_experiment_status") != "Inconclusive"
        or prior.get("required_diagnostic_status") != "DiagnosticPassed"
        or prior.get("required_interpretation")
        != (
            "q8_midpoint_piecewise_constant_at_truth__"
            "chord_control_locally_converged"
        )
    ):
        raise ValueError("Prior q8 local-control identity lock changed.")
    design = config.get("plateau_interval", {})
    equivalence = design.get("equivalence_threshold", {})
    thresholds = design.get("thresholds", {})
    expected_thresholds = {
        "source_replay_relative_l2_max": 1.0e-14,
        "deterministic_repeat_relative_l2_max": 1.0e-14,
        "outside_to_tau_ratio_min": 1.0e6,
        "boundary_resolution_m_max": 1.0e-12,
        "interval_width_m_max": 1.25e-7,
        "endpoint_absolute_error_m_max": 1.25e-7,
        "endpoint_relative_error_max": 0.00625,
        "boundary_margin_m": 1.25e-7,
    }
    if (
        design.get("q8_interface_method") != "subpixel_midpoint_count"
        or int(design.get("q8_interface_factor", 0)) != 8
        or float(equivalence.get("floor_multiplier", 0.0)) != 100.0
        or float(equivalence.get("minimum_relative_l2", 0.0)) != 1.0e-12
        or float(equivalence.get("low_factor", 0.0)) != 0.1
        or float(equivalence.get("high_factor", 0.0)) != 10.0
        or int(design.get("max_adjacent_cells_per_direction", 0)) != 256
        or int(design.get("boundary_bisection_iterations", 0)) != 24
        or set(thresholds) != set(expected_thresholds)
        or any(
            float(thresholds[key]) != value
            for key, value in expected_thresholds.items()
        )
    ):
        raise ValueError("exp051 plateau-interval frozen design changed.")
    output = config.get("output", {})
    if (
        output.get("hdf5_filename") != "exp051_q8_plateau_interval_fit.h5"
        or len(output.get("figure_filenames", [])) != 3
    ):
        raise ValueError("exp051 plateau-interval output contract changed.")


def load_prior_q8_local_control(
    config: Mapping[str, Any], project_root: Path
) -> PriorQ8LocalControlArtifact:
    """Load and hash-check the registered q8 attribution artifact."""

    validate_exp051_plateau_interval_config(config)
    prior = config["prior_q8_local_control"]
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
            raise RuntimeError(f"Missing prior q8 local-control artifact: {path}")
        actual[key] = sha256_file(path)
        if actual[key] != str(prior["expected_sha256"][key]).upper():
            raise RuntimeError(f"Prior q8 local-control {key} hash mismatch.")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    if (
        state.get("status") != "complete"
        or state.get("artifacts_validated") is not True
        or metrics.get("experiment_status")
        != prior["required_experiment_status"]
        or metrics.get("diagnostic_status") != prior["required_diagnostic_status"]
        or metrics.get("interpretation") != prior["required_interpretation"]
        or metrics.get("reference_validated") is not False
        or metrics.get("full_tgv_reference_authorized") is not False
        or metrics.get("p_b_rec_used_as_primary_input") is not False
    ):
        raise RuntimeError("Prior q8 local-control state/provenance differs.")
    return PriorQ8LocalControlArtifact(run_dir, actual, metrics, state)


def build_q8_cell_partition(
    source_config: Mapping[str, Any],
    *,
    bounds_m: tuple[float, float],
    reference_diameter_m: float,
    q: int = 8,
) -> dict[str, NDArray[np.generic]]:
    """Return strict internal breakpoints and sentinel-bounded q8 cells."""

    lower, upper = (float(value) for value in bounds_m)
    mapped = q8_waist_breakpoint_map(
        source_config,
        bounds_m=(lower, upper),
        true_waist_m=float(reference_diameter_m),
        q=q,
    )
    all_breakpoints = np.asarray(mapped["breakpoints_m"], dtype=np.float64)
    all_multiplicity = np.asarray(mapped["multiplicity"], dtype=np.int64)
    selected = (all_breakpoints > lower) & (all_breakpoints < upper)
    breakpoints = all_breakpoints[selected]
    multiplicity = all_multiplicity[selected]
    edges = np.concatenate(
        [np.asarray([lower]), breakpoints, np.asarray([upper])]
    ).astype(np.float64)
    if (
        len(breakpoints) == 0
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise RuntimeError("q8 cell partition is not finite and strictly ordered.")
    return {
        "breakpoints_m": breakpoints,
        "multiplicity": multiplicity,
        "cell_edges_m": edges,
    }


def q8_cell_index(diameter_m: float, cell_edges_m: NDArray[np.float64]) -> int:
    """Map a float64 diameter to its source-convention half-open q8 cell."""

    edges = np.asarray(cell_edges_m, dtype=np.float64)
    value = float(np.float64(diameter_m))
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("cell_edges_m must be a strictly increasing 1D array.")
    if not np.isfinite(value) or not edges[0] <= value <= edges[-1]:
        raise ValueError("diameter lies outside q8 cell partition.")
    if value == edges[-1]:
        return len(edges) - 2
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def q8_cell_midpoint(cell_index: int, cell_edges_m: NDArray[np.float64]) -> float:
    """Return a representable point in one source-convention half-open cell."""

    edges = np.asarray(cell_edges_m, dtype=np.float64)
    if not 0 <= cell_index < len(edges) - 1:
        raise ValueError("cell_index lies outside q8 partition.")
    left = float(edges[cell_index])
    right = float(edges[cell_index + 1])
    midpoint = float(left + 0.5 * (right - left))
    if not left < midpoint < right:
        midpoint = float(np.nextafter(left, right))
    if not left < midpoint < right:
        # Adjacent float64 breakpoints have no strictly interior value.  The
        # left endpoint is the sole representable member of [left, right).
        midpoint = left
    return midpoint


def expand_connected_cell_component(
    member_for_cell: Callable[[int], bool],
    *,
    cell_count: int,
    seed_cell: int,
    max_cells_per_direction: int,
) -> dict[str, Any]:
    """Expand the maximal sampled connected member component around a seed."""

    if cell_count <= 0 or not 0 <= seed_cell < cell_count:
        raise ValueError("seed_cell must lie inside a nonempty partition.")
    if max_cells_per_direction <= 0:
        raise ValueError("max_cells_per_direction must be positive.")
    queried_cells: list[int] = []
    queried_membership: list[bool] = []

    def query(index: int) -> bool:
        member = bool(member_for_cell(index))
        queried_cells.append(index)
        queried_membership.append(member)
        return member

    if not query(seed_cell):
        return {
            "seed_member": False,
            "left_cell_index": -1,
            "right_cell_index": -1,
            "left_outside_cell_index": -1,
            "right_outside_cell_index": -1,
            "left_cap_hit": False,
            "right_cap_hit": False,
            "queried_cell_index": np.asarray(queried_cells, dtype=np.int64),
            "queried_member": np.asarray(queried_membership, dtype=np.bool_),
        }

    left = seed_cell
    left_outside = -1
    left_cap = False
    checked = 0
    while left > 0 and checked < max_cells_per_direction:
        candidate = left - 1
        checked += 1
        if not query(candidate):
            left_outside = candidate
            break
        left = candidate
    if left > 0 and left_outside < 0:
        left_cap = True

    right = seed_cell
    right_outside = -1
    right_cap = False
    checked = 0
    while right < cell_count - 1 and checked < max_cells_per_direction:
        candidate = right + 1
        checked += 1
        if not query(candidate):
            right_outside = candidate
            break
        right = candidate
    if right < cell_count - 1 and right_outside < 0:
        right_cap = True
    return {
        "seed_member": True,
        "left_cell_index": int(left),
        "right_cell_index": int(right),
        "left_outside_cell_index": int(left_outside),
        "right_outside_cell_index": int(right_outside),
        "left_cap_hit": bool(left_cap),
        "right_cap_hit": bool(right_cap),
        "queried_cell_index": np.asarray(queried_cells, dtype=np.int64),
        "queried_member": np.asarray(queried_membership, dtype=np.bool_),
    }


def fixed_membership_bisection(
    objective: Callable[[float], tuple[float, int]],
    *,
    inside_m: float,
    outside_m: float,
    tau_relative_l2: float,
    iterations: int,
) -> dict[str, Any]:
    """Bracket one discontinuous membership transition at fixed budget."""

    if iterations <= 0 or tau_relative_l2 <= 0.0:
        raise ValueError("bisection iterations and tau must be positive.")
    inside_loss, inside_cache = objective(float(inside_m))
    outside_loss, outside_cache = objective(float(outside_m))
    threshold_loss = float(tau_relative_l2**2)
    if inside_loss > threshold_loss or outside_loss <= threshold_loss:
        raise ValueError("bisection requires one inside and one outside endpoint.")
    inside = float(inside_m)
    outside = float(outside_m)
    evaluated_d: list[float] = []
    evaluated_loss: list[float] = []
    evaluated_member: list[bool] = []
    evaluated_cache: list[int] = []
    for _ in range(iterations):
        midpoint = float(inside + 0.5 * (outside - inside))
        loss, cache_index = objective(midpoint)
        member = bool(loss <= threshold_loss)
        evaluated_d.append(midpoint)
        evaluated_loss.append(float(loss))
        evaluated_member.append(member)
        evaluated_cache.append(int(cache_index))
        if member:
            inside = midpoint
            inside_loss = loss
            inside_cache = cache_index
        else:
            outside = midpoint
            outside_loss = loss
            outside_cache = cache_index
    return {
        "iterations": int(iterations),
        "evaluated_d_waist_m": np.asarray(evaluated_d, dtype=np.float64),
        "evaluated_loss": np.asarray(evaluated_loss, dtype=np.float64),
        "evaluated_member": np.asarray(evaluated_member, dtype=np.bool_),
        "evaluated_cache_index": np.asarray(evaluated_cache, dtype=np.int64),
        "final_inside_m": float(inside),
        "final_inside_loss": float(inside_loss),
        "final_inside_cache_index": int(inside_cache),
        "final_outside_m": float(outside),
        "final_outside_loss": float(outside_loss),
        "final_outside_cache_index": int(outside_cache),
        "bracket_lower_m": float(min(inside, outside)),
        "bracket_upper_m": float(max(inside, outside)),
        "bracket_width_m": float(abs(outside - inside)),
    }


def _component_signature(component: Mapping[str, Any]) -> tuple[int, int, bool, bool]:
    return (
        int(component["left_cell_index"]),
        int(component["right_cell_index"]),
        bool(component["lower_closed"]),
        bool(component["upper_closed"]),
    )


def _build_fine_grid(
    coarse_m: NDArray[np.float64],
    coarse_loss: NDArray[np.float64],
    *,
    bounds_m: tuple[float, float],
    half_width_m: float,
    step_m: float,
) -> NDArray[np.float64]:
    coarse_minimum = float(coarse_m[int(np.argmin(coarse_loss))])
    lower = max(bounds_m[0], coarse_minimum - half_width_m)
    upper = min(bounds_m[1], coarse_minimum + half_width_m)
    count = int(np.rint((upper - lower) / step_m))
    if not np.isclose(
        lower + count * step_m,
        upper,
        rtol=0.0,
        atol=32.0 * np.finfo(np.float64).eps,
    ):
        raise RuntimeError("registered fine interval is not divisible by step.")
    return lower + np.arange(count + 1, dtype=np.float64) * step_m


def _node_count_controls(
    source_config: Mapping[str, Any],
    diameters: Mapping[str, float],
    *,
    q: int,
) -> dict[str, Any]:
    stacks: dict[str, NDArray[np.uint8]] = {}
    hashes: dict[str, str] = {}
    for name, diameter in diameters.items():
        fraction, _, _ = make_air_fraction_stack(
            source_config,
            float(diameter),
            builder=make_tgv_air_fraction_slice,
            resolution=q,
        )
        nodes = np.rint(fraction * q**2).astype(np.uint8)
        stacks[name] = nodes
        hashes[name] = sha256_array_bytes(nodes)
    seed = stacks["seed_midpoint"]
    return {
        "diameter_m": {key: float(value) for key, value in diameters.items()},
        "node_count_stack": stacks,
        "node_count_sha256": hashes,
        "equal_to_seed": {
            key: bool(np.array_equal(value, seed))
            for key, value in stacks.items()
        },
    }


def _endpoint_field_controls(
    evaluate: Callable[[float], tuple[NDArray[np.complex128], float, int]],
    target: NDArray[np.complex128],
    diameters: Mapping[str, float],
    *,
    tau_relative_l2: float,
) -> dict[str, Any]:
    names: list[str] = []
    values: list[float] = []
    losses: list[float] = []
    exact: list[bool] = []
    members: list[bool] = []
    cache_indices: list[int] = []
    threshold_loss = tau_relative_l2**2
    for name, diameter in diameters.items():
        probe, loss, cache_index = evaluate(float(diameter))
        names.append(name)
        values.append(float(diameter))
        losses.append(float(loss))
        exact.append(bool(np.array_equal(probe, target)))
        members.append(bool(loss <= threshold_loss))
        cache_indices.append(int(cache_index))
    return {
        "name": names,
        "diameter_m": np.asarray(values, dtype=np.float64),
        "loss": np.asarray(losses, dtype=np.float64),
        "raw_relative_l2": np.sqrt(np.asarray(losses, dtype=np.float64)),
        "exact_member": np.asarray(exact, dtype=np.bool_),
        "tau_member": np.asarray(members, dtype=np.bool_),
        "cache_index": np.asarray(cache_indices, dtype=np.int64),
    }


def run_q8_plateau_interval_fit(
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    target_probe: NDArray[np.complex128],
    candidate_generator: ComplexGenerator,
) -> dict[str, Any]:
    """Run the preregistered fixed-q8 interval-valued oracle fit."""

    validate_exp051_plateau_interval_config(config)
    target = np.asarray(target_probe)
    if target.ndim != 2 or target.dtype != np.complex128:
        raise ValueError("target_probe must be a 2D complex128 field.")
    if not np.all(np.isfinite(target)) or float(np.linalg.norm(target)) <= 0.0:
        raise ValueError("target_probe must be finite with positive norm.")
    fit = config["fit"]
    design = config["plateau_interval"]
    thresholds = design["thresholds"]
    true_waist = float(fit["true_d_waist_m"])
    bounds = tuple(float(value) for value in fit["bounds_m"])
    q = int(design["q8_interface_factor"])

    replay = np.asarray(candidate_generator(true_waist))
    repeat = np.asarray(candidate_generator(true_waist))
    if replay.shape != target.shape or replay.dtype != np.complex128:
        raise ValueError("replay candidate identity differs from target.")
    replay_metrics = replay_error_metrics(replay, target)
    repeat_relative = float(np.sqrt(raw_complex_probe_loss(repeat, replay)))
    replay_metrics["deterministic_repeat_relative_l2"] = repeat_relative
    source_gate = bool(
        replay_metrics["raw_complex_relative_l2"]
        <= float(thresholds["source_replay_relative_l2_max"])
        and repeat_relative
        <= float(thresholds["deterministic_repeat_relative_l2_max"])
    )
    floor_relative = max(
        replay_metrics["raw_complex_relative_l2"],
        repeat_relative,
        np.finfo(np.float64).eps,
    )
    equivalence = design["equivalence_threshold"]
    tau = max(
        float(equivalence["minimum_relative_l2"]),
        float(equivalence["floor_multiplier"]) * floor_relative,
    )
    tau_values = {
        "tau_low": float(tau * float(equivalence["low_factor"])),
        "tau_primary": float(tau),
        "tau_high": float(tau * float(equivalence["high_factor"])),
    }
    cache = _ProbeCache(candidate_generator, target, bounds)
    cache.values[float(np.float64(true_waist))] = 0
    cache.diameters_m.append(true_waist)
    cache.probes.append(replay.copy())
    cache.losses.append(raw_complex_probe_loss(replay, target))
    if not source_gate:
        return {
            "experiment_status": "Inconclusive",
            "interpretation": "artifact_operator_handoff_not_closed",
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "p_b_rec_used_as_primary_input": False,
            "replay": replay_metrics,
            "tau": tau_values,
            "gates": {"source_replay_and_repeat_pass": False},
            "cache": {
                "D_waist_m": np.asarray(cache.diameters_m),
                "loss": np.asarray(cache.losses),
                "P_B_candidate": np.stack(cache.probes),
            },
        }

    def evaluate_loss(diameter_m: float) -> tuple[float, int]:
        _, loss, cache_index = cache.get(diameter_m)
        return loss, cache_index

    def evaluate_field(
        diameter_m: float,
    ) -> tuple[NDArray[np.complex128], float, int]:
        return cache.get(diameter_m)

    partition = build_q8_cell_partition(
        source_config,
        bounds_m=bounds,
        reference_diameter_m=true_waist,
        q=q,
    )
    edges = np.asarray(partition["cell_edges_m"], dtype=np.float64)
    cell_count = len(edges) - 1
    breakpoint_gate = bool(
        np.all(np.isfinite(edges)) and np.all(np.diff(edges) > 0.0)
    )

    coarse = np.asarray(fit["coarse_grid_m"], dtype=np.float64)
    coarse_loss: list[float] = []
    coarse_cache: list[int] = []
    for diameter in coarse:
        loss, cache_index = evaluate_loss(float(diameter))
        coarse_loss.append(loss)
        coarse_cache.append(cache_index)
    coarse_losses = np.asarray(coarse_loss, dtype=np.float64)
    fine = _build_fine_grid(
        coarse,
        coarse_losses,
        bounds_m=bounds,
        half_width_m=float(fit["fine_half_width_m"]),
        step_m=float(fit["fine_step_m"]),
    )
    fine_loss: list[float] = []
    fine_cache: list[int] = []
    for diameter in fine:
        loss, cache_index = evaluate_loss(float(diameter))
        fine_loss.append(loss)
        fine_cache.append(cache_index)
    fine_losses = np.asarray(fine_loss, dtype=np.float64)

    optimizer = fit["optimizer"]
    branches: dict[str, dict[str, Any]] = {}
    for branch_index, start in enumerate(optimizer["starts_m"]):
        branch = equal_budget_bounded_pattern_search(
            evaluate_loss,
            bounds_m=bounds,
            start_m=float(start),
            initial_step_m=float(optimizer["initial_step_m"]),
            evaluation_budget=int(optimizer["evaluation_budget_per_start"]),
        )
        branch["final_raw_relative_l2"] = float(np.sqrt(branch["final_loss"]))
        branch["final_seed_qualifies"] = bool(
            branch["final_raw_relative_l2"] <= tau
        )
        branch["final_cell_index"] = q8_cell_index(
            float(branch["final_estimate_m"]), edges
        )
        branches[f"start_{branch_index:02d}"] = branch
    equal_budget = bool(
        all(
            int(branch["evaluation_count"])
            == int(optimizer["evaluation_budget_per_start"])
            and branch["stopping_reason"] == "evaluation_budget"
            for branch in branches.values()
        )
    )
    all_seeds_qualify = bool(
        all(branch["final_seed_qualifies"] for branch in branches.values())
    )
    if not all_seeds_qualify:
        cache_cells = np.asarray(
            [q8_cell_index(value, edges) for value in cache.diameters_m],
            dtype=np.int64,
        )
        return {
            "experiment_status": "Failed",
            "interpretation": "interval_search_unstable",
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "p_b_rec_used_as_primary_input": False,
            "replay": replay_metrics,
            "tau": tau_values,
            "partition": partition,
            "profile": {
                "coarse_d_waist_m": coarse,
                "coarse_loss": coarse_losses,
                "coarse_cache_index": np.asarray(coarse_cache, dtype=np.int64),
                "fine_d_waist_m": fine,
                "fine_loss": fine_losses,
                "fine_cache_index": np.asarray(fine_cache, dtype=np.int64),
            },
            "optimizer": {
                "branches": branches,
                "equal_budget": equal_budget,
                "all_final_seeds_qualify": False,
            },
            "gates": {
                "source_replay_and_repeat_pass": source_gate,
                "breakpoint_partition_pass": breakpoint_gate,
                "equal_budget_pass": equal_budget,
                "all_branch_seeds_qualify_pass": False,
            },
            "cache": {
                "D_waist_m": np.asarray(cache.diameters_m),
                "loss": np.asarray(cache.losses),
                "q8_cell_index": cache_cells,
                "P_B_candidate": np.stack(cache.probes),
            },
        }

    max_cells = int(design["max_adjacent_cells_per_direction"])
    interval_by_branch: dict[str, dict[str, Any]] = {}
    for name, branch in branches.items():
        seed_cell = int(branch["final_cell_index"])
        components: dict[str, Any] = {}
        membership_specs: dict[str, float | None] = {
            "exact": None,
            **tau_values,
        }
        for component_name, component_tau in membership_specs.items():
            query_details: dict[int, tuple[float, float, int, bool]] = {}

            def member_for_cell(
                cell_index: int,
                component_tau: float | None = component_tau,
                query_details: dict[
                    int, tuple[float, float, int, bool]
                ] = query_details,
            ) -> bool:
                diameter = q8_cell_midpoint(cell_index, edges)
                probe, loss, cache_index = cache.get(diameter)
                member = (
                    bool(np.array_equal(probe, target))
                    if component_tau is None
                    else bool(loss <= float(component_tau) ** 2)
                )
                query_details[cell_index] = (
                    diameter,
                    float(loss),
                    int(cache_index),
                    member,
                )
                return member

            component = expand_connected_cell_component(
                member_for_cell,
                cell_count=cell_count,
                seed_cell=seed_cell,
                max_cells_per_direction=max_cells,
            )
            if not component["seed_member"]:
                component.update(
                    {
                        "lower_boundary_m": np.nan,
                        "upper_boundary_m": np.nan,
                        "lower_closed": False,
                        "upper_closed": False,
                    }
                )
            else:
                left_cell = int(component["left_cell_index"])
                right_cell = int(component["right_cell_index"])
                lower_boundary = float(edges[left_cell])
                upper_boundary = float(edges[right_cell + 1])
                lower_probe, lower_loss, _ = cache.get(lower_boundary)
                upper_probe, upper_loss, _ = cache.get(upper_boundary)
                if component_tau is None:
                    lower_closed = bool(np.array_equal(lower_probe, target))
                    upper_closed = bool(np.array_equal(upper_probe, target))
                else:
                    lower_closed = bool(lower_loss <= float(component_tau) ** 2)
                    upper_closed = bool(upper_loss <= float(component_tau) ** 2)
                component.update(
                    {
                        "lower_boundary_m": lower_boundary,
                        "upper_boundary_m": upper_boundary,
                        "lower_closed": lower_closed,
                        "upper_closed": upper_closed,
                    }
                )
            ordered_details = [
                query_details[int(index)]
                for index in component["queried_cell_index"]
            ]
            component["queried_d_waist_m"] = np.asarray(
                [item[0] for item in ordered_details], dtype=np.float64
            )
            component["queried_loss"] = np.asarray(
                [item[1] for item in ordered_details], dtype=np.float64
            )
            component["queried_cache_index"] = np.asarray(
                [item[2] for item in ordered_details], dtype=np.int64
            )
            components[component_name] = component
        interval_by_branch[name] = {
            "seed_m": float(branch["final_estimate_m"]),
            "seed_loss": float(branch["final_loss"]),
            "seed_cell_index": seed_cell,
            "components": components,
        }

    first_branch = next(iter(interval_by_branch.values()))
    primary_component = first_branch["components"]["tau_primary"]
    left_cell = int(primary_component["left_cell_index"])
    right_cell = int(primary_component["right_cell_index"])
    common_left = float(primary_component["lower_boundary_m"])
    common_right = float(primary_component["upper_boundary_m"])
    seed_midpoint = q8_cell_midpoint(left_cell, edges)
    left_outside_cell = int(primary_component["left_outside_cell_index"])
    right_outside_cell = int(primary_component["right_outside_cell_index"])
    if left_outside_cell < 0 or right_outside_cell < 0:
        raise RuntimeError("Primary interval lacks adjacent outside cells.")
    left_outside_midpoint = q8_cell_midpoint(left_outside_cell, edges)
    right_outside_midpoint = q8_cell_midpoint(right_outside_cell, edges)
    endpoint_diameters = {
        "seed_midpoint": seed_midpoint,
        "left_outside_midpoint": left_outside_midpoint,
        "right_outside_midpoint": right_outside_midpoint,
        "lower_breakpoint": common_left,
        "lower_nextafter_minus": float(np.nextafter(common_left, -np.inf)),
        "lower_nextafter_plus": float(np.nextafter(common_left, np.inf)),
        "upper_breakpoint": common_right,
        "upper_nextafter_minus": float(np.nextafter(common_right, -np.inf)),
        "upper_nextafter_plus": float(np.nextafter(common_right, np.inf)),
    }
    node_controls = _node_count_controls(
        source_config, endpoint_diameters, q=q
    )
    endpoint_fields = _endpoint_field_controls(
        evaluate_field,
        target,
        endpoint_diameters,
        tau_relative_l2=tau,
    )
    node_equal = node_controls["equal_to_seed"]
    field_names = list(endpoint_fields["name"])
    field_tau = {
        name: bool(endpoint_fields["tau_member"][index])
        for index, name in enumerate(field_names)
    }
    field_exact = {
        name: bool(endpoint_fields["exact_member"][index])
        for index, name in enumerate(field_names)
    }
    geometry_endpoint_pass = bool(
        node_equal["seed_midpoint"]
        and not node_equal["left_outside_midpoint"]
        and not node_equal["right_outside_midpoint"]
        and all(
            field_tau[name] == node_equal[name]
            and field_exact[name] == node_equal[name]
            for name in field_names
        )
        and node_equal["lower_breakpoint"]
        == bool(primary_component["lower_closed"])
        and node_equal["upper_breakpoint"]
        == bool(primary_component["upper_closed"])
    )
    geometry_signature = (
        left_cell,
        right_cell,
        bool(node_equal["lower_breakpoint"]),
        bool(node_equal["upper_breakpoint"]),
    )
    component_signatures = [
        _component_signature(branch["components"][component_name])
        for branch in interval_by_branch.values()
        for component_name in ("exact", "tau_low", "tau_primary", "tau_high")
    ]
    threshold_stability = bool(
        all(signature == geometry_signature for signature in component_signatures)
    )
    interval_agreement = bool(
        all(
            _component_signature(branch["components"]["tau_primary"])
            == _component_signature(primary_component)
            for branch in interval_by_branch.values()
        )
    )
    cap_hit = bool(
        any(
            component[side]
            for branch in interval_by_branch.values()
            for component in branch["components"].values()
            for side in ("left_cap_hit", "right_cap_hit")
        )
    )

    outside_losses = np.asarray(
        [
            evaluate_loss(left_outside_midpoint)[0],
            evaluate_loss(right_outside_midpoint)[0],
        ],
        dtype=np.float64,
    )
    outside_relative = np.sqrt(outside_losses)
    outside_ratio = outside_relative / tau
    outside_separation_pass = bool(
        np.min(outside_ratio)
        >= float(thresholds["outside_to_tau_ratio_min"])
    )

    bisection: dict[str, Any] = {}
    bisection_passes: list[bool] = []
    iterations = int(design["boundary_bisection_iterations"])
    for name, branch_interval in interval_by_branch.items():
        component = branch_interval["components"]["tau_primary"]
        inside_left = q8_cell_midpoint(int(component["left_cell_index"]), edges)
        inside_right = q8_cell_midpoint(int(component["right_cell_index"]), edges)
        outside_left = q8_cell_midpoint(
            int(component["left_outside_cell_index"]), edges
        )
        outside_right = q8_cell_midpoint(
            int(component["right_outside_cell_index"]), edges
        )
        lower_control = fixed_membership_bisection(
            evaluate_loss,
            inside_m=inside_left,
            outside_m=outside_left,
            tau_relative_l2=tau,
            iterations=iterations,
        )
        upper_control = fixed_membership_bisection(
            evaluate_loss,
            inside_m=inside_right,
            outside_m=outside_right,
            tau_relative_l2=tau,
            iterations=iterations,
        )
        for control, analytic in (
            (lower_control, float(component["lower_boundary_m"])),
            (upper_control, float(component["upper_boundary_m"])),
        ):
            control["analytic_breakpoint_m"] = analytic
            control["analytic_breakpoint_in_bracket"] = bool(
                control["bracket_lower_m"]
                <= analytic
                <= control["bracket_upper_m"]
            )
            control["pass"] = bool(
                control["iterations"] == iterations
                and control["bracket_width_m"]
                <= float(thresholds["boundary_resolution_m_max"])
                and control["analytic_breakpoint_in_bracket"]
            )
            bisection_passes.append(bool(control["pass"]))
        bisection[name] = {"lower": lower_control, "upper": upper_control}
    bisection_pass = bool(all(bisection_passes))

    repeat_diameters = {
        "seed_midpoint": seed_midpoint,
        "left_outside_midpoint": left_outside_midpoint,
        "right_outside_midpoint": right_outside_midpoint,
    }
    repeat_controls: dict[str, float] = {}
    for name, diameter in repeat_diameters.items():
        reference_probe, _, _ = cache.get(diameter)
        repeated_probe = np.asarray(candidate_generator(diameter))
        repeat_controls[name] = float(
            np.sqrt(raw_complex_probe_loss(repeated_probe, reference_probe))
        )
    repeat_max = max(repeat_controls.values(), default=0.0)
    deterministic_pass = bool(
        max(repeat_relative, repeat_max)
        <= float(thresholds["deterministic_repeat_relative_l2_max"])
    )

    support_diameters = [coarse, fine]
    support_losses = [coarse_losses, fine_losses]
    for branch in branches.values():
        support_diameters.append(
            np.asarray(branch["evaluated_d_waist_m"], dtype=np.float64)
        )
        support_losses.append(np.asarray(branch["evaluated_loss"], dtype=np.float64))
    screen_d = np.concatenate(support_diameters)
    screen_loss = np.concatenate(support_losses)
    screen_qualifies = screen_loss <= tau**2
    screen_cells = np.asarray(
        [q8_cell_index(value, edges) for value in screen_d], dtype=np.int64
    )
    screened_alias = bool(
        np.any(
            screen_qualifies
            & ((screen_cells < left_cell) | (screen_cells > right_cell))
        )
    )
    screen_pass = not screened_alias

    interval_width = common_right - common_left
    lower_closed = bool(primary_component["lower_closed"])
    upper_closed = bool(primary_component["upper_closed"])
    truth_inside = bool(
        (true_waist > common_left or (lower_closed and true_waist == common_left))
        and (
            true_waist < common_right
            or (upper_closed and true_waist == common_right)
        )
    )
    endpoint_errors = np.asarray(
        [abs(common_left - true_waist), abs(common_right - true_waist)],
        dtype=np.float64,
    )
    worst_absolute_error = float(np.max(endpoint_errors))
    worst_relative_error = worst_absolute_error / true_waist
    boundary_margin = min(common_left - bounds[0], bounds[1] - common_right)
    accuracy_pass = bool(
        0.0 < interval_width <= float(thresholds["interval_width_m_max"])
        and truth_inside
        and worst_absolute_error
        <= float(thresholds["endpoint_absolute_error_m_max"])
        and worst_relative_error
        <= float(thresholds["endpoint_relative_error_max"])
        and boundary_margin >= float(thresholds["boundary_margin_m"])
    )
    numerical_controls_pass = bool(
        breakpoint_gate
        and deterministic_pass
        and threshold_stability
        and outside_separation_pass
        and bisection_pass
        and geometry_endpoint_pass
        and not cap_hit
    )
    search_pass = bool(
        equal_budget and all_seeds_qualify and interval_agreement
    )
    if not numerical_controls_pass:
        status = "Inconclusive"
        interpretation = "q8_plateau_interval_numerical_control_not_closed"
    elif not search_pass:
        status = "Failed"
        interpretation = "interval_search_unstable"
    elif not screen_pass:
        status = "Failed"
        interpretation = "screened_equivalence_nonunique"
    elif not accuracy_pass:
        status = "Failed"
        interpretation = "interval_accuracy_failed"
    else:
        status = "Passed"
        interpretation = "q8_plateau_interval_single_parameter_oracle_fit_passed"

    midpoint = float(common_left + 0.5 * interval_width)
    best_probe, best_loss, best_cache_index = cache.get(midpoint)
    residual = np.asarray(best_probe - target, dtype=np.complex128)
    cache_cells = np.asarray(
        [q8_cell_index(value, edges) for value in cache.diameters_m],
        dtype=np.int64,
    )
    return {
        "experiment_status": status,
        "interpretation": interpretation,
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "replay": replay_metrics,
        "tau": {
            **tau_values,
            "replay_floor_relative_l2": floor_relative,
            "primary_loss_threshold": tau**2,
        },
        "partition": partition,
        "profile": {
            "coarse_d_waist_m": coarse,
            "coarse_loss": coarse_losses,
            "coarse_cache_index": np.asarray(coarse_cache, dtype=np.int64),
            "fine_d_waist_m": fine,
            "fine_loss": fine_losses,
            "fine_cache_index": np.asarray(fine_cache, dtype=np.int64),
        },
        "optimizer": {
            "algorithm": "fixed_budget_bounded_pattern_search",
            "branches": branches,
            "equal_budget": equal_budget,
            "all_final_seeds_qualify": all_seeds_qualify,
            "interval_agreement": interval_agreement,
        },
        "interval_by_branch": interval_by_branch,
        "reported_interval": {
            "lower_m": common_left,
            "upper_m": common_right,
            "lower_closed": lower_closed,
            "upper_closed": upper_closed,
            "width_m": interval_width,
            "midpoint_m": midpoint,
            "midpoint_loss": float(best_loss),
            "midpoint_cache_index": int(best_cache_index),
            "left_cell_index": left_cell,
            "right_cell_index": right_cell,
            "truth_inside_simulation_evaluation_only": truth_inside,
            "truth_to_interval_distance_m": 0.0 if truth_inside else float(
                min(abs(true_waist - common_left), abs(true_waist - common_right))
            ),
            "worst_endpoint_absolute_error_m": worst_absolute_error,
            "worst_endpoint_relative_error": worst_relative_error,
            "fitting_bound_margin_m": boundary_margin,
        },
        "outside_control": {
            "diameter_m": np.asarray(
                [left_outside_midpoint, right_outside_midpoint], dtype=np.float64
            ),
            "loss": outside_losses,
            "raw_relative_l2": outside_relative,
            "to_tau_ratio": outside_ratio,
        },
        "geometry_control": node_controls,
        "endpoint_field_control": endpoint_fields,
        "bisection": bisection,
        "deterministic_repeat": {
            "by_point_relative_l2": repeat_controls,
            "maximum_relative_l2": repeat_max,
        },
        "profile_screen": {
            "diameter_m": screen_d,
            "loss": screen_loss,
            "q8_cell_index": screen_cells,
            "qualifies": screen_qualifies,
            "qualifying_count": int(np.count_nonzero(screen_qualifies)),
            "external_qualifying_component_observed": screened_alias,
        },
        "P_B_interval_midpoint_raw": np.asarray(best_probe, dtype=np.complex128),
        "residual_field_raw": residual,
        "gates": {
            "source_replay_and_repeat_pass": source_gate,
            "breakpoint_partition_pass": breakpoint_gate,
            "deterministic_repeat_pass": deterministic_pass,
            "threshold_stability_pass": threshold_stability,
            "outside_separation_pass": outside_separation_pass,
            "boundary_bisection_pass": bisection_pass,
            "geometry_endpoint_convention_pass": geometry_endpoint_pass,
            "cell_expansion_budget_pass": not cap_hit,
            "numerical_controls_pass": numerical_controls_pass,
            "equal_budget_pass": equal_budget,
            "all_branch_seeds_qualify_pass": all_seeds_qualify,
            "multi_start_interval_agreement_pass": interval_agreement,
            "screened_profile_uniqueness_pass": screen_pass,
            "interval_accuracy_pass": accuracy_pass,
        },
        "cache": {
            "D_waist_m": np.asarray(cache.diameters_m, dtype=np.float64),
            "loss": np.asarray(cache.losses, dtype=np.float64),
            "q8_cell_index": cache_cells,
            "P_B_candidate": np.stack(cache.probes).astype(
                np.complex128, copy=False
            ),
        },
    }
