"""Figures for the exp040 R14 iterative-solver scaling experiment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP040_R14_FIGURE_FILENAMES = (
    "r14_axial_pml.png",
    "r14_solver_convergence.png",
    "r14_solver_resource.png",
)


def _solver_case(
    metrics: Mapping[str, Any], solver_id: str, case_id: str
) -> Mapping[str, Any] | None:
    value = metrics["solver_scaling"]["cases"][case_id]["solvers"].get(
        solver_id
    )
    if not isinstance(value, Mapping) or value.get("setup_succeeded") is not True:
        return None
    return value


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    result = float(value)
    return result if np.isfinite(result) else float("nan")


def save_exp040_r14_figures(
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> list[Path]:
    """Write the three pre-registered R14 figures."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / name for name in EXP040_R14_FIGURE_FILENAMES]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for case_id, values in arrays["axial_pml"].items():
        coordinate_um = np.asarray(values["dense_coordinates_m"]) * 1e6
        error = np.abs(
            np.asarray(values["dense_field"])
            - np.asarray(values["dense_truth"])
        )
        order = np.argsort(coordinate_um)
        axes[0].semilogy(
            coordinate_um[order],
            np.maximum(error[order], 1e-16),
            label=case_id,
        )
        measurement_um = np.abs(
            np.asarray(values["measurement_coordinates_m"])
        ) * 1e6
        axes[1].semilogy(
            measurement_um,
            values["incoming_to_outgoing_ratio"],
            marker="o",
            label=f"{case_id}: incoming/outgoing",
        )
        axes[1].semilogy(
            measurement_um,
            values["outgoing_impedance_residual"],
            marker="s",
            linestyle="--",
            label=f"{case_id}: impedance",
        )
    gate = float(
        metrics["thresholds"]["axial_pml_incoming_to_outgoing_ratio_max"]
    )
    axes[1].axhline(gate, color="black", linestyle=":", label="0.1% gate")
    axes[0].set(
        xlabel="physical axial coordinate (um)",
        ylabel="absolute field error",
        title="Axial outgoing plane-wave PML",
    )
    axes[1].set(
        xlabel="distance from driven boundary (um)",
        ylabel="relative diagnostic",
        title="Axial reflection controls",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=180)
    plt.close(fig)

    solver_ids = list(metrics["solver_scaling"]["fixed_solver_order"])
    case_ids = list(metrics["solver_scaling"]["fixed_case_order"])
    unknowns = np.asarray(
        [
            metrics["solver_scaling"]["cases"][case]["active_unknowns"]
            for case in case_ids
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for solver_id in solver_ids:
        for family in metrics["solver_scaling"]["modal_families"]:
            iterations: list[float] = []
            errors: list[float] = []
            for case_id in case_ids:
                value = _solver_case(metrics, solver_id, case_id)
                modal = None if value is None else value["modal_results"].get(family)
                solved = (
                    modal is not None
                    and modal.get("solve_succeeded") is True
                )
                iterations.append(
                    np.nan
                    if not solved
                    else float(modal["gmres"]["inner_iteration_count"])
                )
                errors.append(
                    np.nan
                    if not solved
                    else float(modal["analytic_weighted_relative_l2"])
                )
            label = f"{solver_id}: {family}"
            axes[0].plot(unknowns, iterations, marker="o", label=label)
            axes[1].loglog(unknowns, errors, marker="o", label=label)
    axes[0].axhline(
        float(metrics["thresholds"]["maximum_gmres_inner_iterations"]),
        color="black",
        linestyle=":",
        label="iteration gate",
    )
    axes[1].axhline(
        float(metrics["thresholds"]["analytic_field_weighted_relative_l2_max"]),
        color="black",
        linestyle=":",
        label="1% field gate",
    )
    axes[0].set(
        xscale="log",
        xlabel="active unknowns",
        ylabel="GMRES inner iterations",
        title="Iteration scaling",
    )
    axes[1].set(
        xlabel="active unknowns",
        ylabel="analytic-truth weighted relative L2",
        title="Field accuracy after iterative solve",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=180)
    plt.close(fig)

    summaries = metrics["solver_scaling"]["solver_summaries"]
    projected = [
        _finite_or_nan(summaries[solver].get("projected_peak_gib"))
        for solver in solver_ids
    ]
    largest_iterations = [
        _finite_or_nan(
            summaries[solver].get("maximum_largest_case_iterations")
        )
        for solver in solver_ids
    ]
    colors = [
        "tab:green" if summaries[solver]["solver_gate_pass"] else "tab:red"
        for solver in solver_ids
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].bar(solver_ids, projected, color=colors)
    axes[0].axhline(
        float(metrics["thresholds"]["maximum_projected_peak_gib"]),
        color="black",
        linestyle=":",
        label="10 GiB gate",
    )
    axes[0].set(
        ylabel="projected peak storage (GiB)",
        title="Full-TGV memory projection",
    )
    axes[1].bar(solver_ids, largest_iterations, color=colors)
    axes[1].axhline(
        float(metrics["thresholds"]["maximum_gmres_inner_iterations"]),
        color="black",
        linestyle=":",
        label="300-iteration gate",
    )
    axes[1].set(ylabel="worst core64 iterations", title="Largest-case convergence")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=180)
    plt.close(fig)
    return paths


__all__ = ["EXP040_R14_FIGURE_FILENAMES", "save_exp040_r14_figures"]
