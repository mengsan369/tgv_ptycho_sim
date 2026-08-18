"""Figures for the exp040 R13 reflection and pollution benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP040_R13_FIGURE_FILENAMES = (
    "r13_domain_reflection.png",
    "r13_physical_k_pollution.png",
    "r13_candidate_resource.png",
)


def save_exp040_r13_figures(
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> list[Path]:
    """Write the three pre-registered R13 diagnostic figures."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / name for name in EXP040_R13_FIGURE_FILENAMES]

    domain_arrays = arrays["domain_reflection"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for case_id, values in domain_arrays.items():
        radius_um = np.asarray(values["dense_radii_m"]) * 1e6
        error = np.abs(
            np.asarray(values["dense_field"])
            - np.asarray(values["dense_truth"])
        )
        axes[0].semilogy(radius_um, np.maximum(error, 1e-16), label=case_id)
        measurement_um = np.asarray(values["measurement_radii_m"]) * 1e6
        axes[1].semilogy(
            measurement_um,
            np.asarray(values["incoming_to_outgoing_ratio"]),
            marker="o",
            label=f"{case_id}: incoming/outgoing",
        )
        axes[1].semilogy(
            measurement_um,
            np.asarray(values["outgoing_impedance_residual"]),
            marker="s",
            linestyle="--",
            label=f"{case_id}: impedance",
        )
    axes[0].set(
        xlabel="radius (um)",
        ylabel="absolute field error",
        title="Outgoing Hankel field error before PML",
    )
    axes[1].axhline(
        float(metrics["thresholds"]["pml_incoming_to_outgoing_ratio_max"]),
        color="black",
        linestyle=":",
        label="0.1% gate",
    )
    axes[1].set(
        xlabel="radius (um)",
        ylabel="relative diagnostic",
        title="Reflection and outgoing-condition controls",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=180)
    plt.close(fig)

    pollution = metrics["physical_k_pollution"]["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    families = (("homogeneous", axes[0]), ("glass_air_interface", axes[1]))
    ratios = sorted(
        {float(value["element_size_ratio"]) for value in pollution.values()},
        reverse=True,
    )
    for family, axis in families:
        for ratio in ratios:
            selected = sorted(
                (
                    (int(value["degree"]), float(value[family]["weighted_relative_l2"]))
                    for value in pollution.values()
                    if float(value["element_size_ratio"]) == ratio
                ),
                key=lambda item: item[0],
            )
            axis.semilogy(
                [item[0] for item in selected],
                [item[1] for item in selected],
                marker="o",
                label=f"h/h_formal={ratio:g}",
            )
        axis.axhline(
            float(
                metrics["thresholds"][
                    "physical_k_family_weighted_relative_l2_max"
                ]
            ),
            color="black",
            linestyle=":",
            label="1% gate",
        )
        axis.set(
            xlabel="polynomial degree p",
            title=family.replace("_", " "),
        )
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("analytic-truth weighted relative L2")
    fig.suptitle("Physical-k hp-FEM pollution series (kh_formal=8.85787)")
    fig.tight_layout()
    fig.savefig(paths[1], dpi=180)
    plt.close(fig)

    ordered = list(metrics["physical_k_pollution"]["fixed_case_order"])
    unknowns = np.asarray(
        [
            pollution[case_id]["estimated_full_tgv_active_unknowns"]
            for case_id in ordered
        ]
    )
    eligible = np.asarray(
        [pollution[case_id]["candidate_eligible"] for case_id in ordered]
    )
    selected_id = metrics["physical_k_pollution"]["selected_candidate_id"]
    colors = [
        "tab:red" if case_id == selected_id else ("tab:green" if ok else "tab:gray")
        for case_id, ok in zip(ordered, eligible, strict=True)
    ]
    fig, axis = plt.subplots(figsize=(11.2, 4.4))
    axis.bar(np.arange(len(ordered)), unknowns / 1e6, color=colors)
    axis.axhline(
        float(metrics["full_tgv_projection"]["direct_lu_maximum_unknowns"]) / 1e6,
        color="black",
        linestyle=":",
        label="direct-LU unknown-count ceiling",
    )
    axis.set_xticks(np.arange(len(ordered)), ordered, rotation=45, ha="right")
    axis.set(
        ylabel="projected active unknowns (million)",
        title="Accuracy-eligible hp cases and projected full-TGV size",
    )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(paths[2], dpi=180)
    plt.close(fig)
    return paths


__all__ = ["EXP040_R13_FIGURE_FILENAMES", "save_exp040_r13_figures"]
