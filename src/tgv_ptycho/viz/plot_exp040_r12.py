"""Diagnostic figures for the exp040 R12 reference-closure run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP040_R12_FIGURE_FILENAMES = (
    "r12_domain_fem_controls.png",
    "r12_cartesian_fov_alias_qdht.png",
    "r12_conditional_cross_model.png",
)


def save_exp040_r12_figures(
    output_dir: str | Path,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> list[Path]:
    """Save the fixed three-figure R12 diagnostic set."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / name for name in EXP040_R12_FIGURE_FILENAMES]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    domain = metrics["domain"]
    axes[0].bar(
        ["48→60\npassband", "core60\nguard", "FEM p3\nguard"],
        100.0
        * np.asarray(
            [
                domain["core48_to_core60_passband_radial_l2"],
                domain["core60_outer_guard_rms_ratio"],
                metrics["fem"]["p3_outer_guard_rms_ratio"],
            ]
        ),
        color=["#4C78A8", "#F58518", "#E45756"],
    )
    axes[0].axhline(5.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("relative metric (%)")
    axes[0].set_title("Domain / guard gates")
    fem = metrics["fem"]
    axes[1].bar(
        ["Q2→Q3", "ADC5→Q3\nreport"],
        100.0
        * np.asarray(
            [
                fem["p2_to_p3_passband_radial_l2"],
                fem["adc5_to_p3_passband_radial_l2_report_only"],
            ]
        ),
        color=["#54A24B", "#B279A2"],
    )
    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("weighted radial L2 (%)")
    axes[1].set_title("Higher-order FEM attribution")
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    cartesian = metrics["cartesian"]
    labels = ["FOV64", "FOV96", "FOV128", "FOV128 alias"]
    polar = cartesian["polar_controls"]
    angular = [
        polar[key]["angular_relative_l2"]
        for key in (
            "chord_fov64_standard",
            "chord_fov96_standard",
            "chord_fov128_standard",
            "chord_fov128_alias",
        )
    ]
    axes[0].plot(labels, 100.0 * np.asarray(angular), "o-", linewidth=1.5)
    axes[0].axhline(5.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("polar angular L2 (%)")
    axes[0].set_title("Cartesian anisotropy versus FOV/control")
    axes[0].tick_params(axis="x", rotation=20)
    radius_um = arrays["radial_radius_m"] * 1.0e6
    axes[1].plot(
        radius_um,
        np.abs(arrays["cartesian_alias_radial"]),
        label="Cartesian FOV128 alias",
    )
    axes[1].plot(
        radius_um,
        np.abs(arrays["qdht_radial"]),
        "--",
        label="QDHT report",
    )
    axes[1].set_xlabel("radius (µm)")
    axes[1].set_ylabel("normalized amplitude")
    axes[1].set_title("Axisymmetric report comparator")
    axes[1].legend(fontsize=8)
    fig.savefig(paths[1], dpi=160)
    plt.close(fig)

    cross = metrics["conditional_cross_model"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    if bool(cross["executed"]):
        radius_um = arrays["radial_radius_m"] * 1.0e6
        ax.plot(radius_um, np.abs(arrays["cross_fem_radial"]), label="FEM Q3")
        ax.plot(
            radius_um,
            np.abs(arrays["cross_multislice_radial"]),
            "--",
            label="multislice alias",
        )
        ax.set_xlabel("radius (µm)")
        ax.set_ylabel("normalized amplitude")
        ax.set_title(
            "Conditional scalar comparator: "
            f"{100.0 * cross['passband_radial_l2']:.3f}%"
        )
        ax.legend()
    else:
        ax.axis("off")
        failed = ", ".join(cross["failed_gates"])
        ax.text(
            0.5,
            0.55,
            "Scalar cross-model comparator skipped",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.text(0.5, 0.42, failed, ha="center", va="center", fontsize=9)
    fig.savefig(paths[2], dpi=160)
    plt.close(fig)
    return paths


__all__ = ["EXP040_R12_FIGURE_FILENAMES", "save_exp040_r12_figures"]
