"""Figure for the exp040 R14A axial-control attribution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP040_R14A_FIGURE_FILENAME = "r14a_axial_attribution.png"


def save_exp040_r14a_figure(
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> Path:
    """Plot fixed mesh-order and PML-thickness diagnostics."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / EXP040_R14A_FIGURE_FILENAME
    convergence = metrics["mesh_convergence"]
    h_um = np.asarray(convergence["element_sizes_m"]) * 1e6
    series = (
        (
            "incoming_to_outgoing_ratio",
            "maximum_incoming_to_outgoing_ratio",
            "incoming/outgoing",
            "o",
        ),
        (
            "outgoing_impedance_residual",
            "maximum_outgoing_impedance_residual",
            "impedance residual",
            "s",
        ),
        (
            "dense_field_relative_l2",
            "dense_field_relative_l2",
            "dense field L2",
            "^",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for key, order_key, label, marker in series:
        axes[0].loglog(
            h_um,
            convergence[key],
            marker=marker,
            label=f"{label}; slope={convergence['orders'][order_key]:.2f}",
        )
    axes[0].axhline(
        float(metrics["thresholds"]["incoming_to_outgoing_ratio_max"]),
        color="black",
        linestyle=":",
        label="0.1% original gate",
    )
    axes[0].invert_xaxis()
    axes[0].set(
        xlabel="Q4 element size (um)",
        ylabel="relative diagnostic",
        title="Glass axial mesh-order attribution",
    )

    pml2 = arrays["glass_h24_pml2"]
    pml3 = arrays["glass_h24_pml3"]
    coordinate_um = np.abs(np.asarray(pml2["dense_coordinates_m"])) * 1e6
    axes[1].semilogy(
        coordinate_um,
        np.maximum(
            np.abs(
                np.asarray(pml2["dense_field"])
                - np.asarray(pml3["dense_field"])
            ),
            1e-16,
        ),
        label="|PML2 field - PML3 field|",
    )
    axes[1].semilogy(
        coordinate_um,
        np.maximum(
            np.abs(
                np.asarray(pml2["dense_field"])
                - np.asarray(pml2["dense_truth"])
            ),
            1e-16,
        ),
        label="|PML2 field - truth|",
    )
    axes[1].set(
        xlabel="distance into physical core (um)",
        ylabel="absolute complex-field difference",
        title=(
            "PML-thickness separation; raw L2="
            f"{metrics['pml_separation']['raw_field_relative_l2']:.3e}"
        ),
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


__all__ = ["EXP040_R14A_FIGURE_FILENAME", "save_exp040_r14a_figure"]
