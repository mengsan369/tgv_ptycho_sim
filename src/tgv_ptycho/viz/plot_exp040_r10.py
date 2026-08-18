"""Backend-free plots for the formal exp040 R10 Stage-A comparison."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R10_STAGE_A_FIGURE_FILENAMES = (
    "r10_stage_a_convergence.png",
    "r10_stage_a_residuals.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _finite_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"R10 Stage-A {name} must be a finite 2D array.")
    return array


def save_exp040_r10_stage_a_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the two figures frozen in the R10 Stage-A registration."""

    metrics = _mapping(result.get("metrics"), "R10 Stage-A metrics")
    comparison = _mapping(metrics.get("comparison"), "comparison")
    provenance = _mapping(metrics.get("provenance"), "provenance")
    r9 = _mapping(provenance.get("r9_lateral_cell_average"), "R9 pair")
    threshold = float(
        _mapping(metrics.get("thresholds"), "thresholds")[
            "convergence_relative_l2_max"
        ]
    )
    raw = np.asarray(
        [float(r9["raw_relative_l2"]), float(comparison["raw_relative_l2"])]
    )
    passband = np.asarray(
        [
            float(r9["external_passband_relative_l2"]),
            float(comparison["external_passband_relative_l2"]),
        ]
    )
    convergence = _line_plot(
        np.arange(2, dtype=np.float64),
        [raw, passband, np.full(2, threshold)],
        ["Raw U_A_exit", "External propagating passband", "5.0% gate"],
        title="R10 Stage A: lateral pairwise error",
        x_label="0: dx 0.25->0.125 um (R9); 1: 0.125->0.0625 um (R10)",
        y_label="Unaligned relative L2 [1]",
        log_y=False,
    )

    maps = _mapping(result.get("selected_maps"), "selected_maps")
    current_dx_m = float(
        _mapping(metrics.get("sampling"), "sampling")["current_dx_m"]
    )
    panels = [
        _annotated_scalar_image(
            _finite_map(maps.get("raw_normalized_residual"), "raw residual"),
            cmap="magma",
            title="R10 raw residual",
            colorbar_label="|Delta U| / max|U_ref| [1]",
            dx=current_dx_m,
            vmin=0.0,
        ),
        _annotated_scalar_image(
            _finite_map(
                maps.get("passband_normalized_residual"), "passband residual"
            ),
            cmap="magma",
            title="R10 external-passband residual",
            colorbar_label="|Delta U| / max|U_ref| [1]",
            dx=current_dx_m,
            vmin=0.0,
        ),
        _annotated_scalar_image(
            _finite_map(maps.get("raw_difference_spectrum"), "difference spectrum"),
            cmap="magma",
            title="R10 raw difference spectrum",
            colorbar_label="log10 normalized power [1]",
        ),
        _annotated_scalar_image(
            _finite_map(maps.get("external_passband_mask"), "passband mask"),
            cmap="gray",
            title="External-medium propagating mask",
            colorbar_label="Mask [1]",
            vmin=0.0,
            vmax=1.0,
        ),
    ]

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in EXP040_R10_STAGE_A_FIGURE_FILENAMES]
    iio.imwrite(paths[0], convergence)
    iio.imwrite(paths[1], _join_panels(panels, columns=2))
    return paths


__all__ = [
    "EXP040_R10_STAGE_A_FIGURE_FILENAMES",
    "save_exp040_r10_stage_a_figures",
]
