"""Backend-free plots for exp040 R9 A-exit attribution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R9_FIGURE_FILENAMES = (
    "r9_a_exit_convergence.png",
    "r9_lateral_restriction.png",
    "r9_difference_spectra.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _finite_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"R9 {name} must be a finite 2D array.")
    return array


def save_exp040_r9_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three pre-registered R9 figures."""

    diagnostics = _mapping(result.get("diagnostics_r9"), "diagnostics_r9")
    metrics = _mapping(
        _mapping(result.get("metrics"), "result.metrics").get(
            "diagnostics_r9"
        ),
        "R9 metrics",
    )
    comparisons = _mapping(metrics.get("comparisons"), "R9 comparisons")
    names = (
        "r8_axial_reproduction",
        "axial_refinement",
        "lateral_bilinear",
        "lateral_cell_average",
    )
    raw = np.asarray(
        [float(_mapping(comparisons[name], name)["raw_relative_l2"]) for name in names]
    )
    passband = np.asarray(
        [
            float(
                _mapping(comparisons[name], name)[
                    "external_passband_relative_l2"
                ]
            )
            for name in names
        ]
    )
    threshold = float(
        _mapping(metrics.get("thresholds"), "R9 thresholds")[
            "convergence_relative_l2_max"
        ]
    )
    x = np.arange(len(names), dtype=np.float64)
    convergence = _line_plot(
        x,
        [raw, passband, np.full(len(names), threshold)],
        ["Raw U_A_exit", "External propagating passband", "5.0% gate"],
        title="R9 A-exit attribution: raw versus external passband",
        x_label="0 R8 axial, 1 refined axial, 2 bilinear, 3 cell average",
        y_label="Unaligned relative L2 [1]",
        log_y=False,
    )

    maps = _mapping(diagnostics.get("selected_maps"), "R9 selected maps")
    common_dx = float(
        _mapping(
            _mapping(metrics.get("passband"), "R9 passband").get(
                "native_projection_controls"
            ),
            "R9 native controls",
        )["common_reference"]["dx_m"]
    )
    map_specs = (
        ("lateral_raw_bilinear", "Raw lateral error / bilinear"),
        ("lateral_raw_cell_average", "Raw lateral error / cell average"),
        ("lateral_passband_bilinear", "Passband error / bilinear"),
        ("lateral_passband_cell_average", "Passband error / cell average"),
        ("restriction_disagreement", "Bilinear vs cell-average reference"),
    )
    map_panels = [
        _annotated_scalar_image(
            _finite_map(maps.get(name), name),
            cmap="magma",
            title=title,
            colorbar_label="|Delta U| / max|U_ref| [1]",
            dx=common_dx,
            vmin=0.0,
        )
        for name, title in map_specs
    ]

    spectra = _mapping(
        diagnostics.get("difference_spectra"), "R9 difference spectra"
    )
    spectrum_panels = [
        _annotated_scalar_image(
            _finite_map(spectra.get("axial_refinement"), "axial spectrum"),
            cmap="magma",
            title="Refined axial difference spectrum",
            colorbar_label="log10 normalized power [1]",
        ),
        _annotated_scalar_image(
            _finite_map(
                spectra.get("lateral_cell_average"), "lateral spectrum"
            ),
            cmap="magma",
            title="Lateral cell-average difference spectrum",
            colorbar_label="log10 normalized power [1]",
        ),
        _annotated_scalar_image(
            _finite_map(
                spectra.get("external_passband_mask"), "passband mask"
            ),
            cmap="gray",
            title="External-medium propagating mask",
            colorbar_label="Mask [1]",
            vmin=0.0,
            vmax=1.0,
        ),
    ]

    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R9_FIGURE_FILENAMES]
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(paths[0], convergence)
    iio.imwrite(paths[1], _join_panels(map_panels, columns=3))
    iio.imwrite(paths[2], _join_panels(spectrum_panels, columns=3))
    return paths


__all__ = ["EXP040_R9_FIGURE_FILENAMES", "save_exp040_r9_figures"]
