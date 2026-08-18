"""Backend-free plots for exp040 R4 positive detector quadrature."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R4_FIGURE_FILENAMES = (
    "r4_positive_quadrature_convergence.png",
    "r4_positive_quadrature_controls.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _positive_series(factors: np.ndarray, values: Any) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(values, dtype=np.float64)
    if (
        series.shape != factors.shape
        or not np.all(np.isfinite(series))
        or np.any(series < 0)
    ):
        raise ValueError("R4 plot series is invalid.")
    keep = series > 0.0
    return factors[keep], series[keep]


def save_exp040_r4_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the two pre-registered R4 figures."""

    metrics = _mapping(
        _mapping(result.get("metrics"), "result.metrics").get("diagnostics_r4"),
        "diagnostics_r4",
    )
    convergence = _mapping(metrics.get("convergence"), "R4 convergence")
    factors = np.asarray(convergence.get("factors"), dtype=np.float64)
    if not np.array_equal(factors, np.asarray([2.0, 4.0, 8.0])):
        raise ValueError("R4 plot factors must be [2, 4, 8].")
    relative = _mapping(convergence.get("relative_to_q8"), "R4 relative_to_q8")
    threshold = float(
        _mapping(metrics.get("thresholds"), "R4 thresholds")[
            "convergence_relative_l2_max"
        ]
    )
    x_p, y_p = _positive_series(factors, relative.get("P_B"))
    x_i, y_i = _positive_series(factors, relative.get("I_stack"))
    convergence_panel = _line_plot(
        [x_p, x_i, factors],
        [y_p, y_i, np.full(3, threshold)],
        ["P_B block-mean", "positive pixel I_stack", "5.0% gate"],
        title="R4 staggered quadrature convergence (exact q8 zeros omitted)",
        x_label="Quadrature factor q [1]",
        y_label="log10(relative L2 to q8) [1]",
        log_y=True,
    )

    controls = _mapping(metrics.get("quadrature_controls"), "R4 controls")
    control_values = [
        np.asarray(controls[name], dtype=np.float64)
        for name in (
            "node_geometry_normalized_error_by_factor",
            "constant_max_abs_error_by_factor",
            "sum_relative_error_by_factor",
        )
    ]
    control_labels = ["node geometry", "constant", "sum identity"]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    labels: list[str] = []
    for values, label in zip(control_values, control_labels, strict=True):
        x, y = _positive_series(factors, values)
        if y.size:
            xs.append(x)
            ys.append(y)
            labels.append(label)
    xs.append(factors)
    ys.append(np.full(3, float(metrics["thresholds"]["algebra_relative_l2_max"])))
    labels.append("1e-12 hard gate")
    control_panel = _line_plot(
        xs,
        ys,
        labels,
        title="R4 positive quadrature hard controls (exact zeros omitted)",
        x_label="Quadrature factor q [1]",
        y_label="log10(normalized error) [1]",
        log_y=True,
    )
    selected = np.asarray(
        _mapping(result.get("diagnostics_r4"), "result.diagnostics_r4").get(
            "selected_q8_scan0"
        ),
        dtype=np.float64,
    )
    if selected.ndim != 2 or not np.all(np.isfinite(selected)) or np.any(selected < 0):
        raise ValueError("R4 selected q8 image is invalid.")
    image_panel = _annotated_scalar_image(
        selected,
        cmap="gray",
        title="R4 q8 scan 0 positive pixel integration",
        colorbar_label="Intensity [a.u.]",
        dx=float(result["baseline"]["dx_m"]),
        vmin=0.0,
    )
    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R4_FIGURE_FILENAMES]
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(paths[0], convergence_panel)
    iio.imwrite(paths[1], _join_panels([control_panel, image_panel], columns=1))
    return paths


__all__ = ["EXP040_R4_FIGURE_FILENAMES", "save_exp040_r4_figures"]
