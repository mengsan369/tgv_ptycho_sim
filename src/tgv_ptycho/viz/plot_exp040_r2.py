"""Backend-free plots for the exp040 R2 boundary/alias diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R2_FIGURE_FILENAMES = (
    "r2_period_aligned_convergence.png",
    "r2_alias_method_difference.png",
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{context} must be a mapping."
        raise ValueError(msg)
    return value


def _series(
    mapping: Mapping[str, Any],
    key: str,
    x: NDArray[np.float64],
    context: str,
    *,
    upper_bound: float | None = None,
) -> NDArray[np.float64]:
    values = np.asarray(mapping.get(key), dtype=np.float64)
    if (
        values.shape != x.shape
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or (upper_bound is not None and np.any(values > upper_bound))
    ):
        msg = f"{context}.{key} is invalid or does not match x_values_m."
        raise ValueError(msg)
    return values


def _x_values(mapping: Mapping[str, Any], context: str) -> NDArray[np.float64]:
    values = np.asarray(mapping.get("x_values_m"), dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        msg = f"{context}.x_values_m must be finite, positive, and 1D."
        raise ValueError(msg)
    return values


def _positive_log_panel(
    x_m: NDArray[np.float64],
    values: Mapping[str, NDArray[np.float64]],
    threshold: float,
    *,
    title: str,
    threshold_label: str,
    y_label: str,
) -> NDArray[np.uint8]:
    x_series: list[NDArray[np.float64]] = []
    y_series: list[NDArray[np.float64]] = []
    labels: list[str] = []
    for name, series in values.items():
        positive = series > 0.0
        if np.any(positive):
            x_series.append(x_m[positive] * 1e6)
            y_series.append(series[positive])
            labels.append(name)
    x_series.append(x_m * 1e6)
    y_series.append(np.full(x_m.shape, threshold, dtype=np.float64))
    labels.append(threshold_label)
    return _line_plot(
        x_series,
        y_series,
        labels,
        title=f"{title} (exact-zero reference omitted)",
        x_label="Period-commensurate external FOV [um]",
        y_label=y_label,
        log_y=True,
    )


def plot_r2_period_aligned_convergence(
    period_aligned: Mapping[str, Any],
    threshold: float,
    save_path: str | Path,
) -> None:
    """Plot current and alias-controlled FOV convergence separately."""

    panels: list[NDArray[np.uint8]] = []
    for key, title in (
        ("current_asm", "R2 current ASM period-aligned convergence"),
        (
            "alias_controlled",
            "R2 alias-controlled ASM period-aligned convergence",
        ),
    ):
        group = _mapping(period_aligned.get(key), f"period_aligned.{key}")
        x = _x_values(group, f"period_aligned.{key}")
        values = {
            "P_B": _series(group, "P_B", x, f"period_aligned.{key}"),
            "I_stack": _series(
                group, "I_stack", x, f"period_aligned.{key}"
            ),
        }
        panels.append(
            _positive_log_panel(
                x,
                values,
                threshold,
                title=title,
                threshold_label=f"{threshold:.1%} convergence gate",
                y_label="Relative L2 error to 288 um reference [1]",
            )
        )
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, _join_panels(panels, columns=1))


def plot_r2_alias_method_difference(
    method_difference: Mapping[str, Any],
    alias_masks: Mapping[str, Any],
    threshold: float,
    save_path: str | Path,
) -> None:
    """Plot method differences and AB/BC kept-bin fractions."""

    x = _x_values(method_difference, "method_difference")
    values = {
        "P_B current vs alias": _series(
            method_difference, "P_B", x, "method_difference"
        ),
        "I_stack current vs alias": _series(
            method_difference, "I_stack", x, "method_difference"
        ),
    }
    difference_panel = _positive_log_panel(
        x,
        values,
        threshold,
        title="R2 current-vs-alias-controlled method difference",
        threshold_label=f"{threshold:.1%} materiality gate",
        y_label="Relative L2 difference; alias-controlled reference [1]",
    )
    mask_x = _x_values(alias_masks, "alias_masks")
    if not np.array_equal(mask_x, x):
        msg = "alias mask and method-difference FOV values must match."
        raise ValueError(msg)
    ab = _series(
        alias_masks,
        "AB_kept_bin_fraction",
        mask_x,
        "alias_masks",
        upper_bound=1.0,
    )
    bc = _series(
        alias_masks,
        "BC_kept_bin_fraction",
        mask_x,
        "alias_masks",
        upper_bound=1.0,
    )
    mask_panel = _line_plot(
        [mask_x * 1e6, mask_x * 1e6],
        [ab, bc],
        ["AB kept FFT-bin fraction", "BC kept FFT-bin fraction"],
        title="R2 exact common-ellipse mask support",
        x_label="Period-commensurate external FOV [um]",
        y_label="Kept transfer bins / all FFT bins [1]",
        log_y=False,
    )
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, _join_panels([difference_panel, mask_panel], columns=1))


def save_exp040_r2_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the two pre-registered compact R2 diagnostic figures."""

    metrics_root = _mapping(result.get("metrics"), "result.metrics")
    metrics = _mapping(
        metrics_root.get("diagnostics_r2"), "result.metrics.diagnostics_r2"
    )
    thresholds = _mapping(metrics.get("thresholds"), "R2 thresholds")
    threshold = float(thresholds.get("convergence_relative_l2_max"))
    if not np.isfinite(threshold) or threshold <= 0.0:
        msg = "R2 convergence threshold must be finite and positive."
        raise ValueError(msg)
    period_aligned = _mapping(
        metrics.get("period_aligned"), "R2 period_aligned"
    )
    method_difference = _mapping(
        metrics.get("method_difference"), "R2 method_difference"
    )
    alias_masks = _mapping(metrics.get("alias_masks"), "R2 alias_masks")
    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R2_FIGURE_FILENAMES]
    plot_r2_period_aligned_convergence(period_aligned, threshold, paths[0])
    plot_r2_alias_method_difference(
        method_difference, alias_masks, threshold, paths[1]
    )
    return paths


__all__ = [
    "EXP040_R2_FIGURE_FILENAMES",
    "plot_r2_alias_method_difference",
    "plot_r2_period_aligned_convergence",
    "save_exp040_r2_figures",
]
