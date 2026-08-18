"""Independent backend-free plots for the exp040 R1 diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R1_FIGURE_FILENAMES = (
    "r1_refined_convergence.png",
    "r1_external_padding_convergence.png",
)

_OUTPUTS = ("U_A_exit", "P_B", "I_stack")


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{context} must be a mapping."
        raise ValueError(msg)
    return value


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        msg = f"{context} is missing required key {key!r}."
        raise ValueError(msg)
    return mapping[key]


def _finite_positive_x(value: Any, context: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size < 2
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        msg = f"{context} must be a finite positive 1D array with >=2 entries."
        raise ValueError(msg)
    return array


def _finite_errors(
    value: Any, x: NDArray[np.float64], context: str
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != x.shape
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        msg = f"{context} must be finite, non-negative, and match x_values."
        raise ValueError(msg)
    return array


def _group_series(
    group: Mapping[str, Any],
    outputs: Sequence[str],
    context: str,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    x = _finite_positive_x(_required(group, "x_values", context), f"{context}.x_values")
    errors = {
        name: _finite_errors(
            _required(group, name, context), x, f"{context}.{name}"
        )
        for name in outputs
    }
    return x, errors


def _threshold(
    metrics: Mapping[str, Any],
    result: Mapping[str, Any],
    key: str,
) -> float:
    candidates: list[Any] = []
    if key in metrics:
        candidates.append(metrics[key])
    nested = metrics.get("thresholds")
    if isinstance(nested, Mapping) and key in nested:
        candidates.append(nested[key])
    result_metrics = result.get("metrics")
    if isinstance(result_metrics, Mapping):
        legacy = result_metrics.get("thresholds")
        if isinstance(legacy, Mapping) and key in legacy:
            candidates.append(legacy[key])
    if not candidates:
        msg = f"R1 metrics must provide {key}."
        raise ValueError(msg)
    threshold = float(candidates[0])
    if not np.isfinite(threshold) or threshold <= 0.0:
        msg = f"{key} must be finite and positive."
        raise ValueError(msg)
    return threshold


def _positive_log_series(
    x: NDArray[np.float64],
    errors: Mapping[str, NDArray[np.float64]],
    threshold: float,
    *,
    gate_label: str | None = None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[str]]:
    x_series: list[NDArray[np.float64]] = []
    y_series: list[NDArray[np.float64]] = []
    labels: list[str] = []
    for name, values in errors.items():
        positive = values > 0.0
        if not np.any(positive):
            continue
        x_series.append(x[positive])
        y_series.append(values[positive])
        labels.append(name)
    x_series.append(x)
    y_series.append(np.full(x.shape, threshold, dtype=np.float64))
    labels.append(gate_label or f"{threshold:.1%} convergence gate")
    return x_series, y_series, labels


def _convergence_panel(
    group: Mapping[str, Any],
    *,
    context: str,
    title: str,
    x_label: str,
    threshold: float,
) -> NDArray[np.uint8]:
    x, errors = _group_series(group, _OUTPUTS, context)
    x_series, y_series, labels = _positive_log_series(
        x * 1e6, errors, threshold
    )
    return _line_plot(
        x_series,
        y_series,
        labels,
        title=f"{title} (exact-zero reference points omitted)",
        x_label=x_label,
        y_label="Relative L2 error to R1 reference [1]",
        log_y=True,
    )


def plot_r1_refined_convergence(
    refined: Mapping[str, Any],
    threshold: float,
    save_path: str | Path,
) -> None:
    """Save the three-panel refined axial/lateral/FOV R1 diagnostic."""

    axial = _as_mapping(
        _required(refined, "axial", "refined_convergence"),
        "refined_convergence.axial",
    )
    lateral = _as_mapping(
        _required(refined, "lateral", "refined_convergence"),
        "refined_convergence.lateral",
    )
    fov = _as_mapping(
        _required(refined, "fov", "refined_convergence"),
        "refined_convergence.fov",
    )
    panels = [
        _convergence_panel(
            axial,
            context="refined_convergence.axial",
            title="R1 refined axial convergence",
            x_label="Comparison-case target dz [um]",
            threshold=threshold,
        ),
        _convergence_panel(
            lateral,
            context="refined_convergence.lateral",
            title="R1 refined fixed-FOV lateral convergence",
            x_label="Comparison-grid dx [um]",
            threshold=threshold,
        ),
        _convergence_panel(
            fov,
            context="refined_convergence.fov",
            title="R1 refined internal-FOV convergence on common ROI",
            x_label="Comparison-case FOV width [um]",
            threshold=threshold,
        ),
    ]
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, _join_panels(panels, columns=1))


def plot_r1_external_padding_convergence(
    external: Mapping[str, Any],
    convergence_threshold: float,
    invariance_threshold: float,
    save_path: str | Path,
) -> None:
    """Save external P/I errors and sample-A center-invariance diagnostics."""

    x, propagation_errors = _group_series(
        external,
        ("P_B", "I_stack"),
        "external_padding",
    )
    invariance_key = (
        "U_A_exit_center_invariance"
        if "U_A_exit_center_invariance" in external
        else "U_A_exit"
    )
    invariance = _finite_errors(
        _required(external, invariance_key, "external_padding"),
        x,
        f"external_padding.{invariance_key}",
    )
    propagation_x, propagation_y, propagation_labels = _positive_log_series(
        x * 1e6,
        propagation_errors,
        convergence_threshold,
    )
    invariance_x, invariance_y, invariance_labels = _positive_log_series(
        x * 1e6,
        {"U_A_exit center invariance": invariance},
        invariance_threshold,
        gate_label=f"{invariance_threshold:.1e} A-exit invariance gate",
    )
    panels = [
        _line_plot(
            propagation_x,
            propagation_y,
            propagation_labels,
            title=(
                "R1 external-padding convergence on common ROI "
                "(exact-zero reference omitted)"
            ),
            x_label="Padded external FOV width [um]",
            y_label="Relative L2 error to largest padding [1]",
            log_y=True,
        ),
        _line_plot(
            invariance_x,
            invariance_y,
            invariance_labels,
            title=(
                "R1 fixed A-exit center invariance "
                "(exact-zero reference omitted)"
            ),
            x_label="Padded external FOV width [um]",
            y_label="A-exit common-ROI relative L2 change [1]",
            log_y=True,
        ),
    ]
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, _join_panels(panels, columns=1))


def save_exp040_r1_figures(
    result: dict[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the two R1 figures without changing the original exp040 API."""

    if not isinstance(result, dict):
        msg = "result must be a dictionary."
        raise ValueError(msg)
    diagnostics = _as_mapping(
        _required(result, "diagnostics_r1", "result"),
        "result.diagnostics_r1",
    )
    metrics_root = _as_mapping(
        _required(result, "metrics", "result"), "result.metrics"
    )
    metrics = _as_mapping(
        _required(metrics_root, "diagnostics_r1", "result.metrics"),
        "result.metrics.diagnostics_r1",
    )
    convergence_threshold = _threshold(
        metrics,
        result,
        "convergence_relative_l2_max",
    )
    invariance_threshold = _threshold(
        metrics,
        result,
        "a_exit_center_invariance_max",
    )
    refined_source = diagnostics.get(
        "refined_convergence", metrics.get("refined_convergence")
    )
    external_source = diagnostics.get(
        "external_padding", metrics.get("external_padding")
    )
    refined = _as_mapping(refined_source, "diagnostics_r1.refined_convergence")
    external = _as_mapping(external_source, "diagnostics_r1.external_padding")

    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R1_FIGURE_FILENAMES]
    plot_r1_refined_convergence(refined, convergence_threshold, paths[0])
    plot_r1_external_padding_convergence(
        external,
        convergence_threshold,
        invariance_threshold,
        paths[1],
    )
    return paths


__all__ = [
    "EXP040_R1_FIGURE_FILENAMES",
    "plot_r1_external_padding_convergence",
    "plot_r1_refined_convergence",
    "save_exp040_r1_figures",
]
