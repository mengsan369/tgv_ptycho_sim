"""Backend-free plots for exp040 R3 detector-path diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R3_FIGURE_FILENAMES = (
    "r3_b_exit_and_bc_spectrum.png",
    "r3_detector_sampling_convergence.png",
    "r3_detector_operator_difference.png",
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{context} must be a mapping."
        raise ValueError(msg)
    return value


def _factor_axis(mapping: Mapping[str, Any], context: str) -> NDArray[np.float64]:
    factors = np.asarray(mapping.get("factors"), dtype=np.float64)
    if (
        factors.ndim != 1
        or factors.size != 3
        or not np.array_equal(factors, np.asarray([1.0, 2.0, 4.0]))
    ):
        msg = f"{context}.factors must be [1, 2, 4]."
        raise ValueError(msg)
    return factors


def _series(
    mapping: Mapping[str, Any],
    key: str,
    factors: NDArray[np.float64],
    context: str,
    *,
    upper_bound: float | None = None,
) -> NDArray[np.float64]:
    values = np.asarray(mapping.get(key), dtype=np.float64)
    if (
        values.shape != factors.shape
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or (upper_bound is not None and np.any(values > upper_bound))
    ):
        msg = f"{context}.{key} is invalid or does not match factors."
        raise ValueError(msg)
    return values


def _threshold_panel(
    factors: NDArray[np.float64],
    series: Mapping[str, NDArray[np.float64]],
    threshold: float,
    *,
    title: str,
    y_label: str,
    log_y: bool,
) -> NDArray[np.uint8]:
    x_values: list[NDArray[np.float64]] = []
    y_values: list[NDArray[np.float64]] = []
    labels: list[str] = []
    for label, values in series.items():
        keep = values > 0.0 if log_y else np.ones(values.shape, dtype=np.bool_)
        if np.any(keep):
            x_values.append(factors[keep])
            y_values.append(values[keep])
            labels.append(label)
    x_values.append(factors)
    y_values.append(np.full(factors.shape, threshold, dtype=np.float64))
    labels.append(f"{threshold:.1%} registered gate")
    return _line_plot(
        x_values,
        y_values,
        labels,
        title=f"{title} (exact zeros omitted)" if log_y else title,
        x_label="Computational sampling factor q [1]",
        y_label=y_label,
        log_y=log_y,
    )


def plot_r3_b_exit_and_bc_spectrum(
    metrics: Mapping[str, Any], save_path: str | Path
) -> None:
    """Plot the registered B-exit spectrum and BC method diagnostics."""

    spectra = _mapping(metrics.get("spectra"), "R3 spectra")
    factors = _factor_axis(spectra, "R3 spectra")
    b_exit = _mapping(spectra.get("B_exit"), "R3 spectra.B_exit")
    thresholds = _mapping(metrics.get("thresholds"), "R3 thresholds")
    threshold = float(thresholds["convergence_relative_l2_max"])
    b_panel = _threshold_panel(
        factors,
        {
            "outside BC mask mean": _series(
                b_exit,
                "outside_BC_alias_mask_energy_fraction_mean",
                factors,
                "R3 spectra.B_exit",
                upper_bound=1.0,
            ),
            "outside BC mask max": _series(
                b_exit,
                "outside_BC_alias_mask_energy_fraction_max",
                factors,
                "R3 spectra.B_exit",
                upper_bound=1.0,
            ),
            "outside detector Nyquist mean": _series(
                b_exit,
                "outside_native_detector_nyquist_energy_fraction_mean",
                factors,
                "R3 spectra.B_exit",
                upper_bound=1.0,
            ),
            "outside detector Nyquist max": _series(
                b_exit,
                "outside_native_detector_nyquist_energy_fraction_max",
                factors,
                "R3 spectra.B_exit",
                upper_bound=1.0,
            ),
        },
        threshold,
        title="R3 spectrum after B multiplication",
        y_label="Field-spectrum energy fraction [1]",
        log_y=True,
    )

    bc = _mapping(metrics.get("bc_propagation"), "R3 bc_propagation")
    bc_factors = _factor_axis(bc, "R3 bc_propagation")
    if not np.array_equal(factors, bc_factors):
        msg = "R3 spectrum and BC factor axes disagree."
        raise ValueError(msg)
    bc_panel = _threshold_panel(
        factors,
        {
            "detector field current vs alias": _series(
                bc,
                "detector_field_current_vs_alias_relative_l2",
                factors,
                "R3 bc_propagation",
            ),
            "full intensity current vs alias": _series(
                bc,
                "full_intensity_current_vs_alias_relative_l2",
                factors,
                "R3 bc_propagation",
            ),
        },
        threshold,
        title="R3 BC propagation sampling difference",
        y_label="Relative L2; alias-controlled reference [1]",
        log_y=True,
    )

    detector = _mapping(
        spectra.get("detector_intensity"), "R3 detector spectrum"
    )
    detector_series: dict[str, NDArray[np.float64]] = {}
    for method in ("current_asm", "alias_controlled"):
        group = _mapping(detector.get(method), f"R3 detector spectrum.{method}")
        detector_series[f"{method} outside Nyquist mean"] = _series(
            group,
            "outside_native_detector_nyquist_energy_fraction_mean",
            factors,
            f"R3 detector spectrum.{method}",
            upper_bound=1.0,
        )
        detector_series[f"{method} outside Nyquist max"] = _series(
            group,
            "outside_native_detector_nyquist_energy_fraction_max",
            factors,
            f"R3 detector spectrum.{method}",
            upper_bound=1.0,
        )
    detector_panel = _threshold_panel(
        factors,
        detector_series,
        threshold,
        title="R3 detector-plane intensity spectrum",
        y_label="Intensity-spectrum energy fraction [1]",
        log_y=True,
    )
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        output, _join_panels([b_panel, bc_panel, detector_panel], columns=1)
    )


def plot_r3_detector_sampling_convergence(
    metrics: Mapping[str, Any], save_path: str | Path
) -> None:
    """Plot P_B and four detector-branch sampling convergence series."""

    sampling = _mapping(metrics.get("detector_sampling"), "R3 detector_sampling")
    factors = _factor_axis(sampling, "R3 detector_sampling")
    relative = _mapping(
        sampling.get("relative_to_factor4"),
        "R3 detector_sampling.relative_to_factor4",
    )
    detector = _mapping(relative.get("detector"), "R3 detector convergence")
    values = {
        "P_B": _series(relative, "P_B", factors, "R3 detector convergence"),
    }
    for method in ("current_asm", "alias_controlled"):
        group = _mapping(detector.get(method), f"R3 detector convergence.{method}")
        for branch in ("point_sample", "pixel_box_average"):
            values[f"{method} + {branch}"] = _series(
                group,
                branch,
                factors,
                f"R3 detector convergence.{method}",
            )
    threshold = float(
        _mapping(metrics.get("thresholds"), "R3 thresholds")[
            "convergence_relative_l2_max"
        ]
    )
    panel = _threshold_panel(
        factors,
        values,
        threshold,
        title="R3 fixed-192 um sampling convergence to factor 4",
        y_label="Native 128^2 ROI relative L2 [1]",
        log_y=True,
    )
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, panel)


def plot_r3_detector_operator_difference(
    metrics: Mapping[str, Any],
    selected_scan: Mapping[str, Any],
    detector_pixel_m: float,
    save_path: str | Path,
) -> None:
    """Plot point-vs-pixel differences and the selected primary scan."""

    operator = _mapping(
        metrics.get("detector_operator_difference"),
        "R3 detector_operator_difference",
    )
    factors = _factor_axis(operator, "R3 detector_operator_difference")
    differences = _mapping(
        operator.get("point_vs_pixel_relative_l2"),
        "R3 point_vs_pixel_relative_l2",
    )
    threshold = float(
        _mapping(metrics.get("thresholds"), "R3 thresholds")[
            "convergence_relative_l2_max"
        ]
    )
    line_panel = _threshold_panel(
        factors,
        {
            "current ASM point vs pixel": _series(
                differences,
                "current_asm",
                factors,
                "R3 point_vs_pixel_relative_l2",
            ),
            "alias-controlled point vs pixel": _series(
                differences,
                "alias_controlled",
                factors,
                "R3 point_vs_pixel_relative_l2",
            ),
        },
        threshold,
        title="R3 detector operator materiality",
        y_label="Relative L2; pixel-integrated denominator [1]",
        log_y=True,
    )
    point = np.asarray(selected_scan.get("point_sample"), dtype=np.float64)
    pixel = np.asarray(selected_scan.get("pixel_box_average"), dtype=np.float64)
    difference = np.asarray(
        selected_scan.get("relative_difference"), dtype=np.float64
    )
    if (
        point.ndim != 2
        or pixel.shape != point.shape
        or difference.shape != point.shape
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(pixel))
        or not np.all(np.isfinite(difference))
    ):
        msg = "R3 selected-scan detector arrays are invalid."
        raise ValueError(msg)
    common_max = max(float(np.max(point)), float(np.max(pixel)))
    image_panels = [
        _annotated_scalar_image(
            point,
            cmap="gray",
            title="Factor-4 alias BC: point sample",
            colorbar_label="Intensity [a.u.]",
            dx=detector_pixel_m,
            vmin=0.0,
            vmax=common_max,
        ),
        _annotated_scalar_image(
            pixel,
            cmap="gray",
            title="Factor-4 alias BC: square-pixel average",
            colorbar_label="Intensity [a.u.]",
            dx=detector_pixel_m,
            vmin=0.0,
            vmax=common_max,
        ),
        _annotated_scalar_image(
            difference,
            cmap="coolwarm",
            title="(point - pixel) / max(pixel)",
            colorbar_label="Relative difference [1]",
            dx=detector_pixel_m,
            vmin=-max(float(np.max(np.abs(difference))), np.finfo(float).eps),
            vmax=max(float(np.max(np.abs(difference))), np.finfo(float).eps),
        ),
    ]
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, _join_panels([line_panel, *image_panels], columns=1))


def save_exp040_r3_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three pre-registered R3 detector-path figures."""

    metrics_root = _mapping(result.get("metrics"), "result.metrics")
    metrics = _mapping(
        metrics_root.get("diagnostics_r3"), "result.metrics.diagnostics_r3"
    )
    diagnostic = _mapping(result.get("diagnostics_r3"), "result.diagnostics_r3")
    selected = _mapping(diagnostic.get("selected_scan"), "R3 selected_scan")
    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R3_FIGURE_FILENAMES]
    plot_r3_b_exit_and_bc_spectrum(metrics, paths[0])
    plot_r3_detector_sampling_convergence(metrics, paths[1])
    detector_pixel = float(
        np.asarray(result["baseline"]["dx_m"], dtype=np.float64)
    )
    plot_r3_detector_operator_difference(
        metrics, selected, detector_pixel, paths[2]
    )
    return paths


__all__ = [
    "EXP040_R3_FIGURE_FILENAMES",
    "plot_r3_b_exit_and_bc_spectrum",
    "plot_r3_detector_operator_difference",
    "plot_r3_detector_sampling_convergence",
    "save_exp040_r3_figures",
]
