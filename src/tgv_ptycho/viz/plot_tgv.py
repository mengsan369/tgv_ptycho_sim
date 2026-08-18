"""Backend-independent plots for TGV geometry and observability."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from tgv_ptycho.viz.plot_field import (
    _annotated_scalar_image,
    _colormap_uint8,
    _font,
    _format_tick,
)
from tgv_ptycho.viz.plot_recon import _join_panels

_COLORS = (
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (148, 103, 189),
    (255, 127, 14),
)


def _save(save_path: str | Path, rgb: NDArray[np.uint8]) -> None:
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.asarray(rgb, dtype=np.uint8))


def _line_plot(
    x_values: NDArray[np.floating] | Sequence[NDArray[np.floating]],
    y_series: Sequence[NDArray[np.floating]],
    labels: Sequence[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
    log_y: bool = False,
) -> NDArray[np.uint8]:
    """Render a multi-series line plot without a GUI plotting backend."""

    if len(y_series) != len(labels) or not y_series:
        msg = "y_series and labels must be non-empty and have equal length."
        raise ValueError(msg)

    try:
        candidate_x: NDArray[np.float64] | None = np.asarray(
            x_values, dtype=np.float64
        )
    except ValueError:
        candidate_x = None
    if candidate_x is not None and candidate_x.ndim == 1:
        x_series = [candidate_x] * len(y_series)
    elif len(x_values) == len(y_series):
        x_series = [np.asarray(values, dtype=np.float64) for values in x_values]
    else:
        msg = "x_values must be one shared array or one array per y series."
        raise ValueError(msg)
    if any(
        x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x))
        for x in x_series
    ):
        msg = "Every x series must be a non-empty finite 1D array."
        raise ValueError(msg)

    series: list[NDArray[np.float64]] = []
    for x, values in zip(x_series, y_series, strict=True):
        y = np.asarray(values, dtype=np.float64)
        if y.shape != x.shape or not np.all(np.isfinite(y)):
            msg = "Every y series must be finite and match its x series."
            raise ValueError(msg)
        if log_y:
            y = np.log10(np.maximum(y, np.finfo(float).tiny))
        series.append(y)

    width, height = 900, 460
    left, right, top, bottom = 94, 220, 62, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(13)
    tick_font = _font(11)

    x_min = min(float(np.min(values)) for values in x_series)
    x_max = max(float(np.max(values)) for values in x_series)
    if x_max - x_min <= np.finfo(float).eps:
        x_min -= 0.5
        x_max += 0.5
    y_min = min(float(np.min(values)) for values in series)
    y_max = max(float(np.max(values)) for values in series)
    if y_max - y_min <= np.finfo(float).eps * max(abs(y_max), 1.0):
        padding = 0.05 * max(abs(y_max), 1.0)
    else:
        padding = 0.05 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    draw.text((left, 18), title, fill="black", font=title_font)
    draw.rectangle(
        [left, top, left + plot_width, top + plot_height], outline="black"
    )
    for fraction in np.linspace(0.0, 1.0, 5):
        x_pixel = int(left + fraction * plot_width)
        y_pixel = int(top + fraction * plot_height)
        draw.line(
            [x_pixel, top, x_pixel, top + plot_height], fill=(232, 232, 232)
        )
        draw.line(
            [left, y_pixel, left + plot_width, y_pixel], fill=(232, 232, 232)
        )
        x_tick = _format_tick(x_min + fraction * (x_max - x_min))
        x_bbox = draw.textbbox((0, 0), x_tick, font=tick_font)
        draw.text(
            (x_pixel - (x_bbox[2] - x_bbox[0]) / 2, top + plot_height + 9),
            x_tick,
            fill="black",
            font=tick_font,
        )
        y_tick = _format_tick(y_max - fraction * (y_max - y_min))
        y_bbox = draw.textbbox((0, 0), y_tick, font=tick_font)
        draw.text(
            (left - 10 - (y_bbox[2] - y_bbox[0]), y_pixel - 6),
            y_tick,
            fill="black",
            font=tick_font,
        )

    for index, (x, values) in enumerate(zip(x_series, series, strict=True)):
        points = [
            (
                int(left + (value_x - x_min) / (x_max - x_min) * plot_width),
                int(top + (y_max - value_y) / (y_max - y_min) * plot_height),
            )
            for value_x, value_y in zip(x, values, strict=True)
        ]
        color = _COLORS[index % len(_COLORS)]
        if len(points) == 1:
            px, py = points[0]
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)
        else:
            draw.line(points, fill=color, width=3, joint="curve")
        legend_y = top + 10 + 25 * index
        draw.line(
            [
                left + plot_width + 18,
                legend_y + 7,
                left + plot_width + 48,
                legend_y + 7,
            ],
            fill=color,
            width=3,
        )
        draw.text(
            (left + plot_width + 56, legend_y),
            labels[index],
            fill="black",
            font=tick_font,
        )

    x_bbox = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(
        (left + plot_width / 2 - (x_bbox[2] - x_bbox[0]) / 2, height - 28),
        x_label,
        fill="black",
        font=label_font,
    )
    shown_y_label = f"log10({y_label})" if log_y else y_label
    y_image = Image.new("RGBA", (320, 28), (255, 255, 255, 0))
    ImageDraw.Draw(y_image).text(
        (2, 2), shown_y_label, fill="black", font=label_font
    )
    y_image = y_image.crop(y_image.getbbox()).rotate(90, expand=True)
    canvas.paste(
        y_image,
        (10, int(top + plot_height / 2 - y_image.height / 2)),
        y_image,
    )
    return np.asarray(canvas, dtype=np.uint8)


def plot_diameter_profile(
    z_m: NDArray[np.floating],
    diameter_m: NDArray[np.floating],
    save_path: str | Path,
) -> None:
    """Plot D(z) in micrometers."""

    image = _line_plot(
        np.asarray(z_m) * 1e6,
        [np.asarray(diameter_m) * 1e6],
        ["D(z)"],
        title="Single-TGV piecewise-linear diameter profile",
        x_label="z [um]",
        y_label="D(z) [um]",
    )
    _save(save_path, image)


def plot_radial_profiles(
    radius_m: NDArray[np.floating],
    profiles_m: Sequence[NDArray[np.floating]],
    labels: Sequence[str],
    save_path: str | Path,
    *,
    title: str = "Projected air path length",
    y_label: str = "Air path length [um]",
) -> None:
    """Plot one or more radial profiles with explicit physical units."""

    image = _line_plot(
        np.asarray(radius_m) * 1e6,
        [np.asarray(profile) * 1e6 for profile in profiles_m],
        labels,
        title=title,
        x_label="r [um]",
        y_label=y_label,
    )
    _save(save_path, image)


def plot_phase_profiles(
    radius_m: NDArray[np.floating],
    phase_profiles_rad: Sequence[NDArray[np.floating]],
    labels: Sequence[str],
    save_path: str | Path,
) -> None:
    """Plot unwrapped projected-phase profiles for a waist sweep."""

    image = _line_plot(
        np.asarray(radius_m) * 1e6,
        phase_profiles_rad,
        labels,
        title="Waist sweep: projected unwrapped phase",
        x_label="r [um]",
        y_label="Unwrapped phase [rad]",
    )
    _save(save_path, image)


def plot_effective_transmission(
    transmission: NDArray[np.complexfloating],
    dx_m: float,
    save_path: str | Path,
) -> None:
    """Plot amplitude and wrapped phase without exaggerating amplitude noise."""

    field = np.asarray(transmission, dtype=np.complex128)
    panels = [
        _annotated_scalar_image(
            np.abs(field),
            cmap="magma",
            title="Effective transmission amplitude",
            colorbar_label="Amplitude [1]",
            dx=dx_m,
            vmin=0.999999999999,
            vmax=1.000000000001,
        ),
        _annotated_scalar_image(
            np.angle(field),
            cmap="twilight",
            title="Effective transmission wrapped phase",
            colorbar_label="Phase [rad]",
            dx=dx_m,
            vmin=-np.pi,
            vmax=np.pi,
        ),
    ]
    _save(save_path, _join_panels(panels, columns=2))


def plot_opd_and_phase(
    opd_relative_m: NDArray[np.floating],
    phase_unwrapped_rad: NDArray[np.floating],
    dx_m: float,
    save_path: str | Path,
) -> None:
    """Plot relative OPD and phase before wrapping."""

    panels = [
        _annotated_scalar_image(
            np.asarray(opd_relative_m) * 1e6,
            cmap="coolwarm",
            title="Relative optical path difference",
            colorbar_label="OPD [um]",
            dx=dx_m,
        ),
        _annotated_scalar_image(
            phase_unwrapped_rad,
            cmap="coolwarm",
            title="Unwrapped projected phase",
            colorbar_label="Phase [rad]",
            dx=dx_m,
        ),
    ]
    _save(save_path, _join_panels(panels, columns=2))


def plot_probe_sensitivity_maps(
    baseline: NDArray[np.complexfloating],
    minus: NDArray[np.complexfloating],
    plus: NDArray[np.complexfloating],
    projected_derivative_scaled: NDArray[np.complexfloating],
    dx_m: float,
    save_path: str | Path,
) -> None:
    """Plot true-probe amplitude, phase, and gauge-projected differences."""

    base = np.asarray(baseline, dtype=np.complex128)
    low = np.asarray(minus, dtype=np.complex128)
    high = np.asarray(plus, dtype=np.complex128)
    derivative = np.asarray(projected_derivative_scaled, dtype=np.complex128)
    panels = [
        _annotated_scalar_image(
            np.abs(base),
            cmap="magma",
            title="Baseline P_B amplitude",
            colorbar_label="Amplitude [a.u.]",
            dx=dx_m,
        ),
        _annotated_scalar_image(
            np.angle(base),
            cmap="twilight",
            title="Baseline P_B wrapped phase",
            colorbar_label="Phase [rad]",
            dx=dx_m,
            vmin=-np.pi,
            vmax=np.pi,
        ),
        _annotated_scalar_image(
            np.abs(high) - np.abs(low),
            cmap="coolwarm",
            title="P_B amplitude: plus minus minus",
            colorbar_label="Amplitude difference [a.u.]",
            dx=dx_m,
        ),
        _annotated_scalar_image(
            np.abs(derivative),
            cmap="viridis",
            title="Gauge-projected D_waist signature",
            colorbar_label="Scaled derivative [a.u.]",
            dx=dx_m,
        ),
    ]
    _save(save_path, _join_panels(panels, columns=2))


def plot_intensity_sensitivity(
    baseline_frame: NDArray[np.floating],
    scaled_derivative_frame: NDArray[np.floating],
    per_frame_sensitivity: NDArray[np.floating],
    dx_m: float,
    save_path: str | Path,
) -> None:
    """Plot one detector frame, its derivative, and all frame sensitivities."""

    derivative = np.asarray(scaled_derivative_frame, dtype=np.float64)
    limit = max(float(np.max(np.abs(derivative))), np.finfo(float).eps)
    frame_values = np.asarray(per_frame_sensitivity, dtype=np.float64)
    panels = [
        _annotated_scalar_image(
            np.asarray(baseline_frame),
            cmap="magma",
            title="Baseline detector intensity",
            colorbar_label="Intensity [a.u.]",
            dx=dx_m,
        ),
        _annotated_scalar_image(
            derivative,
            cmap="coolwarm",
            title="Scaled D_waist intensity derivative",
            colorbar_label="Normalized derivative [a.u.]",
            dx=dx_m,
            vmin=-limit,
            vmax=limit,
        ),
        _line_plot(
            np.arange(len(frame_values), dtype=np.float64),
            [frame_values],
            ["frame sensitivity"],
            title="Per-frame intensity sensitivity",
            x_label="Frame index",
            y_label="Normalized sensitivity [1]",
        ),
    ]
    _save(save_path, _join_panels(panels, columns=3))


def plot_sensitivity_curve(
    waist_m: NDArray[np.floating],
    values: NDArray[np.floating],
    save_path: str | Path,
) -> None:
    """Plot gauge-aligned probe difference over the waist sweep."""

    image = _line_plot(
        np.asarray(waist_m) * 1e6,
        [values],
        ["probe difference"],
        title="Local waist sensitivity curve",
        x_label="D_waist [um]",
        y_label="Gauge-aligned probe relative L2 [1]",
    )
    _save(save_path, image)


def plot_step_convergence(
    steps_m: NDArray[np.floating],
    baseline_probe: NDArray[np.floating],
    baseline_intensity: NDArray[np.floating],
    fine_probe: NDArray[np.floating],
    fine_intensity: NDArray[np.floating],
    save_path: str | Path,
) -> None:
    """Plot finite-difference sensitivity versus decreasing waist step."""

    steps_nm = np.asarray(steps_m, dtype=np.float64) * 1e9
    if np.any(steps_nm <= 0.0):
        msg = "steps_m must be positive."
        raise ValueError(msg)
    image = _line_plot(
        np.log10(steps_nm),
        [
            np.asarray(baseline_probe, dtype=np.float64),
            np.asarray(baseline_intensity, dtype=np.float64),
            np.asarray(fine_probe, dtype=np.float64),
            np.asarray(fine_intensity, dtype=np.float64),
        ],
        [
            "probe: dx=0.25 um",
            "intensity: dx=0.25 um",
            "probe: dx=0.125 um",
            "intensity: dx=0.125 um",
        ],
        title="Finite-difference step convergence",
        x_label="log10(Delta D_waist [nm])",
        y_label="Normalized sensitivity [1]",
        log_y=True,
    )
    _save(save_path, image)


def plot_recovered_probe_cases(
    probes: Sequence[NDArray[np.complexfloating]],
    labels: Sequence[str],
    dx_m: float,
    save_path: str | Path,
) -> None:
    """Plot final Stage D probe amplitude and phase for selected variants."""

    if len(probes) != len(labels) or not probes:
        msg = "probes and labels must be non-empty and have equal length."
        raise ValueError(msg)
    panels: list[NDArray[np.uint8]] = []
    for probe, label in zip(probes, labels, strict=True):
        field = np.asarray(probe, dtype=np.complex128)
        panels.extend(
            [
                _annotated_scalar_image(
                    np.abs(field),
                    cmap="magma",
                    title=f"Stage D final P_B amplitude: {label}",
                    colorbar_label="Amplitude [a.u.]",
                    dx=dx_m,
                ),
                _annotated_scalar_image(
                    np.angle(field),
                    cmap="twilight",
                    title=f"Stage D final P_B phase: {label}",
                    colorbar_label="Phase [rad]",
                    dx=dx_m,
                    vmin=-np.pi,
                    vmax=np.pi,
                ),
            ]
        )
    _save(save_path, _join_panels(panels, columns=2))


def plot_loss_curves(
    loss_curves: Sequence[NDArray[np.floating]],
    labels: Sequence[str],
    save_path: str | Path,
) -> None:
    """Plot Stage D loss, including a final frozen reevaluation point."""

    if len(loss_curves) != len(labels) or not loss_curves:
        msg = "loss_curves and labels must be non-empty and have equal length."
        raise ValueError(msg)
    curves = [np.asarray(curve, dtype=np.float64) for curve in loss_curves]
    if any(curve.ndim != 1 or curve.size == 0 for curve in curves):
        msg = "Every loss curve must be a non-empty 1D array."
        raise ValueError(msg)
    image = _line_plot(
        [
            np.arange(1, curve.size + 1, dtype=np.float64)
            for curve in curves
        ],
        curves,
        labels,
        title="Stage D operator-consistency ablation loss",
        x_label="Iteration; last point is frozen final evaluation",
        y_label="Relative detector-amplitude loss",
        log_y=True,
    )
    _save(save_path, image)


def plot_jacobian_correlation(
    correlation: NDArray[np.floating],
    labels: Sequence[str],
    save_path: str | Path,
) -> None:
    """Plot the normalized Jacobian column correlation matrix."""

    values = np.asarray(correlation, dtype=np.float64)
    if values.shape != (len(labels), len(labels)) or not np.all(np.isfinite(values)):
        msg = "correlation must be a finite square matrix matching labels."
        raise ValueError(msg)
    cell = 92
    top, left, right, bottom = 76, 76, 360, 70
    size = len(labels)
    canvas = Image.new(
        "RGB", (left + size * cell + right, top + size * cell + bottom), "white"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(12)
    value_font = _font(13)
    draw.text(
        (left, 20),
        "Gauge-projected Jacobian column correlation",
        fill="black",
        font=title_font,
    )
    colors = _colormap_uint8(values, "coolwarm", vmin=-1.0, vmax=1.0)
    short_labels = [f"P{index + 1}" for index in range(size)]
    for row in range(size):
        for column in range(size):
            x0 = left + column * cell
            y0 = top + row * cell
            color = tuple(int(channel) for channel in colors[row, column])
            draw.rectangle(
                [x0, y0, x0 + cell, y0 + cell], fill=color, outline="black"
            )
            text = f"{values[row, column]:.3f}"
            bbox = draw.textbbox((0, 0), text, font=value_font)
            draw.text(
                (
                    x0 + cell / 2 - (bbox[2] - bbox[0]) / 2,
                    y0 + cell / 2 - (bbox[3] - bbox[1]) / 2,
                ),
                text,
                fill="white" if abs(values[row, column]) > 0.6 else "black",
                font=value_font,
            )
        draw.text(
            (left - 34, top + row * cell + cell / 2 - 7),
            short_labels[row],
            fill="black",
            font=label_font,
        )
        draw.text(
            (left + row * cell + cell / 2 - 8, top + size * cell + 10),
            short_labels[row],
            fill="black",
            font=label_font,
        )

    legend_x = left + size * cell + 28
    draw.text((legend_x, top), "Parameters", fill="black", font=label_font)
    for index, label in enumerate(labels):
        draw.text(
            (legend_x, top + 30 + index * 28),
            f"P{index + 1}: {label}",
            fill="black",
            font=label_font,
        )
    draw.text(
        (legend_x, top + size * cell - 28),
        "Color scale: -1 to +1",
        fill="black",
        font=label_font,
    )
    _save(save_path, np.asarray(canvas, dtype=np.uint8))


def plot_jacobian_singular_values(
    singular_values: NDArray[np.floating],
    save_path: str | Path,
) -> None:
    """Plot local Jacobian singular values on a logarithmic axis."""

    values = np.asarray(singular_values, dtype=np.float64)
    image = _line_plot(
        np.arange(1, len(values) + 1, dtype=np.float64),
        [values],
        ["singular values"],
        title="Gauge-projected local Jacobian spectrum",
        x_label="Singular-value index",
        y_label="singular value [scaled field units]",
        log_y=True,
    )
    _save(save_path, image)
