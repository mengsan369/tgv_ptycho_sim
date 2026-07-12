"""Reconstruction visualizations that do not require an interactive backend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from tgv_ptycho.viz.plot_field import (
    _annotated_scalar_image,
    _font,
    _format_tick,
)


def _join_panels(
    panels: Sequence[NDArray[np.uint8]], *, columns: int, gap: int = 8
) -> NDArray[np.uint8]:
    """Join equally sized RGB panels into a white grid."""

    if not panels or columns <= 0:
        msg = "panels must be non-empty and columns must be positive."
        raise ValueError(msg)
    panel_height = max(panel.shape[0] for panel in panels)
    panel_width = max(panel.shape[1] for panel in panels)
    rows = int(np.ceil(len(panels) / columns))
    canvas = np.full(
        (
            rows * panel_height + (rows - 1) * gap,
            columns * panel_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        y0 = row * (panel_height + gap)
        x0 = column * (panel_width + gap)
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    return canvas


def save_reconstruction_comparison(
    B_true: NDArray[np.complexfloating],
    B_rec: NDArray[np.complexfloating],
    save_path: str | Path,
    *,
    dx: float | tuple[float, float] | None = None,
    mask: NDArray[np.bool_] | None = None,
    field_label: str = "B",
) -> None:
    """Save amplitude and phase truth/reconstruction/error panels."""

    truth = np.asarray(B_true, dtype=np.complex128)
    rec = np.asarray(B_rec, dtype=np.complex128)
    if truth.shape != rec.shape or truth.ndim != 2:
        msg = "B_true and B_rec must be same-shaped 2D arrays."
        raise ValueError(msg)

    amplitude_error = np.abs(rec) - np.abs(truth)
    phase_error = np.angle(np.exp(1j * (np.angle(rec) - np.angle(truth))))
    if mask is None:
        selected = np.ones(truth.shape, dtype=bool)
        amplitude_error_display = amplitude_error
        phase_error_display = phase_error
        error_region = "full field"
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != truth.shape or not np.any(selected):
            msg = "mask must match B_true and select at least one pixel."
            raise ValueError(msg)
        amplitude_error_display = np.where(selected, amplitude_error, np.nan)
        phase_error_display = np.where(selected, phase_error, np.nan)
        error_region = "illuminated"
    amplitude_min = float(min(np.min(np.abs(truth)), np.min(np.abs(rec))))
    amplitude_max = float(max(np.max(np.abs(truth)), np.max(np.abs(rec))))
    amplitude_scale = max(abs(amplitude_min), abs(amplitude_max), 1.0)
    if amplitude_max - amplitude_min < 1e-9 * amplitude_scale:
        padding = 0.05 * amplitude_scale
        amplitude_min -= padding
        amplitude_max += padding
    amplitude_error_limit = max(float(np.max(np.abs(amplitude_error[selected]))), 1e-12)
    phase_error_limit = max(float(np.max(np.abs(phase_error[selected]))), 1e-12)

    panels = [
        _annotated_scalar_image(
            np.abs(truth),
            cmap="magma",
            title=f"{field_label} truth: amplitude",
            colorbar_label="Amplitude [a.u.]",
            dx=dx,
            vmin=amplitude_min,
            vmax=amplitude_max,
        ),
        _annotated_scalar_image(
            np.abs(rec),
            cmap="magma",
            title=f"{field_label} reconstruction: amplitude",
            colorbar_label="Amplitude [a.u.]",
            dx=dx,
            vmin=amplitude_min,
            vmax=amplitude_max,
        ),
        _annotated_scalar_image(
            amplitude_error_display,
            cmap="coolwarm",
            title=f"Amp. error: {error_region}",
            colorbar_label="Error [a.u.]",
            dx=dx,
            vmin=-amplitude_error_limit,
            vmax=amplitude_error_limit,
        ),
        _annotated_scalar_image(
            np.angle(truth),
            cmap="twilight",
            title=f"{field_label} truth: phase",
            colorbar_label="Phase [rad]",
            dx=dx,
            vmin=-np.pi,
            vmax=np.pi,
        ),
        _annotated_scalar_image(
            np.angle(rec),
            cmap="twilight",
            title=f"{field_label} reconstruction: phase",
            colorbar_label="Phase [rad]",
            dx=dx,
            vmin=-np.pi,
            vmax=np.pi,
        ),
        _annotated_scalar_image(
            phase_error_display,
            cmap="coolwarm",
            title=f"Phase error: {error_region}",
            colorbar_label="Error [rad]",
            dx=dx,
            vmin=-phase_error_limit,
            vmax=phase_error_limit,
        ),
    ]
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, _join_panels(panels, columns=3))


def plot_loss_curve(
    loss_curve: NDArray[np.floating],
    save_path: str | Path,
    *,
    title: str = "ePIE data-fidelity loss",
) -> None:
    """Save a log-scale ePIE loss curve as a backend-independent PNG."""

    loss = np.asarray(loss_curve, dtype=np.float64)
    if loss.ndim != 1 or loss.size == 0:
        msg = "loss_curve must be a non-empty 1D array."
        raise ValueError(msg)
    if not np.all(np.isfinite(loss)) or np.any(loss < 0):
        msg = "loss_curve must contain finite, non-negative values."
        raise ValueError(msg)

    width, height = 760, 430
    left, right, top, bottom = 86, 32, 54, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(13)
    tick_font = _font(11)

    safe_loss = np.maximum(loss, np.finfo(float).tiny)
    log_loss = np.log10(safe_loss)
    y_min = float(np.min(log_loss))
    y_max = float(np.max(log_loss))
    if y_max - y_min < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    margin = 0.05 * (y_max - y_min)
    y_min -= margin
    y_max += margin

    draw.text((left, 14), title, fill="black", font=title_font)
    draw.rectangle(
        [left, top, left + plot_width, top + plot_height], outline="black", width=1
    )
    for fraction in np.linspace(0.0, 1.0, 5):
        y = int(top + fraction * plot_height)
        value = y_max - fraction * (y_max - y_min)
        draw.line([left, y, left + plot_width, y], fill=(225, 225, 225), width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (left - 10 - (bbox[2] - bbox[0]), y - 6),
            label,
            fill="black",
            font=tick_font,
        )

    x_ticks = np.unique(np.rint(np.linspace(1, loss.size, 5)).astype(int))
    for iteration in x_ticks:
        fraction = 0.0 if loss.size == 1 else (iteration - 1) / (loss.size - 1)
        x = int(left + fraction * plot_width)
        draw.line([x, top + plot_height, x, top + plot_height + 5], fill="black")
        label = str(iteration)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2, top + plot_height + 10),
            label,
            fill="black",
            font=tick_font,
        )

    points: list[tuple[int, int]] = []
    for index, value in enumerate(log_loss):
        x_fraction = 0.0 if loss.size == 1 else index / (loss.size - 1)
        y_fraction = (y_max - float(value)) / (y_max - y_min)
        points.append(
            (
                int(left + x_fraction * plot_width),
                int(top + y_fraction * plot_height),
            )
        )
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(30, 100, 180))
    else:
        draw.line(points, fill=(30, 100, 180), width=3, joint="curve")

    x_label = "Iteration"
    x_bbox = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(
        (left + plot_width / 2 - (x_bbox[2] - x_bbox[0]) / 2, height - 28),
        x_label,
        fill="black",
        font=label_font,
    )
    y_label = "log10(loss)"
    y_bbox = draw.textbbox((0, 0), y_label, font=label_font)
    label_image = Image.new(
        "RGBA",
        (y_bbox[2] - y_bbox[0] + 8, y_bbox[3] - y_bbox[1] + 8),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(label_image).text((4, 4), y_label, fill="black", font=label_font)
    label_image = label_image.rotate(90, expand=True)
    canvas.paste(
        label_image,
        (8, int(top + plot_height / 2 - label_image.height / 2)),
        label_image,
    )

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, np.asarray(canvas, dtype=np.uint8))


def save_scan_positions(
    scan_positions: NDArray[np.floating],
    save_path: str | Path,
    *,
    title: str = "B scan positions",
) -> None:
    """Save the `(x, y)` scan grid in micrometers with acquisition order."""

    positions = np.asarray(scan_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        msg = "scan_positions must have shape (num_positions, 2)."
        raise ValueError(msg)
    positions_um = positions * 1e6
    width, height = 560, 500
    left, right, top, bottom = 80, 34, 58, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = float(np.min(positions_um[:, 0])), float(np.max(positions_um[:, 0]))
    y_min, y_max = float(np.min(positions_um[:, 1])), float(np.max(positions_um[:, 1]))
    x_padding = max(0.08 * (x_max - x_min), 1.0)
    y_padding = max(0.08 * (y_max - y_min), 1.0)
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    label_font = _font(13)
    tick_font = _font(10)
    draw.text((left, 16), title, fill="black", font=title_font)
    draw.rectangle(
        [left, top, left + plot_width, top + plot_height], outline="black", width=1
    )

    def to_pixel(x_um: float, y_um: float) -> tuple[int, int]:
        x = left + (x_um - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - y_um) / (y_max - y_min) * plot_height
        return int(x), int(y)

    for fraction in np.linspace(0.0, 1.0, 5):
        x_value = x_min + fraction * (x_max - x_min)
        x = int(left + fraction * plot_width)
        draw.line([x, top + plot_height, x, top + plot_height + 5], fill="black")
        label = _format_tick(x_value)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2, top + plot_height + 9),
            label,
            fill="black",
            font=tick_font,
        )
        y_value = y_max - fraction * (y_max - y_min)
        y = int(top + fraction * plot_height)
        draw.line([left - 5, y, left, y], fill="black")
        y_label = _format_tick(y_value)
        y_bbox = draw.textbbox((0, 0), y_label, font=tick_font)
        draw.text(
            (left - 10 - (y_bbox[2] - y_bbox[0]), y - 6),
            y_label,
            fill="black",
            font=tick_font,
        )

    pixels = [to_pixel(x, y) for x, y in positions_um]
    if len(pixels) > 1:
        draw.line(pixels, fill=(170, 185, 200), width=1)
    for index, (x, y) in enumerate(pixels):
        fraction = index / max(len(pixels) - 1, 1)
        color = (
            int(35 + 200 * fraction),
            int(110 - 45 * fraction),
            int(190 - 125 * fraction),
        )
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline="black")

    draw.text(
        (left + plot_width / 2 - 24, height - 27),
        "x [um]",
        fill="black",
        font=label_font,
    )
    draw.text((10, top + plot_height / 2), "y [um]", fill="black", font=label_font)
    draw.text(
        (left + plot_width - 150, 20),
        "color: acquisition order",
        fill="black",
        font=tick_font,
    )
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, np.asarray(canvas, dtype=np.uint8))


def save_diffraction_montage(
    I_stack: NDArray[np.floating],
    save_path: str | Path,
    *,
    dx: float | tuple[float, float] | None = None,
    frame_indices: Sequence[int] | None = None,
) -> None:
    """Save representative detector frames using logarithmic intensity."""

    intensities = np.asarray(I_stack, dtype=np.float64)
    if intensities.ndim != 3 or intensities.shape[0] == 0:
        msg = "I_stack must have shape (num_positions, ny, nx)."
        raise ValueError(msg)
    if frame_indices is None:
        frame_indices = [0, intensities.shape[0] // 2, intensities.shape[0] - 1]
    panels: list[NDArray[np.uint8]] = []
    for index in frame_indices:
        if index < 0 or index >= intensities.shape[0]:
            msg = f"frame index out of range: {index}"
            raise IndexError(msg)
        log_intensity = np.log10(np.maximum(intensities[index], np.finfo(float).tiny))
        panels.append(
            _annotated_scalar_image(
                log_intensity,
                cmap="magma",
                title=f"Detector frame {index}",
                colorbar_label="log10 intensity [a.u.]",
                dx=dx,
            )
        )
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, _join_panels(panels, columns=len(panels)))
