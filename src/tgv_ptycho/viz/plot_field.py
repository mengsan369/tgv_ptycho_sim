"""Complex field visualization."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


def _colormap_uint8(
    values: NDArray[np.floating],
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> NDArray[np.uint8]:
    arr = np.asarray(values, dtype=np.float64)
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    denom = max(vmax - vmin, np.finfo(float).eps)
    finite = np.isfinite(arr)
    normalized = np.where(finite, np.clip((arr - vmin) / denom, 0.0, 1.0), 0.0)
    rgba = matplotlib.colormaps[cmap](normalized)
    rgb = np.rint(rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[~finite] = np.array([210, 210, 210], dtype=np.uint8)
    return rgb


def _normalize_dx(dx: float | tuple[float, float] | None) -> tuple[float, float] | None:
    if dx is None:
        return None
    if isinstance(dx, tuple):
        if len(dx) != 2:
            msg = "dx tuple must be (dy, dx) in meters."
            raise ValueError(msg)
        return float(dx[0]), float(dx[1])
    value = float(dx)
    return value, value


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2g}"


def _format_colorbar(value: float) -> str:
    if value == 0:
        return "0"
    if 1e-2 <= abs(value) < 1e3:
        return f"{value:.3g}"
    return f"{value:.2e}"


def _paste_array(
    canvas: Image.Image, rgb: NDArray[np.uint8], xy: tuple[int, int]
) -> None:
    canvas.paste(Image.fromarray(rgb, mode="RGB"), xy)


def _annotated_scalar_image(
    values: NDArray[np.floating],
    *,
    cmap: str,
    title: str,
    colorbar_label: str,
    dx: float | tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> NDArray[np.uint8]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        msg = "values must be a 2D array."
        raise ValueError(msg)

    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))

    image_rgb = _colormap_uint8(arr, cmap, vmin=vmin, vmax=vmax)
    ny, nx = arr.shape
    left, top, bottom = 78, 48, 64
    cbar_gap, cbar_width, right = 18, 18, 160
    width = left + nx + cbar_gap + cbar_width + right
    height = top + ny + bottom

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(16)
    label_font = _font(12)
    tick_font = _font(10)

    x0, y0 = left, top
    _paste_array(canvas, image_rgb, (x0, y0))
    draw.rectangle([x0, y0, x0 + nx - 1, y0 + ny - 1], outline="black", width=1)
    draw.text((x0, 12), title, fill="black", font=title_font)

    dx_pair = _normalize_dx(dx)
    if dx_pair is None:
        x_values = np.array([0, (nx - 1) / 2, nx - 1], dtype=np.float64)
        y_values = np.array([0, (ny - 1) / 2, ny - 1], dtype=np.float64)
        x_label = "x [px]"
        y_label = "y [px]"
    else:
        dy_m, dx_m = dx_pair
        x_values = (np.array([0, (nx - 1) / 2, nx - 1]) - (nx - 1) / 2) * dx_m * 1e6
        y_values = (np.array([0, (ny - 1) / 2, ny - 1]) - (ny - 1) / 2) * dy_m * 1e6
        x_label = "x [um]"
        y_label = "y [um]"

    x_pixels = [x0, x0 + (nx - 1) // 2, x0 + nx - 1]
    y_pixels = [y0, y0 + (ny - 1) // 2, y0 + ny - 1]
    for pix, value in zip(x_pixels, x_values, strict=True):
        draw.line([pix, y0 + ny, pix, y0 + ny + 5], fill="black")
        text = _format_tick(float(value))
        bbox = draw.textbbox((0, 0), text, font=tick_font)
        draw.text(
            (pix - (bbox[2] - bbox[0]) / 2, y0 + ny + 8),
            text,
            fill="black",
            font=tick_font,
        )
    for pix, value in zip(y_pixels, y_values, strict=True):
        draw.line([x0 - 5, pix, x0, pix], fill="black")
        text = _format_tick(float(value))
        bbox = draw.textbbox((0, 0), text, font=tick_font)
        draw.text(
            (x0 - 10 - (bbox[2] - bbox[0]), pix - 6), text, fill="black", font=tick_font
        )

    x_bbox = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(
        (x0 + nx / 2 - (x_bbox[2] - x_bbox[0]) / 2, height - 24),
        x_label,
        fill="black",
        font=label_font,
    )
    draw.text((8, y0 + ny / 2 - 8), y_label, fill="black", font=label_font)

    cbar_x = x0 + nx + cbar_gap
    grad = np.linspace(vmax, vmin, ny, dtype=np.float64)[:, None]
    cbar_rgb = _colormap_uint8(
        np.repeat(grad, cbar_width, axis=1), cmap, vmin=vmin, vmax=vmax
    )
    _paste_array(canvas, cbar_rgb, (cbar_x, y0))
    draw.rectangle(
        [cbar_x, y0, cbar_x + cbar_width - 1, y0 + ny - 1], outline="black", width=1
    )

    for frac, value in ((0.0, vmax), (0.5, (vmin + vmax) / 2.0), (1.0, vmin)):
        py = int(y0 + frac * (ny - 1))
        draw.line([cbar_x + cbar_width, py, cbar_x + cbar_width + 5, py], fill="black")
        draw.text(
            (cbar_x + cbar_width + 8, py - 6),
            _format_colorbar(float(value)),
            fill="black",
            font=tick_font,
        )
    draw.text((cbar_x, y0 + ny + 10), colorbar_label, fill="black", font=label_font)

    return np.asarray(canvas, dtype=np.uint8)


def plot_complex_field(
    U: NDArray[np.complexfloating],
    save_path: str | Path | None = None,
    title: str | None = None,
    dx: float | tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Plot amplitude and wrapped phase of a complex field."""

    field = np.asarray(U)
    amplitude = np.abs(field)
    phase = np.angle(field)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        amp_title = "Amplitude" if title is None else f"{title}: amplitude"
        phase_title = "Phase" if title is None else f"{title}: phase"
        amp_rgb = _annotated_scalar_image(
            amplitude,
            cmap="magma",
            title=amp_title,
            colorbar_label="Amplitude [a.u.]",
            dx=dx,
        )
        phase_rgb = _annotated_scalar_image(
            phase,
            cmap="twilight",
            title=phase_title,
            colorbar_label="Phase [rad]",
            dx=dx,
            vmin=-np.pi,
            vmax=np.pi,
        )
        gap = np.full((amp_rgb.shape[0], 8, 3), 255, dtype=np.uint8)
        iio.imwrite(output_path, np.concatenate([amp_rgb, gap, phase_rgb], axis=1))
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    if title:
        fig.suptitle(title)
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.08, top=0.86, wspace=0.25)

    amp_im = axes[0].imshow(amplitude, cmap="magma")
    axes[0].set_title("Amplitude")
    axes[0].set_axis_off()
    fig.colorbar(amp_im, ax=axes[0], fraction=0.046, pad=0.04)

    phase_im = axes[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase")
    axes[1].set_axis_off()
    fig.colorbar(phase_im, ax=axes[1], fraction=0.046, pad=0.04)

    return fig


def save_intensity_image(
    intensity: NDArray[np.floating],
    save_path: str | Path,
    cmap: str = "magma",
    dx: float | tuple[float, float] | None = None,
    title: str = "Detector intensity",
) -> None:
    """Save a single intensity map as an image."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = _annotated_scalar_image(
        np.asarray(intensity),
        cmap=cmap,
        title=title,
        colorbar_label="Intensity [a.u.]",
        dx=dx,
    )
    iio.imwrite(output_path, annotated)
