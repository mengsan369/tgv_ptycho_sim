"""Backend-independent visualizations for exp040 multi-slice validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from tgv_ptycho.viz.plot_field import (
    _annotated_scalar_image,
    _colormap_uint8,
    _font,
    _format_colorbar,
    _format_tick,
)
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_FIGURE_FILENAMES = (
    "tgv_geometry_and_index_slices.png",
    "exit_field_multislice.png",
    "projected_limit_comparison.png",
    "dz_convergence.png",
    "lateral_fov_convergence.png",
    "B_plane_probe.png",
    "detector_intensity_baseline.png",
    "detector_visibility.png",
)

_OUTPUT_LABELS = {
    "U_A_exit": "U_A_exit",
    "P_B": "P_B",
    "I_stack": "I_stack",
}


def _save(save_path: str | Path, rgb: NDArray[np.uint8]) -> None:
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.asarray(rgb, dtype=np.uint8))


def _positive_dx(dx: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(dx, tuple):
        if len(dx) != 2:
            msg = "dx tuple must be (dy, dx) in meters."
            raise ValueError(msg)
        dy_m, dx_m = float(dx[0]), float(dx[1])
    else:
        dy_m = dx_m = float(dx)
    if not np.all(np.isfinite([dy_m, dx_m])) or min(dy_m, dx_m) <= 0.0:
        msg = "dx values must be finite and positive."
        raise ValueError(msg)
    return dy_m, dx_m


def _spatial_scalar_image(
    values: NDArray[np.floating],
    *,
    cmap: str,
    title: str,
    colorbar_label: str,
    dx: float | tuple[float, float],
    vmin: float | None = None,
    vmax: float | None = None,
) -> NDArray[np.uint8]:
    """Render a readable spatial panel while preserving its physical extent."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        msg = "spatial panel values must be a non-empty finite 2D array."
        raise ValueError(msg)
    dy_m, dx_m = _positive_dx(dx)
    repeat_y = max(1, int(np.ceil(384 / array.shape[0])))
    repeat_x = max(1, int(np.ceil(384 / array.shape[1])))
    displayed = np.repeat(
        np.repeat(array, repeat_y, axis=0), repeat_x, axis=1
    )
    return _annotated_scalar_image(
        displayed,
        cmap=cmap,
        title=title,
        colorbar_label=colorbar_label,
        dx=(dy_m / repeat_y, dx_m / repeat_x),
        vmin=vmin,
        vmax=vmax,
    )


def _xz_index_panel(
    values: NDArray[np.floating],
    z_m: NDArray[np.floating],
    dx_m: float,
) -> NDArray[np.uint8]:
    """Render an x-z index panel with the actual non-centered z coordinates."""

    array = np.asarray(values, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if array.ndim != 2 or z.shape != (array.shape[0],):
        msg = "x-z values must have one row per z coordinate."
        raise ValueError(msg)
    repeat_y = max(1, int(np.ceil(384 / array.shape[0])))
    repeat_x = max(1, int(np.ceil(384 / array.shape[1])))
    displayed = np.repeat(
        np.repeat(array, repeat_y, axis=0), repeat_x, axis=1
    )
    ny, nx = displayed.shape
    vmin, vmax = float(np.min(array)), float(np.max(array))
    rgb = _colormap_uint8(displayed, "viridis", vmin=vmin, vmax=vmax)

    left, top, bottom = 84, 48, 66
    cbar_gap, cbar_width, right = 18, 18, 168
    width = left + nx + cbar_gap + cbar_width + right
    height = top + ny + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, tick_font = _font(16), _font(12), _font(10)
    x0, y0 = left, top
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (x0, y0))
    draw.rectangle([x0, y0, x0 + nx - 1, y0 + ny - 1], outline="black")
    draw.text(
        (x0, 12),
        "Central x-z refractive-index slice",
        fill="black",
        font=title_font,
    )

    x_extent_um = 0.5 * (array.shape[1] - 1) * dx_m * 1e6
    x_values = np.linspace(-x_extent_um, x_extent_um, 3)
    z_values_um = np.linspace(float(z[0]), float(z[-1]), 3) * 1e6
    for fraction, x_value, z_value in zip(
        np.linspace(0.0, 1.0, 3), x_values, z_values_um, strict=True
    ):
        x_pixel = int(x0 + fraction * (nx - 1))
        y_pixel = int(y0 + fraction * (ny - 1))
        draw.line([x_pixel, y0 + ny, x_pixel, y0 + ny + 5], fill="black")
        x_text = _format_tick(float(x_value))
        x_box = draw.textbbox((0, 0), x_text, font=tick_font)
        draw.text(
            (x_pixel - (x_box[2] - x_box[0]) / 2, y0 + ny + 8),
            x_text,
            fill="black",
            font=tick_font,
        )
        draw.line([x0 - 5, y_pixel, x0, y_pixel], fill="black")
        z_text = _format_tick(float(z_value))
        z_box = draw.textbbox((0, 0), z_text, font=tick_font)
        draw.text(
            (x0 - 10 - (z_box[2] - z_box[0]), y_pixel - 6),
            z_text,
            fill="black",
            font=tick_font,
        )
    draw.text((x0 + nx / 2 - 20, height - 24), "x [um]", font=label_font)
    draw.text((8, y0 + ny / 2 - 8), "z [um]", font=label_font)

    cbar_x = x0 + nx + cbar_gap
    gradient = np.linspace(vmax, vmin, ny, dtype=np.float64)[:, None]
    cbar = _colormap_uint8(
        np.repeat(gradient, cbar_width, axis=1),
        "viridis",
        vmin=vmin,
        vmax=vmax,
    )
    canvas.paste(Image.fromarray(cbar, mode="RGB"), (cbar_x, y0))
    draw.rectangle(
        [cbar_x, y0, cbar_x + cbar_width - 1, y0 + ny - 1],
        outline="black",
    )
    for fraction, value in ((0.0, vmax), (0.5, (vmin + vmax) / 2.0), (1.0, vmin)):
        y_pixel = int(y0 + fraction * (ny - 1))
        draw.text(
            (cbar_x + cbar_width + 8, y_pixel - 6),
            _format_colorbar(float(value)),
            fill="black",
            font=tick_font,
        )
    draw.text(
        (cbar_x, y0 + ny + 10),
        "Refractive index [1]",
        fill="black",
        font=label_font,
    )
    return np.asarray(canvas, dtype=np.uint8)


def _finite_1d(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        msg = f"{name} must be a non-empty finite 1D array."
        raise ValueError(msg)
    return array


def _finite_2d(values: NDArray[np.generic], name: str) -> NDArray[np.generic]:
    array = np.asarray(values)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        msg = f"{name} must be a non-empty finite 2D array."
        raise ValueError(msg)
    return array


def _finite_complex_field(
    values: NDArray[np.complexfloating], name: str
) -> NDArray[np.complex128]:
    return np.asarray(_finite_2d(values, name), dtype=np.complex128)


def _finite_intensity_stack(
    values: NDArray[np.floating], name: str
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[0] == 0
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        msg = f"{name} must be a finite, non-negative (frames, ny, nx) stack."
        raise ValueError(msg)
    return array


def _validated_metric_series(
    x_values: NDArray[np.floating],
    metrics: Mapping[str, NDArray[np.floating]],
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]], list[str]]:
    x = _finite_1d(x_values, "convergence coordinates")
    missing = [name for name in _OUTPUT_LABELS if name not in metrics]
    if missing:
        msg = f"metrics is missing required outputs: {missing}"
        raise ValueError(msg)
    series: list[NDArray[np.float64]] = []
    labels: list[str] = []
    for name, label in _OUTPUT_LABELS.items():
        values = _finite_1d(metrics[name], f"metrics[{name!r}]")
        if values.shape != x.shape or np.any(values < 0.0):
            msg = f"metrics[{name!r}] must match x and be non-negative."
            raise ValueError(msg)
        series.append(values)
        labels.append(label)
    return x, series, labels


def _omit_nonpositive_log_points(
    x_values: NDArray[np.float64],
    series: Sequence[NDArray[np.float64]],
    labels: Sequence[str],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[str]]:
    """Omit exact self-comparison zeros before rendering a logarithmic axis."""

    filtered_x: list[NDArray[np.float64]] = []
    filtered_y: list[NDArray[np.float64]] = []
    filtered_labels: list[str] = []
    for values, label in zip(series, labels, strict=True):
        positive = values > 0.0
        if not np.any(positive):
            continue
        filtered_x.append(x_values[positive])
        filtered_y.append(values[positive])
        filtered_labels.append(label)
    if not filtered_y:
        msg = "convergence plot has no positive non-reference errors to display."
        raise ValueError(msg)
    return filtered_x, filtered_y, filtered_labels


def _field_panels(
    field: NDArray[np.complexfloating],
    dx: float | tuple[float, float],
    *,
    field_name: str,
) -> list[NDArray[np.uint8]]:
    values = _finite_complex_field(field, field_name)
    _positive_dx(dx)
    return [
        _spatial_scalar_image(
            np.abs(values),
            cmap="magma",
            title=f"{field_name}: amplitude",
            colorbar_label="Amplitude [a.u.]",
            dx=dx,
        ),
        _spatial_scalar_image(
            np.angle(values),
            cmap="twilight",
            title=f"{field_name}: wrapped phase",
            colorbar_label="Phase [rad]",
            dx=dx,
            vmin=-np.pi,
            vmax=np.pi,
        ),
        _spatial_scalar_image(
            np.abs(values) ** 2,
            cmap="magma",
            title=f"{field_name}: intensity",
            colorbar_label="Intensity [a.u.]",
            dx=dx,
        ),
    ]


def plot_tgv_geometry_and_index_slices(
    n_volume: NDArray[np.floating],
    z_m: NDArray[np.floating],
    diameter_z_m: NDArray[np.floating],
    dx: float | tuple[float, float],
    save_path: str | Path,
) -> None:
    """Save ``tgv_geometry_and_index_slices.png`` for the baseline volume."""

    volume = np.asarray(n_volume, dtype=np.float64)
    z = _finite_1d(z_m, "z_m")
    diameter = _finite_1d(diameter_z_m, "diameter_z_m")
    dy_m, dx_m = _positive_dx(dx)
    if (
        volume.ndim != 3
        or volume.shape[0] != z.size
        or diameter.shape != z.shape
        or not np.all(np.isfinite(volume))
    ):
        msg = "n_volume must be finite (nz, ny, nx) and match z/D(z)."
        raise ValueError(msg)
    if np.any(np.diff(z) <= 0.0) or np.any(diameter <= 0.0):
        msg = "z_m must increase strictly and diameter_z_m must be positive."
        raise ValueError(msg)

    profile = _line_plot(
        z * 1e6,
        [diameter * 1e6],
        ["D(z)"],
        title="TGV diameter profile at slice centers",
        x_label="z [um]",
        y_label="Diameter [um]",
    )
    center_y = volume.shape[1] // 2
    xz = _xz_index_panel(volume[:, center_y, :], z, dx_m)
    slice_indices = np.unique(
        np.asarray([0, len(z) // 2, len(z) - 1], dtype=np.int64)
    )
    xy_panels = [
        _spatial_scalar_image(
            volume[index],
            cmap="viridis",
            title=f"x-y index at z={z[index] * 1e6:.3g} um",
            colorbar_label="Refractive index [1]",
            dx=(dy_m, dx_m),
        )
        for index in slice_indices
    ]
    _save(save_path, _join_panels([profile, xz, *xy_panels], columns=2))


def plot_exit_field_multislice(
    field: NDArray[np.complexfloating],
    dx: float | tuple[float, float],
    save_path: str | Path,
) -> None:
    """Save A-exit amplitude, phase, and intensity panels."""

    _save(
        save_path,
        _join_panels(
            _field_panels(field, dx, field_name="Sample A exit field"),
            columns=3,
        ),
    )


def plot_projected_limit_comparison(
    phase_screen_product: NDArray[np.complexfloating],
    projected_product: NDArray[np.complexfloating],
    dx: float | tuple[float, float],
    save_path: str | Path,
    *,
    projected_difference: NDArray[np.complexfloating] | None = None,
) -> None:
    """Save discrete phase-screen/projected-product comparison panels."""

    phase_product = _finite_complex_field(
        phase_screen_product, "phase_screen_product"
    )
    projected = _finite_complex_field(projected_product, "projected_product")
    _positive_dx(dx)
    if phase_product.shape != projected.shape:
        msg = "phase_screen_product and projected_product must share one shape."
        raise ValueError(msg)
    if projected_difference is None:
        difference = phase_product - projected
    else:
        difference = _finite_complex_field(
            projected_difference, "projected_difference"
        )
        if difference.shape != phase_product.shape:
            msg = "projected_difference must match the product fields."
            raise ValueError(msg)
    limit = max(float(np.max(np.abs(difference))), np.finfo(float).eps)
    panels = [
        _spatial_scalar_image(
            np.angle(phase_product),
            cmap="twilight",
            title="No-internal-propagation phase-screen product",
            colorbar_label="Wrapped phase [rad]",
            dx=dx,
            vmin=-np.pi,
            vmax=np.pi,
        ),
        _spatial_scalar_image(
            np.angle(projected),
            cmap="twilight",
            title="Discrete projected-phase product",
            colorbar_label="Wrapped phase [rad]",
            dx=dx,
            vmin=-np.pi,
            vmax=np.pi,
        ),
        _spatial_scalar_image(
            np.abs(difference),
            cmap="viridis",
            title="Absolute complex-product difference",
            colorbar_label="Absolute field error [1]",
            dx=dx,
            vmin=0.0,
            vmax=limit,
        ),
        _spatial_scalar_image(
            np.angle(phase_product * np.conj(projected)),
            cmap="coolwarm",
            title="Wrapped phase difference (no alignment)",
            colorbar_label="Phase difference [rad]",
            dx=dx,
            vmin=-np.pi,
            vmax=np.pi,
        ),
    ]
    _save(save_path, _join_panels(panels, columns=2))


def plot_dz_convergence(
    dz_m: NDArray[np.floating],
    relative_l2: Mapping[str, NDArray[np.floating]],
    save_path: str | Path,
) -> None:
    """Save relative-L2 convergence for the registered axial cases."""

    dz, series, labels = _validated_metric_series(dz_m, relative_l2)
    plot_x, plot_y, plot_labels = _omit_nonpositive_log_points(
        dz * 1e6, series, labels
    )
    image = _line_plot(
        plot_x,
        plot_y,
        plot_labels,
        title="Axial convergence (exact-zero reference points omitted)",
        x_label="Comparison-case target dz [um]",
        y_label="Relative L2 error to registered reference [1]",
        log_y=True,
    )
    _save(save_path, image)


def plot_lateral_fov_convergence(
    dx_m: NDArray[np.floating],
    lateral_relative_l2: Mapping[str, NDArray[np.floating]],
    fov_m: NDArray[np.floating],
    fov_relative_l2: Mapping[str, NDArray[np.floating]],
    save_path: str | Path,
) -> None:
    """Save fixed-FOV lateral and fixed-sampling FOV convergence."""

    dx_values, lateral_series, labels = _validated_metric_series(
        dx_m, lateral_relative_l2
    )
    fov_values, fov_series, fov_labels = _validated_metric_series(
        fov_m, fov_relative_l2
    )
    lateral_x, lateral_y, lateral_labels = _omit_nonpositive_log_points(
        dx_values * 1e6, lateral_series, labels
    )
    fov_x, fov_y, fov_plot_labels = _omit_nonpositive_log_points(
        fov_values * 1e6, fov_series, fov_labels
    )
    lateral_plot = _line_plot(
        lateral_x,
        lateral_y,
        lateral_labels,
        title=(
            "Fixed-FOV lateral convergence "
            "(exact-zero reference points omitted)"
        ),
        x_label="Comparison-grid dx [um]",
        y_label="Relative L2 error to registered reference [1]",
        log_y=True,
    )
    fov_plot = _line_plot(
        fov_x,
        fov_y,
        fov_plot_labels,
        title=(
            "Fixed-sampling FOV convergence on common ROI "
            "(exact-zero reference omitted)"
        ),
        x_label="Comparison-case FOV width [um]",
        y_label="Common-ROI relative L2 error to reference [1]",
        log_y=True,
    )
    _save(save_path, _join_panels([lateral_plot, fov_plot], columns=1))


def plot_B_plane_probe(
    field: NDArray[np.complexfloating],
    dx: float | tuple[float, float],
    save_path: str | Path,
) -> None:
    """Save B-plane probe amplitude, phase, and intensity panels."""

    _save(
        save_path,
        _join_panels(
            _field_panels(field, dx, field_name="B-plane probe"),
            columns=3,
        ),
    )


def plot_detector_intensity_baseline(
    intensity_stack: NDArray[np.floating],
    dx: float | tuple[float, float],
    save_path: str | Path,
    *,
    frame_indices: Sequence[int] | None = None,
) -> None:
    """Save representative baseline detector frames on a shared scale."""

    stack = _finite_intensity_stack(intensity_stack, "intensity_stack")
    _positive_dx(dx)
    if frame_indices is None:
        frame_indices = [0, stack.shape[0] // 2, stack.shape[0] - 1]
    if not frame_indices:
        msg = "frame_indices must select at least one frame."
        raise ValueError(msg)
    indices = [int(index) for index in frame_indices]
    if any(index < 0 or index >= stack.shape[0] for index in indices):
        msg = "frame_indices contains an out-of-range frame."
        raise IndexError(msg)
    selected = stack[indices]
    vmin, vmax = float(np.min(selected)), float(np.max(selected))
    panels = [
        _spatial_scalar_image(
            stack[index],
            cmap="magma",
            title=f"Baseline detector frame {index}",
            colorbar_label="Grid-sampled intensity [a.u.]",
            dx=dx,
            vmin=vmin,
            vmax=vmax,
        )
        for index in indices
    ]
    _save(save_path, _join_panels(panels, columns=len(panels)))


def _per_frame_relative_l2(
    values: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    difference_energy = np.sum(
        (values - reference) ** 2, axis=(1, 2), dtype=np.float64
    )
    reference_energy = np.sum(reference**2, axis=(1, 2), dtype=np.float64)
    return np.sqrt(
        difference_energy / np.maximum(reference_energy, np.finfo(float).eps)
    )


def plot_detector_visibility(
    intensity_minus: NDArray[np.floating],
    intensity_baseline: NDArray[np.floating],
    intensity_plus: NDArray[np.floating],
    detector_discretization_floor: float,
    dx: float | tuple[float, float],
    save_path: str | Path,
    *,
    per_frame_minus: NDArray[np.floating] | None = None,
    per_frame_plus: NDArray[np.floating] | None = None,
    most_sensitive_frame: int | None = None,
) -> None:
    """Save waist differences, per-frame changes, floor, and peak frame."""

    minus = _finite_intensity_stack(intensity_minus, "intensity_minus")
    baseline = _finite_intensity_stack(intensity_baseline, "intensity_baseline")
    plus = _finite_intensity_stack(intensity_plus, "intensity_plus")
    _positive_dx(dx)
    if minus.shape != baseline.shape or plus.shape != baseline.shape:
        msg = "minus, baseline, and plus detector stacks must share one shape."
        raise ValueError(msg)
    floor = float(detector_discretization_floor)
    if not np.isfinite(floor) or floor < 0.0:
        msg = "detector_discretization_floor must be finite and non-negative."
        raise ValueError(msg)

    if per_frame_minus is None:
        minus_relative = _per_frame_relative_l2(minus, baseline)
    else:
        minus_relative = _finite_1d(per_frame_minus, "per_frame_minus")
    if per_frame_plus is None:
        plus_relative = _per_frame_relative_l2(plus, baseline)
    else:
        plus_relative = _finite_1d(per_frame_plus, "per_frame_plus")
    if (
        minus_relative.shape != (baseline.shape[0],)
        or plus_relative.shape != (baseline.shape[0],)
        or np.any(minus_relative < 0.0)
        or np.any(plus_relative < 0.0)
    ):
        msg = "per-frame visibility arrays must be non-negative and match frames."
        raise ValueError(msg)
    sensitivity = np.maximum(minus_relative, plus_relative)
    if most_sensitive_frame is None:
        most_sensitive = int(np.argmax(sensitivity))
    else:
        most_sensitive = int(most_sensitive_frame)
        if most_sensitive < 0 or most_sensitive >= baseline.shape[0]:
            msg = "most_sensitive_frame is out of range."
            raise IndexError(msg)
    minus_difference = minus[most_sensitive] - baseline[most_sensitive]
    plus_difference = plus[most_sensitive] - baseline[most_sensitive]
    difference_limit = max(
        float(np.max(np.abs(minus_difference))),
        float(np.max(np.abs(plus_difference))),
        np.finfo(float).eps,
    )
    frame_axis = np.arange(baseline.shape[0], dtype=np.float64)
    curve = _line_plot(
        frame_axis,
        [
            minus_relative,
            plus_relative,
            np.full(baseline.shape[0], floor, dtype=np.float64),
        ],
        ["D_waist minus", "D_waist plus", "detector discretization floor"],
        title=(
            "Waist detector visibility by frame; "
            f"most sensitive frame={most_sensitive}"
        ),
        x_label="Scan-frame index [1]",
        y_label="Relative intensity L2 change [1]",
        log_y=True,
    )
    panels = [
        _spatial_scalar_image(
            baseline[most_sensitive],
            cmap="magma",
            title=f"Baseline intensity: most-sensitive frame {most_sensitive}",
            colorbar_label="Grid-sampled intensity [a.u.]",
            dx=dx,
        ),
        _spatial_scalar_image(
            minus_difference,
            cmap="coolwarm",
            title=f"Waist minus - baseline: frame {most_sensitive}",
            colorbar_label="Intensity difference [a.u.]",
            dx=dx,
            vmin=-difference_limit,
            vmax=difference_limit,
        ),
        _spatial_scalar_image(
            plus_difference,
            cmap="coolwarm",
            title=f"Waist plus - baseline: frame {most_sensitive}",
            colorbar_label="Intensity difference [a.u.]",
            dx=dx,
            vmin=-difference_limit,
            vmax=difference_limit,
        ),
        curve,
    ]
    _save(save_path, _join_panels(panels, columns=2))


def _mapping_value(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        msg = f"{context} is missing required key {key!r}."
        raise ValueError(msg)
    return mapping[key]


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{context} must be a mapping."
        raise ValueError(msg)
    return value


def _visibility_value(
    visibility: Mapping[str, Any], names: Sequence[str], context: str
) -> Any:
    for name in names:
        if name in visibility:
            return visibility[name]
    msg = f"{context} requires one of the keys {list(names)}."
    raise ValueError(msg)


def save_exp040_figures(
    result: dict[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save all eight pre-registered exp040 figures from one runner result.

    The function only renders already-computed fields and metrics. It does not
    derive convergence errors, choose a detector floor, or run any forward
    model. Returned paths follow :data:`EXP040_FIGURE_FILENAMES` order.
    """

    if not isinstance(result, dict):
        msg = "result must be a dictionary."
        raise ValueError(msg)
    baseline = _as_mapping(
        _mapping_value(result, "baseline", "result"), "result['baseline']"
    )
    controls = _as_mapping(
        _mapping_value(result, "controls", "result"), "result['controls']"
    )
    convergence = _as_mapping(
        _mapping_value(result, "convergence", "result"),
        "result['convergence']",
    )
    sweep = _as_mapping(
        _mapping_value(result, "sweep", "result"), "result['sweep']"
    )
    metrics = _as_mapping(
        _mapping_value(result, "metrics", "result"), "result['metrics']"
    )
    visibility = _as_mapping(
        _mapping_value(metrics, "visibility", "result['metrics']"),
        "result['metrics']['visibility']",
    )

    axial = _as_mapping(
        _mapping_value(convergence, "axial", "result['convergence']"),
        "result['convergence']['axial']",
    )
    lateral = _as_mapping(
        _mapping_value(convergence, "lateral", "result['convergence']"),
        "result['convergence']['lateral']",
    )
    fov = _as_mapping(
        _mapping_value(convergence, "fov", "result['convergence']"),
        "result['convergence']['fov']",
    )
    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_FIGURE_FILENAMES]
    dx = _mapping_value(baseline, "dx_m", "result['baseline']")

    plot_tgv_geometry_and_index_slices(
        _mapping_value(baseline, "n_volume", "result['baseline']"),
        _mapping_value(baseline, "z_m", "result['baseline']"),
        _mapping_value(baseline, "diameter_z_m", "result['baseline']"),
        dx,
        paths[0],
    )
    plot_exit_field_multislice(
        _mapping_value(baseline, "U_A_exit", "result['baseline']"),
        dx,
        paths[1],
    )
    plot_projected_limit_comparison(
        _mapping_value(
            controls, "phase_screen_product", "result['controls']"
        ),
        _mapping_value(controls, "projected_phase", "result['controls']"),
        dx,
        paths[2],
        projected_difference=_mapping_value(
            controls, "projected_difference", "result['controls']"
        ),
    )
    plot_dz_convergence(
        _mapping_value(axial, "x_values", "result['convergence']['axial']"),
        axial,
        paths[3],
    )
    plot_lateral_fov_convergence(
        _mapping_value(
            lateral, "x_values", "result['convergence']['lateral']"
        ),
        lateral,
        _mapping_value(fov, "x_values", "result['convergence']['fov']"),
        fov,
        paths[4],
    )
    plot_B_plane_probe(
        _mapping_value(baseline, "P_B", "result['baseline']"),
        dx,
        paths[5],
    )
    plot_detector_intensity_baseline(
        _mapping_value(baseline, "I_stack", "result['baseline']"),
        dx,
        paths[6],
    )

    sweep_stack = np.asarray(
        _mapping_value(sweep, "I_stack", "result['sweep']"), dtype=np.float64
    )
    case_ids = [
        str(value)
        for value in _mapping_value(sweep, "case_ids", "result['sweep']")
    ]
    required_cases = ["waist_minus", "baseline", "waist_plus"]
    if (
        sweep_stack.ndim != 4
        or sweep_stack.shape[0] != len(case_ids)
        or any(case_id not in case_ids for case_id in required_cases)
    ):
        msg = (
            "sweep I_stack must have (cases, frames, ny, nx) and include "
            "waist_minus, baseline, and waist_plus case_ids."
        )
        raise ValueError(msg)
    selected = [sweep_stack[case_ids.index(case_id)] for case_id in required_cases]
    plot_detector_visibility(
        selected[0],
        selected[1],
        selected[2],
        float(
            _visibility_value(
                visibility,
                ["floor", "detector_discretization_floor"],
                "visibility floor",
            )
        ),
        dx,
        paths[7],
        per_frame_minus=_visibility_value(
            visibility,
            ["per_frame_minus", "waist_minus_per_frame_relative_l2"],
            "minus per-frame visibility",
        ),
        per_frame_plus=_visibility_value(
            visibility,
            ["per_frame_plus", "waist_plus_per_frame_relative_l2"],
            "plus per-frame visibility",
        ),
        most_sensitive_frame=int(
            _visibility_value(
                visibility,
                ["most_sensitive_frame", "most_sensitive_frame_index"],
                "most-sensitive frame",
            )
        ),
    )
    return paths


__all__ = [
    "EXP040_FIGURE_FILENAMES",
    "plot_tgv_geometry_and_index_slices",
    "plot_exit_field_multislice",
    "plot_projected_limit_comparison",
    "plot_dz_convergence",
    "plot_lateral_fov_convergence",
    "plot_B_plane_probe",
    "plot_detector_intensity_baseline",
    "plot_detector_visibility",
    "save_exp040_figures",
]
