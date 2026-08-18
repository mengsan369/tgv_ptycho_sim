"""Backend-free plots for exp040 R6 sample-B support sensitivity."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R6_FIGURE_FILENAMES = (
    "r6_b_support_effect_matrix.png",
    "r6_b_support_nominal_difference.png",
    "r6_b_support_selected_detector.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _finite_nonnegative(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != shape
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        raise ValueError(f"{name} is invalid.")
    return array


def save_exp040_r6_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three pre-registered R6 figures."""

    metrics = _mapping(
        _mapping(result.get("metrics"), "result.metrics").get("diagnostics_r6"),
        "diagnostics_r6",
    )
    family = _mapping(metrics.get("support_family"), "R6 support family")
    widths_m = _finite_nonnegative(
        family.get("support_width_m"), (3,), "R6 support widths"
    )
    tapers_m = _finite_nonnegative(
        family.get("edge_taper_width_m"), (3,), "R6 taper widths"
    )
    nominal_width = float(family["nominal_support_width_m"])
    if not np.allclose(
        widths_m / nominal_width, [5.0 / 6.0, 1.0, 7.0 / 6.0]
    ):
        raise ValueError("R6 support-width ratios are invalid.")
    threshold = float(
        _mapping(metrics.get("thresholds"), "R6 thresholds")[
            "materiality_relative_l2"
        ]
    )
    effects = _mapping(metrics.get("support_effects"), "R6 support effects")
    effect_matrix = _finite_nonnegative(
        effects.get("relative_l2_matrix"), (3, 3), "R6 effect matrix"
    )
    widths_um = widths_m * 1.0e6
    taper_labels = [f"taper {value * 1.0e6:.0f} um" for value in tapers_m]
    effect_panel = _line_plot(
        [widths_um, widths_um, widths_um, widths_um],
        [
            effect_matrix[:, 0],
            effect_matrix[:, 1],
            effect_matrix[:, 2],
            np.full(3, threshold),
        ],
        [*taper_labels, "5.0% materiality"],
        title="R6 periodic-vs-finite B support effect",
        x_label="Finite coded support width [um]",
        y_label="Relative L2; finite case denominator [1]",
        log_y=False,
    )

    nominal = _mapping(
        metrics.get("nominal_sensitivity"), "R6 nominal sensitivity"
    )
    nominal_matrix = _finite_nonnegative(
        nominal.get("relative_l2_matrix"),
        (3, 3),
        "R6 nominal-difference matrix",
    )
    nominal_panel = _line_plot(
        [widths_um, widths_um, widths_um, widths_um],
        [
            nominal_matrix[:, 0],
            nominal_matrix[:, 1],
            nominal_matrix[:, 2],
            np.full(3, threshold),
        ],
        [*taper_labels, "5.0% reference"],
        title="R6 detector sensitivity to nominal 96 um hard edge",
        x_label="Finite coded support width [um]",
        y_label="Relative L2; nominal denominator [1]",
        log_y=False,
    )

    selected = _mapping(
        _mapping(result.get("diagnostics_r6"), "result.diagnostics_r6").get(
            "selected_scan0"
        ),
        "R6 selected scan",
    )
    selected_cases = _mapping(metrics.get("selected_cases"), "R6 selected cases")
    minimum = _mapping(selected_cases.get("minimum_effect"), "R6 minimum case")
    maximum = _mapping(selected_cases.get("maximum_effect"), "R6 maximum case")
    titles = (
        "periodic comparator",
        "nominal 96 um / hard edge",
        (
            f"minimum: {minimum['support_width_m'] * 1e6:.0f} um / "
            f"{minimum['edge_taper_width_m'] * 1e6:.0f} um taper"
        ),
        (
            f"maximum: {maximum['support_width_m'] * 1e6:.0f} um / "
            f"{maximum['edge_taper_width_m'] * 1e6:.0f} um taper"
        ),
    )
    keys = ("periodic", "nominal", "minimum_effect", "maximum_effect")
    images: list[np.ndarray] = []
    image_shape: tuple[int, int] | None = None
    for key in keys:
        image = np.asarray(selected.get(key), dtype=np.float64)
        if (
            image.ndim != 2
            or not np.all(np.isfinite(image))
            or np.any(image < 0.0)
            or (image_shape is not None and image.shape != image_shape)
        ):
            raise ValueError(f"R6 selected detector image {key} is invalid.")
        image_shape = image.shape
        images.append(image)
    dx_m = float(_mapping(result.get("baseline"), "result.baseline")["dx_m"])
    panels = [
        _annotated_scalar_image(
            image,
            cmap="gray",
            title=title,
            colorbar_label="Intensity [a.u.]",
            dx=dx_m,
            vmin=0.0,
        )
        for image, title in zip(images, titles, strict=True)
    ]

    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R6_FIGURE_FILENAMES]
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(paths[0], effect_panel)
    iio.imwrite(paths[1], nominal_panel)
    iio.imwrite(paths[2], _join_panels(panels, columns=2))
    return paths


__all__ = ["EXP040_R6_FIGURE_FILENAMES", "save_exp040_r6_figures"]
