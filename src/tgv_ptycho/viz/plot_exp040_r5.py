"""Backend-free plots for exp040 R5 support/boundary diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R5_FIGURE_FILENAMES = (
    "r5_open_boundary_convergence.png",
    "r5_support_boundary_effects.png",
    "r5_detector_comparison.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _nonnegative_series(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != shape
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        raise ValueError(f"{name} is invalid.")
    return array


def save_exp040_r5_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three pre-registered R5 figures."""

    metrics = _mapping(
        _mapping(result.get("metrics"), "result.metrics").get("diagnostics_r5"),
        "diagnostics_r5",
    )
    convergence = _mapping(
        metrics.get("open_boundary_convergence"), "R5 convergence"
    )
    fov_m = _nonnegative_series(
        convergence.get("padding_fov_m"), (3,), "R5 padding FOV"
    )
    if not (
        fov_m[0] > 0.0
        and np.allclose(fov_m / fov_m[0], [1.0, 1.5, 2.0])
    ):
        raise ValueError("R5 padding FOV series is invalid.")
    relative = _nonnegative_series(
        convergence.get("relative_to_384"), (3,), "R5 convergence series"
    )
    threshold = float(
        _mapping(metrics.get("thresholds"), "R5 thresholds")[
            "convergence_and_materiality_relative_l2"
        ]
    )
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("R5 convergence threshold is invalid.")
    fov_um = fov_m * 1.0e6
    keep = relative > 0.0
    convergence_panel = _line_plot(
        [fov_um[keep], fov_um],
        [relative[keep], np.full(3, threshold)],
        ["finite-open I_stack", "5.0% registered gate"],
        title="R5 open-boundary padding convergence (exact zero omitted)",
        x_label="Residual propagation FOV [um]",
        y_label="log10(relative L2 to largest FOV) [1]",
        log_y=True,
    )

    effects = _mapping(metrics.get("effects"), "R5 effects")
    effect_values = _nonnegative_series(
        [
            effects.get("support_relative_l2"),
            effects.get("boundary_relative_l2"),
            effects.get("combined_relative_l2"),
        ],
        (3,),
        "R5 boundary effects",
    )
    effect_x = np.asarray([1.0, 2.0, 3.0])
    effect_panel = _line_plot(
        [effect_x, effect_x],
        [effect_values, np.full(3, threshold)],
        ["registered effects", "5.0% materiality"],
        title="R5 finite-support and open-boundary effects",
        x_label="Effect: 1=support, 2=boundary, 3=combined",
        y_label="Relative L2 [1]",
        log_y=False,
    )

    selected = _mapping(
        _mapping(result.get("diagnostics_r5"), "result.diagnostics_r5").get(
            "selected_scan0"
        ),
        "R5 selected scan",
    )
    names = (
        "periodic_circular_192",
        "finite_circular_192",
        "finite_open_384",
    )
    images = []
    shape: tuple[int, int] | None = None
    for name in names:
        image = np.asarray(selected.get(name), dtype=np.float64)
        if (
            image.ndim != 2
            or not np.all(np.isfinite(image))
            or np.any(image < 0.0)
            or (shape is not None and image.shape != shape)
        ):
            raise ValueError(f"R5 selected image {name} is invalid.")
        shape = image.shape
        images.append(image)
    dx_m = float(_mapping(result.get("baseline"), "result.baseline")["dx_m"])
    image_panels = [
        _annotated_scalar_image(
            image,
            cmap="gray",
            title=title,
            colorbar_label="Intensity [a.u.]",
            dx=dx_m,
            vmin=0.0,
        )
        for image, title in zip(
            images,
            ("periodic/circular", "finite/circular", "finite/open 384 um"),
            strict=True,
        )
    ]
    difference_scale = max(float(np.max(images[-1])), np.finfo(float).eps)
    for image, title in (
        (np.abs(images[0] - images[1]), "|periodic - finite circular|"),
        (np.abs(images[1] - images[2]), "|finite circular - open|"),
    ):
        image_panels.append(
            _annotated_scalar_image(
                image / difference_scale,
                cmap="magma",
                title=title,
                colorbar_label="Difference / max(open) [1]",
                dx=dx_m,
                vmin=0.0,
            )
        )

    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R5_FIGURE_FILENAMES]
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(paths[0], convergence_panel)
    iio.imwrite(paths[1], effect_panel)
    iio.imwrite(paths[2], _join_panels(image_panels, columns=3))
    return paths


__all__ = ["EXP040_R5_FIGURE_FILENAMES", "save_exp040_r5_figures"]
