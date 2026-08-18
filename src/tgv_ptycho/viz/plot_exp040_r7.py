"""Backend-free plots for exp040 R7 subvoxel TGV interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R7_FIGURE_FILENAMES = (
    "r7_interface_fraction_slice.png",
    "r7_interface_convergence.png",
    "r7_interface_selected_detector.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def save_exp040_r7_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three pre-registered R7 figures."""

    baseline = _mapping(result.get("baseline"), "result.baseline")
    diagnostics = _mapping(result.get("diagnostics_r7"), "diagnostics_r7")
    metrics = _mapping(
        _mapping(result.get("metrics"), "result.metrics").get("diagnostics_r7"),
        "R7 metrics",
    )
    factors = np.asarray(
        _mapping(metrics.get("sampling"), "R7 sampling")["interface_factors"],
        dtype=np.int64,
    )
    if not np.array_equal(factors, [1, 2, 4, 8]):
        raise ValueError("R7 interface factors are invalid.")
    dx_m = float(baseline["dx_m"])

    fractions = _mapping(
        diagnostics.get("selected_fractions"), "R7 selected fractions"
    )
    fraction_panels = []
    for factor in factors:
        image = np.asarray(fractions.get(f"q{factor}"), dtype=np.float64)
        if (
            image.ndim != 2
            or not np.all(np.isfinite(image))
            or np.any(image < 0.0)
            or np.any(image > 1.0)
        ):
            raise ValueError(f"R7 q{factor} fraction slice is invalid.")
        fraction_panels.append(
            _annotated_scalar_image(
                image,
                cmap="gray",
                title=f"q{factor} air fraction",
                colorbar_label="Air area fraction [1]",
                dx=float(metrics["sampling"]["sample_a_dx_m"]),
                vmin=0.0,
                vmax=1.0,
            )
        )

    convergence = _mapping(metrics.get("convergence"), "R7 convergence")
    series = _mapping(convergence.get("relative_to_q8"), "R7 series")
    threshold = float(
        _mapping(metrics.get("thresholds"), "R7 thresholds")[
            "convergence_and_materiality_relative_l2"
        ]
    )
    convergence_panel = _line_plot(
        [factors, factors, factors, factors],
        [
            np.asarray(series["U_A_exit"], dtype=np.float64),
            np.asarray(series["P_B"], dtype=np.float64),
            np.asarray(series["I_stack"], dtype=np.float64),
            np.full(len(factors), threshold),
        ],
        ["U_A_exit", "P_B", "I_stack", "5.0% gate"],
        title="R7 subvoxel-interface convergence to q8",
        x_label="Lateral interface factor q",
        y_label="Unaligned relative L2; q8 denominator [1]",
        log_y=False,
    )

    selected = _mapping(diagnostics.get("selected_scan0"), "R7 selected scan")
    detector_panels = []
    for factor in factors:
        image = np.asarray(selected.get(f"q{factor}"), dtype=np.float64)
        if image.ndim != 2 or not np.all(np.isfinite(image)) or np.any(image < 0):
            raise ValueError(f"R7 q{factor} selected detector is invalid.")
        detector_panels.append(
            _annotated_scalar_image(
                image,
                cmap="gray",
                title=f"q{factor} interface / scan 0",
                colorbar_label="Intensity [a.u.]",
                dx=dx_m,
                vmin=0.0,
            )
        )

    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R7_FIGURE_FILENAMES]
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(paths[0], _join_panels(fraction_panels, columns=2))
    iio.imwrite(paths[1], convergence_panel)
    iio.imwrite(paths[2], _join_panels(detector_panels, columns=2))
    return paths


__all__ = ["EXP040_R7_FIGURE_FILENAMES", "save_exp040_r7_figures"]
