"""Backend-free plots for exp040 R8 unified convergence and visibility."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R8_FIGURE_FILENAMES = (
    "r8_unified_convergence.png",
    "r8_waist_visibility.png",
    "r8_selected_detector.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def save_exp040_r8_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three pre-registered R8 figures."""

    baseline = _mapping(result.get("baseline"), "result.baseline")
    diagnostics = _mapping(result.get("diagnostics_r8"), "diagnostics_r8")
    metrics = _mapping(
        _mapping(result.get("metrics"), "result.metrics").get("diagnostics_r8"),
        "R8 metrics",
    )
    convergence = _mapping(metrics.get("convergence"), "R8 convergence")
    axial = _mapping(convergence.get("axial"), "R8 axial")
    lateral = _mapping(convergence.get("lateral"), "R8 lateral")
    open_metrics = _mapping(convergence.get("open"), "R8 open")
    threshold = float(
        _mapping(metrics.get("thresholds"), "R8 thresholds")[
            "convergence_relative_l2_max"
        ]
    )
    convergence_panel = _line_plot(
        [
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 1.0, 2.0]),
        ],
        [
            np.asarray(
                [
                    axial["acceptance"]["U_A_exit"],
                    lateral["acceptance"]["U_A_exit"],
                ]
            ),
            np.asarray(
                [
                    axial["acceptance"]["P_B"],
                    lateral["acceptance"]["P_B"],
                ]
            ),
            np.asarray(
                [
                    axial["acceptance"]["I_stack"],
                    lateral["acceptance"]["I_stack"],
                    open_metrics["I_stack"],
                ]
            ),
            np.full(3, threshold),
        ],
        ["U_A_exit", "P_B", "I_stack", "5.0% gate"],
        title="R8 unified q8 convergence: axial / lateral / open",
        x_label="Component index: 0 axial, 1 lateral, 2 open",
        y_label="Unaligned relative L2 [1]",
        log_y=False,
    )

    visibility = _mapping(metrics.get("visibility"), "R8 visibility")
    signals = _mapping(visibility.get("signals"), "R8 signals")
    floor = float(visibility["numerical_floor"]["I_stack"])
    visibility_gate = float(
        _mapping(metrics.get("thresholds"), "R8 thresholds")[
            "detector_visibility_signal_to_floor_min"
        ]
    )
    x_signal = np.asarray([-1.0, 1.0])
    detector_signals = np.asarray(
        [
            signals["waist_minus"]["I_stack"],
            signals["waist_plus"]["I_stack"],
        ]
    )
    visibility_panel = _line_plot(
        [x_signal, x_signal, x_signal],
        [
            detector_signals,
            np.full(2, floor),
            np.full(2, visibility_gate * floor),
        ],
        ["Detector waist signal", "Numerical floor", "3 x floor gate"],
        title="R8 detector waist visibility on the finest unified forward",
        x_label="Waist perturbation sign (-1: 18 um, +1: 22 um)",
        y_label="Full-stack relative L2 [1]",
        log_y=False,
    )

    selected = _mapping(diagnostics.get("selected_scan0"), "R8 selected scan")
    images = {
        name: np.asarray(selected.get(name), dtype=np.float64)
        for name in ("waist_minus", "finest_baseline", "waist_plus")
    }
    for name, image in images.items():
        if image.ndim != 2 or not np.all(np.isfinite(image)) or np.any(image < 0):
            raise ValueError(f"R8 {name} selected detector is invalid.")
    baseline_image = images["finest_baseline"]
    denominator = max(float(np.max(baseline_image)), np.finfo(float).eps)
    differences = {
        "waist_minus_difference": np.abs(images["waist_minus"] - baseline_image)
        / denominator,
        "waist_plus_difference": np.abs(images["waist_plus"] - baseline_image)
        / denominator,
    }
    detector_dx = float(baseline["dx_m"])
    detector_panels = [
        _annotated_scalar_image(
            images[name],
            cmap="gray",
            title=title,
            colorbar_label="Intensity [a.u.]",
            dx=detector_dx,
            vmin=0.0,
        )
        for name, title in (
            ("waist_minus", "D_waist = 18 um / scan 0"),
            ("finest_baseline", "D_waist = 20 um / scan 0"),
            ("waist_plus", "D_waist = 22 um / scan 0"),
        )
    ]
    detector_panels.extend(
        _annotated_scalar_image(
            differences[name],
            cmap="magma",
            title=title,
            colorbar_label="|Delta I| / max(I_20) [1]",
            dx=detector_dx,
            vmin=0.0,
        )
        for name, title in (
            ("waist_minus_difference", "|I_18 - I_20| / max(I_20)"),
            ("waist_plus_difference", "|I_22 - I_20| / max(I_20)"),
        )
    )

    output_dir = Path(figures_dir)
    paths = [output_dir / name for name in EXP040_R8_FIGURE_FILENAMES]
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(paths[0], convergence_panel)
    iio.imwrite(paths[1], visibility_panel)
    iio.imwrite(paths[2], _join_panels(detector_panels, columns=3))
    return paths


__all__ = ["EXP040_R8_FIGURE_FILENAMES", "save_exp040_r8_figures"]
