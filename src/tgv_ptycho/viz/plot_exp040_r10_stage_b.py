"""Backend-free figures for exp040 R10 Stage-B."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R10_STAGE_B_FIGURE_FILENAMES = (
    "r10_stage_b_reference_controls.png",
    "r10_stage_b_cross_model.png",
    "r10_stage_b_radial_profiles.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _finite_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"R10 Stage-B {name} must be a finite 2D array.")
    return array


def save_exp040_r10_stage_b_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three figures frozen in the Stage-B registration."""

    metrics = _mapping(result.get("metrics"), "metrics")
    comparisons = _mapping(metrics.get("comparisons"), "comparisons")
    reference = _mapping(metrics.get("reference_controls"), "reference controls")
    thresholds = _mapping(metrics.get("thresholds"), "thresholds")
    cases = _mapping(metrics.get("case_controls"), "case controls")
    guard = np.asarray(
        [
            float(_mapping(cases[name], name)["outer_guard_rms_ratio"])
            for name in (
                "coarse_nominal",
                "fine_nominal",
                "fine_enlarged_pml",
            )
        ]
    )
    control_values = np.asarray(
        [
            float(_mapping(comparisons["mesh"], "mesh")["passband_radial_l2"]),
            float(_mapping(comparisons["pml"], "PML")["passband_radial_l2"]),
            float(
                _mapping(comparisons["cross_model"], "cross model")[
                    "passband_radial_l2"
                ]
            ),
            float(reference["multislice_azimuthal_anisotropy_relative_l2"]),
            float(np.max(guard)),
        ],
        dtype=np.float64,
    )
    gate = float(thresholds["reference_passband_relative_l2_max"])
    controls_figure = _line_plot(
        np.arange(control_values.size, dtype=np.float64),
        [control_values, np.full(control_values.size, gate)],
        ["Measured", "5.0% registered gate"],
        title="R10 Stage B: reference and cross-model diagnostics",
        x_label="0 mesh; 1 PML; 2 cross; 3 MS anisotropy; 4 max guard",
        y_label="Relative metric [1]",
        log_y=False,
    )

    maps = _mapping(result.get("selected_maps"), "selected maps")
    dx_m = float(_mapping(metrics.get("sampling"), "sampling")["cartesian_dx_m"])
    cross_panels = [
        _annotated_scalar_image(
            _finite_map(maps.get("helmholtz_passband_amplitude"), "H amplitude"),
            cmap="viridis",
            title="Candidate Helmholtz reference amplitude",
            colorbar_label="|v_H| [1]",
            dx=dx_m,
        ),
        _annotated_scalar_image(
            _finite_map(maps.get("multislice_passband_amplitude"), "MS amplitude"),
            cmap="viridis",
            title="Multislice passband amplitude",
            colorbar_label="|v_MS| [1]",
            dx=dx_m,
        ),
        _annotated_scalar_image(
            _finite_map(maps.get("normalized_cross_residual"), "cross residual"),
            cmap="magma",
            title="Cross-model normalized residual",
            colorbar_label="|Delta v| / max|v_H| [1]",
            dx=dx_m,
            vmin=0.0,
        ),
        _annotated_scalar_image(
            _finite_map(maps.get("cross_phase_difference_rad"), "phase difference"),
            cmap="twilight",
            title="Wrapped cross-model phase difference",
            colorbar_label="arg(v_MS conj(v_H)) [rad]",
            dx=dx_m,
            vmin=-np.pi,
            vmax=np.pi,
        ),
    ]
    cross_figure = _join_panels(cross_panels, columns=2)

    radial = _mapping(result.get("radial_profiles"), "radial profiles")
    radius_um = np.asarray(radial["radius_m"], dtype=np.float64) * 1.0e6
    helmholtz = np.asarray(radial["fine_enlarged_pml_passband"])
    multislice = np.asarray(radial["multislice_fine_1024_passband"])
    amplitude_panel = _line_plot(
        radius_um,
        [np.abs(helmholtz), np.abs(multislice)],
        ["Helmholtz fine enlarged-PML", "Multislice fine"],
        title="R10 Stage B passband radial amplitude",
        x_label="Radius [um]",
        y_label="Normalized amplitude [1]",
        log_y=False,
    )
    phase_panel = _line_plot(
        radius_um,
        [np.angle(helmholtz), np.angle(multislice)],
        ["Helmholtz fine enlarged-PML", "Multislice fine"],
        title="R10 Stage B passband radial phase",
        x_label="Radius [um]",
        y_label="Wrapped phase [rad]",
        log_y=False,
    )
    radial_figure = _join_panels([amplitude_panel, phase_panel], columns=1)

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in EXP040_R10_STAGE_B_FIGURE_FILENAMES]
    for path, image in zip(
        paths,
        (controls_figure, cross_figure, radial_figure),
        strict=True,
    ):
        iio.imwrite(path, image)
    return paths


__all__ = [
    "EXP040_R10_STAGE_B_FIGURE_FILENAMES",
    "save_exp040_r10_stage_b_figures",
]
