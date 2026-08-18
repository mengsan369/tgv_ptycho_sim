"""Backend-free figures for exp040 R11 reference-closure diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tgv_ptycho.viz.plot_field import _annotated_scalar_image
from tgv_ptycho.viz.plot_recon import _join_panels
from tgv_ptycho.viz.plot_tgv import _line_plot

EXP040_R11_FIGURE_FILENAMES = (
    "r11_domain_mesh_controls.png",
    "r11_anisotropy_attribution.png",
    "r11_conditional_cross_model.png",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"R11 {name} must be a mapping.")
    return value


def _finite_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"R11 {name} must be a finite 2D array.")
    return array


def save_exp040_r11_figures(
    result: Mapping[str, Any], figures_dir: str | Path
) -> list[Path]:
    """Save the three figures fixed by the R11 registration."""

    metrics = _mapping(result.get("metrics"), "metrics")
    comparisons = _mapping(metrics.get("comparisons"), "comparisons")
    gates = _mapping(metrics.get("gates"), "gates")
    thresholds = _mapping(metrics.get("thresholds"), "thresholds")
    domain = _mapping(comparisons.get("domain"), "domain comparisons")
    mesh = _mapping(comparisons.get("mesh"), "mesh comparisons")
    anisotropy = _mapping(metrics.get("anisotropy"), "anisotropy")

    control_values = np.asarray(
        [
            float(
                _mapping(domain["core36_to_core48"], "domain gate")[
                    "passband_radial_l2"
                ]
            ),
            float(domain["core48_outer_guard_rms_ratio"]),
            float(_mapping(mesh["adc5"], "ADC5 mesh")["passband_radial_l2"]),
            float(
                _mapping(mesh["standard_report_only"], "standard mesh")[
                    "passband_radial_l2"
                ]
            ),
            float(anisotropy["chord_lateral_passband_relative_l2"]),
            float(anisotropy["maximum_formal_polar_angular_relative_l2"]),
        ],
        dtype=np.float64,
    )
    gate = float(thresholds["domain_passband_relative_l2_max"])
    controls_image = _line_plot(
        np.arange(control_values.size, dtype=np.float64),
        [control_values, np.full(control_values.size, gate)],
        ["Measured", "5% registered gate"],
        title="R11 physical-domain, mesh, and Cartesian controls",
        x_label="0 domain; 1 guard; 2 ADC mesh; 3 std mesh; 4 grid; 5 angular",
        y_label="Relative metric [1]",
        log_y=False,
    )

    polar = _mapping(anisotropy.get("polar_controls"), "polar controls")
    polar_names = (
        "q8_native_1024",
        "q8_restricted_512",
        "chord512",
        "chord1024_native",
        "chord1024_restricted",
    )
    polar_values = np.asarray(
        [
            float(_mapping(polar[name], name)["angular_relative_l2"])
            for name in polar_names
        ],
        dtype=np.float64,
    )
    polar_panel = _line_plot(
        np.arange(polar_values.size, dtype=np.float64),
        [polar_values, np.full(polar_values.size, gate)],
        ["Fixed-radius polar residual", "5% registered gate"],
        title=(
            "R11 angular attribution: q8 native/restricted, chord 512/native/restricted"
        ),
        x_label="Registered field index",
        y_label="Angular relative L2 [1]",
        log_y=False,
    )
    maps = _mapping(result.get("selected_maps"), "selected maps")
    q8_delta = _annotated_scalar_image(
        _finite_map(maps.get("q8_vs_chord_normalized_residual"), "q8 residual"),
        cmap="magma",
        title="q8 versus chord-cell passband residual",
        colorbar_label="|Delta v| / max|v_chord| [1]",
        dx=6.25e-8,
        vmin=0.0,
    )
    chord_amplitude = _annotated_scalar_image(
        _finite_map(maps.get("chord1024_passband_amplitude"), "chord amplitude"),
        cmap="viridis",
        title="Chord-cell 1024 passband amplitude",
        colorbar_label="|v| [1]",
        dx=6.25e-8,
    )
    anisotropy_image = _join_panels(
        [polar_panel, q8_delta, chord_amplitude], columns=1
    )

    conditional = _mapping(metrics.get("conditional_cross_model"), "cross model")
    if bool(conditional.get("executed")):
        cross_panels = [
            _annotated_scalar_image(
                _finite_map(maps.get("helmholtz_passband_amplitude"), "H amplitude"),
                cmap="viridis",
                title="Validated ADC5 Helmholtz reference amplitude",
                colorbar_label="|v_H| [1]",
                dx=1.25e-7,
            ),
            _annotated_scalar_image(
                _finite_map(maps.get("multislice_passband_amplitude"), "MS amplitude"),
                cmap="viridis",
                title="Validated chord-cell multislice amplitude",
                colorbar_label="|v_MS| [1]",
                dx=1.25e-7,
            ),
            _annotated_scalar_image(
                _finite_map(maps.get("normalized_cross_residual"), "cross residual"),
                cmap="magma",
                title="Conditional cross-model normalized residual",
                colorbar_label="|Delta v| / max|v_H| [1]",
                dx=1.25e-7,
                vmin=0.0,
            ),
            _annotated_scalar_image(
                _finite_map(maps.get("cross_phase_difference_rad"), "cross phase"),
                cmap="twilight",
                title="Conditional cross-model phase difference",
                colorbar_label="arg(v_MS conj(v_H)) [rad]",
                dx=1.25e-7,
                vmin=-np.pi,
                vmax=np.pi,
            ),
        ]
        cross_image = _join_panels(cross_panels, columns=2)
    else:
        gate_values = np.asarray(
            [
                float(bool(gates["domain_gate_pass"])),
                float(bool(gates["adc5_mesh_gate_pass"])),
                float(bool(gates["cartesian_anisotropy_gate_pass"])),
            ]
        )
        cross_image = _line_plot(
            np.arange(3, dtype=np.float64),
            [gate_values, np.ones(3, dtype=np.float64)],
            ["Gate state", "Required"],
            title="R11 cross-model skipped by pre-registered reference gates",
            x_label="0 physical domain; 1 ADC5 mesh; 2 Cartesian anisotropy",
            y_label="Boolean gate [0/1]",
            log_y=False,
        )

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in EXP040_R11_FIGURE_FILENAMES]
    for path, image in zip(
        paths,
        (controls_image, anisotropy_image, cross_image),
        strict=True,
    ):
        iio.imwrite(path, image)
    return paths


__all__ = ["EXP040_R11_FIGURE_FILENAMES", "save_exp040_r11_figures"]
