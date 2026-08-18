from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import tgv_ptycho.viz.plot_exp040 as plot_module
from tgv_ptycho.viz.plot_exp040 import (
    EXP040_FIGURE_FILENAMES,
    plot_detector_visibility,
    plot_projected_limit_comparison,
    plot_tgv_geometry_and_index_slices,
    save_exp040_figures,
)


def _exp040_result() -> dict[str, Any]:
    nz, ny, nx, num_frames = 6, 16, 18, 5
    y, x = np.indices((ny, nx), dtype=np.float64)
    phase = 0.04 * x + 0.02 * y
    exit_field = np.exp(1j * phase).astype(np.complex128)
    probe = (1.0 + 0.1 * np.cos(x / 3.0)) * np.exp(0.5j * phase)
    baseline_stack = np.stack(
        [
            1.0 + 0.01 * index + 0.1 * np.sin((x + index) / 4.0) ** 2
            for index in range(num_frames)
        ]
    ).astype(np.float64)
    minus_stack = baseline_stack * (
        1.0 - 0.01 * np.cos(x / 5.0)[None, :, :]
    )
    plus_stack = baseline_stack * (
        1.0 + 0.012 * np.sin(y / 5.0)[None, :, :]
    )
    product = np.exp(1j * 0.1 * np.sin(x / 4.0)).astype(np.complex128)
    projected = product * np.exp(1e-6j * np.cos(y / 3.0))
    convergence = {
        "U_A_exit": np.asarray([0.04, 0.02, 0.0]),
        "P_B": np.asarray([0.05, 0.025, 0.0]),
        "I_stack": np.asarray([0.06, 0.03, 0.0]),
    }
    return {
        "baseline": {
            "z_m": (np.arange(nz, dtype=np.float64) + 0.5) * 1e-6,
            "diameter_z_m": np.linspace(30e-6, 20e-6, nz),
            "n_volume": np.where(
                ((x - (nx - 1) / 2.0) ** 2 + (y - (ny - 1) / 2.0) ** 2)[
                    None, :, :
                ]
                < np.linspace(5.0, 3.0, nz)[:, None, None] ** 2,
                1.0,
                1.5,
            ),
            "dx_m": 0.5e-6,
            "U_A_exit": exit_field,
            "P_B": probe.astype(np.complex128),
            "I_stack": baseline_stack,
        },
        "controls": {
            "phase_screen_product": product,
            "projected_phase": projected,
            "projected_difference": product - projected,
        },
        "convergence": {
            "axial": {"x_values": np.asarray([2e-6, 1e-6, 0.5e-6]), **convergence},
            "lateral": {
                "x_values": np.asarray([1e-6, 0.5e-6, 0.25e-6]),
                **convergence,
            },
            "fov": {
                "x_values": np.asarray([64e-6, 80e-6, 96e-6]),
                **convergence,
            },
        },
        "sweep": {
            "case_ids": ["waist_minus", "baseline", "waist_plus"],
            "I_stack": np.stack([minus_stack, baseline_stack, plus_stack]),
        },
        "metrics": {
            "visibility": {
                "per_frame_minus": np.linspace(0.01, 0.02, num_frames),
                "per_frame_plus": np.linspace(0.012, 0.022, num_frames),
                "floor": 0.004,
                "most_sensitive_frame": num_frames - 1,
            }
        },
    }


def test_save_exp040_figures_writes_eight_named_readable_pngs(
    tmp_path: Path,
) -> None:
    paths = save_exp040_figures(_exp040_result(), tmp_path)

    assert [path.name for path in paths] == list(EXP040_FIGURE_FILENAMES)
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 0
            assert image.height > 0


def test_geometry_plot_rejects_volume_axis_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="match z/D"):
        plot_tgv_geometry_and_index_slices(
            np.ones((3, 8, 8)),
            np.asarray([0.5e-6, 1.5e-6]),
            np.asarray([3e-6, 2e-6]),
            0.5e-6,
            tmp_path / "bad_geometry.png",
        )


def test_projected_plot_rejects_nonfinite_difference(tmp_path: Path) -> None:
    field = np.ones((8, 8), dtype=np.complex128)
    difference = np.zeros((8, 8), dtype=np.complex128)
    difference[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 2D"):
        plot_projected_limit_comparison(
            field,
            field,
            0.5e-6,
            tmp_path / "bad_projected.png",
            projected_difference=difference,
        )


def test_visibility_plot_rejects_mismatched_stacks(tmp_path: Path) -> None:
    baseline = np.ones((3, 8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="share one shape"):
        plot_detector_visibility(
            np.ones((2, 8, 8), dtype=np.float64),
            baseline,
            baseline,
            0.01,
            0.5e-6,
            tmp_path / "bad_visibility.png",
        )


def test_convergence_plots_omit_exact_zero_before_log_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_line_plot(
        x_values: Any,
        y_series: Any,
        labels: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        calls.append(
            {
                "x_values": x_values,
                "y_series": y_series,
                "labels": labels,
                **kwargs,
            }
        )
        return np.full((20, 30, 3), 255, dtype=np.uint8)

    monkeypatch.setattr(plot_module, "_line_plot", fake_line_plot)
    errors = {
        "U_A_exit": np.asarray([0.04, 0.02, 0.0]),
        "P_B": np.asarray([0.05, 0.025, 0.0]),
        "I_stack": np.asarray([0.06, 0.03, 0.0]),
    }
    plot_module.plot_dz_convergence(
        np.asarray([2e-6, 1e-6, 0.5e-6]),
        errors,
        tmp_path / "dz.png",
    )
    plot_module.plot_lateral_fov_convergence(
        np.asarray([1e-6, 0.5e-6, 0.25e-6]),
        errors,
        np.asarray([64e-6, 80e-6, 96e-6]),
        errors,
        tmp_path / "lateral_fov.png",
    )

    assert len(calls) == 3
    for call in calls:
        assert call["log_y"] is True
        assert "reference" in call["y_label"]
        for values in call["y_series"]:
            assert np.all(np.asarray(values) > 0.0)
            assert np.asarray(values).shape == (2,)
