from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import tgv_ptycho.viz.plot_exp040_r3 as plot_module
from tgv_ptycho.viz.plot_exp040_r3 import (
    EXP040_R3_FIGURE_FILENAMES,
    save_exp040_r3_figures,
)


def _result() -> dict[str, Any]:
    factors = np.asarray([1, 2, 4])
    fraction = np.asarray([0.0, 0.08, 0.12])
    convergence = np.asarray([0.2, 0.04, 0.0])
    spectra = {
        "factors": factors,
        "B_exit": {
            "outside_BC_alias_mask_energy_fraction_mean": fraction,
            "outside_BC_alias_mask_energy_fraction_max": fraction + 0.02,
            "outside_native_detector_nyquist_energy_fraction_mean": fraction,
            "outside_native_detector_nyquist_energy_fraction_max": fraction + 0.02,
        },
        "detector_intensity": {
            method: {
                "outside_native_detector_nyquist_energy_fraction_mean": fraction,
                "outside_native_detector_nyquist_energy_fraction_max": fraction
                + 0.02,
            }
            for method in ("current_asm", "alias_controlled")
        },
    }
    detector_convergence = {
        method: {
            "point_sample": convergence,
            "pixel_box_average": convergence * 0.8,
        }
        for method in ("current_asm", "alias_controlled")
    }
    yy, xx = np.mgrid[:16, :16]
    point = np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 16.0)
    pixel = 0.95 * point + 0.01
    return {
        "baseline": {"dx_m": 1.0e-6},
        "diagnostics_r3": {
            "selected_scan": {
                "point_sample": point,
                "pixel_box_average": pixel,
                "relative_difference": (point - pixel) / np.max(pixel),
            }
        },
        "metrics": {
            "diagnostics_r3": {
                "spectra": spectra,
                "bc_propagation": {
                    "factors": factors,
                    "detector_field_current_vs_alias_relative_l2": np.asarray(
                        [0.3, 0.2, 0.1]
                    ),
                    "full_intensity_current_vs_alias_relative_l2": np.asarray(
                        [0.4, 0.25, 0.12]
                    ),
                },
                "detector_sampling": {
                    "factors": factors,
                    "relative_to_factor4": {
                        "P_B": convergence * 0.5,
                        "detector": detector_convergence,
                    },
                },
                "detector_operator_difference": {
                    "factors": factors,
                    "point_vs_pixel_relative_l2": {
                        "current_asm": np.asarray([0.2, 0.15, 0.1]),
                        "alias_controlled": np.asarray([0.18, 0.12, 0.08]),
                    },
                },
                "thresholds": {"convergence_relative_l2_max": 0.05},
            }
        },
    }


def test_save_exp040_r3_figures_writes_three_readable_pngs(
    tmp_path: Path,
) -> None:
    paths = save_exp040_r3_figures(_result(), tmp_path)

    assert [path.name for path in paths] == list(EXP040_R3_FIGURE_FILENAMES)
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 0
            assert image.height > 0


def test_r3_plot_rejects_fraction_above_one(tmp_path: Path) -> None:
    result = _result()
    result["metrics"]["diagnostics_r3"]["spectra"]["B_exit"][
        "outside_BC_alias_mask_energy_fraction_mean"
    ][-1] = 1.1

    with pytest.raises(ValueError, match="outside_BC_alias_mask"):
        save_exp040_r3_figures(result, tmp_path)


def test_r3_log_plots_omit_exact_zero_reference_points(
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
    save_exp040_r3_figures(_result(), tmp_path)

    assert len(calls) == 5
    assert all(call["log_y"] is True for call in calls)
    for call in calls:
        for values in call["y_series"]:
            assert np.all(np.asarray(values) > 0.0)
        assert "exact zeros omitted" in call["title"]
