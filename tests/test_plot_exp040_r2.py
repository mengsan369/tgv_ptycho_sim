from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import tgv_ptycho.viz.plot_exp040_r2 as plot_module
from tgv_ptycho.viz.plot_exp040_r2 import (
    EXP040_R2_FIGURE_FILENAMES,
    save_exp040_r2_figures,
)


def _result() -> dict[str, Any]:
    x = np.asarray([96e-6, 192e-6, 288e-6])
    method = {
        "x_values_m": x,
        "P_B": np.asarray([0.2, 0.04, 0.0]),
        "I_stack": np.asarray([0.4, 0.06, 0.0]),
    }
    return {
        "metrics": {
            "diagnostics_r2": {
                "period_aligned": {
                    "current_asm": method,
                    "alias_controlled": {
                        **method,
                        "P_B": np.asarray([0.1, 0.02, 0.0]),
                        "I_stack": np.asarray([0.2, 0.03, 0.0]),
                    },
                },
                "method_difference": {
                    "x_values_m": x,
                    "P_B": np.asarray([0.3, 0.2, 0.1]),
                    "I_stack": np.asarray([0.5, 0.4, 0.3]),
                },
                "alias_masks": {
                    "x_values_m": x,
                    "AB_kept_bin_fraction": np.asarray([0.1, 0.2, 0.3]),
                    "BC_kept_bin_fraction": np.asarray([0.05, 0.1, 0.2]),
                },
                "thresholds": {"convergence_relative_l2_max": 0.05},
            }
        }
    }


def test_save_exp040_r2_figures_writes_two_readable_pngs(
    tmp_path: Path,
) -> None:
    paths = save_exp040_r2_figures(_result(), tmp_path)

    assert [path.name for path in paths] == list(EXP040_R2_FIGURE_FILENAMES)
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 0
            assert image.height > 0


def test_r2_plot_uses_frozen_convergence_and_materiality_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    save_exp040_r2_figures(_result(), tmp_path)

    assert len(calls) == 4
    assert all(call["log_y"] is True for call in calls[:3])
    assert calls[3]["log_y"] is False
    assert all("5.0% convergence gate" in call["labels"] for call in calls[:2])
    assert "5.0% materiality gate" in calls[2]["labels"]
    for call in calls[:3]:
        for values in call["y_series"]:
            assert np.all(np.asarray(values) > 0.0)


def test_r2_plot_rejects_invalid_mask_fraction(tmp_path: Path) -> None:
    result = _result()
    result["metrics"]["diagnostics_r2"]["alias_masks"][
        "AB_kept_bin_fraction"
    ][0] = 1.1

    with pytest.raises(ValueError, match="AB_kept_bin_fraction"):
        save_exp040_r2_figures(result, tmp_path)
