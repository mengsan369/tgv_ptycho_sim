from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import tgv_ptycho.viz.plot_exp040_r1 as r1_plot_module
from tgv_ptycho.viz.plot_exp040 import EXP040_FIGURE_FILENAMES
from tgv_ptycho.viz.plot_exp040_r1 import (
    EXP040_R1_FIGURE_FILENAMES,
    plot_r1_external_padding_convergence,
    save_exp040_r1_figures,
)


def _errors() -> dict[str, np.ndarray]:
    return {
        "U_A_exit": np.asarray([0.04, 0.018, 0.0]),
        "P_B": np.asarray([0.045, 0.021, 0.0]),
        "I_stack": np.asarray([0.048, 0.024, 0.0]),
    }


def _result() -> dict[str, Any]:
    return {
        "diagnostics_r1": {
            "refined_convergence": {
                "axial": {
                    "x_values": np.asarray([1e-6, 0.5e-6, 0.25e-6]),
                    **_errors(),
                },
                "lateral": {
                    "x_values": np.asarray([0.5e-6, 0.25e-6, 0.125e-6]),
                    **_errors(),
                },
                "fov": {
                    "x_values": np.asarray([96e-6, 112e-6, 128e-6]),
                    **_errors(),
                },
            },
            "external_padding": {
                "x_values": np.asarray([96e-6, 112e-6, 128e-6]),
                "P_B": np.asarray([0.044, 0.019, 0.0]),
                "I_stack": np.asarray([0.049, 0.023, 0.0]),
                "U_A_exit_center_invariance": np.asarray([2e-14, 1e-14, 0.0]),
            },
        },
        "metrics": {
            "diagnostics_r1": {
                "thresholds": {
                    "convergence_relative_l2_max": 0.05,
                    "a_exit_center_invariance_max": 1e-12,
                }
            },
            "thresholds": {"convergence_relative_l2_max": 0.05},
        },
    }


def test_save_exp040_r1_figures_writes_two_named_readable_pngs(
    tmp_path: Path,
) -> None:
    paths = save_exp040_r1_figures(_result(), tmp_path)

    assert EXP040_FIGURE_FILENAMES == (
        "tgv_geometry_and_index_slices.png",
        "exit_field_multislice.png",
        "projected_limit_comparison.png",
        "dz_convergence.png",
        "lateral_fov_convergence.png",
        "B_plane_probe.png",
        "detector_intensity_baseline.png",
        "detector_visibility.png",
    )
    assert [path.name for path in paths] == list(EXP040_R1_FIGURE_FILENAMES)
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 0
            assert image.height > 0


def test_r1_log_rendering_omits_exact_zero_and_keeps_gate(
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

    monkeypatch.setattr(r1_plot_module, "_line_plot", fake_line_plot)
    paths = save_exp040_r1_figures(_result(), tmp_path)

    assert len(calls) == 5
    assert all(path.is_file() for path in paths)
    for call in calls[:4]:
        assert call["log_y"] is True
        assert "[1]" in call["y_label"]
        assert any("5.0% convergence gate" == label for label in call["labels"])
        for values in call["y_series"]:
            assert np.all(np.asarray(values) > 0.0)
            assert np.min(np.asarray(values)) > np.finfo(float).tiny

    invariance_call = calls[4]
    assert invariance_call["log_y"] is True
    assert "[1]" in invariance_call["y_label"]
    assert "1.0e-12 A-exit invariance gate" in invariance_call["labels"]
    assert "5.0% convergence gate" not in invariance_call["labels"]
    np.testing.assert_allclose(invariance_call["y_series"][-1], 1e-12)


def test_external_padding_plot_rejects_mismatched_error_shape(
    tmp_path: Path,
) -> None:
    external = _result()["diagnostics_r1"]["external_padding"]
    external["P_B"] = np.asarray([0.1, 0.0])

    with pytest.raises(ValueError, match="match x_values"):
        plot_r1_external_padding_convergence(
            external,
            0.05,
            1e-12,
            tmp_path / "bad_external.png",
        )


def test_r1_plot_rejects_nonfinite_invariance(tmp_path: Path) -> None:
    result = _result()
    result["diagnostics_r1"]["external_padding"][
        "U_A_exit_center_invariance"
    ][0] = np.nan

    with pytest.raises(ValueError, match="finite, non-negative"):
        save_exp040_r1_figures(result, tmp_path)


def test_r1_plot_requires_registered_a_exit_invariance_gate(
    tmp_path: Path,
) -> None:
    result = _result()
    del result["metrics"]["diagnostics_r1"]["thresholds"][
        "a_exit_center_invariance_max"
    ]

    with pytest.raises(
        ValueError,
        match="must provide a_exit_center_invariance_max",
    ):
        save_exp040_r1_figures(result, tmp_path)
