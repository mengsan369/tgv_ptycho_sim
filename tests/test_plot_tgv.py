from __future__ import annotations

from pathlib import Path

import numpy as np

from tgv_ptycho.viz.plot_tgv import plot_loss_curves


def test_plot_loss_curves_accepts_different_iteration_counts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "loss_curves.png"

    plot_loss_curves(
        [
            np.linspace(1.0, 0.1, 6, dtype=np.float64),
            np.linspace(1.0, 0.01, 11, dtype=np.float64),
        ],
        ["short", "long"],
        output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
