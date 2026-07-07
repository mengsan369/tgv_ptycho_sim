"""Scan-stage calibration interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_stage_positions(path: str | Path) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Load scan-stage positions and metadata.

    TODO: Support CSV/HDF5 logs, unit conversion to meters, coordinate-axis
    conventions, and synchronization with detector frames.
    """

    raise NotImplementedError("Stage-position loading is not implemented yet.")


def refine_stage_positions(
    I_stack: NDArray[np.floating],
    initial_positions: NDArray[np.floating],
    method: str = "phase_correlation",
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Refine scan positions from measured data.

    TODO: Add position refinement after the first reconstruction prototypes.
    """

    raise NotImplementedError("Stage-position refinement is not implemented yet.")
