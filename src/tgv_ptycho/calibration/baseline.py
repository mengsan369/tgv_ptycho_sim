"""Baseline experimental metadata interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_baseline_metadata(
    camera: dict[str, Any],
    stage: dict[str, Any],
    geometry: dict[str, Any],
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine camera, stage, geometry, and preprocessing metadata.

    TODO: Define a stable baseline schema for each experimental run.
    """

    raise NotImplementedError("Baseline metadata assembly is not implemented yet.")


def save_baseline_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Save baseline metadata for an experimental run.

    TODO: Decide whether baseline metadata is YAML-first, HDF5-first, or both.
    """

    raise NotImplementedError("Baseline metadata saving is not implemented yet.")


def load_baseline_metadata(path: str | Path) -> dict[str, Any]:
    """Load baseline metadata for an experimental run."""

    raise NotImplementedError("Baseline metadata loading is not implemented yet.")
