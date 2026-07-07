"""JSON and HDF5 persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    msg = f"Object of type {type(value).__name__} is not JSON serializable."
    raise TypeError(msg)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON with NumPy-aware conversion."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


def _write_item(group: h5py.Group, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        subgroup = group.require_group(name)
        for key, child in value.items():
            _write_item(subgroup, str(key), child)
        return
    if isinstance(value, str):
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=value, dtype=dtype)
        return
    if isinstance(value, Path):
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=str(value), dtype=dtype)
        return
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, str) for item in value):
            dtype = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=list(value), dtype=dtype)
            return
        group.create_dataset(name, data=np.asarray(value))
        return

    array = np.asarray(value)
    if array.dtype.kind in {"U", "O"}:
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=array.astype(str), dtype=dtype)
        return
    group.create_dataset(name, data=array)


def save_ptycho_hdf5(
    path: str | Path,
    *,
    I_stack: NDArray[np.floating] | None = None,
    scan_positions: NDArray[np.floating] | None = None,
    instrument: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
    truth: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    reconstruction: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    metadata: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save data using the project's internal CXI/NeXus-inspired HDF5 layout.

    Simulated datasets may include a `truth` group. Experimental datasets
    should omit `truth` and may include `calibration` and `preprocessing`
    groups instead. Run config, metadata, and metrics are stored as separate
    `/entry/config_yaml`, `/entry/metadata`, and `/entry/metrics` nodes.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        entry = h5.require_group("entry")
        data_group = entry.require_group("data")
        if I_stack is not None:
            _write_item(data_group, "I_stack", np.asarray(I_stack))
        if scan_positions is not None:
            _write_item(data_group, "scan_positions", np.asarray(scan_positions))

        if instrument is not None:
            _write_item(entry, "instrument", instrument)
        if sample is not None:
            _write_item(entry, "sample", sample)
        if truth is not None:
            _write_item(entry, "truth", truth)
        if calibration is not None:
            _write_item(entry, "calibration", calibration)
        if preprocessing is not None:
            _write_item(entry, "preprocessing", preprocessing)
        if reconstruction is not None:
            _write_item(entry, "reconstruction", reconstruction)
        if config_yaml is not None:
            _write_item(entry, "config_yaml", config_yaml)
        if metadata is not None:
            _write_item(entry, "metadata", metadata)
        if metrics is not None:
            _write_item(entry, "metrics", metrics)


def load_ptycho_hdf5(path: str | Path) -> h5py.File:
    """Open an internal ptychography HDF5 file in read-only mode."""

    return h5py.File(Path(path), "r")
