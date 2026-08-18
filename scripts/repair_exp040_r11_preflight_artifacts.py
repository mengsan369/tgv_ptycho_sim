"""Create a separate HDF5 artifact from the locked failed R11 preflight run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.io.config import config_to_yaml, load_config  # noqa: E402
from tgv_ptycho.io.metadata import created_at_utc  # noqa: E402
from tgv_ptycho.io.save_load import (  # noqa: E402
    save_json,
    save_ptycho_hdf5,
)

REGISTERED_RUN_RELATIVE = Path(
    "runs/exp040_TGV_3d_multislice_r11_preflight_20260815_193925"
)
REGISTERED_SOURCE_CONFIG_SHA256 = (
    "2FEAA121E7B6EA4F2B3F3BC0AC3C2843891AC31214FBF2156FD369D072252CF4"
)
REGISTERED_INPUT_SHA256 = {
    "config.yaml": "FEE2891ACF3A3171DF798041BDEA784E77DF88B97ABF800F9C10F1FB6E609912",
    "metadata.json": "90A2EF833D0D48CBD67EBA8FB7D925DF6783A8F28EEA1749D504BC376236B8E7",
    "metrics.json": "D9D2F82D1B808281243A8966614B1E8FF4B9D179B3899C59BCCA105BD3F2BB58",
    "run_progress.json": (
        "1E81B1DA823A299949936011D3FCBD5A19028C9FA36236DFB6842E25ED05AC3B"
    ),
    "run_state.json": (
        "77C5EB4CF748655FA3FBD5C36570BD4F437ED242E7D3DAE7C5E0982DA39E872B"
    ),
    "outputs/exp040_r11_preflight.h5": (
        "CDFFD2EE6E2F34D585FEEBF1A15DC1F3D182DEDD03593D664A3DFBCAA04D5EF3"
    ),
}
REPAIRED_HDF5_RELATIVE = Path("outputs/exp040_r11_preflight_repaired.h5")
REPAIR_RECORD_RELATIVE = Path("artifact_repair.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"repair input must be a JSON mapping: {path}")
    return value


def _hdf5_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _hdf5_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)) and any(
        isinstance(child, Mapping) for child in value
    ):
        return {
            "__sequence_encoding__": "indexed_mapping_v1",
            "length": len(value),
            "items": {
                f"{index:06d}": _hdf5_safe(child)
                for index, child in enumerate(value)
            },
        }
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _hdf5_to_plain(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _hdf5_to_plain(node[key]) for key in node}
    return _plain(node[()])


def _decode_hdf5_sequences(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__sequence_encoding__", "items", "length"}:
            if value["__sequence_encoding__"] != "indexed_mapping_v1":
                raise ValueError("unknown repair HDF5 sequence encoding")
            length = int(value["length"])
            items = value["items"]
            if not isinstance(items, Mapping):
                raise ValueError("encoded sequence items must be a mapping")
            expected = [f"{index:06d}" for index in range(length)]
            if list(sorted(items)) != expected:
                raise ValueError("encoded sequence indices differ")
            return [_decode_hdf5_sequences(items[key]) for key in expected]
        return {
            str(key): _decode_hdf5_sequences(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_decode_hdf5_sequences(child) for child in value]
    return value


def _verify_registered_inputs(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_run = (PROJECT_ROOT / REGISTERED_RUN_RELATIVE).resolve()
    if run_dir.resolve() != expected_run:
        raise ValueError(f"artifact repair only accepts: {expected_run}")
    source_config = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "exp040_TGV_3d_multislice_r11_preflight.yaml"
    )
    if _sha256(source_config) != REGISTERED_SOURCE_CONFIG_SHA256:
        raise ValueError("registered R11 preflight source config differs")
    for relative, expected_hash in REGISTERED_INPUT_SHA256.items():
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"registered repair input differs: {relative}")

    state = _load_json(run_dir / "run_state.json")
    if (
        state.get("status") != "failed_during_execution"
        or state.get("error_type") != "TypeError"
        or state.get("error")
        != "Object dtype dtype('O') has no native HDF5 equivalent"
        or state.get("scientific_result") is not False
        or state.get("formal_r11_allowed") is not False
    ):
        raise ValueError("registered preflight artifact failure differs")

    metrics = _load_json(run_dir / "metrics.json")
    if (
        metrics.get("status") != "Passed"
        or metrics.get("interpretation_code") != "r11_formal_preflight_passed"
        or metrics.get("hard_controls_pass") is not True
        or metrics.get("formal_r11_allowed") is not True
        or metrics.get("scientific_result") is not False
        or not all(metrics.get("control_pass", {}).values())
    ):
        raise ValueError("registered preflight controls did not all pass")
    metadata = _load_json(run_dir / "metadata.json")
    if (
        metadata.get("status") != "Passed"
        or metadata.get("formal_r11_allowed") is not True
        or metadata.get("scientific_result") is not False
    ):
        raise ValueError("registered preflight metadata differs")
    return metrics, metadata, state


def _validate_repaired_hdf5(
    path: Path,
    *,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    instrument: Mapping[str, Any],
) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        entry = h5["entry"]
        if set(entry) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
        }:
            raise RuntimeError("repaired R11 preflight HDF5 layout differs")
        if len(entry["data"]) != 0:
            raise RuntimeError("repaired preflight data group must be empty")
        if "truth" in entry or "reconstruction" in entry:
            raise RuntimeError("preflight HDF5 must not claim truth/reconstruction")
        stored_metrics = _decode_hdf5_sequences(
            _hdf5_to_plain(entry["metrics"])
        )
        stored_metadata = _decode_hdf5_sequences(
            _hdf5_to_plain(entry["metadata"])
        )
        stored_instrument = _decode_hdf5_sequences(
            _hdf5_to_plain(entry["instrument"])
        )
    if stored_metrics != metrics:
        raise RuntimeError("repaired HDF5 metrics differ from locked JSON")
    if stored_metadata != metadata:
        raise RuntimeError("repaired HDF5 metadata differ from locked JSON")
    if stored_instrument != instrument:
        raise RuntimeError("repaired HDF5 instrument differs")
    return {
        "entry_keys_exact": True,
        "data_group_empty": True,
        "truth_absent": True,
        "reconstruction_absent": True,
        "full_metrics_exact_round_trip": True,
        "full_metadata_exact_round_trip": True,
        "instrument_exact_round_trip": True,
    }


def repair(run_dir: Path) -> Path:
    """Repair persistence only; no preflight or scientific control is rerun."""

    run_dir = run_dir.resolve()
    repaired_path = run_dir / REPAIRED_HDF5_RELATIVE
    record_path = run_dir / REPAIR_RECORD_RELATIVE
    if repaired_path.exists() or record_path.exists():
        raise FileExistsError("registered R11 preflight repair already exists")
    metrics, metadata, state = _verify_registered_inputs(run_dir)
    config = load_config(run_dir / "config.yaml")
    instrument = {
        "wavelength_m": float(config["physics"]["wavelength_m"]),
        "formal_grid_controls": metrics["formal_grid_controls"],
        "resource_controls": metrics["resource_controls"],
    }
    save_ptycho_hdf5(
        repaired_path,
        instrument=_hdf5_safe(instrument),
        config_yaml=config_to_yaml(config),
        metadata=_hdf5_safe(metadata),
        metrics=_hdf5_safe(metrics),
    )
    validation = _validate_repaired_hdf5(
        repaired_path,
        metrics=metrics,
        metadata=metadata,
        instrument=instrument,
    )
    input_hashes_after = {
        relative: _sha256(run_dir / relative)
        for relative in REGISTERED_INPUT_SHA256
    }
    if input_hashes_after != REGISTERED_INPUT_SHA256:
        raise RuntimeError("R11 preflight repair modified a registered input")
    record = {
        "version": "R11_preflight_artifact_repair_v1",
        "completed_at": created_at_utc(),
        "original_run_state_preserved": True,
        "original_shell_exit_code_preserved": 1,
        "original_failure": state["error"],
        "scientific_result": False,
        "preflight_control_recomputation": False,
        "scientific_forward_recomputation": False,
        "sequence_encoding": "indexed_mapping_v1",
        "registered_source_config_sha256": REGISTERED_SOURCE_CONFIG_SHA256,
        "registered_input_sha256": dict(REGISTERED_INPUT_SHA256),
        "registered_input_sha256_after_repair": input_hashes_after,
        "output_sha256": {
            REPAIRED_HDF5_RELATIVE.as_posix(): _sha256(repaired_path)
        },
        "validation": validation,
        "external_metrics_status": metrics["status"],
        "external_metrics_interpretation_code": metrics[
            "interpretation_code"
        ],
        "external_metrics_formal_r11_allowed": metrics["formal_r11_allowed"],
    }
    save_json(record_path, record)
    print(f"repaired_hdf5: {repaired_path}", flush=True)
    print(f"repaired_hdf5_sha256: {_sha256(repaired_path)}", flush=True)
    print(f"repair_record: {record_path}", flush=True)
    print(f"repair_record_sha256: {_sha256(record_path)}", flush=True)
    return repaired_path


def main() -> None:
    args = _parse_args()
    repair(args.run_dir)


if __name__ == "__main__":
    main()
