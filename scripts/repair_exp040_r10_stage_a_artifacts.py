"""Repair only the compact HDF5 artifact of the completed R10 Stage-A run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.io.config import load_config  # noqa: E402
from tgv_ptycho.io.metadata import created_at_utc  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402

REGISTERED_RUN_RELATIVE = Path(
    "runs/exp040_TGV_3d_multislice_r10_stage_a_20260815_162021"
)
REGISTERED_INPUT_SHA256 = {
    "config.yaml": "5D49274C41DEBE9742D00183FEFD46E6D4F5551F45C59BBEC32A8A468A042892",
    "metadata.json": "FEBE74B948FC59BC716DC43745C9BCF25758ECF77C6127BFAFB13A94BD934DC9",
    "metrics.json": "3346A2463E7B374EBE850E4CE621A133307F359EF5EBC66D70458E91F0602817",
    "run_progress.json": (
        "009533D2B32F20BA12D28834E5D1E02ED8FD56F0EABCA95F4EE8C926C557B9C6"
    ),
    "run_state.json": (
        "9A54A9F68464216F78EE1DDC04095DECDEEC6B65D59A36EE94418E02D79265DE"
    ),
    "outputs/exp040_r10_stage_a.h5": (
        "18034DB747102515633D55C86EDE57F7304240D7BBC4B29E3798292A29DF52A8"
    ),
}
REPAIRED_HDF5_RELATIVE = Path("outputs/exp040_r10_stage_a_repaired.h5")
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
        raise ValueError(f"Repair input must be a JSON mapping: {path}")
    return value


def _verify_exact_run_dir(run_dir: Path) -> None:
    expected = (PROJECT_ROOT / REGISTERED_RUN_RELATIVE).resolve()
    if run_dir.resolve() != expected:
        msg = f"Artifact repair only accepts the registered run: {expected}"
        raise ValueError(msg)


def _verify_registered_inputs(run_dir: Path) -> dict[str, Any]:
    for relative, expected_hash in REGISTERED_INPUT_SHA256.items():
        path = run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Registered repair input is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            msg = f"Repair input hash differs for {relative}: {actual_hash}"
            raise ValueError(msg)

    state = _load_json(run_dir / "run_state.json")
    if state.get("status") != "failed_during_execution":
        raise ValueError("Registered formal failure state is not preserved.")
    if state.get("error_type") != "TypeError" or "Object dtype" not in str(
        state.get("error")
    ):
        raise ValueError("Registered HDF5 object-dtype failure signature differs.")
    if state.get("formal_attempt_retained") is not True:
        raise ValueError("Formal-attempt retention flag is missing.")

    metrics = _load_json(run_dir / "metrics.json")
    comparison = metrics.get("comparison")
    outcome = metrics.get("outcome")
    if not isinstance(comparison, dict) or not isinstance(outcome, dict):
        raise ValueError("Registered scientific metrics are incomplete.")
    expected_science = {
        "raw_relative_l2": 0.04574990331167789,
        "external_passband_relative_l2": 0.023787308028510038,
        "status": "Passed",
        "interpretation_code": "scalar_lateral_reference_closed",
        "stage_b_allowed": True,
    }
    actual_science = {
        "raw_relative_l2": comparison.get("raw_relative_l2"),
        "external_passband_relative_l2": comparison.get(
            "external_passband_relative_l2"
        ),
        "status": outcome.get("status"),
        "interpretation_code": outcome.get("interpretation_code"),
        "stage_b_allowed": outcome.get("stage_b_allowed"),
    }
    if actual_science != expected_science:
        raise ValueError("Registered Stage-A scientific values differ.")

    progress = _load_json(run_dir / "run_progress.json")
    events = progress.get("events")
    if not isinstance(events, list):
        raise ValueError("Registered progress events are missing.")
    completed = [event for event in events if event.get("event") == "case_completed"]
    if [event.get("case_id") for event in completed] != [
        "current_512",
        "fine_1024",
    ] or [event.get("slice_count") for event in completed] != [400, 400]:
        raise ValueError("Both registered 400-slice case completions are required.")
    if progress.get("latest_event", {}).get("event") != "artifacts_writing_started":
        raise ValueError("Registered formal progress endpoint differs.")
    return metrics


def _normalise_metrics_for_hdf5(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert only the registered list of case mappings to keyed groups."""

    normalized = deepcopy(metrics)
    controls = normalized.get("case_controls")
    if not isinstance(controls, list) or not all(
        isinstance(item, dict) for item in controls
    ):
        raise ValueError("case_controls must be the registered list of mappings.")
    ids = [item.get("id") for item in controls]
    if ids != ["current_512", "fine_1024"]:
        raise ValueError("case_controls order or ids differ from registration.")
    normalized["case_controls"] = {
        str(item["id"]): item for item in controls
    }
    return normalized


def _plain_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain_value(value.item())
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _hdf5_to_plain(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _hdf5_to_plain(node[key]) for key in node}
    return _plain_value(node[()])


def _write_repaired_hdf5(
    run_dir: Path,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
    normalized_metrics: dict[str, Any],
) -> Path:
    output_path = run_dir / REPAIRED_HDF5_RELATIVE
    if output_path.exists():
        raise FileExistsError(f"Repair output already exists: {output_path}")
    save_ptycho_hdf5(
        output_path,
        instrument={
            "wavelength_m": float(config["physics"]["wavelength_m"]),
            "internal_reference_index": float(
                config["physics"]["internal_reference_index"]
            ),
            "external_medium_index": float(
                config["physics"]["external_medium_index"]
            ),
            "angular_spectrum_bandlimit": bool(
                config["physics"]["angular_spectrum_bandlimit"]
            ),
            "sampling": normalized_metrics["sampling"],
        },
        sample={
            "type": "single_axisymmetric_air_filled_tgv_in_glass",
            "geometry": dict(config["physics"]),
            "interface": dict(config["stage_a"]["interface"]),
        },
        config_yaml=(run_dir / "config.yaml").read_text(encoding="utf-8"),
        metadata=metadata,
        metrics=normalized_metrics,
    )
    return output_path


def _validate_repaired_hdf5(
    path: Path, normalized_metrics: dict[str, Any]
) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        entry = h5["entry"]
        expected_entry = {
            "config_yaml",
            "data",
            "instrument",
            "sample",
            "metadata",
            "metrics",
        }
        if set(entry) != expected_entry:
            raise RuntimeError(f"Repaired HDF5 entry keys differ: {sorted(entry)}")
        if len(entry["data"]) != 0:
            raise RuntimeError("Repaired HDF5 data group must remain empty.")
        if "truth" in entry or "reconstruction" in entry:
            raise RuntimeError("Repaired HDF5 must not contain truth/reconstruction.")
        stored_metrics = _hdf5_to_plain(entry["metrics"])
    if stored_metrics != normalized_metrics:
        raise RuntimeError("Repaired HDF5 metrics differ from normalized JSON.")
    return {
        "entry_keys_exact": True,
        "data_group_empty": True,
        "truth_absent": True,
        "reconstruction_absent": True,
        "full_metrics_exact_round_trip": True,
        "case_control_ids": sorted(normalized_metrics["case_controls"]),
        "raw_relative_l2": normalized_metrics["comparison"]["raw_relative_l2"],
        "external_passband_relative_l2": normalized_metrics["comparison"][
            "external_passband_relative_l2"
        ],
        "status": normalized_metrics["outcome"]["status"],
        "stage_b_allowed": normalized_metrics["outcome"]["stage_b_allowed"],
    }


def repair(run_dir: Path) -> Path:
    """Create a new compact HDF5 from locked JSON without scientific work."""

    run_dir = run_dir.resolve()
    _verify_exact_run_dir(run_dir)
    repair_record_path = run_dir / REPAIR_RECORD_RELATIVE
    repaired_path = run_dir / REPAIRED_HDF5_RELATIVE
    if repair_record_path.exists() or repaired_path.exists():
        raise FileExistsError("Registered artifact repair has already been attempted.")
    metrics = _verify_registered_inputs(run_dir)
    normalized_metrics = _normalise_metrics_for_hdf5(metrics)
    config = load_config(run_dir / "config.yaml")
    metadata = _load_json(run_dir / "metadata.json")
    repaired_path = _write_repaired_hdf5(
        run_dir,
        config=config,
        metadata=metadata,
        normalized_metrics=normalized_metrics,
    )
    validation = _validate_repaired_hdf5(repaired_path, normalized_metrics)
    original_hashes_after = {
        relative: _sha256(run_dir / relative)
        for relative in REGISTERED_INPUT_SHA256
    }
    if original_hashes_after != REGISTERED_INPUT_SHA256:
        raise RuntimeError("Artifact repair modified a registered original input.")
    record = {
        "version": "R10_stage_a_artifact_repair_v1",
        "completed_at": created_at_utc(),
        "formal_run_state_preserved": True,
        "formal_shell_exit_code_preserved": 1,
        "original_failure": "HDF5 list-of-mappings object dtype serialization",
        "scientific_recomputation": False,
        "forward_or_fft_called": False,
        "plots_recomputed": False,
        "normalization": (
            "case_controls ordered list converted to id-keyed HDF5 groups only"
        ),
        "registered_input_sha256": dict(REGISTERED_INPUT_SHA256),
        "original_input_sha256_after_repair": original_hashes_after,
        "repaired_hdf5_relative_path": REPAIRED_HDF5_RELATIVE.as_posix(),
        "repaired_hdf5_sha256": _sha256(repaired_path),
        "validation": validation,
    }
    save_json(repair_record_path, record)
    print(f"repaired_hdf5: {repaired_path}", flush=True)
    print(f"repaired_hdf5_sha256: {record['repaired_hdf5_sha256']}", flush=True)
    print("scientific_recomputation: false", flush=True)
    return repaired_path


def main() -> None:
    args = _parse_args()
    repair(args.run_dir)


if __name__ == "__main__":
    main()
