"""Repair Stage-B HDF5/figures only from the locked formal checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.io.config import load_config  # noqa: E402
from tgv_ptycho.io.metadata import created_at_utc  # noqa: E402
from tgv_ptycho.io.save_load import save_json  # noqa: E402

REGISTERED_RUN_RELATIVE = Path(
    "runs/exp040_TGV_3d_multislice_r10_stage_b_20260815_181819"
)
REGISTERED_INPUT_SHA256 = {
    "config.yaml": "B77B7D02B80BF45F3BD917BB821AFC0452E24F38418EAF88FDC4EA62AD5541AA",
    "metadata.json": "052CE5781219A58906967A85E1AD3221349E0C959570138A827AE9101E208ABA",
    "metrics.json": "60DAE2C59FB89EA2D201D53C105BCEAF98A6F36D163D9401146058F0F27DB11A",
    "run_progress.json": (
        "FC8947CD6151C262E2F4A3744E863E662BF49DBEE2180923D3C1954CE999083E"
    ),
    "run_state.json": (
        "559B8B51CBF19C1F42221FD917E28446DAEAC37999ABD1C11B37A8DAAFBB0F12"
    ),
    "checkpoints/coarse_nominal.npz": (
        "899237BB35003FA50AE0C010D3D9A4F047C1CE6BB89ED8838B7A2277060DB1A7"
    ),
    "checkpoints/fine_nominal.npz": (
        "F2EE0DBF1DF82342DD7434FB77CE0606015670A3E3531EDC22D1DBF405F32640"
    ),
    "checkpoints/fine_enlarged_pml.npz": (
        "3C81CB4D4A5BC9D50ED239BD269B219DB34F312CD01930D4C554FC1843B7B802"
    ),
    "checkpoints/multislice_fine_1024.npz": (
        "5D75B300BAE6B3F0A5202A55AE99942C094D1C6267E0C3B68139E00B42824C9D"
    ),
}
REPAIRED_HDF5_RELATIVE = Path("outputs/exp040_r10_stage_b_repaired.h5")
REPAIRED_FIGURES_RELATIVE = Path("repaired_figures")
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


def _formal_runner() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_exp040_r10_stage_b.py"
    spec = importlib.util.spec_from_file_location("r10_stage_b_locked", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the locked Stage-B runner.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_exact_run_dir(run_dir: Path) -> None:
    expected = (PROJECT_ROOT / REGISTERED_RUN_RELATIVE).resolve()
    if run_dir.resolve() != expected:
        raise ValueError(f"Artifact repair only accepts: {expected}")


def _verify_registered_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for relative, expected_hash in REGISTERED_INPUT_SHA256.items():
        path = run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Registered repair input is missing: {path}")
        actual = _sha256(path)
        if actual != expected_hash:
            raise ValueError(f"Repair input hash differs for {relative}: {actual}")
    state = _load_json(run_dir / "run_state.json")
    if (
        state.get("status") != "failed_during_execution"
        or state.get("error_type") != "KeyError"
        or state.get("error") != "'multislice_passband'"
        or state.get("formal_attempt_retained") is not True
    ):
        raise ValueError("Registered Stage-B plot failure signature differs.")
    metrics = _load_json(run_dir / "metrics.json")
    expected_science = {
        "status": "Failed",
        "interpretation_code": "helmholtz_reference_not_validated",
        "mesh": 1.1063644834324253,
        "pml": 1.0821141299585554e-05,
        "cross": 1.1128332252351876,
        "reference_validated": False,
    }
    actual_science = {
        "status": metrics.get("status"),
        "interpretation_code": metrics.get("interpretation_code"),
        "mesh": metrics.get("comparisons", {})
        .get("mesh", {})
        .get("passband_radial_l2"),
        "pml": metrics.get("comparisons", {})
        .get("pml", {})
        .get("passband_radial_l2"),
        "cross": metrics.get("comparisons", {})
        .get("cross_model", {})
        .get("passband_radial_l2"),
        "reference_validated": metrics.get("reference_controls", {}).get(
            "reference_validated"
        ),
    }
    if actual_science != expected_science:
        raise ValueError("Registered Stage-B scientific values differ.")
    progress = _load_json(run_dir / "run_progress.json")
    events = progress.get("events")
    if not isinstance(events, list):
        raise ValueError("Registered Stage-B progress events are missing.")
    helmholtz_completed = [
        event.get("case_id")
        for event in events
        if event.get("event") == "helmholtz_case_completed"
    ]
    if helmholtz_completed != [
        "coarse_nominal",
        "fine_nominal",
        "fine_enlarged_pml",
    ]:
        raise ValueError("All three registered Helmholtz completions are required.")
    multislice_events = [
        event for event in events if event.get("event") == "multislice_case_completed"
    ]
    if len(multislice_events) != 1 or multislice_events[0].get("slice_count") != 400:
        raise ValueError("The registered 400-slice completion is required.")
    post_events = [
        event for event in events if event.get("event") == "postprocessing_completed"
    ]
    if len(post_events) != 1 or post_events[0].get("status") != "Failed":
        raise ValueError("The registered scientific postprocessing is incomplete.")
    if progress.get("latest_event", {}).get("event") != "artifacts_writing_started":
        raise ValueError("Registered Stage-B progress endpoint differs.")
    return metrics, state


def _load_checkpoints(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    helmholtz: dict[str, Any] = {}
    for case_id in (
        "coarse_nominal",
        "fine_nominal",
        "fine_enlarged_pml",
    ):
        path = run_dir / "checkpoints" / f"{case_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            helmholtz[case_id] = {
                "radius_m": np.asarray(data["radius_m"]).copy(),
                "normalized_total_trace": np.asarray(
                    data["normalized_total_trace"]
                ).copy(),
                "normalized_scattered_trace": np.asarray(
                    data["normalized_scattered_trace"]
                ).copy(),
                "controls": json.loads(str(data["controls_json"])),
            }
    path = run_dir / "checkpoints" / "multislice_fine_1024.npz"
    with np.load(path, allow_pickle=False) as data:
        multislice = {
            "normalized_native_field": np.asarray(
                data["normalized_native_field"]
            ).copy(),
            "controls": json.loads(str(data["controls_json"])),
        }
    return helmholtz, multislice


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("Scientific forward recomputation is forbidden in repair.")


def _replay_postprocessing(
    runner: ModuleType,
    config: dict[str, Any],
    helmholtz: dict[str, Any],
    multislice: dict[str, Any],
    locked_metrics: dict[str, Any],
) -> dict[str, Any]:
    for name in (
        "_solve_helmholtz_case",
        "_multislice_reference",
        "solve_sparse_direct",
        "multislice_propagate_streamed_A",
        "angular_spectrum_propagate",
    ):
        setattr(runner, name, _forbidden)
    post = runner._postprocess_once(config, helmholtz, multislice)
    checks = {
        "comparisons": _plain(post["comparisons"])
        == locked_metrics["comparisons"],
        "projection_controls": _plain(post["projection_controls"])
        == locked_metrics["projection_controls"],
        "restriction_controls": _plain(post["restriction_controls"])
        == locked_metrics["restriction_controls"],
        "annular_constant": float(post["annular_constant_max_abs_error"])
        == float(locked_metrics["annular_constant_max_abs_error"]),
        "anisotropy": float(
            post["multislice_azimuthal_anisotropy_relative_l2"]
        )
        == float(
            locked_metrics["reference_controls"][
                "multislice_azimuthal_anisotropy_relative_l2"
            ]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Checkpoint replay differs from locked metrics: {checks}")
    result = {
        "metrics": locked_metrics,
        "selected_maps": {
            "helmholtz_passband_amplitude": np.abs(
                post["selected_maps"]["helmholtz_passband"]
            ),
            "multislice_passband_amplitude": np.abs(
                post["selected_maps"]["multislice_passband"]
            ),
            "normalized_cross_residual": post["selected_maps"][
                "normalized_cross_residual"
            ],
            "cross_phase_difference_rad": post["selected_maps"][
                "cross_phase_difference_rad"
            ],
        },
        "radial_profiles": {
            "radius_m": post["radial_radius_m"],
            **{
                f"{case_id}_raw": values
                for case_id, values in post["radial_raw"].items()
            },
            **{
                f"{case_id}_passband": values
                for case_id, values in post["radial_pass"].items()
            },
        },
        "native_traces": {
            case_id: {
                "radius_m": values["radius_m"],
                "normalized_total_trace": values["normalized_total_trace"],
                "normalized_scattered_trace": values[
                    "normalized_scattered_trace"
                ],
            }
            for case_id, values in helmholtz.items()
        },
        "selected_complex_fields": {
            "helmholtz_reference_passband": post["selected_maps"][
                "helmholtz_passband"
            ],
            "multislice_passband": post["selected_maps"][
                "multislice_passband"
            ],
        },
        "replay_checks": checks,
    }
    return result


def _hdf5_to_plain(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _hdf5_to_plain(node[key]) for key in node}
    return _plain(node[()])


def _validate_outputs(
    hdf5_path: Path,
    figure_paths: list[Path],
    result: dict[str, Any],
) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as h5:
        entry = h5["entry"]
        if set(entry) != {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
        }:
            raise RuntimeError("Repaired Stage-B HDF5 entry layout differs.")
        if "truth" in entry or "reconstruction" in entry:
            raise RuntimeError("Repaired HDF5 must not claim truth/reconstruction.")
        if _hdf5_to_plain(entry["metrics"]) != result["metrics"]:
            raise RuntimeError("Repaired HDF5 metrics differ from locked JSON.")
        stored_h = np.asarray(
            entry["data/selected_complex_fields/helmholtz_reference_passband"]
        )
        stored_ms = np.asarray(
            entry["data/selected_complex_fields/multislice_passband"]
        )
    if not np.array_equal(
        stored_h, result["selected_complex_fields"]["helmholtz_reference_passband"]
    ) or not np.array_equal(
        stored_ms, result["selected_complex_fields"]["multislice_passband"]
    ):
        raise RuntimeError("Repaired HDF5 selected fields differ from replay.")
    for path in figure_paths:
        image = np.asarray(iio.imread(path))
        if image.ndim not in (2, 3) or image.size == 0:
            raise RuntimeError(f"Repaired figure is invalid: {path}")
    return {
        "entry_keys_exact": True,
        "truth_absent": True,
        "reconstruction_absent": True,
        "full_metrics_exact_round_trip": True,
        "selected_complex_fields_exact_round_trip": True,
        "figure_count": len(figure_paths),
        "figures_readable": True,
        "status": result["metrics"]["status"],
        "interpretation_code": result["metrics"]["interpretation_code"],
    }


def repair(run_dir: Path) -> Path:
    """Replay only frozen postprocessing and create separate repaired artifacts."""

    run_dir = run_dir.resolve()
    _verify_exact_run_dir(run_dir)
    hdf5_path = run_dir / REPAIRED_HDF5_RELATIVE
    figures_dir = run_dir / REPAIRED_FIGURES_RELATIVE
    record_path = run_dir / REPAIR_RECORD_RELATIVE
    if hdf5_path.exists() or figures_dir.exists() or record_path.exists():
        raise FileExistsError("Registered Stage-B artifact repair already exists.")
    metrics, state = _verify_registered_inputs(run_dir)
    config = load_config(run_dir / "config.yaml")
    metadata = _load_json(run_dir / "metadata.json")
    helmholtz, multislice = _load_checkpoints(run_dir)
    runner = _formal_runner()
    result = _replay_postprocessing(
        runner, config, helmholtz, multislice, metrics
    )
    plot_result = dict(result)
    plot_radial = dict(result["radial_profiles"])
    plot_radial["multislice_passband"] = plot_radial[
        "multislice_fine_1024_passband"
    ]
    plot_result["radial_profiles"] = plot_radial
    figure_paths = runner.save_exp040_r10_stage_b_figures(
        plot_result, figures_dir
    )
    runner._write_hdf5(
        hdf5_path,
        config=config,
        metadata=metadata,
        result=result,
    )
    validation = _validate_outputs(hdf5_path, figure_paths, result)
    input_hashes_after = {
        relative: _sha256(run_dir / relative)
        for relative in REGISTERED_INPUT_SHA256
    }
    if input_hashes_after != REGISTERED_INPUT_SHA256:
        raise RuntimeError("Stage-B repair modified a registered formal input.")
    output_hashes = {
        REPAIRED_HDF5_RELATIVE.as_posix(): _sha256(hdf5_path),
        **{
            path.relative_to(run_dir).as_posix(): _sha256(path)
            for path in figure_paths
        },
    }
    record = {
        "version": "R10_stage_b_artifact_repair_v1",
        "completed_at": created_at_utc(),
        "formal_run_state_preserved": True,
        "formal_shell_exit_code_preserved": 1,
        "formal_failure": state["error"],
        "scientific_recomputation": False,
        "helmholtz_or_multislice_called": False,
        "checkpoint_postprocessing_replayed": True,
        "fft_passband_postprocessing_replayed": True,
        "plot_alias_only": (
            "multislice_passband := multislice_fine_1024_passband"
        ),
        "registered_input_sha256": dict(REGISTERED_INPUT_SHA256),
        "registered_input_sha256_after_repair": input_hashes_after,
        "output_sha256": output_hashes,
        "replay_checks": result["replay_checks"],
        "validation": validation,
    }
    save_json(record_path, record)
    print(f"repaired_hdf5: {hdf5_path}", flush=True)
    repaired_hash = output_hashes[REPAIRED_HDF5_RELATIVE.as_posix()]
    print(f"repaired_hdf5_sha256: {repaired_hash}", flush=True)
    print("scientific_recomputation: false", flush=True)
    print("helmholtz_or_multislice_called: false", flush=True)
    return hdf5_path


def main() -> None:
    args = _parse_args()
    repair(args.run_dir)


if __name__ == "__main__":
    main()
