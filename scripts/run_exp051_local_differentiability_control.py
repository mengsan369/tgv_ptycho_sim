from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

_CONDA_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _CONDA_DLL_DIR.is_dir():
    _path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(_CONDA_DLL_DIR) not in _path_entries:
        os.environ["PATH"] = str(_CONDA_DLL_DIR) + os.pathsep + os.environ.get(
            "PATH", ""
        )

import h5py  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tgv_ptycho.inverse.exp051 import (  # noqa: E402
    Exp051SourceArtifact,
    load_exp051_source_true_probe,
    make_exp051_candidate_generator,
    sha256_file,
)
from tgv_ptycho.inverse.exp051_local_control import (  # noqa: E402
    PriorExp051Artifact,
    load_prior_exp051_formal,
    run_local_differentiability_control,
    validate_exp051_local_control_config,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402
from tgv_ptycho.objects.tgv3d import (  # noqa: E402
    make_tgv_air_fraction_slice_chord_quadrature,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exp051 q8 local-differentiability control."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _stack_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        return {}
    return {
        key: np.asarray([record[key] for record in records])
        for key in records[0]
    }


def _series_summary(series: dict[str, Any] | None) -> dict[str, Any] | None:
    if series is None:
        return None
    return {
        "steps_m": series["steps_m"],
        "normalized_jacobian_per_m": series["normalized_jacobian_per_m"],
        "pairwise_jacobian_relative_l2": series[
            "pairwise_jacobian_relative_l2"
        ],
        "one_sided_probe_relative_l2": series[
            "one_sided_probe_relative_l2"
        ],
        "second_difference_asymmetry": series[
            "second_difference_asymmetry"
        ],
        "zero_pair": series["zero_pair"],
    }


def _metrics_summary(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    runtime_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    breakpoint = {
        key: value
        for key, value in result["breakpoint"].items()
        if key not in {"breakpoints_m", "multiplicity"}
    }
    chord = result["chord_control"]
    return {
        "experiment_status": result["experiment_status"],
        "diagnostic_status": result["diagnostic_status"],
        "interpretation": result["interpretation"],
        "truth_identity": config["experiment"]["truth_identity"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "source_replay_relative_l2": result["source_replay_relative_l2"],
        "q8_deterministic_repeat_relative_l2_max": result[
            "q8_deterministic_repeat_relative_l2_max"
        ],
        "derived_diameters": result["derived_diameters"],
        "breakpoint": breakpoint,
        "q8": {
            "series": _series_summary(result["q8"]["series"]),
            "plateau_geometry": result["q8"]["plateau_geometry"],
            "geometry_step_minus": _stack_records(
                result["q8"]["geometry_step_minus"]
            ),
            "geometry_step_plus": _stack_records(
                result["q8"]["geometry_step_plus"]
            ),
        },
        "chord_control": {
            "geometry_order_relative_l2": chord[
                "geometry_order_relative_l2"
            ],
            "geometry_volume_relative_error": chord[
                "geometry_volume_relative_error"
            ],
            "geometry_order_pass": chord["geometry_order_pass"],
            "deterministic_repeat_relative_l2": chord[
                "deterministic_repeat_relative_l2"
            ],
            "q8_to_chord_center_relative_l2": chord[
                "q8_to_chord_center_relative_l2"
            ],
            "derivative_pass": chord["derivative_pass"],
            "series": _series_summary(chord["series"]),
        },
        "selected_slice_index": result["selected_slice_index"],
        "gates": result["gates"],
        "runtime_seconds": runtime_seconds,
        "peak_rss_bytes_sampled": peak_rss_bytes,
        "runtime_within_registered_contract": bool(
            runtime_seconds <= config["output"]["expected_runtime_seconds_max"]
        ),
        "memory_within_registered_contract": bool(
            peak_rss_bytes
            <= config["output"]["expected_peak_memory_bytes_max"]
        ),
    }


def _metadata(
    config: dict[str, Any],
    config_path: Path,
    source: Exp051SourceArtifact,
    prior: PriorExp051Artifact,
) -> dict[str, Any]:
    return {
        "experiment_id": "exp051",
        "created_at_utc": created_at_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_config": str(config_path),
        "run_role": config["experiment"]["role"],
        "scientific_claim_boundary": (
            "q8 local-differentiability attribution within the selected "
            "exp040 scalar working model only"
        ),
        "truth_identity": config["experiment"]["truth_identity"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "primary_input_is_raw_p_b_true": True,
        "p_b_rec_used_as_primary_input": False,
        "phase_or_scale_alignment": "none",
        "source_run": str(config["source"]["run"]),
        "source_hdf5_path": str(source.hdf5_path),
        "source_target_hdf5_path": source.target_hdf5_path,
        "source_file_sha256": source.file_sha256,
        "source_target_dataset_sha256": source.target_dataset_sha256,
        "prior_exp051_run": str(prior.run_dir),
        "prior_exp051_file_sha256": prior.file_sha256,
        "prior_exp051_interpretation": prior.metrics["interpretation"],
        "q8_primary_operator_unchanged": True,
        "chord_control_is_non_primary": True,
    }


def _data_payload(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    chord = result["chord_control"]
    return {
        "local_differentiability": {
            "design": config["local_control"],
            "derived_diameters": result["derived_diameters"],
            "breakpoint": result["breakpoint"],
            "q8": {
                "series": result["q8"]["series"],
                "plateau_geometry": result["q8"]["plateau_geometry"],
                "geometry_step_minus": _stack_records(
                    result["q8"]["geometry_step_minus"]
                ),
                "geometry_step_plus": _stack_records(
                    result["q8"]["geometry_step_plus"]
                ),
                "plateau_probe": result["q8"]["plateau_probe"],
                "truth_fraction_selected_slice": result["q8"][
                    "truth_fraction_selected_slice"
                ],
            },
            "chord_control": {
                "geometry_order_relative_l2": chord[
                    "geometry_order_relative_l2"
                ],
                "geometry_volume_relative_error": chord[
                    "geometry_volume_relative_error"
                ],
                "geometry_order_pass": chord["geometry_order_pass"],
                "deterministic_repeat_relative_l2": chord[
                    "deterministic_repeat_relative_l2"
                ],
                "q8_to_chord_center_relative_l2": chord[
                    "q8_to_chord_center_relative_l2"
                ],
                "derivative_pass": chord["derivative_pass"],
                "series": chord["series"],
                "order64_fraction_selected_slice": chord[
                    "order64_fraction_selected_slice"
                ],
                "order128_fraction_selected_slice": chord[
                    "order128_fraction_selected_slice"
                ],
            },
            "selected_slice_index": result["selected_slice_index"],
            "gates": result["gates"],
        }
    }


def _save_figures(result: dict[str, Any], paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    steps_um = np.asarray(result["q8"]["series"]["steps_m"]) * 1.0e6
    q8_response = np.asarray(
        result["q8"]["series"]["one_sided_probe_relative_l2"]
    )
    q8_minus_nodes = np.asarray(
        [
            row["changed_subpixel_node_count"]
            for row in result["q8"]["geometry_step_minus"]
        ]
    )
    q8_plus_nodes = np.asarray(
        [
            row["changed_subpixel_node_count"]
            for row in result["q8"]["geometry_step_plus"]
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].loglog(steps_um, q8_response[:, 0], "o-", label="minus")
    axes[0].loglog(steps_um, q8_response[:, 1], "o-", label="plus")
    axes[0].set_xlabel("finite-difference step h (um)")
    axes[0].set_ylabel("raw probe relative L2")
    axes[0].set_title("q8 probe response")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogx(steps_um, q8_minus_nodes, "o-", label="minus")
    axes[1].semilogx(steps_um, q8_plus_nodes, "o-", label="plus")
    axes[1].set_xlabel("finite-difference step h (um)")
    axes[1].set_ylabel("changed q8 subpixel nodes")
    axes[1].set_title("fixed-node geometry transitions")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle(
        f"exp051 q8 breakpoint response — {result['diagnostic_status']}"
    )
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)

    q8_series = result["q8"]["series"]
    chord_series = result["chord_control"]["series"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].loglog(
        steps_um,
        q8_series["normalized_jacobian_per_m"],
        "o-",
        label="q8 midpoint",
    )
    if chord_series is not None:
        axes[0].loglog(
            steps_um,
            chord_series["normalized_jacobian_per_m"],
            "o-",
            label="chord GL64",
        )
    axes[0].set_xlabel("finite-difference step h (um)")
    axes[0].set_ylabel("normalized ||J_h|| (1/m)")
    axes[0].set_title("local signature")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    pair_steps_um = steps_um[1:]
    axes[1].loglog(
        pair_steps_um,
        q8_series["pairwise_jacobian_relative_l2"],
        "o-",
        label="q8 midpoint",
    )
    if chord_series is not None:
        axes[1].loglog(
            pair_steps_um,
            chord_series["pairwise_jacobian_relative_l2"],
            "o-",
            label="chord GL64",
        )
    axes[1].axhline(0.10, color="black", linestyle="--", label="chord gate")
    axes[1].set_xlabel("finer step in halving pair (um)")
    axes[1].set_ylabel("relative L2(J_h, J_h/2)")
    axes[1].set_title("step convergence")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle("q8 midpoint versus non-primary chord-cell control")
    fig.savefig(paths[1], dpi=160)
    plt.close(fig)

    q8_jacobian = np.asarray(q8_series["jacobian"])
    if chord_series is None:
        chord_field = np.zeros_like(q8_jacobian[-1])
    else:
        chord_field = np.asarray(chord_series["jacobian"])[-1]
    panels = [
        (np.abs(q8_jacobian[1]), "q8 |J|, h=0.25 um"),
        (np.abs(q8_jacobian[2]), "q8 |J|, h=0.125 um"),
        (np.abs(chord_field), "chord |J|, h=0.00390625 um"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, (image, title) in zip(axes, panels, strict=True):
        shown = axis.imshow(image, origin="lower", cmap="magma")
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.8)
    fig.suptitle("Representative raw-complex Jacobian magnitudes (field/m)")
    fig.savefig(paths[2], dpi=160)
    plt.close(fig)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_json_group(
    group: h5py.Group, payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if set(group) != set(payload):
        errors.append(f"key mismatch at {group.name}")
    for key, expected in payload.items():
        if key not in group:
            continue
        node = group[key]
        if isinstance(expected, dict):
            if not isinstance(node, h5py.Group):
                errors.append(f"expected group at {node.name}")
            else:
                errors.extend(_compare_json_group(node, expected))
            continue
        actual = node[()]
        expected_array = np.asarray(expected)
        if expected_array.ndim > 0:
            if actual.dtype.kind in "SO":
                actual = np.asarray([_decode(value) for value in actual])
            if not np.array_equal(actual, expected_array):
                errors.append(f"array mismatch at {node.name}")
        elif _decode(actual) != expected:
            errors.append(f"scalar mismatch at {node.name}")
    return errors


def _all_numeric_finite(group: h5py.Group) -> bool:
    finite = True

    def visitor(_: str, node: h5py.Group | h5py.Dataset) -> None:
        nonlocal finite
        if isinstance(node, h5py.Dataset) and node.dtype.kind in "biufc":
            finite = bool(finite and np.all(np.isfinite(node[...])))

    group.visititems(visitor)
    return finite


def _validate_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    source: Exp051SourceArtifact,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    if load_config(run_dir / "config.yaml") != config:
        raise RuntimeError("Run config differs from executed local-control config.")
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        saved_metadata = json.load(handle)
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        saved_metrics = json.load(handle)
    if saved_metadata != metadata or saved_metrics != json.loads(
        json.dumps(metrics, default=lambda value: value.tolist())
    ):
        raise RuntimeError("In-memory and external JSON artifacts differ.")
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    with h5py.File(hdf5_path, "r") as h5:
        expected_entry = {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        if set(h5["entry"]) != expected_entry:
            raise RuntimeError("Unexpected exp051 local-control /entry tree.")
        if set(h5["entry/data"]) != {"local_differentiability"}:
            raise RuntimeError("Local-control data group differs from contract.")
        names: list[str] = []
        h5["entry"].visit(names.append)
        if any(name.rsplit("/", 1)[-1] == "P_B_rec" for name in names):
            raise RuntimeError("exp051 local control may not contain P_B_rec.")
        if not np.array_equal(h5["entry/truth/P_B_true"][...], source.P_B_true):
            raise RuntimeError("Saved target differs from source P_B_true.")
        if _compare_json_group(h5["entry/metadata"], metadata):
            raise RuntimeError("metadata.json and HDF5 metadata differ.")
        if _compare_json_group(h5["entry/metrics"], metrics):
            raise RuntimeError("metrics.json and HDF5 metrics differ.")
        if not _all_numeric_finite(h5["entry"]):
            raise RuntimeError("Local-control HDF5 contains non-finite data.")
        if bool(h5["entry/truth/reference_validated"][()]) or bool(
            h5["entry/truth/full_tgv_reference_authorized"][()]
        ):
            raise RuntimeError("Local-control truth flags were promoted.")
        root = h5["entry/data/local_differentiability"]
        q8_cache = root["q8/series/cache/P_B_candidate"]
        if q8_cache.shape != (17, 96, 96) or q8_cache.dtype != np.complex128:
            raise RuntimeError("Unexpected q8 local-control cache identity.")
        chord_series = root.get("chord_control/series")
        if chord_series is not None:
            chord_cache = chord_series["cache/P_B_candidate"]
            if chord_cache.shape != (17, 96, 96):
                raise RuntimeError("Unexpected chord local-control cache shape.")
    figure_paths = sorted((run_dir / "figures").glob("*.png"))
    if [path.name for path in figure_paths] != sorted(
        config["output"]["figure_filenames"]
    ):
        raise RuntimeError("Local-control figure set differs from config.")
    for path in figure_paths:
        image = plt.imread(path)
        if image.size == 0 or not np.all(np.isfinite(image)):
            raise RuntimeError(f"Unreadable local-control figure: {path}")
    if hdf5_path.stat().st_size > config["output"]["expected_hdf5_bytes_max"]:
        raise RuntimeError("Local-control HDF5 exceeds the registered limit.")


def run(config_path: Path) -> Path:
    resolved_config = config_path
    if not resolved_config.is_absolute():
        resolved_config = (PROJECT_ROOT / resolved_config).resolve()
    config = load_config(resolved_config)
    validate_exp051_local_control_config(config)
    output_root = Path(config["output"]["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = make_run_dir(output_root, config["output"]["run_name"])
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {"status": "running", "artifacts_validated": False},
    )
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    def sample_memory() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)

    try:
        source = load_exp051_source_true_probe(config, PROJECT_ROOT)
        prior = load_prior_exp051_formal(config, PROJECT_ROOT)
        q8_generator = make_exp051_candidate_generator(
            source.source_config, memory_callback=sample_memory
        )
        chord_generator = make_exp051_candidate_generator(
            source.source_config,
            memory_callback=sample_memory,
            air_fraction_builder=make_tgv_air_fraction_slice_chord_quadrature,
            interface_resolution=int(
                config["local_control"]["chord_formal_order"]
            ),
        )
        result = run_local_differentiability_control(
            config,
            source.source_config,
            source.P_B_true,
            q8_generator,
            chord_generator,
            memory_callback=sample_memory,
        )
        sample_memory()
        runtime_seconds = time.perf_counter() - started
        metadata = _metadata(config, resolved_config, source, prior)
        metrics = _metrics_summary(
            config,
            result,
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss,
        )
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
        identity = config["source"]["expected_identity"]
        save_ptycho_hdf5(
            run_dir / "outputs" / config["output"]["hdf5_filename"],
            data=_data_payload(config, result),
            instrument={
                "wavelength_m": source.source_config["optics"]["wavelength_m"],
                "internal_reference_index": source.source_config["optics"][
                    "internal_reference_index"
                ],
                "external_medium_index": source.source_config["optics"][
                    "external_medium_index"
                ],
                "z_AB_m": source.source_config["optics"]["z_AB_m"],
                "probe_grid": identity,
            },
            sample={"sample_a": source.source_config["sample_a"]},
            truth={
                "identity": config["experiment"]["truth_identity"],
                "reference_validated": False,
                "full_tgv_reference_authorized": False,
                "P_B_true": source.P_B_true,
                "D_waist_true_m": config["fit"]["true_d_waist_m"],
                "D_z_m": source.D_z_m,
                "z_m": source.z_m,
                "slice_widths_m": source.slice_widths_m,
            },
            config_yaml=config_to_yaml(config),
            metadata=metadata,
            metrics=metrics,
        )
        figure_paths = [
            run_dir / "figures" / name
            for name in config["output"]["figure_filenames"]
        ]
        _save_figures(result, figure_paths)
        _validate_artifacts(
            run_dir, config, source, metadata, metrics
        )
        state = {
            "status": "complete",
            "artifacts_validated": True,
            "completed_at_utc": created_at_utc(),
            "runtime_seconds": runtime_seconds,
            "diagnostic_status": result["diagnostic_status"],
            "config_sha256": sha256_file(run_dir / "config.yaml"),
            "metadata_sha256": sha256_file(run_dir / "metadata.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
            "hdf5_sha256": sha256_file(
                run_dir / "outputs" / config["output"]["hdf5_filename"]
            ),
            "figure_sha256": {
                path.name: sha256_file(path) for path in figure_paths
            },
            "figure_count": len(figure_paths),
        }
        save_json(run_dir / "run_state.json", state)
    except Exception as error:
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "artifacts_validated": False,
                "failed_at_utc": created_at_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    print(f"run_dir: {run_dir}")
    print(f"diagnostic_status: {result['diagnostic_status']}")
    print(f"interpretation: {result['interpretation']}")
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
