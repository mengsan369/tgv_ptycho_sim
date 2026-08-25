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
from tgv_ptycho.inverse.exp051_plateau_interval import (  # noqa: E402
    PriorQ8LocalControlArtifact,
    load_prior_q8_local_control,
    run_q8_plateau_interval_fit,
    validate_exp051_plateau_interval_config,
)
from tgv_ptycho.io.config import (  # noqa: E402
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit  # noqa: E402
from tgv_ptycho.io.naming import make_run_dir  # noqa: E402
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exp051 fixed-q8 plateau-interval oracle fit."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _component_summary(component: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed_member": component["seed_member"],
        "left_cell_index": component["left_cell_index"],
        "right_cell_index": component["right_cell_index"],
        "left_outside_cell_index": component["left_outside_cell_index"],
        "right_outside_cell_index": component["right_outside_cell_index"],
        "lower_boundary_m": component["lower_boundary_m"],
        "upper_boundary_m": component["upper_boundary_m"],
        "lower_closed": component["lower_closed"],
        "upper_closed": component["upper_closed"],
        "left_cap_hit": component["left_cap_hit"],
        "right_cap_hit": component["right_cap_hit"],
        "queried_cell_count": len(component["queried_cell_index"]),
    }


def _metrics_summary(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    runtime_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    optimizer_branches = {
        name: {
            "start_m": branch["start_m"],
            "evaluation_budget": branch["evaluation_budget"],
            "evaluation_count": branch["evaluation_count"],
            "stopping_reason": branch["stopping_reason"],
            "final_estimate_m": branch["final_estimate_m"],
            "final_loss": branch["final_loss"],
            "final_raw_relative_l2": branch["final_raw_relative_l2"],
            "final_seed_qualifies": branch["final_seed_qualifies"],
            "final_cell_index": branch["final_cell_index"],
        }
        for name, branch in result["optimizer"]["branches"].items()
    }
    interval_branches = {
        name: {
            "seed_m": branch["seed_m"],
            "seed_loss": branch["seed_loss"],
            "seed_cell_index": branch["seed_cell_index"],
            "components": {
                component_name: _component_summary(component)
                for component_name, component in branch["components"].items()
            },
        }
        for name, branch in result["interval_by_branch"].items()
    }
    bisection = {
        branch_name: {
            side: {
                "iterations": control["iterations"],
                "analytic_breakpoint_m": control["analytic_breakpoint_m"],
                "analytic_breakpoint_in_bracket": control[
                    "analytic_breakpoint_in_bracket"
                ],
                "bracket_lower_m": control["bracket_lower_m"],
                "bracket_upper_m": control["bracket_upper_m"],
                "bracket_width_m": control["bracket_width_m"],
                "pass": control["pass"],
            }
            for side, control in branch.items()
        }
        for branch_name, branch in result["bisection"].items()
    }
    geometry = result["geometry_control"]
    return {
        "experiment_status": result["experiment_status"],
        "interpretation": result["interpretation"],
        "truth_identity": config["experiment"]["truth_identity"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "p_b_rec_used_as_primary_input": False,
        "replay": result["replay"],
        "tau": result["tau"],
        "breakpoint_count": len(result["partition"]["breakpoints_m"]),
        "cell_count": len(result["partition"]["cell_edges_m"]) - 1,
        "profile": {
            "coarse_d_waist_m": result["profile"]["coarse_d_waist_m"],
            "coarse_loss": result["profile"]["coarse_loss"],
            "fine_d_waist_m": result["profile"]["fine_d_waist_m"],
            "fine_loss": result["profile"]["fine_loss"],
        },
        "optimizer": {
            "algorithm": result["optimizer"]["algorithm"],
            "equal_budget": result["optimizer"]["equal_budget"],
            "all_final_seeds_qualify": result["optimizer"][
                "all_final_seeds_qualify"
            ],
            "interval_agreement": result["optimizer"]["interval_agreement"],
            "branches": optimizer_branches,
        },
        "interval_by_branch": interval_branches,
        "reported_interval": result["reported_interval"],
        "outside_control": result["outside_control"],
        "geometry_control": {
            "diameter_m": geometry["diameter_m"],
            "node_count_sha256": geometry["node_count_sha256"],
            "equal_to_seed": geometry["equal_to_seed"],
        },
        "endpoint_field_control": result["endpoint_field_control"],
        "bisection": bisection,
        "deterministic_repeat": result["deterministic_repeat"],
        "profile_screen": {
            "qualifying_count": result["profile_screen"]["qualifying_count"],
            "external_qualifying_component_observed": result[
                "profile_screen"
            ]["external_qualifying_component_observed"],
        },
        "candidate_cache_count": len(result["cache"]["D_waist_m"]),
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
    prior: PriorQ8LocalControlArtifact,
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
            "interval-valued fixed-q8 true-probe oracle fit within the "
            "selected exp040 scalar working model only"
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
        "prior_q8_local_control_run": str(prior.run_dir),
        "prior_q8_local_control_file_sha256": prior.file_sha256,
        "prior_q8_local_control_interpretation": prior.metrics[
            "interpretation"
        ],
        "q8_primary_operator_unchanged": True,
        "chord_primary_operator_used": False,
        "set_valued_primary_estimate": True,
    }


def _reconstruction_payload(
    config: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "waist_fit": {
            "plateau_interval": {
                "design": config["plateau_interval"],
                "source": {
                    "target_hdf5_path": config["source"]["target_hdf5_path"],
                    "prior_q8_local_control": config[
                        "prior_q8_local_control"
                    ],
                },
                "partition": result["partition"],
                "profile": result["profile"],
                "optimizer": result["optimizer"],
                "interval_by_branch": result["interval_by_branch"],
                "reported_interval": result["reported_interval"],
                "outside_control": result["outside_control"],
                "geometry_control": result["geometry_control"],
                "endpoint_field_control": result["endpoint_field_control"],
                "bisection": result["bisection"],
                "deterministic_repeat": result["deterministic_repeat"],
                "profile_screen": result["profile_screen"],
                "cache": result["cache"],
                "P_B_interval_midpoint_raw": result[
                    "P_B_interval_midpoint_raw"
                ],
                "residual_field_raw": result["residual_field_raw"],
                "gates": result["gates"],
            }
        }
    }


def _save_figures(result: dict[str, Any], paths: list[Path]) -> None:
    interval = result["reported_interval"]
    tau_loss = float(result["tau"]["primary_loss_threshold"])
    profile = result["profile"]

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.semilogy(
        np.asarray(profile["coarse_d_waist_m"]) * 1.0e6,
        np.maximum(profile["coarse_loss"], np.finfo(np.float64).tiny),
        "o-",
        label="coarse profile",
    )
    axis.semilogy(
        np.asarray(profile["fine_d_waist_m"]) * 1.0e6,
        np.maximum(profile["fine_loss"], np.finfo(np.float64).tiny),
        "s-",
        label="fine profile",
    )
    axis.axhline(tau_loss, color="black", linestyle="--", label="tau²")
    axis.axvspan(
        interval["lower_m"] * 1.0e6,
        interval["upper_m"] * 1.0e6,
        color="tab:green",
        alpha=0.25,
        label="reported interval",
    )
    axis.set_xlabel("D_waist (um)")
    axis.set_ylabel("raw normalized complex loss")
    axis.set_title(f"exp051 q8 global profile — {result['experiment_status']}")
    axis.legend()
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)

    first_bisection = next(iter(result["bisection"].values()))
    local_d = [np.asarray(result["endpoint_field_control"]["diameter_m"])]
    local_rho = [
        np.asarray(result["endpoint_field_control"]["raw_relative_l2"])
    ]
    for side in ("lower", "upper"):
        local_d.append(first_bisection[side]["evaluated_d_waist_m"])
        local_rho.append(np.sqrt(first_bisection[side]["evaluated_loss"]))
    local_d_values = np.concatenate(local_d)
    local_rho_values = np.concatenate(local_rho)
    order = np.argsort(local_d_values, kind="stable")
    center_m = 0.5 * (interval["lower_m"] + interval["upper_m"])
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.semilogy(
        (local_d_values[order] - center_m) * 1.0e9,
        np.maximum(local_rho_values[order], np.finfo(np.float64).tiny),
        ".-",
        markersize=4,
    )
    axis.axhline(
        result["tau"]["tau_primary"], color="black", linestyle="--"
    )
    axis.axvline(
        (interval["lower_m"] - center_m) * 1.0e9,
        color="tab:green",
        linestyle=":",
    )
    axis.axvline(
        (interval["upper_m"] - center_m) * 1.0e9,
        color="tab:green",
        linestyle=":",
    )
    axis.set_xlabel("D_waist - interval midpoint (nm)")
    axis.set_ylabel("raw complex relative L2")
    axis.set_title("q8 local staircase and fixed-budget boundary brackets")
    fig.savefig(paths[1], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    branch_items = list(result["interval_by_branch"].items())
    for row, (_name, branch) in enumerate(branch_items):
        component = branch["components"]["tau_primary"]
        axis.hlines(
            row,
            (component["lower_boundary_m"] - center_m) * 1.0e9,
            (component["upper_boundary_m"] - center_m) * 1.0e9,
            linewidth=5,
            color="tab:green",
        )
        axis.plot(
            (branch["seed_m"] - center_m) * 1.0e9,
            row,
            "o",
            color="tab:blue",
        )
    axis.set_yticks(range(len(branch_items)), [name for name, _ in branch_items])
    axis.set_xlabel("D_waist - common interval midpoint (nm)")
    axis.set_title("Equal-budget multi-start interval agreement")
    fig.savefig(paths[2], dpi=160)
    plt.close(fig)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_json_group(group: h5py.Group, payload: dict[str, Any]) -> list[str]:
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
    result: dict[str, Any],
) -> None:
    if load_config(run_dir / "config.yaml") != config:
        raise RuntimeError("Run config differs from executed plateau config.")
    saved_metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    saved_metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    normalized_metrics = json.loads(
        json.dumps(metrics, default=lambda value: value.tolist())
    )
    if saved_metadata != metadata or saved_metrics != normalized_metrics:
        raise RuntimeError("In-memory and external JSON artifacts differ.")
    hdf5_path = run_dir / "outputs" / config["output"]["hdf5_filename"]
    with h5py.File(hdf5_path, "r") as h5:
        expected_entry = {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "reconstruction",
            "sample",
            "truth",
        }
        if set(h5["entry"]) != expected_entry:
            raise RuntimeError("Unexpected exp051 plateau /entry tree.")
        if set(h5["entry/data"]):
            raise RuntimeError("exp051 plateau /entry/data must remain empty.")
        names: list[str] = []
        h5["entry"].visit(names.append)
        if any(name.rsplit("/", 1)[-1] == "P_B_rec" for name in names):
            raise RuntimeError("exp051 plateau fit may not contain P_B_rec.")
        if "calibration" in h5["entry"] or "preprocessing" in h5["entry"]:
            raise RuntimeError("exp051 plateau fit wrote a fake processing group.")
        if not np.array_equal(h5["entry/truth/P_B_true"][...], source.P_B_true):
            raise RuntimeError("Saved target differs from source P_B_true.")
        if _compare_json_group(h5["entry/metadata"], metadata):
            raise RuntimeError("metadata.json and HDF5 metadata differ.")
        if _compare_json_group(h5["entry/metrics"], metrics):
            raise RuntimeError("metrics.json and HDF5 metrics differ.")
        if not _all_numeric_finite(h5["entry"]):
            raise RuntimeError("Plateau-interval HDF5 contains non-finite data.")
        if bool(h5["entry/truth/reference_validated"][()]) or bool(
            h5["entry/truth/full_tgv_reference_authorized"][()]
        ):
            raise RuntimeError("Plateau-interval provenance flags were promoted.")
        root = h5["entry/reconstruction/waist_fit/plateau_interval"]
        cache = root["cache"]
        if (
            cache["P_B_candidate"].shape[1:] != (96, 96)
            or cache["P_B_candidate"].dtype != np.complex128
            or len(cache["D_waist_m"]) != len(result["cache"]["D_waist_m"])
        ):
            raise RuntimeError("Unexpected plateau candidate cache identity.")
        midpoint_index = int(root["reported_interval/midpoint_cache_index"][()])
        if not np.array_equal(
            root["P_B_interval_midpoint_raw"][...],
            cache["P_B_candidate"][midpoint_index],
        ):
            raise RuntimeError("Interval midpoint probe/cache mapping differs.")
        if not np.array_equal(
            root["P_B_interval_midpoint_raw"][...] - source.P_B_true,
            root["residual_field_raw"][...],
        ):
            raise RuntimeError("Saved raw residual identity differs.")
        branches = root["optimizer/branches"]
        if len(branches) != 4 or any(
            int(branch["evaluation_count"][()]) != 41
            for branch in branches.values()
        ):
            raise RuntimeError("Optimizer equal-budget artifact differs.")
        bisection = root["bisection"]
        if any(
            int(side["iterations"][()]) != 24
            for branch in bisection.values()
            for side in branch.values()
        ):
            raise RuntimeError("Boundary bisection budget differs.")
    if hdf5_path.stat().st_size > config["output"]["expected_hdf5_bytes_max"]:
        raise RuntimeError("Plateau-interval HDF5 exceeds registered limit.")
    figure_paths = sorted((run_dir / "figures").glob("*.png"))
    if [path.name for path in figure_paths] != sorted(
        config["output"]["figure_filenames"]
    ):
        raise RuntimeError("Plateau-interval figure set differs from config.")
    for path in figure_paths:
        image = plt.imread(path)
        if image.size == 0 or not np.all(np.isfinite(image)):
            raise RuntimeError(f"Unreadable plateau-interval figure: {path}")


def run(config_path: Path) -> Path:
    resolved_config = config_path
    if not resolved_config.is_absolute():
        resolved_config = (PROJECT_ROOT / resolved_config).resolve()
    config = load_config(resolved_config)
    validate_exp051_plateau_interval_config(config)
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
        prior = load_prior_q8_local_control(config, PROJECT_ROOT)
        generator = make_exp051_candidate_generator(
            source.source_config, memory_callback=sample_memory
        )
        result = run_q8_plateau_interval_fit(
            config, source.source_config, source.P_B_true, generator
        )
        sample_memory()
        runtime_seconds = time.perf_counter() - started
        runtime_pass = bool(
            runtime_seconds <= config["output"]["expected_runtime_seconds_max"]
        )
        memory_pass = bool(
            peak_rss <= config["output"]["expected_peak_memory_bytes_max"]
        )
        result["gates"]["runtime_contract_pass"] = runtime_pass
        result["gates"]["memory_contract_pass"] = memory_pass
        if not runtime_pass or not memory_pass:
            result["experiment_status"] = "Inconclusive"
            result["interpretation"] = "runtime_or_memory_contract_not_closed"
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
            reconstruction=_reconstruction_payload(config, result),
            config_yaml=config_to_yaml(config),
            metadata=metadata,
            metrics=metrics,
        )
        figure_paths = [
            run_dir / "figures" / name
            for name in config["output"]["figure_filenames"]
        ]
        _save_figures(result, figure_paths)
        _validate_artifacts(run_dir, config, source, metadata, metrics, result)
        state = {
            "status": "complete",
            "artifacts_validated": True,
            "completed_at_utc": created_at_utc(),
            "runtime_seconds": runtime_seconds,
            "experiment_status": result["experiment_status"],
            "interpretation": result["interpretation"],
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
    print(f"experiment_status: {result['experiment_status']}")
    print(f"interpretation: {result['interpretation']}")
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
