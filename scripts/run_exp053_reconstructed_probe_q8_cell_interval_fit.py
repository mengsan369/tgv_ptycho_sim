from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tgv_ptycho.inverse.exp051 import sha256_file  # noqa: E402
from tgv_ptycho.inverse.exp053 import (  # noqa: E402
    Exp051OracleIntervalArtifact,
    Exp053SourceArtifact,
    load_exp051_oracle_interval,
    load_exp053_source_reconstructed_probe,
    make_exp053_candidate_generator,
    run_reconstructed_probe_q8_interval_fit,
    validate_exp053_config,
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
        description="Run exp053 reconstructed-probe q8 cell interval fit."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _branch_summary(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_m": branch["start_m"],
        "evaluation_budget": branch["evaluation_budget"],
        "evaluation_count": branch["evaluation_count"],
        "stopping_reason": branch["stopping_reason"],
        "final_estimate_m": branch["final_estimate_m"],
        "final_loss": branch["final_loss"],
        "final_raw_relative_l2": branch["final_raw_relative_l2"],
        "final_cell_index": branch["final_cell_index"],
        "final_best_field_relative_l2": branch[
            "final_best_field_relative_l2"
        ],
        "final_seed_qualifies": branch["final_seed_qualifies"],
    }


def _interval_branch_summary(item: dict[str, Any]) -> dict[str, Any]:
    component = item["primary_component"]
    return {
        "seed_m": item["seed_m"],
        "seed_loss": item["seed_loss"],
        "seed_cell_index": item["seed_cell_index"],
        "final_best_field_relative_l2": item[
            "final_best_field_relative_l2"
        ],
        "final_seed_qualifies": item["final_seed_qualifies"],
        "lower_m": item["source_exact_lower_boundary_m"],
        "upper_m": item["source_exact_upper_boundary_m"],
        "analytic_lower_m": item["analytic_lower_boundary_m"],
        "analytic_upper_m": item["analytic_upper_boundary_m"],
        "boundary_semantics": "source_exact_float64_membership_transition",
        "lower_closed": component["lower_closed"],
        "upper_closed": component["upper_closed"],
        "left_cell_index": component["left_cell_index"],
        "right_cell_index": component["right_cell_index"],
    }


def _bisection_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for branch_name, branch in result["bisection"].items():
        summary[branch_name] = {}
        for side, control in branch.items():
            summary[branch_name][side] = {
                "iterations": control["iterations"],
                "bracket_lower_m": control["bracket_lower_m"],
                "bracket_upper_m": control["bracket_upper_m"],
                "bracket_width_m": control["bracket_width_m"],
                "analytic_breakpoint_m": control["analytic_breakpoint_m"],
                "analytic_breakpoint_in_bracket": control[
                    "analytic_breakpoint_in_bracket"
                ],
                "source_exact_breakpoint_m": control[
                    "source_exact_breakpoint_m"
                ],
                "source_exact_ulp_offset_from_analytic": control[
                    "source_exact_ulp_offset_from_analytic"
                ],
                "source_exact_breakpoint_in_bracket": control[
                    "source_exact_breakpoint_in_bracket"
                ],
                "source_exact_mapping_pass": control[
                    "source_exact_mapping_pass"
                ],
                "pass": control["pass"],
            }
    return summary


def _metrics_summary(
    config: dict[str, Any],
    source: Exp053SourceArtifact,
    oracle: Exp051OracleIntervalArtifact,
    result: dict[str, Any],
    *,
    runtime_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    branches = {
        name: _branch_summary(branch)
        for name, branch in result["optimizer"]["branches"].items()
    }
    interval_branches = {
        name: _interval_branch_summary(item)
        for name, item in result["interval_by_branch"].items()
    }
    return {
        "experiment_status": result["experiment_status"],
        "interpretation": result["interpretation"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "primary_input_is_raw_matched_q4_p_b_rec": True,
        "truth_used_by_fitter": False,
        "phase_or_scale_alignment": "none",
        "source": {
            "run": str(source.run_dir),
            "hdf5_path": str(source.hdf5_path),
            "target_hdf5_path": source.target_hdf5_path,
            "operator_reference_hdf5_path": (
                source.operator_reference_hdf5_path
            ),
            "file_sha256": source.file_sha256,
            "target_dataset_sha256": source.target_dataset_sha256,
            "operator_reference_dataset_sha256": (
                source.operator_reference_dataset_sha256
            ),
            "target_shape": list(source.P_B_rec_raw.shape),
            "target_dtype": str(source.P_B_rec_raw.dtype),
        },
        "oracle_source": {
            "run": str(oracle.run_dir),
            "hdf5_path": str(oracle.hdf5_path),
            "file_sha256": oracle.file_sha256,
            "lower_m": oracle.lower_m,
            "upper_m": oracle.upper_m,
            "lower_closed": oracle.lower_closed,
            "upper_closed": oracle.upper_closed,
        },
        "candidate_operator_replay": result["candidate_operator_replay"],
        "reconstructed_target_mismatch": result[
            "reconstructed_target_mismatch"
        ],
        "field_equivalence_threshold": result[
            "field_equivalence_threshold"
        ],
        "source_exact_boundary_control": result[
            "source_exact_boundary_control"
        ],
        "partition": {
            "breakpoint_count": len(result["partition"]["breakpoints_m"]),
            "cell_count": len(result["partition"]["cell_edges_m"]) - 1,
        },
        "profile": {
            "coarse_d_waist_m": result["profile"]["coarse_d_waist_m"],
            "coarse_loss": result["profile"]["coarse_loss"],
            "fine_d_waist_m": result["profile"]["fine_d_waist_m"],
            "fine_loss": result["profile"]["fine_loss"],
        },
        "optimizer": {
            "algorithm": result["optimizer"]["algorithm"],
            "numerical_closure_control": result["optimizer"][
                "numerical_closure_control"
            ],
            "equal_budget": result["optimizer"]["equal_budget"],
            "all_final_seeds_qualify": result["optimizer"][
                "all_final_seeds_qualify"
            ],
            "interval_agreement": result["optimizer"][
                "interval_agreement"
            ],
            "branches": branches,
        },
        "interval_by_branch": interval_branches,
        "reported_interval": result["reported_interval"],
        "loss_comparison": result["loss_comparison"],
        "oracle_interval_comparison": result["oracle_interval_comparison"],
        "simulation_evaluation_only": result[
            "simulation_evaluation_only"
        ],
        "outside_control": result["outside_control"],
        "deterministic_repeat": result["deterministic_repeat"],
        "profile_screen": {
            "loss_tied_count": result["profile_screen"]["loss_tied_count"],
            "external_loss_tied_cell_observed": result["profile_screen"][
                "external_loss_tied_cell_observed"
            ],
        },
        "bisection": _bisection_summary(result),
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
    source: Exp053SourceArtifact,
    oracle: Exp051OracleIntervalArtifact,
) -> dict[str, Any]:
    return {
        "experiment_id": "exp053",
        "created_at_utc": created_at_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_config": str(config_path),
        "run_role": config["experiment"]["role"],
        "scientific_claim_boundary": (
            "fixed-q8 single-parameter interval fit to one raw matched-q4 "
            "known-B noiseless reconstruction in the selected exp040 scalar "
            "working model only"
        ),
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "primary_input_is_raw_matched_q4_p_b_rec": True,
        "primary_input_truth_aligned": False,
        "truth_used_by_fitter": False,
        "phase_or_scale_alignment": "none",
        "source_run": str(source.run_dir),
        "source_hdf5_path": str(source.hdf5_path),
        "source_target_hdf5_path": source.target_hdf5_path,
        "source_target_dataset_sha256": source.target_dataset_sha256,
        "source_hdf5_sha256": source.file_sha256["hdf5"],
        "oracle_run": str(oracle.run_dir),
        "oracle_hdf5_sha256": oracle.file_sha256["hdf5"],
        "primary_interval_definition": config[
            "reconstruction_aware_interval"
        ]["primary_definition"],
        "optimizer_evaluation_budget_per_start": config["fit"]["optimizer"][
            "evaluation_budget_per_start"
        ],
        "optimizer_numerical_closure_control": config["fit"]["optimizer"][
            "numerical_closure_control"
        ],
        "source_exact_boundary_control": config[
            "reconstruction_aware_interval"
        ]["source_exact_boundary_control"],
        "reconstruction_error_used_to_scale_threshold": False,
    }


def _reconstruction_payload(
    config: dict[str, Any],
    source: Exp053SourceArtifact,
    oracle: Exp051OracleIntervalArtifact,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "waist_fit": {
            "reconstructed_target_q8_cell_interval": {
                "design": config["reconstruction_aware_interval"],
                "source": {
                    "raw_P_B_rec_input": source.P_B_rec_raw,
                    "target_hdf5_path": source.target_hdf5_path,
                    "operator_reference_hdf5_path": (
                        source.operator_reference_hdf5_path
                    ),
                    "file_sha256": source.file_sha256,
                    "target_dataset_sha256": source.target_dataset_sha256,
                    "operator_reference_dataset_sha256": (
                        source.operator_reference_dataset_sha256
                    ),
                    "plane": "B",
                    "axis_order": ["y", "x"],
                    "native_shape": [96, 96],
                    "dtype": "complex128",
                    "node_dx_m": 5.0e-7,
                    "raw_unaligned": True,
                },
                "oracle_source": {
                    "file_sha256": oracle.file_sha256,
                    "lower_m": oracle.lower_m,
                    "upper_m": oracle.upper_m,
                    "lower_closed": oracle.lower_closed,
                    "upper_closed": oracle.upper_closed,
                },
                **result,
            }
        }
    }


def _save_figures(
    result: dict[str, Any], paths: list[Path], diagnostic_path: Path | None
) -> None:
    profile = result["profile"]
    interval = result["reported_interval"]
    oracle = result["oracle_interval_comparison"]
    truth_m = result["simulation_evaluation_only"]["D_waist_true_m"]

    fig, axis = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    axis.semilogy(
        np.asarray(profile["coarse_d_waist_m"]) * 1.0e6,
        np.asarray(profile["coarse_loss"]),
        "o-",
        label="coarse raw target loss",
    )
    axis.semilogy(
        np.asarray(profile["fine_d_waist_m"]) * 1.0e6,
        np.asarray(profile["fine_loss"]),
        "s-",
        label="fine raw target loss",
    )
    axis.axvline(
        interval["lower_m"] * 1.0e6,
        color="tab:green",
        linestyle="--",
        label="reconstructed q8 endpoints",
    )
    axis.axvline(
        interval["upper_m"] * 1.0e6,
        color="tab:green",
        linestyle="--",
    )
    axis.axvline(
        oracle["oracle_lower_m"] * 1.0e6,
        color="tab:blue",
        linestyle="-.",
        label="exp051 oracle endpoints",
    )
    axis.axvline(
        oracle["oracle_upper_m"] * 1.0e6,
        color="tab:blue",
        linestyle="-.",
    )
    axis.axvline(
        truth_m * 1.0e6,
        color="black",
        linestyle=":",
        label="D true (simulation evaluation only)",
    )
    axis.set_xlabel("D_waist (um)")
    axis.set_ylabel("raw normalized complex loss")
    axis.set_title(
        "exp053 reconstructed-target profile — "
        f"{result['experiment_status']}"
    )
    axis.legend(fontsize=8)
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)

    center_m = float(interval["midpoint_m"])
    cache = result["cache"]
    cache_d = np.asarray(cache["D_waist_m"])
    cache_loss = np.asarray(cache["loss"])
    local = np.abs(cache_d - center_m) <= 4.0e-7
    order = np.argsort(cache_d[local], kind="stable")
    fig, axis = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    axis.semilogy(
        (cache_d[local][order] - center_m) * 1.0e9,
        cache_loss[local][order],
        ".",
        markersize=4,
        label="evaluated raw target loss",
    )
    axis.axvline(
        (interval["lower_m"] - center_m) * 1.0e9,
        color="tab:green",
        linestyle="--",
        label="reconstructed endpoints",
    )
    axis.axvline(
        (interval["upper_m"] - center_m) * 1.0e9,
        color="tab:green",
        linestyle="--",
    )
    axis.axvline(
        (oracle["oracle_lower_m"] - center_m) * 1.0e9,
        color="tab:blue",
        linestyle="-.",
        label="oracle endpoints",
    )
    axis.axvline(
        (oracle["oracle_upper_m"] - center_m) * 1.0e9,
        color="tab:blue",
        linestyle="-.",
    )
    axis.axvline(
        (truth_m - center_m) * 1.0e9,
        color="black",
        linestyle=":",
    )
    axis.set_xlabel("D_waist - reconstructed interval midpoint (nm)")
    axis.set_ylabel("raw reconstructed-target loss")
    axis.set_title("q8 local staircase and half-open interval endpoints")
    axis.legend(fontsize=8)
    fig.savefig(paths[1], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    for name, branch in result["optimizer"]["branches"].items():
        track = np.asarray(branch["incumbent_d_waist_m"]) * 1.0e6
        axis.plot(np.arange(len(track)), track, ".-", label=name)
    axis.axhline(
        interval["lower_m"] * 1.0e6,
        color="tab:green",
        linestyle="--",
        label="reconstructed endpoints",
    )
    axis.axhline(
        interval["upper_m"] * 1.0e6,
        color="tab:green",
        linestyle="--",
    )
    axis.axhline(
        oracle["oracle_lower_m"] * 1.0e6,
        color="tab:blue",
        linestyle="-.",
        label="oracle endpoints",
    )
    axis.axhline(
        oracle["oracle_upper_m"] * 1.0e6,
        color="tab:blue",
        linestyle="-.",
    )
    axis.set_xlabel("pattern-search update index")
    axis.set_ylabel("incumbent D_waist (um)")
    axis.set_title("Four equal-budget optimizer tracks")
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(paths[2], dpi=160)
    plt.close(fig)

    directional = result["simulation_evaluation_only"][
        "directional_diagnostic"
    ]
    if diagnostic_path is not None and directional["performed"]:
        directions = directional["directions"]
        names = list(directions)
        real_cosine = [directions[name]["real_cosine"] for name in names]
        coherence = [directions[name]["complex_coherence"] for name in names]
        x = np.arange(len(names))
        fig, axis = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
        axis.plot(x, real_cosine, "o-", label="real cosine")
        axis.plot(x, coherence, "s-", label="complex coherence")
        axis.set_xticks(x, names, rotation=15, ha="right")
        axis.set_ylim(-1.05, 1.05)
        axis.set_ylabel("normalized projection/correlation")
        axis.set_title("Reconstruction error vs waist-sensitive directions")
        axis.legend()
        fig.savefig(diagnostic_path, dpi=160)
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
    source: Exp053SourceArtifact,
    oracle: Exp051OracleIntervalArtifact,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    result: dict[str, Any],
) -> None:
    if load_config(run_dir / "config.yaml") != config:
        raise RuntimeError("Run config differs from executed exp053 config.")
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
        raise RuntimeError("In-memory and external exp053 JSON differ.")
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
        if set(h5["entry"]) != expected_entry or set(h5["entry/data"]):
            raise RuntimeError("Unexpected exp053 /entry layout.")
        if "calibration" in h5["entry"] or "preprocessing" in h5["entry"]:
            raise RuntimeError("exp053 wrote a fake processing group.")
        root = h5[
            "/entry/reconstruction/waist_fit/"
            "reconstructed_target_q8_cell_interval"
        ]
        if not np.array_equal(
            root["source/raw_P_B_rec_input"][...], source.P_B_rec_raw
        ):
            raise RuntimeError("Saved exp053 raw target differs from source.")
        names: list[str] = []
        root.visit(names.append)
        forbidden_aligned_tokens = (
            "global_phase_aligned",
            "truth_aligned",
            "p_b_rec_aligned",
        )
        if any(
            token in name.lower()
            for name in names
            for token in forbidden_aligned_tokens
        ):
            raise RuntimeError("exp053 artifact contains a forbidden aligned copy.")
        if _compare_json_group(h5["entry/metadata"], metadata):
            raise RuntimeError("metadata.json and exp053 HDF5 metadata differ.")
        if _compare_json_group(h5["entry/metrics"], metrics):
            raise RuntimeError("metrics.json and exp053 HDF5 metrics differ.")
        if not _all_numeric_finite(h5["entry"]):
            raise RuntimeError("exp053 HDF5 contains non-finite numeric data.")
        if bool(h5["entry/truth/reference_validated"][()]) or bool(
            h5["entry/truth/full_tgv_reference_authorized"][()]
        ):
            raise RuntimeError("exp053 provenance flags were promoted.")
        cache = root["cache"]
        if (
            cache["P_B_candidate"].shape[1:] != (96, 96)
            or cache["P_B_candidate"].dtype != np.complex128
            or len(cache["D_waist_m"])
            != len(result["cache"]["D_waist_m"])
        ):
            raise RuntimeError("Unexpected exp053 candidate cache identity.")
        midpoint_index = int(root["reported_interval/midpoint_cache_index"][()])
        if not np.array_equal(
            root["P_B_best_candidate_raw"][...],
            cache["P_B_candidate"][midpoint_index],
        ):
            raise RuntimeError("Best candidate/cache mapping differs.")
        if not np.array_equal(
            root["P_B_best_candidate_raw"][...] - source.P_B_rec_raw,
            root["residual_field_raw"][...],
        ):
            raise RuntimeError("Saved exp053 raw residual identity differs.")
        branches = root["optimizer/branches"]
        expected_budget = int(
            config["fit"]["optimizer"]["evaluation_budget_per_start"]
        )
        if len(branches) != 4 or any(
            int(branch["evaluation_count"][()]) != expected_budget
            for branch in branches.values()
        ):
            raise RuntimeError("exp053 optimizer equal-budget artifact differs.")
        if _compare_json_group(
            root["optimizer/numerical_closure_control"],
            config["fit"]["optimizer"]["numerical_closure_control"],
        ):
            raise RuntimeError("exp053 numerical-closure contract differs.")
        if _compare_json_group(
            root["source_exact_boundary_control"],
            result["source_exact_boundary_control"],
        ):
            raise RuntimeError("exp053 source-exact boundary control differs.")
        bisection = root["bisection"]
        expected_bisection = result["bisection"]
        if set(bisection) != set(expected_bisection) or any(
            set(bisection[branch_name]) != set(expected_branch)
            or any(
                int(bisection[branch_name][side_name]["iterations"][()])
                != int(expected_side["iterations"])
                or float(
                    bisection[branch_name][side_name][
                        "source_exact_breakpoint_m"
                    ][()]
                )
                != float(expected_side["source_exact_breakpoint_m"])
                or bool(
                    bisection[branch_name][side_name][
                        "source_exact_breakpoint_in_bracket"
                    ][()]
                )
                is not bool(
                    expected_side["source_exact_breakpoint_in_bracket"]
                )
                for side_name, expected_side in expected_branch.items()
            )
            for branch_name, expected_branch in expected_bisection.items()
        ):
            raise RuntimeError("exp053 boundary bisection budget differs.")
        if _decode(root["source/target_hdf5_path"][()]) != (
            source.target_hdf5_path
        ):
            raise RuntimeError("exp053 source dataset identity differs.")
        if _decode(root["source/target_dataset_sha256"][()]) != (
            source.target_dataset_sha256
        ):
            raise RuntimeError("exp053 source dataset hash differs.")
        if float(root["oracle_source/lower_m"][()]) != oracle.lower_m or float(
            root["oracle_source/upper_m"][()]
        ) != oracle.upper_m:
            raise RuntimeError("exp053 oracle interval artifact differs.")
    if hdf5_path.stat().st_size > config["output"]["expected_hdf5_bytes_max"]:
        raise RuntimeError("exp053 HDF5 exceeds registered size limit.")
    expected_figures = list(config["output"]["required_figure_filenames"])
    directional = result["simulation_evaluation_only"][
        "directional_diagnostic"
    ]
    if directional["performed"]:
        expected_figures.append(
            config["output"]["directional_diagnostic_figure_filename"]
        )
    figure_paths = sorted((run_dir / "figures").glob("*.png"))
    if [path.name for path in figure_paths] != sorted(expected_figures):
        raise RuntimeError("exp053 figure set differs from frozen contract.")
    for path in figure_paths:
        image = plt.imread(path)
        if image.size == 0 or not np.all(np.isfinite(image)):
            raise RuntimeError(f"Unreadable exp053 figure: {path}")


def run(config_path: Path) -> Path:
    resolved_config = config_path
    if not resolved_config.is_absolute():
        resolved_config = (PROJECT_ROOT / resolved_config).resolve()
    config = load_config(resolved_config)
    validate_exp053_config(config)
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
        source = load_exp053_source_reconstructed_probe(config, PROJECT_ROOT)
        oracle = load_exp051_oracle_interval(config, PROJECT_ROOT)
        generator = make_exp053_candidate_generator(
            source.source_config, memory_callback=sample_memory
        )
        result = run_reconstructed_probe_q8_interval_fit(
            config,
            source.source_config,
            source.P_B_rec_raw,
            source.P_B_true_operator_reference,
            oracle,
            generator,
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
        metadata = _metadata(config, resolved_config, source, oracle)
        metrics = _metrics_summary(
            config,
            source,
            oracle,
            result,
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss,
        )
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)
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
                "probe_grid": config["source"]["expected_identity"],
            },
            sample={"sample_a": source.source_config["sample_a"]},
            truth={
                "identity": (
                    "simulation truth under selected exp040 scalar working model; "
                    "not a fitter target"
                ),
                "reference_validated": False,
                "full_tgv_reference_authorized": False,
                "P_B_true_operator_reference": (
                    source.P_B_true_operator_reference
                ),
                "D_waist_true_m": config["fit"][
                    "true_d_waist_m_simulation_evaluation_only"
                ],
                "D_z_m": source.D_z_m,
                "z_m": source.z_m,
                "slice_widths_m": source.slice_widths_m,
            },
            reconstruction=_reconstruction_payload(
                config, source, oracle, result
            ),
            config_yaml=config_to_yaml(config),
            metadata=metadata,
            metrics=metrics,
        )
        figure_paths = [
            run_dir / "figures" / name
            for name in config["output"]["required_figure_filenames"]
        ]
        directional = result["simulation_evaluation_only"][
            "directional_diagnostic"
        ]
        diagnostic_path = (
            run_dir
            / "figures"
            / config["output"]["directional_diagnostic_figure_filename"]
            if directional["performed"]
            else None
        )
        _save_figures(result, figure_paths, diagnostic_path)
        _validate_artifacts(
            run_dir, config, source, oracle, metadata, metrics, result
        )
        all_figure_paths = [*figure_paths]
        if diagnostic_path is not None:
            all_figure_paths.append(diagnostic_path)
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
                path.name: sha256_file(path) for path in all_figure_paths
            },
            "figure_count": len(all_figure_paths),
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
    interval = result["reported_interval"]
    print(f"run_dir: {run_dir}")
    print(f"experiment_status: {result['experiment_status']}")
    print(f"interpretation: {result['interpretation']}")
    print(
        "reported_interval_um: "
        f"[{interval['lower_m'] * 1.0e6}, "
        f"{interval['upper_m'] * 1.0e6})"
    )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
