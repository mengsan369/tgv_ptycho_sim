"""Run exp040: validated 3D TGV multi-slice forward simulation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgv_ptycho.forward.exp040 import (
    build_exp040_hdf5_payload,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import (
    config_to_yaml,
    load_config,
    save_config,
)
from tgv_ptycho.io.metadata import created_at_utc, get_git_commit
from tgv_ptycho.io.naming import make_run_dir
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5
from tgv_ptycho.viz.plot_exp040 import (
    EXP040_FIGURE_FILENAMES,
    save_exp040_figures,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _metadata(
    config: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    status: str,
    diagnostics_r1_metrics: Mapping[str, Any] | None = None,
    diagnostics_r2_metrics: Mapping[str, Any] | None = None,
    diagnostics_r3_metrics: Mapping[str, Any] | None = None,
    diagnostics_r4_metrics: Mapping[str, Any] | None = None,
    diagnostics_r5_metrics: Mapping[str, Any] | None = None,
    diagnostics_r6_metrics: Mapping[str, Any] | None = None,
    diagnostics_r7_metrics: Mapping[str, Any] | None = None,
    diagnostics_r8_metrics: Mapping[str, Any] | None = None,
    diagnostics_r9_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build human- and HDF5-readable run metadata."""

    metadata: dict[str, Any] = {
        "experiment_id": "exp040",
        "phase": "Phase 4",
        "run_name": config["run"]["name"],
        "run_path": str(run_dir.resolve()),
        "source_config": str(config_path.resolve()),
        "created_at": created_at_utc(),
        "git_commit": get_git_commit(PROJECT_ROOT) or "unavailable",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config_status_at_launch": str(config["experiment"]["status"]),
        "experiment_status": status,
        "model": "centered_symmetric_split_step_scalar_multislice",
        "internal_reference_index": float(
            config["optics"]["internal_reference_index"]
        ),
        "external_medium_index": float(
            config["optics"]["external_medium_index"]
        ),
        "array_axes": {
            "field": ["y", "x"],
            "volume": ["z", "y", "x"],
            "intensity_stack": ["scan", "y", "x"],
            "scan_position_columns": ["x", "y"],
        },
        "planes": {
            "sample_A_input": "z=0 entrance boundary",
            "sample_A_output": "z=L exit boundary",
            "z_AB_origin": "sample_A_exit",
        },
        "units": {
            "length": "m",
            "phase": "rad",
            "intensity": "a.u.",
        },
        "random_seeds": {
            "sample_B": int(config["sample_b"]["seed"]),
            "scan_jitter": int(config["scan"]["jitter_seed"]),
            "incident_field": "not_applicable_deterministic_plane_wave",
        },
        "incident_field_randomized": False,
        "truth_use": "simulation forward evaluation and figures only",
        "reconstruction_executed": False,
        "calibration_executed": False,
        "preprocessing_executed": False,
        "limitations": [
            "scalar monochromatic unidirectional phase-screen model",
            "no interface reflection, backward wave, or polarization",
            "voxel-center binary TGV boundary",
            "FFT periodic lateral boundary",
            "periodic integer-pixel sample-B shifts",
            "ideal noiseless grid-sampled detector intensity",
        ],
    }
    if _r9_enabled(config):
        if diagnostics_r9_metrics is None:
            msg = "R9 metadata requires R9 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R9"
        metadata["diagnostics_r9_status"] = str(
            diagnostics_r9_metrics["status"]
        )
        metadata["limitations"] = [
            item
            for item in metadata["limitations"]
            if item != "voxel-center binary TGV boundary"
        ]
        metadata["limitations"].extend(
            [
                "R9 uses a q8 numerical cell-average interface, not a "
                "physical effective-medium law",
                "R9 external passband is a scalar external-medium "
                "propagating projector, not a measured system NA",
                "R9 does not recompute the detector path and preserves the "
                "formal R8 result as provenance",
            ]
        )
    elif _r8_enabled(config):
        if diagnostics_r8_metrics is None:
            msg = "R8 metadata requires R8 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R8"
        metadata["diagnostics_r8_status"] = str(
            diagnostics_r8_metrics["status"]
        )
        metadata["limitations"] = [
            item
            for item in metadata["limitations"]
            if item
            not in {
                "voxel-center binary TGV boundary",
                "periodic integer-pixel sample-B shifts",
                "ideal noiseless grid-sampled detector intensity",
            }
        ]
        metadata["limitations"].extend(
            [
                "R8 uses a q8 numerical cell-average interface, not a "
                "physical effective-medium law",
                "R8 fixes the virtual nominal 96 um hard-edge B; the R6 "
                "support envelope remains outside the visibility gate",
                "R8 uses ideal noiseless q4 positive square-pixel quadrature",
            ]
        )
    elif _r7_enabled(config):
        if diagnostics_r7_metrics is None:
            msg = "R7 metadata requires R7 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R7"
        metadata["diagnostics_r7_status"] = str(
            diagnostics_r7_metrics["status"]
        )
        metadata["limitations"].extend(
            [
                "R7 subvoxel index is a numerical cell average, not a "
                "physical effective-medium law",
                "R7 fixes the virtual nominal 96 um hard-edge B; the R6 "
                "support envelope remains an independent uncertainty context",
            ]
        )
    elif _r6_enabled(config):
        if diagnostics_r6_metrics is None:
            msg = "R6 metadata requires R6 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R6"
        metadata["diagnostics_r6_status"] = str(
            diagnostics_r6_metrics["status"]
        )
        metadata["limitations"].append(
            "R6 is a virtual finite-B support/taper sensitivity envelope, "
            "not an empirical calibration of the physical sample"
        )
    elif _r5_enabled(config):
        if diagnostics_r5_metrics is None:
            msg = "R5 metadata requires R5 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R5"
        metadata["diagnostics_r5_status"] = str(
            diagnostics_r5_metrics["status"]
        )
        metadata["limitations"].append(
            "R5 uses a finite 96 um coded B with transparent exterior and "
            "zero-padded residual propagation; illumination remains an "
            "infinite plane-wave background"
        )
    elif _r4_enabled(config):
        if diagnostics_r4_metrics is None:
            msg = "R4 metadata requires R4 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R4"
        metadata["diagnostics_r4_status"] = str(
            diagnostics_r4_metrics["status"]
        )
        metadata["limitations"].append(
            "R4 uses periodic sample B and FFT boundaries while evaluating "
            "positive staggered midpoint detector quadrature"
        )
    elif _r3_enabled(config):
        if (
            diagnostics_r1_metrics is None
            or diagnostics_r2_metrics is None
            or diagnostics_r3_metrics is None
        ):
            msg = "R3 metadata requires R1, R2, and R3 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R3"
        metadata["diagnostics_r1_status"] = str(
            diagnostics_r1_metrics["status"]
        )
        metadata["diagnostics_r2_status"] = str(
            diagnostics_r2_metrics["status"]
        )
        metadata["diagnostics_r3_status"] = str(
            diagnostics_r3_metrics["status"]
        )
        metadata["limitations"].append(
            "R3 pixel branch uses an ideal periodic square-pixel sinc MTF, "
            "not a measured detector response"
        )
    elif _r2_enabled(config):
        if diagnostics_r1_metrics is None or diagnostics_r2_metrics is None:
            msg = "R2 metadata requires both R1 and R2 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R2"
        metadata["diagnostics_r1_status"] = str(
            diagnostics_r1_metrics["status"]
        )
        metadata["diagnostics_r2_status"] = str(
            diagnostics_r2_metrics["status"]
        )
    elif _r1_enabled(config):
        if diagnostics_r1_metrics is None:
            msg = "R1 metadata requires diagnostics_r1 metrics."
            raise RuntimeError(msg)
        metadata["diagnostic_stage"] = "R1"
        metadata["diagnostics_r1_status"] = str(
            diagnostics_r1_metrics["status"]
        )
    return metadata


def _r1_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r1")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r2_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r2")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r3_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r3")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r4_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r4")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r5_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r5")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r6_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r6")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r7_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r7")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r8_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r8")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _r9_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("diagnostics_r9")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _expected_figure_filenames(config: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(EXP040_FIGURE_FILENAMES)
    if _r1_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r1 import EXP040_R1_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R1_FIGURE_FILENAMES))
    if _r2_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r2 import EXP040_R2_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R2_FIGURE_FILENAMES))
    if _r3_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r3 import EXP040_R3_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R3_FIGURE_FILENAMES))
    if _r4_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r4 import EXP040_R4_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R4_FIGURE_FILENAMES))
    if _r5_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r5 import EXP040_R5_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R5_FIGURE_FILENAMES))
    if _r6_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r6 import EXP040_R6_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R6_FIGURE_FILENAMES))
    if _r7_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r7 import EXP040_R7_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R7_FIGURE_FILENAMES))
    if _r8_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r8 import EXP040_R8_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R8_FIGURE_FILENAMES))
    if _r9_enabled(config):
        from tgv_ptycho.viz.plot_exp040_r9 import EXP040_R9_FIGURE_FILENAMES

        names = (*names, *tuple(EXP040_R9_FIGURE_FILENAMES))
    return names


def _save_r1_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r1 import save_exp040_r1_figures

    return save_exp040_r1_figures(result, figures_dir)


def _save_r2_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r2 import save_exp040_r2_figures

    return save_exp040_r2_figures(result, figures_dir)


def _save_r3_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r3 import save_exp040_r3_figures

    return save_exp040_r3_figures(result, figures_dir)


def _save_r4_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r4 import save_exp040_r4_figures

    return save_exp040_r4_figures(result, figures_dir)


def _save_r5_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r5 import save_exp040_r5_figures

    return save_exp040_r5_figures(result, figures_dir)


def _save_r6_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r6 import save_exp040_r6_figures

    return save_exp040_r6_figures(result, figures_dir)


def _save_r7_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r7 import save_exp040_r7_figures

    return save_exp040_r7_figures(result, figures_dir)


def _save_r8_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r8 import save_exp040_r8_figures

    return save_exp040_r8_figures(result, figures_dir)


def _save_r9_figures(result: dict[str, Any], figures_dir: Path) -> list[Path]:
    from tgv_ptycho.viz.plot_exp040_r9 import save_exp040_r9_figures

    return save_exp040_r9_figures(result, figures_dir)


def _decode_hdf5_value(value: Any) -> Any:
    """Convert one HDF5 value to JSON-compatible Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_hdf5_value(value.item())
        return [_decode_hdf5_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _decode_hdf5_value(value.item())
    return value


def _read_hdf5_tree(group: h5py.Group) -> dict[str, Any]:
    """Read a nested HDF5 group into a JSON-compatible mapping."""

    return {
        name: (
            _read_hdf5_tree(item)
            if isinstance(item, h5py.Group)
            else _decode_hdf5_value(item[()])
        )
        for name, item in group.items()
    }


def _first_tree_difference(expected: Any, actual: Any, path: str) -> str | None:
    """Return the first recursive tree mismatch, if one exists."""

    expected = _decode_hdf5_value(expected)
    actual = _decode_hdf5_value(actual)
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path}: expected mapping, got {type(actual).__name__}"
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path}: missing keys={missing}, extra keys={extra}"
        for key in sorted(expected_keys):
            difference = _first_tree_difference(
                expected[key], actual[key], f"{path}/{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return f"{path}: expected sequence, got {type(actual).__name__}"
        if len(expected) != len(actual):
            return f"{path}: expected length {len(expected)}, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            difference = _first_tree_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def _require_equal_trees(expected: Any, actual: Any, name: str) -> None:
    difference = _first_tree_difference(expected, actual, name)
    if difference is not None:
        msg = f"{name} trees disagree: {difference}"
        raise RuntimeError(msg)


def _validate_r1_metrics(
    config: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> None:
    """Validate the compact R1 metrics persisted to JSON and HDF5."""

    r1_config = config.get("diagnostics_r1")
    if not isinstance(r1_config, Mapping) or r1_config.get("enabled") is not True:
        msg = "R1 metrics are only valid for diagnostics_r1.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "canonical_b_validation",
        "refined_convergence",
        "external_padding",
        "refined_floor",
        "visibility_report",
        "thresholds",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r1 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != r1_config["version"]:
        msg = "diagnostics_r1 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(r1_config["methods"], diagnostics["methods"], "R1 methods")
    status = diagnostics["status"]
    if status not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r1 status: {status!r}."
        raise RuntimeError(msg)
    for name in ("all_finite", "all_intensity_nonnegative", "hard_checks_pass"):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r1 {name} must be boolean."
            raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r1 reports non-finite outputs."
        raise RuntimeError(msg)
    if not bool(diagnostics["all_intensity_nonnegative"]):
        msg = "diagnostics_r1 reports negative intensity."
        raise RuntimeError(msg)

    refined_config = {
        "axial": ("acceptance_pair_m", r1_config["refined_axial"]),
        "lateral": ("acceptance_pair_dx_m", r1_config["refined_lateral"]),
        "fov": ("acceptance_pair_shapes", r1_config["refined_fov"]),
    }
    refined_metrics = diagnostics["refined_convergence"]
    if not isinstance(refined_metrics, Mapping):
        msg = "diagnostics_r1 refined_convergence must be a mapping."
        raise RuntimeError(msg)
    for group_name, (pair_key, group_config) in refined_config.items():
        group_metrics = refined_metrics.get(group_name)
        if not isinstance(group_metrics, Mapping) or pair_key not in group_metrics:
            msg = f"R1 {group_name} metrics must contain {pair_key}."
            raise RuntimeError(msg)
        _require_equal_trees(
            group_config[pair_key],
            group_metrics[pair_key],
            f"R1 {group_name} acceptance pair",
        )
    external_metrics = diagnostics["external_padding"]
    if not isinstance(external_metrics, Mapping):
        msg = "diagnostics_r1 external_padding must be a mapping."
        raise RuntimeError(msg)
    pair_key = "acceptance_pair_shapes"
    if pair_key not in external_metrics:
        msg = "R1 external_padding metrics must contain acceptance_pair_shapes."
        raise RuntimeError(msg)
    _require_equal_trees(
        r1_config["external_padding"][pair_key],
        external_metrics[pair_key],
        "R1 external-padding acceptance pair",
    )


def _validate_r2_metrics(
    config: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> None:
    """Validate compact R2 metrics before JSON/HDF5 persistence."""

    r2_config = config.get("diagnostics_r2")
    if not isinstance(r2_config, Mapping) or r2_config.get("enabled") is not True:
        msg = "R2 metrics are only valid for diagnostics_r2.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "canonical_b_validation",
        "a_exit_center_invariance",
        "period_aligned",
        "method_difference",
        "alias_masks",
        "determinism",
        "r1_external_comparator",
        "thresholds",
        "outcome_flags",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r2 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != r2_config["version"]:
        msg = "diagnostics_r2 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(r2_config["methods"], diagnostics["methods"], "R2 methods")
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r2 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in ("all_finite", "all_intensity_nonnegative", "hard_checks_pass"):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r2 {name} must be boolean."
            raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r2 reports non-finite outputs."
        raise RuntimeError(msg)
    if not bool(diagnostics["all_intensity_nonnegative"]):
        msg = "diagnostics_r2 reports negative intensity."
        raise RuntimeError(msg)
    aligned = diagnostics["period_aligned"]
    if not isinstance(aligned, Mapping) or set(aligned) != {
        "current_asm",
        "alias_controlled",
    }:
        msg = "diagnostics_r2 period_aligned methods are invalid."
        raise RuntimeError(msg)
    pair = r2_config["period_commensurate"]["acceptance_pair_shapes"]
    for method_name, method_metrics in aligned.items():
        if not isinstance(method_metrics, Mapping):
            msg = f"R2 {method_name} metrics must be a mapping."
            raise RuntimeError(msg)
        _require_equal_trees(
            pair,
            method_metrics.get("acceptance_pair_shapes"),
            f"R2 {method_name} acceptance pair",
        )
        if set(method_metrics.get("acceptance", {})) != {"P_B", "I_stack"}:
            msg = f"R2 {method_name} acceptance outputs are invalid."
            raise RuntimeError(msg)
    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    if not isinstance(thresholds, Mapping) or (
        float(thresholds["convergence_relative_l2_max"])
        != float(acceptance["convergence_relative_l2_max"])
        or float(thresholds["determinism_relative_l2_max"])
        != float(acceptance["determinism_relative_l2_max"])
    ):
        msg = "R2 metrics do not reuse the registered thresholds."
        raise RuntimeError(msg)


def _validate_r3_metrics(
    config: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> None:
    """Validate compact R3 metrics before JSON/HDF5 persistence."""

    r3_config = config.get("diagnostics_r3")
    if not isinstance(r3_config, Mapping) or r3_config.get("enabled") is not True:
        msg = "R3 metrics are only valid for diagnostics_r3.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "sampling",
        "canonical_b_validation",
        "a_exit_native_recovery",
        "alias_masks",
        "spectra",
        "bc_propagation",
        "detector_sampling",
        "detector_operator_difference",
        "pixel_operator_controls",
        "determinism",
        "thresholds",
        "outcome_flags",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r3 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != r3_config["version"]:
        msg = "diagnostics_r3 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(r3_config["methods"], diagnostics["methods"], "R3 methods")
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r3 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in ("all_finite", "all_intensity_nonnegative", "hard_checks_pass"):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r3 {name} must be boolean."
            raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r3 reports non-finite outputs."
        raise RuntimeError(msg)

    sampling = diagnostics["sampling"]
    if not isinstance(sampling, Mapping):
        msg = "diagnostics_r3 sampling must be a mapping."
        raise RuntimeError(msg)
    registered_sampling = r3_config["sampling"]
    for config_name, metric_name in (
        ("factors", "factors"),
        ("dx_m", "dx_m"),
        ("shapes", "shapes"),
        ("external_fov_m", "external_fov_m"),
        ("native_sample_offsets_px", "native_sample_offset_px"),
        ("physical_origin_compensation_m", "physical_origin_compensation_m"),
        ("native_roi_shape", "native_roi_shape"),
    ):
        _require_equal_trees(
            registered_sampling[config_name],
            sampling.get(metric_name),
            f"R3 sampling {metric_name}",
        )
    if sampling.get("full_detector_stacks_retained") is not False:
        msg = "R3 must not retain full detector stacks."
        raise RuntimeError(msg)

    convergence = diagnostics["detector_sampling"]
    if not isinstance(convergence, Mapping):
        msg = "diagnostics_r3 detector_sampling must be a mapping."
        raise RuntimeError(msg)
    _require_equal_trees(
        r3_config["sampling"]["acceptance_pair_factors"],
        convergence.get("acceptance_pair_factors"),
        "R3 acceptance factor pair",
    )
    detector_acceptance = convergence.get("acceptance", {}).get("detector", {})
    if set(detector_acceptance) != {"current_asm", "alias_controlled"}:
        msg = "R3 detector acceptance BC methods are invalid."
        raise RuntimeError(msg)
    for method in detector_acceptance.values():
        if not isinstance(method, Mapping) or set(method) != {
            "point_sample",
            "pixel_box_average",
        }:
            msg = "R3 detector acceptance branches are invalid."
            raise RuntimeError(msg)

    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    if not isinstance(thresholds, Mapping) or (
        float(thresholds["convergence_relative_l2_max"])
        != float(acceptance["convergence_relative_l2_max"])
        or float(thresholds["mapping_and_pixel_relative_max"])
        != float(acceptance["algebra_relative_l2_max"])
        or float(thresholds["determinism_relative_l2_max"])
        != float(acceptance["determinism_relative_l2_max"])
    ):
        msg = "R3 metrics do not reuse the registered thresholds."
        raise RuntimeError(msg)
    selected = r3_config["detector_sampling"]
    operator = diagnostics["detector_operator_difference"]
    determinism = diagnostics["determinism"]
    if (
        not isinstance(operator, Mapping)
        or not isinstance(determinism, Mapping)
        or int(operator["selected_factor"]) != int(selected["selected_factor"])
        or int(operator["selected_scan_index"])
        != int(selected["selected_scan_index"])
        or int(determinism["selected_factor"]) != int(selected["selected_factor"])
        or int(determinism["selected_scan_index"])
        != int(selected["selected_scan_index"])
    ):
        msg = "R3 selected factor/scan metrics disagree with config."
        raise RuntimeError(msg)


def _validate_r4_metrics(
    config: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> None:
    """Validate compact R4 metrics before JSON/HDF5 persistence."""

    r4_config = config.get("diagnostics_r4")
    if not isinstance(r4_config, Mapping) or r4_config.get("enabled") is not True:
        msg = "R4 metrics are only valid for diagnostics_r4.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "r3_provenance",
        "sampling",
        "canonical_b_validation",
        "quadrature_controls",
        "convergence",
        "determinism",
        "thresholds",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r4 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != r4_config["version"]:
        msg = "diagnostics_r4 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(r4_config["methods"], diagnostics["methods"], "R4 methods")
    _require_equal_trees(
        r4_config["r3_provenance"],
        diagnostics["r3_provenance"],
        "R4 R3 provenance",
    )
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r4 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in ("all_finite", "all_intensity_nonnegative", "hard_checks_pass"):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r4 {name} must be boolean."
            raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r4 reports non-finite outputs."
        raise RuntimeError(msg)

    sampling = diagnostics["sampling"]
    if not isinstance(sampling, Mapping):
        msg = "diagnostics_r4 sampling must be a mapping."
        raise RuntimeError(msg)
    registered_sampling = r4_config["sampling"]
    for config_name, metric_name in (
        ("factors", "factors"),
        ("node_dx_m", "node_dx_m"),
        ("node_shapes", "node_shapes"),
        ("native_roi_shape", "native_roi_shape"),
    ):
        _require_equal_trees(
            registered_sampling[config_name],
            sampling.get(metric_name),
            f"R4 sampling {metric_name}",
        )
    expected_scan_count = int(config["scan"]["num_x"]) * int(
        config["scan"]["num_y"]
    )
    if int(sampling.get("scan_count", -1)) != expected_scan_count:
        msg = "R4 sampling scan count disagrees with config."
        raise RuntimeError(msg)
    if sampling.get("full_node_stacks_retained") is not False:
        msg = "R4 must not retain full node-grid stacks."
        raise RuntimeError(msg)

    convergence = diagnostics["convergence"]
    if not isinstance(convergence, Mapping):
        msg = "diagnostics_r4 convergence must be a mapping."
        raise RuntimeError(msg)
    _require_equal_trees(
        registered_sampling["factors"],
        convergence.get("factors"),
        "R4 convergence factors",
    )
    _require_equal_trees(
        registered_sampling["acceptance_pair_factors"],
        convergence.get("acceptance_pair_factors"),
        "R4 acceptance factor pair",
    )
    if set(convergence.get("acceptance", {})) != {"P_B", "I_stack"}:
        msg = "R4 convergence acceptance outputs are invalid."
        raise RuntimeError(msg)

    determinism = diagnostics["determinism"]
    selected = r4_config["determinism"]
    if (
        not isinstance(determinism, Mapping)
        or int(determinism["factor"]) != int(selected["selected_factor"])
        or int(determinism["scan_index"]) != int(selected["selected_scan_index"])
    ):
        msg = "R4 determinism factor/scan metrics disagree with config."
        raise RuntimeError(msg)

    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    if not isinstance(thresholds, Mapping) or (
        float(thresholds["convergence_relative_l2_max"])
        != float(acceptance["convergence_relative_l2_max"])
        or float(thresholds["algebra_relative_l2_max"])
        != float(acceptance["algebra_relative_l2_max"])
        or float(thresholds["determinism_relative_l2_max"])
        != float(acceptance["determinism_relative_l2_max"])
    ):
        msg = "R4 metrics do not reuse the registered thresholds."
        raise RuntimeError(msg)


def _validate_r5_metrics(
    config: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> None:
    """Validate compact R5 metrics before JSON/HDF5 persistence."""

    r5_config = config.get("diagnostics_r5")
    if not isinstance(r5_config, Mapping) or r5_config.get("enabled") is not True:
        msg = "R5 metrics are only valid for diagnostics_r5.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "r4_provenance",
        "sampling",
        "finite_support",
        "source_containment",
        "open_boundary_convergence",
        "effects",
        "controls",
        "determinism",
        "thresholds",
        "outcome_flags",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r5 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != r5_config["version"]:
        msg = "diagnostics_r5 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(r5_config["methods"], diagnostics["methods"], "R5 methods")
    _require_equal_trees(
        r5_config["r4_provenance"],
        diagnostics["r4_provenance"],
        "R5 R4 provenance",
    )
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r5 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in ("all_finite", "all_intensity_nonnegative", "hard_checks_pass"):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r5 {name} must be boolean."
            raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r5 reports non-finite outputs."
        raise RuntimeError(msg)

    sampling = diagnostics["sampling"]
    registered = r5_config["sampling"]
    if not isinstance(sampling, Mapping):
        msg = "diagnostics_r5 sampling must be a mapping."
        raise RuntimeError(msg)
    for config_name, metric_name in (
        ("quadrature_factor", "quadrature_factor"),
        ("node_dx_m", "node_dx_m"),
        ("base_fov_m", "base_fov_m"),
        ("base_node_shape", "base_node_shape"),
        ("padding_fov_m", "padding_fov_m"),
        ("padding_node_shapes", "padding_node_shapes"),
        ("native_roi_shape", "native_roi_shape"),
        ("boundary_ring_width_m", "boundary_ring_width_m"),
    ):
        _require_equal_trees(
            registered[config_name],
            sampling.get(metric_name),
            f"R5 sampling {metric_name}",
        )
    expected_scan_count = int(config["scan"]["num_x"]) * int(
        config["scan"]["num_y"]
    )
    if int(sampling.get("scan_count", -1)) != expected_scan_count:
        msg = "R5 sampling scan count disagrees with config."
        raise RuntimeError(msg)
    if sampling.get("full_node_stacks_retained") is not False:
        msg = "R5 must not retain full node-grid stacks."
        raise RuntimeError(msg)

    finite_support = diagnostics["finite_support"]
    support_config = r5_config["finite_support"]
    if not isinstance(finite_support, Mapping):
        msg = "diagnostics_r5 finite_support must be a mapping."
        raise RuntimeError(msg)
    _require_equal_trees(
        support_config["physical_shape_m"],
        finite_support.get("physical_shape_m"),
        "R5 finite support physical shape",
    )
    _require_equal_trees(
        support_config["canonical_phase_cells"],
        finite_support.get("canonical_phase_cells"),
        "R5 finite support phase cells",
    )

    convergence = diagnostics["open_boundary_convergence"]
    if not isinstance(convergence, Mapping):
        msg = "diagnostics_r5 open convergence must be a mapping."
        raise RuntimeError(msg)
    _require_equal_trees(
        registered["padding_fov_m"],
        convergence.get("padding_fov_m"),
        "R5 open padding FOVs",
    )
    _require_equal_trees(
        registered["acceptance_pair_fov_m"],
        convergence.get("acceptance_pair_fov_m"),
        "R5 open acceptance pair",
    )
    effects = diagnostics["effects"]
    expected_effects = {
        "support_relative_l2",
        "boundary_relative_l2",
        "combined_relative_l2",
        "support_material",
        "boundary_material",
        "combined_material",
    }
    if not isinstance(effects, Mapping) or set(effects) != expected_effects:
        msg = "R5 effect metrics are invalid."
        raise RuntimeError(msg)

    determinism = diagnostics["determinism"]
    selected = r5_config["determinism"]
    if (
        not isinstance(determinism, Mapping)
        or float(determinism["padding_fov_m"])
        != float(selected["selected_padding_fov_m"])
        or int(determinism["scan_index"]) != int(selected["selected_scan_index"])
    ):
        msg = "R5 determinism FOV/scan metrics disagree with config."
        raise RuntimeError(msg)

    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    if not isinstance(thresholds, Mapping) or (
        float(thresholds["convergence_and_materiality_relative_l2"])
        != float(acceptance["convergence_relative_l2_max"])
        or float(thresholds["algebra_relative_l2_max"])
        != float(acceptance["algebra_relative_l2_max"])
        or float(thresholds["determinism_relative_l2_max"])
        != float(acceptance["determinism_relative_l2_max"])
    ):
        msg = "R5 metrics do not reuse the registered thresholds."
        raise RuntimeError(msg)


def _validate_r6_metrics(
    config: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> None:
    """Validate compact R6 metrics before JSON/HDF5 persistence."""

    r6_config = config.get("diagnostics_r6")
    if not isinstance(r6_config, Mapping) or r6_config.get("enabled") is not True:
        msg = "R6 metrics are only valid for diagnostics_r6.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "r5_provenance",
        "sampling",
        "support_family",
        "support_effects",
        "nominal_sensitivity",
        "selected_cases",
        "controls",
        "determinism",
        "thresholds",
        "outcome_flags",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r6 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != r6_config["version"]:
        msg = "diagnostics_r6 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(r6_config["methods"], diagnostics["methods"], "R6 methods")
    _require_equal_trees(
        r6_config["r5_provenance"],
        diagnostics["r5_provenance"],
        "R6 R5 provenance",
    )
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r6 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in ("all_finite", "all_intensity_nonnegative", "hard_checks_pass"):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r6 {name} must be boolean."
            raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r6 reports non-finite outputs."
        raise RuntimeError(msg)

    sampling = diagnostics["sampling"]
    registered_sampling = r6_config["sampling"]
    if not isinstance(sampling, Mapping):
        msg = "diagnostics_r6 sampling must be a mapping."
        raise RuntimeError(msg)
    for config_name, metric_name in (
        ("quadrature_factor", "quadrature_factor"),
        ("node_dx_m", "node_dx_m"),
        ("fov_m", "fov_m"),
        ("node_shape", "node_shape"),
        ("native_roi_shape", "native_roi_shape"),
    ):
        _require_equal_trees(
            registered_sampling[config_name],
            sampling.get(metric_name),
            f"R6 sampling {metric_name}",
        )
    expected_scan_count = int(config["scan"]["num_x"]) * int(
        config["scan"]["num_y"]
    )
    if (
        int(sampling.get("scan_count", -1)) != expected_scan_count
        or int(sampling.get("case_count", -1)) != 9
        or sampling.get("full_node_stacks_retained") is not False
    ):
        msg = "R6 scan/case/streaming metrics disagree with config."
        raise RuntimeError(msg)

    family = diagnostics["support_family"]
    registered_family = r6_config["support_family"]
    if not isinstance(family, Mapping):
        msg = "diagnostics_r6 support_family must be a mapping."
        raise RuntimeError(msg)
    for name in (
        "support_width_m",
        "edge_taper_width_m",
        "nominal_support_width_m",
        "nominal_edge_taper_width_m",
    ):
        _require_equal_trees(
            registered_family[name], family.get(name), f"R6 support family {name}"
        )
    if np.asarray(family.get("case_support_width_m")).shape != (9,) or np.asarray(
        family.get("case_edge_taper_width_m")
    ).shape != (9,):
        msg = "R6 case-family axes must contain nine cases."
        raise RuntimeError(msg)

    effects = diagnostics["support_effects"]
    nominal = diagnostics["nominal_sensitivity"]
    if (
        not isinstance(effects, Mapping)
        or np.asarray(effects.get("relative_l2_matrix")).shape != (3, 3)
        or np.asarray(effects.get("material_matrix")).shape != (3, 3)
        or not isinstance(nominal, Mapping)
        or np.asarray(nominal.get("relative_l2_matrix")).shape != (3, 3)
    ):
        msg = "R6 effect/sensitivity matrices are invalid."
        raise RuntimeError(msg)
    if not isinstance(effects.get("all_cases_material"), (bool, np.bool_)):
        msg = "R6 all_cases_material must be boolean."
        raise RuntimeError(msg)

    determinism = diagnostics["determinism"]
    selected = r6_config["determinism"]
    if (
        not isinstance(determinism, Mapping)
        or float(determinism["support_width_m"])
        != float(selected["support_width_m"])
        or float(determinism["edge_taper_width_m"])
        != float(selected["edge_taper_width_m"])
        or int(determinism["scan_index"]) != int(selected["scan_index"])
    ):
        msg = "R6 determinism case/scan metrics disagree with config."
        raise RuntimeError(msg)

    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    if not isinstance(thresholds, Mapping) or (
        float(thresholds["materiality_relative_l2"])
        != float(acceptance["convergence_relative_l2_max"])
        or float(thresholds["algebra_relative_l2_max"])
        != float(acceptance["algebra_relative_l2_max"])
        or float(thresholds["determinism_relative_l2_max"])
        != float(acceptance["determinism_relative_l2_max"])
    ):
        msg = "R6 metrics do not reuse the registered thresholds."
        raise RuntimeError(msg)


def _validate_r7_metrics(
    config: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> None:
    """Validate compact R7 metrics before JSON/HDF5 persistence."""

    registered = config.get("diagnostics_r7")
    if not isinstance(registered, Mapping) or registered.get("enabled") is not True:
        msg = "R7 metrics are only valid for diagnostics_r7.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "r6_provenance",
        "sampling",
        "interface_controls",
        "finite_b_controls",
        "detector_controls",
        "convergence",
        "binary_effect",
        "determinism",
        "model_uncertainty_context",
        "thresholds",
        "outcome_flags",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r7 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != registered["version"]:
        msg = "diagnostics_r7 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(registered["methods"], diagnostics["methods"], "R7 methods")
    _require_equal_trees(
        registered["r6_provenance"],
        diagnostics["r6_provenance"],
        "R7 R6 provenance",
    )
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r7 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in ("all_finite", "all_intensity_nonnegative", "hard_checks_pass"):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r7 {name} must be boolean."
            raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r7 reports non-finite outputs."
        raise RuntimeError(msg)

    sampling = diagnostics["sampling"]
    if not isinstance(sampling, Mapping):
        msg = "diagnostics_r7 sampling must be a mapping."
        raise RuntimeError(msg)
    a_registered = registered["sample_a_sampling"]
    d_registered = registered["detector_sampling"]
    for expected, actual, name in (
        (
            registered["interface"]["factors"],
            sampling.get("interface_factors"),
            "factors",
        ),
        (a_registered["shape"], sampling.get("sample_a_shape"), "A shape"),
        (a_registered["dx_m"], sampling.get("sample_a_dx_m"), "A dx"),
        (a_registered["dz_m"], sampling.get("sample_a_dz_m"), "A dz"),
        (
            d_registered["quadrature_factor"],
            sampling.get("detector_quadrature_factor"),
            "detector q",
        ),
        (
            d_registered["node_dx_m"],
            sampling.get("detector_node_dx_m"),
            "detector dx",
        ),
        (
            d_registered["base_node_shape"],
            sampling.get("base_node_shape"),
            "base shape",
        ),
        (
            d_registered["open_node_shape"],
            sampling.get("open_node_shape"),
            "open shape",
        ),
        (d_registered["native_roi_shape"], sampling.get("native_roi_shape"), "ROI"),
    ):
        _require_equal_trees(expected, actual, f"R7 sampling {name}")
    expected_scans = int(config["scan"]["num_x"]) * int(config["scan"]["num_y"])
    if (
        int(sampling.get("scan_count", -1)) != expected_scans
        or sampling.get("full_volumes_retained") is not False
        or sampling.get("full_node_stacks_retained") is not False
    ):
        msg = "R7 scan/streaming metrics disagree with config."
        raise RuntimeError(msg)

    convergence = diagnostics["convergence"]
    binary = diagnostics["binary_effect"]
    if not isinstance(convergence, Mapping) or not isinstance(binary, Mapping):
        msg = "R7 comparison metrics must be mappings."
        raise RuntimeError(msg)
    for name in ("U_A_exit", "P_B", "I_stack"):
        if np.asarray(convergence["relative_to_q8"][name]).shape != (4,):
            msg = f"R7 {name} convergence series must have four cases."
            raise RuntimeError(msg)
        if not isinstance(convergence["pass"][name], (bool, np.bool_)):
            msg = f"R7 {name} convergence flag must be boolean."
            raise RuntimeError(msg)
        if not isinstance(binary["material_by_output"][name], (bool, np.bool_)):
            msg = f"R7 {name} materiality flag must be boolean."
            raise RuntimeError(msg)
    context = diagnostics["model_uncertainty_context"]
    if (
        not isinstance(context, Mapping)
        or float(context["r6_maximum_nominal_b_variation"])
        != float(registered["r6_provenance"]["maximum_nominal_b_variation"])
        or context.get("combined_with_r7_metrics") is not False
    ):
        msg = "R7 must preserve R6 uncertainty as a separate context field."
        raise RuntimeError(msg)
    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    if not isinstance(thresholds, Mapping) or (
        float(thresholds["convergence_and_materiality_relative_l2"])
        != float(acceptance["convergence_relative_l2_max"])
        or float(thresholds["algebra_relative_l2_max"])
        != float(acceptance["algebra_relative_l2_max"])
        or float(thresholds["determinism_relative_l2_max"])
        != float(acceptance["determinism_relative_l2_max"])
    ):
        msg = "R7 metrics do not reuse the registered thresholds."
        raise RuntimeError(msg)


def _validate_r8_metrics(
    config: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> None:
    """Validate compact R8 metrics before JSON/HDF5 persistence."""

    registered = config.get("diagnostics_r8")
    if not isinstance(registered, Mapping) or registered.get("enabled") is not True:
        msg = "R8 metrics are only valid for diagnostics_r8.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "r7_provenance",
        "sampling",
        "interface_controls",
        "mapping_controls",
        "detector_controls",
        "convergence",
        "visibility",
        "determinism",
        "model_uncertainty_context",
        "thresholds",
        "outcome_flags",
        "legacy_experiment_status_preserved",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r8 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != registered["version"]:
        msg = "diagnostics_r8 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(registered["methods"], diagnostics["methods"], "R8 methods")
    _require_equal_trees(
        registered["r7_provenance"],
        diagnostics["r7_provenance"],
        "R8 R7 provenance",
    )
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r8 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in (
        "legacy_experiment_status_preserved",
        "all_finite",
        "all_intensity_nonnegative",
        "hard_checks_pass",
    ):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r8 {name} must be boolean."
            raise RuntimeError(msg)
    if diagnostics["legacy_experiment_status_preserved"] is not True:
        msg = "R8 must preserve the legacy experiment status."
        raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r8 reports non-finite outputs."
        raise RuntimeError(msg)

    sampling = diagnostics["sampling"]
    if not isinstance(sampling, Mapping):
        msg = "diagnostics_r8 sampling must be a mapping."
        raise RuntimeError(msg)
    cases = registered["sample_a_cases"]["cases"]
    expected_case_ids = [case["id"] for case in cases]
    expected_shapes = [case["shape"] for case in cases]
    expected_dx = [case["dx_m"] for case in cases]
    expected_dz = [case["dz_m"] for case in cases]
    expected_waists = [case["d_waist_m"] for case in cases]
    detector = registered["detector_sampling"]
    open_control = registered["open_control"]
    for expected, actual, name in (
        (registered["interface"]["factor"], sampling.get("interface_factor"), "q"),
        (expected_case_ids, sampling.get("case_ids"), "case IDs"),
        (expected_shapes, sampling.get("sample_a_shapes"), "A shapes"),
        (expected_dx, sampling.get("sample_a_dx_m"), "A dx"),
        (expected_dz, sampling.get("sample_a_dz_m"), "A dz"),
        (expected_waists, sampling.get("d_waist_m"), "waists"),
        (
            detector["quadrature_factor"],
            sampling.get("detector_quadrature_factor"),
            "detector q",
        ),
        (detector["node_dx_m"], sampling.get("detector_node_dx_m"), "detector dx"),
        (detector["base_node_shape"], sampling.get("base_node_shape"), "base shape"),
        (
            detector["primary_open_node_shape"],
            sampling.get("primary_open_node_shape"),
            "primary open shape",
        ),
        (
            open_control["node_shapes"],
            sampling.get("open_control_node_shapes"),
            "open-control shapes",
        ),
        (detector["native_roi_shape"], sampling.get("native_roi_shape"), "ROI"),
    ):
        _require_equal_trees(expected, actual, f"R8 sampling {name}")
    expected_scans = int(config["scan"]["num_x"]) * int(config["scan"]["num_y"])
    if (
        int(sampling.get("scan_count", -1)) != expected_scans
        or np.asarray(sampling.get("slice_counts")).shape != (5,)
        or sampling.get("full_volumes_retained") is not False
        or sampling.get("full_node_stacks_retained") is not False
    ):
        msg = "R8 scan/slice/streaming metrics disagree with config."
        raise RuntimeError(msg)

    convergence = diagnostics["convergence"]
    if not isinstance(convergence, Mapping):
        msg = "R8 convergence metrics must be a mapping."
        raise RuntimeError(msg)
    for group_name, pair, denominator in (
        (
            "axial",
            registered["sample_a_cases"]["axial_pair"],
            registered["sample_a_cases"]["axial_reference"],
        ),
        (
            "lateral",
            registered["sample_a_cases"]["lateral_pair"],
            registered["sample_a_cases"]["lateral_reference"],
        ),
    ):
        group = convergence.get(group_name)
        if (
            not isinstance(group, Mapping)
            or list(group.get("pair", [])) != list(pair)
            or group.get("denominator") != denominator
        ):
            msg = f"R8 {group_name} comparison disagrees with config."
            raise RuntimeError(msg)
        for output in ("U_A_exit", "P_B", "I_stack"):
            if not np.isfinite(float(group["acceptance"][output])):
                msg = f"R8 {group_name} {output} acceptance is non-finite."
                raise RuntimeError(msg)
            if not isinstance(group["pass"][output], (bool, np.bool_)):
                msg = f"R8 {group_name} {output} pass flag must be boolean."
                raise RuntimeError(msg)
    open_metrics = convergence.get("open")
    if (
        not isinstance(open_metrics, Mapping)
        or list(open_metrics.get("pair", []))
        != list(open_control["acceptance_pair"])
        or open_metrics.get("denominator") != open_control["denominator"]
        or not np.isfinite(float(open_metrics["I_stack"]))
        or not isinstance(open_metrics["pass"], (bool, np.bool_))
        or not isinstance(convergence.get("all_pass"), (bool, np.bool_))
    ):
        msg = "R8 open comparison metrics disagree with config."
        raise RuntimeError(msg)

    visibility = diagnostics["visibility"]
    if not isinstance(visibility, Mapping):
        msg = "R8 visibility metrics must be a mapping."
        raise RuntimeError(msg)
    for label in ("waist_minus", "waist_plus"):
        for output in ("U_A_exit", "P_B", "I_stack"):
            if not np.isfinite(float(visibility["signals"][label][output])):
                msg = f"R8 {label} {output} signal is non-finite."
                raise RuntimeError(msg)
        per_frame = np.asarray(
            visibility["per_frame_I_stack_relative_l2"][label]
        )
        if per_frame.shape != (expected_scans,) or not np.all(np.isfinite(per_frame)):
            msg = f"R8 {label} per-frame signal has the wrong shape."
            raise RuntimeError(msg)
    if not isinstance(visibility["detector_visibility_pass"], (bool, np.bool_)):
        msg = "R8 detector visibility flag must be boolean."
        raise RuntimeError(msg)

    context = diagnostics["model_uncertainty_context"]
    _require_equal_trees(
        registered["r6_context"], context, "R8 independent R6 context"
    )
    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    expected_thresholds = {
        "convergence_relative_l2_max": acceptance["convergence_relative_l2_max"],
        "algebra_relative_l2_max": acceptance["algebra_relative_l2_max"],
        "determinism_relative_l2_max": acceptance["determinism_relative_l2_max"],
        "detector_visibility_signal_to_floor_min": acceptance[
            "detector_visibility_signal_to_floor_min"
        ],
    }
    _require_equal_trees(expected_thresholds, thresholds, "R8 thresholds")


def _validate_r9_metrics(
    config: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> None:
    """Validate compact R9 metrics before JSON/HDF5 persistence."""

    registered = config.get("diagnostics_r9")
    if not isinstance(registered, Mapping) or registered.get("enabled") is not True:
        msg = "R9 metrics are only valid for diagnostics_r9.enabled=true."
        raise RuntimeError(msg)
    required = {
        "version",
        "methods",
        "r8_provenance",
        "sampling",
        "passband",
        "interface_controls",
        "r8_reproduction",
        "comparisons",
        "restriction_controls",
        "spectral_controls",
        "determinism",
        "thresholds",
        "outcome_flags",
        "legacy_experiment_status_preserved",
        "all_finite",
        "hard_checks_pass",
        "status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        msg = f"diagnostics_r9 metrics are missing keys: {missing}"
        raise RuntimeError(msg)
    if diagnostics["version"] != registered["version"]:
        msg = "diagnostics_r9 version disagrees with config."
        raise RuntimeError(msg)
    _require_equal_trees(registered["methods"], diagnostics["methods"], "R9 methods")
    _require_equal_trees(
        registered["r8_provenance"],
        diagnostics["r8_provenance"],
        "R9 R8 provenance",
    )
    if diagnostics["status"] not in {"Passed", "Inconclusive", "Failed"}:
        msg = f"Invalid diagnostics_r9 status: {diagnostics['status']!r}."
        raise RuntimeError(msg)
    for name in (
        "legacy_experiment_status_preserved",
        "all_finite",
        "hard_checks_pass",
    ):
        if not isinstance(diagnostics[name], (bool, np.bool_)):
            msg = f"diagnostics_r9 {name} must be boolean."
            raise RuntimeError(msg)
    if diagnostics["legacy_experiment_status_preserved"] is not True:
        msg = "R9 must preserve the legacy experiment status."
        raise RuntimeError(msg)
    if not bool(diagnostics["all_finite"]):
        msg = "diagnostics_r9 reports non-finite outputs."
        raise RuntimeError(msg)

    sampling = diagnostics["sampling"]
    if not isinstance(sampling, Mapping):
        msg = "diagnostics_r9 sampling must be a mapping."
        raise RuntimeError(msg)
    cases = registered["sample_a_cases"]["cases"]
    for expected, actual, name in (
        (registered["interface"]["factor"], sampling.get("interface_factor"), "q"),
        ([case["id"] for case in cases], sampling.get("case_ids"), "case IDs"),
        ([case["shape"] for case in cases], sampling.get("sample_a_shapes"), "shapes"),
        ([case["dx_m"] for case in cases], sampling.get("sample_a_dx_m"), "dx"),
        ([case["dz_m"] for case in cases], sampling.get("sample_a_dz_m"), "dz"),
    ):
        _require_equal_trees(expected, actual, f"R9 sampling {name}")
    if (
        np.asarray(sampling.get("slice_counts")).shape != (4,)
        or sampling.get("full_volumes_retained") is not False
        or sampling.get("detector_path_recomputed") is not False
    ):
        msg = "R9 slice/streaming metrics disagree with config."
        raise RuntimeError(msg)

    passband = diagnostics["passband"]
    if not isinstance(passband, Mapping):
        msg = "R9 passband metrics must be a mapping."
        raise RuntimeError(msg)
    expected_cutoff = registered["physical_passband"][
        "cutoff_cycles_per_m"
    ]
    if float(passband.get("cutoff_cycles_per_m", np.nan)) != float(
        expected_cutoff
    ):
        msg = "R9 passband cutoff disagrees with config."
        raise RuntimeError(msg)
    native_controls = passband.get("native_projection_controls")
    expected_ids = [case["id"] for case in cases]
    if not isinstance(native_controls, Mapping) or set(native_controls) != set(
        expected_ids
    ):
        msg = "R9 native projection controls have the wrong cases."
        raise RuntimeError(msg)
    for case, case_id in zip(cases, expected_ids, strict=True):
        controls = native_controls[case_id]
        if not isinstance(controls, Mapping):
            msg = f"R9 projection controls for {case_id} must be a mapping."
            raise RuntimeError(msg)
        _require_equal_trees(
            case["shape"], controls.get("shape"), f"R9 {case_id} shape"
        )
        if (
            float(controls.get("dx_m", np.nan)) != float(case["dx_m"])
            or int(controls.get("mask_true_count", 0)) <= 0
            or int(controls.get("mask_total_count", 0)) <= 0
            or controls.get("all_finite") is not True
        ):
            msg = f"R9 projection controls for {case_id} are invalid."
            raise RuntimeError(msg)
        for key in (
            "mask_fraction",
            "retained_reference_energy_fraction",
            "repeat_relative_l2",
            "idempotence_relative_l2",
            "constant_max_abs_error",
        ):
            if not np.isfinite(float(controls[key])):
                msg = f"R9 projection control {case_id}/{key} is non-finite."
                raise RuntimeError(msg)

    interface = diagnostics["interface_controls"]
    if not isinstance(interface, Mapping):
        msg = "R9 interface controls must be a mapping."
        raise RuntimeError(msg)
    for key in (
        "fraction_bound_error_by_case",
        "index_bound_error_by_case",
        "subnode_count_identity_error_by_case",
        "air_volume_relative_error_by_case",
        "slice_width_sum_absolute_error_m_by_case",
    ):
        values = np.asarray(interface.get(key), dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            msg = f"R9 interface control {key} has the wrong shape."
            raise RuntimeError(msg)

    reproduction = diagnostics["r8_reproduction"]
    if not isinstance(reproduction, Mapping):
        msg = "R9 reproduction metrics must be a mapping."
        raise RuntimeError(msg)
    for key in (
        "raw_axial_absolute_error",
        "raw_lateral_bilinear_absolute_error",
    ):
        if not np.isfinite(float(reproduction[key])):
            msg = f"R9 reproduction metric {key} is non-finite."
            raise RuntimeError(msg)
    if not isinstance(reproduction.get("pass"), (bool, np.bool_)):
        msg = "R9 reproduction pass flag must be boolean."
        raise RuntimeError(msg)

    comparisons = diagnostics["comparisons"]
    expected_comparisons = {
        "r8_axial_reproduction": (
            ["axial_coarse", "common_reference"],
            "common_reference",
            "direct_same_grid",
        ),
        "axial_refinement": (
            ["common_reference", "axial_fine_reference"],
            "axial_fine_reference",
            "direct_same_grid",
        ),
        "lateral_bilinear": (
            ["common_reference", "lateral_fine_reference"],
            "restricted_lateral_fine_reference",
            "centered_bilinear_complex_field",
        ),
        "lateral_cell_average": (
            ["common_reference", "lateral_fine_reference"],
            "restricted_lateral_fine_reference",
            "aligned_2x2_complex_cell_average",
        ),
    }
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(
        expected_comparisons
    ):
        msg = "R9 comparison groups do not match the registration."
        raise RuntimeError(msg)
    for name, (pair, denominator, restriction) in expected_comparisons.items():
        values = comparisons[name]
        if (
            not isinstance(values, Mapping)
            or list(values.get("pair", [])) != pair
            or values.get("denominator") != denominator
            or values.get("restriction") != restriction
        ):
            msg = f"R9 comparison {name} disagrees with config."
            raise RuntimeError(msg)
        for key in ("raw_relative_l2", "external_passband_relative_l2"):
            if not np.isfinite(float(values[key])):
                msg = f"R9 comparison {name}/{key} is non-finite."
                raise RuntimeError(msg)
        pass_flags = values.get("pass")
        if not isinstance(pass_flags, Mapping) or any(
            not isinstance(pass_flags.get(key), (bool, np.bool_))
            for key in ("raw", "external_passband")
        ):
            msg = f"R9 comparison {name} pass flags are invalid."
            raise RuntimeError(msg)
        energy = values.get("difference_energy")
        if not isinstance(energy, Mapping) or any(
            not np.isfinite(float(energy[key]))
            for key in (
                "total",
                "inside_external_passband",
                "outside_external_passband",
                "inside_fraction",
                "outside_fraction",
                "parseval_closure_relative_error",
                "inside_outside_orthogonality_relative_error",
            )
        ):
            msg = f"R9 comparison {name} energy attribution is invalid."
            raise RuntimeError(msg)

    restriction = diagnostics["restriction_controls"]
    if (
        not isinstance(restriction, Mapping)
        or list(restriction.get("methods", []))
        != list(registered["lateral_restrictions"]["methods"])
        or not isinstance(restriction.get("pass"), (bool, np.bool_))
    ):
        msg = "R9 restriction controls disagree with config."
        raise RuntimeError(msg)
    spectral = diagnostics["spectral_controls"]
    if not isinstance(spectral, Mapping) or any(
        not np.isfinite(float(spectral[key]))
        for key in (
            "maximum_parseval_closure_relative_error",
            "maximum_inside_outside_orthogonality_relative_error",
        )
    ):
        msg = "R9 spectral controls are invalid."
        raise RuntimeError(msg)
    determinism = diagnostics["determinism"]
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("scope")
        != registered["determinism"]["scope"]
        or not np.isfinite(float(determinism["relative_l2"]))
        or not isinstance(determinism.get("pass"), (bool, np.bool_))
    ):
        msg = "R9 determinism controls are invalid."
        raise RuntimeError(msg)

    thresholds = diagnostics["thresholds"]
    acceptance = config["acceptance"]
    expected_thresholds = {
        "convergence_relative_l2_max": acceptance[
            "convergence_relative_l2_max"
        ],
        "algebra_relative_l2_max": acceptance["algebra_relative_l2_max"],
        "determinism_relative_l2_max": acceptance[
            "determinism_relative_l2_max"
        ],
    }
    _require_equal_trees(expected_thresholds, thresholds, "R9 thresholds")
    flags = diagnostics["outcome_flags"]
    if not isinstance(flags, Mapping) or any(
        not isinstance(flags.get(name), (bool, np.bool_))
        for name in ("passband_convergence_pass", "raw_convergence_pass")
    ):
        msg = "R9 outcome flags are invalid."
        raise RuntimeError(msg)
    hard_pass = bool(diagnostics["hard_checks_pass"])
    passband_pass = bool(flags["passband_convergence_pass"])
    raw_pass = bool(flags["raw_convergence_pass"])
    expected_status = (
        "Failed" if not hard_pass else "Passed" if passband_pass else "Inconclusive"
    )
    if diagnostics["status"] != expected_status:
        msg = "R9 status is inconsistent with frozen outcome logic."
        raise RuntimeError(msg)
    if not hard_pass:
        expected_code = "a_exit_attribution_blocked"
    elif not passband_pass:
        expected_code = "external_propagating_band_discrepancy_remains"
    elif raw_pass:
        expected_code = "raw_and_external_passband_a_exit_converged"
    else:
        expected_code = (
            "raw_discrepancy_attributed_outside_external_propagating_gate"
        )
    if flags.get("interpretation_code") != expected_code:
        msg = "R9 interpretation code is inconsistent with frozen logic."
        raise RuntimeError(msg)


def _validate_hdf5(
    path: Path,
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config_yaml: str,
) -> None:
    """Check required exp040 paths, shapes, finiteness, and absent groups."""

    required = (
        "entry/data/I_stack",
        "entry/data/scan_positions",
        "entry/instrument/wavelength",
        "entry/instrument/dx",
        "entry/instrument/z_AB",
        "entry/instrument/z_BC",
        "entry/instrument/detector_pixel_size",
        "entry/instrument/internal_reference_index",
        "entry/instrument/external_medium_index",
        "entry/sample/sample_A_type",
        "entry/sample/tgv_parameters",
        "entry/sample/sample_B_type",
        "entry/sample/sample_B_parameters",
        "entry/truth/n_volume",
        "entry/truth/z_m",
        "entry/truth/slice_thickness_m",
        "entry/truth/diameter_z_m",
        "entry/truth/incident_field_true",
        "entry/truth/U_A_exit_true",
        "entry/truth/P_B_true",
        "entry/truth/B_true",
        "entry/truth/parameter_sweep/case_ids",
        "entry/truth/parameter_sweep/d_waist_m",
        "entry/truth/parameter_sweep/U_A_exit_true",
        "entry/truth/parameter_sweep/P_B_true",
        "entry/truth/parameter_sweep/I_stack_true",
        "entry/config_yaml",
        "entry/metadata",
        "entry/metrics",
    )
    with h5py.File(path, "r") as h5:
        missing = [name for name in required if name not in h5]
        if missing:
            msg = f"HDF5 is missing required exp040 paths: {missing}"
            raise RuntimeError(msg)
        entry = h5["entry"]
        expected_top_level = {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        if set(entry) != expected_top_level:
            msg = (
                "Unexpected /entry children: "
                f"expected {sorted(expected_top_level)}, got {sorted(entry)}."
            )
            raise RuntimeError(msg)
        forbidden = [
            name
            for name in ("reconstruction", "calibration", "preprocessing")
            if name in entry
        ]
        if forbidden:
            msg = f"HDF5 contains inapplicable groups: {forbidden}"
            raise RuntimeError(msg)
        stored_config_yaml = _decode_hdf5_value(entry["config_yaml"][()])
        if stored_config_yaml != config_yaml:
            msg = "HDF5 config_yaml does not match the executed config."
            raise RuntimeError(msg)
        _require_equal_trees(
            dict(metadata), _read_hdf5_tree(entry["metadata"]), "metadata"
        )
        _require_equal_trees(
            dict(metrics), _read_hdf5_tree(entry["metrics"]), "metrics"
        )

        enabled_stages = [
            stage
            for stage, enabled in (
                ("R1", _r1_enabled(config)),
                ("R2", _r2_enabled(config)),
                ("R3", _r3_enabled(config)),
                ("R4", _r4_enabled(config)),
                ("R5", _r5_enabled(config)),
                ("R6", _r6_enabled(config)),
                ("R7", _r7_enabled(config)),
                ("R8", _r8_enabled(config)),
                ("R9", _r9_enabled(config)),
            )
            if enabled
        ]
        if enabled_stages:
            expected_stage = enabled_stages[-1]
            if metadata.get("diagnostic_stage") != expected_stage:
                msg = (
                    "Enabled diagnostic metadata must set diagnostic_stage="
                    f"{expected_stage}."
                )
                raise RuntimeError(msg)
        elif "diagnostic_stage" in metadata:
            msg = "Legacy exp040 metadata must not set diagnostic_stage."
            raise RuntimeError(msg)

        if _r1_enabled(config):
            if "diagnostics_r1" not in metrics:
                msg = "Enabled R1 run is missing metrics/diagnostics_r1."
                raise RuntimeError(msg)
            r1_metrics = metrics["diagnostics_r1"]
            if not isinstance(r1_metrics, Mapping):
                msg = "metrics/diagnostics_r1 must be a mapping."
                raise RuntimeError(msg)
            _validate_r1_metrics(config, r1_metrics)
            if metadata.get("diagnostics_r1_status") != r1_metrics["status"]:
                msg = "R1 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r1" in metrics
            or "diagnostics_r1_status" in metadata
        ):
            msg = "Legacy exp040 outputs must not contain R1-only metadata/metrics."
            raise RuntimeError(msg)

        if _r2_enabled(config):
            if "diagnostics_r2" not in metrics:
                msg = "Enabled R2 run is missing metrics/diagnostics_r2."
                raise RuntimeError(msg)
            r2_metrics = metrics["diagnostics_r2"]
            if not isinstance(r2_metrics, Mapping):
                msg = "metrics/diagnostics_r2 must be a mapping."
                raise RuntimeError(msg)
            _validate_r2_metrics(config, r2_metrics)
            if metadata.get("diagnostics_r2_status") != r2_metrics["status"]:
                msg = "R2 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r2" in metrics
            or "diagnostics_r2_status" in metadata
        ):
            msg = "Non-R2 exp040 outputs must not contain R2-only fields."
            raise RuntimeError(msg)

        if _r3_enabled(config):
            if "diagnostics_r3" not in metrics:
                msg = "Enabled R3 run is missing metrics/diagnostics_r3."
                raise RuntimeError(msg)
            r3_metrics = metrics["diagnostics_r3"]
            if not isinstance(r3_metrics, Mapping):
                msg = "metrics/diagnostics_r3 must be a mapping."
                raise RuntimeError(msg)
            _validate_r3_metrics(config, r3_metrics)
            if metadata.get("diagnostics_r3_status") != r3_metrics["status"]:
                msg = "R3 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r3" in metrics
            or "diagnostics_r3_status" in metadata
        ):
            msg = "Non-R3 exp040 outputs must not contain R3-only fields."
            raise RuntimeError(msg)

        if _r4_enabled(config):
            if "diagnostics_r4" not in metrics:
                msg = "Enabled R4 run is missing metrics/diagnostics_r4."
                raise RuntimeError(msg)
            r4_metrics = metrics["diagnostics_r4"]
            if not isinstance(r4_metrics, Mapping):
                msg = "metrics/diagnostics_r4 must be a mapping."
                raise RuntimeError(msg)
            _validate_r4_metrics(config, r4_metrics)
            if metadata.get("diagnostics_r4_status") != r4_metrics["status"]:
                msg = "R4 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r4" in metrics
            or "diagnostics_r4_status" in metadata
        ):
            msg = "Non-R4 exp040 outputs must not contain R4-only fields."
            raise RuntimeError(msg)

        if _r5_enabled(config):
            if "diagnostics_r5" not in metrics:
                msg = "Enabled R5 run is missing metrics/diagnostics_r5."
                raise RuntimeError(msg)
            r5_metrics = metrics["diagnostics_r5"]
            if not isinstance(r5_metrics, Mapping):
                msg = "metrics/diagnostics_r5 must be a mapping."
                raise RuntimeError(msg)
            _validate_r5_metrics(config, r5_metrics)
            if metadata.get("diagnostics_r5_status") != r5_metrics["status"]:
                msg = "R5 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r5" in metrics
            or "diagnostics_r5_status" in metadata
        ):
            msg = "Non-R5 exp040 outputs must not contain R5-only fields."
            raise RuntimeError(msg)

        if _r6_enabled(config):
            if "diagnostics_r6" not in metrics:
                msg = "Enabled R6 run is missing metrics/diagnostics_r6."
                raise RuntimeError(msg)
            r6_metrics = metrics["diagnostics_r6"]
            if not isinstance(r6_metrics, Mapping):
                msg = "metrics/diagnostics_r6 must be a mapping."
                raise RuntimeError(msg)
            _validate_r6_metrics(config, r6_metrics)
            if metadata.get("diagnostics_r6_status") != r6_metrics["status"]:
                msg = "R6 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r6" in metrics
            or "diagnostics_r6_status" in metadata
        ):
            msg = "Non-R6 exp040 outputs must not contain R6-only fields."
            raise RuntimeError(msg)

        if _r7_enabled(config):
            if "diagnostics_r7" not in metrics:
                msg = "Enabled R7 run is missing metrics/diagnostics_r7."
                raise RuntimeError(msg)
            r7_metrics = metrics["diagnostics_r7"]
            if not isinstance(r7_metrics, Mapping):
                msg = "metrics/diagnostics_r7 must be a mapping."
                raise RuntimeError(msg)
            _validate_r7_metrics(config, r7_metrics)
            if metadata.get("diagnostics_r7_status") != r7_metrics["status"]:
                msg = "R7 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r7" in metrics
            or "diagnostics_r7_status" in metadata
        ):
            msg = "Non-R7 exp040 outputs must not contain R7-only fields."
            raise RuntimeError(msg)

        if _r8_enabled(config):
            if "diagnostics_r8" not in metrics:
                msg = "Enabled R8 run is missing metrics/diagnostics_r8."
                raise RuntimeError(msg)
            r8_metrics = metrics["diagnostics_r8"]
            if not isinstance(r8_metrics, Mapping):
                msg = "metrics/diagnostics_r8 must be a mapping."
                raise RuntimeError(msg)
            _validate_r8_metrics(config, r8_metrics)
            if metadata.get("diagnostics_r8_status") != r8_metrics["status"]:
                msg = "R8 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r8" in metrics
            or "diagnostics_r8_status" in metadata
        ):
            msg = "Non-R8 exp040 outputs must not contain R8-only fields."
            raise RuntimeError(msg)

        if _r9_enabled(config):
            if "diagnostics_r9" not in metrics:
                msg = "Enabled R9 run is missing metrics/diagnostics_r9."
                raise RuntimeError(msg)
            r9_metrics = metrics["diagnostics_r9"]
            if not isinstance(r9_metrics, Mapping):
                msg = "metrics/diagnostics_r9 must be a mapping."
                raise RuntimeError(msg)
            _validate_r9_metrics(config, r9_metrics)
            if metadata.get("diagnostics_r9_status") != r9_metrics["status"]:
                msg = "R9 metadata and metrics status disagree."
                raise RuntimeError(msg)
        elif (
            "diagnostics_r9" in metrics
            or "diagnostics_r9_status" in metadata
        ):
            msg = "Non-R9 exp040 outputs must not contain R9-only fields."
            raise RuntimeError(msg)

        launch_status = str(config["experiment"]["status"])
        if metadata["config_status_at_launch"] != launch_status:
            msg = "metadata config_status_at_launch disagrees with config."
            raise RuntimeError(msg)
        if metadata["experiment_status"] != metrics["experiment_status"]:
            msg = "metadata and metrics experiment_status disagree."
            raise RuntimeError(msg)

        optics = config["optics"]
        internal_index = float(optics["internal_reference_index"])
        external_index = float(optics["external_medium_index"])
        if (
            float(entry["instrument/internal_reference_index"][()])
            != internal_index
            or float(metadata["internal_reference_index"]) != internal_index
        ):
            msg = "Internal reference index is inconsistent across outputs."
            raise RuntimeError(msg)
        if (
            float(entry["instrument/external_medium_index"][()])
            != external_index
            or float(metadata["external_medium_index"]) != external_index
        ):
            msg = "External medium index is inconsistent across outputs."
            raise RuntimeError(msg)
        expected_axes = {
            "field": ["y", "x"],
            "volume": ["z", "y", "x"],
            "intensity_stack": ["scan", "y", "x"],
            "scan_position_columns": ["x", "y"],
        }
        expected_planes = {
            "sample_A_input": "z=0 entrance boundary",
            "sample_A_output": "z=L exit boundary",
            "z_AB_origin": "sample_A_exit",
        }
        _require_equal_trees(expected_axes, metadata["array_axes"], "array_axes")
        _require_equal_trees(expected_planes, metadata["planes"], "planes")

        baseline_shape = tuple(int(value) for value in optics["baseline_shape"])
        num_positions = int(config["scan"]["num_x"]) * int(
            config["scan"]["num_y"]
        )
        baseline_intensity = entry["data/I_stack"][...]
        baseline_exit = entry["truth/U_A_exit_true"][...]
        baseline_probe = entry["truth/P_B_true"][...]
        sweep_intensity = entry["truth/parameter_sweep/I_stack_true"][...]
        sweep_exit = entry["truth/parameter_sweep/U_A_exit_true"][...]
        sweep_probe = entry["truth/parameter_sweep/P_B_true"][...]
        case_ids = _decode_hdf5_value(
            entry["truth/parameter_sweep/case_ids"][()]
        )
        if case_ids != ["waist_minus", "baseline", "waist_plus"]:
            msg = "Unexpected parameter-sweep case order."
            raise RuntimeError(msg)
        baseline_index = case_ids.index("baseline")
        baseline_pairs = (
            ("I_stack", baseline_intensity, sweep_intensity[baseline_index]),
            ("U_A_exit", baseline_exit, sweep_exit[baseline_index]),
            ("P_B", baseline_probe, sweep_probe[baseline_index]),
        )
        for name, baseline_values, sweep_values in baseline_pairs:
            if not np.array_equal(baseline_values, sweep_values):
                msg = f"baseline {name} must exactly equal its sweep baseline."
                raise RuntimeError(msg)

        volume = entry["truth/n_volume"][...]
        z_m = entry["truth/z_m"][...]
        widths = entry["truth/slice_thickness_m"][...]
        diameter = entry["truth/diameter_z_m"][...]
        incident = entry["truth/incident_field_true"][...]
        sample_b = entry["truth/B_true"][...]
        positions = entry["data/scan_positions"][...]
        nz = len(widths)
        expected_shapes = {
            "I_stack": (num_positions, *baseline_shape),
            "scan_positions": (num_positions, 2),
            "n_volume": (nz, *baseline_shape),
            "z_m": (nz,),
            "slice_thickness_m": (nz,),
            "diameter_z_m": (nz,),
            "incident_field_true": baseline_shape,
            "U_A_exit_true": baseline_shape,
            "P_B_true": baseline_shape,
            "B_true": baseline_shape,
            "sweep_I_stack": (3, num_positions, *baseline_shape),
            "sweep_U_A_exit": (3, *baseline_shape),
            "sweep_P_B": (3, *baseline_shape),
        }
        arrays = {
            "I_stack": baseline_intensity,
            "scan_positions": positions,
            "n_volume": volume,
            "z_m": z_m,
            "slice_thickness_m": widths,
            "diameter_z_m": diameter,
            "incident_field_true": incident,
            "U_A_exit_true": baseline_exit,
            "P_B_true": baseline_probe,
            "B_true": sample_b,
            "sweep_I_stack": sweep_intensity,
            "sweep_U_A_exit": sweep_exit,
            "sweep_P_B": sweep_probe,
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                msg = (
                    f"{name} has shape {arrays[name].shape}; "
                    f"expected {expected_shape}."
                )
                raise RuntimeError(msg)
        real_float64 = (
            "I_stack",
            "scan_positions",
            "n_volume",
            "z_m",
            "slice_thickness_m",
            "diameter_z_m",
            "sweep_I_stack",
        )
        complex128 = (
            "incident_field_true",
            "U_A_exit_true",
            "P_B_true",
            "B_true",
            "sweep_U_A_exit",
            "sweep_P_B",
        )
        if any(arrays[name].dtype != np.float64 for name in real_float64):
            msg = "One or more real exp040 arrays are not float64."
            raise RuntimeError(msg)
        if any(arrays[name].dtype != np.complex128 for name in complex128):
            msg = "One or more complex exp040 fields are not complex128."
            raise RuntimeError(msg)
        if not np.all(widths > 0.0):
            msg = "slice widths must be finite and positive."
            raise RuntimeError(msg)
        thickness = float(config["sample_a"]["thickness_m"])
        tolerance_cfg = config["acceptance"][
            "geometry_thickness_absolute_tolerance_m"
        ]
        width_tolerance = max(
            float(tolerance_cfg["fixed_floor_m"]),
            float(tolerance_cfg["floating_point_factor"])
            * np.finfo(np.float64).eps
            * thickness,
        )
        width_error = abs(float(np.sum(widths, dtype=np.float64)) - thickness)
        if width_error > width_tolerance:
            msg = (
                f"slice widths sum error {width_error:.6e} m exceeds "
                f"{width_tolerance:.6e} m."
            )
            raise RuntimeError(msg)
        if np.any(baseline_intensity < 0.0) or np.any(sweep_intensity < 0.0):
            msg = "Detector intensity must be non-negative."
            raise RuntimeError(msg)

        nonfinite_paths: list[str] = []

        def visitor(name: str, item: h5py.Dataset | h5py.Group) -> None:
            if not isinstance(item, h5py.Dataset):
                return
            if item.dtype.kind not in {"b", "i", "u", "f", "c"}:
                return
            values = item[...]
            if not np.all(np.isfinite(values)):
                nonfinite_paths.append(name)

        entry.visititems(visitor)
        if nonfinite_paths:
            msg = f"HDF5 contains non-finite numeric datasets: {nonfinite_paths}"
            raise RuntimeError(msg)


def _validate_external_files(
    run_dir: Path,
    output_path: Path,
    *,
    config_yaml: str,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    figure_paths: list[Path],
    config: Mapping[str, Any] | None = None,
) -> None:
    required_files = [
        run_dir / "config.yaml",
        run_dir / "metadata.json",
        run_dir / "metrics.json",
        output_path,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        msg = f"Run is missing required artifacts: {missing}"
        raise RuntimeError(msg)
    expected_figures = {
        str((run_dir / "figures" / name).resolve())
        for name in (
            _expected_figure_filenames(config)
            if config is not None
            else EXP040_FIGURE_FILENAMES
        )
    }
    actual_figures = {str(path.resolve()) for path in figure_paths}
    if expected_figures != actual_figures:
        msg = "Figure output names do not match the pre-registered set."
        raise RuntimeError(msg)
    if any(not path.is_file() or path.stat().st_size == 0 for path in figure_paths):
        msg = "One or more exp040 figures are missing or empty."
        raise RuntimeError(msg)
    external_config_yaml = config_to_yaml(load_config(run_dir / "config.yaml"))
    if external_config_yaml != config_yaml:
        msg = "External config.yaml does not match the executed config."
        raise RuntimeError(msg)
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        external_metadata = json.load(handle)
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        external_metrics = json.load(handle)
    _require_equal_trees(dict(metadata), external_metadata, "external metadata")
    _require_equal_trees(dict(metrics), external_metrics, "external metrics")
    with h5py.File(output_path, "r") as h5:
        entry = h5["entry"]
        stored_config_yaml = _decode_hdf5_value(entry["config_yaml"][()])
        if stored_config_yaml != external_config_yaml:
            msg = "External and HDF5 config disagree."
            raise RuntimeError(msg)
        _require_equal_trees(
            external_metadata,
            _read_hdf5_tree(entry["metadata"]),
            "external/HDF5 metadata",
        )
        _require_equal_trees(
            external_metrics,
            _read_hdf5_tree(entry["metrics"]),
            "external/HDF5 metrics",
        )


def _save_runtime_progress(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save non-scientific runtime progress for crash diagnosis."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    save_json(temporary_path, payload)
    temporary_path.replace(path)


def run(config_path: Path) -> Path:
    """Execute exp040 and return the newly created run directory."""

    config = load_config(config_path)
    validate_exp040_config(config)
    run_cfg = config["run"]
    run_dir = make_run_dir(
        PROJECT_ROOT / str(run_cfg["output_root"]),
        str(run_cfg["name"]),
    )
    save_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "created_at": created_at_utc(),
            "source_config": str(config_path.resolve()),
        },
    )
    runtime_events: list[dict[str, Any]] = []
    runtime_progress_path = run_dir / "run_progress.json"

    def record_runtime_progress(
        event: str, details: Mapping[str, Any]
    ) -> None:
        record = {
            "event": event,
            "recorded_at": created_at_utc(),
            "details": dict(details),
        }
        runtime_events.append(record)
        _save_runtime_progress(
            runtime_progress_path,
            {
                "purpose": "non_scientific_execution_diagnostic",
                "latest_event": record,
                "events": runtime_events,
            },
        )
        serialized_details = json.dumps(
            record["details"], sort_keys=True, separators=(",", ":")
        )
        print(
            f"runtime_progress: {event} {serialized_details}",
            flush=True,
        )

    r9_progress_callback = record_runtime_progress if _r9_enabled(config) else None
    if r9_progress_callback is not None:
        record_runtime_progress(
            "run_created", {"run_dir": str(run_dir.resolve())}
        )

    try:
        result = run_exp040_experiment(
            config, progress_callback=r9_progress_callback
        )
        metrics = result["metrics"]
        status = str(metrics["experiment_status"])
        diagnostics_r1_metrics = metrics.get("diagnostics_r1")
        if diagnostics_r1_metrics is not None and not isinstance(
            diagnostics_r1_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r1 must be a mapping."
            raise RuntimeError(msg)
        if _r1_enabled(config):
            if diagnostics_r1_metrics is None:
                msg = "R1-enabled experiment did not produce diagnostics_r1 metrics."
                raise RuntimeError(msg)
            _validate_r1_metrics(config, diagnostics_r1_metrics)
        diagnostics_r2_metrics = metrics.get("diagnostics_r2")
        if diagnostics_r2_metrics is not None and not isinstance(
            diagnostics_r2_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r2 must be a mapping."
            raise RuntimeError(msg)
        if _r2_enabled(config):
            if diagnostics_r2_metrics is None:
                msg = "R2-enabled experiment did not produce diagnostics_r2 metrics."
                raise RuntimeError(msg)
            _validate_r2_metrics(config, diagnostics_r2_metrics)
        diagnostics_r3_metrics = metrics.get("diagnostics_r3")
        if diagnostics_r3_metrics is not None and not isinstance(
            diagnostics_r3_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r3 must be a mapping."
            raise RuntimeError(msg)
        if _r3_enabled(config):
            if diagnostics_r3_metrics is None:
                msg = "R3-enabled experiment did not produce diagnostics_r3 metrics."
                raise RuntimeError(msg)
            _validate_r3_metrics(config, diagnostics_r3_metrics)
        diagnostics_r4_metrics = metrics.get("diagnostics_r4")
        if diagnostics_r4_metrics is not None and not isinstance(
            diagnostics_r4_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r4 must be a mapping."
            raise RuntimeError(msg)
        if _r4_enabled(config):
            if diagnostics_r4_metrics is None:
                msg = "R4-enabled experiment did not produce diagnostics_r4 metrics."
                raise RuntimeError(msg)
            _validate_r4_metrics(config, diagnostics_r4_metrics)
        diagnostics_r5_metrics = metrics.get("diagnostics_r5")
        if diagnostics_r5_metrics is not None and not isinstance(
            diagnostics_r5_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r5 must be a mapping."
            raise RuntimeError(msg)
        if _r5_enabled(config):
            if diagnostics_r5_metrics is None:
                msg = "R5-enabled experiment did not produce diagnostics_r5 metrics."
                raise RuntimeError(msg)
            _validate_r5_metrics(config, diagnostics_r5_metrics)
        diagnostics_r6_metrics = metrics.get("diagnostics_r6")
        if diagnostics_r6_metrics is not None and not isinstance(
            diagnostics_r6_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r6 must be a mapping."
            raise RuntimeError(msg)
        if _r6_enabled(config):
            if diagnostics_r6_metrics is None:
                msg = "R6-enabled experiment did not produce diagnostics_r6 metrics."
                raise RuntimeError(msg)
            _validate_r6_metrics(config, diagnostics_r6_metrics)
        diagnostics_r7_metrics = metrics.get("diagnostics_r7")
        if diagnostics_r7_metrics is not None and not isinstance(
            diagnostics_r7_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r7 must be a mapping."
            raise RuntimeError(msg)
        if _r7_enabled(config):
            if diagnostics_r7_metrics is None:
                msg = "R7-enabled experiment did not produce diagnostics_r7 metrics."
                raise RuntimeError(msg)
            _validate_r7_metrics(config, diagnostics_r7_metrics)
        diagnostics_r8_metrics = metrics.get("diagnostics_r8")
        if diagnostics_r8_metrics is not None and not isinstance(
            diagnostics_r8_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r8 must be a mapping."
            raise RuntimeError(msg)
        if _r8_enabled(config):
            if diagnostics_r8_metrics is None:
                msg = "R8-enabled experiment did not produce diagnostics_r8 metrics."
                raise RuntimeError(msg)
            _validate_r8_metrics(config, diagnostics_r8_metrics)
        diagnostics_r9_metrics = metrics.get("diagnostics_r9")
        if diagnostics_r9_metrics is not None and not isinstance(
            diagnostics_r9_metrics, Mapping
        ):
            msg = "metrics.diagnostics_r9 must be a mapping."
            raise RuntimeError(msg)
        if _r9_enabled(config):
            if diagnostics_r9_metrics is None:
                msg = "R9-enabled experiment did not produce diagnostics_r9 metrics."
                raise RuntimeError(msg)
            _validate_r9_metrics(config, diagnostics_r9_metrics)
        metadata = _metadata(
            config,
            config_path,
            run_dir,
            status,
            diagnostics_r1_metrics=diagnostics_r1_metrics,
            diagnostics_r2_metrics=diagnostics_r2_metrics,
            diagnostics_r3_metrics=diagnostics_r3_metrics,
            diagnostics_r4_metrics=diagnostics_r4_metrics,
            diagnostics_r5_metrics=diagnostics_r5_metrics,
            diagnostics_r6_metrics=diagnostics_r6_metrics,
            diagnostics_r7_metrics=diagnostics_r7_metrics,
            diagnostics_r8_metrics=diagnostics_r8_metrics,
            diagnostics_r9_metrics=diagnostics_r9_metrics,
        )
        config_yaml = config_to_yaml(config)
        save_json(run_dir / "metadata.json", metadata)
        save_json(run_dir / "metrics.json", metrics)

        figure_paths = save_exp040_figures(result, run_dir / "figures")
        if _r1_enabled(config):
            if config["output"].get("save_r1_figures") is not True:
                msg = "R1-enabled run requires output.save_r1_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r1_figures(result, run_dir / "figures"))
        if _r2_enabled(config):
            if config["output"].get("save_r2_figures") is not True:
                msg = "R2-enabled run requires output.save_r2_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r2_figures(result, run_dir / "figures"))
        if _r3_enabled(config):
            if config["output"].get("save_r3_figures") is not True:
                msg = "R3-enabled run requires output.save_r3_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r3_figures(result, run_dir / "figures"))
        if _r4_enabled(config):
            if config["output"].get("save_r4_figures") is not True:
                msg = "R4-enabled run requires output.save_r4_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r4_figures(result, run_dir / "figures"))
        if _r5_enabled(config):
            if config["output"].get("save_r5_figures") is not True:
                msg = "R5-enabled run requires output.save_r5_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r5_figures(result, run_dir / "figures"))
        if _r6_enabled(config):
            if config["output"].get("save_r6_figures") is not True:
                msg = "R6-enabled run requires output.save_r6_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r6_figures(result, run_dir / "figures"))
        if _r7_enabled(config):
            if config["output"].get("save_r7_figures") is not True:
                msg = "R7-enabled run requires output.save_r7_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r7_figures(result, run_dir / "figures"))
        if _r8_enabled(config):
            if config["output"].get("save_r8_figures") is not True:
                msg = "R8-enabled run requires output.save_r8_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r8_figures(result, run_dir / "figures"))
        if _r9_enabled(config):
            if config["output"].get("save_r9_figures") is not True:
                msg = "R9-enabled run requires output.save_r9_figures=true."
                raise RuntimeError(msg)
            figure_paths.extend(_save_r9_figures(result, run_dir / "figures"))
        output_path = run_dir / "outputs" / str(
            config["output"]["hdf5_filename"]
        )
        payload = build_exp040_hdf5_payload(
            result,
            config,
            config_yaml=config_yaml,
            metadata=metadata,
        )
        save_ptycho_hdf5(output_path, **payload)
        _validate_hdf5(
            output_path,
            config=config,
            metadata=metadata,
            metrics=metrics,
            config_yaml=config_yaml,
        )
        _validate_external_files(
            run_dir,
            output_path,
            config_yaml=config_yaml,
            metadata=metadata,
            metrics=metrics,
            figure_paths=figure_paths,
            config=config,
        )
        if r9_progress_callback is not None:
            record_runtime_progress(
                "artifacts_validated", {"experiment_status": status}
            )
        save_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "experiment_status": status,
                "completed_at": created_at_utc(),
                "artifacts_validated": True,
            },
        )
    except Exception as error:
        if r9_progress_callback is not None:
            record_runtime_progress(
                "python_exception",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        save_json(
            run_dir / "run_state.json",
            {
                "status": "failed_during_execution",
                "failed_at": created_at_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    convergence = metrics["convergence"]
    visibility = metrics["visibility"]
    print(f"run_dir: {run_dir.resolve()}")
    print(f"experiment_status: {metrics['experiment_status']}")
    for group_name in ("axial", "lateral", "fov"):
        values = convergence[group_name]["acceptance"]
        print(
            f"{group_name}_relative_l2: "
            f"U_A_exit={values['U_A_exit']:.6e}, "
            f"P_B={values['P_B']:.6e}, "
            f"I_stack={values['I_stack']:.6e}"
        )
    print(
        "detector_signal_to_floor_min: "
        f"{visibility['detector_signal_to_floor_min']:.6e}"
    )
    if _r2_enabled(config):
        r2 = metrics["diagnostics_r2"]
        current = r2["period_aligned"]["current_asm"]["acceptance"]
        controlled = r2["period_aligned"]["alias_controlled"]["acceptance"]
        difference = r2["method_difference"]["largest_fov"]
        print(f"diagnostics_r2_status: {r2['status']}")
        print(
            "r2_current_aligned_relative_l2: "
            f"P_B={current['P_B']:.6e}, I_stack={current['I_stack']:.6e}"
        )
        print(
            "r2_alias_controlled_relative_l2: "
            f"P_B={controlled['P_B']:.6e}, "
            f"I_stack={controlled['I_stack']:.6e}"
        )
        print(
            "r2_largest_fov_method_difference: "
            f"P_B={difference['P_B']:.6e}, "
            f"I_stack={difference['I_stack']:.6e}"
        )
        print(
            "r2_interpretation: "
            f"{r2['outcome_flags']['interpretation_code']}"
        )
    if _r3_enabled(config):
        r3 = metrics["diagnostics_r3"]
        acceptance = r3["detector_sampling"]["acceptance"]
        detector_acceptance = acceptance["detector"]
        print(f"diagnostics_r3_status: {r3['status']}")
        print(
            "r3_factor2_to4_relative_l2: "
            f"P_B={acceptance['P_B']:.6e}, "
            "alias_point="
            f"{detector_acceptance['alias_controlled']['point_sample']:.6e}, "
            "alias_pixel="
            f"{detector_acceptance['alias_controlled']['pixel_box_average']:.6e}"
        )
        print(
            "r3_interpretation: "
            f"{r3['outcome_flags']['interpretation_code']}"
        )
    if _r4_enabled(config):
        r4 = metrics["diagnostics_r4"]
        acceptance = r4["convergence"]["acceptance"]
        controls = r4["quadrature_controls"]
        print(f"diagnostics_r4_status: {r4['status']}")
        print(
            "r4_q4_to_q8_relative_l2: "
            f"P_B={acceptance['P_B']:.6e}, "
            f"I_stack={acceptance['I_stack']:.6e}"
        )
        print(
            "r4_positive_quadrature_controls: "
            f"geometry={controls['max_node_geometry_normalized_error']:.6e}, "
            f"constant={controls['max_constant_abs_error']:.6e}, "
            f"sum={controls['max_sum_relative_error']:.6e}, "
            f"nonnegative={controls['all_outputs_nonnegative']}"
        )
    if _r5_enabled(config):
        r5 = metrics["diagnostics_r5"]
        effects = r5["effects"]
        print(f"diagnostics_r5_status: {r5['status']}")
        print(
            "r5_open_288_to_384_relative_l2: "
            f"{r5['open_boundary_convergence']['acceptance']:.6e}"
        )
        print(
            "r5_boundary_effects: "
            f"support={effects['support_relative_l2']:.6e}, "
            f"boundary={effects['boundary_relative_l2']:.6e}, "
            f"combined={effects['combined_relative_l2']:.6e}"
        )
        print(
            "r5_interpretation: "
            f"{r5['outcome_flags']['interpretation_code']}"
        )
    if _r6_enabled(config):
        r6 = metrics["diagnostics_r6"]
        effects = r6["support_effects"]
        nominal = r6["nominal_sensitivity"]
        print(f"diagnostics_r6_status: {r6['status']}")
        print(
            "r6_support_effect_envelope: "
            f"min={effects['minimum']:.6e}, "
            f"max={effects['maximum']:.6e}, "
            f"span={effects['span']:.6e}"
        )
        print(
            "r6_nominal_sensitivity_max: "
            f"{nominal['maximum']:.6e}"
        )
        print(
            "r6_interpretation: "
            f"{r6['outcome_flags']['interpretation_code']}"
        )
    if _r7_enabled(config):
        r7 = metrics["diagnostics_r7"]
        final_pair = r7["convergence"]["acceptance"]
        binary = r7["binary_effect"]["relative_l2_q1_to_q8"]
        print(f"diagnostics_r7_status: {r7['status']}")
        print(
            "r7_q4_to_q8_relative_l2: "
            f"U_A_exit={final_pair['U_A_exit']:.6e}, "
            f"P_B={final_pair['P_B']:.6e}, "
            f"I_stack={final_pair['I_stack']:.6e}"
        )
        print(
            "r7_q1_to_q8_relative_l2: "
            f"U_A_exit={binary['U_A_exit']:.6e}, "
            f"P_B={binary['P_B']:.6e}, "
            f"I_stack={binary['I_stack']:.6e}"
        )
        print(
            "r7_interpretation: "
            f"{r7['outcome_flags']['interpretation_code']}"
        )
    if _r8_enabled(config):
        r8 = metrics["diagnostics_r8"]
        axial = r8["convergence"]["axial"]["acceptance"]
        lateral = r8["convergence"]["lateral"]["acceptance"]
        print(f"diagnostics_r8_status: {r8['status']}")
        print(
            "r8_axial_relative_l2: "
            f"U_A_exit={axial['U_A_exit']:.6e}, "
            f"P_B={axial['P_B']:.6e}, "
            f"I_stack={axial['I_stack']:.6e}"
        )
        print(
            "r8_lateral_relative_l2: "
            f"U_A_exit={lateral['U_A_exit']:.6e}, "
            f"P_B={lateral['P_B']:.6e}, "
            f"I_stack={lateral['I_stack']:.6e}"
        )
        print(
            "r8_open_288_to_384_relative_l2: "
            f"{r8['convergence']['open']['I_stack']:.6e}"
        )
        print(
            "r8_detector_signal_to_floor_min: "
            f"{r8['visibility']['detector_signal_to_floor_min']:.6e}"
        )
        print(
            "r8_interpretation: "
            f"{r8['outcome_flags']['interpretation_code']}"
        )
    if _r9_enabled(config):
        r9 = metrics["diagnostics_r9"]
        comparisons = r9["comparisons"]
        axial = comparisons["axial_refinement"]
        lateral_bilinear = comparisons["lateral_bilinear"]
        lateral_average = comparisons["lateral_cell_average"]
        print(f"diagnostics_r9_status: {r9['status']}")
        print(
            "r9_axial_raw_and_passband_relative_l2: "
            f"raw={axial['raw_relative_l2']:.6e}, "
            f"passband={axial['external_passband_relative_l2']:.6e}"
        )
        print(
            "r9_lateral_bilinear_raw_and_passband_relative_l2: "
            f"raw={lateral_bilinear['raw_relative_l2']:.6e}, "
            "passband="
            f"{lateral_bilinear['external_passband_relative_l2']:.6e}"
        )
        print(
            "r9_lateral_cell_average_raw_and_passband_relative_l2: "
            f"raw={lateral_average['raw_relative_l2']:.6e}, "
            "passband="
            f"{lateral_average['external_passband_relative_l2']:.6e}"
        )
        print(
            "r9_interpretation: "
            f"{r9['outcome_flags']['interpretation_code']}"
        )
    return run_dir


def main() -> None:
    args = _parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
