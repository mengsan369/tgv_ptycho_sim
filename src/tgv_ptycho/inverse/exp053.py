"""Reconstruction-aware fixed-q8 interval fitting for exp053."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.inverse.exp051 import (
    make_exp051_candidate_generator,
    sha256_array_bytes,
    sha256_file,
)
from tgv_ptycho.inverse.exp051_local_control import make_air_fraction_stack
from tgv_ptycho.inverse.exp051_plateau_interval import (
    build_q8_cell_partition,
    expand_connected_cell_component,
    fixed_membership_bisection,
    q8_cell_index,
    q8_cell_midpoint,
)
from tgv_ptycho.inverse.waist_fit import (
    equal_budget_bounded_pattern_search,
    raw_complex_probe_loss,
    replay_error_metrics,
)
from tgv_ptycho.io.config import load_config
from tgv_ptycho.objects.tgv3d import make_tgv_air_fraction_slice

_LEGACY_MATCHED_Q4_TARGET_PATH = (
    "/entry/reconstruction/detector_quadrature_ablation/branches/"
    "matched_q4/P_B_rec"
)
_EXP053_FEEDBACK_MATCHED_Q4_TARGET_PATH = (
    "/entry/reconstruction/exp053_feedback_control/branches/"
    "spectrally_damped_gn_cg/P_B_rec"
)

ComplexGenerator = Callable[[float], NDArray[np.complex128]]


@dataclass(frozen=True)
class Exp053SourceArtifact:
    """Validated raw matched-q4 reconstructed-probe handoff from exp042."""

    run_dir: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    run_state_path: Path
    hdf5_path: Path
    target_hdf5_path: str
    operator_reference_hdf5_path: str
    source_config: dict[str, Any]
    source_metadata: dict[str, Any]
    source_metrics: dict[str, Any]
    source_state: dict[str, Any]
    file_sha256: dict[str, str]
    target_dataset_sha256: str
    operator_reference_dataset_sha256: str
    P_B_rec_raw: NDArray[np.complex128]
    P_B_true_operator_reference: NDArray[np.complex128]
    D_z_m: NDArray[np.float64]
    z_m: NDArray[np.float64]
    slice_widths_m: NDArray[np.float64]
    embedded_config_yaml: str


@dataclass(frozen=True)
class Exp051OracleIntervalArtifact:
    """Validated exp051 oracle interval used only for paired comparison."""

    run_dir: Path
    hdf5_path: Path
    file_sha256: dict[str, str]
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    run_state: dict[str, Any]
    lower_m: float
    upper_m: float
    lower_closed: bool
    upper_closed: bool


@dataclass
class _ProbeCache:
    generator: ComplexGenerator
    target: NDArray[np.complex128]
    bounds_m: tuple[float, float]
    values: dict[float, int] = field(default_factory=dict)
    diameters_m: list[float] = field(default_factory=list)
    probes: list[NDArray[np.complex128]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def get(self, diameter_m: float) -> tuple[NDArray[np.complex128], float, int]:
        """Return a cached candidate and its raw loss to reconstructed target."""

        key = float(np.float64(diameter_m))
        lower, upper = self.bounds_m
        if not np.isfinite(key) or not lower <= key <= upper:
            raise ValueError("candidate D_waist lies outside registered bounds.")
        index = self.values.get(key)
        if index is not None:
            return self.probes[index], self.losses[index], index
        probe = np.asarray(self.generator(key))
        if probe.shape != self.target.shape or probe.dtype != np.complex128:
            raise ValueError(
                "candidate generator must return target-shaped complex128."
            )
        if not np.all(np.isfinite(probe)):
            raise ValueError("candidate generator returned non-finite values.")
        loss = raw_complex_probe_loss(probe, self.target)
        index = len(self.probes)
        self.values[key] = index
        self.diameters_m.append(key)
        self.probes.append(probe.copy())
        self.losses.append(float(loss))
        return self.probes[index], float(loss), index


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"Missing registered {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected).upper():
        raise RuntimeError(f"Registered {label} hash mismatch.")
    return actual


def validate_exp053_config(config: Mapping[str, Any]) -> None:
    """Validate the formal-frozen exp053 reconstruction-aware contract."""

    experiment = config.get("experiment", {})
    if experiment.get("id") != "exp053":
        raise ValueError("exp053 config must declare experiment.id=exp053.")
    if experiment.get("role") != (
        "3d_scalar_multislice_reconstructed_probe_q8_cell_interval_fit"
    ):
        raise ValueError("Unexpected exp053 experiment role.")
    if (
        experiment.get("reference_validated") is not False
        or experiment.get("full_tgv_reference_authorized") is not False
    ):
        raise ValueError("exp053 provenance flags must remain false.")
    source = config.get("source", {})
    target_path = str(source.get("target_hdf5_path", ""))
    allowed_targets = {
        _LEGACY_MATCHED_Q4_TARGET_PATH,
        _EXP053_FEEDBACK_MATCHED_Q4_TARGET_PATH,
    }
    if target_path not in allowed_targets:
        raise ValueError("exp053 primary input must be matched_q4 raw P_B_rec.")
    lowered = target_path.lower()
    if any(
        str(token).lower() in lowered
        for token in source.get("forbidden_primary_input_tokens", [])
    ):
        raise ValueError("exp053 primary input selects a forbidden dataset.")
    if source.get("operator_reference_hdf5_path") != "/entry/truth/P_B_true":
        raise ValueError("exp053 operator reference identity changed.")
    identity = source.get("expected_identity", {})
    if (
        identity.get("plane") != "B"
        or identity.get("axis_order") != ["y", "x"]
        or identity.get("native_shape") != [96, 96]
        or identity.get("dtype") != "complex128"
        or float(identity.get("node_dx_m", 0.0)) != 5.0e-7
        or identity.get("detector_branch") != "matched_q4"
        or identity.get("detector_data_source") != "primary_q4_data"
        or identity.get("detector_data_readout")
        != "positive_pixel_average"
        or identity.get("detector_model_role") != "exp040_matched_reference"
    ):
        raise ValueError("exp053 target plane/grid/branch identity changed.")
    fixed = config.get("fixed_model", {})
    if (
        fixed.get("interface_factor") != 8
        or fixed.get("shape") != [96, 96]
        or fixed.get("complex_dtype") != "complex128"
        or fixed.get("internal_alias_control") is not False
        or fixed.get("external_alias_control") is not True
    ):
        raise ValueError("exp053 fixed scalar candidate identity changed.")
    fit = config.get("fit", {})
    bounds = np.asarray(fit.get("bounds_m", []), dtype=np.float64)
    coarse = np.asarray(fit.get("coarse_grid_m", []), dtype=np.float64)
    optimizer = fit.get("optimizer", {})
    closure = optimizer.get("numerical_closure_control", {})
    starts = np.asarray(optimizer.get("starts_m", []), dtype=np.float64)
    if (
        fit.get("primary_loss") != "raw_complex_normalized_squared_l2"
        or fit.get("mask") != "full_native_field_all_true"
        or fit.get("phase_or_scale_alignment") != "none"
        or bounds.shape != (2,)
        or not np.array_equal(bounds, np.asarray([1.6e-5, 2.4e-5]))
        or not np.allclose(
            coarse,
            np.arange(16.0, 25.0) * 1.0e-6,
            rtol=0.0,
            atol=1.0e-20,
        )
        or not np.allclose(
            starts,
            np.asarray([16.5, 18.5, 21.5, 23.5]) * 1.0e-6,
            rtol=0.0,
            atol=1.0e-20,
        )
        or int(optimizer.get("evaluation_budget_per_start", 0)) != 43
        or int(closure.get("baseline_evaluation_budget_per_start", 0)) != 41
        or int(
            closure.get("additional_complete_pattern_updates_per_start", 0)
        )
        != 1
        or int(closure.get("evaluations_per_pattern_update", 0)) != 2
        or int(closure.get("added_evaluations_per_start", 0)) != 2
        or closure.get("selection_basis")
        != "existing_raw_target_loss_and_41_call_tracks_only"
        or closure.get("equal_budget_all_starts") is not True
        or closure.get(
            "truth_used_for_method_parameter_or_stopping_selection"
        )
        is not False
    ):
        raise ValueError("exp053 frozen loss/bounds/grid/multi-start changed.")
    design = config.get("reconstruction_aware_interval", {})
    source_boundary = design.get("source_exact_boundary_control", {})
    equivalence = design.get("field_equivalence_threshold", {})
    thresholds = design.get("thresholds", {})
    expected_thresholds = {
        "candidate_operator_replay_relative_l2_max": 1.0e-14,
        "deterministic_repeat_relative_l2_max": 1.0e-14,
        "boundary_resolution_m_max": 1.0e-12,
        "interval_width_m_max": 1.25e-7,
        "truth_to_interval_distance_m_max_simulation_evaluation_only": (
            1.25e-7
        ),
        "endpoint_absolute_error_m_max_simulation_evaluation_only": 1.25e-7,
        "endpoint_relative_error_max_simulation_evaluation_only": 0.00625,
        "boundary_margin_m": 1.25e-7,
    }
    if (
        design.get("primary_definition")
        != "connected_best_candidate_q8_field_equivalence_component"
        or design.get("q8_interface_method") != "subpixel_midpoint_count"
        or int(design.get("q8_interface_factor", 0)) != 8
        or source_boundary.get("method")
        != "bounded_float64_nextafter_transition_scan"
        or int(source_boundary.get("max_ulps_each_direction", 0)) != 8
        or source_boundary.get("lower_boundary_definition")
        != "first_representable_member_after_outside"
        or source_boundary.get("upper_boundary_definition")
        != "first_representable_outside_after_member"
        or source_boundary.get("membership_contract")
        != "tau_primary_exact_field_and_node_count_all_agree"
        or source_boundary.get("primary_boundary_reference")
        != "source_exact_float64_membership_transition"
        or source_boundary.get("analytic_breakpoint_role")
        != "retained_algebraic_diagnostic_only"
        or source_boundary.get("truth_used_for_mapping_selection_or_stopping")
        is not False
        or float(equivalence.get("primary_relative_l2", 0.0)) != 1.0e-12
        or float(equivalence.get("low_factor", 0.0)) != 0.1
        or float(equivalence.get("high_factor", 0.0)) != 10.0
        or float(design.get("screened_loss_tie_absolute", -1.0)) != 1.0e-14
        or int(design.get("max_adjacent_cells_per_direction", 0)) != 256
        or int(design.get("boundary_bisection_iterations", 0)) != 24
        or set(thresholds) != set(expected_thresholds)
        or any(
            float(thresholds[key]) != value
            for key, value in expected_thresholds.items()
        )
    ):
        raise ValueError("exp053 reconstruction-aware frozen design changed.")
    oracle = config.get("oracle_comparison", {})
    expected_interval = oracle.get("expected_interval_m", {})
    if (
        oracle.get("required_experiment_status") != "Passed"
        or oracle.get("required_interpretation")
        != "q8_plateau_interval_single_parameter_oracle_fit_passed"
        or float(expected_interval.get("lower", 0.0))
        != 1.9999972701052885e-5
        or float(expected_interval.get("upper", 0.0))
        != 2.0000143340468626e-5
        or expected_interval.get("lower_closed") is not True
        or expected_interval.get("upper_closed") is not False
    ):
        raise ValueError("exp051 oracle comparison contract changed.")


def load_exp053_source_reconstructed_probe(
    config: Mapping[str, Any], project_root: Path
) -> Exp053SourceArtifact:
    """Load and strictly validate raw matched-q4 ``P_B_rec`` and provenance."""

    validate_exp053_config(config)
    source = config["source"]
    target_path = str(source["target_hdf5_path"])
    feedback_control_source = (
        target_path == _EXP053_FEEDBACK_MATCHED_Q4_TARGET_PATH
    )
    run_dir = Path(source["run"])
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    run_dir = run_dir.resolve()
    paths = {
        "config": run_dir / source["config_filename"],
        "metadata": run_dir / source["metadata_filename"],
        "metrics": run_dir / source["metrics_filename"],
        "run_state": run_dir / source["run_state_filename"],
        "hdf5": run_dir / source["hdf5_relative_path"],
    }
    file_sha = {
        key: _require_hash(path, source["expected_sha256"][key], f"exp042 {key}")
        for key, path in paths.items()
    }
    source_config = load_config(paths["config"])
    source_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    source_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    source_state = json.loads(paths["run_state"].read_text(encoding="utf-8"))
    identity = source["expected_identity"]
    provenance = source_config.get("provenance", {})
    grid = source_config.get("probe_grid", {})
    ablation = source_config.get("verification", {}).get(
        "detector_quadrature_ablation", {}
    )
    branch_specs = {
        branch["name"]: branch for branch in ablation.get("branches", [])
    }
    matched_spec = branch_specs.get("matched_q4", {})
    if (
        source_state.get("status") != "complete"
        or source_state.get("artifacts_validated") is not True
        or source_state.get("hdf5_sha256") != file_sha["hdf5"]
        or source_metadata.get("experiment_id") != identity["source_experiment"]
        or source_metadata.get("git_commit") != identity["source_git_commit"]
        or source_metadata.get("reference_validated") is not False
        or source_metadata.get("full_tgv_reference_authorized") is not False
        or source_metadata.get("truth_used_by_optimizer") is not False
        or provenance.get("source_branch") != identity["source_exp040_branch"]
        or provenance.get("development_case_id")
        != identity["development_case_id"]
        or provenance.get("reference_validated") is not False
        or provenance.get("full_tgv_reference_authorized") is not False
        or grid.get("plane") != identity["plane"]
        or grid.get("axis_order") != identity["axis_order"]
        or grid.get("native_shape") != identity["native_shape"]
        or float(grid.get("node_dx_m")) != float(identity["node_dx_m"])
        or grid.get("open_shape") != identity["open_shape"]
        or grid.get("residual_embedding") != identity["residual_embedding"]
        or source_config.get("optics", {}).get("angular_spectrum_bandlimit")
        is not identity["angular_spectrum_bandlimit"]
        or source_config.get("optics", {}).get("alias_control_external")
        is not identity["external_alias_control"]
        or ablation.get("reference_branch") != "matched_q4"
        or matched_spec.get("data_source") != identity["detector_data_source"]
        or matched_spec.get("data_readout")
        != identity["detector_data_readout"]
        or matched_spec.get("detector_readout")
        != identity["detector_data_readout"]
        or matched_spec.get("model_role") != identity["detector_model_role"]
        or ablation.get("truth_used_by_branch_selection") is not False
        or ablation.get("truth_used_by_stopping") is not False
    ):
        raise RuntimeError("Registered exp042 source state/provenance differs.")

    truth_path = str(source["operator_reference_hdf5_path"])
    with h5py.File(paths["hdf5"], "r") as h5:
        required = [
            target_path,
            truth_path,
            "/entry/truth/D_z_m",
            "/entry/truth/z_m",
            "/entry/truth/slice_widths_m",
            "/entry/truth/reference_validated",
            "/entry/truth/full_tgv_reference_authorized",
            "/entry/instrument/probe_grid/plane",
            "/entry/instrument/probe_grid/axis_order",
            "/entry/instrument/probe_grid/native_shape",
            "/entry/instrument/probe_grid/node_dx_m",
            "/entry/instrument/probe_grid/open_shape",
            "/entry/sample/sample_a/d_waist_m",
            "/entry/config_yaml",
        ]
        if feedback_control_source:
            feedback_branch = target_path.removesuffix("/P_B_rec")
            required.extend(
                [
                    "/entry/metadata/matched_q4_data_and_operator",
                    "/entry/metadata/raw_control_probe_truth_aligned",
                    (
                        "/entry/reconstruction/exp053_feedback_control/design/"
                        "same_exp040_source_b_scan_q4_data_operator_"
                        "initialization_seed"
                    ),
                    (
                        "/entry/metrics/truth_free_comparison/"
                        "same_I_stack_operator_B_scan"
                    ),
                    f"{feedback_branch}/truth_used_by_optimizer",
                    f"{feedback_branch}/truth_used_by_spectral_radius",
                    f"{feedback_branch}/truth_used_by_damping",
                    f"{feedback_branch}/truth_used_by_cg",
                    (
                        f"{feedback_branch}/simulation_evaluation_only/"
                        "P_B_rec_global_phase_aligned"
                    ),
                ]
            )
        else:
            required.extend(
                [
                    (
                        "/entry/reconstruction/detector_quadrature_ablation/"
                        "branches/matched_q4/operator_spec"
                    ),
                    (
                        "/entry/reconstruction/detector_quadrature_ablation/"
                        "branches/matched_q4/truth_used_by_optimizer"
                    ),
                    (
                        "/entry/reconstruction/detector_quadrature_ablation/"
                        "branches/matched_q4/simulation_evaluation_only/"
                        "P_B_rec_global_phase_aligned"
                    ),
                ]
            )
        missing = [path for path in required if path not in h5]
        if missing:
            raise RuntimeError(f"Registered exp042 source HDF5 is missing: {missing}")
        target = np.asarray(h5[target_path][...])
        operator_reference = np.asarray(h5[truth_path][...])
        d_z_m = np.asarray(h5["/entry/truth/D_z_m"][...], dtype=np.float64)
        z_m = np.asarray(h5["/entry/truth/z_m"][...], dtype=np.float64)
        widths = np.asarray(
            h5["/entry/truth/slice_widths_m"][...], dtype=np.float64
        )
        embedded = str(_decode(h5["/entry/config_yaml"][()]))
        hdf_plane = _decode(h5["/entry/instrument/probe_grid/plane"][()])
        hdf_axis = [
            _decode(value)
            for value in h5["/entry/instrument/probe_grid/axis_order"][...]
        ]
        hdf_shape = h5["/entry/instrument/probe_grid/native_shape"][...].tolist()
        hdf_dx = float(h5["/entry/instrument/probe_grid/node_dx_m"][()])
        hdf_open_shape = h5[
            "/entry/instrument/probe_grid/open_shape"
        ][...].tolist()
        hdf_flags = (
            bool(h5["/entry/truth/reference_validated"][()]),
            bool(h5["/entry/truth/full_tgv_reference_authorized"][()]),
        )
        hdf_waist = float(h5["/entry/sample/sample_a/d_waist_m"][()])
        if feedback_control_source:
            feedback_branch = target_path.removesuffix("/P_B_rec")
            operator_identity_ok = bool(
                h5["/entry/metadata/matched_q4_data_and_operator"][()]
            ) and not bool(
                h5["/entry/metadata/raw_control_probe_truth_aligned"][()]
            )
            operator_identity_ok = operator_identity_ok and bool(
                h5[
                    "/entry/reconstruction/exp053_feedback_control/design/"
                    "same_exp040_source_b_scan_q4_data_operator_"
                    "initialization_seed"
                ][()]
            )
            operator_identity_ok = operator_identity_ok and bool(
                h5[
                    "/entry/metrics/truth_free_comparison/"
                    "same_I_stack_operator_B_scan"
                ][()]
            )
            operator_identity_ok = operator_identity_ok and all(
                not bool(h5[f"{feedback_branch}/{name}"][()])
                for name in (
                    "truth_used_by_optimizer",
                    "truth_used_by_spectral_radius",
                    "truth_used_by_damping",
                    "truth_used_by_cg",
                )
            )
        else:
            operator_group = h5[
                "/entry/reconstruction/detector_quadrature_ablation/"
                "branches/matched_q4/operator_spec"
            ]
            operator_spec = {
                key: _decode(operator_group[key][()]) for key in operator_group
            }
            operator_identity_ok = operator_spec == {
                "data_readout": identity["detector_data_readout"],
                "data_source": identity["detector_data_source"],
                "detector_readout": identity["detector_data_readout"],
                "model_role": identity["detector_model_role"],
                "name": identity["detector_branch"],
            } and not bool(
                h5[
                    "/entry/reconstruction/detector_quadrature_ablation/"
                    "branches/matched_q4/truth_used_by_optimizer"
                ][()]
            )
    if (
        target.shape != tuple(identity["native_shape"])
        or target.dtype != np.complex128
        or operator_reference.shape != target.shape
        or operator_reference.dtype != np.complex128
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(operator_reference))
        or float(np.sum(np.abs(target) ** 2, dtype=np.float64)) <= 0.0
        or hdf_plane != identity["plane"]
        or hdf_axis != identity["axis_order"]
        or hdf_shape != identity["native_shape"]
        or hdf_dx != float(identity["node_dx_m"])
        or hdf_open_shape != identity["open_shape"]
        or hdf_flags != (False, False)
        or hdf_waist
        != float(config["fit"]["true_d_waist_m_simulation_evaluation_only"])
        or not operator_identity_ok
    ):
        raise RuntimeError("Registered raw P_B_rec plane/grid/operator differs.")
    target_sha = sha256_array_bytes(target)
    truth_sha = sha256_array_bytes(operator_reference)
    if (
        target_sha != str(source["expected_sha256"]["target_dataset_bytes"])
        or truth_sha
        != str(source["expected_sha256"]["operator_reference_dataset_bytes"])
    ):
        raise RuntimeError("Registered source probe dataset byte hash mismatch.")
    if feedback_control_source and (
        source_state.get("raw_control_dataset_path") != target_path
        or source_state.get("raw_control_dataset_sha256") != target_sha
        or source_metadata.get("matched_q4_data_and_operator") is not True
        or source_metadata.get("raw_control_probe_truth_aligned") is not False
    ):
        raise RuntimeError("Registered feedback-control raw probe identity differs.")
    external_config = paths["config"].read_text(encoding="utf-8")
    if embedded.replace("\r\n", "\n") != external_config.replace("\r\n", "\n"):
        raise RuntimeError("Source HDF5 config_yaml differs from config.yaml.")
    raw_mismatch = float(
        np.sqrt(raw_complex_probe_loss(target, operator_reference))
    )
    if feedback_control_source:
        reported_mismatch = float(
            source_metrics["branches"]["spectrally_damped_gn_cg"]
            ["simulation_evaluation_only"]["probe_error"]
            ["raw_complex_relative_l2"]
        )
    else:
        reported_mismatch = float(
            source_metrics["detector_quadrature_ablation"]["branches"]
            ["matched_q4"]["simulation_evaluation_only"]
            ["final_probe_raw_relative_l2"]
        )
    if not np.isclose(
        raw_mismatch,
        reported_mismatch,
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    ):
        raise RuntimeError("Source raw reconstruction mismatch metric differs.")
    return Exp053SourceArtifact(
        run_dir=run_dir,
        config_path=paths["config"],
        metadata_path=paths["metadata"],
        metrics_path=paths["metrics"],
        run_state_path=paths["run_state"],
        hdf5_path=paths["hdf5"],
        target_hdf5_path=target_path,
        operator_reference_hdf5_path=truth_path,
        source_config=source_config,
        source_metadata=source_metadata,
        source_metrics=source_metrics,
        source_state=source_state,
        file_sha256=file_sha,
        target_dataset_sha256=target_sha,
        operator_reference_dataset_sha256=truth_sha,
        P_B_rec_raw=np.asarray(target, dtype=np.complex128),
        P_B_true_operator_reference=np.asarray(
            operator_reference, dtype=np.complex128
        ),
        D_z_m=d_z_m,
        z_m=z_m,
        slice_widths_m=widths,
        embedded_config_yaml=embedded,
    )


def load_exp051_oracle_interval(
    config: Mapping[str, Any], project_root: Path
) -> Exp051OracleIntervalArtifact:
    """Load and hash-check the authoritative exp051 oracle interval."""

    validate_exp053_config(config)
    oracle = config["oracle_comparison"]
    run_dir = Path(oracle["run"])
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    run_dir = run_dir.resolve()
    paths = {
        "config": run_dir / "config.yaml",
        "metadata": run_dir / "metadata.json",
        "metrics": run_dir / "metrics.json",
        "run_state": run_dir / "run_state.json",
        "hdf5": run_dir / oracle["hdf5_relative_path"],
    }
    file_sha = {
        key: _require_hash(path, oracle["expected_sha256"][key], f"oracle {key}")
        for key, path in paths.items()
    }
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    state = json.loads(paths["run_state"].read_text(encoding="utf-8"))
    interval = metrics.get("reported_interval", {})
    expected = oracle["expected_interval_m"]
    if (
        state.get("status") != "complete"
        or state.get("artifacts_validated") is not True
        or state.get("experiment_status") != oracle["required_experiment_status"]
        or metrics.get("experiment_status") != oracle["required_experiment_status"]
        or metrics.get("interpretation") != oracle["required_interpretation"]
        or metrics.get("reference_validated") is not False
        or metrics.get("full_tgv_reference_authorized") is not False
        or metrics.get("p_b_rec_used_as_primary_input") is not False
        or metadata.get("reference_validated") is not False
        or metadata.get("full_tgv_reference_authorized") is not False
        or float(interval.get("lower_m", 0.0)) != float(expected["lower"])
        or float(interval.get("upper_m", 0.0)) != float(expected["upper"])
        or interval.get("lower_closed") is not expected["lower_closed"]
        or interval.get("upper_closed") is not expected["upper_closed"]
    ):
        raise RuntimeError("Registered exp051 oracle state/interval differs.")
    with h5py.File(paths["hdf5"], "r") as h5:
        root = h5["/entry/metrics/reported_interval"]
        hdf_interval = {
            "lower_m": float(root["lower_m"][()]),
            "upper_m": float(root["upper_m"][()]),
            "lower_closed": bool(root["lower_closed"][()]),
            "upper_closed": bool(root["upper_closed"][()]),
        }
    if any(hdf_interval[key] != interval[key] for key in hdf_interval):
        raise RuntimeError("Oracle JSON/HDF5 interval identity differs.")
    return Exp051OracleIntervalArtifact(
        run_dir=run_dir,
        hdf5_path=paths["hdf5"],
        file_sha256=file_sha,
        metadata=metadata,
        metrics=metrics,
        run_state=state,
        lower_m=float(interval["lower_m"]),
        upper_m=float(interval["upper_m"]),
        lower_closed=bool(interval["lower_closed"]),
        upper_closed=bool(interval["upper_closed"]),
    )


def make_exp053_candidate_generator(
    source_config: Mapping[str, Any],
    *,
    memory_callback: Callable[[], None] | None = None,
) -> ComplexGenerator:
    """Return the unchanged default-q8 exp051 candidate generator."""

    return make_exp051_candidate_generator(
        source_config, memory_callback=memory_callback
    )


def _build_fine_grid(
    coarse_m: NDArray[np.float64],
    coarse_loss: NDArray[np.float64],
    *,
    bounds_m: tuple[float, float],
    half_width_m: float,
    step_m: float,
) -> NDArray[np.float64]:
    center = float(coarse_m[int(np.argmin(coarse_loss))])
    lower = max(bounds_m[0], center - half_width_m)
    upper = min(bounds_m[1], center + half_width_m)
    count = int(np.rint((upper - lower) / step_m))
    if not np.isclose(
        lower + count * step_m,
        upper,
        rtol=0.0,
        atol=32.0 * np.finfo(np.float64).eps,
    ):
        raise RuntimeError("registered fine interval is not divisible by step.")
    return lower + np.arange(count + 1, dtype=np.float64) * step_m


def _node_count_stack(
    source_config: Mapping[str, Any], diameter_m: float, q: int
) -> NDArray[np.uint8]:
    fraction, _, _ = make_air_fraction_stack(
        source_config,
        float(diameter_m),
        builder=make_tgv_air_fraction_slice,
        resolution=q,
    )
    return np.rint(fraction * q**2).astype(np.uint8)


def _source_exact_boundary_scan(
    analytic_breakpoint_m: float,
    *,
    side: str,
    max_ulps_each_direction: int,
    membership: Callable[[float], tuple[bool, bool, bool, float, int]],
) -> dict[str, Any]:
    """Locate one source-exact float64 membership transition near an edge."""

    if side not in {"lower", "upper"}:
        raise ValueError("source-exact boundary side must be lower or upper.")
    if max_ulps_each_direction <= 0:
        raise ValueError("source-exact boundary ULP budget must be positive.")
    analytic = float(np.float64(analytic_breakpoint_m))
    if not np.isfinite(analytic):
        raise ValueError("analytic breakpoint must be finite.")

    below: list[float] = []
    value = analytic
    for _ in range(max_ulps_each_direction):
        value = float(np.nextafter(value, -np.inf))
        below.append(value)
    above: list[float] = []
    value = analytic
    for _ in range(max_ulps_each_direction):
        value = float(np.nextafter(value, np.inf))
        above.append(value)
    values = np.asarray(
        [*reversed(below), analytic, *above], dtype=np.float64
    )
    ulp_offset = np.arange(
        -max_ulps_each_direction,
        max_ulps_each_direction + 1,
        dtype=np.int64,
    )

    field_member: list[bool] = []
    exact_member: list[bool] = []
    node_member: list[bool] = []
    field_relative_l2: list[float] = []
    cache_index: list[int] = []
    for diameter in values:
        field, exact, node, relative, index = membership(float(diameter))
        field_member.append(bool(field))
        exact_member.append(bool(exact))
        node_member.append(bool(node))
        field_relative_l2.append(float(relative))
        cache_index.append(int(index))

    field_values = np.asarray(field_member, dtype=np.bool_)
    exact_values = np.asarray(exact_member, dtype=np.bool_)
    node_values = np.asarray(node_member, dtype=np.bool_)
    agreement = bool(
        np.array_equal(field_values, exact_values)
        and np.array_equal(field_values, node_values)
    )
    transitions = np.flatnonzero(field_values[1:] != field_values[:-1])
    unique_transition = bool(len(transitions) == 1)
    transition_index = int(transitions[0] + 1) if unique_transition else -1
    if side == "lower":
        expected_endpoints = bool(not field_values[0] and field_values[-1])
        expected_transition = bool(
            unique_transition
            and not field_values[transition_index - 1]
            and field_values[transition_index]
        )
    else:
        expected_endpoints = bool(field_values[0] and not field_values[-1])
        expected_transition = bool(
            unique_transition
            and field_values[transition_index - 1]
            and not field_values[transition_index]
        )
    monotone = bool(
        expected_transition
        and np.all(field_values[:transition_index] == field_values[0])
        and np.all(field_values[transition_index:] == field_values[-1])
    )
    cap_hit = bool(
        not expected_endpoints
        or transition_index <= 0
        or transition_index >= len(values)
    )
    passed = bool(agreement and monotone and not cap_hit)
    source_exact = float(values[transition_index]) if passed else analytic
    source_offset = int(ulp_offset[transition_index]) if passed else 0
    previous = (
        float(values[transition_index - 1]) if passed else analytic
    )
    following = (
        float(values[min(transition_index + 1, len(values) - 1)])
        if passed
        else analytic
    )
    return {
        "method": "bounded_float64_nextafter_transition_scan",
        "side": side,
        "analytic_breakpoint_m": analytic,
        "max_ulps_each_direction": int(max_ulps_each_direction),
        "ulp_offset": ulp_offset,
        "diameter_m": values,
        "tau_primary_member": field_values,
        "exact_best_field_member": exact_values,
        "node_count_equal_to_best": node_values,
        "field_relative_l2": np.asarray(
            field_relative_l2, dtype=np.float64
        ),
        "cache_index": np.asarray(cache_index, dtype=np.int64),
        "membership_triplet_agreement": agreement,
        "unique_monotone_transition": monotone,
        "scan_cap_hit": cap_hit,
        "transition_index": transition_index,
        "source_exact_breakpoint_m": source_exact,
        "source_exact_ulp_offset_from_analytic": source_offset,
        "source_exact_previous_m": previous,
        "source_exact_next_m": following,
        "pass": passed,
    }


def _component_signature(component: Mapping[str, Any]) -> tuple[int, int, bool, bool]:
    return (
        int(component["left_cell_index"]),
        int(component["right_cell_index"]),
        bool(component["lower_closed"]),
        bool(component["upper_closed"]),
    )


def _field_norm(values: NDArray[np.complex128]) -> float:
    return float(np.sqrt(np.sum(np.abs(values) ** 2, dtype=np.float64)))


def _direction_metrics(
    error: NDArray[np.complex128], direction: NDArray[np.complex128]
) -> dict[str, Any]:
    error_norm = _field_norm(error)
    direction_norm = _field_norm(direction)
    denominator = error_norm * direction_norm
    if denominator <= 0.0:
        return {
            "valid": False,
            "direction_norm": direction_norm,
            "real_cosine": 0.0,
            "complex_coherence": 0.0,
            "real_projection_coefficient": 0.0,
        }
    inner = np.sum(np.conjugate(direction) * error, dtype=np.complex128)
    return {
        "valid": True,
        "direction_norm": direction_norm,
        "real_cosine": float(np.real(inner) / denominator),
        "complex_coherence": float(np.abs(inner) / denominator),
        "real_projection_coefficient": float(
            np.real(inner) / (direction_norm**2)
        ),
    }


def _interval_comparison(
    lower_m: float,
    upper_m: float,
    oracle: Exp051OracleIntervalArtifact,
) -> dict[str, Any]:
    overlap = bool(max(lower_m, oracle.lower_m) < min(upper_m, oracle.upper_m))
    distance = float(
        max(0.0, oracle.lower_m - upper_m, lower_m - oracle.upper_m)
    )
    lower_displacement = float(lower_m - oracle.lower_m)
    upper_displacement = float(upper_m - oracle.upper_m)
    return {
        "oracle_lower_m": oracle.lower_m,
        "oracle_upper_m": oracle.upper_m,
        "oracle_lower_closed": oracle.lower_closed,
        "oracle_upper_closed": oracle.upper_closed,
        "intervals_overlap": overlap,
        "interval_distance_m": distance,
        "midpoint_displacement_m": float(
            0.5 * (lower_m + upper_m)
            - 0.5 * (oracle.lower_m + oracle.upper_m)
        ),
        "lower_endpoint_displacement_m": lower_displacement,
        "upper_endpoint_displacement_m": upper_displacement,
        "endpoint_hausdorff_displacement_m": float(
            max(abs(lower_displacement), abs(upper_displacement))
        ),
    }


def run_reconstructed_probe_q8_interval_fit(
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    reconstructed_target: NDArray[np.complex128],
    operator_reference_probe: NDArray[np.complex128],
    oracle: Exp051OracleIntervalArtifact,
    candidate_generator: ComplexGenerator,
) -> dict[str, Any]:
    """Run the preregistered reconstruction-aware q8 cell-interval fit."""

    validate_exp053_config(config)
    target = np.asarray(reconstructed_target)
    operator_reference = np.asarray(operator_reference_probe)
    if (
        target.ndim != 2
        or target.dtype != np.complex128
        or operator_reference.shape != target.shape
        or operator_reference.dtype != np.complex128
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(operator_reference))
        or _field_norm(target) <= 0.0
    ):
        raise ValueError("exp053 target/operator reference identity differs.")
    fit = config["fit"]
    design = config["reconstruction_aware_interval"]
    thresholds = design["thresholds"]
    true_waist = float(fit["true_d_waist_m_simulation_evaluation_only"])
    bounds = tuple(float(value) for value in fit["bounds_m"])
    q = int(design["q8_interface_factor"])
    tau = float(design["field_equivalence_threshold"]["primary_relative_l2"])
    tau_values = {
        "tau_low": tau
        * float(design["field_equivalence_threshold"]["low_factor"]),
        "tau_primary": tau,
        "tau_high": tau
        * float(design["field_equivalence_threshold"]["high_factor"]),
    }

    replay = np.asarray(candidate_generator(true_waist))
    repeat = np.asarray(candidate_generator(true_waist))
    if replay.shape != target.shape or replay.dtype != np.complex128:
        raise ValueError("candidate replay identity differs from target grid.")
    operator_replay = replay_error_metrics(replay, operator_reference)
    repeat_relative = float(np.sqrt(raw_complex_probe_loss(repeat, replay)))
    operator_replay["deterministic_repeat_relative_l2"] = repeat_relative
    mismatch_to_target_norm = replay_error_metrics(replay, target)
    mismatch_to_truth_norm = float(
        np.sqrt(raw_complex_probe_loss(target, operator_reference))
    )
    reconstructed_mismatch = {
        "raw_complex_relative_l2": mismatch_to_truth_norm,
        "raw_complex_relative_l2_truth_denominator": mismatch_to_truth_norm,
        "raw_complex_relative_l2_reconstructed_target_denominator": (
            mismatch_to_target_norm["raw_complex_relative_l2"]
        ),
        "amplitude_relative_l2_reconstructed_target_denominator": (
            mismatch_to_target_norm["amplitude_relative_l2"]
        ),
        (
            "amplitude_weighted_phase_sensitive_relative_l2_"
            "reconstructed_target_denominator"
        ): mismatch_to_target_norm[
            "amplitude_weighted_phase_sensitive_relative_l2"
        ],
    }
    operator_gate = bool(
        operator_replay["raw_complex_relative_l2"]
        <= float(thresholds["candidate_operator_replay_relative_l2_max"])
        and repeat_relative
        <= float(thresholds["deterministic_repeat_relative_l2_max"])
    )
    cache = _ProbeCache(candidate_generator, target, bounds)
    cache.values[float(np.float64(true_waist))] = 0
    cache.diameters_m.append(true_waist)
    cache.probes.append(replay.copy())
    cache.losses.append(raw_complex_probe_loss(replay, target))
    if not operator_gate:
        return {
            "experiment_status": "Inconclusive",
            "interpretation": "artifact_operator_handoff_not_closed",
            "reference_validated": False,
            "full_tgv_reference_authorized": False,
            "primary_input_is_raw_matched_q4_p_b_rec": True,
            "truth_used_by_fitter": False,
            "candidate_operator_replay": operator_replay,
            "reconstructed_target_mismatch": reconstructed_mismatch,
            "field_equivalence_threshold": tau_values,
            "gates": {"candidate_operator_replay_pass": False},
            "cache": {
                "D_waist_m": np.asarray(cache.diameters_m),
                "loss": np.asarray(cache.losses),
                "P_B_candidate": np.stack(cache.probes),
            },
        }

    def evaluate_loss(diameter_m: float) -> tuple[float, int]:
        _, loss, cache_index = cache.get(diameter_m)
        return loss, cache_index

    partition = build_q8_cell_partition(
        source_config,
        bounds_m=bounds,
        reference_diameter_m=true_waist,
        q=q,
    )
    edges = np.asarray(partition["cell_edges_m"], dtype=np.float64)
    cell_count = len(edges) - 1
    breakpoint_pass = bool(
        np.all(np.isfinite(edges)) and np.all(np.diff(edges) > 0.0)
    )

    coarse = np.asarray(fit["coarse_grid_m"], dtype=np.float64)
    coarse_loss: list[float] = []
    coarse_cache: list[int] = []
    for diameter in coarse:
        loss, cache_index = evaluate_loss(float(diameter))
        coarse_loss.append(loss)
        coarse_cache.append(cache_index)
    coarse_losses = np.asarray(coarse_loss, dtype=np.float64)
    fine = _build_fine_grid(
        coarse,
        coarse_losses,
        bounds_m=bounds,
        half_width_m=float(fit["fine_half_width_m"]),
        step_m=float(fit["fine_step_m"]),
    )
    fine_loss: list[float] = []
    fine_cache: list[int] = []
    for diameter in fine:
        loss, cache_index = evaluate_loss(float(diameter))
        fine_loss.append(loss)
        fine_cache.append(cache_index)
    fine_losses = np.asarray(fine_loss, dtype=np.float64)

    optimizer = fit["optimizer"]
    branches: dict[str, dict[str, Any]] = {}
    for branch_index, start in enumerate(optimizer["starts_m"]):
        branch = equal_budget_bounded_pattern_search(
            evaluate_loss,
            bounds_m=bounds,
            start_m=float(start),
            initial_step_m=float(optimizer["initial_step_m"]),
            evaluation_budget=int(optimizer["evaluation_budget_per_start"]),
        )
        branch["final_raw_relative_l2"] = float(np.sqrt(branch["final_loss"]))
        branch["final_cell_index"] = q8_cell_index(
            float(branch["final_estimate_m"]), edges
        )
        branches[f"start_{branch_index:02d}"] = branch
    equal_budget = bool(
        all(
            int(branch["evaluation_count"])
            == int(optimizer["evaluation_budget_per_start"])
            and branch["stopping_reason"] == "evaluation_budget"
            for branch in branches.values()
        )
    )

    screen_d = np.concatenate(
        [
            coarse,
            fine,
            *[
                np.asarray(branch["evaluated_d_waist_m"], dtype=np.float64)
                for branch in branches.values()
            ],
        ]
    )
    screen_loss = np.concatenate(
        [
            coarse_losses,
            fine_losses,
            *[
                np.asarray(branch["evaluated_loss"], dtype=np.float64)
                for branch in branches.values()
            ],
        ]
    )
    screen_cache = np.concatenate(
        [
            np.asarray(coarse_cache, dtype=np.int64),
            np.asarray(fine_cache, dtype=np.int64),
            *[
                np.asarray(branch["evaluated_cache_index"], dtype=np.int64)
                for branch in branches.values()
            ],
        ]
    )
    unique_support_cache = np.unique(screen_cache)
    best_cache_before_interval = min(
        (int(index) for index in unique_support_cache),
        key=lambda index: (
            cache.losses[index],
            q8_cell_index(cache.diameters_m[index], edges),
            cache.diameters_m[index],
        ),
    )
    best_cell = q8_cell_index(
        cache.diameters_m[best_cache_before_interval], edges
    )
    best_midpoint = q8_cell_midpoint(best_cell, edges)
    best_probe, best_loss, best_cache_index = cache.get(best_midpoint)
    selected_probe = cache.probes[best_cache_before_interval]
    best_cell_consistency = bool(
        np.array_equal(best_probe, selected_probe)
        and best_loss == cache.losses[best_cache_before_interval]
    )

    node_cache: dict[int, NDArray[np.uint8]] = {}

    def nodes_for_cell(cell_index: int) -> NDArray[np.uint8]:
        nodes = node_cache.get(cell_index)
        if nodes is None:
            nodes = _node_count_stack(
                source_config, q8_cell_midpoint(cell_index, edges), q
            )
            node_cache[cell_index] = nodes
        return nodes

    best_nodes = nodes_for_cell(best_cell)
    component_queries: dict[str, dict[int, tuple[float, float, int, bool]]] = {}

    def build_component(name: str, seed_cell: int) -> dict[str, Any]:
        query_details: dict[int, tuple[float, float, int, bool]] = {}

        def member(cell_index: int) -> bool:
            diameter = q8_cell_midpoint(cell_index, edges)
            probe, _, cache_index = cache.get(diameter)
            if name == "geometry":
                is_member = bool(np.array_equal(nodes_for_cell(cell_index), best_nodes))
                relative = 0.0 if is_member else 1.0
            elif name == "exact_best_field":
                is_member = bool(np.array_equal(probe, best_probe))
                relative = 0.0 if is_member else float(
                    np.sqrt(raw_complex_probe_loss(probe, best_probe))
                )
            else:
                relative = float(
                    np.sqrt(raw_complex_probe_loss(probe, best_probe))
                )
                is_member = bool(relative <= tau_values[name])
            query_details[cell_index] = (
                diameter,
                relative,
                int(cache_index),
                is_member,
            )
            return is_member

        component = expand_connected_cell_component(
            member,
            cell_count=cell_count,
            seed_cell=seed_cell,
            max_cells_per_direction=int(
                design["max_adjacent_cells_per_direction"]
            ),
        )
        if component["seed_member"]:
            left_cell = int(component["left_cell_index"])
            right_cell = int(component["right_cell_index"])
            lower_boundary = float(edges[left_cell])
            upper_boundary = float(edges[right_cell + 1])

            def boundary_member(value: float) -> bool:
                boundary_cell = q8_cell_index(value, edges)
                if name == "geometry":
                    return bool(
                        np.array_equal(nodes_for_cell(boundary_cell), best_nodes)
                    )
                probe, _, _ = cache.get(value)
                if name == "exact_best_field":
                    return bool(np.array_equal(probe, best_probe))
                return bool(
                    np.sqrt(raw_complex_probe_loss(probe, best_probe))
                    <= tau_values[name]
                )

            component.update(
                {
                    "lower_boundary_m": lower_boundary,
                    "upper_boundary_m": upper_boundary,
                    "lower_closed": True,
                    "upper_closed": False,
                    "actual_lower_breakpoint_member": boundary_member(
                        lower_boundary
                    ),
                    "actual_upper_breakpoint_member": boundary_member(
                        upper_boundary
                    ),
                }
            )
        else:
            component.update(
                {
                    "lower_boundary_m": 0.0,
                    "upper_boundary_m": 0.0,
                    "lower_closed": False,
                    "upper_closed": False,
                    "actual_lower_breakpoint_member": False,
                    "actual_upper_breakpoint_member": False,
                }
            )
        ordered = [
            query_details[int(index)]
            for index in component["queried_cell_index"]
        ]
        component["queried_d_waist_m"] = np.asarray(
            [item[0] for item in ordered], dtype=np.float64
        )
        component["queried_best_field_relative_l2"] = np.asarray(
            [item[1] for item in ordered], dtype=np.float64
        )
        component["queried_cache_index"] = np.asarray(
            [item[2] for item in ordered], dtype=np.int64
        )
        component_queries[name] = query_details
        return component

    best_components = {
        name: build_component(name, best_cell)
        for name in (
            "geometry",
            "exact_best_field",
            "tau_low",
            "tau_primary",
            "tau_high",
        )
    }
    primary_component = best_components["tau_primary"]
    primary_signature = _component_signature(primary_component)
    component_signatures = {
        name: _component_signature(component)
        for name, component in best_components.items()
    }
    threshold_stability = bool(
        all(
            signature == primary_signature
            for signature in component_signatures.values()
        )
    )
    cap_hit = bool(
        any(
            component[side]
            for component in best_components.values()
            for side in ("left_cap_hit", "right_cap_hit")
        )
    )

    analytic_common_left = float(primary_component["lower_boundary_m"])
    analytic_common_right = float(primary_component["upper_boundary_m"])
    boundary_node_cache: dict[float, NDArray[np.uint8]] = {}

    def boundary_membership(
        diameter_m: float,
    ) -> tuple[bool, bool, bool, float, int]:
        probe, _, cache_index = cache.get(diameter_m)
        relative = float(
            np.sqrt(raw_complex_probe_loss(probe, best_probe))
        )
        key = float(np.float64(diameter_m))
        nodes = boundary_node_cache.get(key)
        if nodes is None:
            nodes = _node_count_stack(source_config, key, q)
            boundary_node_cache[key] = nodes
        return (
            bool(relative <= tau),
            bool(np.array_equal(probe, best_probe)),
            bool(np.array_equal(nodes, best_nodes)),
            relative,
            int(cache_index),
        )

    source_boundary_spec = design["source_exact_boundary_control"]
    max_boundary_ulps = int(source_boundary_spec["max_ulps_each_direction"])
    source_exact_boundary_control = {
        "method": source_boundary_spec["method"],
        "max_ulps_each_direction": max_boundary_ulps,
        "membership_contract": source_boundary_spec["membership_contract"],
        "primary_boundary_reference": source_boundary_spec[
            "primary_boundary_reference"
        ],
        "analytic_breakpoint_role": source_boundary_spec[
            "analytic_breakpoint_role"
        ],
        "truth_used_for_mapping_selection_or_stopping": False,
        "boundaries": {
            "lower": _source_exact_boundary_scan(
                analytic_common_left,
                side="lower",
                max_ulps_each_direction=max_boundary_ulps,
                membership=boundary_membership,
            ),
            "upper": _source_exact_boundary_scan(
                analytic_common_right,
                side="upper",
                max_ulps_each_direction=max_boundary_ulps,
                membership=boundary_membership,
            ),
        },
    }
    source_boundary_pass = bool(
        all(
            control["pass"]
            for control in source_exact_boundary_control["boundaries"].values()
        )
    )
    source_exact_boundary_control["pass"] = source_boundary_pass
    common_left = (
        float(
            source_exact_boundary_control["boundaries"]["lower"][
                "source_exact_breakpoint_m"
            ]
        )
        if source_boundary_pass
        else analytic_common_left
    )
    common_right = (
        float(
            source_exact_boundary_control["boundaries"]["upper"][
                "source_exact_breakpoint_m"
            ]
        )
        if source_boundary_pass
        else analytic_common_right
    )

    interval_by_branch: dict[str, dict[str, Any]] = {}
    for name, branch in branches.items():
        final_probe, _, _ = cache.get(float(branch["final_estimate_m"]))
        final_field_relative = float(
            np.sqrt(raw_complex_probe_loss(final_probe, best_probe))
        )
        branch_component = build_component(
            "tau_primary", int(branch["final_cell_index"])
        )
        final_qualifies = bool(final_field_relative <= tau)
        branch["final_best_field_relative_l2"] = final_field_relative
        branch["final_seed_qualifies"] = final_qualifies
        interval_by_branch[name] = {
            "seed_m": float(branch["final_estimate_m"]),
            "seed_loss": float(branch["final_loss"]),
            "seed_cell_index": int(branch["final_cell_index"]),
            "final_best_field_relative_l2": final_field_relative,
            "final_seed_qualifies": final_qualifies,
            "analytic_lower_boundary_m": float(
                branch_component["lower_boundary_m"]
            ),
            "analytic_upper_boundary_m": float(
                branch_component["upper_boundary_m"]
            ),
            "source_exact_lower_boundary_m": common_left,
            "source_exact_upper_boundary_m": common_right,
            "primary_component": branch_component,
        }
    all_seeds_qualify = bool(
        all(branch["final_seed_qualifies"] for branch in branches.values())
    )
    interval_agreement = bool(
        all(
            item["final_seed_qualifies"]
            and _component_signature(item["primary_component"])
            == primary_signature
            for item in interval_by_branch.values()
        )
    )
    cap_hit = bool(
        cap_hit
        or any(
            item["primary_component"][side]
            for item in interval_by_branch.values()
            for side in ("left_cap_hit", "right_cap_hit")
        )
    )

    left_cell = int(primary_component["left_cell_index"])
    right_cell = int(primary_component["right_cell_index"])
    left_outside_cell = int(primary_component["left_outside_cell_index"])
    right_outside_cell = int(primary_component["right_outside_cell_index"])
    adjacent_available = bool(left_outside_cell >= 0 and right_outside_cell >= 0)
    left_outside_midpoint = (
        q8_cell_midpoint(left_outside_cell, edges)
        if left_outside_cell >= 0
        else bounds[0]
    )
    right_outside_midpoint = (
        q8_cell_midpoint(right_outside_cell, edges)
        if right_outside_cell >= 0
        else bounds[1]
    )
    endpoint_diameters = {
        "best_midpoint": best_midpoint,
        "left_outside_midpoint": left_outside_midpoint,
        "right_outside_midpoint": right_outside_midpoint,
        "analytic_lower_breakpoint": analytic_common_left,
        "analytic_upper_breakpoint": analytic_common_right,
        "lower_breakpoint": common_left,
        "lower_nextafter_minus": float(np.nextafter(common_left, -np.inf)),
        "lower_nextafter_plus": float(np.nextafter(common_left, np.inf)),
        "upper_breakpoint": common_right,
        "upper_nextafter_minus": float(np.nextafter(common_right, -np.inf)),
        "upper_nextafter_plus": float(np.nextafter(common_right, np.inf)),
    }
    endpoint_names: list[str] = []
    endpoint_values: list[float] = []
    endpoint_target_loss: list[float] = []
    endpoint_best_relative: list[float] = []
    endpoint_exact: list[bool] = []
    endpoint_member: list[bool] = []
    endpoint_node_equal: list[bool] = []
    endpoint_cache_indices: list[int] = []
    endpoint_node_stacks: dict[str, NDArray[np.uint8]] = {}
    for name, diameter in endpoint_diameters.items():
        probe, target_loss, cache_index = cache.get(diameter)
        field_relative = float(
            np.sqrt(raw_complex_probe_loss(probe, best_probe))
        )
        nodes = _node_count_stack(source_config, diameter, q)
        endpoint_names.append(name)
        endpoint_values.append(diameter)
        endpoint_target_loss.append(target_loss)
        endpoint_best_relative.append(field_relative)
        endpoint_exact.append(bool(np.array_equal(probe, best_probe)))
        endpoint_member.append(bool(field_relative <= tau))
        endpoint_node_equal.append(bool(np.array_equal(nodes, best_nodes)))
        endpoint_cache_indices.append(cache_index)
        endpoint_node_stacks[name] = nodes
    endpoint_control = {
        "boundary_reference": "source_exact_float64_membership_transition",
        "name": endpoint_names,
        "diameter_m": np.asarray(endpoint_values, dtype=np.float64),
        "reconstructed_target_loss": np.asarray(
            endpoint_target_loss, dtype=np.float64
        ),
        "best_field_relative_l2": np.asarray(
            endpoint_best_relative, dtype=np.float64
        ),
        "exact_best_field_member": np.asarray(endpoint_exact, dtype=np.bool_),
        "tau_primary_member": np.asarray(endpoint_member, dtype=np.bool_),
        "node_count_equal_to_best": np.asarray(
            endpoint_node_equal, dtype=np.bool_
        ),
        "cache_index": np.asarray(endpoint_cache_indices, dtype=np.int64),
        "node_count_stack": endpoint_node_stacks,
    }
    endpoint_field_map = dict(zip(endpoint_names, endpoint_member, strict=True))
    endpoint_exact_map = dict(zip(endpoint_names, endpoint_exact, strict=True))
    endpoint_node_map = dict(
        zip(endpoint_names, endpoint_node_equal, strict=True)
    )
    endpoint_convention_pass = bool(
        source_boundary_pass
        and primary_component["lower_closed"]
        and not primary_component["upper_closed"]
        and all(
            endpoint_field_map[name]
            == endpoint_exact_map[name]
            == endpoint_node_map[name]
            for name in endpoint_names
        )
        and endpoint_field_map["lower_breakpoint"]
        and not endpoint_field_map["lower_nextafter_minus"]
        and endpoint_field_map["lower_nextafter_plus"]
        and endpoint_field_map["upper_nextafter_minus"]
        and not endpoint_field_map["upper_breakpoint"]
        and not endpoint_field_map["upper_nextafter_plus"]
    )

    def best_field_objective(diameter_m: float) -> tuple[float, int]:
        probe, _, cache_index = cache.get(diameter_m)
        return raw_complex_probe_loss(probe, best_probe), cache_index

    bisection: dict[str, Any] = {}
    bisection_passes: list[bool] = []
    iterations = int(design["boundary_bisection_iterations"])
    if adjacent_available and all_seeds_qualify:
        for name, item in interval_by_branch.items():
            component = item["primary_component"]
            controls: dict[str, Any] = {}
            for side, inside_cell, outside_cell, analytic in (
                (
                    "lower",
                    int(component["left_cell_index"]),
                    int(component["left_outside_cell_index"]),
                    float(component["lower_boundary_m"]),
                ),
                (
                    "upper",
                    int(component["right_cell_index"]),
                    int(component["right_outside_cell_index"]),
                    float(component["upper_boundary_m"]),
                ),
            ):
                control = fixed_membership_bisection(
                    best_field_objective,
                    inside_m=q8_cell_midpoint(inside_cell, edges),
                    outside_m=q8_cell_midpoint(outside_cell, edges),
                    tau_relative_l2=tau,
                    iterations=iterations,
                )
                boundary_mapping = source_exact_boundary_control[
                    "boundaries"
                ][side]
                source_exact = float(
                    boundary_mapping["source_exact_breakpoint_m"]
                )
                control["analytic_breakpoint_m"] = analytic
                control["analytic_breakpoint_in_bracket"] = bool(
                    control["bracket_lower_m"]
                    <= analytic
                    <= control["bracket_upper_m"]
                )
                control["source_exact_breakpoint_m"] = source_exact
                control["source_exact_ulp_offset_from_analytic"] = int(
                    boundary_mapping[
                        "source_exact_ulp_offset_from_analytic"
                    ]
                )
                control["source_exact_breakpoint_in_bracket"] = bool(
                    control["bracket_lower_m"]
                    <= source_exact
                    <= control["bracket_upper_m"]
                )
                control["source_exact_mapping_pass"] = bool(
                    boundary_mapping["pass"]
                )
                control["pass"] = bool(
                    control["iterations"] == iterations
                    and control["bracket_width_m"]
                    <= float(thresholds["boundary_resolution_m_max"])
                    and control["source_exact_mapping_pass"]
                    and control["source_exact_breakpoint_in_bracket"]
                )
                bisection_passes.append(bool(control["pass"]))
                controls[side] = control
            bisection[name] = controls
    bisection_pass = bool(len(bisection_passes) == 8 and all(bisection_passes))

    repeat_points = {
        "best_midpoint": best_midpoint,
        "left_outside_midpoint": left_outside_midpoint,
        "right_outside_midpoint": right_outside_midpoint,
    }
    repeat_controls: dict[str, float] = {}
    for name, diameter in repeat_points.items():
        reference, _, _ = cache.get(diameter)
        repeated = np.asarray(candidate_generator(diameter))
        repeat_controls[name] = float(
            np.sqrt(raw_complex_probe_loss(repeated, reference))
        )
    repeat_max = max(repeat_controls.values(), default=0.0)
    deterministic_pass = bool(
        max(repeat_relative, repeat_max)
        <= float(thresholds["deterministic_repeat_relative_l2_max"])
    )

    screen_cells = np.asarray(
        [q8_cell_index(value, edges) for value in screen_d], dtype=np.int64
    )
    tie_tolerance = float(design["screened_loss_tie_absolute"])
    screen_tied = screen_loss <= best_loss + tie_tolerance
    external_tie = bool(
        np.any(
            screen_tied
            & ((screen_cells < left_cell) | (screen_cells > right_cell))
        )
    )
    outside_target_losses = np.asarray(
        [
            cache.get(left_outside_midpoint)[1],
            cache.get(right_outside_midpoint)[1],
        ],
        dtype=np.float64,
    )
    adjacent_loss_separation = bool(
        adjacent_available
        and np.all(outside_target_losses > best_loss + tie_tolerance)
    )
    screen_pass = bool(not external_tie and adjacent_loss_separation)

    interval_width = common_right - common_left
    truth_inside = bool(common_left <= true_waist < common_right)
    truth_distance = float(
        0.0
        if truth_inside
        else min(abs(true_waist - common_left), abs(true_waist - common_right))
    )
    endpoint_errors = np.asarray(
        [abs(common_left - true_waist), abs(common_right - true_waist)],
        dtype=np.float64,
    )
    worst_endpoint_error = float(np.max(endpoint_errors))
    worst_relative_error = worst_endpoint_error / true_waist
    boundary_margin = min(common_left - bounds[0], bounds[1] - common_right)
    accuracy_pass = bool(
        0.0 < interval_width <= float(thresholds["interval_width_m_max"])
        and truth_distance
        <= float(
            thresholds[
                "truth_to_interval_distance_m_max_simulation_evaluation_only"
            ]
        )
        and worst_endpoint_error
        <= float(
            thresholds[
                "endpoint_absolute_error_m_max_simulation_evaluation_only"
            ]
        )
        and worst_relative_error
        <= float(
            thresholds[
                "endpoint_relative_error_max_simulation_evaluation_only"
            ]
        )
        and boundary_margin >= float(thresholds["boundary_margin_m"])
    )
    numerical_pass = bool(
        operator_gate
        and breakpoint_pass
        and best_cell_consistency
        and deterministic_pass
        and threshold_stability
        and source_boundary_pass
        and endpoint_convention_pass
        and bisection_pass
        and adjacent_available
        and not cap_hit
    )
    search_pass = bool(equal_budget and all_seeds_qualify and interval_agreement)
    if not numerical_pass:
        status = "Inconclusive"
        interpretation = "reconstructed_q8_interval_numerical_control_not_closed"
    elif not search_pass:
        status = "Failed"
        interpretation = "reconstructed_target_interval_search_unstable"
    elif not screen_pass:
        status = "Failed"
        interpretation = "reconstructed_target_screened_minimum_nonunique"
    elif not accuracy_pass:
        status = "Failed"
        interpretation = "reconstructed_target_interval_accuracy_failed"
    else:
        status = "Passed"
        interpretation = "reconstructed_probe_q8_cell_interval_fit_passed"

    midpoint = float(common_left + 0.5 * interval_width)
    best_probe, best_loss, best_cache_index = cache.get(midpoint)
    target_at_truth_loss = float(cache.get(true_waist)[1])
    loss_improvement = float(target_at_truth_loss - best_loss)
    comparison = _interval_comparison(common_left, common_right, oracle)
    directional_triggered = bool(status != "Passed" or not truth_inside)
    directional: dict[str, Any] = {
        "performed": directional_triggered,
        "trigger": design["directional_diagnostic"]["trigger"],
        "simulation_evaluation_only": True,
        "enters_fitter": False,
    }
    if directional_triggered:
        reconstruction_error = np.asarray(
            target - operator_reference, dtype=np.complex128
        )
        direction_results: dict[str, Any] = {}
        for offset in design["directional_diagnostic"][
            "finite_change_offsets_m"
        ]:
            diameter = true_waist + float(offset)
            probe, _, cache_index = cache.get(diameter)
            direction = np.asarray(
                probe - operator_reference, dtype=np.complex128
            )
            metrics = _direction_metrics(reconstruction_error, direction)
            metrics.update(
                {
                    "diameter_m": diameter,
                    "offset_m": float(offset),
                    "cache_index": int(cache_index),
                }
            )
            direction_results[
                "minus_0p125um" if float(offset) < 0.0 else "plus_0p125um"
            ] = metrics
        best_direction = np.asarray(
            best_probe - operator_reference, dtype=np.complex128
        )
        best_direction_metrics = _direction_metrics(
            reconstruction_error, best_direction
        )
        best_direction_metrics.update(
            {
                "diameter_m": midpoint,
                "offset_m": midpoint - true_waist,
                "cache_index": int(best_cache_index),
            }
        )
        direction_results["best_candidate_manifold"] = best_direction_metrics
        directional.update(
            {
                "reconstruction_error_raw_relative_l2": float(
                    np.sqrt(
                        raw_complex_probe_loss(target, operator_reference)
                    )
                ),
                "directions": direction_results,
            }
        )

    cache_cells = np.asarray(
        [q8_cell_index(value, edges) for value in cache.diameters_m],
        dtype=np.int64,
    )
    return {
        "experiment_status": status,
        "interpretation": interpretation,
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "primary_input_is_raw_matched_q4_p_b_rec": True,
        "truth_used_by_fitter": False,
        "phase_or_scale_alignment": "none",
        "candidate_operator_replay": operator_replay,
        "reconstructed_target_mismatch": reconstructed_mismatch,
        "field_equivalence_threshold": {
            **tau_values,
            "screened_loss_tie_absolute": tie_tolerance,
            "derived_from_reconstruction_error": False,
        },
        "partition": partition,
        "profile": {
            "coarse_d_waist_m": coarse,
            "coarse_loss": coarse_losses,
            "coarse_cache_index": np.asarray(coarse_cache, dtype=np.int64),
            "fine_d_waist_m": fine,
            "fine_loss": fine_losses,
            "fine_cache_index": np.asarray(fine_cache, dtype=np.int64),
        },
        "optimizer": {
            "algorithm": "fixed_budget_bounded_pattern_search",
            "numerical_closure_control": dict(
                optimizer["numerical_closure_control"]
            ),
            "branches": branches,
            "equal_budget": equal_budget,
            "all_final_seeds_qualify": all_seeds_qualify,
            "interval_agreement": interval_agreement,
        },
        "best_reference_components": best_components,
        "source_exact_boundary_control": source_exact_boundary_control,
        "interval_by_branch": interval_by_branch,
        "reported_interval": {
            "lower_m": common_left,
            "upper_m": common_right,
            "analytic_lower_m": analytic_common_left,
            "analytic_upper_m": analytic_common_right,
            "boundary_semantics": (
                "source_exact_float64_membership_transition"
            ),
            "lower_closed": bool(primary_component["lower_closed"]),
            "upper_closed": bool(primary_component["upper_closed"]),
            "width_m": interval_width,
            "midpoint_m": midpoint,
            "midpoint_loss": best_loss,
            "midpoint_cache_index": int(best_cache_index),
            "left_cell_index": left_cell,
            "right_cell_index": right_cell,
            "best_screened_cell_index": best_cell,
            "best_screened_cache_index": int(best_cache_before_interval),
            "fitting_bound_margin_m": boundary_margin,
        },
        "loss_comparison": {
            "L_rec_at_true_diameter": target_at_truth_loss,
            "L_rec_at_best_interval": best_loss,
            "absolute_loss_improvement": loss_improvement,
            "relative_loss_improvement": float(
                loss_improvement / max(target_at_truth_loss, np.finfo(np.float64).eps)
            ),
        },
        "oracle_interval_comparison": comparison,
        "simulation_evaluation_only": {
            "D_waist_true_m": true_waist,
            "truth_inside_reported_interval": truth_inside,
            "truth_to_interval_distance_m": truth_distance,
            "lower_endpoint_absolute_error_m": float(endpoint_errors[0]),
            "upper_endpoint_absolute_error_m": float(endpoint_errors[1]),
            "worst_endpoint_absolute_error_m": worst_endpoint_error,
            "worst_endpoint_relative_error": worst_relative_error,
            "directional_diagnostic": directional,
        },
        "outside_control": {
            "diameter_m": np.asarray(
                [left_outside_midpoint, right_outside_midpoint],
                dtype=np.float64,
            ),
            "reconstructed_target_loss": outside_target_losses,
            "loss_minus_best": outside_target_losses - best_loss,
            "separated_from_best_by_tie_tolerance": adjacent_loss_separation,
        },
        "endpoint_control": endpoint_control,
        "bisection": bisection,
        "deterministic_repeat": {
            "by_point_relative_l2": repeat_controls,
            "maximum_relative_l2": repeat_max,
        },
        "profile_screen": {
            "diameter_m": screen_d,
            "loss": screen_loss,
            "q8_cell_index": screen_cells,
            "cache_index": screen_cache,
            "loss_tied_to_best": screen_tied,
            "loss_tied_count": int(np.count_nonzero(screen_tied)),
            "external_loss_tied_cell_observed": external_tie,
        },
        "P_B_best_candidate_raw": np.asarray(best_probe, dtype=np.complex128),
        "residual_field_raw": np.asarray(best_probe - target, dtype=np.complex128),
        "gates": {
            "candidate_operator_replay_pass": operator_gate,
            "breakpoint_partition_pass": breakpoint_pass,
            "best_cell_field_consistency_pass": best_cell_consistency,
            "deterministic_repeat_pass": deterministic_pass,
            "threshold_stability_pass": threshold_stability,
            "source_exact_boundary_mapping_pass": source_boundary_pass,
            "endpoint_half_open_convention_pass": endpoint_convention_pass,
            "boundary_bisection_pass": bisection_pass,
            "adjacent_outside_cells_available_pass": adjacent_available,
            "cell_expansion_budget_pass": not cap_hit,
            "numerical_controls_pass": numerical_pass,
            "equal_budget_pass": equal_budget,
            "all_branch_seeds_qualify_pass": all_seeds_qualify,
            "multi_start_interval_agreement_pass": interval_agreement,
            "screened_minimum_uniqueness_pass": screen_pass,
            "interval_accuracy_pass_simulation_evaluation_only": accuracy_pass,
        },
        "cache": {
            "D_waist_m": np.asarray(cache.diameters_m, dtype=np.float64),
            "loss": np.asarray(cache.losses, dtype=np.float64),
            "q8_cell_index": cache_cells,
            "P_B_candidate": np.stack(cache.probes).astype(
                np.complex128, copy=False
            ),
        },
    }
