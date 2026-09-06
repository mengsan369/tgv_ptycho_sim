"""Contracts and estimators for exp052 reconstructed-probe waist fitting."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.integer_shift import shift_field_integer_pixels
from tgv_ptycho.inverse.exp050 import (
    load_exp030_source,
    make_exp050_candidate_generator,
    sha256_array,
    sha256_file,
)
from tgv_ptycho.optics.angular_spectrum import (
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)
from tgv_ptycho.recon.epie import epie_reconstruct
from tgv_ptycho.recon.initialization import (
    initialize_probe_by_detector_backpropagation,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_relative_l2(values: NDArray[Any], reference: NDArray[Any]) -> float:
    numerator = float(np.sum(np.abs(np.asarray(values) - np.asarray(reference)) ** 2))
    denominator = float(np.sum(np.abs(np.asarray(reference)) ** 2))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).eps)))


def _require_dataset(
    h5: h5py.File,
    path: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise ValueError(f"Required source dataset is missing: {path}.")
    values = np.asarray(h5[path][()])
    if values.shape != shape or values.dtype != dtype:
        raise ValueError(
            f"Source dataset {path} has {values.shape}/{values.dtype}, "
            f"expected {shape}/{dtype}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Source dataset contains non-finite values: {path}.")
    return values


def validate_exp052_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate the registered exp052 identity and non-truth-selected sources."""

    experiment = config.get("experiment", {})
    if experiment.get("id") != "exp052" or experiment.get("role") != (
        "2d_projected_known_b_and_blind_reconstructed_probe_waist_fit"
    ):
        raise ValueError("Unexpected exp052 experiment identity or role.")
    if mode not in {"preflight", "formal"}:
        raise ValueError("mode must be 'preflight' or 'formal'.")
    source = config.get("source", {})
    exp030 = source.get("exp030", {})
    if exp030.get("target_dataset") != "/entry/truth/P_B_true":
        raise ValueError("exp052 exp030 target identity must be raw P_B_true.")
    forbidden_tokens = (
        "exp040",
        "exp042",
        "exp051",
        "exp053",
        "aligned",
        "common_gauge",
    )
    serialized_source = json.dumps(source, sort_keys=True).lower()
    if any(token in serialized_source for token in forbidden_tokens):
        raise ValueError(
            "exp052 source contract contains a forbidden 3D/aligned source."
        )
    blind = config.get("part_b", {})
    if blind.get("selection_rule") != (
        "fixed_authoritative_blind_long_final_1000_not_checkpoint_selected"
    ):
        raise ValueError("Part B must use the fixed non-truth-selected final state.")
    raw_probe = str(blind.get("raw_probe_dataset", ""))
    if not raw_probe.endswith("/blind_long_study/baseline/P_B_rec_raw"):
        raise ValueError("Part B raw target must be the blind-long raw final probe.")
    if any(
        token in raw_probe.lower() for token in ("aligned", "truth", "common_gauge")
    ):
        raise ValueError("Aligned or truth-selected Part B targets are forbidden.")
    known = config.get("part_a", {})
    if known.get("initialization") != "measurement_mean_amplitude_backpropagation":
        raise ValueError("Part A must use the registered truth-free initialization.")
    if not bool(known.get("fixed_known_b", False)):
        raise ValueError("Part A must freeze known sample B.")
    design = config.get(mode, {})
    lower, upper = (float(value) for value in design.get("bounds_m", (0.0, 0.0)))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Fitting bounds must be finite and increasing.")
    for name in (
        "global_step_m",
        "fine_half_width_m",
        "fine_step_m",
        "refine_half_width_m",
        "refine_step_m",
    ):
        if float(design.get(name, 0.0)) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if config.get("gauge", {}).get("primary") != "global_phase_profiled_full_field":
        raise ValueError("The frozen primary gauge must profile global phase only.")


def load_exp052_sources(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    """Load and verify the exp030 measurement chain and exp050 oracle."""

    exp030_cfg = config["source"]["exp030"]
    adapter = {
        "source": exp030_cfg,
        "instrument": config["instrument"],
        "sample_a": config["sample_a"],
    }
    source = load_exp030_source(adapter, project_root)
    run_dir = project_root / str(exp030_cfg["run"])
    hdf5_path = run_dir / str(exp030_cfg["hdf5_relative_path"])
    shape = tuple(int(value) for value in exp030_cfg["shape_ny_nx"])
    paths = config["source"]["datasets"]
    expected_hashes = config["source"]["dataset_bytes_sha256"]
    with h5py.File(hdf5_path, "r") as h5:
        intensity = _require_dataset(
            h5,
            str(paths["I_stack"]),
            shape=(int(exp030_cfg["num_frames"]), *shape),
            dtype=np.dtype(np.float64),
        )
        positions = _require_dataset(
            h5,
            str(paths["scan_positions"]),
            shape=(int(exp030_cfg["num_frames"]), 2),
            dtype=np.dtype(np.float64),
        )
        sample_b = _require_dataset(
            h5, str(paths["B_true"]), shape=shape, dtype=np.dtype(np.complex128)
        )
        probe_true = _require_dataset(
            h5, str(paths["P_B_true"]), shape=shape, dtype=np.dtype(np.complex128)
        )
        arrays = {
            "I_stack": intensity,
            "scan_positions": positions,
            "B_true": sample_b,
            "P_B_true": probe_true,
        }
        for name, values in arrays.items():
            actual = sha256_array(np.asarray(values))
            if actual != str(expected_hashes[name]).upper():
                raise ValueError(f"exp030 dataset byte hash mismatch for {name}.")

        blind_paths = config["part_b"]
        blind_probe = _require_dataset(
            h5,
            str(blind_paths["raw_probe_dataset"]),
            shape=shape,
            dtype=np.dtype(np.complex128),
        )
        blind_object = _require_dataset(
            h5,
            str(blind_paths["raw_object_dataset"]),
            shape=shape,
            dtype=np.dtype(np.complex128),
        )
        blind_probe_init = _require_dataset(
            h5,
            str(blind_paths["probe_init_dataset"]),
            shape=shape,
            dtype=np.dtype(np.complex128),
        )
        blind_object_init = _require_dataset(
            h5,
            str(blind_paths["object_init_dataset"]),
            shape=shape,
            dtype=np.dtype(np.complex128),
        )
        blind_loss = np.asarray(h5[str(blind_paths["loss_curve_dataset"])][()])
        if (
            sha256_array(blind_probe)
            != str(blind_paths["expected_raw_probe_sha256"]).upper()
        ):
            raise ValueError("Part B raw probe hash mismatch.")
        if (
            sha256_array(blind_object)
            != str(blind_paths["expected_raw_object_sha256"]).upper()
        ):
            raise ValueError("Part B raw object hash mismatch.")
        if (
            blind_loss.shape != (1000,)
            or blind_loss.dtype != np.float64
            or not np.all(np.isfinite(blind_loss))
        ):
            raise ValueError(
                "Part B loss trajectory must be finite float64 with 1000 entries."
            )
        blind_final_loss = float(h5[str(blind_paths["final_loss_dataset"])][()])
        blind_settings_group = h5[str(blind_paths["settings_group"])]
        blind_settings = {
            name: _decode(dataset[()])
            for name, dataset in blind_settings_group.items()
            if isinstance(dataset, h5py.Dataset)
        }
        required_false = (
            "uses_simulation_truth_B_as_input",
            "uses_simulation_truth_probe_as_input",
        )
        if any(bool(blind_settings.get(name, True)) for name in required_false):
            raise ValueError("Part B optimizer provenance reports truth use.")
        if not bool(blind_settings.get("update_probe")) or not bool(
            blind_settings.get("update_object")
        ):
            raise ValueError("Part B did not jointly update probe and object.")
        if int(blind_settings.get("num_iters", 0)) != 1000:
            raise ValueError(
                "Part B final source is not the registered 1000-iteration state."
            )
        checkpoints: dict[str, dict[str, Any]] = {}
        for iteration in (200, 500, 1000):
            root = str(blind_paths["checkpoint_group_template"]).format(
                iteration=iteration
            )
            checkpoint_probe = _require_dataset(
                h5, f"{root}/P_B_rec", shape=shape, dtype=np.dtype(np.complex128)
            )
            checkpoint_object = _require_dataset(
                h5, f"{root}/B_rec", shape=shape, dtype=np.dtype(np.complex128)
            )
            checkpoints[str(iteration)] = {
                "P_B_rec_raw": checkpoint_probe,
                "B_rec_raw": checkpoint_object,
                "data_fidelity_loss": float(h5[f"{root}/data_fidelity_loss"][()]),
            }
        if not np.array_equal(checkpoints["1000"]["P_B_rec_raw"], blind_probe):
            raise ValueError("Part B final raw probe differs from checkpoint 1000.")
        if not np.array_equal(checkpoints["1000"]["B_rec_raw"], blind_object):
            raise ValueError("Part B final raw object differs from checkpoint 1000.")

    oracle_cfg = config["source"]["exp050_oracle"]
    oracle_run = project_root / str(oracle_cfg["run"])
    oracle_paths = {
        "config": oracle_run / "config.yaml",
        "metadata": oracle_run / "metadata.json",
        "metrics": oracle_run / "metrics.json",
        "run_state": oracle_run / "run_state.json",
        "hdf5": oracle_run / str(oracle_cfg["hdf5_relative_path"]),
    }
    for name, expected in oracle_cfg["sha256"].items():
        if sha256_file(oracle_paths[name]) != str(expected).upper():
            raise ValueError(f"exp050 oracle artifact hash mismatch for {name}.")
    oracle_state = json.loads(oracle_paths["run_state"].read_text(encoding="utf-8"))
    oracle_metrics = json.loads(oracle_paths["metrics"].read_text(encoding="utf-8"))
    if (
        oracle_state.get("status") != "complete"
        or not oracle_state.get("artifacts_validated")
        or oracle_metrics.get("experiment_status") != "Passed"
    ):
        raise ValueError("exp050 oracle is not complete, validated, and Passed.")
    oracle_fit = oracle_metrics["fit"]

    durable_checkpoint = run_dir / str(
        config["part_b"]["durable_checkpoint_relative_path"]
    )
    if (
        sha256_file(durable_checkpoint)
        != str(config["part_b"]["durable_checkpoint_sha256"]).upper()
    ):
        raise ValueError("Part B durable checkpoint hash mismatch.")
    with h5py.File(durable_checkpoint, "r") as checkpoint_h5:
        checkpoint_metadata = {
            "format": str(checkpoint_h5.attrs.get("format", "")),
            "case_id": str(checkpoint_h5.attrs.get("case_id", "")),
            "completed_iterations": int(
                checkpoint_h5["optimizer_state/completed_iterations"][()]
            ),
            "problem_signature": _decode(
                checkpoint_h5["optimizer_state/problem_signature"][()]
            ),
            "rng_bit_generator": _decode(
                checkpoint_h5["optimizer_state/rng_bit_generator"][()]
            ),
        }
    if (
        checkpoint_metadata["completed_iterations"] != 1000
        or checkpoint_metadata["case_id"] != "baseline"
    ):
        raise ValueError("Part B durable checkpoint is not the baseline final state.")
    if str(checkpoint_metadata["problem_signature"]) != str(
        blind_settings["problem_signature"]
    ):
        raise ValueError("Part B HDF5 and durable checkpoint signatures disagree.")

    source.update(
        {
            **arrays,
            "exp030_hdf5_path": hdf5_path,
            "dataset_hashes": {
                name: sha256_array(np.asarray(value)) for name, value in arrays.items()
            },
            "blind": {
                "P_B_init": blind_probe_init,
                "B_init": blind_object_init,
                "P_B_rec_raw": blind_probe,
                "B_rec_raw": blind_object,
                "loss_curve": blind_loss,
                "final_data_fidelity_loss": blind_final_loss,
                "settings": blind_settings,
                "checkpoints": checkpoints,
                "durable_checkpoint_path": durable_checkpoint,
                "durable_checkpoint_sha256": sha256_file(durable_checkpoint),
                "durable_checkpoint_metadata": checkpoint_metadata,
            },
            "oracle": {
                "run_dir": oracle_run,
                "paths": {name: str(path) for name, path in oracle_paths.items()},
                "estimate_m": float(oracle_fit["D_waist_estimate_m"]),
                "interval_m": np.asarray(
                    oracle_fit["profile"]["reported_resolution_interval_m"],
                    dtype=np.float64,
                ),
                "interval_width_m": float(
                    oracle_fit["profile"]["reported_resolution_interval_width_m"]
                ),
                "loss_curvature_per_m2": float(
                    oracle_fit["finite_difference"]["loss_curvature_per_m2"]
                ),
                "config_sha256": str(oracle_cfg["frozen_source_config_sha256"]),
                "frozen_body_sha256": str(oracle_cfg["frozen_body_sha256"]),
            },
        }
    )
    return source


def replay_measurement_operator(source: Mapping[str, Any]) -> dict[str, float]:
    """Replay the saved detector stack and one propagation adjoint dot test."""

    intensity = np.asarray(source["I_stack"], dtype=np.float64)
    positions = np.asarray(source["scan_positions"], dtype=np.float64)
    sample_b = np.asarray(source["B_true"], dtype=np.complex128)
    probe = np.asarray(source["P_B_true"], dtype=np.complex128)
    instrument = source["instrument"]
    transfer = make_angular_spectrum_transfer(
        probe.shape,
        float(instrument["dx_m"]),
        float(instrument["wavelength_m"]),
        float(instrument["z_BC_m"]),
        n=float(instrument["medium_index"]),
        bandlimit=True,
    )
    replay = np.empty_like(intensity)
    for index, position in enumerate(positions):
        shifted = shift_field_integer_pixels(
            sample_b, position, float(instrument["dx_m"]), boundary="periodic"
        )
        field = apply_angular_spectrum_transfer(probe * shifted, transfer)
        replay[index] = np.abs(field) ** 2
    intensity_replay = _array_relative_l2(replay, intensity)
    amplitude_replay = _array_relative_l2(np.sqrt(replay), np.sqrt(intensity))
    rng = np.random.default_rng(20260902)
    left = (rng.normal(size=probe.shape) + 1j * rng.normal(size=probe.shape)).astype(
        np.complex128
    )
    right = (rng.normal(size=probe.shape) + 1j * rng.normal(size=probe.shape)).astype(
        np.complex128
    )
    lhs = np.sum(np.conj(apply_angular_spectrum_transfer(left, transfer)) * right)
    rhs = np.sum(
        np.conj(left) * apply_angular_spectrum_transfer(right, np.conj(transfer))
    )
    adjoint_error = float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).eps))
    return {
        "intensity_relative_l2": intensity_replay,
        "amplitude_relative_l2": amplitude_replay,
        "adjoint_inner_product_relative_error": adjoint_error,
    }


def frozen_data_fidelity(
    probe: ComplexArray,
    sample_b: ComplexArray,
    source: Mapping[str, Any],
) -> float:
    """Independently recompute the mean frozen detector-amplitude residual."""

    intensity = np.asarray(source["I_stack"], dtype=np.float64)
    positions = np.asarray(source["scan_positions"], dtype=np.float64)
    instrument = source["instrument"]
    transfer = make_angular_spectrum_transfer(
        probe.shape,
        float(instrument["dx_m"]),
        float(instrument["wavelength_m"]),
        float(instrument["z_BC_m"]),
        n=float(instrument["medium_index"]),
        bandlimit=True,
    )
    total = 0.0
    for index, position in enumerate(positions):
        shifted = shift_field_integer_pixels(
            sample_b, position, float(instrument["dx_m"]), boundary="periodic"
        )
        field = apply_angular_spectrum_transfer(probe * shifted, transfer)
        total += _array_relative_l2(np.abs(field), np.sqrt(intensity[index]))
    return float(total / len(positions))


def run_known_b_reconstruction(
    source: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    repeat: bool,
) -> dict[str, Any]:
    """Run the truth-free-initialized known-B probe-only reconstruction."""

    instrument = source["instrument"]
    intensity = np.asarray(source["I_stack"], dtype=np.float64)
    positions = np.asarray(source["scan_positions"], dtype=np.float64)
    sample_b = np.asarray(source["B_true"], dtype=np.complex128)
    init_probe = initialize_probe_by_detector_backpropagation(
        intensity,
        float(instrument["dx_m"]),
        float(instrument["wavelength_m"]),
        float(instrument["z_BC_m"]),
    )

    def execute() -> dict[str, Any]:
        return epie_reconstruct(
            intensity,
            positions,
            dx=float(instrument["dx_m"]),
            wavelength=float(instrument["wavelength_m"]),
            z_BC=float(instrument["z_BC_m"]),
            num_iters=int(settings["num_iters"]),
            beta_probe=float(settings["beta_probe"]),
            beta_object=0.0,
            init_probe=init_probe,
            init_object=sample_b,
            update_probe=True,
            update_object=False,
            shuffle_positions=bool(settings["shuffle_positions"]),
            seed=int(settings["seed"]),
            object_amplitude_bounds=None,
            correction_mode="adjoint_residual",
            denominator_mode="epie",
            object_boundary="periodic",
            checkpoint_iters=tuple(
                int(value) for value in settings["checkpoint_iterations"]
            ),
            show_progress=False,
        )

    result = execute()
    repeated = execute() if repeat else None
    probe = np.asarray(result["P_B_rec"], dtype=np.complex128)
    object_b = np.asarray(result["B_rec"], dtype=np.complex128)
    independent_loss = frozen_data_fidelity(probe, object_b, source)
    checkpoints: dict[str, Any] = {}
    for iteration, checkpoint in result["checkpoints"].items():
        checkpoint_probe = np.asarray(checkpoint["P_B_rec"], dtype=np.complex128)
        checkpoint_object = np.asarray(checkpoint["B_rec"], dtype=np.complex128)
        checkpoint_state = checkpoint["optimizer_state"]
        checkpoints[iteration] = {
            "P_B_rec_raw": checkpoint_probe,
            "B_fixed": checkpoint_object,
            "data_fidelity_loss": float(checkpoint["data_fidelity_loss"]),
            "independent_data_fidelity_loss": frozen_data_fidelity(
                checkpoint_probe, checkpoint_object, source
            ),
            "loss_curve": np.asarray(checkpoint_state["loss_curve"], dtype=np.float64),
            "optimizer_state": {
                "completed_iterations": int(checkpoint_state["completed_iterations"]),
                "problem_signature": str(checkpoint_state["problem_signature"]),
                "rng_bit_generator": str(checkpoint_state["rng_bit_generator"]),
                "rng_state_json": json.dumps(
                    checkpoint_state["rng_state"], sort_keys=True
                ),
            },
        }
    repeat_metrics = {
        "executed": repeated is not None,
        "P_B_relative_l2": 0.0,
        "B_relative_l2": 0.0,
        "loss_curve_relative_l2": 0.0,
        "final_loss_absolute_difference": 0.0,
        "bitwise_equal": True,
    }
    if repeated is not None:
        repeat_metrics = {
            "executed": True,
            "P_B_relative_l2": _array_relative_l2(repeated["P_B_rec"], probe),
            "B_relative_l2": _array_relative_l2(repeated["B_rec"], object_b),
            "loss_curve_relative_l2": _array_relative_l2(
                repeated["loss_curve"], result["loss_curve"]
            ),
            "final_loss_absolute_difference": abs(
                float(repeated["final_data_fidelity_loss"])
                - float(result["final_data_fidelity_loss"])
            ),
            "bitwise_equal": bool(
                np.array_equal(repeated["P_B_rec"], probe)
                and np.array_equal(repeated["B_rec"], object_b)
                and np.array_equal(repeated["loss_curve"], result["loss_curve"])
            ),
        }
    return {
        "P_B_init": init_probe,
        "B_known": sample_b,
        "P_B_rec_raw": probe,
        "B_fixed": object_b,
        "loss_curve": np.asarray(result["loss_curve"], dtype=np.float64),
        "initial_data_fidelity_loss": float(result["initial_data_fidelity_loss"]),
        "final_data_fidelity_loss": float(result["final_data_fidelity_loss"]),
        "independent_data_fidelity_loss": independent_loss,
        "fixed_B_max_abs_change": float(np.max(np.abs(object_b - sample_b))),
        "settings": result["metadata"],
        "optimizer_state": {
            "completed_iterations": int(
                result["optimizer_state"]["completed_iterations"]
            ),
            "problem_signature": str(result["optimizer_state"]["problem_signature"]),
            "rng_bit_generator": str(result["optimizer_state"]["rng_bit_generator"]),
            "rng_state_json": json.dumps(
                result["optimizer_state"]["rng_state"], sort_keys=True
            ),
        },
        "checkpoints": checkpoints,
        "repeat": repeat_metrics,
    }


def profile_global_phase(
    candidate: ComplexArray,
    target: ComplexArray,
) -> dict[str, Any]:
    """Return raw, global-phase-profiled, and complex-gain diagnostic losses."""

    candidate_values = np.asarray(candidate)
    target_values = np.asarray(target)
    if (
        candidate_values.dtype != np.complex128
        or target_values.dtype != np.complex128
        or candidate_values.shape != target_values.shape
        or candidate_values.ndim != 2
    ):
        raise ValueError(
            "candidate and target must be same-shaped 2D complex128 fields."
        )
    if not np.all(np.isfinite(candidate_values)) or not np.all(
        np.isfinite(target_values)
    ):
        raise ValueError("candidate and target must be finite.")
    target_energy = float(np.sum(np.abs(target_values) ** 2))
    candidate_energy = float(np.sum(np.abs(candidate_values) ** 2))
    if min(target_energy, candidate_energy) <= np.finfo(float).eps:
        raise ValueError("candidate and target must have positive energy.")
    correlation = np.sum(np.conj(candidate_values) * target_values)
    phase_gain = (
        complex(np.exp(1j * np.angle(correlation))) if correlation != 0 else 1.0 + 0.0j
    )
    complex_gain = complex(correlation / candidate_energy)
    raw_residual = candidate_values - target_values
    primary_residual = phase_gain * candidate_values - target_values
    gain_residual = complex_gain * candidate_values - target_values
    return {
        "raw_loss": float(np.sum(np.abs(raw_residual) ** 2) / target_energy),
        "primary_loss": float(np.sum(np.abs(primary_residual) ** 2) / target_energy),
        "complex_gain_profiled_loss_diagnostic": float(
            np.sum(np.abs(gain_residual) ** 2) / target_energy
        ),
        "global_phase_gain": phase_gain,
        "global_phase_rad": float(np.angle(phase_gain)),
        "complex_gain_diagnostic": complex_gain,
        "complex_gain_magnitude_diagnostic": float(abs(complex_gain)),
        "raw_residual": np.asarray(raw_residual, dtype=np.complex128),
        "primary_residual": np.asarray(primary_residual, dtype=np.complex128),
    }


@dataclass
class CandidateCache:
    """Deterministic candidate cache shared by both exp052 arms."""

    generator: Callable[[float], Mapping[str, Any]]
    target_shape: tuple[int, int]
    index_by_key: dict[float, int] = field(default_factory=dict)
    diameters_m: list[float] = field(default_factory=list)
    probes: list[ComplexArray] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)

    def get(self, diameter_m: float) -> tuple[ComplexArray, int]:
        key = float(np.float64(diameter_m))
        if not np.isfinite(key):
            raise ValueError("candidate diameter must be finite.")
        index = self.index_by_key.get(key)
        if index is not None:
            return self.probes[index], index
        result = self.generator(key)
        probe = np.asarray(result["P_B"])
        if (
            probe.shape != self.target_shape
            or probe.dtype != np.complex128
            or not np.all(np.isfinite(probe))
        ):
            raise ValueError("candidate generator returned an invalid probe.")
        index = len(self.probes)
        self.index_by_key[key] = index
        self.diameters_m.append(key)
        self.probes.append(probe.copy())
        self.hashes.append(sha256_array(probe))
        return self.probes[index], index


def _regular_grid(lower: float, upper: float, step: float) -> FloatArray:
    count = int(np.rint((upper - lower) / step))
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(upper))
    if count <= 0 or not np.isclose(
        lower + count * step, upper, rtol=0.0, atol=tolerance
    ):
        raise ValueError("profile interval must be exactly divisible by its step.")
    grid = lower + np.arange(count + 1, dtype=np.float64) * step
    grid[-1] = upper
    return grid


def _bounded_profile_interval(
    center: float,
    half_width: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Translate a fixed-width local interval so it stays inside frozen bounds."""

    width = 2.0 * half_width
    if width > upper_bound - lower_bound:
        raise ValueError("local profile interval is wider than registered bounds.")
    interval_lower = min(max(center - half_width, lower_bound), upper_bound - width)
    interval_upper = interval_lower + width
    return float(interval_lower), float(interval_upper)


def _profile_grid(
    target: ComplexArray,
    diameters: FloatArray,
    cache: CandidateCache,
) -> dict[str, Any]:
    raw: list[float] = []
    primary: list[float] = []
    gain: list[float] = []
    phase: list[float] = []
    complex_gain: list[complex] = []
    cache_indices: list[int] = []
    hashes: list[str] = []
    for diameter in diameters:
        candidate, index = cache.get(float(diameter))
        metrics = profile_global_phase(candidate, target)
        raw.append(float(metrics["raw_loss"]))
        primary.append(float(metrics["primary_loss"]))
        gain.append(float(metrics["complex_gain_profiled_loss_diagnostic"]))
        phase.append(float(metrics["global_phase_rad"]))
        complex_gain.append(complex(metrics["complex_gain_diagnostic"]))
        cache_indices.append(index)
        hashes.append(cache.hashes[index])
    return {
        "D_waist_m": np.asarray(diameters, dtype=np.float64),
        "raw_loss": np.asarray(raw, dtype=np.float64),
        "primary_loss": np.asarray(primary, dtype=np.float64),
        "complex_gain_profiled_loss_diagnostic": np.asarray(gain, dtype=np.float64),
        "global_phase_rad": np.asarray(phase, dtype=np.float64),
        "complex_gain_diagnostic": np.asarray(complex_gain, dtype=np.complex128),
        "cache_index": np.asarray(cache_indices, dtype=np.int64),
        "candidate_sha256": np.asarray(hashes, dtype=object),
    }


def _local_minima_indices(values: FloatArray, tolerance: float) -> list[int]:
    minima: list[int] = []
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else np.inf
        right = values[index + 1] if index + 1 < len(values) else np.inf
        if value <= left + tolerance and value <= right + tolerance:
            minima.append(index)
    return minima


def fit_reconstructed_probe(
    target: ComplexArray,
    cache: CandidateCache,
    *,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one raw reconstructed target with the registered global-phase loss."""

    target_values = np.asarray(target)
    if (
        target_values.dtype != np.complex128
        or target_values.shape != cache.target_shape
        or not np.all(np.isfinite(target_values))
    ):
        raise ValueError("target must be finite complex128 with the registered shape.")
    lower, upper = (float(value) for value in design["bounds_m"])
    global_grid = _regular_grid(lower, upper, float(design["global_step_m"]))
    global_profile = _profile_grid(target_values, global_grid, cache)
    global_index = int(np.argmin(global_profile["primary_loss"]))
    global_center = float(global_grid[global_index])
    fine_lower, fine_upper = _bounded_profile_interval(
        global_center,
        float(design["fine_half_width_m"]),
        lower,
        upper,
    )
    fine_grid = _regular_grid(fine_lower, fine_upper, float(design["fine_step_m"]))
    fine_profile = _profile_grid(target_values, fine_grid, cache)
    fine_index = int(np.argmin(fine_profile["primary_loss"]))
    fine_center = float(fine_grid[fine_index])
    refine_lower, refine_upper = _bounded_profile_interval(
        fine_center,
        float(design["refine_half_width_m"]),
        lower,
        upper,
    )
    refine_grid = _regular_grid(
        refine_lower, refine_upper, float(design["refine_step_m"])
    )
    refine_profile = _profile_grid(target_values, refine_grid, cache)
    refine_index = int(np.argmin(refine_profile["primary_loss"]))
    estimate = float(refine_grid[refine_index])
    tolerance = float(design["loss_tie_tolerance"])
    minima = _local_minima_indices(
        np.asarray(global_profile["primary_loss"]), tolerance
    )
    minimum_loss = float(refine_profile["primary_loss"][refine_index])
    equivalent_indices = np.flatnonzero(
        np.asarray(refine_profile["primary_loss"]) <= minimum_loss + tolerance
    )
    components = (
        int(1 + np.count_nonzero(np.diff(equivalent_indices) > 1))
        if equivalent_indices.size
        else 0
    )
    component_bounds: list[tuple[float, float]] = []
    if equivalent_indices.size:
        starts = np.r_[0, np.flatnonzero(np.diff(equivalent_indices) > 1) + 1]
        stops = np.r_[starts[1:] - 1, equivalent_indices.size - 1]
        for start, stop in zip(starts, stops, strict=True):
            component_bounds.append(
                (
                    float(refine_grid[equivalent_indices[start]]),
                    float(refine_grid[equivalent_indices[stop]]),
                )
            )
    if refine_index == 0 or refine_index == len(refine_grid) - 1:
        interval = np.asarray([estimate, estimate], dtype=np.float64)
    else:
        interval = np.asarray(
            [
                0.5 * (refine_grid[refine_index - 1] + refine_grid[refine_index]),
                0.5 * (refine_grid[refine_index] + refine_grid[refine_index + 1]),
            ],
            dtype=np.float64,
        )
    if 0 < refine_index < len(refine_grid) - 1:
        x = refine_grid[refine_index - 1 : refine_index + 2] - estimate
        y = np.asarray(refine_profile["primary_loss"])[
            refine_index - 1 : refine_index + 2
        ]
        coefficients = np.polyfit(x, y, 2)
        quadratic_offset = (
            float(-coefficients[1] / (2.0 * coefficients[0]))
            if coefficients[0] > 0
            else 0.0
        )
        quadratic_estimate = estimate + quadratic_offset
        curvature = float(2.0 * coefficients[0])
    else:
        quadratic_estimate = estimate
        curvature = 0.0
    best_candidate, best_cache_index = cache.get(estimate)
    best = profile_global_phase(best_candidate, target_values)
    return {
        "global_profile": global_profile,
        "fine_profile": fine_profile,
        "refinement_profile": refine_profile,
        "global_minimum_index": global_index,
        "fine_minimum_index": fine_index,
        "refinement_minimum_index": refine_index,
        "global_local_minima_indices": np.asarray(minima, dtype=np.int64),
        "global_local_minima_count": len(minima),
        "equivalence_indices": equivalent_indices.astype(np.int64),
        "equivalence_D_waist_m": np.asarray(
            refine_grid[equivalent_indices], dtype=np.float64
        ),
        "equivalence_components_m": np.asarray(
            component_bounds, dtype=np.float64
        ).reshape(-1, 2),
        "equivalence_component_count": components,
        "estimate_m": estimate,
        "reported_interval_m": interval,
        "reported_interval_width_m": float(interval[1] - interval[0]),
        "reported_interval_role": (
            "registered_profile_grid_cell_not_physical_uncertainty"
        ),
        "minimum_primary_loss": minimum_loss,
        "minimum_raw_loss": float(best["raw_loss"]),
        "minimum_complex_gain_profiled_loss_diagnostic": float(
            best["complex_gain_profiled_loss_diagnostic"]
        ),
        "global_phase_gain": complex(best["global_phase_gain"]),
        "global_phase_rad": float(best["global_phase_rad"]),
        "complex_gain_diagnostic": complex(best["complex_gain_diagnostic"]),
        "quadratic_crosscheck_estimate_m": float(quadratic_estimate),
        "quadratic_profile_agreement_m": abs(float(quadratic_estimate) - estimate),
        "local_loss_curvature_per_m2": curvature,
        "boundary_hit": bool(
            refine_index in {0, len(refine_grid) - 1}
            or estimate <= lower
            or estimate >= upper
        ),
        "best_cache_index": best_cache_index,
        "P_B_best_raw": best_candidate,
        "residual_field_raw": best["raw_residual"],
        "residual_field_primary": best["primary_residual"],
    }


def evaluate_fit_gates(
    fit: Mapping[str, Any],
    *,
    true_waist_m: float,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate one arm's frozen fit gates after the landscape is complete."""

    estimate = float(fit["estimate_m"])
    error = abs(estimate - true_waist_m)
    interval = np.asarray(fit["reported_interval_m"], dtype=np.float64)
    truth_distance = max(
        float(interval[0] - true_waist_m), float(true_waist_m - interval[1]), 0.0
    )
    coordinate_scale = np.float64(
        max(
            abs(estimate), abs(true_waist_m), *(abs(float(value)) for value in interval)
        )
    )
    comparison_tolerance_m = float(32.0 * abs(np.spacing(coordinate_scale)))

    def within(value: float, threshold_name: str) -> bool:
        return value <= float(thresholds[threshold_name]) + comparison_tolerance_m

    unique = int(fit["global_local_minima_count"]) == 1
    single_component = int(fit["equivalence_component_count"]) == 1
    interior = not bool(fit["boundary_hit"])
    curvature_positive = bool(
        np.isfinite(float(fit["local_loss_curvature_per_m2"]))
        and float(fit["local_loss_curvature_per_m2"]) > 0.0
    )
    accuracy_pass = within(error, "absolute_error_m_max")
    truth_interval_pass = within(truth_distance, "truth_to_interval_distance_m_max")
    interval_width_pass = within(
        float(fit["reported_interval_width_m"]), "interval_width_m_max"
    )
    method_agreement_pass = within(
        float(fit["quadratic_profile_agreement_m"]), "method_agreement_m_max"
    )
    pass_value = bool(
        unique
        and single_component
        and interior
        and curvature_positive
        and accuracy_pass
        and truth_interval_pass
        and interval_width_pass
        and method_agreement_pass
    )
    return {
        "pass": pass_value,
        "absolute_error_m_simulation_evaluation_only": error,
        "truth_to_interval_distance_m_simulation_evaluation_only": truth_distance,
        "comparison_tolerance_m": comparison_tolerance_m,
        "unique_global_minimum": unique,
        "single_equivalence_component": single_component,
        "interior": interior,
        "positive_finite_curvature": curvature_positive,
        "absolute_error_pass": accuracy_pass,
        "truth_to_interval_distance_pass": truth_interval_pass,
        "interval_width_pass": interval_width_pass,
        "method_agreement_pass": method_agreement_pass,
    }


def combine_status(
    *,
    shared_status: str,
    part_a_reconstruction_status: str,
    part_a_fit_status: str,
    part_b_reconstruction_status: str,
    part_b_fit_status: str,
) -> dict[str, str]:
    """Apply the mutually exclusive exp052 arm and overall status matrix."""

    valid = {"Passed", "Failed", "Inconclusive"}
    statuses = {
        "shared": shared_status,
        "part_a_reconstruction": part_a_reconstruction_status,
        "part_a_fit": part_a_fit_status,
        "part_b_reconstruction": part_b_reconstruction_status,
        "part_b_fit": part_b_fit_status,
    }
    if any(value not in valid for value in statuses.values()):
        raise ValueError(
            "All exp052 component statuses must be mutually exclusive values."
        )
    if shared_status != "Passed":
        part_a = "Inconclusive"
        part_b = "Inconclusive"
        overall = "Inconclusive"
    else:
        part_a = (
            "Passed"
            if part_a_reconstruction_status == part_a_fit_status == "Passed"
            else (
                "Failed"
                if "Failed" in {part_a_reconstruction_status, part_a_fit_status}
                else "Inconclusive"
            )
        )
        part_b = (
            "Passed"
            if part_b_reconstruction_status == part_b_fit_status == "Passed"
            else (
                "Failed"
                if "Failed" in {part_b_reconstruction_status, part_b_fit_status}
                else "Inconclusive"
            )
        )
        overall = (
            "Passed"
            if part_a == part_b == "Passed"
            else ("Failed" if "Failed" in {part_a, part_b} else "Inconclusive")
        )
    return {
        **statuses,
        "part_a_overall": part_a,
        "part_b_overall": part_b,
        "overall": overall,
    }


def make_candidate_cache(source: Mapping[str, Any]) -> CandidateCache:
    """Create the shared exp050-compatible projected candidate cache."""

    return CandidateCache(
        make_exp050_candidate_generator(source),
        tuple(int(value) for value in source["shape_ny_nx"]),
    )


def summarize_checkpoint_fits(
    targets: Mapping[str, ComplexArray],
    cache: CandidateCache,
    *,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit every registered checkpoint without selecting one by truth error."""

    return {
        name: fit_reconstructed_probe(np.asarray(target), cache, design=design)
        for name, target in targets.items()
    }


def truth_evaluation(
    probe: ComplexArray,
    object_b: ComplexArray,
    source: Mapping[str, Any],
) -> dict[str, float]:
    """Return explicitly simulation-only probe/object reconstruction metrics."""

    true_probe = np.asarray(source["P_B_true"], dtype=np.complex128)
    true_object = np.asarray(source["B_true"], dtype=np.complex128)
    probe_metrics = profile_global_phase(
        np.asarray(probe, dtype=np.complex128), true_probe
    )
    object_metrics = profile_global_phase(
        np.asarray(object_b, dtype=np.complex128), true_object
    )
    return {
        "probe_global_phase_profiled_relative_l2": float(
            np.sqrt(probe_metrics["primary_loss"])
        ),
        "probe_complex_gain_profiled_relative_l2_diagnostic": float(
            np.sqrt(probe_metrics["complex_gain_profiled_loss_diagnostic"])
        ),
        "object_global_phase_profiled_relative_l2": float(
            np.sqrt(object_metrics["primary_loss"])
        ),
        "object_complex_gain_profiled_relative_l2_diagnostic": float(
            np.sqrt(object_metrics["complex_gain_profiled_loss_diagnostic"])
        ),
    }


__all__ = [
    "CandidateCache",
    "combine_status",
    "evaluate_fit_gates",
    "fit_reconstructed_probe",
    "frozen_data_fidelity",
    "load_exp052_sources",
    "make_candidate_cache",
    "profile_global_phase",
    "replay_measurement_operator",
    "run_known_b_reconstruction",
    "sha256_array",
    "sha256_file",
    "summarize_checkpoint_fits",
    "truth_evaluation",
    "validate_exp052_config",
]
