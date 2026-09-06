"""Blind probe/sample-B reconstruction on the frozen exp042 operator.

The implementation keeps the exp042 reference-plus-residual, q4 detector
operator and mean half squared intensity loss.  It adds a variable finite-B
modulation and alternating, spectrally damped block Gauss--Newton/CG updates.
Simulation truth is loaded only for operator controls and post-hoc evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.exp040 import center_crop, relative_l2
from tgv_ptycho.forward.integer_shift import (
    shift_field_integer_pixels,
    unshift_field_delta_integer_pixels,
)
from tgv_ptycho.optics.angular_spectrum import apply_angular_spectrum_transfer
from tgv_ptycho.recon.exp042 import (
    MatchedKnownBProbeOperator,
    _center_embed,
    _l2_norm,
    _real_inner_product,
    build_matched_development_case,
    detector_relative_residual,
    operator_consistency_metrics,
    reconstruct_known_b_probe_damped_gn_cg,
)

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]
BlockName = Literal["probe", "sample_b"]


def sha256_file(path: str | Path) -> str:
    """Return an uppercase SHA256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array_bytes(values: NDArray[np.generic]) -> str:
    """Hash the contiguous C-order bytes of one persisted dataset."""

    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest().upper()


def _require_digest(actual: str, expected: Any, label: str) -> None:
    expected_value = str(expected).upper()
    if len(expected_value) != 64 or actual != expected_value:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected_value}.")


def centered_support_mask(
    open_shape: tuple[int, int], support_shape: tuple[int, int]
) -> NDArray[np.bool_]:
    """Return a centered finite support mask on an open ``(ny, nx)`` grid."""

    oy, ox = (int(value) for value in open_shape)
    sy, sx = (int(value) for value in support_shape)
    if min(oy, ox, sy, sx) <= 0 or sy > oy or sx > ox:
        raise ValueError("support_shape must be positive and fit open_shape.")
    if (oy - sy) % 2 or (ox - sx) % 2:
        raise ValueError("support and open grids must have aligned centers.")
    mask = np.zeros((oy, ox), dtype=np.bool_)
    y0 = (oy - sy) // 2
    x0 = (ox - sx) // 2
    mask[y0 : y0 + sy, x0 : x0 + sx] = True
    return mask


@dataclass(frozen=True)
class Exp042SourceBundle:
    """Audited source arrays and regenerated frozen measurement operator."""

    source_run: Path
    source_hdf5: Path
    source_config: dict[str, Any]
    operator: MatchedKnownBProbeOperator
    I_stack: NDArray[np.float64]
    scan_positions: NDArray[np.float64]
    P_B_true: NDArray[np.complex128]
    B_true: NDArray[np.complex128]
    authoritative_P_B_rec: NDArray[np.complex128]
    support_mask: NDArray[np.bool_]
    provenance: dict[str, Any]
    replay: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping at {path}.")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping at {path}.")
    return value


def load_exp042_authoritative_source(
    project_root: str | Path, source: Mapping[str, Any]
) -> Exp042SourceBundle:
    """Load, hash-audit, and independently replay the exp042 source artifact.

    Dataset paths and expected hashes come exclusively from ``source``.  The
    persisted intensity stack and scan positions are the reconstruction data;
    deterministic regeneration is used only to reconstruct and verify the
    frozen operator identity.
    """

    root = Path(project_root).resolve()
    source_run = (root / str(source["run"])).resolve()
    source_hdf5 = source_run / str(source["hdf5"])
    source_config_path = source_run / str(source["config"])
    metadata_path = source_run / str(source["metadata"])
    metrics_path = source_run / str(source["metrics"])
    state_path = source_run / str(source["run_state"])
    for path in (
        source_hdf5,
        source_config_path,
        metadata_path,
        metrics_path,
        state_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    file_hashes = {
        "hdf5": sha256_file(source_hdf5),
        "config": sha256_file(source_config_path),
        "metadata": sha256_file(metadata_path),
        "metrics": sha256_file(metrics_path),
        "run_state": sha256_file(state_path),
    }
    expected_files = source["file_sha256"]
    for key, digest in file_hashes.items():
        _require_digest(digest, expected_files[key], f"source {key}")

    metadata = _load_json(metadata_path)
    state = _load_json(state_path)
    if (
        state.get("status") != "complete"
        or state.get("artifacts_validated") is not True
    ):
        raise ValueError("The exp042 source is not complete and artifact-validated.")
    if metadata.get("experiment_id") != "exp042":
        raise ValueError("The configured source is not an exp042 artifact.")
    if metadata.get("reference_validated") is not False:
        raise ValueError("The exp042 reference boundary unexpectedly changed.")
    if metadata.get("full_tgv_reference_authorized") is not False:
        raise ValueError("The exp042 full-reference boundary unexpectedly changed.")

    paths = source["datasets"]
    arrays: dict[str, NDArray[Any]] = {}
    dataset_info: dict[str, Any] = {}
    with h5py.File(source_hdf5, "r") as h5:
        for key, path_value in paths.items():
            path = str(path_value)
            if path not in h5 or not isinstance(h5[path], h5py.Dataset):
                raise KeyError(f"Missing exp042 source dataset: {path}.")
            array = np.asarray(h5[path][...])
            arrays[key] = array
            digest = sha256_array_bytes(array)
            _require_digest(digest, source["dataset_sha256"][key], f"dataset {path}")
            dataset_info[key] = {
                "path": path,
                "sha256": digest,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }

    I_stack = np.asarray(arrays["I_stack"], dtype=np.float64)
    positions = np.asarray(arrays["scan_positions"], dtype=np.float64)
    p_true = np.asarray(arrays["P_B_true"], dtype=np.complex128)
    b_true = np.asarray(arrays["B_true"], dtype=np.complex128)
    p_authoritative = np.asarray(arrays["authoritative_P_B_rec"], dtype=np.complex128)
    if (
        I_stack.ndim != 3
        or positions.shape != (I_stack.shape[0], 2)
        or p_true.ndim != 2
        or b_true.ndim != 2
        or p_authoritative.shape != p_true.shape
    ):
        raise ValueError("The exp042 source shapes are inconsistent.")
    if not all(
        np.all(np.isfinite(value))
        for value in (I_stack, positions, p_true, b_true, p_authoritative)
    ):
        raise ValueError("The exp042 source contains non-finite values.")

    source_config = _load_yaml(source_config_path)
    generated = build_matched_development_case(source_config)
    generated_operator = generated["operator"]
    operator = replace(
        generated_operator,
        positions_m=positions.copy(),
        finite_b_modulation_open=np.asarray(b_true - 1.0, dtype=np.complex128),
    )
    regenerated_checks = {
        "P_B_true_exact": bool(np.array_equal(generated["P_B_true"], p_true)),
        "B_true_exact": bool(np.array_equal(generated["B_true"], b_true)),
        "scan_positions_exact": bool(
            np.array_equal(generated["scan_positions"], positions)
        ),
        "generated_I_stack_exact": bool(np.array_equal(generated["I_stack"], I_stack)),
    }
    replay_prediction = operator.predict_stack(p_true)
    replay_relative_l2 = relative_l2(replay_prediction, I_stack)
    replay_exact = bool(np.array_equal(replay_prediction, I_stack))
    if not all(regenerated_checks.values()) or not replay_exact:
        raise ValueError("The regenerated exp042 operator is not byte-exact.")

    support_shape = tuple(int(v) for v in source_config["sample_b"]["support_shape"])
    support_mask = centered_support_mask(operator.open_shape, support_shape)
    exterior_error = float(np.max(np.abs(b_true[~support_mask] - 1.0)))
    if exterior_error != 0.0:
        raise ValueError("The source B violates its transparent exterior.")

    provenance = {
        "source_experiment": "exp042",
        "source_run": str(source_run),
        "source_hdf5": str(source_hdf5),
        "file_sha256": file_hashes,
        "datasets": dataset_info,
        "operator_branch": metadata["operator_branch"],
        "git_commit": metadata["git_commit"],
        "reference_validated": False,
        "full_tgv_reference_authorized": False,
        "axis_order": ["y", "x"],
        "scan_position_columns": ["x", "y"],
        "scan_position_units": "m",
        "intensity_units": "arbitrary",
        "node_dx_m": float(operator.node_dx_m),
        "native_fov_m": [
            float(operator.native_shape[0] * operator.node_dx_m),
            float(operator.native_shape[1] * operator.node_dx_m),
        ],
        "open_fov_m": [
            float(operator.open_shape[0] * operator.node_dx_m),
            float(operator.open_shape[1] * operator.node_dx_m),
        ],
        "shift_convention": "constant-zero shift of B_minus_1",
        "detector_readout": "matched q4 positive pixel average",
    }
    replay = {
        **regenerated_checks,
        "independent_forward_replay_exact": replay_exact,
        "independent_forward_replay_relative_l2": replay_relative_l2,
        "transparent_exterior_max_abs_error": exterior_error,
    }
    return Exp042SourceBundle(
        source_run=source_run,
        source_hdf5=source_hdf5,
        source_config=source_config,
        operator=operator,
        I_stack=I_stack,
        scan_positions=positions,
        P_B_true=p_true,
        B_true=b_true,
        authoritative_P_B_rec=p_authoritative,
        support_mask=support_mask,
        provenance=provenance,
        replay=replay,
    )


@dataclass(frozen=True)
class MatchedBlindProbeBOperator:
    """Variable-probe/variable-finite-B form of the exp042 operator."""

    reference: MatchedKnownBProbeOperator
    support_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        mask = np.asarray(self.support_mask, dtype=np.bool_)
        if mask.shape != self.reference.open_shape or not np.any(mask):
            raise ValueError("support_mask must be nonempty and match open_shape.")

    @property
    def native_shape(self) -> tuple[int, int]:
        return self.reference.native_shape

    @property
    def open_shape(self) -> tuple[int, int]:
        return self.reference.open_shape

    @property
    def data_shape(self) -> tuple[int, int, int]:
        return (
            len(self.reference.positions_m),
            *self.reference.detector_roi_shape,
        )

    def project_modulation(self, modulation: ComplexArray) -> NDArray[np.complex128]:
        values = np.asarray(modulation, dtype=np.complex128)
        if values.shape != self.open_shape or not np.all(np.isfinite(values)):
            raise ValueError("B modulation must be finite and match open_shape.")
        return np.where(self.support_mask, values, 0.0j).astype(np.complex128)

    def probe_open(self, probe_native: ComplexArray) -> NDArray[np.complex128]:
        probe = np.asarray(probe_native, dtype=np.complex128)
        if probe.shape != self.native_shape or not np.all(np.isfinite(probe)):
            raise ValueError("probe must be finite and match native_shape.")
        delta = _center_embed(
            probe - self.reference.homogeneous_probe_native, self.open_shape
        )
        return np.asarray(self.reference.homogeneous_probe_open + delta)

    def shifted_modulation(
        self, modulation: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        values = self.project_modulation(modulation)
        return np.asarray(
            shift_field_integer_pixels(
                values,
                self.reference.positions_m[scan_index],
                self.reference.node_dx_m,
                boundary="constant",
                fill_value=0.0j,
            ),
            dtype=np.complex128,
        )

    def detector_field(
        self, probe_native: ComplexArray, modulation: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        probe_open = self.probe_open(probe_native)
        delta_probe = probe_open - self.reference.homogeneous_probe_open
        shifted = self.shifted_modulation(modulation, scan_index)
        residual_exit = delta_probe + probe_open * shifted
        return np.asarray(
            self.reference.homogeneous_detector_open
            + apply_angular_spectrum_transfer(
                residual_exit, self.reference.transfer_bc
            ),
            dtype=np.complex128,
        )

    def predict_stack(
        self, probe_native: ComplexArray, modulation: ComplexArray
    ) -> NDArray[np.float64]:
        frames = []
        for scan_index in range(len(self.reference.positions_m)):
            field = self.detector_field(probe_native, modulation, scan_index)
            pixels = self.reference._readout_intensity(field)
            frames.append(
                np.asarray(
                    center_crop(pixels, self.reference.detector_roi_shape),
                    dtype=np.float64,
                )
            )
        return np.stack(frames)

    def probe_field_direction(
        self, direction_native: ComplexArray, modulation: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        direction = np.asarray(direction_native, dtype=np.complex128)
        if direction.shape != self.native_shape or not np.all(np.isfinite(direction)):
            raise ValueError("probe direction must be finite and native-shaped.")
        embedded = _center_embed(direction, self.open_shape)
        transmission = 1.0 + self.shifted_modulation(modulation, scan_index)
        return np.asarray(
            apply_angular_spectrum_transfer(
                transmission * embedded, self.reference.transfer_bc
            ),
            dtype=np.complex128,
        )

    def b_field_direction(
        self, probe_native: ComplexArray, direction: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        direction_values = self.project_modulation(direction)
        shifted = shift_field_integer_pixels(
            direction_values,
            self.reference.positions_m[scan_index],
            self.reference.node_dx_m,
            boundary="constant",
            fill_value=0.0j,
        )
        return np.asarray(
            apply_angular_spectrum_transfer(
                self.probe_open(probe_native) * shifted,
                self.reference.transfer_bc,
            ),
            dtype=np.complex128,
        )

    def intensity_jacobian_direction(
        self,
        probe_native: ComplexArray,
        modulation: ComplexArray,
        probe_direction: ComplexArray | None = None,
        b_direction: ComplexArray | None = None,
    ) -> NDArray[np.float64]:
        if probe_direction is None and b_direction is None:
            raise ValueError("At least one Jacobian direction is required.")
        frames = []
        for scan_index in range(len(self.reference.positions_m)):
            field = self.detector_field(probe_native, modulation, scan_index)
            field_direction = np.zeros(self.open_shape, dtype=np.complex128)
            if probe_direction is not None:
                field_direction += self.probe_field_direction(
                    probe_direction, modulation, scan_index
                )
            if b_direction is not None:
                field_direction += self.b_field_direction(
                    probe_native, b_direction, scan_index
                )
            full = self.reference._readout_intensity_direction(field, field_direction)
            frames.append(
                np.asarray(
                    center_crop(full, self.reference.detector_roi_shape),
                    dtype=np.float64,
                )
            )
        return np.stack(frames)

    def intensity_jacobian_adjoint(
        self,
        probe_native: ComplexArray,
        modulation: ComplexArray,
        detector_values: FloatArray,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        residual = np.asarray(detector_values, dtype=np.float64)
        if residual.shape != self.data_shape or not np.all(np.isfinite(residual)):
            raise ValueError("detector_values must be finite and match data_shape.")
        full_pixel_shape = tuple(
            length // self.reference.quadrature_factor for length in self.open_shape
        )
        p_adjoint = np.zeros(self.native_shape, dtype=np.complex128)
        b_adjoint = np.zeros(self.open_shape, dtype=np.complex128)
        probe_open = self.probe_open(probe_native)
        for scan_index in range(len(self.reference.positions_m)):
            field = self.detector_field(probe_native, modulation, scan_index)
            residual_full = _center_embed(residual[scan_index], full_pixel_shape)
            detector_adjoint = self.reference._readout_intensity_adjoint_to_nodes(
                field, residual_full
            )
            exit_adjoint = apply_angular_spectrum_transfer(
                detector_adjoint, np.conj(self.reference.transfer_bc)
            )
            transmission = 1.0 + self.shifted_modulation(modulation, scan_index)
            p_adjoint += center_crop(
                np.conj(transmission) * exit_adjoint, self.native_shape
            )
            shifted_b_adjoint = np.conj(probe_open) * exit_adjoint
            b_adjoint += unshift_field_delta_integer_pixels(
                shifted_b_adjoint,
                self.reference.positions_m[scan_index],
                self.reference.node_dx_m,
                boundary="constant",
            )
        return (
            np.asarray(p_adjoint, dtype=np.complex128),
            self.project_modulation(b_adjoint),
        )

    def loss_and_gradients(
        self,
        probe_native: ComplexArray,
        modulation: ComplexArray,
        measured: FloatArray,
    ) -> tuple[
        float,
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.float64],
    ]:
        data = np.asarray(measured, dtype=np.float64)
        if data.shape != self.data_shape or not np.all(np.isfinite(data)):
            raise ValueError("measured must be finite and match data_shape.")
        prediction = self.predict_stack(probe_native, modulation)
        residual = prediction - data
        loss = 0.5 * float(np.mean(np.square(residual), dtype=np.float64))
        p_adjoint, b_adjoint = self.intensity_jacobian_adjoint(
            probe_native, modulation, residual
        )
        normalization = float(data.size)
        return (
            loss,
            np.asarray(p_adjoint / normalization, dtype=np.complex128),
            np.asarray(b_adjoint / normalization, dtype=np.complex128),
            prediction,
        )


def detector_amplitude_relative_residual(
    prediction: FloatArray, measured: FloatArray
) -> float:
    """Return the nonnegative square-root intensity residual relative L2."""

    predicted = np.asarray(prediction, dtype=np.float64)
    data = np.asarray(measured, dtype=np.float64)
    if predicted.shape != data.shape or np.min(predicted) < 0.0 or np.min(data) < 0.0:
        raise ValueError(
            "Amplitude residual inputs must be same-shaped and nonnegative."
        )
    return relative_l2(np.sqrt(predicted), np.sqrt(data))


def _block_shape(
    operator: MatchedBlindProbeBOperator, block: BlockName
) -> tuple[int, int]:
    return operator.native_shape if block == "probe" else operator.open_shape


def block_normal_action(
    operator: MatchedBlindProbeBOperator,
    probe: ComplexArray,
    modulation: ComplexArray,
    direction: ComplexArray,
    block: BlockName,
) -> NDArray[np.complex128]:
    """Apply the mean-normalized real block Gauss--Newton normal operator."""

    direction_values = np.asarray(direction, dtype=np.complex128)
    if direction_values.shape != _block_shape(operator, block):
        raise ValueError("block direction has the wrong shape.")
    if block == "probe":
        jacobian = operator.intensity_jacobian_direction(
            probe, modulation, probe_direction=direction_values
        )
    else:
        jacobian = operator.intensity_jacobian_direction(
            probe, modulation, b_direction=operator.project_modulation(direction_values)
        )
    p_adjoint, b_adjoint = operator.intensity_jacobian_adjoint(
        probe, modulation, jacobian
    )
    result = p_adjoint if block == "probe" else b_adjoint
    return np.asarray(result / float(np.prod(operator.data_shape)), dtype=np.complex128)


def phase_direction_to_modulation(
    operator: MatchedBlindProbeBOperator,
    modulation: ComplexArray,
    phase_direction: NDArray[np.floating] | ComplexArray,
) -> NDArray[np.complex128]:
    """Map a real active-support phase direction to a complex B direction."""

    m = operator.project_modulation(modulation)
    direction = np.asarray(phase_direction)
    if direction.shape != operator.open_shape or not np.all(np.isfinite(direction)):
        raise ValueError("phase_direction must be finite and open-shaped.")
    if np.iscomplexobj(direction) and np.max(np.abs(np.imag(direction))) > 1.0e-14:
        raise ValueError("phase_direction must be real-valued.")
    real_direction = np.where(operator.support_mask, np.real(direction), 0.0)
    return np.asarray(1j * (1.0 + m) * real_direction, dtype=np.complex128)


def modulation_adjoint_to_phase(
    operator: MatchedBlindProbeBOperator,
    modulation: ComplexArray,
    modulation_adjoint: ComplexArray,
) -> NDArray[np.complex128]:
    """Map a complex modulation adjoint to the real phase parameterization."""

    m = operator.project_modulation(modulation)
    adjoint = operator.project_modulation(modulation_adjoint)
    basis = 1j * (1.0 + m)
    values = np.real(np.conj(basis) * adjoint)
    return np.asarray(np.where(operator.support_mask, values, 0.0), dtype=np.complex128)


def phase_normal_action(
    operator: MatchedBlindProbeBOperator,
    probe: ComplexArray,
    modulation: ComplexArray,
    phase_direction: NDArray[np.floating] | ComplexArray,
) -> NDArray[np.complex128]:
    """Apply the mean-normalized GN normal action in real B-phase coordinates."""

    modulation_direction = phase_direction_to_modulation(
        operator, modulation, phase_direction
    )
    jacobian = operator.intensity_jacobian_direction(
        probe, modulation, b_direction=modulation_direction
    )
    _, modulation_adjoint = operator.intensity_jacobian_adjoint(
        probe, modulation, jacobian
    )
    phase_adjoint = modulation_adjoint_to_phase(
        operator, modulation, modulation_adjoint
    )
    return np.asarray(
        phase_adjoint / float(np.prod(operator.data_shape)),
        dtype=np.complex128,
    )


def _estimate_action_spectral_radius(
    normal_action: Callable[[ComplexArray], NDArray[np.complex128]],
    shape: tuple[int, int],
    *,
    iterations: int,
    seed: int,
    support_mask: NDArray[np.bool_] | None = None,
    real_parameter: bool = False,
) -> dict[str, Any]:
    """Estimate a positive self-adjoint action's largest eigenvalue."""

    if iterations < 1 or seed < 0:
        raise ValueError("spectral iterations/seed are invalid.")
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=shape)
    if not real_parameter:
        vector = vector + 1j * rng.normal(size=shape)
    if support_mask is not None:
        vector = np.where(support_mask, vector, 0.0)
    vector = np.asarray(vector / _l2_norm(vector), dtype=np.complex128)
    rayleigh = []
    for _ in range(iterations):
        action = normal_action(vector)
        value = _real_inner_product(vector, action)
        norm = _l2_norm(action)
        if not np.isfinite(value) or value <= 0.0 or norm <= np.finfo(float).eps:
            raise RuntimeError("Spectral-radius estimate failed.")
        rayleigh.append(float(value))
        vector = np.asarray(action / norm, dtype=np.complex128)
    return {
        "spectral_radius_estimate": rayleigh[-1],
        "rayleigh_curve": np.asarray(rayleigh, dtype=np.float64),
        "normal_operator_action_count": iterations,
        "seed": seed,
        "truth_used": False,
    }


def estimate_block_spectral_radius(
    operator: MatchedBlindProbeBOperator,
    probe: ComplexArray,
    modulation: ComplexArray,
    block: BlockName,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate one block's local normal-operator spectral radius."""

    shape = _block_shape(operator, block)
    action = partial(block_normal_action, operator, probe, modulation, block=block)
    return _estimate_action_spectral_radius(
        action,
        shape,
        iterations=iterations,
        seed=seed,
        support_mask=(operator.support_mask if block == "sample_b" else None),
    )


def block_cg_direction(
    normal_action: Callable[[ComplexArray], NDArray[np.complex128]],
    gradient: ComplexArray,
    *,
    damping: float,
    max_iterations: int,
    relative_residual_tolerance: float,
) -> dict[str, Any]:
    """Solve one damped real-complex block normal equation by CG."""

    right = np.asarray(gradient, dtype=np.complex128)
    if right.ndim != 2 or not np.all(np.isfinite(right)):
        raise ValueError("gradient must be a finite 2D complex field.")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite.")
    if max_iterations < 1 or not 0.0 <= relative_residual_tolerance < 1.0:
        raise ValueError("CG settings are invalid.")
    direction = np.zeros_like(right)
    residual = right.copy()
    search = residual.copy()
    residual_squared = _real_inner_product(residual, residual)
    initial = float(np.sqrt(max(residual_squared, 0.0)))
    residual_curve = [initial]
    curvature_curve = []
    stopping_reason = "iteration_budget"
    completed = 0
    epsilon = np.finfo(np.float64).eps
    for _ in range(max_iterations):
        action = normal_action(search) + damping * search
        curvature = _real_inner_product(search, action)
        curvature_curve.append(curvature)
        if not np.isfinite(curvature) or curvature <= epsilon:
            stopping_reason = "nonpositive_curvature"
            break
        alpha = residual_squared / curvature
        direction = np.asarray(direction + alpha * search, dtype=np.complex128)
        residual = np.asarray(residual - alpha * action, dtype=np.complex128)
        next_squared = _real_inner_product(residual, residual)
        completed += 1
        residual_curve.append(float(np.sqrt(max(next_squared, 0.0))))
        if residual_curve[-1] <= relative_residual_tolerance * max(initial, epsilon):
            stopping_reason = "relative_residual"
            residual_squared = next_squared
            break
        beta = next_squared / max(residual_squared, epsilon)
        search = np.asarray(residual + beta * search, dtype=np.complex128)
        residual_squared = next_squared
    descent = _real_inner_product(right, direction)
    if completed < 1 or not np.isfinite(descent) or descent <= 0.0:
        raise RuntimeError("Block CG did not produce a descent direction.")
    return {
        "direction": direction,
        "iterations_completed": completed,
        "stopping_reason": stopping_reason,
        "residual_l2_curve": np.asarray(residual_curve, dtype=np.float64),
        "curvature_curve": np.asarray(curvature_curve, dtype=np.float64),
        "relative_residual": residual_curve[-1] / max(initial, epsilon),
        "gradient_direction_real_inner_product": descent,
        "normal_operator_action_count": completed,
    }


def optimize_block(
    operator: MatchedBlindProbeBOperator,
    measured: FloatArray,
    probe: ComplexArray,
    modulation: ComplexArray,
    block: BlockName,
    settings: Mapping[str, Any],
    *,
    spectral_seed: int,
) -> dict[str, Any]:
    """Apply a bounded number of spectrally damped GN-CG block steps."""

    for key in (
        "truth_used_by_optimizer",
        "truth_used_by_spectral_radius",
        "truth_used_by_damping",
        "truth_used_by_stopping",
    ):
        if settings.get(key) is not False:
            raise ValueError(f"Truth boundary violated at {key}.")
    outer_iterations = int(settings["outer_iterations"])
    power_iterations = int(settings["spectral_radius_power_iterations"])
    cg_iterations = int(settings["cg_max_iterations"])
    damping_relative = float(settings["damping_relative_to_spectral_radius"])
    cg_tolerance = float(settings["cg_relative_residual_tolerance"])
    backtracking_factor = float(settings["backtracking_factor"])
    armijo_c = float(settings["armijo_c"])
    max_backtracking = int(settings["max_backtracking_steps"])
    minimum_step = float(settings["minimum_step"])
    line_search_scale = float(settings["line_search_initial_scale"])
    gradient_stop = float(settings["gradient_norm_stop"])
    b_parameterization = str(
        settings.get("parameterization", "complex_active_transmission")
    )
    if block == "sample_b" and b_parameterization not in {
        "complex_active_transmission",
        "phase_only_unit_modulus_on_active_support",
    }:
        raise ValueError("Unsupported sample-B block parameterization.")
    if min(outer_iterations, power_iterations, cg_iterations, max_backtracking) < 1:
        raise ValueError("All block budgets must be positive.")
    if not 0.0 < backtracking_factor < 1.0 or not 0.0 < armijo_c < 1.0:
        raise ValueError("Armijo settings are invalid.")
    if min(damping_relative, minimum_step, line_search_scale) <= 0.0:
        raise ValueError("Damping and line-search scales must be positive.")

    p = np.asarray(probe, dtype=np.complex128).copy()
    m = operator.project_modulation(modulation)
    data = np.asarray(measured, dtype=np.float64)
    loss, p_gradient, b_gradient, prediction = operator.loss_and_gradients(p, m, data)
    phase_only_b = (
        block == "sample_b"
        and b_parameterization == "phase_only_unit_modulus_on_active_support"
    )
    if block == "probe":
        gradient = p_gradient
    elif phase_only_b:
        gradient = modulation_adjoint_to_phase(operator, m, b_gradient)
    else:
        gradient = b_gradient
    if phase_only_b:
        spectral_action = partial(phase_normal_action, operator, p, m)
        spectral = _estimate_action_spectral_radius(
            spectral_action,
            operator.open_shape,
            iterations=power_iterations,
            seed=spectral_seed,
            support_mask=operator.support_mask,
            real_parameter=True,
        )
    else:
        spectral = estimate_block_spectral_radius(
            operator,
            p,
            m,
            block,
            iterations=power_iterations,
            seed=spectral_seed,
        )
    radius = float(spectral["spectral_radius_estimate"])
    damping = damping_relative * radius
    losses = [loss]
    residuals = [detector_relative_residual(prediction, data)]
    amplitude_residuals = [detector_amplitude_relative_residual(prediction, data)]
    gradient_norms = [_l2_norm(gradient)]
    accepted_steps = [0.0]
    backtracking_counts = [0]
    cg_iterations_curve = []
    cg_relative_residual_curve = []
    cg_min_curvature_curve = []
    update_relative_curve = [0.0]
    normal_actions = int(spectral["normal_operator_action_count"])
    loss_gradient_evaluations = 1
    stopping_reason = "iteration_budget"

    for _ in range(outer_iterations):
        gradient_norm = _l2_norm(gradient)
        if gradient_norm <= max(gradient_stop, np.finfo(float).eps):
            stopping_reason = "gradient_norm"
            break

        if phase_only_b:
            action = partial(phase_normal_action, operator, p, m)
        else:
            action = partial(block_normal_action, operator, p, m, block=block)

        solve = block_cg_direction(
            action,
            gradient,
            damping=damping,
            max_iterations=cg_iterations,
            relative_residual_tolerance=cg_tolerance,
        )
        direction = np.asarray(solve["direction"], dtype=np.complex128)
        slope = -float(solve["gradient_direction_real_inner_product"])
        step = line_search_scale
        accepted = False
        candidate_p = p
        candidate_m = m
        candidate = (loss, p_gradient, b_gradient, prediction)
        backtracks = 0
        for attempt in range(max_backtracking):
            backtracks = attempt
            if block == "probe":
                candidate_p = np.asarray(p - step * direction, dtype=np.complex128)
                candidate_m = m
            else:
                candidate_p = p
                if phase_only_b:
                    phase_step = np.real(direction)
                    transmission = (1.0 + m) * np.exp(-1j * step * phase_step)
                    candidate_m = operator.project_modulation(transmission - 1.0)
                else:
                    candidate_m = operator.project_modulation(m - step * direction)
            candidate = operator.loss_and_gradients(candidate_p, candidate_m, data)
            loss_gradient_evaluations += 1
            if candidate[0] <= loss + armijo_c * step * slope:
                accepted = True
                break
            step *= backtracking_factor
            if step < minimum_step:
                break
        normal_actions += int(solve["normal_operator_action_count"])
        if not accepted:
            stopping_reason = "line_search_failed"
            break
        old_value = p if block == "probe" else m
        new_value = candidate_p if block == "probe" else candidate_m
        if block == "probe":
            update_denominator = _l2_norm(old_value)
        else:
            update_denominator = _l2_norm(1.0 + old_value)
        update_relative_curve.append(
            _l2_norm(new_value - old_value)
            / max(update_denominator, np.finfo(float).eps)
        )
        p = candidate_p
        m = candidate_m
        loss, p_gradient, b_gradient, prediction = candidate
        if block == "probe":
            gradient = p_gradient
        elif phase_only_b:
            gradient = modulation_adjoint_to_phase(operator, m, b_gradient)
        else:
            gradient = b_gradient
        losses.append(float(loss))
        residuals.append(detector_relative_residual(prediction, data))
        amplitude_residuals.append(
            detector_amplitude_relative_residual(prediction, data)
        )
        gradient_norms.append(_l2_norm(gradient))
        accepted_steps.append(step)
        backtracking_counts.append(backtracks)
        cg_iterations_curve.append(int(solve["iterations_completed"]))
        cg_relative_residual_curve.append(float(solve["relative_residual"]))
        curvatures = np.asarray(solve["curvature_curve"], dtype=np.float64)
        cg_min_curvature_curve.append(float(np.min(curvatures)))

    return {
        "probe": p,
        "modulation": m,
        "prediction_final": prediction,
        "loss_curve": np.asarray(losses, dtype=np.float64),
        "detector_relative_residual_curve": np.asarray(residuals, dtype=np.float64),
        "detector_amplitude_relative_residual_curve": np.asarray(
            amplitude_residuals, dtype=np.float64
        ),
        "gradient_l2_norm_curve": np.asarray(gradient_norms, dtype=np.float64),
        "accepted_step_curve": np.asarray(accepted_steps, dtype=np.float64),
        "backtracking_count_curve": np.asarray(backtracking_counts, dtype=np.int64),
        "update_relative_l2_curve": np.asarray(update_relative_curve, dtype=np.float64),
        "spectral_radius_estimate": radius,
        "spectral_radius_rayleigh_curve": spectral["rayleigh_curve"],
        "damping": damping,
        "cg_iterations_curve": np.asarray(cg_iterations_curve, dtype=np.int64),
        "cg_relative_residual_curve": np.asarray(
            cg_relative_residual_curve, dtype=np.float64
        ),
        "cg_min_curvature_curve": np.asarray(cg_min_curvature_curve, dtype=np.float64),
        "iterations_completed": len(losses) - 1,
        "stopping_reason": stopping_reason,
        "normal_operator_action_count": normal_actions,
        "loss_gradient_evaluation_count": loss_gradient_evaluations,
        "operator_action_budget_units": normal_actions + loss_gradient_evaluations,
        "truth_used_by_optimizer": False,
        "sample_b_parameterization": b_parameterization,
    }


def make_b_initialization(
    support_mask: NDArray[np.bool_], settings: Mapping[str, Any]
) -> NDArray[np.complex128]:
    """Create a deterministic truth-free finite-B modulation initialization."""

    mask = np.asarray(support_mask, dtype=np.bool_)
    kind = str(settings["kind"])
    modulation = np.zeros(mask.shape, dtype=np.complex128)
    if kind == "homogeneous_transmission":
        return modulation
    if kind != "seeded_zero_mean_phase":
        raise ValueError("Unsupported blind B initialization.")
    seed = int(settings["seed"])
    phase_rms = float(settings["phase_rms_rad"])
    if seed < 0 or not np.isfinite(phase_rms) or not 0.0 < phase_rms < np.pi:
        raise ValueError("Invalid seeded B initialization settings.")
    rng = np.random.default_rng(seed)
    phase = rng.normal(size=int(np.count_nonzero(mask)))
    phase -= np.mean(phase)
    phase *= phase_rms / float(np.sqrt(np.mean(np.square(phase))))
    modulation[mask] = np.exp(1j * phase) - 1.0
    return modulation


def reciprocal_exit_product_gauge_error(
    probe_open: ComplexArray, transmission: ComplexArray, factor: complex
) -> float:
    """Return algebraic product error under ``P->cP, B->B/c``."""

    p = np.asarray(probe_open, dtype=np.complex128)
    b = np.asarray(transmission, dtype=np.complex128)
    c = complex(factor)
    if p.shape != b.shape or not np.isfinite(c) or abs(c) <= np.finfo(float).eps:
        raise ValueError("Gauge inputs are invalid.")
    return relative_l2((c * p) * (b / c), p * b)


def measurement_only_canonical_pair(
    probe: ComplexArray, modulation: ComplexArray
) -> dict[str, Any]:
    """Return the identity canonical copy required by the pinned exp042 model.

    The frozen homogeneous field outside the native unknown-probe window and
    transparent B exterior make a non-unit reciprocal complex scaling
    unrepresentable.  Consequently exp043 does not alter the raw primary pair;
    it records this physical-reference convention explicitly instead.
    """

    return {
        "P_B_canonical": np.asarray(probe, dtype=np.complex128).copy(),
        "B_canonical": np.asarray(1.0 + modulation, dtype=np.complex128).copy(),
        "complex_factor": np.complex128(1.0 + 0.0j),
        "convention": (
            "identity_under_frozen_homogeneous_reference_and_transparent_exterior"
        ),
        "prediction_invariant_by_construction": True,
        "uses_simulation_truth": False,
    }


def alternating_blind_reconstruction(
    operator: MatchedBlindProbeBOperator,
    measured: FloatArray,
    init_probe: ComplexArray,
    init_modulation: ComplexArray,
    settings: Mapping[str, Any],
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Alternate finite P and B block GN-CG sweeps with a fixed final rule."""

    if settings.get("algorithm") != "alternating_block_spectrally_damped_gn_cg":
        raise ValueError("Unsupported exp043 primary algorithm.")
    if settings.get("checkpoint_selection_rule") != "fixed_final_outer_sweep":
        raise ValueError("The blind checkpoint rule must be fixed-final.")
    for key in (
        "truth_used_by_initialization",
        "truth_used_by_optimizer",
        "truth_used_by_checkpoint_selection",
        "truth_used_by_stopping",
    ):
        if settings.get(key) is not False:
            raise ValueError(f"Blind truth boundary violated at {key}.")
    sweeps = int(settings["outer_sweeps"])
    if sweeps < 1:
        raise ValueError("outer_sweeps must be positive.")
    p = np.asarray(init_probe, dtype=np.complex128).copy()
    m = operator.project_modulation(init_modulation)
    initial_prediction = operator.predict_stack(p, m)
    losses = [0.5 * float(np.mean(np.square(initial_prediction - measured)))]
    residuals = [detector_relative_residual(initial_prediction, measured)]
    amplitude_residuals = [
        detector_amplitude_relative_residual(initial_prediction, measured)
    ]
    p_history = [p.copy()]
    b_history = [(1.0 + m).copy()]
    p_updates = [0.0]
    b_updates = [0.0]
    gauge_probe_norm = [_l2_norm(p)]
    gauge_b_rms = [
        float(np.sqrt(np.mean(np.abs((1.0 + m)[operator.support_mask]) ** 2)))
    ]
    gauge_b_mean_phase = [float(np.angle(np.mean((1.0 + m)[operator.support_mask])))]
    block_records: list[dict[str, Any]] = []
    actions = 0
    stopping_reason = "outer_sweep_budget"
    prediction = initial_prediction

    for sweep in range(sweeps):
        p_before = p.copy()
        p_result = optimize_block(
            operator,
            measured,
            p,
            m,
            "probe",
            settings["probe_block"],
            spectral_seed=int(settings["probe_block"]["spectral_radius_seed"])
            + seed_offset
            + sweep,
        )
        p = p_result["probe"]
        actions += int(p_result["operator_action_budget_units"])
        b_before = m.copy()
        b_result = optimize_block(
            operator,
            measured,
            p,
            m,
            "sample_b",
            settings["sample_b_block"],
            spectral_seed=int(settings["sample_b_block"]["spectral_radius_seed"])
            + seed_offset
            + sweep,
        )
        m = b_result["modulation"]
        prediction = b_result["prediction_final"]
        actions += int(b_result["operator_action_budget_units"])
        loss = float(b_result["loss_curve"][-1])
        losses.append(loss)
        residuals.append(detector_relative_residual(prediction, measured))
        amplitude_residuals.append(
            detector_amplitude_relative_residual(prediction, measured)
        )
        p_updates.append(_l2_norm(p - p_before) / max(_l2_norm(p_before), 1e-30))
        b_updates.append(_l2_norm(m - b_before) / max(_l2_norm(1.0 + b_before), 1e-30))
        p_history.append(p.copy())
        b_history.append((1.0 + m).copy())
        gauge_probe_norm.append(_l2_norm(p))
        gauge_b_rms.append(
            float(np.sqrt(np.mean(np.abs((1.0 + m)[operator.support_mask]) ** 2)))
        )
        gauge_b_mean_phase.append(
            float(np.angle(np.mean((1.0 + m)[operator.support_mask])))
        )
        block_records.append(
            {"outer_sweep": sweep + 1, "probe": p_result, "sample_b": b_result}
        )
        if p_result["stopping_reason"] == "line_search_failed":
            stopping_reason = "probe_line_search_failed"
            break
        if b_result["stopping_reason"] == "line_search_failed":
            stopping_reason = "sample_b_line_search_failed"
            break

    canonical = measurement_only_canonical_pair(p, m)
    return {
        "P_B_init_raw": np.asarray(init_probe, dtype=np.complex128).copy(),
        "B_init_raw": np.asarray(1.0 + init_modulation, dtype=np.complex128),
        "P_B_rec_raw": p,
        "B_rec_raw": np.asarray(1.0 + m, dtype=np.complex128),
        "P_B_rec_canonical": canonical["P_B_canonical"],
        "B_rec_canonical": canonical["B_canonical"],
        "prediction_final": prediction,
        "loss_curve": np.asarray(losses, dtype=np.float64),
        "detector_relative_residual_curve": np.asarray(residuals, dtype=np.float64),
        "detector_amplitude_relative_residual_curve": np.asarray(
            amplitude_residuals, dtype=np.float64
        ),
        "probe_update_relative_l2_curve": np.asarray(p_updates, dtype=np.float64),
        "sample_b_update_relative_l2_curve": np.asarray(b_updates, dtype=np.float64),
        "gauge_probe_l2_norm_curve": np.asarray(gauge_probe_norm, dtype=np.float64),
        "gauge_b_active_rms_amplitude_curve": np.asarray(gauge_b_rms, dtype=np.float64),
        "gauge_b_active_mean_phase_curve": np.asarray(
            gauge_b_mean_phase, dtype=np.float64
        ),
        "P_B_checkpoint_history": np.stack(p_history),
        "B_checkpoint_history": np.stack(b_history),
        "block_records": block_records,
        "outer_sweeps_completed": len(losses) - 1,
        "stopping_reason": stopping_reason,
        "operator_action_budget_units": actions,
        "checkpoint_selection_rule": "fixed_final_outer_sweep",
        "canonicalization": canonical,
        "truth_used_by_initialization": False,
        "truth_used_by_optimizer": False,
        "truth_used_by_checkpoint_selection": False,
        "truth_used_by_stopping": False,
    }


def align_pair_simulation_only(
    probe: ComplexArray,
    transmission: ComplexArray,
    probe_true: ComplexArray,
    transmission_true: ComplexArray,
    support_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Apply a post-freeze truth-derived reciprocal gain for diagnostics only."""

    p = np.asarray(probe, dtype=np.complex128)
    b = np.asarray(transmission, dtype=np.complex128)
    p_true = np.asarray(probe_true, dtype=np.complex128)
    b_true = np.asarray(transmission_true, dtype=np.complex128)
    mask = np.asarray(support_mask, dtype=np.bool_)
    denominator = np.sum(np.abs(p) ** 2, dtype=np.float64)
    if denominator <= np.finfo(float).eps:
        raise ValueError("Cannot align a zero recovered probe.")
    factor = np.sum(np.conj(p) * p_true, dtype=np.complex128) / denominator
    if abs(factor) <= np.finfo(float).eps:
        raise ValueError("Truth-derived diagnostic gain is singular.")
    p_aligned = np.asarray(factor * p, dtype=np.complex128)
    b_aligned = b.copy()
    b_aligned[mask] = b[mask] / factor
    return {
        "P_B_rec_reciprocal_gain_aligned": p_aligned,
        "B_rec_reciprocal_gain_aligned": b_aligned,
        "complex_factor": np.complex128(factor),
        "probe_raw_relative_l2": relative_l2(p, p_true),
        "probe_aligned_relative_l2": relative_l2(p_aligned, p_true),
        "B_raw_active_relative_l2": relative_l2(b[mask], b_true[mask]),
        "B_aligned_active_relative_l2": relative_l2(b_aligned[mask], b_true[mask]),
        "simulation_evaluation_only": True,
        "enters_optimizer": False,
        "enters_selection": False,
        "enters_stopping": False,
    }


def exit_wave_product_relative_error(
    operator: MatchedBlindProbeBOperator,
    probe: ComplexArray,
    modulation: ComplexArray,
    probe_true: ComplexArray,
    modulation_true: ComplexArray,
) -> float:
    """Return the full scan-stack B-exit product error."""

    probe_open = operator.probe_open(probe)
    probe_true_open = operator.probe_open(probe_true)
    recovered = []
    truth = []
    for scan_index in range(len(operator.reference.positions_m)):
        recovered.append(
            probe_open * (1.0 + operator.shifted_modulation(modulation, scan_index))
        )
        truth.append(
            probe_true_open
            * (1.0 + operator.shifted_modulation(modulation_true, scan_index))
        )
    return relative_l2(np.stack(recovered), np.stack(truth))


def simulation_evaluation_only(
    operator: MatchedBlindProbeBOperator,
    result: Mapping[str, Any],
    source: Exp042SourceBundle,
) -> dict[str, Any]:
    """Compute truth-aware metrics after a raw result has been frozen."""

    p = np.asarray(result["P_B_rec_raw"], dtype=np.complex128)
    b = np.asarray(result["B_rec_raw"], dtype=np.complex128)
    aligned = align_pair_simulation_only(
        p,
        b,
        source.P_B_true,
        source.B_true,
        source.support_mask,
    )
    aligned["exit_wave_product_relative_l2"] = exit_wave_product_relative_error(
        operator,
        p,
        b - 1.0,
        source.P_B_true,
        source.B_true - 1.0,
    )
    aligned["prediction_relative_l2"] = relative_l2(
        result["prediction_final"], source.I_stack
    )
    return aligned


def known_b_probe_control(source: Exp042SourceBundle) -> dict[str, Any]:
    """Re-run the exact exp042 GN-CG known-B branch and compare raw bytes."""

    reconstruction = source.source_config["reconstruction"]
    control = reconstruction["exp053_feedback_control"]
    result = reconstruct_known_b_probe_damped_gn_cg(
        source.operator,
        source.I_stack,
        source.operator.homogeneous_probe_native,
        reconstruction,
        control,
    )
    raw = np.asarray(result["P_B_rec"], dtype=np.complex128)
    result["authoritative_raw_relative_l2"] = relative_l2(
        raw, source.authoritative_P_B_rec
    )
    result["authoritative_raw_exact"] = bool(
        np.array_equal(raw, source.authoritative_P_B_rec)
    )
    result["authoritative_raw_sha256"] = sha256_array_bytes(raw)
    result["source_authoritative_raw_sha256"] = sha256_array_bytes(
        source.authoritative_P_B_rec
    )
    return result


def known_probe_b_control(
    operator: MatchedBlindProbeBOperator,
    source: Exp042SourceBundle,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover B with fixed true probe as a simulation diagnostic control."""

    init = make_b_initialization(source.support_mask, settings["initialization"])
    result = optimize_block(
        operator,
        source.I_stack,
        source.P_B_true,
        init,
        "sample_b",
        settings["optimizer"],
        spectral_seed=int(settings["optimizer"]["spectral_radius_seed"]),
    )
    result["B_init_raw"] = np.asarray(1.0 + init, dtype=np.complex128)
    result["B_rec_raw"] = np.asarray(1.0 + result["modulation"], dtype=np.complex128)
    result["P_B_fixed_simulation_diagnostic_only"] = source.P_B_true.copy()
    result["B_active_relative_l2_simulation_evaluation_only"] = relative_l2(
        result["B_rec_raw"][source.support_mask],
        source.B_true[source.support_mask],
    )
    result["truth_used_by_optimizer"] = False
    result["true_probe_role"] = "simulation_diagnostic_only_fixed_input"
    return result


def operator_control_metrics(
    operator: MatchedBlindProbeBOperator,
    source: Exp042SourceBundle,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Run paired P/B/combined derivative and normal-operator controls."""

    seed = int(settings["random_seed"])
    step_relative = float(settings["finite_difference_relative_step"])
    rng = np.random.default_rng(seed)
    p = source.P_B_true
    m = source.B_true - 1.0
    p_direction = rng.normal(size=p.shape) + 1j * rng.normal(size=p.shape)
    p_direction = np.asarray(p_direction / _l2_norm(p_direction), dtype=np.complex128)
    b_direction = rng.normal(size=m.shape) + 1j * rng.normal(size=m.shape)
    b_direction = operator.project_modulation(b_direction)
    b_direction = np.asarray(b_direction / _l2_norm(b_direction), dtype=np.complex128)
    scale = max(_l2_norm(m), 1.0)
    step = step_relative * scale
    finite_b = (
        operator.predict_stack(p, m + step * b_direction)
        - operator.predict_stack(p, m - step * b_direction)
    ) / (2.0 * step)
    analytic_b = operator.intensity_jacobian_direction(p, m, b_direction=b_direction)
    b_directional = relative_l2(finite_b, analytic_b)

    residual = rng.normal(size=operator.data_shape)
    _, b_adjoint = operator.intensity_jacobian_adjoint(p, m, residual)
    left = float(np.sum(analytic_b * residual, dtype=np.float64))
    right = _real_inner_product(b_direction, b_adjoint)
    b_adjoint_error = abs(left - right) / max(abs(left), abs(right), 1e-30)

    loss, p_gradient, b_gradient, _ = operator.loss_and_gradients(
        p + 0.01 * p_direction, m + 0.01 * b_direction, source.I_stack
    )
    evaluation_p = p + 0.01 * p_direction
    evaluation_m = m + 0.01 * b_direction
    combined_step = step_relative * max(
        _l2_norm(evaluation_p), _l2_norm(evaluation_m), 1.0
    )
    loss_plus = operator.loss_and_gradients(
        evaluation_p + combined_step * p_direction,
        evaluation_m + combined_step * b_direction,
        source.I_stack,
    )[0]
    loss_minus = operator.loss_and_gradients(
        evaluation_p - combined_step * p_direction,
        evaluation_m - combined_step * b_direction,
        source.I_stack,
    )[0]
    finite_gradient = (loss_plus - loss_minus) / (2.0 * combined_step)
    analytic_gradient = _real_inner_product(
        p_gradient, p_direction
    ) + _real_inner_product(b_gradient, b_direction)
    combined_gradient_error = abs(finite_gradient - analytic_gradient) / max(
        abs(finite_gradient), abs(analytic_gradient), 1e-30
    )

    b_second = rng.normal(size=m.shape) + 1j * rng.normal(size=m.shape)
    b_second = operator.project_modulation(b_second)
    b_second = np.asarray(b_second / _l2_norm(b_second), dtype=np.complex128)
    normal_first = block_normal_action(
        operator, evaluation_p, evaluation_m, b_direction, "sample_b"
    )
    normal_second = block_normal_action(
        operator, evaluation_p, evaluation_m, b_second, "sample_b"
    )
    symmetry_left = _real_inner_product(b_second, normal_first)
    symmetry_right = _real_inner_product(b_direction, normal_second)
    symmetry_error = abs(symmetry_left - symmetry_right) / max(
        abs(symmetry_left), abs(symmetry_right), 1e-30
    )
    psd = _real_inner_product(b_direction, normal_first)

    phase_direction = rng.normal(size=m.shape)
    phase_direction = np.where(operator.support_mask, phase_direction, 0.0)
    phase_direction = np.asarray(
        phase_direction / _l2_norm(phase_direction), dtype=np.complex128
    )
    phase_step = step_relative * max(_l2_norm(np.angle(1.0 + m)), 1.0)
    b_transmission = 1.0 + m
    m_phase_plus = operator.project_modulation(
        b_transmission * np.exp(1j * phase_step * np.real(phase_direction)) - 1.0
    )
    m_phase_minus = operator.project_modulation(
        b_transmission * np.exp(-1j * phase_step * np.real(phase_direction)) - 1.0
    )
    finite_phase = (
        operator.predict_stack(p, m_phase_plus)
        - operator.predict_stack(p, m_phase_minus)
    ) / (2.0 * phase_step)
    analytic_phase = operator.intensity_jacobian_direction(
        p,
        m,
        b_direction=phase_direction_to_modulation(operator, m, phase_direction),
    )
    phase_directional_error = relative_l2(finite_phase, analytic_phase)
    _, phase_modulation_adjoint = operator.intensity_jacobian_adjoint(p, m, residual)
    phase_adjoint = modulation_adjoint_to_phase(operator, m, phase_modulation_adjoint)
    phase_left = float(np.sum(analytic_phase * residual, dtype=np.float64))
    phase_right = _real_inner_product(phase_direction, phase_adjoint)
    phase_adjoint_error = abs(phase_left - phase_right) / max(
        abs(phase_left), abs(phase_right), 1e-30
    )
    phase_second = rng.normal(size=m.shape)
    phase_second = np.where(operator.support_mask, phase_second, 0.0)
    phase_second = np.asarray(
        phase_second / _l2_norm(phase_second), dtype=np.complex128
    )
    phase_normal_first = phase_normal_action(
        operator, evaluation_p, evaluation_m, phase_direction
    )
    phase_normal_second = phase_normal_action(
        operator, evaluation_p, evaluation_m, phase_second
    )
    phase_symmetry_left = _real_inner_product(phase_second, phase_normal_first)
    phase_symmetry_right = _real_inner_product(phase_direction, phase_normal_second)
    phase_symmetry_error = abs(phase_symmetry_left - phase_symmetry_right) / max(
        abs(phase_symmetry_left), abs(phase_symmetry_right), 1e-30
    )
    phase_psd = _real_inner_product(phase_direction, phase_normal_first)

    known_controls = operator_consistency_metrics(
        {
            "operator": source.operator,
            "I_stack": source.I_stack,
            "P_B_true": source.P_B_true,
            "B_true": source.B_true,
        },
        source.source_config,
    )
    gauge_factor = complex(
        settings["gauge_test_factor_real"], settings["gauge_test_factor_imag"]
    )
    algebraic_gauge_error = reciprocal_exit_product_gauge_error(
        operator.probe_open(source.P_B_true), source.B_true, gauge_factor
    )

    scaled_probe = gauge_factor * source.P_B_true
    scaled_b = source.B_true.copy()
    scaled_b[source.support_mask] /= gauge_factor
    representable_prediction = operator.predict_stack(scaled_probe, scaled_b - 1.0)
    model_gauge_change = relative_l2(representable_prediction, source.I_stack)
    return {
        "exp042_known_b_operator": known_controls,
        "B_intensity_jacobian_directional_relative_error": b_directional,
        "B_intensity_jacobian_real_adjoint_relative_error": b_adjoint_error,
        "combined_loss_directional_gradient_relative_error": combined_gradient_error,
        "B_normal_action_symmetry_relative_error": symmetry_error,
        "B_normal_action_psd_quadratic_form": psd,
        "B_phase_jacobian_directional_relative_error": phase_directional_error,
        "B_phase_jacobian_real_adjoint_relative_error": phase_adjoint_error,
        "B_phase_normal_action_symmetry_relative_error": phase_symmetry_error,
        "B_phase_normal_action_psd_quadratic_form": phase_psd,
        "support_projection_exterior_max_abs": float(
            np.max(np.abs(operator.project_modulation(m)[~source.support_mask]))
        ),
        "truth_loss": float(operator.loss_and_gradients(p, m, source.I_stack)[0]),
        "algebraic_reciprocal_product_gauge_relative_error": algebraic_gauge_error,
        "model_representable_reciprocal_scale_prediction_change": model_gauge_change,
        "gauge_interpretation": (
            "algebraic_factorization_gauge_is_exact_but_nonunit_scale_is_pinned_"
            "by_the_frozen_homogeneous_probe_reference_and_transparent_B_exterior"
        ),
        "evaluation_loss_for_gradient_check": loss,
        "simulation_truth_used_for_operator_controls_only": True,
    }


def pairwise_blind_stability(
    branches: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare fixed-final blind branches without reading simulation truth."""

    names = list(branches)
    count = len(names)
    prediction = np.zeros((count, count), dtype=np.float64)
    probe = np.zeros((count, count), dtype=np.float64)
    for row, left_name in enumerate(names):
        left = branches[left_name]
        for column, right_name in enumerate(names):
            right = branches[right_name]
            prediction[row, column] = relative_l2(
                left["prediction_final"], right["prediction_final"]
            )
            left_probe = np.asarray(left["P_B_rec_raw"], dtype=np.complex128)
            right_probe = np.asarray(right["P_B_rec_raw"], dtype=np.complex128)
            gain = np.sum(np.conj(right_probe) * left_probe) / max(
                np.sum(np.abs(right_probe) ** 2), 1e-30
            )
            probe[row, column] = relative_l2(gain * right_probe, left_probe)
    upper = np.triu_indices(count, 1)
    return {
        "branch_names": names,
        "pairwise_prediction_relative_l2": prediction,
        "pairwise_probe_complex_gain_aligned_relative_l2": probe,
        "maximum_pairwise_prediction_relative_l2": float(
            np.max(prediction[upper]) if count > 1 else 0.0
        ),
        "maximum_pairwise_probe_aligned_relative_l2": float(
            np.max(probe[upper]) if count > 1 else 0.0
        ),
        "truth_used": False,
    }


def determine_status(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, str]:
    """Apply the frozen fail-closed exp043 status matrix."""

    source_ok = bool(metrics["source_operator_gates_passed"])
    p_ok = bool(metrics["known_b_probe_control_gates_passed"])
    b_ok = bool(metrics["known_probe_B_control_gates_passed"])
    if not source_ok:
        return {
            "scientific_status": "Inconclusive",
            "reason": "source_or_operator_not_closed",
        }
    if not p_ok or not b_ok:
        return {
            "scientific_status": "Inconclusive",
            "reason": "block_control_not_closed",
        }
    blind = metrics["blind_primary"]
    component_identifiable = (
        blind["representative_probe_aligned_relative_l2_simulation_only"]
        <= float(thresholds["blind_probe_aligned_relative_l2_max"])
        and blind["representative_B_aligned_active_relative_l2_simulation_only"]
        <= float(thresholds["blind_B_aligned_active_relative_l2_max"])
        and blind["representative_exit_wave_product_relative_l2_simulation_only"]
        <= float(thresholds["blind_exit_wave_product_relative_l2_max"])
    )
    measurement_ok = (
        blind["maximum_branch_detector_relative_residual"]
        <= float(thresholds["blind_detector_relative_residual_max"])
        and blind["maximum_pairwise_prediction_relative_l2"]
        <= float(thresholds["repeat_prediction_relative_l2_max"])
        and blind["all_loss_nonincreasing"]
        and blind["all_finite"]
        and blind["all_fixed_final_selection"]
    )
    if measurement_ok and component_identifiable:
        return {
            "scientific_status": "Passed",
            "reason": "all_preregistered_gates_passed",
        }
    if measurement_ok and not component_identifiable:
        return {
            "scientific_status": "Failed",
            "reason": "measurement_consistent_but_component_recovery_non_identifiable",
        }
    return {
        "scientific_status": "Failed",
        "reason": "blind_convergence_or_repeatability_gate_failed",
    }
