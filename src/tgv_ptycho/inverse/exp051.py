"""Artifact handoff and candidate generation for exp051/exp053."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.exp040 import build_scalar_working_model_probe
from tgv_ptycho.io.config import load_config
from tgv_ptycho.objects.tgv3d import make_tgv_air_fraction_slice


@dataclass(frozen=True)
class Exp051SourceArtifact:
    """Validated matched raw true-probe handoff from the registered exp042 run."""

    run_dir: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    run_state_path: Path
    hdf5_path: Path
    target_hdf5_path: str
    source_config: dict[str, Any]
    source_metadata: dict[str, Any]
    source_metrics: dict[str, Any]
    source_state: dict[str, Any]
    file_sha256: dict[str, str]
    target_dataset_sha256: str
    P_B_true: NDArray[np.complex128]
    D_z_m: NDArray[np.float64]
    z_m: NDArray[np.float64]
    slice_widths_m: NDArray[np.float64]
    embedded_config_yaml: str


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_array_bytes(values: NDArray[np.generic]) -> str:
    """Return an uppercase SHA256 of one C-contiguous array byte stream."""

    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest().upper()


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_exp051_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen exp051 true-probe single-parameter contract."""

    if config.get("experiment", {}).get("id") != "exp051":
        raise ValueError("exp051 config must declare experiment.id=exp051.")
    experiment = config["experiment"]
    if (
        experiment.get("reference_validated") is not False
        or experiment.get("full_tgv_reference_authorized") is not False
    ):
        raise ValueError("exp051 provenance flags must remain false.")
    source = config["source"]
    target_path = str(source["target_hdf5_path"])
    if target_path != "/entry/truth/P_B_true":
        raise ValueError("exp051 primary input must be /entry/truth/P_B_true.")
    lowered = target_path.lower()
    for token in source["forbidden_primary_input_tokens"]:
        if str(token).lower() in lowered:
            raise ValueError("exp051 primary input may not reference reconstruction.")
    identity = source["expected_identity"]
    if (
        identity.get("plane") != "B"
        or identity.get("axis_order") != ["y", "x"]
        or identity.get("native_shape") != [96, 96]
        or identity.get("dtype") != "complex128"
        or float(identity.get("node_dx_m")) != 5.0e-7
    ):
        raise ValueError("exp051 target plane/grid identity changed.")
    fixed = config["fixed_model"]
    if (
        fixed.get("interface_factor") != 8
        or fixed.get("shape") != [96, 96]
        or fixed.get("complex_dtype") != "complex128"
        or fixed.get("internal_alias_control") is not False
        or fixed.get("external_alias_control") is not True
    ):
        raise ValueError("exp051 frozen scalar working-model identity changed.")
    fit = config["fit"]
    if (
        fit.get("primary_loss") != "raw_complex_normalized_squared_l2"
        or fit.get("mask") != "full_native_field_all_true"
        or fit.get("phase_or_scale_alignment") != "none"
    ):
        raise ValueError("exp051 primary loss/reference convention changed.")
    bounds = np.asarray(fit["bounds_m"], dtype=np.float64)
    coarse = np.asarray(fit["coarse_grid_m"], dtype=np.float64)
    starts = np.asarray(fit["optimizer"]["starts_m"], dtype=np.float64)
    if (
        bounds.shape != (2,)
        or not np.array_equal(bounds, np.asarray([1.6e-5, 2.4e-5]))
        or not np.allclose(
            coarse,
            np.arange(16.0, 25.0, dtype=np.float64) * 1.0e-6,
            rtol=0.0,
            atol=1.0e-20,
        )
        or not np.allclose(
            starts,
            np.asarray([16.5, 18.5, 21.5, 23.5]) * 1.0e-6,
            rtol=0.0,
            atol=1.0e-20,
        )
        or int(fit["optimizer"]["evaluation_budget_per_start"]) != 41
    ):
        raise ValueError("exp051 registered bounds/grid/multi-start changed.")


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"Missing registered exp051 source {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected).upper():
        raise RuntimeError(f"Registered exp051 source {label} hash mismatch.")
    return actual


def load_exp051_source_true_probe(
    config: Mapping[str, Any], project_root: Path
) -> Exp051SourceArtifact:
    """Load and strictly validate the registered raw ``P_B_true`` artifact.

    This function never reads any ``/entry/reconstruction`` dataset.
    """

    validate_exp051_config(config)
    source = config["source"]
    run_dir = Path(source["run"])
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    run_dir = run_dir.resolve()
    config_path = run_dir / source["config_filename"]
    metadata_path = run_dir / source["metadata_filename"]
    metrics_path = run_dir / source["metrics_filename"]
    state_path = run_dir / source["run_state_filename"]
    hdf5_path = run_dir / source["hdf5_relative_path"]
    expected = source["expected_sha256"]
    file_sha = {
        "config": _require_hash(config_path, expected["config"], "config"),
        "metadata": _require_hash(
            metadata_path, expected["metadata"], "metadata"
        ),
        "metrics": _require_hash(metrics_path, expected["metrics"], "metrics"),
        "run_state": _require_hash(
            state_path, expected["run_state"], "run_state"
        ),
        "hdf5": _require_hash(hdf5_path, expected["hdf5"], "HDF5"),
    }
    source_config = load_config(config_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    with metrics_path.open("r", encoding="utf-8") as handle:
        source_metrics = json.load(handle)
    with state_path.open("r", encoding="utf-8") as handle:
        source_state = json.load(handle)
    if (
        source_state.get("status") != "complete"
        or source_state.get("artifacts_validated") is not True
    ):
        raise RuntimeError("Registered exp042 source run is not complete/validated.")
    if (
        source_metadata.get("reference_validated") is not False
        or source_metadata.get("full_tgv_reference_authorized") is not False
        or source_config["provenance"].get("reference_validated") is not False
        or source_config["provenance"].get("full_tgv_reference_authorized")
        is not False
    ):
        raise RuntimeError("Registered source promoted a forbidden provenance flag.")

    target_path = str(source["target_hdf5_path"])
    with h5py.File(hdf5_path, "r") as h5:
        required = [
            target_path,
            "/entry/truth/D_z_m",
            "/entry/truth/z_m",
            "/entry/truth/slice_widths_m",
            "/entry/truth/reference_validated",
            "/entry/truth/full_tgv_reference_authorized",
            "/entry/instrument/probe_grid/plane",
            "/entry/instrument/probe_grid/axis_order",
            "/entry/instrument/probe_grid/native_shape",
            "/entry/instrument/probe_grid/node_dx_m",
            "/entry/instrument/wavelength_m",
            "/entry/instrument/internal_reference_index",
            "/entry/instrument/external_medium_index",
            "/entry/instrument/z_AB_m",
            "/entry/sample/sample_a/d_waist_m",
            "/entry/config_yaml",
        ]
        missing = [path for path in required if path not in h5]
        if missing:
            raise RuntimeError(f"Registered source HDF5 is missing: {missing}")
        target = np.asarray(h5[target_path][...])
        d_z_m = np.asarray(h5["/entry/truth/D_z_m"][...], dtype=np.float64)
        z_m = np.asarray(h5["/entry/truth/z_m"][...], dtype=np.float64)
        widths = np.asarray(
            h5["/entry/truth/slice_widths_m"][...], dtype=np.float64
        )
        embedded_config = _decode_scalar(h5["/entry/config_yaml"][()])
        plane = _decode_scalar(h5["/entry/instrument/probe_grid/plane"][()])
        axis = [
            _decode_scalar(value)
            for value in h5["/entry/instrument/probe_grid/axis_order"][...]
        ]
        shape = h5["/entry/instrument/probe_grid/native_shape"][...].tolist()
        node_dx = float(h5["/entry/instrument/probe_grid/node_dx_m"][()])
        hdf_flags = (
            bool(h5["/entry/truth/reference_validated"][()]),
            bool(h5["/entry/truth/full_tgv_reference_authorized"][()]),
        )
        hdf_waist = float(h5["/entry/sample/sample_a/d_waist_m"][()])
        instrument_values = {
            "wavelength_m": float(h5["/entry/instrument/wavelength_m"][()]),
            "internal_reference_index": float(
                h5["/entry/instrument/internal_reference_index"][()]
            ),
            "external_medium_index": float(
                h5["/entry/instrument/external_medium_index"][()]
            ),
            "z_AB_m": float(h5["/entry/instrument/z_AB_m"][()]),
        }
    identity = source["expected_identity"]
    if (
        target.shape != tuple(identity["native_shape"])
        or target.dtype != np.complex128
        or plane != identity["plane"]
        or axis != identity["axis_order"]
        or shape != identity["native_shape"]
        or node_dx != float(identity["node_dx_m"])
        or hdf_flags != (False, False)
        or hdf_waist != float(config["fit"]["true_d_waist_m"])
    ):
        raise RuntimeError(
            "Registered source target plane/grid/truth identity differs."
        )
    fixed = config["fixed_model"]
    for key, actual in instrument_values.items():
        if actual != float(fixed[key]):
            raise RuntimeError(f"Registered source instrument differs at {key}.")
    dataset_sha = sha256_array_bytes(target)
    if dataset_sha != str(expected["target_dataset_bytes"]).upper():
        raise RuntimeError("Registered source target dataset byte hash mismatch.")
    normalized_embedded = embedded_config.replace("\r\n", "\n")
    normalized_external = config_path.read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    )
    if normalized_embedded != normalized_external:
        raise RuntimeError("Source HDF5 config_yaml differs from source config.yaml.")
    if (
        not np.all(np.isfinite(target))
        or not np.all(np.isfinite(d_z_m))
        or not np.all(np.isfinite(z_m))
        or not np.all(np.isfinite(widths))
    ):
        raise RuntimeError("Registered source truth contains non-finite values.")
    return Exp051SourceArtifact(
        run_dir=run_dir,
        config_path=config_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        run_state_path=state_path,
        hdf5_path=hdf5_path,
        target_hdf5_path=target_path,
        source_config=source_config,
        source_metadata=source_metadata,
        source_metrics=source_metrics,
        source_state=source_state,
        file_sha256=file_sha,
        target_dataset_sha256=dataset_sha,
        P_B_true=np.asarray(target, dtype=np.complex128),
        D_z_m=d_z_m,
        z_m=z_m,
        slice_widths_m=widths,
        embedded_config_yaml=str(embedded_config),
    )


def make_exp051_candidate_generator(
    source_config: Mapping[str, Any],
    *,
    memory_callback: Callable[[], None] | None = None,
    air_fraction_builder: Callable[..., NDArray[np.float64]] = (
        make_tgv_air_fraction_slice
    ),
    interface_resolution: int | None = None,
) -> Callable[[float], NDArray[np.complex128]]:
    """Return a generator that changes only ``sample_a.d_waist_m``.

    The optional interface arguments are for preregistered numerical controls.
    Defaults preserve the matched q8 source operator exactly.
    """

    frozen_source = deepcopy(dict(source_config))

    def generate(d_waist_m: float) -> NDArray[np.complex128]:
        if not np.isfinite(d_waist_m) or d_waist_m <= 0.0:
            raise ValueError("D_waist must be finite and positive.")
        candidate_config = deepcopy(frozen_source)
        candidate_config["sample_a"]["d_waist_m"] = float(d_waist_m)
        generated = build_scalar_working_model_probe(
            candidate_config,
            air_fraction_builder=air_fraction_builder,
            interface_resolution=interface_resolution,
        )
        if memory_callback is not None:
            memory_callback()
        return np.asarray(generated["P_B"], dtype=np.complex128)

    return generate
