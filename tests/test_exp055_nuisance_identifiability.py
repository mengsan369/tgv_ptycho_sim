from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp055_nuisance_identifiability as runner

from tgv_ptycho.inverse import exp055
from tgv_ptycho.inverse.exp051 import Exp051SourceArtifact, sha256_file
from tgv_ptycho.inverse.exp055 import (
    connected_components_3d,
    load_exp055_true_probe,
    make_exp055_candidate_generator,
    run_exp055_formal,
    run_exp055_preflight,
    validate_exp055_config,
)
from tgv_ptycho.io.config import load_config, save_config

CONFIG_PATH = Path(
    "configs/experiments/"
    "exp055_TGV_3d_multislice_nuisance_identifiability.yaml"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registered() -> tuple[dict[str, Any], Exp051SourceArtifact]:
    config = load_config(CONFIG_PATH)
    validate_exp055_config(config)
    source = load_exp055_true_probe(config, PROJECT_ROOT)
    return config, source


def _synthetic_generator(
    target: np.ndarray,
) -> exp055.CandidateGenerator:
    basis = np.zeros((3, *target.shape), dtype=np.complex128)
    basis[0].flat[0] = 1.0
    basis[1].flat[1] = 1.0j
    basis[2].flat[2] = 1.0 + 1.0j
    scales = np.asarray([1.25e-7, 1.25e-7, 5.0e-7])
    nominal = np.asarray([2.0e-5, 3.0e-5, 5.0e-5])

    def generate(
        d_waist_m: float, d_surface_m: float, z_waist_m: float
    ) -> np.ndarray:
        parameters = np.asarray([d_waist_m, d_surface_m, z_waist_m])
        coefficients = (parameters - nominal) / scales
        field = target + 1.0e-3 * np.tensordot(coefficients, basis, axes=1)
        return np.asarray(field, dtype=np.complex128)

    return generate


def test_config_source_identity_and_real_exact_replay(
    registered: tuple[dict[str, Any], Exp051SourceArtifact],
) -> None:
    config, source = registered
    assert source.source_state["status"] == "complete"
    assert source.source_state["artifacts_validated"] is True
    assert source.target_hdf5_path == "/entry/truth/P_B_true"
    assert source.target_dataset_sha256 == (
        "FA61264AF0D96BF3393EC461E2147992FDE6926D0132B2346090790E40E8EBFD"
    )
    generator = make_exp055_candidate_generator(source.source_config)
    replay = generator(2.0e-5, 3.0e-5, 5.0e-5)
    assert np.array_equal(replay, source.P_B_true)

    invalid = deepcopy(config)
    invalid["source"]["target_hdf5_path"] = (
        "/entry/reconstruction/P_B_rec"
    )
    with pytest.raises(ValueError, match="raw /entry/truth/P_B_true"):
        validate_exp055_config(invalid)
    invalid = deepcopy(config)
    invalid["parameters"]["nuisance"].append(
        {"name": "n_glass", "bounds": [1.49, 1.51]}
    )
    with pytest.raises(ValueError, match="minimal parameter set"):
        validate_exp055_config(invalid)


def test_connected_components_six_neighbor_topology() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.bool_)
    mask[1, 1, 1] = True
    mask[1, 1, 2] = True
    mask[3, 3, 3] = True
    result = connected_components_3d(mask)
    assert result["count"] == 2
    assert sorted(len(component) for component in result["components"]) == [1, 2]
    assert result["labels"][1, 1, 1] == result["labels"][1, 1, 2]
    assert result["labels"][3, 3, 3] != result["labels"][1, 1, 1]


def test_synthetic_preflight_is_correctness_only(
    registered: tuple[dict[str, Any], Exp051SourceArtifact],
) -> None:
    config, _ = registered
    target = np.ones((4, 4), dtype=np.complex128)
    result = run_exp055_preflight(
        config, target, _synthetic_generator(target)
    )
    assert result["preflight_status"] == "PreflightPassed"
    assert result["scientific_status"] == "NotEvaluated"
    assert result["replay_relative_l2"] == 0.0
    assert result["deterministic_repeat_relative_l2"] == 0.0
    assert result["cache"]["entry_count"] == 7


def test_synthetic_joint_profile_multistart_and_status_order(
    monkeypatch: pytest.MonkeyPatch,
    registered: tuple[dict[str, Any], Exp051SourceArtifact],
) -> None:
    config, _ = registered
    target = np.ones((4, 4), dtype=np.complex128)

    def valid_fiber(
        *_args: Any,
        seed_m: tuple[float, float, float],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "seed_m": np.asarray(seed_m),
            "reported_interval": {
                "lower_m": 1.9999e-5,
                "upper_m": 2.0001e-5,
                "width_m": 2.0e-9,
                "lower_closed": True,
                "upper_closed": False,
            },
            "outside_to_tau_ratio": np.asarray([1.0e7, 1.0e7]),
            "valid": True,
        }

    monkeypatch.setattr(exp055, "_q8_fiber", valid_fiber)
    result = run_exp055_formal(
        config, {}, target, _synthetic_generator(target)
    )
    assert result["experiment_status"] == "Passed"
    assert result["equivalence"]["global_screened_component_count"] == 1
    assert result["multistart"]["all_terminal_common_component"] is True
    assert result["equivalence"]["projected_d_waist_interval"][
        "width_m"
    ] == pytest.approx(2.0e-9)
    assert result["local_diagnostic"]["enters_status_gate"] is False
    assert np.all(
        np.isfinite(
            result["local_diagnostic"]["normalized_singular_values"]
        )
    )


def test_tiny_runner_hdf5_raw_identity_and_artifact_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered: tuple[dict[str, Any], Exp051SourceArtifact],
) -> None:
    config, source = registered
    local_config = deepcopy(config)
    local_config["output"]["root"] = "runs"
    local_config["output"]["run_name"] = "tiny_exp055_preflight"
    config_path = tmp_path / "exp055.yaml"
    save_config(config_path, local_config)

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_git_commit", lambda *_: "TEST")
    monkeypatch.setattr(runner, "load_exp055_true_probe", lambda *_: source)
    monkeypatch.setattr(
        runner,
        "make_exp055_candidate_generator",
        lambda *_args, **_kwargs: _synthetic_generator(source.P_B_true),
    )
    monkeypatch.setattr(
        runner, "FROZEN_ROOT_CONFIG_SHA256", sha256_file(config_path)
    )
    run_dir = runner.run(config_path, mode="preflight")
    state = json.loads(
        (run_dir / "run_state.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert state["preflight_status"] == "PreflightPassed"
    assert state["scientific_status"] == "NotEvaluated"
    assert state["figure_count"] == 5
    assert metrics["scientific_status"] == "NotEvaluated"
    hdf5_path = run_dir / "outputs" / local_config["output"]["hdf5_filename"]
    with h5py.File(hdf5_path, "r") as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "reconstruction",
            "sample",
            "truth",
        }
        assert len(h5["entry/data"]) == 0
        assert "calibration" not in h5["entry"]
        assert "preprocessing" not in h5["entry"]
        base = (
            "entry/reconstruction/waist_fit/nuisance_identifiability/"
            "true_probe"
        )
        raw = h5[f"{base}/source/P_B_true_raw_input"][...]
        best = h5[f"{base}/result/P_B_best_raw"][...]
        residual = h5[f"{base}/result/residual_field_raw"][...]
        assert np.array_equal(raw, source.P_B_true)
        assert np.array_equal(best - raw, residual)
        names: list[str] = []
        h5["entry"].visit(names.append)
        assert not any("aligned" in name.lower() for name in names)
        assert not any(name.endswith("I_stack") for name in names)
