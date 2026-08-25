from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp051_true_probe_waist_fit as runner

from tgv_ptycho.inverse.exp051 import (
    Exp051SourceArtifact,
    load_exp051_source_true_probe,
    make_exp051_candidate_generator,
    validate_exp051_config,
)
from tgv_ptycho.inverse.waist_fit import (
    extract_profile_minimum,
    finite_difference_controls,
    fit_waist_from_probe,
    raw_complex_probe_loss,
    replay_error_metrics,
)
from tgv_ptycho.io.config import load_config, save_config

CONFIG_PATH = Path(
    "configs/experiments/exp051_TGV_3d_multislice_true_probe_waist_fit.yaml"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registered_source() -> tuple[dict[str, Any], Exp051SourceArtifact]:
    config = load_config(CONFIG_PATH)
    return config, load_exp051_source_true_probe(config, PROJECT_ROOT)


@pytest.fixture(scope="module")
def exact_replay(
    registered_source: tuple[dict[str, Any], Exp051SourceArtifact],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config, source = registered_source
    generator = make_exp051_candidate_generator(source.source_config)
    true_waist = float(config["fit"]["true_d_waist_m"])
    return source.P_B_true, generator(true_waist), generator(true_waist)


def _linear_generator(target: np.ndarray, true_waist_m: float):
    yy, xx = np.indices(target.shape, dtype=np.float64)
    direction = (5.0e4 * target) * (
        1.0 + 0.1j * (xx - yy) / max(target.shape)
    )

    def generate(d_waist_m: float) -> np.ndarray:
        if not np.isfinite(d_waist_m) or d_waist_m <= 0.0:
            raise ValueError("D_waist must be finite and positive.")
        return np.asarray(
            target + (float(d_waist_m) - true_waist_m) * direction,
            dtype=np.complex128,
        )

    return generate


def _run_fit(config: dict[str, Any], target: np.ndarray) -> dict[str, Any]:
    fit = config["fit"]
    true_waist = float(fit["true_d_waist_m"])
    return fit_waist_from_probe(
        target,
        _linear_generator(target, true_waist),
        true_waist_m=true_waist,
        bounds_m=tuple(float(value) for value in fit["bounds_m"]),
        coarse_grid_m=fit["coarse_grid_m"],
        fine_half_width_m=float(fit["fine_half_width_m"]),
        fine_step_m=float(fit["fine_step_m"]),
        finite_difference_steps_m=tuple(
            float(value) for value in fit["finite_difference_steps_m"]
        ),
        optimizer_starts_m=fit["optimizer"]["starts_m"],
        optimizer_initial_step_m=float(fit["optimizer"]["initial_step_m"]),
        optimizer_evaluation_budget=int(
            fit["optimizer"]["evaluation_budget_per_start"]
        ),
        thresholds=config["thresholds"],
    )


def test_config_source_identity_and_forbidden_reconstruction_input(
    registered_source: tuple[dict[str, Any], Exp051SourceArtifact],
) -> None:
    config, source = registered_source
    validate_exp051_config(config)
    assert source.target_hdf5_path == "/entry/truth/P_B_true"
    assert "reconstruction" not in source.target_hdf5_path
    assert "P_B_rec" not in source.target_hdf5_path
    assert source.P_B_true.shape == (96, 96)
    assert source.P_B_true.dtype == np.complex128
    assert source.D_z_m.shape == (100,)
    assert source.z_m.shape == (100,)
    assert source.slice_widths_m.shape == (100,)
    assert np.isclose(np.sum(source.slice_widths_m), 1.0e-4)
    assert source.source_state["status"] == "complete"
    assert source.source_state["artifacts_validated"] is True
    assert source.source_metadata["reference_validated"] is False
    assert source.source_metadata["full_tgv_reference_authorized"] is False

    forbidden = deepcopy(config)
    forbidden["source"]["target_hdf5_path"] = (
        "/entry/reconstruction/detector_quadrature_ablation/"
        "branches/matched_q4/P_B_rec"
    )
    with pytest.raises(ValueError, match="P_B_true"):
        validate_exp051_config(forbidden)

    promoted = deepcopy(config)
    promoted["experiment"]["reference_validated"] = True
    with pytest.raises(ValueError, match="must remain false"):
        validate_exp051_config(promoted)


def test_true_waist_exact_replay_shape_dtype_axis_and_determinism(
    exact_replay: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    target, replay, repeat = exact_replay
    metrics = replay_error_metrics(replay, target)
    assert replay.shape == target.shape == (96, 96)
    assert replay.dtype == repeat.dtype == target.dtype == np.complex128
    assert metrics["raw_complex_relative_l2"] == 0.0
    assert metrics["amplitude_relative_l2"] == 0.0
    assert (
        metrics["amplitude_weighted_phase_sensitive_relative_l2"] == 0.0
    )
    assert raw_complex_probe_loss(repeat, replay) == 0.0


def test_raw_loss_has_fixed_scale_and_phase_reference() -> None:
    target = np.ones((8, 10), dtype=np.complex128)
    assert raw_complex_probe_loss(target, target) == 0.0
    assert raw_complex_probe_loss(2.0 * target, target) == pytest.approx(1.0)
    phase = 0.3
    expected = abs(np.exp(1j * phase) - 1.0) ** 2
    assert raw_complex_probe_loss(
        np.exp(1j * phase) * target, target
    ) == pytest.approx(expected)
    phase_metrics = replay_error_metrics(np.exp(1j * phase) * target, target)
    assert phase_metrics["amplitude_relative_l2"] < 1.0e-15
    assert phase_metrics[
        "amplitude_weighted_phase_sensitive_relative_l2"
    ] > 0.0
    with pytest.raises(ValueError, match="same shape"):
        raw_complex_probe_loss(target[:, :-1], target)


def test_profile_minimum_and_finite_difference_consistency() -> None:
    profile = extract_profile_minimum(
        [18.0e-6, 19.0e-6, 20.0e-6, 21.0e-6],
        [4.0, 1.0, 0.0, 1.0],
        uniqueness_tolerance=1.0e-20,
    )
    assert profile["unique"] is True
    assert profile["minimum_index"] == 2
    target = np.ones((6, 7), dtype=np.complex128)
    true_waist = 20.0e-6
    generator = _linear_generator(target, true_waist)
    h1, h2 = 0.25e-6, 0.125e-6
    controls = finite_difference_controls(
        target,
        generator(true_waist),
        generator(true_waist - h1),
        generator(true_waist + h1),
        generator(true_waist - h2),
        generator(true_waist + h2),
        h1_m=h1,
        h2_m=h2,
    )
    assert controls["normalized_jacobian_h1_per_m"] > 1.0e4
    assert controls["loss_curvature_per_m2"] > 0.0
    assert controls["jacobian_step_relative_l2"] < 1.0e-12


def test_fit_bounds_profile_and_equal_budget_multistart(
    registered_source: tuple[dict[str, Any], Exp051SourceArtifact],
) -> None:
    config, source = registered_source
    result = _run_fit(config, source.P_B_true)
    assert result["status"] == "Passed"
    assert result["stage_a_replay_pass"] is True
    assert result["profile"]["pass"] is True
    assert result["profile"]["minimum"]["unique"] is True
    assert result["profile"]["minimum"]["minimum_d_waist_m"] == pytest.approx(
        20.0e-6
    )
    assert result["finite_difference"]["pass"] is True
    optimizer = result["optimizer"]
    assert optimizer["equal_budget"] is True
    assert optimizer["pass"] is True
    assert len(optimizer["branches"]) == 4
    for branch in optimizer["branches"].values():
        assert branch["evaluation_count"] == 41
        assert branch["stopping_reason"] == "evaluation_budget"
        assert len(branch["evaluated_d_waist_m"]) == 41
        assert branch["boundary_hit"] is False
        assert branch["pass"] is True

    bad_bounds = deepcopy(config)
    bad_bounds["fit"]["bounds_m"] = [15.0e-6, 24.0e-6]
    with pytest.raises(ValueError, match="bounds/grid"):
        validate_exp051_config(bad_bounds)
    generator = make_exp051_candidate_generator(source.source_config)
    with pytest.raises(ValueError, match="positive"):
        generator(-1.0e-6)


def test_runner_hdf5_layout_provenance_and_no_p_b_rec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered_source: tuple[dict[str, Any], Exp051SourceArtifact],
) -> None:
    config, source = registered_source
    local_config = deepcopy(config)
    local_config["output"]["root"] = "runs"
    local_config["output"]["run_name"] = "tiny_exp051"
    config_path = tmp_path / "exp051.yaml"
    save_config(config_path, local_config)
    true_waist = float(config["fit"]["true_d_waist_m"])
    generator = _linear_generator(source.P_B_true, true_waist)

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "load_exp051_source_true_probe",
        lambda *_: source,
    )
    monkeypatch.setattr(
        runner,
        "make_exp051_candidate_generator",
        lambda *_args, **_kwargs: generator,
    )
    run_dir = runner.run(config_path)
    with (run_dir / "run_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert state["figure_count"] == 4
    assert metrics["experiment_status"] == "Passed"
    assert metrics["reference_validated"] is False
    assert metrics["full_tgv_reference_authorized"] is False
    assert metrics["p_b_rec_used_as_primary_input"] is False
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
        assert "calibration" not in h5["entry"]
        assert "preprocessing" not in h5["entry"]
        names: list[str] = []
        h5["entry"].visit(names.append)
        assert not any(name.rsplit("/", 1)[-1] == "P_B_rec" for name in names)
        assert h5["entry/truth/P_B_true"].shape == (96, 96)
        assert h5["entry/truth/P_B_true"].dtype == np.complex128
        assert h5[
            "entry/reconstruction/waist_fit/cache/P_B_candidate"
        ].shape[1:] == (96, 96)
        assert h5[
            "entry/reconstruction/waist_fit/finite_difference/jacobian_h1"
        ].shape == (96, 96)
        branches = h5["entry/reconstruction/waist_fit/optimizer/branches"]
        assert len(branches) == 4
        assert all(
            int(branch["evaluation_count"][()]) == 41
            for branch in branches.values()
        )
        assert bool(h5["entry/truth/reference_validated"][()]) is False
        assert bool(
            h5["entry/truth/full_tgv_reference_authorized"][()]
        ) is False
