from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp051_local_differentiability_control as runner

from tgv_ptycho.inverse.exp051 import (
    Exp051SourceArtifact,
    load_exp051_source_true_probe,
    make_exp051_candidate_generator,
)
from tgv_ptycho.inverse.exp051_local_control import (
    PriorExp051Artifact,
    air_fraction_change_metrics,
    central_field_step_series,
    load_prior_exp051_formal,
    make_air_fraction_stack,
    q8_waist_breakpoint_map,
    run_local_differentiability_control,
    validate_exp051_local_control_config,
)
from tgv_ptycho.io.config import load_config, save_config
from tgv_ptycho.objects.tgv3d import (
    make_tgv_air_fraction_slice,
    make_tgv_air_fraction_slice_chord_quadrature,
)

CONFIG_PATH = Path(
    "configs/experiments/"
    "exp051_TGV_3d_multislice_local_differentiability_control.yaml"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registered() -> tuple[
    dict[str, Any], Exp051SourceArtifact, PriorExp051Artifact
]:
    config = load_config(CONFIG_PATH)
    validate_exp051_local_control_config(config)
    source = load_exp051_source_true_probe(config, PROJECT_ROOT)
    prior = load_prior_exp051_formal(config, PROJECT_ROOT)
    return config, source, prior


@pytest.fixture(scope="module")
def local_result(
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorExp051Artifact
    ],
) -> dict[str, Any]:
    config, source, _ = registered
    q8_generator = make_exp051_candidate_generator(source.source_config)
    chord_generator = make_exp051_candidate_generator(
        source.source_config,
        air_fraction_builder=make_tgv_air_fraction_slice_chord_quadrature,
        interface_resolution=64,
    )
    return run_local_differentiability_control(
        config,
        source.source_config,
        source.P_B_true,
        q8_generator,
        chord_generator,
    )


def test_local_config_prior_and_forbidden_provenance(
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorExp051Artifact
    ],
) -> None:
    config, source, prior = registered
    assert source.target_hdf5_path == "/entry/truth/P_B_true"
    assert "P_B_rec" not in source.target_hdf5_path
    assert prior.run_state["status"] == "complete"
    assert prior.run_state["artifacts_validated"] is True
    assert prior.metrics["experiment_status"] == "Inconclusive"
    assert prior.metrics["reference_validated"] is False
    assert prior.metrics["full_tgv_reference_authorized"] is False

    changed = deepcopy(config)
    changed["local_control"]["step_family_m"][-1] = 2.0e-9
    with pytest.raises(ValueError, match="step family"):
        validate_exp051_local_control_config(changed)
    promoted = deepcopy(config)
    promoted["experiment"]["reference_validated"] = True
    with pytest.raises(ValueError, match="must remain false"):
        validate_exp051_local_control_config(promoted)


def test_q8_breakpoints_predict_inside_and_crossing_geometry(
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorExp051Artifact
    ],
) -> None:
    config, source, _ = registered
    true_waist = float(config["fit"]["true_d_waist_m"])
    breakpoint = q8_waist_breakpoint_map(
        source.source_config,
        bounds_m=tuple(config["fit"]["bounds_m"]),
        true_waist_m=true_waist,
        q=8,
    )
    assert breakpoint["lower_gap_m"] > 0.0
    assert breakpoint["upper_gap_m"] > 0.0
    assert breakpoint["equality_at_truth_multiplicity"] == 0
    assert np.all(np.diff(breakpoint["breakpoints_m"]) > 0.0)

    truth, _, widths = make_air_fraction_stack(
        source.source_config,
        true_waist,
        builder=make_tgv_air_fraction_slice,
        resolution=8,
    )
    dx_m = float(source.source_config["probe_grid"]["node_dx_m"])
    for direction, gap in (
        (-1.0, breakpoint["lower_gap_m"]),
        (1.0, breakpoint["upper_gap_m"]),
    ):
        inside, _, _ = make_air_fraction_stack(
            source.source_config,
            true_waist + direction * 0.5 * gap,
            builder=make_tgv_air_fraction_slice,
            resolution=8,
        )
        crossing, _, _ = make_air_fraction_stack(
            source.source_config,
            true_waist + direction * 1.5 * gap,
            builder=make_tgv_air_fraction_slice,
            resolution=8,
        )
        inside_metrics = air_fraction_change_metrics(
            inside, truth, widths_m=widths, dx_m=dx_m, q=8
        )
        crossing_metrics = air_fraction_change_metrics(
            crossing, truth, widths_m=widths, dx_m=dx_m, q=8
        )
        assert inside_metrics["bitwise_equal"] is True
        assert crossing_metrics["changed_subpixel_node_count"] > 0


def test_default_q8_generator_replays_and_chord_injection_is_deterministic(
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorExp051Artifact
    ],
) -> None:
    config, source, _ = registered
    true_waist = float(config["fit"]["true_d_waist_m"])
    q8 = make_exp051_candidate_generator(source.source_config)
    replay = q8(true_waist)
    np.testing.assert_array_equal(replay, source.P_B_true)
    chord = make_exp051_candidate_generator(
        source.source_config,
        air_fraction_builder=make_tgv_air_fraction_slice_chord_quadrature,
        interface_resolution=64,
    )
    first = chord(true_waist)
    second = chord(true_waist)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (96, 96)
    assert first.dtype == np.complex128


def test_central_series_recovers_a_smooth_synthetic_direction() -> None:
    center_diameter = 20.0e-6
    center = np.ones((7, 9), dtype=np.complex128)
    yy, xx = np.indices(center.shape)
    direction = (2.0e4 + 3.0e4j) * (1.0 + 0.01 * (xx + yy))

    def generator(diameter_m: float) -> np.ndarray:
        return center + (diameter_m - center_diameter) * direction

    series = central_field_step_series(
        generator,
        center_diameter_m=center_diameter,
        steps_m=np.asarray([0.5e-6, 0.25e-6, 0.125e-6]),
    )
    assert series["cache"]["P_B_candidate"].shape == (7, 7, 9)
    assert np.max(series["pairwise_jacobian_relative_l2"]) < 1.0e-12
    assert np.all(series["normalized_jacobian_per_m"] > 1.0e4)
    assert not np.any(series["zero_pair"])


def test_local_result_and_runner_hdf5_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered: tuple[
        dict[str, Any], Exp051SourceArtifact, PriorExp051Artifact
    ],
    local_result: dict[str, Any],
) -> None:
    config, source, prior = registered
    assert local_result["diagnostic_status"] in {
        "DiagnosticPassed",
        "DiagnosticFailed",
        "DiagnosticInconclusive",
    }
    assert local_result["experiment_status"] == "Inconclusive"
    assert local_result["reference_validated"] is False
    assert local_result["full_tgv_reference_authorized"] is False
    assert local_result["q8"]["series"]["jacobian"].shape == (8, 96, 96)
    assert local_result["q8"]["series"]["cache"][
        "P_B_candidate"
    ].shape == (17, 96, 96)

    local_config = deepcopy(config)
    local_config["output"]["root"] = "runs"
    local_config["output"]["run_name"] = "tiny_exp051_local_control"
    config_path = tmp_path / "local_control.yaml"
    save_config(config_path, local_config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_git_commit", lambda *_: "TEST")
    monkeypatch.setattr(
        runner, "load_exp051_source_true_probe", lambda *_: source
    )
    monkeypatch.setattr(runner, "load_prior_exp051_formal", lambda *_: prior)
    monkeypatch.setattr(
        runner,
        "make_exp051_candidate_generator",
        lambda *_args, **_kwargs: lambda _: source.P_B_true,
    )
    monkeypatch.setattr(
        runner,
        "run_local_differentiability_control",
        lambda *_args, **_kwargs: local_result,
    )
    run_dir = runner.run(config_path)
    state = json.loads(
        (run_dir / "run_state.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert metrics["experiment_status"] == "Inconclusive"
    assert metrics["reference_validated"] is False
    hdf5_path = (
        run_dir / "outputs" / local_config["output"]["hdf5_filename"]
    )
    with h5py.File(hdf5_path, "r") as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        assert "reconstruction" not in h5["entry"]
        assert "calibration" not in h5["entry"]
        assert "preprocessing" not in h5["entry"]
        names: list[str] = []
        h5["entry"].visit(names.append)
        assert not any(name.rsplit("/", 1)[-1] == "P_B_rec" for name in names)
        assert h5[
            "entry/data/local_differentiability/q8/series/jacobian"
        ].shape == (8, 96, 96)
        assert bool(h5["entry/truth/reference_validated"][()]) is False
        assert bool(
            h5["entry/truth/full_tgv_reference_authorized"][()]
        ) is False
