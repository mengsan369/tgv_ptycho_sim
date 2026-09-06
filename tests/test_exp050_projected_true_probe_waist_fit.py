from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp050_projected_true_probe_waist_fit as runner

from tgv_ptycho.forward.exp030 import (
    build_exp030_radial_operator,
    make_exp030_projected_probe,
)
from tgv_ptycho.inverse.exp050 import (
    ProbeCache,
    fixed_budget_golden_section_search,
    load_exp030_source,
    make_exp050_candidate_generator,
    run_exp050_estimator,
    sha256_array,
    validate_exp050_config,
)
from tgv_ptycho.io.config import load_config, save_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/exp050_waist_parametric_fit.yaml"


def _synthetic_generator(true_waist_m: float, shape: tuple[int, int] = (8, 10)):
    yy, xx = np.indices(shape, dtype=np.float64)
    direction = (0.4 + yy / shape[0]) + 1j * (0.2 + xx / shape[1])
    radial_direction = np.linspace(0.2, 1.0, 13) * (1.0 + 0.3j)
    target = np.ones(shape, dtype=np.complex128)
    radial_target = np.ones(13, dtype=np.complex128)

    def generate(diameter_m: float) -> dict[str, np.ndarray]:
        delta = (float(diameter_m) - true_waist_m) / true_waist_m
        return {
            "P_B": (target + delta * direction).astype(np.complex128),
            "P_B_radial": (
                radial_target + delta * radial_direction
            ).astype(np.complex128),
        }

    return target, radial_target, generate


def _synthetic_design(true_waist_m: float) -> dict[str, Any]:
    return {
        "bounds_m": [true_waist_m - 1e-6, true_waist_m + 1e-6],
        "global_step_m": 0.5e-6,
        "fine_half_width_m": 0.5e-6,
        "fine_step_m": 0.1e-6,
        "finite_difference_steps_m": [0.2e-6, 0.1e-6],
        "golden_iterations": 8,
        "thresholds": {
            "replay_raw_relative_l2_max": 1e-12,
            "replay_amplitude_relative_l2_max": 1e-12,
            "replay_phase_sensitive_relative_l2_max": 1e-12,
            "radial_replay_relative_l2_max": 1e-12,
            "deterministic_repeat_relative_l2_max": 1e-14,
            "profile_uniqueness_loss_floor": 1e-20,
            "profile_absolute_error_m_max": 0.1e-6,
            "profile_minimum_loss_max": 1e-20,
            "profile_second_best_to_floor_ratio_min": 1e3,
            "normalized_jacobian_per_m_min": 1e3,
            "jacobian_step_relative_l2_max": 1e-10,
            "signature_to_floor_ratio_min": 1e3,
            "golden_final_bracket_width_m_max": 0.1e-6,
            "golden_absolute_error_m_max": 0.1e-6,
            "method_agreement_m_max": 0.1e-6,
            "boundary_margin_m": 0.1e-6,
        },
    }


def test_config_rejects_reconstructed_probe_and_invalid_bounds() -> None:
    config = load_config(CONFIG_PATH)
    forbidden = deepcopy(config)
    forbidden["source"]["target_dataset"] = (
        "/entry/reconstruction/cases/baseline/P_B_rec"
    )
    with pytest.raises(ValueError, match="P_B_true"):
        validate_exp050_config(forbidden, mode="formal")

    invalid = deepcopy(config)
    invalid["formal"]["bounds_m"] = [3.4e-5, 3.2e-5]
    with pytest.raises(ValueError, match="bounds"):
        validate_exp050_config(invalid, mode="formal")


def test_source_identity_exact_replay_and_determinism() -> None:
    config = load_config(CONFIG_PATH)
    source = load_exp030_source(config, PROJECT_ROOT)
    generator = make_exp050_candidate_generator(source)
    first = generator(config["sample_a"]["d_waist_true_m"])
    second = generator(config["sample_a"]["d_waist_true_m"])
    target = np.asarray(source["target"])

    assert source["state"]["status"] == "complete"
    assert target.shape == (384, 384)
    assert target.dtype == np.complex128
    assert source["target_bytes_sha256"] == config["source"][
        "target_bytes_sha256"
    ]
    assert np.linalg.norm(first["P_B"] - target) / np.linalg.norm(target) < 1e-12
    assert np.array_equal(first["P_B"], second["P_B"])
    assert np.array_equal(first["P_B_radial"], second["P_B_radial"])


def test_public_exp030_forward_rejects_invalid_inputs() -> None:
    radius = np.asarray([0.1e-6, 0.2e-6])
    weights = np.asarray([0.1e-6, 0.1e-6])
    output = np.asarray([0.0, 0.2e-6, 0.4e-6])
    with pytest.raises(ValueError, match="positive finite"):
        build_exp030_radial_operator(
            radius,
            np.asarray([0.1e-6, np.nan]),
            output,
            shape=(2, 2),
            dx_m=0.1e-6,
            center_xy_m=(0.0, 0.0),
            wavelength_m=532e-9,
            propagation_distance_m=1e-3,
            medium_index=1.0,
            incident_amplitude=1.0,
        )
    operator = build_exp030_radial_operator(
        radius,
        weights,
        output,
        shape=(2, 2),
        dx_m=0.1e-6,
        center_xy_m=(0.0, 0.0),
        wavelength_m=532e-9,
        propagation_distance_m=1e-3,
        medium_index=1.0,
        incident_amplitude=1.0,
    )
    with pytest.raises(ValueError, match="waist"):
        make_exp030_projected_probe(
            operator,
            thickness_m=100e-6,
            d_top_m=30e-6,
            d_waist_m=31e-6,
            d_bottom_m=30e-6,
            z_waist_m=50e-6,
            n_glass=1.5,
            n_air=1.0,
            wavelength_m=532e-9,
        )


def test_cache_and_golden_search_are_deterministic() -> None:
    target, _radial, generator = _synthetic_generator(3e-6, (3, 4))
    cache = ProbeCache(generator, target.shape)
    first, first_index = cache.get(3e-6)
    second, second_index = cache.get(3e-6)
    assert first_index == second_index == 0
    assert np.array_equal(first, second)
    assert len(cache.probes) == 1

    def objective(value: float) -> tuple[float, int]:
        return (value - 3e-6) ** 2, int(round(value * 1e9))

    first_search = fixed_budget_golden_section_search(
        objective, bracket_m=(2e-6, 4e-6), iterations=9
    )
    second_search = fixed_budget_golden_section_search(
        objective, bracket_m=(2e-6, 4e-6), iterations=9
    )
    assert np.array_equal(
        first_search["evaluated_d_waist_m"],
        second_search["evaluated_d_waist_m"],
    )
    assert first_search["stopping_reason"] == "fixed_iteration_budget"


def test_profile_fd_estimator_and_status_logic() -> None:
    truth = 3e-6
    target, radial, generator = _synthetic_generator(truth)
    design = _synthetic_design(truth)
    result = run_exp050_estimator(
        target, radial, generator, true_waist_m=truth, design=design
    )
    assert result["status"] == "Passed"
    assert result["profile"]["fine_minimum"]["unique"]
    assert result["finite_difference"]["pass"]
    assert result["estimator_crosscheck"]["pass"]
    assert np.array_equal(
        result["P_B_best_raw"] - target, result["residual_field_raw"]
    )

    bad_fd = deepcopy(design)
    bad_fd["thresholds"]["normalized_jacobian_per_m_min"] = 1e12
    inconclusive = run_exp050_estimator(
        target, radial, generator, true_waist_m=truth, design=bad_fd
    )
    assert inconclusive["status"] == "Inconclusive"

    bad_search = deepcopy(design)
    bad_search["thresholds"]["golden_final_bracket_width_m_max"] = 1e-20
    failed = run_exp050_estimator(
        target, radial, generator, true_waist_m=truth, design=bad_search
    )
    assert failed["status"] == "Failed"


def test_invalid_shape_dtype_and_nonfinite_target_are_rejected() -> None:
    truth = 3e-6
    target, radial, generator = _synthetic_generator(truth)
    design = _synthetic_design(truth)
    with pytest.raises(ValueError, match="complex128"):
        run_exp050_estimator(
            target.real, radial, generator, true_waist_m=truth, design=design
        )
    nonfinite = target.copy()
    nonfinite[0, 0] = np.nan + 0j
    with pytest.raises(ValueError, match="finite"):
        run_exp050_estimator(
            nonfinite, radial, generator, true_waist_m=truth, design=design
        )
    wrong_shape = target[:-1]
    with pytest.raises(ValueError, match="shape"):
        run_exp050_estimator(
            wrong_shape,
            radial,
            generator,
            true_waist_m=truth,
            design=design,
        )


def test_tiny_runner_hdf5_provenance_shape_dtype_and_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = 3e-6
    target, radial, generator = _synthetic_generator(truth, (8, 8))
    config = load_config(CONFIG_PATH)
    config["run"]["output_root"] = str(tmp_path)
    config["sample_a"]["d_waist_true_m"] = truth
    config["sample_a"]["d_top_m"] = 5e-6
    config["sample_a"]["d_bottom_m"] = 5e-6
    config["sample_a"]["thickness_m"] = 10e-6
    config["sample_a"]["z_waist_m"] = 5e-6
    config["formal"] = _synthetic_design(truth)
    config_path = tmp_path / "exp050_tiny.yaml"
    save_config(config_path, config)
    source = {
        "run_dir": PROJECT_ROOT
        / "runs/exp030_TGV_2d_effective_phase_20260810_121124",
        "paths": {"hdf5": "synthetic.h5"},
        "hashes": {name: "0" * 64 for name in config["source"]["sha256"]},
        "metadata": {"git_commit": config["source"]["git_commit"]},
        "state": {"status": "complete", "experiment_status": "Passed"},
        "source_artifacts_validated_field_present": False,
        "target": target,
        "target_bytes_sha256": sha256_array(target),
        "target_radial_probe": radial,
        "shape_ny_nx": target.shape,
        "axis_order": "y_x",
        "instrument": {
            "wavelength_m": 532e-9,
            "dx_m": 0.25e-6,
            "z_AB_m": 1e-3,
            "z_BC_m": 1e-3,
            "detector_pixel_size_m": 0.25e-6,
            "medium_index": 1.0,
        },
        "sample_a": config["sample_a"],
        "center_xy_m": (0.0, 0.0),
        "fov_yx_m": (2e-6, 2e-6),
        "coordinate_convention": "(index-(N-1)/2)*dx",
        "coordinate_endpoints_x_m": (-0.875e-6, 0.875e-6),
        "coordinate_endpoints_y_m": (-0.875e-6, 0.875e-6),
        "target_dataset_units_attribute_present": False,
        "source_radius_m": np.linspace(0.1e-6, 1.0e-6, 13),
        "source_weights_m": np.full(13, 0.1e-6),
        "output_radius_m": np.linspace(0.0, 2e-6, 13),
    }
    monkeypatch.setattr(runner, "load_exp030_source", lambda *_args: source)
    monkeypatch.setattr(
        runner, "make_exp050_candidate_generator", lambda _source: generator
    )
    run_dir = runner.run(config_path, mode="formal")
    state = json_load(run_dir / "run_state.json")
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    with h5py.File(
        run_dir / "outputs/exp050_projected_true_probe_waist_fit.h5", "r"
    ) as h5:
        assert list(h5["entry/data"].keys()) == []
        assert "calibration" not in h5["entry"]
        assert "preprocessing" not in h5["entry"]
        saved = h5["entry/truth/P_B_true"]
        assert saved.shape == (8, 8)
        assert saved.dtype == np.complex128
        assert h5["entry/instrument/axis_order"].asstr()[()] == "y_x"
        assert (
            h5["entry/instrument/field_units"].asstr()[()]
            == "arbitrary complex field amplitude"
        )
        assert not bool(
            h5[
                "entry/reconstruction/waist_fit/source_provenance/P_B_rec_used"
            ][()]
        )


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
