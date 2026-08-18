from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pytest
from scipy.special import j0

from tgv_ptycho.forward.helmholtz_axisymmetric import (
    adc5_shifted_wavenumber_squared,
    cartesian_polar_angular_diagnostics,
)
from tgv_ptycho.io.config import load_config
from tgv_ptycho.objects.tgv3d import (
    make_tgv_air_fraction_slice_chord_quadrature,
)
from tgv_ptycho.viz.plot_exp040_r11 import (
    EXP040_R11_FIGURE_FILENAMES,
    save_exp040_r11_figures,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "exp040_TGV_3d_multislice_r11.yaml"
)
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_exp040_r11.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("exp040_r11", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _postprocess_stubs() -> tuple[dict[str, Any], dict[str, Any]]:
    field = np.ones((4, 4), dtype=np.complex128)
    radius = (np.arange(4, dtype=np.float64) + 0.5) * 1.25e-7
    projection = {
        "repeat_relative_l2": 0.0,
        "idempotence_relative_l2": 0.0,
        "constant_max_abs_error": 0.0,
        "all_finite": True,
    }
    helmholtz = {
        "comparisons": {
            "domain": {
                "core24_to_core36": {
                    "raw_radial_l2": 0.01,
                    "passband_radial_l2": 0.01,
                },
                "core36_to_core48": {
                    "raw_radial_l2": 0.01,
                    "passband_radial_l2": 0.01,
                },
                "core48_outer_guard_rms_ratio": 0.01,
            },
            "mesh": {
                "adc5": {
                    "raw_radial_l2": 0.01,
                    "passband_radial_l2": 0.01,
                },
                "standard_report_only": {
                    "raw_radial_l2": 0.01,
                    "passband_radial_l2": 0.01,
                },
            },
        },
        "projection_controls": {"adc_fine_core48": dict(projection)},
        "radial_radius_m": radius,
        "radial_raw": {"adc_fine_core48": np.ones(4, dtype=np.complex128)},
        "radial_pass": {"adc_fine_core48": np.ones(4, dtype=np.complex128)},
        "cartesian_pass": {"adc_fine_core48": field},
    }
    polar = {
        "angular_relative_l2": 0.01,
        "rotation_45deg_relative_l2": 0.01,
        "harmonic_relative_l2": {"m4": 0.01, "m8": 0.01},
        "all_finite": True,
    }
    multislice = {
        "legacy_reproduction_absolute_error": 0.0,
        "maximum_formal_polar_angular_relative_l2": 0.01,
        "restriction_angular_increase": 0.01,
        "chord_lateral_passband_relative_l2": 0.01,
        "restriction_controls": {
            "shape_matches": True,
            "constant_max_abs_error": 0.0,
            "area_weighted_complex_mean_relative_error": 0.0,
        },
        "projection_controls": {"chord1024": dict(projection)},
        "polar_controls": {"chord1024_restricted": dict(polar)},
        "fields": {"chord1024_restricted_passband": field},
        "selected_maps": {},
        "polar_radius_m": radius,
        "polar_means": {
            "chord1024_restricted": np.ones(4, dtype=np.complex128)
        },
    }
    return helmholtz, multislice


def test_formal_config_is_valid_and_exposes_only_the_registered_lock_state() -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    runner.validate_r11_config(config)
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest().upper()
    if runner.REGISTERED_CONFIG_SHA256 == "__LOCK_AFTER_PREFLIGHT__":
        assert config["provenance"]["preflight_run"] == "__LOCK_AFTER_PREFLIGHT__"
    else:
        assert actual == runner.REGISTERED_CONFIG_SHA256
        assert "__LOCK_AFTER_PREFLIGHT__" not in set(config["provenance"].values())

    changed = copy.deepcopy(config)
    changed["conditional_execution"]["vector_model_enabled"] = True
    with pytest.raises(ValueError, match="conditional execution differs"):
        runner.validate_r11_config(changed)


def test_formal_config_locks_repaired_preflight_provenance() -> None:
    runner = _load_runner()
    if runner.REGISTERED_CONFIG_SHA256 == "__LOCK_AFTER_PREFLIGHT__":
        pytest.skip("formal config hash is locked only after preflight repair")
    metrics = runner._load_and_validate_provenance(load_config(CONFIG_PATH))
    assert metrics["status"] == "Passed"
    assert metrics["formal_r11_allowed"] is True
    assert metrics["scientific_result"] is False


def test_adc5_is_positive_and_matches_the_registered_symbol_identity() -> None:
    spacing = 8.333333333333333e-8
    wavenumber = 2.0 * np.pi / 5.32e-7 * np.asarray([1.0, 1.5])
    shifted = adc5_shifted_wavenumber_squared(wavenumber, spacing)
    axis = (2.0 / spacing * np.sin(0.5 * wavenumber * spacing)) ** 2
    diagonal = (
        2.0
        * np.sqrt(2.0)
        / spacing
        * np.sin(wavenumber * spacing / (2.0 * np.sqrt(2.0)))
    ) ** 2
    np.testing.assert_allclose(shifted, 0.5 * (axis + diagonal), rtol=2e-15)
    assert np.all(np.isfinite(shifted))
    assert np.all(shifted > 0.0)


def test_chord_cell_average_order_and_area_controls() -> None:
    shape = (128, 128)
    spacing = 2.5e-7
    diameter = 2.0e-5
    formal = make_tgv_air_fraction_slice_chord_quadrature(
        shape, spacing, diameter, 64
    )
    reference = make_tgv_air_fraction_slice_chord_quadrature(
        shape, spacing, diameter, 128
    )
    order_error = np.linalg.norm(formal - reference) / np.linalg.norm(reference)
    area = float(np.sum(formal) * spacing**2)
    expected_area = float(np.pi * (0.5 * diameter) ** 2)
    assert order_error <= 1.0e-5
    assert abs(area - expected_area) / expected_area <= 1.0e-5
    assert float(np.min(formal)) >= 0.0
    assert float(np.max(formal)) <= 1.0


def test_fixed_radius_polar_diagnostic_passes_radial_manufactured_field() -> None:
    shape = (128, 128)
    spacing = 2.5e-7
    coordinate = (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * spacing
    radial = np.hypot(coordinate[:, None], coordinate[None, :])
    field = j0(2.0 * np.pi * 0.75e6 * radial).astype(np.complex128)
    radius = (np.arange(48, dtype=np.float64) + 0.5) * spacing
    theta = 2.0 * np.pi * np.arange(720, dtype=np.float64) / 720.0
    controls, angular_mean = cartesian_polar_angular_diagnostics(
        field,
        dx_m=spacing,
        radius_m=radius,
        theta_rad=theta,
        interpolation_order=3,
    )
    assert controls["angular_relative_l2"] <= 1.0e-2
    assert controls["all_finite"] is True
    assert angular_mean.shape == radius.shape


def test_checkpoint_round_trip_preserves_fields_and_controls(tmp_path: Path) -> None:
    runner = _load_runner()
    radius = np.asarray([0.5, 1.5]) * 1.0e-7
    trace = np.asarray([1.0 + 0.5j, 0.75 - 0.25j])
    helmholtz = {
        "radius_m": radius,
        "normalized_total_trace": trace,
        "normalized_scattered_trace": trace - 1.0,
        "controls": {"all_finite": True, "value": np.float64(0.25)},
    }
    helmholtz_path = tmp_path / "helmholtz.npz"
    runner._save_helmholtz_checkpoint(helmholtz_path, helmholtz)
    with np.load(helmholtz_path) as data:
        np.testing.assert_array_equal(data["radius_m"], radius)
        np.testing.assert_array_equal(data["normalized_total_trace"], trace)
        assert json.loads(str(data["controls_json"])) == {
            "all_finite": True,
            "value": 0.25,
        }

    field = np.arange(16, dtype=np.float64).reshape(4, 4).astype(np.complex128)
    multislice_path = tmp_path / "multislice.npz"
    runner._save_multislice_checkpoint(
        multislice_path,
        {"normalized_native_field": field, "controls": {"slice_count": 2}},
    )
    with np.load(multislice_path) as data:
        np.testing.assert_array_equal(data["normalized_native_field"], field)
        assert json.loads(str(data["controls_json"])) == {"slice_count": 2}


@pytest.mark.parametrize(
    "failed_control",
    ("hard", "domain", "guard", "mesh", "legacy", "polar", "restriction", "lateral"),
)
def test_cross_model_is_never_called_when_any_registered_control_fails(
    monkeypatch: pytest.MonkeyPatch, failed_control: str
) -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    helmholtz, multislice = _postprocess_stubs()
    if failed_control == "domain":
        helmholtz["comparisons"]["domain"]["core36_to_core48"][
            "passband_radial_l2"
        ] = 0.1
    elif failed_control == "guard":
        helmholtz["comparisons"]["domain"]["core48_outer_guard_rms_ratio"] = 0.1
    elif failed_control == "mesh":
        helmholtz["comparisons"]["mesh"]["adc5"]["passband_radial_l2"] = 0.1
    elif failed_control == "legacy":
        multislice["legacy_reproduction_absolute_error"] = 1.0
    elif failed_control == "polar":
        multislice["maximum_formal_polar_angular_relative_l2"] = 0.1
    elif failed_control == "restriction":
        multislice["restriction_angular_increase"] = 0.1
    elif failed_control == "lateral":
        multislice["chord_lateral_passband_relative_l2"] = 0.1

    monkeypatch.setattr(runner, "_helmholtz_postprocess", lambda *_: helmholtz)
    monkeypatch.setattr(runner, "_multislice_postprocess", lambda *_: multislice)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cross-model comparator was called before closure")

    monkeypatch.setattr(runner, "_conditional_cross_model", forbidden)
    post = runner._postprocess_once(
        config,
        {},
        {},
        np.ones((4, 4), dtype=np.complex128),
        hard_controls_prepass=failed_control != "hard",
    )
    assert post["conditional_cross_model"]["executed"] is False


def test_cross_model_runs_once_only_after_all_registered_controls_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    helmholtz, multislice = _postprocess_stubs()
    calls = 0

    monkeypatch.setattr(runner, "_helmholtz_postprocess", lambda *_: helmholtz)
    monkeypatch.setattr(runner, "_multislice_postprocess", lambda *_: multislice)

    def conditional(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            {"executed": True, "passband_radial_l2": 0.01},
            {"normalized_cross_residual": np.zeros((4, 4))},
            {"cross_profile": np.ones(4, dtype=np.complex128)},
        )

    monkeypatch.setattr(runner, "_conditional_cross_model", conditional)
    post = runner._postprocess_once(
        config,
        {},
        {},
        np.ones((4, 4), dtype=np.complex128),
        hard_controls_prepass=True,
    )
    assert calls == 1
    assert post["conditional_cross_model"]["executed"] is True


def test_preliminary_postprocessing_cannot_cross_before_final_hard_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    helmholtz_post, multislice_post = _postprocess_stubs()

    def helmholtz_case(_config, case_id, **_kwargs):
        controls = {
            "solver_controls": {"relative_residual": 0.0},
            "pml_physical_core_identity_max_abs_error": 0.0,
            "matrix_controls": {"complex_symmetric_max_abs_error": 0.0},
            "material_controls": {
                "fraction_bound_error": 0.0,
                "annular_to_subnode_volume_relative_error": 0.0,
            },
            "observation_controls": {"upper_weight": 0.5},
            "all_finite": True,
        }
        return {
            "radius_m": np.asarray([0.5, 1.5]) * 1.0e-7,
            "normalized_total_trace": np.ones(2, dtype=np.complex128),
            "normalized_scattered_trace": np.zeros(2, dtype=np.complex128),
            "controls": controls,
        }

    def multislice_case(_config, case_id, **_kwargs):
        return {
            "normalized_native_field": np.ones((4, 4), dtype=np.complex128),
            "controls": {
                "fraction_bound_error": 0.0,
                "index_bound_error": 0.0,
                "homogeneous_control_relative_l2": 0.0,
                "all_finite": True,
            },
        }

    monkeypatch.setattr(runner, "_solve_helmholtz_case", helmholtz_case)
    monkeypatch.setattr(runner, "_multislice_chord_case", multislice_case)
    monkeypatch.setattr(runner, "_save_helmholtz_checkpoint", lambda *_: None)
    monkeypatch.setattr(runner, "_save_multislice_checkpoint", lambda *_: None)
    monkeypatch.setattr(runner, "_sha256", lambda *_: "TEST")
    monkeypatch.setattr(
        runner,
        "_load_q8_checkpoint",
        lambda *_: (np.ones((4, 4), dtype=np.complex128), {"interface_factor": 8}),
    )
    monkeypatch.setattr(runner, "_helmholtz_postprocess", lambda *_: helmholtz_post)
    monkeypatch.setattr(runner, "_multislice_postprocess", lambda *_: multislice_post)
    monkeypatch.setattr(runner, "_postprocessing_repeat_error", lambda *_: 1.0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cross-model comparator bypassed final hard controls")

    monkeypatch.setattr(runner, "_conditional_cross_model", forbidden)
    result = runner._run_r11(
        config,
        run_dir=tmp_path,
        preflight_metrics={"formal_r11_allowed": True, "maximum_algebra_error": 0.0},
        progress_callback=None,
    )
    assert result["metrics"]["hard_controls"]["determinism_pass"] is False
    assert result["metrics"]["conditional_cross_model"]["executed"] is False


def test_r11_plot_contract_supports_a_skipped_cross_model(tmp_path: Path) -> None:
    shape = (16, 16)
    field = np.ones(shape, dtype=np.float64)
    polar_names = (
        "q8_native_1024",
        "q8_restricted_512",
        "chord512",
        "chord1024_native",
        "chord1024_restricted",
    )
    result = {
        "metrics": {
            "comparisons": {
                "domain": {
                    "core36_to_core48": {"passband_radial_l2": 0.01},
                    "core48_outer_guard_rms_ratio": 0.01,
                },
                "mesh": {
                    "adc5": {"passband_radial_l2": 0.01},
                    "standard_report_only": {"passband_radial_l2": 0.02},
                },
            },
            "anisotropy": {
                "chord_lateral_passband_relative_l2": 0.01,
                "maximum_formal_polar_angular_relative_l2": 0.01,
                "polar_controls": {
                    name: {"angular_relative_l2": 0.01} for name in polar_names
                },
            },
            "gates": {
                "domain_gate_pass": False,
                "adc5_mesh_gate_pass": True,
                "cartesian_anisotropy_gate_pass": True,
            },
            "thresholds": {"domain_passband_relative_l2_max": 0.05},
            "conditional_cross_model": {"executed": False},
        },
        "selected_maps": {
            "q8_vs_chord_normalized_residual": np.zeros(shape),
            "chord1024_passband_amplitude": field,
        },
    }
    paths = save_exp040_r11_figures(result, tmp_path)
    assert [path.name for path in paths] == list(EXP040_R11_FIGURE_FILENAMES)
    assert all(np.asarray(iio.imread(path)).size > 0 for path in paths)


def test_r11_plot_contract_supports_an_executed_cross_model(tmp_path: Path) -> None:
    shape = (16, 16)
    field = np.ones(shape, dtype=np.float64)
    polar_names = (
        "q8_native_1024",
        "q8_restricted_512",
        "chord512",
        "chord1024_native",
        "chord1024_restricted",
    )
    result = {
        "metrics": {
            "comparisons": {
                "domain": {
                    "core36_to_core48": {"passband_radial_l2": 0.01},
                    "core48_outer_guard_rms_ratio": 0.01,
                },
                "mesh": {
                    "adc5": {"passband_radial_l2": 0.01},
                    "standard_report_only": {"passband_radial_l2": 0.02},
                },
            },
            "anisotropy": {
                "chord_lateral_passband_relative_l2": 0.01,
                "maximum_formal_polar_angular_relative_l2": 0.01,
                "polar_controls": {
                    name: {"angular_relative_l2": 0.01} for name in polar_names
                },
            },
            "gates": {
                "domain_gate_pass": True,
                "adc5_mesh_gate_pass": True,
                "cartesian_anisotropy_gate_pass": True,
            },
            "thresholds": {"domain_passband_relative_l2_max": 0.05},
            "conditional_cross_model": {"executed": True},
        },
        "selected_maps": {
            "q8_vs_chord_normalized_residual": np.zeros(shape),
            "chord1024_passband_amplitude": field,
            "helmholtz_passband_amplitude": field,
            "multislice_passband_amplitude": field,
            "normalized_cross_residual": np.zeros(shape),
            "cross_phase_difference_rad": np.zeros(shape),
        },
    }
    paths = save_exp040_r11_figures(result, tmp_path)
    assert [path.name for path in paths] == list(EXP040_R11_FIGURE_FILENAMES)
    assert all(np.asarray(iio.imread(path)).size > 0 for path in paths)
