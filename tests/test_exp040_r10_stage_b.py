from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tgv_ptycho.io.config import load_config
from tgv_ptycho.viz.plot_exp040_r10_stage_b import (
    EXP040_R10_STAGE_B_FIGURE_FILENAMES,
    save_exp040_r10_stage_b_figures,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "exp040_TGV_3d_multislice_r10_stage_b.yaml"
)
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_exp040_r10_stage_b.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("r10_stage_b", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_config_freezes_mesh_sampling_and_thresholds() -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest().upper()
    assert actual == runner.REGISTERED_CONFIG_SHA256
    runner.validate_stage_b_config(config)
    changed = copy.deepcopy(config)
    changed["helmholtz"]["cases"]["fine_nominal"]["dr_m"] *= 2.0
    with pytest.raises(ValueError, match="Helmholtz cases differs"):
        runner.validate_stage_b_config(changed)
    changed = copy.deepcopy(config)
    changed["thresholds"]["reference_passband_relative_l2_max"] = 0.1
    with pytest.raises(ValueError, match="thresholds differs"):
        runner.validate_stage_b_config(changed)


def test_formal_config_locks_passed_stage_a_and_preflight_provenance() -> None:
    runner = _load_runner()
    metrics = runner._load_and_validate_provenance(load_config(CONFIG_PATH))
    assert metrics["status"] == "Passed"
    assert metrics["formal_stage_b_allowed"] is True


def test_tiny_scattered_field_case_is_finite_and_checkpointable(tmp_path) -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    config["physics"].update(
        {
            "wavelength_m": 0.8e-6,
            "background_interface_z_m": 2.0e-6,
            "sample_thickness_m": 2.0e-6,
            "d_top_m": 2.0e-6,
            "d_waist_m": 1.0e-6,
            "d_bottom_m": 2.0e-6,
            "z_waist_m": 1.0e-6,
        }
    )
    config["helmholtz"].update(
        {
            "radial_core_max_m": 2.0e-6,
            "z_core_min_m": -0.5e-6,
            "z_core_max_m": 2.5e-6,
            "axial_material_subnodes": 2,
        }
    )
    config["helmholtz"]["cases"]["coarse_nominal"] = {
        "dr_m": 0.25e-6,
        "dz_m": 0.25e-6,
        "pml_thickness_m": 0.5e-6,
        "expected_nr": 10,
        "expected_nz": 16,
        "expected_unknowns": 160,
    }
    config["observation"]["z_m"] = 2.25e-6
    config["comparison"].update(
        {
            "guard_inner_max_radius_m": 1.0e-6,
            "guard_min_radius_m": 1.5e-6,
            "guard_max_radius_m": 2.0e-6,
        }
    )
    result = runner._solve_helmholtz_case(
        config,
        "coarse_nominal",
        case_index=1,
        progress_callback=None,
    )
    assert result["controls"]["all_finite"] is True
    assert result["controls"]["solver_controls"]["relative_residual"] < 1e-10
    assert result["normalized_total_trace"].shape == (10,)
    checkpoint = tmp_path / "coarse_nominal.npz"
    runner._save_helmholtz_checkpoint(checkpoint, result)
    with np.load(checkpoint) as data:
        np.testing.assert_allclose(
            data["normalized_total_trace"], result["normalized_total_trace"]
        )


def test_registered_postprocess_preserves_constant_fields() -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    config["comparison"].update(
        {
            "cartesian_shape": [16, 16],
            "cartesian_dx_m": 0.25e-6,
            "trace_support_radius_m": 1.5e-6,
            "physical_passband_cutoff_cycles_per_m": 1.0e6,
            "annular_bin_width_m": 0.25e-6,
            "annular_maximum_radius_m": 1.0e-6,
        }
    )
    config["multislice"]["dx_m"] = 0.125e-6
    radius = (np.arange(8) + 0.5) * 0.25e-6
    helmholtz = {
        name: {
            "radius_m": radius,
            "normalized_total_trace": np.ones(8, dtype=np.complex128),
        }
        for name in config["helmholtz"]["fixed_case_order"]
    }
    multislice = {
        "normalized_native_field": np.ones((32, 32), dtype=np.complex128)
    }
    post = runner._postprocess_once(config, helmholtz, multislice)
    for name in ("mesh", "pml", "cross_model"):
        assert post["comparisons"][name]["raw_radial_l2"] == 0.0
        assert post["comparisons"][name]["passband_radial_l2"] == 0.0
    assert post["multislice_azimuthal_anisotropy_relative_l2"] == 0.0
    assert post["annular_constant_max_abs_error"] == 0.0


def test_stage_b_plot_contract(tmp_path) -> None:
    shape = (16, 16)
    radius = (np.arange(4) + 0.5) * 0.25e-6
    field = np.ones(shape, dtype=np.float64)
    metrics = {
        "comparisons": {
            name: {"passband_radial_l2": 0.01}
            for name in ("mesh", "pml", "cross_model")
        },
        "reference_controls": {
            "multislice_azimuthal_anisotropy_relative_l2": 0.01
        },
        "thresholds": {"reference_passband_relative_l2_max": 0.05},
        "case_controls": {
            name: {"outer_guard_rms_ratio": 0.01}
            for name in (
                "coarse_nominal",
                "fine_nominal",
                "fine_enlarged_pml",
            )
        },
        "sampling": {"cartesian_dx_m": 0.25e-6},
    }
    result = {
        "metrics": metrics,
        "selected_maps": {
            "helmholtz_passband_amplitude": field,
            "multislice_passband_amplitude": field,
            "normalized_cross_residual": np.zeros(shape),
            "cross_phase_difference_rad": np.zeros(shape),
        },
        "radial_profiles": {
            "radius_m": radius,
            "fine_enlarged_pml_passband": np.ones(4, dtype=np.complex128),
            "multislice_fine_1024_passband": np.ones(
                4, dtype=np.complex128
            ),
        },
    }
    paths = save_exp040_r10_stage_b_figures(result, tmp_path)
    assert [path.name for path in paths] == list(
        EXP040_R10_STAGE_B_FIGURE_FILENAMES
    )
    assert all(path.stat().st_size > 0 for path in paths)
