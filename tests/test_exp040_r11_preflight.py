from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import pytest

from tgv_ptycho.io.config import load_config
from tgv_ptycho.io.save_load import save_ptycho_hdf5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "exp040_TGV_3d_multislice_r11_preflight.yaml"
)
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_exp040_r11_preflight.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("exp040_r11_preflight", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_config_hash_and_frozen_controls() -> None:
    runner = _load_runner()
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest().upper()
    assert actual == runner.REGISTERED_CONFIG_SHA256
    runner.validate_preflight_config(load_config(CONFIG_PATH))


def test_preflight_rejects_geometry_or_polar_control_change() -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    changed_geometry = copy.deepcopy(config)
    changed_geometry["geometry_control"]["formal_order"] = 32
    with pytest.raises(ValueError, match="geometry control differs"):
        runner.validate_preflight_config(changed_geometry)
    changed_polar = copy.deepcopy(config)
    changed_polar["polar_control"]["theta_count"] = 360
    with pytest.raises(ValueError, match="polar control differs"):
        runner.validate_preflight_config(changed_polar)


def test_preflight_grid_counts_adc_and_probe_solve_controls() -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    grids = runner._formal_grid_controls(config)
    assert all(row["expected_match"] for row in grids)
    assert max(row["unknown_count"] for row in grids) == 777600

    adc = runner._adc_algebra_controls(config)
    assert adc["positive"] is True
    assert adc["all_finite"] is True
    assert adc["midpoint_identity_max_relative_error"] <= 1.0e-12

    probe = runner._probe_solve_controls(config)
    assert probe["all_finite"] is True
    assert probe["solver_controls"]["relative_residual"] <= 1.0e-9
    assert probe["manufactured_recovery_relative_l2"] <= 1.0e-8


def test_preflight_registered_geometry_and_polar_controls_pass() -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    geometry = runner._geometry_controls(config)
    assert geometry["maximum_order_relative_l2"] <= 1.0e-5
    assert geometry["maximum_area_relative_error"] <= 1.0e-5
    assert geometry["fraction_bound_error"] == 0.0
    assert geometry["all_finite"] is True

    polar = runner._polar_controls(config)
    assert polar["maximum_angular_relative_l2"] <= 1.0e-2
    assert polar["all_finite"] is True


def test_preflight_hdf5_sequence_adapter_round_trips_mapping_rows(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    original = {
        "rows": [
            {"id": "first", "shape": [2, 3], "pass": True},
            {"id": "second", "shape": [4, 5], "pass": False},
        ],
        "scalars": [1.0, 2.0],
    }
    path = tmp_path / "sequence_adapter.h5"
    save_ptycho_hdf5(path, metrics=runner._hdf5_safe(original))
    with h5py.File(path, "r") as h5:
        decoded = runner._decode_hdf5_sequences(
            runner._hdf5_to_plain(h5["entry/metrics"])
        )
    assert decoded == json.loads(json.dumps(original))
