from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from tgv_ptycho.io.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "exp040_TGV_3d_multislice_r10_stage_b_preflight.yaml"
)
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_exp040_r10_stage_b_preflight.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("r10_stage_b_preflight", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_config_hash_and_frozen_controls() -> None:
    runner = _load_runner()
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest().upper()
    assert actual == runner.REGISTERED_CONFIG_SHA256
    runner.validate_preflight_config(load_config(CONFIG_PATH))


def test_preflight_rejects_registered_grid_or_threshold_change() -> None:
    runner = _load_runner()
    config = load_config(CONFIG_PATH)
    changed_grid = copy.deepcopy(config)
    changed_grid["grid"]["dr_m"] *= 2.0
    with pytest.raises(ValueError, match="grid differs"):
        runner.validate_preflight_config(changed_grid)
    changed_threshold = copy.deepcopy(config)
    changed_threshold["thresholds"]["process_peak_rss_gib_max"] = 8.0
    with pytest.raises(ValueError, match="thresholds differs"):
        runner.validate_preflight_config(changed_threshold)
