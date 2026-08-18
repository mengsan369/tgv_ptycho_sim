from __future__ import annotations

from pathlib import Path

import numpy as np
import scripts.run_exp040_r14 as formal_runner
import scripts.run_exp040_r14b as formal_entry
import scripts.run_exp040_r14b_release as release

from tgv_ptycho.io.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/exp040_TGV_3d_multislice_r14b.yaml"
)
LEGACY_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/exp040_TGV_3d_multislice_r14.yaml"
)
RELEASE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/exp040_TGV_3d_multislice_r14b_release.yaml"
)
SCIENTIFIC_CONTRACT_SHA256 = (
    "F4247CC4298E61092363AD7FAD65016820088608D9A5FF42009ACEB50F1D1D37"
)


def test_r14b_reuses_axial_controls_and_preserves_solver_contract() -> None:
    formal = load_config(FORMAL_CONFIG_PATH)
    legacy = load_config(LEGACY_CONFIG_PATH)
    formal_runner.validate_r14_config(formal)
    assert formal_runner.scientific_contract_sha256(formal) == (
        SCIENTIFIC_CONTRACT_SHA256
    )
    assert formal_runner._sha256(FORMAL_CONFIG_PATH) == (
        formal_entry.REGISTERED_CONFIG_SHA256
    )
    for key in (
        "solver_scaling",
        "solvers",
        "memory_projection",
        "thresholds",
        "conditional_execution",
    ):
        assert formal[key] == legacy[key]
    metrics, arrays = formal_runner._load_reused_axial_pml(formal)
    assert set(metrics) == {"air_upward", "glass_downward"}
    assert set(arrays) == {"glass_downward"}
    assert all(
        np.all(np.isfinite(value))
        for value in arrays["glass_downward"].values()
    )
    assert metrics["air_upward"]["maximum_incoming_to_outgoing_ratio"] < 1e-3
    assert metrics["glass_downward"]["maximum_incoming_to_outgoing_ratio"] < 1e-3
    provenance = formal_runner._load_and_validate_provenance(formal)
    assert provenance["preflight_metrics"]["formal_r14_allowed"] is True


def test_r14b_release_is_hash_locked_and_authorizes_no_computation() -> None:
    config = load_config(RELEASE_CONFIG_PATH)
    release.validate_release_config(config)
    assert release._sha256(RELEASE_CONFIG_PATH) == (
        release.REGISTERED_CONFIG_SHA256
    )
    metrics, formal = release._validate_release_inputs(config)
    assert metrics["formal_r14_allowed"] is True
    assert all(metrics["gates"].values())
    assert formal["experiment"]["stage"] == "R14B"
    assert config["resource_controls"]["assemble_formal_matrix"] is False
    assert config["resource_controls"]["rerun_axial_control"] is False
