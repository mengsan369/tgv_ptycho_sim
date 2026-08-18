from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scripts.run_exp040_r12 as runner

from tgv_ptycho.io.config import load_config
from tgv_ptycho.optics.hankel import make_qdht_plan
from tgv_ptycho.viz.plot_exp040_r12 import save_exp040_r12_figures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/exp040_TGV_3d_multislice_r12.yaml"
)


def _radial_result(radius: np.ndarray, guard: float = 0.0):
    total = np.ones(radius.shape, dtype=np.complex128)
    return {
        "radius_m": radius,
        "normalized_total_trace": total,
        "normalized_scattered_trace": total - 1.0,
        "controls": {"outer_guard_rms_ratio": guard, "all_finite": True},
    }


def test_r12_config_and_provenance_are_locked() -> None:
    config = load_config(CONFIG_PATH)
    runner.validate_r12_config(config)
    provenance = runner._load_and_validate_provenance(config)

    assert runner._sha256(CONFIG_PATH) == runner.REGISTERED_CONFIG_SHA256
    assert provenance["preflight_metrics"]["formal_r12_allowed"] is True


def test_r12_postprocess_constant_fields_closes_all_numeric_differences() -> None:
    config = load_config(CONFIG_PATH)
    dr = float(config["comparison"]["trace_sampling_m"])
    core48_radius = (np.arange(576) + 0.5) * dr
    core60_radius = (np.arange(720) + 0.5) * dr
    plan = make_qdht_plan(512, float(config["qdht"]["radial_max_m"]))
    core48 = _radial_result(core48_radius)
    core60 = _radial_result(core60_radius)
    fem = {
        "fem_p2_core60": _radial_result(core60_radius),
        "fem_p3_core60": _radial_result(core60_radius),
    }
    cartesian = {
        "chord_fov64_standard": {
            "normalized_native_field": np.ones((512, 512), dtype=np.complex128)
        },
        "chord_fov96_standard": {
            "normalized_native_field": np.ones((768, 768), dtype=np.complex128)
        },
        "chord_fov128_standard": {
            "normalized_native_field": np.ones((1024, 1024), dtype=np.complex128)
        },
        "chord_fov128_alias": {
            "normalized_native_field": np.ones((1024, 1024), dtype=np.complex128)
        },
    }
    qdht = _radial_result(plan.radial_nodes_m)

    metrics, arrays = runner._postprocess(
        config, core48, core60, fem, cartesian, qdht
    )

    assert metrics["domain"]["core48_to_core60_passband_radial_l2"] == 0.0
    assert metrics["fem"]["p2_to_p3_passband_radial_l2"] == 0.0
    assert metrics["cartesian"]["fov96_to_fov128_passband_relative_l2"] == 0.0
    assert (
        metrics["cartesian"]["polar_controls"]["chord_fov128_alias"]
        ["angular_relative_l2"]
        <= 1.0e-12
    )
    assert arrays["fem_p3_passband_cartesian"].shape == (512, 512)


def test_r12_hdf5_and_plots_accept_closed_synthetic_contract(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    radius = (np.arange(160, dtype=np.float64) + 0.5) * 1.25e-7
    complex_radial = np.ones(160, dtype=np.complex128)
    arrays = {
        "radial_radius_m": radius,
        "cartesian_alias_radial": complex_radial,
        "qdht_radial": complex_radial,
        "fem_p3_radial": complex_radial,
        "adc_core60_radial": complex_radial,
        "fem_p3_passband_cartesian": np.ones((512, 512), dtype=np.complex128),
        "cartesian_alias_passband": np.ones((512, 512), dtype=np.complex128),
        "cross_fem_radial": complex_radial,
        "cross_multislice_radial": complex_radial,
    }
    polar_controls = {
        key: {"angular_relative_l2": 0.0}
        for key in (
            "chord_fov64_standard",
            "chord_fov96_standard",
            "chord_fov128_standard",
            "chord_fov128_alias",
        )
    }
    metrics = {
        "sampling": {},
        "domain": {
            "core48_to_core60_passband_radial_l2": 0.0,
            "core60_outer_guard_rms_ratio": 0.0,
        },
        "fem": {
            "p2_to_p3_passband_radial_l2": 0.0,
            "p3_outer_guard_rms_ratio": 0.0,
            "adc5_to_p3_passband_radial_l2_report_only": 0.0,
        },
        "cartesian": {"polar_controls": polar_controls},
        "conditional_cross_model": {
            "executed": True,
            "passband_radial_l2": 0.0,
            "failed_gates": [],
        },
    }
    qdht = {
        "radius_m": np.linspace(1.0e-7, 6.3e-5, 512),
        "normalized_total_trace": np.ones(512, dtype=np.complex128),
    }
    output = tmp_path / "synthetic.h5"
    runner._write_hdf5(output, config, {"synthetic": True}, metrics, arrays, qdht)
    paths = save_exp040_r12_figures(tmp_path / "figures", metrics, arrays)

    with h5py.File(output, "r") as handle:
        assert set(handle["entry/data"]) == {
            "polar_means",
            "qdht_native",
            "radial_profiles",
            "selected_complex_fields",
        }
    assert len(paths) == 3
    assert all(path.stat().st_size > 0 for path in paths)
