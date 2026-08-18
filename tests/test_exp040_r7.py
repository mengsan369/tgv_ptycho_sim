from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scripts import run_exp040_multislice_forward as runner

from tgv_ptycho.forward.exp040 import (
    _r7_outcome_code,
    _r7_streamed_tgv_exit,
    validate_exp040_config,
)
from tgv_ptycho.io.config import load_config, save_config
from tgv_ptycho.objects.tgv3d import make_tgv_air_fraction_slice
from tgv_ptycho.objects.tgv_geometry import diameter_profile, midpoint_z_grid


def _tiny_r7_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            Path(
                "configs/experiments/"
                "exp040_TGV_3d_multislice_r7_subvoxel_interface.yaml"
            )
        )
    )
    config["optics"].update(
        baseline_shape=[16, 16],
        baseline_dx_m=1.0e-6,
        z_AB_m=20.0e-6,
        z_BC_m=30.0e-6,
    )
    config["optics"]["detector"]["pixel_size_m"] = 1.0e-6
    config["sample_a"].update(
        thickness_m=8.0e-6,
        d_top_m=6.0e-6,
        d_waist_m=4.0e-6,
        d_bottom_m=6.0e-6,
        z_waist_m=4.0e-6,
    )
    config["waist_perturbation"].update(
        delta_d_waist_m=0.5e-6,
        d_waist_m=[3.5e-6, 4.0e-6, 4.5e-6],
    )
    config["multislice"]["target_dz_m"] = 2.0e-6
    config["sample_b"].update(physical_feature_size_m=0.5e-6)
    config["sample_b"]["canonical_grid"].update(
        shape=[48, 48], dx_m=0.5e-6, fov_m=[24.0e-6, 24.0e-6]
    )
    config["scan"].update(
        num_x=1,
        num_y=1,
        step_m=2.0e-6,
        max_jitter_px=0,
        jitter_quantum_m=2.0e-6,
    )
    config["convergence"]["axial"].update(
        fixed_shape=[16, 16], fixed_dx_m=1.0e-6
    )
    config["convergence"]["lateral_fixed_fov"].update(
        fov_m=[16.0e-6, 16.0e-6],
        cases=[
            {"shape": [8, 8], "dx_m": 2.0e-6},
            {"shape": [16, 16], "dx_m": 1.0e-6},
            {"shape": [32, 32], "dx_m": 0.5e-6},
        ],
        acceptance_pair_dx_m=[1.0e-6, 0.5e-6],
        comparison_grid_shape=[16, 16],
        comparison_grid_dx_m=1.0e-6,
    )
    config["convergence"]["fov"].update(
        fixed_dx_m=1.0e-6,
        shapes=[[16, 16], [20, 20], [24, 24]],
        common_center_roi_shape=[16, 16],
        common_center_roi_fov_m=[16.0e-6, 16.0e-6],
        acceptance_pair_shapes=[[20, 20], [24, 24]],
    )
    r1 = config["diagnostics_r1"]["sample_b_refinement"]
    r1["base_grid"].update(
        shape=[192, 192],
        dx_m=0.125e-6,
        fov_m=[24.0e-6, 24.0e-6],
        coarse_phase_cell_shape=[48, 48],
    )
    r1["working_grid"].update(
        shape=[256, 256], dx_m=0.125e-6, fov_m=[32.0e-6, 32.0e-6]
    )
    r1["extension_each_side_px"] = [32, 32]
    r7 = config["diagnostics_r7"]
    r7["sample_a_sampling"].update(
        shape=[32, 32],
        dx_m=0.5e-6,
        fov_m=[16.0e-6, 16.0e-6],
        dz_m=1.0e-6,
    )
    r7["detector_sampling"].update(
        node_dx_m=0.25e-6,
        base_fov_m=[48.0e-6, 48.0e-6],
        base_node_shape=[192, 192],
        open_fov_m=[96.0e-6, 96.0e-6],
        open_node_shape=[384, 384],
        native_roi_shape=[16, 16],
    )
    r7["finite_b"]["physical_shape_m"] = [24.0e-6, 24.0e-6]
    return config


def test_air_fraction_q1_is_exact_binary_and_all_q_are_positive() -> None:
    q1 = make_tgv_air_fraction_slice((17, 19), 1.0e-6, 8.0e-6, 1)
    q8 = make_tgv_air_fraction_slice((17, 19), 1.0e-6, 8.0e-6, 8)

    assert set(np.unique(q1)) <= {0.0, 1.0}
    assert np.all((q8 >= 0.0) & (q8 <= 1.0))
    np.testing.assert_array_equal(q8 * 64.0, np.rint(q8 * 64.0))
    assert np.any((q8 > 0.0) & (q8 < 1.0))


def test_streamed_interface_q1_matches_binary_volume_propagation() -> None:
    shape = (18, 20)
    dx_m = 0.7e-6
    z_m, widths = midpoint_z_grid(6.0e-6, 1.5e-6)
    diameters = diameter_profile(
        z_m, 6.0e-6, 6.0e-6, 4.0e-6, 6.0e-6, 3.0e-6
    )
    incident = np.ones(shape, dtype=np.complex128)
    first, selected, controls = _r7_streamed_tgv_exit(
        incident=incident,
        shape=shape,
        dx_m=dx_m,
        widths=widths,
        diameters=diameters,
        interface_factor=1,
        center_xy_m=(0.0, 0.0),
        n_glass=1.5,
        n_air=1.0,
        wavelength=532.0e-9,
        n_ref=1.5,
        bandlimit=True,
        selected_slice_index=1,
    )
    second, _, _ = _r7_streamed_tgv_exit(
        incident=incident,
        shape=shape,
        dx_m=dx_m,
        widths=widths,
        diameters=diameters,
        interface_factor=1,
        center_xy_m=(0.0, 0.0),
        n_glass=1.5,
        n_air=1.0,
        wavelength=532.0e-9,
        n_ref=1.5,
        bandlimit=True,
        selected_slice_index=1,
    )

    np.testing.assert_array_equal(first, second)
    assert set(np.unique(selected)) <= {0.0, 1.0}
    assert controls["q1_identity_error"] == 0.0
    assert controls["all_finite"] is True


@pytest.mark.parametrize(
    ("status", "material", "expected"),
    [
        ("Failed", False, "subvoxel_interface_attribution_blocked"),
        ("Inconclusive", True, "subvoxel_interface_quadrature_not_converged"),
        ("Passed", True, "binary_interface_material_for_at_least_one_output"),
        ("Passed", False, "binary_interface_nonmaterial_on_registered_outputs"),
    ],
)
def test_r7_outcome_table_is_frozen(
    status: str, material: bool, expected: str
) -> None:
    flags = {name: material for name in ("U_A_exit", "P_B", "I_stack")}
    assert _r7_outcome_code(
        status=status, binary_material_by_output=flags
    ) == expected


def test_official_and_tiny_r7_configs_validate() -> None:
    official = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r7_subvoxel_interface.yaml"
        )
    )
    validate_exp040_config(official)
    validate_exp040_config(_tiny_r7_config())


def test_tiny_r7_runs_all_four_interfaces_and_validates_metrics() -> None:
    config = _tiny_r7_config()
    from tgv_ptycho.forward.exp040 import run_exp040_experiment

    result = run_exp040_experiment(config)
    metrics = result["metrics"]["diagnostics_r7"]
    runner._validate_r7_metrics(config, metrics)

    assert all(f"diagnostics_r{stage}" not in result for stage in range(1, 7))
    assert metrics["sampling"]["interface_factors"].tolist() == [1, 2, 4, 8]
    assert metrics["sampling"]["slice_count"] == 8
    assert metrics["sampling"]["scan_count"] == 1
    assert metrics["sampling"]["full_volumes_retained"] is False
    assert metrics["sampling"]["full_node_stacks_retained"] is False
    assert metrics["interface_controls"]["q1_binary_identity_max_abs_error"] == 0
    assert metrics["determinism"]["pass"] is True
    assert metrics["all_finite"] is True
    assert metrics["all_intensity_nonnegative"] is True
    assert metrics["status"] in {"Passed", "Inconclusive"}
    assert set(result["diagnostics_r7"]["selected_fractions"]) == {
        "q1",
        "q2",
        "q4",
        "q8",
    }
    assert set(result["diagnostics_r7"]["selected_scan0"]) == {
        "q1",
        "q2",
        "q4",
        "q8",
    }


def test_tiny_r7_runner_writes_hdf5_and_eleven_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_r7_config()
    config["run"].update(name="tiny_r7", output_root="runs")
    config["output"]["hdf5_filename"] = "tiny_r7.h5"
    config_path = tmp_path / "tiny_r7.yaml"
    save_config(config_path, config)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    run_dir = runner.run(config_path)

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["artifacts_validated"] is True
    assert len(list((run_dir / "figures").glob("*.png"))) == 11
    with runner.h5py.File(run_dir / "outputs" / "tiny_r7.h5", "r") as h5:
        assert "entry/metrics/diagnostics_r7" in h5
        assert "entry/truth/diagnostics_r7" not in h5
        assert h5["entry/metadata/diagnostic_stage"][()].decode() == "R7"
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }


def test_r7_rejects_post_registered_factor_change() -> None:
    config = _tiny_r7_config()
    config["diagnostics_r7"]["interface"]["factors"][-1] = 16

    with pytest.raises(ValueError, match="interface cases"):
        validate_exp040_config(config)
