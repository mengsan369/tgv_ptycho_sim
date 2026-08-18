from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest
from scripts import run_exp040_multislice_forward as exp040_runner

from tgv_ptycho.forward.exp040 import (
    _assemble_metrics,
    build_exp040_hdf5_payload,
    center_crop,
    resample_centered_grid,
    run_exp040_experiment,
    validate_exp040_config,
)
from tgv_ptycho.io.config import config_to_yaml, load_config, save_config
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5
from tgv_ptycho.viz.plot_exp040 import (
    EXP040_FIGURE_FILENAMES,
    save_exp040_figures,
)


def _tiny_config() -> dict[str, object]:
    path = Path("configs/experiments/exp040_TGV_3d_multislice_forward.yaml")
    config = deepcopy(load_config(path))
    config["optics"].update(  # type: ignore[union-attr]
        baseline_shape=[16, 16],
        baseline_dx_m=1e-6,
        z_AB_m=20e-6,
        z_BC_m=30e-6,
    )
    config["optics"]["detector"]["pixel_size_m"] = 1e-6  # type: ignore[index]
    config["sample_a"].update(  # type: ignore[union-attr]
        thickness_m=8e-6,
        d_top_m=6e-6,
        d_waist_m=4e-6,
        d_bottom_m=6e-6,
        z_waist_m=4e-6,
    )
    config["multislice"]["target_dz_m"] = 2e-6  # type: ignore[index]
    config["sample_b"]["physical_feature_size_m"] = 2e-6  # type: ignore[index]
    config["sample_b"]["canonical_grid"].update(  # type: ignore[index,union-attr]
        shape=[48, 48],
        dx_m=0.5e-6,
        fov_m=[24e-6, 24e-6],
    )
    config["scan"].update(  # type: ignore[union-attr]
        num_x=3,
        num_y=3,
        step_m=2e-6,
        max_jitter_px=0,
        jitter_quantum_m=2e-6,
    )
    config["waist_perturbation"].update(  # type: ignore[union-attr]
        delta_d_waist_m=0.5e-6,
        d_waist_m=[3.5e-6, 4e-6, 4.5e-6],
    )
    config["convergence"]["axial"].update(  # type: ignore[index,union-attr]
        fixed_shape=[16, 16],
        fixed_dx_m=1e-6,
        dz_cases_m=[4e-6, 2e-6, 1e-6],
        acceptance_pair_m=[2e-6, 1e-6],
    )
    config["convergence"]["lateral_fixed_fov"].update(  # type: ignore[index,union-attr]
        fov_m=[16e-6, 16e-6],
        cases=[
            {"shape": [8, 8], "dx_m": 2e-6},
            {"shape": [16, 16], "dx_m": 1e-6},
            {"shape": [32, 32], "dx_m": 0.5e-6},
        ],
        acceptance_pair_dx_m=[1e-6, 0.5e-6],
        comparison_grid_shape=[16, 16],
        comparison_grid_dx_m=1e-6,
    )
    config["convergence"]["fov"].update(  # type: ignore[index,union-attr]
        fixed_dx_m=1e-6,
        shapes=[[16, 16], [20, 20], [24, 24]],
        common_center_roi_shape=[16, 16],
        common_center_roi_fov_m=[16e-6, 16e-6],
        acceptance_pair_shapes=[[20, 20], [24, 24]],
    )
    return config


def test_centered_mapping_preserves_constant_and_complex_dtype() -> None:
    source = np.full((2, 32, 32), 2.0 + 3.0j, dtype=np.complex128)
    mapped = resample_centered_grid(source, 0.5e-6, (16, 16), 1e-6)
    cropped = center_crop(source, (16, 16))

    assert mapped.shape == (2, 16, 16)
    assert mapped.dtype == np.complex128
    assert np.array_equal(mapped, np.full_like(mapped, 2.0 + 3.0j))
    assert cropped.shape == (2, 16, 16)


def test_config_rejects_scan_not_integer_on_every_grid() -> None:
    config = _tiny_config()
    config["scan"]["jitter_quantum_m"] = 0.5e-6  # type: ignore[index]

    with pytest.raises(ValueError, match="integer multiple"):
        validate_exp040_config(config)


def test_tiny_exp040_is_deterministic_and_reuses_baseline_data() -> None:
    config = _tiny_config()
    first = run_exp040_experiment(config)
    second = run_exp040_experiment(config)

    assert first["baseline"]["n_volume"].shape == (4, 16, 16)
    assert first["baseline"]["U_A_exit"].shape == (16, 16)
    assert first["baseline"]["P_B"].shape == (16, 16)
    assert first["baseline"]["I_stack"].shape == (9, 16, 16)
    assert first["sweep"]["I_stack"].shape == (3, 9, 16, 16)
    assert np.array_equal(
        first["baseline"]["I_stack"], first["sweep"]["I_stack"][1]
    )
    assert np.array_equal(
        first["baseline"]["U_A_exit"], first["sweep"]["U_A_exit"][1]
    )
    assert np.array_equal(
        first["baseline"]["P_B"], first["sweep"]["P_B"][1]
    )
    assert np.array_equal(
        first["baseline"]["U_A_exit"], second["baseline"]["U_A_exit"]
    )
    assert np.array_equal(first["sweep"]["I_stack"], second["sweep"]["I_stack"])
    assert first["shared_inputs"]["same_physical_sample_b"] is True
    assert first["shared_inputs"]["same_scan_positions"] is True
    assert first["metrics"]["algebraic_controls"]["pass"] is True
    assert first["metrics"]["finite_control"]["pass"] is True
    determinism = first["metrics"]["determinism_control"]
    assert set(determinism["waist_sweep"]) == {
        "waist_minus",
        "baseline",
        "waist_plus",
    }
    assert determinism["baseline"]["U_A_exit_relative_l2"] == determinism[
        "U_A_exit_relative_l2"
    ]
    assert determinism["scope"] == (
        "all_saved_baseline_truth_data_and_waist_sweep_outputs"
    )
    assert {
        "n_volume_relative_l2",
        "z_m_relative_l2",
        "slice_thickness_m_relative_l2",
        "diameter_z_m_relative_l2",
        "incident_field_relative_l2",
        "B_relative_l2",
        "scan_positions_relative_l2",
        "projected_phase_product_relative_l2",
    } <= set(determinism["baseline"])
    assert all(
        value == 0.0
        for key, value in determinism["baseline"].items()
        if key.endswith("_relative_l2")
    )
    all_determinism_errors = [
        value
        for case in determinism["waist_sweep"].values()
        for key, value in case.items()
        if key.endswith("_relative_l2")
    ]
    assert determinism["max_relative_l2"] == max(all_determinism_errors)
    convergence = first["metrics"]["convergence"]
    np.testing.assert_array_equal(
        convergence["axial"]["acceptance_pair_m"], [2e-6, 1e-6]
    )
    np.testing.assert_array_equal(
        convergence["lateral"]["acceptance_pair_m"], [1e-6, 0.5e-6]
    )
    np.testing.assert_array_equal(
        convergence["fov"]["acceptance_pair_shapes"], [[20, 20], [24, 24]]
    )


def test_finite_control_checks_convergence_intensity_nonnegative() -> None:
    config = _tiny_config()
    field = np.ones((2, 2), dtype=np.complex128)
    intensity = np.ones((1, 2, 2), dtype=np.float64)

    def case(case_intensity: np.ndarray = intensity) -> dict[str, object]:
        return {
            "n_volume": np.ones((1, 2, 2), dtype=np.float64),
            "U_A_exit": field,
            "P_B": field,
            "I_stack": case_intensity,
        }

    baseline = case()
    sweep_cases = [case(), case(), case()]
    negative_intensity = intensity.copy()
    negative_intensity[0, 0, 0] = -1.0
    convergence = {
        "axial": {
            "acceptance": {"U_A_exit": 0.0, "P_B": 0.0, "I_stack": 0.0},
            "case_values_m": np.asarray([2e-6, 1e-6]),
            "acceptance_pair_m": np.asarray([2e-6, 1e-6]),
            "cases": [case(negative_intensity)],
        },
        "lateral": {
            "acceptance": {"U_A_exit": 0.0, "P_B": 0.0, "I_stack": 0.0},
            "case_values_m": np.asarray([1e-6, 0.5e-6]),
            "acceptance_pair_m": np.asarray([1e-6, 0.5e-6]),
            "cases": [case()],
        },
        "fov": {
            "acceptance": {"U_A_exit": 0.0, "P_B": 0.0, "I_stack": 0.0},
            "case_values_m": np.asarray([20e-6, 24e-6]),
            "acceptance_pair_shapes": np.asarray([[20, 20], [24, 24]]),
            "cases": [case()],
        },
    }
    controls = {
        "pass": True,
        "geometry": {"pass": True},
        "determinism": {"pass": True},
    }

    metrics = _assemble_metrics(
        config,
        baseline,
        sweep_cases,
        controls,
        convergence,
    )

    assert metrics["finite_control"]["all_outputs_finite"] is True
    assert metrics["finite_control"]["all_intensity_nonnegative"] is False
    assert metrics["finite_control"]["pass"] is False
    assert metrics["experiment_status"] == "Failed"


def test_exp040_hdf5_layout_has_truth_but_no_fake_workflow_groups(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    result = run_exp040_experiment(config)
    config_yaml = config_to_yaml(config)
    payload = build_exp040_hdf5_payload(
        result,
        config,
        config_yaml=config_yaml,
        metadata={"run_name": "pytest_exp040", "created_at": "test"},
    )
    output = tmp_path / "exp040.h5"
    save_ptycho_hdf5(output, **payload)

    with h5py.File(output, "r") as h5:
        entry = h5["entry"]
        required = (
            "data/I_stack",
            "data/scan_positions",
            "instrument/wavelength",
            "instrument/dx",
            "instrument/z_AB",
            "instrument/z_BC",
            "instrument/detector_pixel_size",
            "instrument/internal_reference_index",
            "instrument/external_medium_index",
            "sample/sample_A_type",
            "sample/tgv_parameters",
            "sample/sample_B_type",
            "sample/sample_B_parameters",
            "truth/n_volume",
            "truth/z_m",
            "truth/slice_thickness_m",
            "truth/diameter_z_m",
            "truth/incident_field_true",
            "truth/U_A_exit_true",
            "truth/P_B_true",
            "truth/B_true",
            "truth/parameter_sweep/d_waist_m",
            "truth/parameter_sweep/U_A_exit_true",
            "truth/parameter_sweep/P_B_true",
            "truth/parameter_sweep/I_stack_true",
            "config_yaml",
            "metadata",
            "metrics",
        )
        assert all(path in entry for path in required)
        assert "reconstruction" not in entry
        assert "calibration" not in entry
        assert "preprocessing" not in entry
        assert entry["truth/n_volume"].dtype == np.float64
        assert entry["truth/U_A_exit_true"].dtype == np.complex128
        assert entry["data/I_stack"].dtype == np.float64
        assert np.array_equal(
            entry["data/I_stack"][...],
            entry["truth/parameter_sweep/I_stack_true"][1],
        )
        assert np.isclose(
            np.sum(entry["truth/slice_thickness_m"][...]),
            config["sample_a"]["thickness_m"],  # type: ignore[index]
        )


def test_exp040_result_schema_generates_all_registered_figures(
    tmp_path: Path,
) -> None:
    result = run_exp040_experiment(_tiny_config())
    paths = save_exp040_figures(result, tmp_path)

    assert [path.name for path in paths] == list(EXP040_FIGURE_FILENAMES)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_exp040_runner_metadata_and_validators_are_strict(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    config["output"]["hdf5_filename"] = "custom_exp040_output.h5"  # type: ignore[index]
    result = run_exp040_experiment(config)
    metrics = result["metrics"]
    status = str(metrics["experiment_status"])
    run_dir = tmp_path / "run"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True)
    config_path = Path(
        "configs/experiments/exp040_TGV_3d_multislice_forward.yaml"
    )
    metadata = exp040_runner._metadata(config, config_path, run_dir, status)
    config_yaml = config_to_yaml(config)
    save_config(run_dir / "config.yaml", config)
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "metrics.json", metrics)

    output = run_dir / "outputs" / str(config["output"]["hdf5_filename"])  # type: ignore[index]
    payload = build_exp040_hdf5_payload(
        result,
        config,
        config_yaml=config_yaml,
        metadata=metadata,
    )
    save_ptycho_hdf5(output, **payload)
    figure_paths = [figures_dir / name for name in EXP040_FIGURE_FILENAMES]
    for figure_path in figure_paths:
        figure_path.write_bytes(b"test")

    exp040_runner._validate_hdf5(
        output,
        config=config,
        metadata=metadata,
        metrics=metrics,
        config_yaml=config_yaml,
    )
    exp040_runner._validate_external_files(
        run_dir,
        output,
        config_yaml=config_yaml,
        metadata=metadata,
        metrics=metrics,
        figure_paths=figure_paths,
    )

    assert metadata["config_status_at_launch"] == "Planned"
    assert metadata["experiment_status"] == status
    assert metadata["internal_reference_index"] == 1.5
    assert metadata["external_medium_index"] == 1.0
    assert metadata["incident_field_randomized"] is False
    assert (
        metadata["random_seeds"]["incident_field"]
        == "not_applicable_deterministic_plane_wave"
    )
    with h5py.File(output, "r") as h5:
        entry = h5["entry"]
        assert entry["metadata/internal_reference_index"][()] == 1.5
        assert entry["metadata/external_medium_index"][()] == 1.0
        assert (
            entry["metadata/random_seeds/incident_field"][()].decode("utf-8")
            == "not_applicable_deterministic_plane_wave"
        )

    with h5py.File(output, "r+") as h5:
        h5["entry/truth/parameter_sweep/P_B_true"][1, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="baseline P_B"):
        exp040_runner._validate_hdf5(
            output,
            config=config,
            metadata=metadata,
            metrics=metrics,
            config_yaml=config_yaml,
        )

    with h5py.File(output, "r+") as h5:
        h5["entry/truth/parameter_sweep/P_B_true"][1] = h5[
            "entry/truth/P_B_true"
        ][...]
        h5["entry/truth/slice_thickness_m"][0] = -1.0
    with pytest.raises(RuntimeError, match="slice widths must be finite and positive"):
        exp040_runner._validate_hdf5(
            output,
            config=config,
            metadata=metadata,
            metrics=metrics,
            config_yaml=config_yaml,
        )
