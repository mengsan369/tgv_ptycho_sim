from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scripts import run_exp040_multislice_forward as runner

from tgv_ptycho.io.config import config_to_yaml, load_config, save_config
from tgv_ptycho.io.save_load import save_json, save_ptycho_hdf5
from tgv_ptycho.viz.plot_exp040 import EXP040_FIGURE_FILENAMES


def _config(*, r1: bool) -> dict[str, Any]:
    filename = (
        "exp040_TGV_3d_multislice_refinement.yaml"
        if r1
        else "exp040_TGV_3d_multislice_forward.yaml"
    )
    config = load_config(Path("configs/experiments") / filename)
    config["optics"]["baseline_shape"] = [4, 4]
    config["scan"]["num_x"] = 1
    config["scan"]["num_y"] = 1
    config["sample_a"]["thickness_m"] = 2.0e-6
    config["output"]["hdf5_filename"] = (
        "custom_r1.h5" if r1 else "custom_legacy.h5"
    )
    return config


def _r1_metrics(config: dict[str, Any]) -> dict[str, Any]:
    r1 = config["diagnostics_r1"]
    zero_acceptance = {"U_A_exit": 0.0, "P_B": 0.0, "I_stack": 0.0}
    return {
        "version": "R1",
        "methods": deepcopy(r1["methods"]),
        "canonical_b_validation": {"max_complex_error": 0.0, "pass": True},
        "refined_convergence": {
            "axial": {
                "acceptance_pair_m": deepcopy(
                    r1["refined_axial"]["acceptance_pair_m"]
                ),
                "acceptance": dict(zero_acceptance),
                "pass": True,
            },
            "lateral": {
                "acceptance_pair_dx_m": deepcopy(
                    r1["refined_lateral"]["acceptance_pair_dx_m"]
                ),
                "acceptance": dict(zero_acceptance),
                "pass": True,
            },
            "fov": {
                "acceptance_pair_shapes": deepcopy(
                    r1["refined_fov"]["acceptance_pair_shapes"]
                ),
                "acceptance": dict(zero_acceptance),
                "pass": True,
            },
        },
        "external_padding": {
            "acceptance_pair_shapes": deepcopy(
                r1["external_padding"]["acceptance_pair_shapes"]
            ),
            "acceptance": dict(zero_acceptance),
            "pass": True,
        },
        "refined_floor": {"I_stack": 0.0},
        "visibility_report": {"detector_signal_to_floor_min": 4.0},
        "thresholds": {"convergence_relative_l2_max": 0.05},
        "all_finite": True,
        "all_intensity_nonnegative": True,
        "hard_checks_pass": True,
        "status": "Passed",
    }


def _r2_config() -> dict[str, Any]:
    config = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r2_boundary_alias.yaml"
        )
    )
    config["optics"]["baseline_shape"] = [4, 4]
    config["scan"]["num_x"] = 1
    config["scan"]["num_y"] = 1
    config["sample_a"]["thickness_m"] = 2.0e-6
    config["output"]["hdf5_filename"] = "custom_r2.h5"
    return config


def _r2_metrics(config: dict[str, Any]) -> dict[str, Any]:
    r2 = config["diagnostics_r2"]
    period = r2["period_commensurate"]
    x_values = deepcopy(period["fov_m"])
    shapes = deepcopy(period["shapes"])
    pair = deepcopy(period["acceptance_pair_shapes"])
    method = {
        "x_values_m": x_values,
        "shapes": shapes,
        "acceptance_pair_shapes": pair,
        "P_B": [0.01, 0.005, 0.0],
        "I_stack": [0.02, 0.01, 0.0],
        "acceptance": {"P_B": 0.005, "I_stack": 0.01},
        "pass": True,
    }
    return {
        "version": "R2",
        "methods": deepcopy(r2["methods"]),
        "canonical_b_validation": {"pass": True},
        "a_exit_center_invariance": {"max": 0.0, "pass": True},
        "period_aligned": {
            "current_asm": deepcopy(method),
            "alias_controlled": deepcopy(method),
        },
        "method_difference": {
            "x_values_m": x_values,
            "P_B": [0.01, 0.01, 0.01],
            "I_stack": [0.02, 0.02, 0.02],
            "largest_fov": {"P_B": 0.01, "I_stack": 0.02},
            "material": False,
            "reference_method": "alias_controlled",
        },
        "alias_masks": {
            "x_values_m": x_values,
            "AB_kept_bin_fraction": [0.2, 0.4, 0.6],
            "BC_kept_bin_fraction": [0.1, 0.2, 0.3],
            "pass": True,
        },
        "determinism": {"P_B": 0.0, "I_stack": 0.0, "pass": True},
        "r1_external_comparator": {
            "acceptance_pair_shapes": [[224, 224], [256, 256]],
            "acceptance": {"P_B": 0.2, "I_stack": 0.7},
        },
        "thresholds": {
            "convergence_relative_l2_max": 0.05,
            "method_difference_material_relative_l2_min": 0.05,
            "a_exit_center_invariance_max": 1e-12,
            "canonical_b_mapping_max_complex_error": 1e-12,
            "determinism_relative_l2_max": 1e-14,
        },
        "outcome_flags": {
            "interpretation_code": "period_aligned_fov_supported"
        },
        "all_finite": True,
        "all_intensity_nonnegative": True,
        "hard_checks_pass": True,
        "status": "Passed",
    }


def _write_valid_run(
    tmp_path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path, list[Path]]:
    run_dir = tmp_path / "run"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True)
    save_config(run_dir / "config.yaml", config)
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "metrics.json", metrics)

    shape = (4, 4)
    intensity = np.ones((1, *shape), dtype=np.float64)
    field = np.ones(shape, dtype=np.complex128)
    widths = np.asarray([1.0e-6, 1.0e-6], dtype=np.float64)
    output = run_dir / "outputs" / str(config["output"]["hdf5_filename"])
    save_ptycho_hdf5(
        output,
        I_stack=intensity,
        scan_positions=np.zeros((1, 2), dtype=np.float64),
        instrument={
            "wavelength": float(config["optics"]["wavelength_m"]),
            "dx": float(config["optics"]["baseline_dx_m"]),
            "z_AB": float(config["optics"]["z_AB_m"]),
            "z_BC": float(config["optics"]["z_BC_m"]),
            "detector_pixel_size": float(
                config["optics"]["detector"]["pixel_size_m"]
            ),
            "internal_reference_index": float(
                config["optics"]["internal_reference_index"]
            ),
            "external_medium_index": float(
                config["optics"]["external_medium_index"]
            ),
        },
        sample={
            "sample_A_type": "test_tgv",
            "tgv_parameters": {"thickness_m": 2.0e-6},
            "sample_B_type": "test_phase",
            "sample_B_parameters": {"seed": 1},
        },
        truth={
            "n_volume": np.ones((2, *shape), dtype=np.float64),
            "z_m": np.asarray([0.5e-6, 1.5e-6], dtype=np.float64),
            "slice_thickness_m": widths,
            "diameter_z_m": np.ones(2, dtype=np.float64),
            "incident_field_true": field,
            "U_A_exit_true": field,
            "P_B_true": field,
            "B_true": field,
            "parameter_sweep": {
                "case_ids": ["waist_minus", "baseline", "waist_plus"],
                "d_waist_m": np.asarray([1.0, 2.0, 3.0]),
                "U_A_exit_true": np.stack([field, field, field]),
                "P_B_true": np.stack([field, field, field]),
                "I_stack_true": np.stack([intensity, intensity, intensity]),
            },
        },
        config_yaml=config_to_yaml(config),
        metadata=metadata,
        metrics=metrics,
    )
    figure_paths = [
        figures_dir / name for name in runner._expected_figure_filenames(config)
    ]
    for path in figure_paths:
        path.write_bytes(b"test")
    return run_dir, output, figure_paths


def test_legacy_runner_keeps_eight_figures_and_no_r1_metadata(
    tmp_path: Path,
) -> None:
    config = _config(r1=False)
    metrics = {"experiment_status": "Inconclusive"}
    metadata = runner._metadata(
        config, Path("legacy.yaml"), tmp_path / "run", "Inconclusive"
    )
    run_dir, output, figures = _write_valid_run(
        tmp_path, config, metrics, metadata
    )

    assert runner._expected_figure_filenames(config) == EXP040_FIGURE_FILENAMES
    assert "diagnostic_stage" not in metadata
    assert "diagnostics_r1_status" not in metadata
    runner._validate_hdf5(
        output,
        config=config,
        metadata=metadata,
        metrics=metrics,
        config_yaml=config_to_yaml(config),
    )
    runner._validate_external_files(
        run_dir,
        output,
        config_yaml=config_to_yaml(config),
        metadata=metadata,
        metrics=metrics,
        figure_paths=figures,
        config=config,
    )


def test_r1_runner_persists_compact_metrics_and_ten_figure_contract(
    tmp_path: Path,
) -> None:
    config = _config(r1=True)
    diagnostics = _r1_metrics(config)
    runner._validate_r1_metrics(config, diagnostics)
    metrics = {
        "experiment_status": "Inconclusive",
        "diagnostics_r1": diagnostics,
    }
    metadata = runner._metadata(
        config,
        Path("r1.yaml"),
        tmp_path / "run",
        "Inconclusive",
        diagnostics_r1_metrics=diagnostics,
    )
    run_dir, output, figures = _write_valid_run(
        tmp_path, config, metrics, metadata
    )

    assert len(runner._expected_figure_filenames(config)) == 10
    assert metadata["diagnostic_stage"] == "R1"
    assert metadata["diagnostics_r1_status"] == "Passed"
    runner._validate_hdf5(
        output,
        config=config,
        metadata=metadata,
        metrics=metrics,
        config_yaml=config_to_yaml(config),
    )
    runner._validate_external_files(
        run_dir,
        output,
        config_yaml=config_to_yaml(config),
        metadata=metadata,
        metrics=metrics,
        figure_paths=figures,
        config=config,
    )
    with h5py.File(output, "r") as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        assert "entry/metrics/diagnostics_r1" in h5
        assert "entry/truth/diagnostics_r1" not in h5


def test_r2_runner_persists_metrics_and_twelve_figure_contract(
    tmp_path: Path,
) -> None:
    config = _r2_config()
    r1_diagnostics = _r1_metrics(config)
    r2_diagnostics = _r2_metrics(config)
    runner._validate_r2_metrics(config, r2_diagnostics)
    metrics = {
        "experiment_status": "Inconclusive",
        "diagnostics_r1": r1_diagnostics,
        "diagnostics_r2": r2_diagnostics,
    }
    metadata = runner._metadata(
        config,
        Path("r2.yaml"),
        tmp_path / "run",
        "Inconclusive",
        diagnostics_r1_metrics=r1_diagnostics,
        diagnostics_r2_metrics=r2_diagnostics,
    )
    run_dir, output, figures = _write_valid_run(
        tmp_path, config, metrics, metadata
    )

    assert len(runner._expected_figure_filenames(config)) == 12
    assert metadata["diagnostic_stage"] == "R2"
    assert metadata["diagnostics_r1_status"] == "Passed"
    assert metadata["diagnostics_r2_status"] == "Passed"
    runner._validate_hdf5(
        output,
        config=config,
        metadata=metadata,
        metrics=metrics,
        config_yaml=config_to_yaml(config),
    )
    runner._validate_external_files(
        run_dir,
        output,
        config_yaml=config_to_yaml(config),
        metadata=metadata,
        metrics=metrics,
        figure_paths=figures,
        config=config,
    )
    with h5py.File(output, "r") as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        assert "entry/metrics/diagnostics_r1" in h5
        assert "entry/metrics/diagnostics_r2" in h5
        assert "entry/truth/diagnostics_r2" not in h5


@pytest.mark.parametrize("failure", ["method", "pair", "finite", "status"])
def test_r1_metric_validator_rejects_semantic_mismatch(failure: str) -> None:
    config = _config(r1=True)
    diagnostics = _r1_metrics(config)
    if failure == "method":
        diagnostics["methods"]["external_field_padding"] = "direct_zero"
    elif failure == "pair":
        diagnostics["external_padding"]["acceptance_pair_shapes"] = [
            [192, 192],
            [224, 224],
        ]
    elif failure == "finite":
        diagnostics["all_finite"] = False
    else:
        diagnostics["status"] = "Complete"

    with pytest.raises(RuntimeError):
        runner._validate_r1_metrics(config, diagnostics)


@pytest.mark.parametrize("failure", ["method", "pair", "finite", "status"])
def test_r2_metric_validator_rejects_semantic_mismatch(failure: str) -> None:
    config = _r2_config()
    diagnostics = _r2_metrics(config)
    if failure == "method":
        diagnostics["methods"]["alias_controlled_asm"] = "unregistered"
    elif failure == "pair":
        diagnostics["period_aligned"]["current_asm"][
            "acceptance_pair_shapes"
        ] = [[192, 192], [384, 384]]
    elif failure == "finite":
        diagnostics["all_finite"] = False
    else:
        diagnostics["status"] = "Resolved"

    with pytest.raises(RuntimeError):
        runner._validate_r2_metrics(config, diagnostics)


def _r3_config() -> dict[str, Any]:
    config = load_config(
        Path(
            "configs/experiments/"
            "exp040_TGV_3d_multislice_r3_detector_path.yaml"
        )
    )
    config["optics"]["baseline_shape"] = [4, 4]
    config["scan"]["num_x"] = 1
    config["scan"]["num_y"] = 1
    config["sample_a"]["thickness_m"] = 2.0e-6
    config["output"]["hdf5_filename"] = "custom_r3.h5"
    return config


def _r3_metrics(config: dict[str, Any]) -> dict[str, Any]:
    r3 = config["diagnostics_r3"]
    sampling = r3["sampling"]
    factors = deepcopy(sampling["factors"])
    convergence = [0.02, 0.01, 0.0]
    branch_acceptance = {
        "point_sample": 0.01,
        "pixel_box_average": 0.008,
    }
    return {
        "version": "R3",
        "methods": deepcopy(r3["methods"]),
        "sampling": {
            "factors": factors,
            "dx_m": deepcopy(sampling["dx_m"]),
            "shapes": deepcopy(sampling["shapes"]),
            "external_fov_m": deepcopy(sampling["external_fov_m"]),
            "native_sample_offset_px": deepcopy(
                sampling["native_sample_offsets_px"]
            ),
            "physical_origin_compensation_m": deepcopy(
                sampling["physical_origin_compensation_m"]
            ),
            "native_roi_shape": deepcopy(sampling["native_roi_shape"]),
            "scan_count": 1,
            "full_detector_stacks_retained": False,
        },
        "canonical_b_validation": {"pass": True},
        "a_exit_native_recovery": {"pass": True},
        "alias_masks": {"pass": True},
        "spectra": {"factors": factors},
        "bc_propagation": {"factors": factors},
        "detector_sampling": {
            "factors": factors,
            "acceptance_pair_factors": deepcopy(
                sampling["acceptance_pair_factors"]
            ),
            "relative_to_factor4": {
                "P_B": convergence,
                "detector": {
                    method: {
                        "point_sample": convergence,
                        "pixel_box_average": convergence,
                    }
                    for method in ("current_asm", "alias_controlled")
                },
            },
            "acceptance": {
                "P_B": 0.005,
                "detector": {
                    "current_asm": deepcopy(branch_acceptance),
                    "alias_controlled": deepcopy(branch_acceptance),
                },
            },
            "P_B_pass": True,
            "primary_detector_pass": True,
        },
        "detector_operator_difference": {
            "factors": factors,
            "point_vs_pixel_relative_l2": {
                "current_asm": [0.02, 0.02, 0.02],
                "alias_controlled": [0.01, 0.01, 0.01],
            },
            "selected_factor": 4,
            "selected_scan_index": 0,
        },
        "pixel_operator_controls": {"pass": True},
        "determinism": {
            "selected_factor": 4,
            "selected_scan_index": 0,
            "primary_relative_l2": 0.0,
            "pass": True,
        },
        "thresholds": {
            "convergence_relative_l2_max": 0.05,
            "spectral_and_method_material_relative_l2_min": 0.05,
            "mapping_and_pixel_relative_max": 1e-12,
            "determinism_relative_l2_max": 1e-14,
        },
        "outcome_flags": {
            "interpretation_code": "detector_path_sampling_converged"
        },
        "all_finite": True,
        "all_intensity_nonnegative": True,
        "hard_checks_pass": True,
        "status": "Passed",
    }


def test_r3_runner_persists_metrics_and_fifteen_figure_contract(
    tmp_path: Path,
) -> None:
    config = _r3_config()
    r1_diagnostics = _r1_metrics(config)
    r2_diagnostics = _r2_metrics(config)
    r3_diagnostics = _r3_metrics(config)
    runner._validate_r3_metrics(config, r3_diagnostics)
    metrics = {
        "experiment_status": "Inconclusive",
        "diagnostics_r1": r1_diagnostics,
        "diagnostics_r2": r2_diagnostics,
        "diagnostics_r3": r3_diagnostics,
    }
    metadata = runner._metadata(
        config,
        Path("r3.yaml"),
        tmp_path / "run",
        "Inconclusive",
        diagnostics_r1_metrics=r1_diagnostics,
        diagnostics_r2_metrics=r2_diagnostics,
        diagnostics_r3_metrics=r3_diagnostics,
    )
    run_dir, output, figures = _write_valid_run(
        tmp_path, config, metrics, metadata
    )

    assert len(runner._expected_figure_filenames(config)) == 15
    assert metadata["diagnostic_stage"] == "R3"
    assert metadata["diagnostics_r1_status"] == "Passed"
    assert metadata["diagnostics_r2_status"] == "Passed"
    assert metadata["diagnostics_r3_status"] == "Passed"
    runner._validate_hdf5(
        output,
        config=config,
        metadata=metadata,
        metrics=metrics,
        config_yaml=config_to_yaml(config),
    )
    runner._validate_external_files(
        run_dir,
        output,
        config_yaml=config_to_yaml(config),
        metadata=metadata,
        metrics=metrics,
        figure_paths=figures,
        config=config,
    )
    with h5py.File(output, "r") as h5:
        assert set(h5["entry"]) == {
            "config_yaml",
            "data",
            "instrument",
            "metadata",
            "metrics",
            "sample",
            "truth",
        }
        assert "entry/metrics/diagnostics_r3" in h5
        assert "entry/truth/diagnostics_r3" not in h5


def test_r3_metric_validator_rejects_sampling_origin_mismatch() -> None:
    config = _r3_config()
    diagnostics = _r3_metrics(config)
    diagnostics["sampling"]["native_sample_offset_px"] = [0, 1, 1]

    with pytest.raises(RuntimeError, match="native_sample_offset"):
        runner._validate_r3_metrics(config, diagnostics)
