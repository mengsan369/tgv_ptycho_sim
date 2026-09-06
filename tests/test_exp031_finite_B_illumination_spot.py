from __future__ import annotations

import json

import h5py
import numpy as np
import scripts.run_exp031_finite_B_illumination_spot as runner

from tgv_ptycho.forward.exp031 import (
    adjoint_detector_roi_to_active,
    gaussian_reference_radial,
    make_asm_transfer_complex64,
    make_open_shape,
    make_tgv_radial_probe_result,
    plane_wave_reference,
    propagate_residual_batch_to_roi,
    reference_plus_residual_identity_error,
)
from tgv_ptycho.inverse.exp031 import (
    gaussian_amplitude_for_total_power,
    gaussian_square_captured_power_fraction,
    gaussian_square_side_for_omitted_power,
    gaussian_total_power,
    poisson_fisher_per_incident_photon,
)


def test_gaussian_one_over_e2_diameter_and_power_normalizations() -> None:
    waist = 50.0e-6
    radius = np.asarray([0.0, waist])
    field = gaussian_reference_radial(
        radius,
        waist_radius_m=waist,
        amplitude=1.0,
        wavelength_m=532.0e-9,
        distance_m=0.0,
    )
    assert np.isclose(abs(field[1]) ** 2 / abs(field[0]) ** 2, np.exp(-2.0))
    amplitude = gaussian_amplitude_for_total_power(1.0, waist)
    assert np.isclose(gaussian_total_power(amplitude, waist), 1.0)
    assert np.isclose(gaussian_total_power(1.0, waist), np.pi * waist**2 / 2.0)


def test_formal_square_fov_has_requested_analytic_power() -> None:
    waist = 100.0e-6
    side = gaussian_square_side_for_omitted_power(waist, 1.0e-3)
    captured = gaussian_square_captured_power_fraction(side, waist)
    assert np.isclose(captured, 1.0 - 1.0e-3, rtol=1.0e-12)


def test_change06_case_matrix_separates_support_padding_and_reporting_roi() -> None:
    config = runner.load_config(
        runner.REPO_ROOT
        / "configs"
        / "experiments"
        / "exp031_finite_B_illumination_spot.yaml"
    )
    cases = runner._case_descriptors(config)
    by_id = {case["id"]: case for case in cases}

    formal_50 = by_id["gaussian_050um_formal"]
    control_50 = by_id["gaussian_050um_strict_fov"]
    assert formal_50["omitted_power_target"] == 1.0e-7
    assert control_50["omitted_power_target"] == 1.0e-8
    assert formal_50["support_basis"] == "residual_edge_convergence"
    assert control_50["active_shape"][0] > formal_50["active_shape"][0]

    padding_cases = {
        case["padding_guard_m_per_side"]: case
        for case in cases
        if case["id"]
        in {
            "gaussian_100um_padding_032um_control",
            "gaussian_100um_formal",
            "gaussian_100um_padding_128um_control",
        }
    }
    assert list(sorted(padding_cases)) == [32.0e-6, 64.0e-6, 128.0e-6]
    open_sizes = [
        padding_cases[guard]["open_shape"][0] for guard in sorted(padding_cases)
    ]
    assert open_sizes == sorted(open_sizes)
    assert len(set(open_sizes)) == 3

    comparison_shape = runner._probe_envelope_comparison_shape(config, cases)
    assert comparison_shape == (352, 352)
    geometry = runner._geometry(config, runner._make_scan(config))
    assert geometry["probe_envelope_comparison_shape"] == [352, 352]
    assert np.allclose(
        geometry["probe_envelope_comparison_size_yx_m"], [88.0e-6, 88.0e-6]
    )


def test_large_waist_gaussian_approaches_plane_wave_on_fixed_roi() -> None:
    radius = np.linspace(0.0, 50.0e-6, 100)
    gaussian = gaussian_reference_radial(
        radius,
        waist_radius_m=1.0,
        amplitude=1.0,
        wavelength_m=532.0e-9,
        distance_m=1.0e-3,
    )
    plane = plane_wave_reference(
        (1, radius.size),
        amplitude=1.0,
        wavelength_m=532.0e-9,
        distance_m=1.0e-3,
    )[0]
    assert np.linalg.norm(gaussian - plane) / np.linalg.norm(plane) < 1.0e-7


def test_reference_plus_residual_identity_and_compact_open_propagation() -> None:
    rng = np.random.default_rng(31)
    reference = np.ones((12, 16), dtype=np.complex128)
    delta = 0.01 * (
        rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)
    )
    sample_b = np.exp(0.2j * rng.normal(size=reference.shape))
    assert reference_plus_residual_identity_error(reference, delta, sample_b) < 1.0e-14

    open_shape = make_open_shape(reference.shape, 0.5e-6, 2.0e-6)
    transfer = make_asm_transfer_complex64(
        open_shape,
        dx_m=0.5e-6,
        wavelength_m=532.0e-9,
        distance_m=1.0e-3,
    )
    first = propagate_residual_batch_to_roi(
        delta, transfer, reference.shape, workers=1
    )[0]
    second = propagate_residual_batch_to_roi(
        delta[None, ...], transfer, reference.shape, workers=1
    )[0]
    assert np.array_equal(first, second)


def test_zero_contrast_compact_tgv_perturbation_is_zero() -> None:
    result = make_tgv_radial_probe_result(
        tgv={
            "thickness_m": 700.0e-6,
            "d_top_m": 50.0e-6,
            "d_waist_m": 33.3e-6,
            "d_bottom_m": 50.0e-6,
            "z_waist_m": 350.0e-6,
            "n_glass": 1.5,
            "n_air": 1.5,
        },
        wavelength_m=532.0e-9,
        z_AB_m=1.0e-3,
        radial_output_max_m=20.0e-6,
        radial_output_step_m=1.0e-6,
        finite_difference_step_m=0.5e-9,
        core_step_m=1.0e-6,
        transition_step_m=0.5e-6,
        gaussian_waist_radius_m=50.0e-6,
    )
    assert np.max(np.abs(result.probe_delta_baseline)) == 0.0
    assert np.max(np.abs(result.probe_delta_derivative_per_m)) == 0.0


def test_poisson_fisher_is_invariant_to_uniform_field_scaling() -> None:
    intensity = np.arange(1.0, 25.0).reshape(2, 3, 4)
    derivative = 0.2 * intensity
    first = poisson_fisher_per_incident_photon(
        intensity,
        derivative,
        pixel_area_m2=1.0,
        incident_power_per_frame=10.0,
        mu_epsilon=1.0e-15,
    )
    second = poisson_fisher_per_incident_photon(
        7.0 * intensity,
        7.0 * derivative,
        pixel_area_m2=1.0,
        incident_power_per_frame=70.0,
        mu_epsilon=1.0e-15,
    )
    assert np.isclose(
        first["fisher_information_per_incident_photon_per_m2"],
        second["fisher_information_per_incident_photon_per_m2"],
    )


def test_open_propagation_and_roi_crop_have_paired_adjoint() -> None:
    rng = np.random.default_rng(310)
    active_shape = (9, 11)
    open_shape = (16, 18)
    roi_shape = (7, 8)
    transfer = make_asm_transfer_complex64(
        open_shape,
        dx_m=0.5e-6,
        wavelength_m=532.0e-9,
        distance_m=1.0e-3,
    )
    source = (
        rng.normal(size=active_shape) + 1j * rng.normal(size=active_shape)
    ).astype(np.complex64)
    detector = (rng.normal(size=roi_shape) + 1j * rng.normal(size=roi_shape)).astype(
        np.complex64
    )
    forward = propagate_residual_batch_to_roi(source, transfer, roi_shape, workers=1)[0]
    adjoint = adjoint_detector_roi_to_active(
        detector, transfer, active_shape, workers=1
    )[0]
    left = np.sum(np.conj(forward) * detector, dtype=np.complex128)
    right = np.sum(np.conj(source) * adjoint, dtype=np.complex128)
    relative = abs(left - right) / max(abs(left), abs(right))
    assert relative < 2.0e-6


def test_exp031_hdf5_artifact_audit_layout(tmp_path) -> None:
    hdf5_path = tmp_path / "exp031.h5"
    metadata = {"experiment_id": "exp031", "shape": (2, 3)}
    metrics = {"status": "diagnostic", "values": (1.0, 2.0)}
    config_yaml = "experiment:\n  id: exp031\n"
    with h5py.File(hdf5_path, "w") as h5:
        entry = h5.require_group("entry")
        runner._write_h5(entry, "config_yaml", config_yaml)
        data = entry.require_group("data")
        data.create_dataset("scan_positions", data=np.zeros((1, 2)))
        data.create_dataset("I_stack", data=np.ones((1, 4, 4)))
        runner._write_h5(entry, "instrument", {"illumination": {"kind": "test"}})
        runner._write_h5(entry, "sample", {"sample_B_parameters": {"seed": 31}})
        runner._write_h5(
            entry,
            "truth",
            {
                "sample_B_phase_cells": np.zeros((2, 2)),
                "illumination_cases": {"spot_diameter_1e2_intensity_m": [100.0e-6]},
                "P_B_true": np.ones((4, 4), dtype=np.complex64),
            },
        )
        runner._write_h5(
            entry,
            "metadata",
            {**metadata, "json": runner._json_text(metadata)},
        )
        runner._write_h5(
            entry,
            "metrics",
            {
                "B_coverage": {"minimum_margin_m": 1.0},
                "illumination": {"captured": 1.0},
                "probe_sensitivity": {"value": 1.0},
                "detector_sensitivity": {"value": 1.0},
                "poisson_fisher": {"value": 1.0},
                "numerical_convergence": {"finite": True},
                "json": runner._json_text(metrics),
            },
        )
    audit = runner._audit_hdf5(
        hdf5_path,
        config_yaml,
        json.loads(runner._json_text(metadata)),
        json.loads(runner._json_text(metrics)),
        [],
        tmp_path,
    )
    assert audit["passed"]
    assert not audit["reconstruction_group_present"]
