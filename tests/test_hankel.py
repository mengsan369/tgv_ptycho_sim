from __future__ import annotations

import numpy as np

from tgv_ptycho.forward.multislice_radial import (
    radial_multislice_contrast_propagate,
)
from tgv_ptycho.optics.hankel import make_qdht_plan, qdht_plan_controls


def test_qdht_roundtrip_and_scaled_parseval() -> None:
    plan = make_qdht_plan(64, 40.0e-6)
    controls = qdht_plan_controls(plan)

    assert controls["transform_involution_probe_relative_l2"] <= 1.0e-10
    assert controls["physical_roundtrip_relative_l2"] <= 1.0e-10
    assert controls["scaled_parseval_relative_error"] <= 1.0e-10
    assert controls["all_finite"] is True


def test_qdht_zero_distance_without_projection_is_identity() -> None:
    plan = make_qdht_plan(48, 30.0e-6)
    contrast = np.exp(-((plan.radial_nodes_m / 8.0e-6) ** 2)) * (
        0.2 + 0.1j
    )

    propagated = plan.propagate_contrast(
        contrast,
        wavelength_m=532.0e-9,
        distance_m=0.0,
        refractive_index=1.5,
        bandlimit=False,
    )

    np.testing.assert_allclose(propagated, contrast, rtol=1.0e-10, atol=1.0e-12)


def test_radial_multislice_preserves_zero_contrast_background() -> None:
    plan = make_qdht_plan(40, 25.0e-6)
    field, controls = radial_multislice_contrast_propagate(
        plan,
        np.asarray([1.0e-6, 1.0e-6]),
        np.asarray([0.25e-6, 0.25e-6]),
        wavelength_m=532.0e-9,
        n_glass=1.5,
        n_air=1.5,
        post_exit_air_distance_m=1.0e-6,
    )

    np.testing.assert_allclose(field, 1.0, rtol=0.0, atol=1.0e-14)
    assert controls["maximum_outer_transmission_error"] == 0.0
    assert controls["all_finite"] is True
