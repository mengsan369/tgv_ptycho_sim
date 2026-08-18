"""Quasi-discrete Hankel transforms for axisymmetric scalar fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import jn_zeros, jv


@dataclass(frozen=True)
class QDHTPlan:
    """Precomputed Bessel-zero lattice and scaled QDHT matrix.

    The transform convention follows Guizar-Sicairos and Gutiérrez-Vega
    (JOSA A 21, 53--58, 2004).  ``forward`` and ``inverse`` return physical
    radial and spatial-frequency samples; ``propagate_contrast`` uses the
    equivalent scaled, nearly unitary representation internally.
    """

    order: int
    radial_max_m: float
    zeros: NDArray[np.float64]
    radial_nodes_m: NDArray[np.float64]
    radial_frequency_nodes_per_m: NDArray[np.float64]
    radial_wavenumber_nodes_per_m: NDArray[np.float64]
    transform_matrix: NDArray[np.float64]
    spatial_scale: NDArray[np.float64]
    frequency_scale: NDArray[np.float64]

    @property
    def sample_count(self) -> int:
        """Number of radial samples."""

        return int(self.radial_nodes_m.size)

    def forward(
        self, values: NDArray[np.complexfloating]
    ) -> NDArray[np.complex128]:
        """Apply the physical QDHT to samples on ``radial_nodes_m``."""

        field = _validated_vector(values, self.sample_count, "values")
        scaled = field / self.spatial_scale
        return np.asarray(
            self.frequency_scale * (self.transform_matrix @ scaled),
            dtype=np.complex128,
        )

    def inverse(
        self, spectrum: NDArray[np.complexfloating]
    ) -> NDArray[np.complex128]:
        """Apply the inverse physical QDHT."""

        values = _validated_vector(spectrum, self.sample_count, "spectrum")
        scaled = values / self.frequency_scale
        return np.asarray(
            self.spatial_scale * (self.transform_matrix @ scaled),
            dtype=np.complex128,
        )

    def propagate_contrast(
        self,
        contrast: NDArray[np.complexfloating],
        *,
        wavelength_m: float,
        distance_m: float,
        refractive_index: float,
        bandlimit: bool = True,
    ) -> NDArray[np.complex128]:
        """Propagate a boundary-decaying contrast in a carrier-normalized frame."""

        values = _validated_vector(contrast, self.sample_count, "contrast")
        wavelength = _positive(wavelength_m, "wavelength_m")
        index = _positive(refractive_index, "refractive_index")
        distance = float(distance_m)
        if not np.isfinite(distance):
            raise ValueError("distance_m must be finite.")
        if not isinstance(bandlimit, bool):
            raise TypeError("bandlimit must be a bool.")

        k_medium = 2.0 * np.pi * index / wavelength
        kr = self.radial_wavenumber_nodes_per_m
        kz_squared = k_medium**2 - kr**2
        if bandlimit:
            propagating = kz_squared >= 0.0
            transfer = np.zeros(self.sample_count, dtype=np.complex128)
            kz = np.sqrt(np.maximum(kz_squared[propagating], 0.0))
            transfer[propagating] = np.exp(
                1j * (kz - k_medium) * distance
            )
        else:
            kz = np.sqrt(kz_squared.astype(np.complex128))
            transfer = np.exp(1j * (kz - k_medium) * distance)

        spatial_scaled = values / self.spatial_scale
        spectral_scaled = self.transform_matrix @ spatial_scaled
        propagated_scaled = self.transform_matrix @ (
            transfer * spectral_scaled
        )
        return np.asarray(
            self.spatial_scale * propagated_scaled,
            dtype=np.complex128,
        )


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _validated_vector(
    values: NDArray[np.complexfloating], sample_count: int, name: str
) -> NDArray[np.complex128]:
    result = np.asarray(values, dtype=np.complex128)
    if result.shape != (sample_count,):
        raise ValueError(f"{name} must have shape ({sample_count},).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def make_qdht_plan(
    sample_count: int,
    radial_max_m: float,
    *,
    order: int = 0,
) -> QDHTPlan:
    """Construct an integer-order QDHT plan on a finite radial support."""

    count = int(sample_count)
    if count < 2 or count != sample_count:
        raise ValueError("sample_count must be an integer of at least two.")
    radius = _positive(radial_max_m, "radial_max_m")
    transform_order = int(order)
    if transform_order < 0 or transform_order != order:
        raise ValueError("order must be a non-negative integer.")

    all_zeros = np.asarray(
        jn_zeros(transform_order, count + 1), dtype=np.float64
    )
    alpha = all_zeros[:-1]
    alpha_boundary = float(all_zeros[-1])
    adjacent = np.asarray(jv(transform_order + 1, alpha), dtype=np.float64)
    denominator = alpha_boundary * np.outer(adjacent, adjacent)
    transform = 2.0 * jv(
        transform_order, np.outer(alpha, alpha) / alpha_boundary
    ) / denominator
    maximum_frequency = alpha_boundary / (2.0 * np.pi * radius)
    radial_nodes = alpha * radius / alpha_boundary
    frequency_nodes = alpha / (2.0 * np.pi * radius)
    spatial_scale = adjacent / radius
    frequency_scale = adjacent / maximum_frequency
    arrays = (
        all_zeros,
        radial_nodes,
        frequency_nodes,
        transform,
        spatial_scale,
        frequency_scale,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise RuntimeError("QDHT construction produced non-finite values.")
    return QDHTPlan(
        order=transform_order,
        radial_max_m=radius,
        zeros=all_zeros,
        radial_nodes_m=radial_nodes,
        radial_frequency_nodes_per_m=frequency_nodes,
        radial_wavenumber_nodes_per_m=2.0 * np.pi * frequency_nodes,
        transform_matrix=np.asarray(transform, dtype=np.float64),
        spatial_scale=np.asarray(spatial_scale, dtype=np.float64),
        frequency_scale=np.asarray(frequency_scale, dtype=np.float64),
    )


def qdht_plan_controls(plan: QDHTPlan) -> dict[str, float | int | bool]:
    """Return algebra and sampling controls for one plan."""

    probe = (
        np.exp(-((plan.radial_nodes_m / (0.21 * plan.radial_max_m)) ** 2))
        * (1.0 + 0.2j)
    )
    scaled_probe = probe / plan.spatial_scale
    twice_transformed = plan.transform_matrix @ (
        plan.transform_matrix @ scaled_probe
    )
    involution_error = float(
        np.linalg.norm(twice_transformed - scaled_probe)
        / np.linalg.norm(scaled_probe)
    )
    spectrum = plan.forward(probe)
    recovered = plan.inverse(spectrum)
    roundtrip = float(
        np.linalg.norm(recovered - probe) / np.linalg.norm(probe)
    )
    scaled_spatial = probe / plan.spatial_scale
    scaled_frequency = spectrum / plan.frequency_scale
    parseval = float(
        abs(
            np.vdot(scaled_spatial, scaled_spatial).real
            - np.vdot(scaled_frequency, scaled_frequency).real
        )
        / np.vdot(scaled_spatial, scaled_spatial).real
    )
    return {
        "order": plan.order,
        "sample_count": plan.sample_count,
        "radial_max_m": plan.radial_max_m,
        "maximum_radial_node_m": float(plan.radial_nodes_m[-1]),
        "maximum_radial_frequency_per_m": float(
            plan.radial_frequency_nodes_per_m[-1]
        ),
        "transform_involution_probe_relative_l2": involution_error,
        "physical_roundtrip_relative_l2": roundtrip,
        "scaled_parseval_relative_error": parseval,
        "all_finite": bool(
            np.all(np.isfinite(spectrum))
            and np.all(np.isfinite(recovered))
        ),
    }


__all__ = ["QDHTPlan", "make_qdht_plan", "qdht_plan_controls"]
