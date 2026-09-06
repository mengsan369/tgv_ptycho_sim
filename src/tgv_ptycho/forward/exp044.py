"""Finite-master-B and localized-illumination operators for exp044.

The module composes the exp031 physical scan-window contract with the exp040
scalar multislice and exp042 q4 detector conventions.  Array axes are
``(y, x)``; scan columns are ``(x, y)`` in meters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.forward.camera import positive_midpoint_pixel_average
from tgv_ptycho.forward.exp040 import center_crop, relative_l2
from tgv_ptycho.forward.multislice_A import multislice_propagate_streamed_A
from tgv_ptycho.forward.scan_windows import (
    ScanWindowPlan,
    extract_scan_window,
    scatter_add_scan_windows,
)
from tgv_ptycho.objects.sample_b import (
    PhysicalPhaseCellMap,
    make_physical_phase_cell_map,
    rasterize_physical_phase_cells,
)
from tgv_ptycho.objects.tgv3d import make_tgv_air_fraction_slice
from tgv_ptycho.objects.tgv_geometry import diameter_profile, midpoint_z_grid
from tgv_ptycho.optics.angular_spectrum import (
    angular_spectrum_propagate,
    apply_angular_spectrum_transfer,
    make_angular_spectrum_transfer,
)

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]


def _shape(value: Any, name: str) -> tuple[int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{name} must be a positive (ny, nx) pair.")
    return shape


def center_embed(values: NDArray[np.generic], shape: tuple[int, int]) -> NDArray[Any]:
    """Center a 2D field in a larger same-parity zero array."""

    source = np.asarray(values)
    target = _shape(shape, "shape")
    if source.ndim != 2:
        raise ValueError("values must be a 2D array.")
    delta = (target[0] - source.shape[0], target[1] - source.shape[1])
    if min(delta) < 0 or delta[0] % 2 or delta[1] % 2:
        raise ValueError("shape must contain values with aligned centers.")
    return np.pad(
        source,
        ((delta[0] // 2, delta[0] // 2), (delta[1] // 2, delta[1] // 2)),
        mode="constant",
    )


def gaussian_incident_field(
    shape: tuple[int, int],
    dx_m: float,
    diameter_1e2_intensity_m: float,
    total_power: float,
    *,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray[np.complex128], dict[str, float]]:
    r"""Return ``A0 exp(-r^2/w^2)`` with fixed analytic total power.

    ``diameter_1e2_intensity_m`` is ``2*w``.  The infinite-plane analytic
    power is ``pi*w^2*A0^2/2``.  The returned captured fraction is the
    pixel-center quadrature on the finite requested canvas.
    """

    ny, nx = _shape(shape, "shape")
    dx = float(dx_m)
    diameter = float(diameter_1e2_intensity_m)
    power = float(total_power)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_m must be finite and positive.")
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("diameter_1e2_intensity_m must be finite and positive.")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("total_power must be finite and positive.")
    center_x, center_y = (float(value) for value in center_xy_m)
    if not np.all(np.isfinite([center_x, center_y])):
        raise ValueError("center_xy_m must be finite.")
    w = 0.5 * diameter
    amplitude = float(np.sqrt(2.0 * power / (np.pi * w**2)))
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * dx - center_x
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * dx - center_y
    yy, xx = np.meshgrid(y, x, indexing="ij")
    field = amplitude * np.exp(-(xx**2 + yy**2) / w**2)
    captured = float(np.sum(field**2, dtype=np.float64) * dx**2)
    return np.asarray(field, dtype=np.complex128), {
        "waist_radius_m": w,
        "diameter_1e2_intensity_m": diameter,
        "center_amplitude": amplitude,
        "center_intensity": amplitude**2,
        "analytic_total_power": power,
        "captured_power": captured,
        "captured_power_fraction": captured / power,
        "peak_sampled_intensity": float(np.max(field**2)),
    }


def residual_edge_energy_fraction(
    values: ComplexArray, *, ring_width_px: int = 8
) -> float:
    """Return the fraction of field energy in a rectangular outer ring."""

    field = np.asarray(values, dtype=np.complex128)
    width = int(ring_width_px)
    if field.ndim != 2 or width <= 0 or 2 * width >= min(field.shape):
        raise ValueError("field/ring_width_px do not define a nonempty inner region.")
    energy = np.abs(field) ** 2
    inner = energy[width:-width, width:-width]
    return float((np.sum(energy) - np.sum(inner)) / max(np.sum(energy), 1.0e-300))


def build_localized_multislice_probe(
    config: Mapping[str, Any],
    *,
    open_shape: tuple[int, int] | None = None,
    zero_contrast: bool = False,
    diameter_1e2_intensity_m: float | None = None,
    incident_type: str = "gaussian",
) -> dict[str, Any]:
    """Propagate an A-entrance field through the selected scalar model.

    The homogeneous Gaussian reference is propagated separately.  The TGV
    contribution is propagated as ``A_exit - homogeneous_exit`` and is added
    back at B.  No B-plane envelope multiplication is used.  ``plane_wave``
    is the exact unit-field organization bridge for the C0/C1 controls.
    """

    optics = config["optics"]
    sample = config["sample_a"]
    illumination = config["illumination"]
    native_shape = _shape(sample["native_shape"], "sample_a.native_shape")
    shape = (
        _shape(open_shape, "open_shape")
        if open_shape is not None
        else _shape(sample["open_shape"], "sample_a.open_shape")
    )
    dx = float(sample["dx_m"])
    wavelength = float(optics["wavelength_m"])
    n_ref = float(optics["internal_reference_index"])
    n_external = float(optics["external_medium_index"])
    thickness = float(sample["thickness_m"])
    diameter = (
        float(illumination["spot_diameter_1e2_intensity_m"])
        if diameter_1e2_intensity_m is None
        else float(diameter_1e2_intensity_m)
    )
    if incident_type == "gaussian":
        incident, power = gaussian_incident_field(
            shape,
            dx,
            diameter,
            float(illumination["analytic_total_power_reference_m2"]),
            center_xy_m=tuple(float(v) for v in illumination["center_xy_m"]),
        )
    elif incident_type == "plane_wave":
        incident = np.ones(shape, dtype=np.complex128)
        captured = float(np.prod(shape) * dx**2)
        power = {
            "waist_radius_m": 0.0,
            "diameter_1e2_intensity_m": 0.0,
            "center_amplitude": 1.0,
            "center_intensity": 1.0,
            "analytic_total_power": captured,
            "captured_power": captured,
            "captured_power_fraction": 1.0,
            "peak_sampled_intensity": 1.0,
            "is_exact_plane_wave": True,
        }
    else:
        raise ValueError("incident_type must be gaussian or plane_wave.")
    z_m, widths = midpoint_z_grid(thickness, float(sample["target_dz_m"]))
    diameters = diameter_profile(
        z_m,
        thickness,
        float(sample["d_top_m"]),
        float(sample["d_waist_m"]),
        float(sample["d_bottom_m"]),
        float(sample["z_waist_m"]),
    )
    n_glass = float(sample["n_glass"])
    n_air = n_glass if zero_contrast else float(sample["n_air"])
    interface_factor = int(sample["interface_factor"])
    center_xy = tuple(float(v) for v in sample["center_xy_m"])

    def slices() -> Any:
        for local_diameter in diameters:
            fraction = make_tgv_air_fraction_slice(
                shape, dx, float(local_diameter), interface_factor, center_xy
            )
            yield n_glass + fraction * (n_air - n_glass)

    homogeneous_exit = angular_spectrum_propagate(
        incident,
        dx,
        wavelength,
        thickness,
        n=n_ref,
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=False,
    )
    a_exit = multislice_propagate_streamed_A(
        incident,
        slices(),
        dx,
        widths,
        wavelength,
        n_ref=n_ref,
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=False,
    )
    transfer_ab = make_angular_spectrum_transfer(
        shape,
        dx,
        wavelength,
        float(optics["z_AB_m"]),
        n=n_external,
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=bool(optics["alias_control_external"]),
    )
    reference_probe = apply_angular_spectrum_transfer(homogeneous_exit, transfer_ab)
    a_residual = np.asarray(a_exit - homogeneous_exit, dtype=np.complex128)
    probe_residual = apply_angular_spectrum_transfer(a_residual, transfer_ab)
    probe = np.asarray(reference_probe + probe_residual, dtype=np.complex128)
    direct_probe = apply_angular_spectrum_transfer(a_exit, transfer_ab)
    transfer_bc = make_angular_spectrum_transfer(
        shape,
        dx,
        wavelength,
        float(optics["z_BC_m"]),
        n=n_external,
        bandlimit=bool(optics["angular_spectrum_bandlimit"]),
        alias_control=bool(optics["alias_control_external"]),
    )
    reference_detector = apply_angular_spectrum_transfer(reference_probe, transfer_bc)
    zero_contrast_error = relative_l2(a_exit, homogeneous_exit)
    return {
        "incident_open": incident,
        "U_A_exit_open": np.asarray(a_exit, dtype=np.complex128),
        "U_A_homogeneous_open": np.asarray(homogeneous_exit, dtype=np.complex128),
        "U_A_residual_open": a_residual,
        "P_B_open": probe,
        "P_B_reference_open": np.asarray(reference_probe, dtype=np.complex128),
        "P_B_residual_open": np.asarray(probe_residual, dtype=np.complex128),
        "P_B_native": np.asarray(center_crop(probe, native_shape), dtype=np.complex128),
        "P_B_reference_native": np.asarray(
            center_crop(reference_probe, native_shape), dtype=np.complex128
        ),
        "U_A_exit_native": np.asarray(center_crop(a_exit, native_shape)),
        "transfer_BC": np.asarray(transfer_bc, dtype=np.complex128),
        "P_detector_reference_open": np.asarray(
            reference_detector, dtype=np.complex128
        ),
        "z_m": np.asarray(z_m, dtype=np.float64),
        "slice_widths_m": np.asarray(widths, dtype=np.float64),
        "D_z_m": np.asarray(diameters, dtype=np.float64),
        "power": power,
        "controls": {
            "incident_type": incident_type,
            "zero_contrast": bool(zero_contrast),
            "zero_contrast_relative_l2": zero_contrast_error,
            "reference_plus_residual_A_to_B_relative_l2": relative_l2(
                probe, direct_probe
            ),
            "A_residual_edge_energy_fraction": residual_edge_energy_fraction(
                a_residual
            ),
            "P_B_residual_edge_energy_fraction": residual_edge_energy_fraction(
                probe_residual
            ),
            "all_finite": bool(
                all(
                    np.all(np.isfinite(value))
                    for value in (incident, a_exit, reference_probe, probe, transfer_bc)
                )
            ),
        },
    }


def build_nested_master_b(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a finite master whose centered legacy core is byte-identical."""

    settings = config["sample_b"]
    master_cells_shape = _shape(settings["master_cells_shape"], "master_cells_shape")
    core_cells_shape = _shape(settings["core_cells_shape"], "core_cells_shape")
    pairs = zip(core_cells_shape, master_cells_shape, strict=True)
    if any(core > master for core, master in pairs):
        raise ValueError("core_cells_shape must fit inside master_cells_shape.")
    pairs = zip(core_cells_shape, master_cells_shape, strict=True)
    if any((master - core) % 2 for core, master in pairs):
        raise ValueError("master/core cell maps must have aligned centers.")
    feature = float(settings["feature_size_m"])
    phase_range = float(settings["phase_range_rad"])
    extension = make_physical_phase_cell_map(
        master_cells_shape,
        feature_size_m=feature,
        phase_range_rad=phase_range,
        seed=int(settings["extension_seed"]),
    )
    core = make_physical_phase_cell_map(
        core_cells_shape,
        feature_size_m=feature,
        phase_range_rad=phase_range,
        seed=int(settings["core_seed"]),
    )
    phase = extension.phase_rad.copy()
    start_y = (master_cells_shape[0] - core_cells_shape[0]) // 2
    start_x = (master_cells_shape[1] - core_cells_shape[1]) // 2
    phase[
        start_y : start_y + core_cells_shape[0],
        start_x : start_x + core_cells_shape[1],
    ] = core.phase_rad
    master_map = PhysicalPhaseCellMap(
        phase_rad=phase,
        feature_size_m=feature,
        origin_xy_m=extension.origin_xy_m,
        phase_range_rad=phase_range,
        seed=int(settings["extension_seed"]),
    )
    master_shape = _shape(settings["master_shape"], "master_shape")
    dx = float(settings["dx_m"])
    master = rasterize_physical_phase_cells(master_map, master_shape, dx)
    core_shape_px = tuple(
        int(round(length * feature / dx)) for length in core_cells_shape
    )
    core_raster = rasterize_physical_phase_cells(core, core_shape_px, dx)
    return {
        "B_master_true": master,
        "B_core_true": core_raster,
        "phase_cells_rad": phase,
        "core_phase_cells_rad": core.phase_rad,
        "master_map": master_map,
    }


@dataclass(frozen=True)
class FiniteMasterBlindProbeBOperator:
    """Matched q4 operator with a finite master-B extract/scatter pair.

    The master parameter array and detector open grid share ``open_shape`` in
    the formal exp044 design.  Each physical scan extracts an interaction
    patch from the master and embeds it in the detector open grid.  The exact
    B adjoint crops that interaction patch and scatter-adds it to the master.
    """

    positions_m: NDArray[np.float64]
    node_dx_m: float
    quadrature_factor: int
    native_shape: tuple[int, int]
    open_shape: tuple[int, int]
    detector_roi_shape: tuple[int, int]
    homogeneous_probe_native: NDArray[np.complex128]
    homogeneous_probe_open: NDArray[np.complex128]
    homogeneous_detector_open: NDArray[np.complex128]
    transfer_bc: NDArray[np.complex128]
    scan_plan: ScanWindowPlan
    support_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.scan_plan.large_shape != self.open_shape:
            raise ValueError("scan_plan.large_shape must equal open_shape.")
        if tuple(self.support_mask.shape) != self.open_shape:
            raise ValueError("support_mask must match open_shape.")
        if not np.array_equal(self.support_mask, self.scan_plan.visited_region):
            raise ValueError("support_mask must be the exact scan-window union.")
        if self.homogeneous_probe_native.shape != self.native_shape:
            raise ValueError("homogeneous_probe_native has the wrong shape.")
        for field in (
            self.homogeneous_probe_open,
            self.homogeneous_detector_open,
            self.transfer_bc,
        ):
            if field.shape != self.open_shape:
                raise ValueError("open reference fields have the wrong shape.")

    @property
    def data_shape(self) -> tuple[int, int, int]:
        return (len(self.positions_m), *self.detector_roi_shape)

    def project_modulation(self, modulation: ComplexArray) -> NDArray[np.complex128]:
        values = np.asarray(modulation, dtype=np.complex128)
        if values.shape != self.open_shape or not np.all(np.isfinite(values)):
            raise ValueError("B modulation must be finite and master-shaped.")
        return np.asarray(np.where(self.support_mask, values, 0.0j))

    def probe_open(self, probe_native: ComplexArray) -> NDArray[np.complex128]:
        probe = np.asarray(probe_native, dtype=np.complex128)
        if probe.shape != self.native_shape or not np.all(np.isfinite(probe)):
            raise ValueError("probe_native must be finite and native-shaped.")
        delta = center_embed(probe - self.homogeneous_probe_native, self.open_shape)
        return np.asarray(self.homogeneous_probe_open + delta, dtype=np.complex128)

    def shifted_modulation(
        self, modulation: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        master = self.project_modulation(modulation)
        patch = extract_scan_window(master, self.scan_plan, int(scan_index))
        return np.asarray(center_embed(patch, self.open_shape), dtype=np.complex128)

    def _readout_intensity(self, field: ComplexArray) -> NDArray[np.float64]:
        return np.asarray(
            positive_midpoint_pixel_average(
                np.abs(np.asarray(field, dtype=np.complex128)) ** 2,
                self.quadrature_factor,
            ),
            dtype=np.float64,
        )

    def _readout_direction(
        self, field: ComplexArray, direction: ComplexArray
    ) -> NDArray[np.float64]:
        node = 2.0 * np.real(
            np.conj(np.asarray(field, dtype=np.complex128))
            * np.asarray(direction, dtype=np.complex128)
        )
        return np.asarray(
            positive_midpoint_pixel_average(node, self.quadrature_factor),
            dtype=np.float64,
        )

    def _readout_adjoint(
        self, field: ComplexArray, pixels: FloatArray
    ) -> NDArray[np.complex128]:
        values = np.asarray(pixels, dtype=np.float64)
        repeated = np.repeat(
            np.repeat(values, self.quadrature_factor, axis=-2),
            self.quadrature_factor,
            axis=-1,
        ) / float(self.quadrature_factor**2)
        return np.asarray(2.0 * repeated * field, dtype=np.complex128)

    def detector_field(
        self, probe_native: ComplexArray, modulation: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        probe = self.probe_open(probe_native)
        delta_probe = probe - self.homogeneous_probe_open
        residual_exit = delta_probe + probe * self.shifted_modulation(
            modulation, scan_index
        )
        return np.asarray(
            self.homogeneous_detector_open
            + apply_angular_spectrum_transfer(residual_exit, self.transfer_bc),
            dtype=np.complex128,
        )

    def predict_stack(
        self, probe_native: ComplexArray, modulation: ComplexArray
    ) -> NDArray[np.float64]:
        return np.stack(
            [
                center_crop(
                    self._readout_intensity(
                        self.detector_field(probe_native, modulation, index)
                    ),
                    self.detector_roi_shape,
                )
                for index in range(len(self.positions_m))
            ]
        ).astype(np.float64)

    def probe_field_direction(
        self, direction_native: ComplexArray, modulation: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        direction = center_embed(
            np.asarray(direction_native, dtype=np.complex128), self.open_shape
        )
        transmission = 1.0 + self.shifted_modulation(modulation, scan_index)
        return np.asarray(
            apply_angular_spectrum_transfer(
                transmission * direction, self.transfer_bc
            ),
            dtype=np.complex128,
        )

    def b_field_direction(
        self, probe_native: ComplexArray, direction: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        patch_direction = self.shifted_modulation(direction, scan_index)
        return np.asarray(
            apply_angular_spectrum_transfer(
                self.probe_open(probe_native) * patch_direction, self.transfer_bc
            ),
            dtype=np.complex128,
        )

    def intensity_jacobian_direction(
        self,
        probe_native: ComplexArray,
        modulation: ComplexArray,
        probe_direction: ComplexArray | None = None,
        b_direction: ComplexArray | None = None,
    ) -> NDArray[np.float64]:
        if probe_direction is None and b_direction is None:
            raise ValueError("At least one direction is required.")
        frames = []
        for index in range(len(self.positions_m)):
            field = self.detector_field(probe_native, modulation, index)
            direction = np.zeros(self.open_shape, dtype=np.complex128)
            if probe_direction is not None:
                direction += self.probe_field_direction(
                    probe_direction, modulation, index
                )
            if b_direction is not None:
                direction += self.b_field_direction(probe_native, b_direction, index)
            frames.append(
                center_crop(
                    self._readout_direction(field, direction),
                    self.detector_roi_shape,
                )
            )
        return np.asarray(np.stack(frames), dtype=np.float64)

    def intensity_jacobian_adjoint(
        self,
        probe_native: ComplexArray,
        modulation: ComplexArray,
        detector_values: FloatArray,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        residual = np.asarray(detector_values, dtype=np.float64)
        if residual.shape != self.data_shape or not np.all(np.isfinite(residual)):
            raise ValueError("detector_values must be finite and data-shaped.")
        full_pixel_shape = tuple(
            length // self.quadrature_factor for length in self.open_shape
        )
        probe_open = self.probe_open(probe_native)
        probe_adjoint = np.zeros(self.native_shape, dtype=np.complex128)
        patch_adjoints = np.zeros(
            (len(self.positions_m), *self.scan_plan.window_shape),
            dtype=np.complex128,
        )
        for index in range(len(self.positions_m)):
            field = self.detector_field(probe_native, modulation, index)
            pixels_full = center_embed(residual[index], full_pixel_shape)
            detector_adjoint = self._readout_adjoint(field, pixels_full)
            exit_adjoint = apply_angular_spectrum_transfer(
                detector_adjoint, np.conj(self.transfer_bc)
            )
            transmission = 1.0 + self.shifted_modulation(modulation, index)
            probe_adjoint += center_crop(
                np.conj(transmission) * exit_adjoint, self.native_shape
            )
            patch_adjoints[index] = center_crop(
                np.conj(probe_open) * exit_adjoint, self.scan_plan.window_shape
            )
        b_adjoint = scatter_add_scan_windows(patch_adjoints, self.scan_plan)
        return (
            np.asarray(probe_adjoint, dtype=np.complex128),
            self.project_modulation(b_adjoint),
        )

    def loss_and_gradients(
        self, probe_native: ComplexArray, modulation: ComplexArray, measured: FloatArray
    ) -> tuple[
        float,
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.float64],
    ]:
        prediction = self.predict_stack(probe_native, modulation)
        data = np.asarray(measured, dtype=np.float64)
        if data.shape != self.data_shape:
            raise ValueError("measured must match data_shape.")
        residual = prediction - data
        p_adjoint, b_adjoint = self.intensity_jacobian_adjoint(
            probe_native, modulation, residual
        )
        normalization = float(data.size)
        return (
            0.5 * float(np.mean(residual**2, dtype=np.float64)),
            np.asarray(p_adjoint / normalization, dtype=np.complex128),
            np.asarray(b_adjoint / normalization, dtype=np.complex128),
            prediction,
        )


@dataclass(frozen=True)
class FiniteMasterKnownBProbeOperator:
    """Known-B view of :class:`FiniteMasterBlindProbeBOperator`."""

    blind: FiniteMasterBlindProbeBOperator
    modulation: NDArray[np.complex128]

    @property
    def positions_m(self) -> NDArray[np.float64]:
        return self.blind.positions_m

    @property
    def node_dx_m(self) -> float:
        return self.blind.node_dx_m

    @property
    def quadrature_factor(self) -> int:
        return self.blind.quadrature_factor

    @property
    def native_shape(self) -> tuple[int, int]:
        return self.blind.native_shape

    @property
    def open_shape(self) -> tuple[int, int]:
        return self.blind.open_shape

    @property
    def detector_roi_shape(self) -> tuple[int, int]:
        return self.blind.detector_roi_shape

    @property
    def homogeneous_probe_native(self) -> NDArray[np.complex128]:
        return self.blind.homogeneous_probe_native

    def detector_field(
        self, probe_native: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        return self.blind.detector_field(probe_native, self.modulation, scan_index)

    def predict_stack(self, probe_native: ComplexArray) -> NDArray[np.float64]:
        return self.blind.predict_stack(probe_native, self.modulation)

    def linear_detector_field(
        self, direction_native: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        return self.blind.probe_field_direction(
            direction_native, self.modulation, scan_index
        )

    def linear_detector_field_adjoint(
        self, detector_delta: ComplexArray, scan_index: int
    ) -> NDArray[np.complex128]:
        delta = np.asarray(detector_delta, dtype=np.complex128)
        back = apply_angular_spectrum_transfer(
            delta, np.conj(self.blind.transfer_bc)
        )
        transmission = 1.0 + self.blind.shifted_modulation(
            self.modulation, scan_index
        )
        return np.asarray(
            center_crop(np.conj(transmission) * back, self.native_shape),
            dtype=np.complex128,
        )

    def intensity_jacobian_direction(
        self, probe_native: ComplexArray, direction_native: ComplexArray
    ) -> NDArray[np.float64]:
        return self.blind.intensity_jacobian_direction(
            probe_native, self.modulation, probe_direction=direction_native
        )

    def intensity_jacobian_adjoint(
        self, probe_native: ComplexArray, detector_values: FloatArray
    ) -> NDArray[np.complex128]:
        probe, _ = self.blind.intensity_jacobian_adjoint(
            probe_native, self.modulation, detector_values
        )
        return probe

    def loss_and_gradient(
        self, probe_native: ComplexArray, measured: FloatArray
    ) -> tuple[float, NDArray[np.complex128], NDArray[np.float64]]:
        loss, probe, _, prediction = self.blind.loss_and_gradients(
            probe_native, self.modulation, measured
        )
        return loss, probe, prediction


def operator_control_metrics(
    known: Any,
    blind: Any,
    probe: ComplexArray,
    modulation: ComplexArray,
    measured: FloatArray,
    *,
    seed: int,
    finite_difference_step: float = 1.0e-6,
) -> dict[str, float | bool]:
    """Return deterministic field/intensity adjoint and derivative controls."""

    rng = np.random.default_rng(int(seed))
    p = np.asarray(probe, dtype=np.complex128)
    m = np.asarray(modulation, dtype=np.complex128)
    dp = rng.normal(size=p.shape) + 1j * rng.normal(size=p.shape)
    dp = np.asarray(dp / np.linalg.norm(dp), dtype=np.complex128)
    dtheta = np.where(blind.support_mask, rng.normal(size=m.shape), 0.0)
    dtheta /= max(float(np.linalg.norm(dtheta)), 1.0e-300)
    dm = np.asarray(1j * (1.0 + m) * dtheta, dtype=np.complex128)
    detector_test = rng.normal(size=blind.data_shape)
    j_direction = blind.intensity_jacobian_direction(
        p, m, probe_direction=dp, b_direction=dm
    )
    p_adj, b_adj = blind.intensity_jacobian_adjoint(p, m, detector_test)
    left = float(np.sum(j_direction * detector_test, dtype=np.float64))
    right = float(
        np.real(
            np.sum(np.conj(dp) * p_adj, dtype=np.complex128)
            + np.sum(np.conj(dm) * b_adj, dtype=np.complex128)
        )
    )
    adjoint_error = abs(left - right) / max(abs(left), abs(right), 1.0e-300)
    step = float(finite_difference_step)
    plus = blind.predict_stack(p + step * dp, m + step * dm)
    minus = blind.predict_stack(p - step * dp, m - step * dm)
    finite_difference = (plus - minus) / (2.0 * step)
    directional_error = relative_l2(finite_difference, j_direction)

    probe_nodes = rng.normal(size=known.open_shape) + 1j * rng.normal(
        size=known.open_shape
    )
    linear = known.linear_detector_field(dp, 0)
    linear_adjoint = known.linear_detector_field_adjoint(probe_nodes, 0)
    linear_left = np.sum(np.conj(linear) * probe_nodes, dtype=np.complex128)
    linear_right = np.sum(np.conj(dp) * linear_adjoint, dtype=np.complex128)
    linear_error = float(
        abs(linear_left - linear_right)
        / max(abs(linear_left), abs(linear_right), 1.0e-300)
    )
    prediction = known.predict_stack(p)
    return {
        "combined_intensity_jacobian_real_adjoint_relative_error": float(
            adjoint_error
        ),
        "combined_intensity_jacobian_directional_relative_error": float(
            directional_error
        ),
        "propagation_linear_adjoint_relative_error": linear_error,
        "repeat_prediction_relative_l2": relative_l2(
            prediction, known.predict_stack(p)
        ),
        "intensity_all_finite": bool(np.all(np.isfinite(prediction))),
        "intensity_nonnegative": bool(np.min(prediction) >= 0.0),
        "forward_replay_relative_l2": relative_l2(prediction, measured),
    }


def random_direction_gain_metrics(
    blind: Any,
    probe: ComplexArray,
    modulation: ComplexArray,
    *,
    seed: int,
    count: int,
) -> dict[str, Any]:
    """Report a bounded truth-free random-direction Jacobian gain proxy."""

    if count < 1:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(int(seed))
    p = np.asarray(probe, dtype=np.complex128)
    m = np.asarray(modulation, dtype=np.complex128)
    probe_gains = []
    b_gains = []
    joint_gains = []
    for _ in range(int(count)):
        dp = rng.normal(size=p.shape) + 1j * rng.normal(size=p.shape)
        dp = np.asarray(dp / np.linalg.norm(dp), dtype=np.complex128)
        phase = np.where(blind.support_mask, rng.normal(size=m.shape), 0.0)
        phase /= max(float(np.linalg.norm(phase)), 1.0e-300)
        dm = np.asarray(1j * (1.0 + m) * phase, dtype=np.complex128)
        jp = blind.intensity_jacobian_direction(
            p, m, probe_direction=dp
        )
        jb = blind.intensity_jacobian_direction(p, m, b_direction=dm)
        probe_gains.append(float(np.sqrt(np.mean(jp**2))))
        b_gains.append(float(np.sqrt(np.mean(jb**2))))
        joint_gains.append(float(np.sqrt(np.mean((jp + jb) ** 2))))

    def summary(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "values": array,
            "minimum": float(np.min(array)),
            "median": float(np.median(array)),
            "maximum": float(np.max(array)),
            "near_zero_fraction_at_1e3_of_median": float(
                np.mean(array <= 1.0e-3 * max(float(np.median(array)), 1.0e-300))
            ),
        }

    return {
        "probe": summary(probe_gains),
        "sample_b_phase": summary(b_gains),
        "joint": summary(joint_gains),
        "seed": int(seed),
        "count": int(count),
        "truth_used_for_case_selection": False,
    }


__all__ = [
    "FiniteMasterBlindProbeBOperator",
    "FiniteMasterKnownBProbeOperator",
    "build_localized_multislice_probe",
    "build_nested_master_b",
    "center_embed",
    "gaussian_incident_field",
    "operator_control_metrics",
    "random_direction_gain_metrics",
    "residual_edge_energy_fraction",
]
