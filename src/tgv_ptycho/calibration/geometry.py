"""Experimental geometry calibration interfaces."""

from __future__ import annotations

from typing import Any


def build_geometry_metadata(
    wavelength_m: float,
    detector_pixel_size_m: float,
    dx_m: float,
    z_AB_m: float | None = None,
    z_BC_m: float | None = None,
    detector_distance_m: float | None = None,
) -> dict[str, Any]:
    """Build geometry metadata for the unified HDF5 instrument group.

    TODO: Validate consistency between effective sample-plane sampling,
    detector pixel size, propagation distances, and reconstruction geometry.
    """

    raise NotImplementedError("Geometry metadata validation is not implemented yet.")


def estimate_detector_distance(*args, **kwargs) -> dict[str, Any]:
    """Estimate detector distance from calibration measurements.

    TODO: Add calibration target or diffraction-ring based distance estimation.
    """

    raise NotImplementedError("Detector-distance calibration is not implemented yet.")
