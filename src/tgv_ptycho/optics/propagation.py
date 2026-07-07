"""Propagation dispatch helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate
from tgv_ptycho.optics.fresnel import fresnel_propagate


def propagate(
    U: NDArray[np.complexfloating],
    dx: float | tuple[float, float],
    wavelength: float,
    z: float,
    n: float = 1.0,
    method: str = "angular_spectrum",
) -> NDArray[np.complexfloating]:
    """Propagate a field with the selected method."""

    if method == "angular_spectrum":
        return angular_spectrum_propagate(U, dx, wavelength, z, n=n)
    if method == "fresnel":
        return fresnel_propagate(U, dx, wavelength, z, n=n)
    msg = f"Unknown propagation method: {method}"
    raise ValueError(msg)
