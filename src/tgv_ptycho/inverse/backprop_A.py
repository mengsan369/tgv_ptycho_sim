"""Backpropagation from the B plane toward sample A."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate


def backpropagate_probe_to_A(
    P_B_rec: NDArray[np.complexfloating],
    dx: float | tuple[float, float],
    wavelength: float,
    z_AB: float,
) -> NDArray[np.complex128]:
    """Backpropagate recovered B-plane probe by `-z_AB`."""

    return angular_spectrum_propagate(P_B_rec, dx, wavelength, -z_AB)
