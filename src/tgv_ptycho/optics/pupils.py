"""Pupil and aperture helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.fields import make_circular_aperture


def make_circular_pupil(
    shape: tuple[int, int],
    dx: float | tuple[float, float],
    numerical_aperture: float,
    wavelength: float,
) -> NDArray[np.float64]:
    """Create a placeholder real-space circular pupil mask.

    TODO: Replace this with a Fourier-domain pupil model tied to NA and
    wavelength. The current helper keeps the API location ready.
    """

    radius = numerical_aperture * wavelength
    return make_circular_aperture(shape, dx, radius)
