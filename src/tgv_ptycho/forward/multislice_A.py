"""Multi-slice propagation through sample A."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tgv_ptycho.optics.angular_spectrum import angular_spectrum_propagate


def multislice_propagate_A(
    incident_field: NDArray[np.complexfloating],
    n_volume: NDArray[np.floating],
    dx: float | tuple[float, float],
    dz: float,
    wavelength: float,
    n_ref: float = 1.5,
) -> NDArray[np.complex128]:
    r"""Propagate through a refractive-index volume with a thin-slice model.

    Each slice is converted to a transmission function

    ```text
    t_m(x, y) = exp(i * k0 * (n_m(x, y) - n_ref) * dz)
    ```

    and the field is propagated by `dz` between adjacent slices. The output is
    the field immediately after the final slice. This is a starting point, not
    yet a rigorously validated multi-slice solver.
    """

    field = np.asarray(incident_field, dtype=np.complex128).copy()
    n_vol = np.asarray(n_volume, dtype=np.float64)
    if n_vol.ndim != 3:
        msg = "n_volume must have shape (nz, ny, nx)."
        raise ValueError(msg)
    if field.shape != n_vol.shape[1:]:
        msg = "incident_field shape must match each n_volume slice."
        raise ValueError(msg)
    if dz <= 0:
        msg = "dz must be positive."
        raise ValueError(msg)

    k0 = 2.0 * np.pi / wavelength
    nz = n_vol.shape[0]
    for iz in range(nz):
        transmission = np.exp(1j * k0 * (n_vol[iz] - n_ref) * dz)
        field *= transmission
        if iz < nz - 1:
            field = angular_spectrum_propagate(field, dx, wavelength, dz, n=n_ref)
    return field.astype(np.complex128, copy=False)
