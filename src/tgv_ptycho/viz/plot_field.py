"""Complex field visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_complex_field(
    U: NDArray[np.complexfloating],
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot amplitude and wrapped phase of a complex field."""

    field = np.asarray(U)
    amplitude = np.abs(field)
    phase = np.angle(field)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    if title:
        fig.suptitle(title)

    amp_im = axes[0].imshow(amplitude, cmap="magma")
    axes[0].set_title("Amplitude")
    axes[0].set_axis_off()
    fig.colorbar(amp_im, ax=axes[0], fraction=0.046, pad=0.04)

    phase_im = axes[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase")
    axes[1].set_axis_off()
    fig.colorbar(phase_im, ax=axes[1], fraction=0.046, pad=0.04)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig


def save_intensity_image(
    intensity: NDArray[np.floating],
    save_path: str | Path,
    cmap: str = "magma",
) -> None:
    """Save a single intensity map as an image."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, np.asarray(intensity), cmap=cmap)
