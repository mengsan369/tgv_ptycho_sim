"""Preprocessing interfaces for experimental ptychography data."""

from tgv_ptycho.preprocess.bad_pixels import correct_bad_pixels, detect_bad_pixels
from tgv_ptycho.preprocess.dark_flat import apply_dark_flat_correction
from tgv_ptycho.preprocess.normalize import normalize_by_exposure, normalize_stack
from tgv_ptycho.preprocess.roi import crop_roi, crop_stack_roi

__all__ = [
    "apply_dark_flat_correction",
    "normalize_by_exposure",
    "normalize_stack",
    "crop_roi",
    "crop_stack_roi",
    "detect_bad_pixels",
    "correct_bad_pixels",
]
