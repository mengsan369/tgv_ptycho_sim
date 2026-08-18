"""Gauge-aware sensitivity and local-observability helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _squared_norm(values: NDArray[np.generic]) -> float:
    return float(np.sum(np.abs(values) ** 2, dtype=np.float64))


def relative_l2(
    values: NDArray[np.generic], reference: NDArray[np.generic]
) -> float:
    """Return a finite relative L2 difference for same-shaped arrays."""

    array = np.asarray(values)
    ref = np.asarray(reference)
    if array.shape != ref.shape:
        msg = "values and reference must have the same shape."
        raise ValueError(msg)
    numerator = _squared_norm(array - ref)
    denominator = _squared_norm(ref)
    return float(np.sqrt(numerator / (denominator + np.finfo(float).eps)))


def align_complex_scale(
    field: NDArray[np.complexfloating],
    reference: NDArray[np.complexfloating],
) -> tuple[NDArray[np.complex128], complex]:
    """Fit and remove one global complex scale relative to ``reference``."""

    candidate = np.asarray(field, dtype=np.complex128)
    ref = np.asarray(reference, dtype=np.complex128)
    if candidate.shape != ref.shape:
        msg = "field and reference must have the same shape."
        raise ValueError(msg)
    denominator = np.sum(np.conj(candidate) * candidate)
    gain = (
        complex(np.sum(np.conj(candidate) * ref) / denominator)
        if abs(denominator) > np.finfo(float).eps
        else 1.0 + 0.0j
    )
    return (gain * candidate).astype(np.complex128), gain


def compare_probe_sensitivity(
    P1: NDArray[np.complexfloating],
    P2: NDArray[np.complexfloating],
) -> dict[str, float]:
    """Compare probes before and after global-complex-scale alignment."""

    probe_1 = np.asarray(P1, dtype=np.complex128)
    probe_2 = np.asarray(P2, dtype=np.complex128)
    if probe_1.shape != probe_2.shape:
        msg = "P1 and P2 must have the same shape."
        raise ValueError(msg)
    aligned_2, gain = align_complex_scale(probe_2, probe_1)
    aligned_difference = aligned_2 - probe_1
    wrapped_phase_difference = np.angle(aligned_2 * np.conj(probe_1))
    return {
        "relative_l2": relative_l2(probe_2, probe_1),
        "amplitude_relative_l2": relative_l2(
            np.abs(probe_2), np.abs(probe_1)
        ),
        "gauge_aligned_complex_relative_l2": relative_l2(aligned_2, probe_1),
        "gauge_aligned_amplitude_relative_l2": relative_l2(
            np.abs(aligned_2), np.abs(probe_1)
        ),
        "gauge_aligned_wrapped_phase_rmse_rad": float(
            np.sqrt(np.mean(wrapped_phase_difference**2))
        ),
        "gauge_aligned_difference_max_abs": float(
            np.max(np.abs(aligned_difference))
        ),
        "gauge_aligned_difference_rms": float(
            np.sqrt(np.mean(np.abs(aligned_difference) ** 2))
        ),
        "alignment_gain_magnitude": float(abs(gain)),
        "alignment_gain_phase_rad": float(np.angle(gain)),
    }


def central_finite_difference(
    minus: NDArray[np.generic],
    plus: NDArray[np.generic],
    step: float,
) -> NDArray[np.generic]:
    """Return ``(plus - minus) / (2 * step)`` with input validation."""

    lower = np.asarray(minus)
    upper = np.asarray(plus)
    if lower.shape != upper.shape:
        msg = "minus and plus arrays must have the same shape."
        raise ValueError(msg)
    if not np.isfinite(step) or step <= 0.0:
        msg = "step must be finite and positive."
        raise ValueError(msg)
    derivative = (upper - lower) / (2.0 * step)
    if not np.all(np.isfinite(derivative)):
        msg = "finite difference produced NaN or Inf."
        raise ValueError(msg)
    return derivative


def gauge_project_complex_derivative(
    derivative: NDArray[np.complexfloating],
    reference: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Remove the global complex-scale tangent from a field derivative."""

    column = np.asarray(derivative, dtype=np.complex128)
    ref = np.asarray(reference, dtype=np.complex128)
    if column.shape != ref.shape:
        msg = "derivative and reference must have the same shape."
        raise ValueError(msg)
    denominator = np.sum(np.conj(ref) * ref)
    if abs(denominator) <= np.finfo(float).eps:
        msg = "reference must have non-zero energy."
        raise ValueError(msg)
    coefficient = np.sum(np.conj(ref) * column) / denominator
    return (column - coefficient * ref).astype(np.complex128)


def normalized_complex_sensitivity(
    minus: NDArray[np.complexfloating],
    plus: NDArray[np.complexfloating],
    reference: NDArray[np.complexfloating],
    step: float,
    parameter_scale: float,
) -> tuple[NDArray[np.complex128], float]:
    """Return gauge-projected derivative and dimensionless sensitivity norm."""

    if not np.isfinite(parameter_scale) or parameter_scale <= 0.0:
        msg = "parameter_scale must be finite and positive."
        raise ValueError(msg)
    derivative = central_finite_difference(minus, plus, step)
    projected = gauge_project_complex_derivative(derivative, reference)
    ref_energy = _squared_norm(np.asarray(reference, dtype=np.complex128))
    normalized = parameter_scale * np.sqrt(
        _squared_norm(projected) / (ref_energy + np.finfo(float).eps)
    )
    return projected, float(normalized)


def normalized_real_sensitivity(
    minus: NDArray[np.floating],
    plus: NDArray[np.floating],
    reference: NDArray[np.floating],
    step: float,
    parameter_scale: float,
) -> tuple[NDArray[np.float64], float]:
    """Return real-valued derivative and dimensionless sensitivity norm."""

    if not np.isfinite(parameter_scale) or parameter_scale <= 0.0:
        msg = "parameter_scale must be finite and positive."
        raise ValueError(msg)
    derivative = np.asarray(
        central_finite_difference(minus, plus, step), dtype=np.float64
    )
    ref_energy = _squared_norm(np.asarray(reference, dtype=np.float64))
    normalized = parameter_scale * np.sqrt(
        _squared_norm(derivative) / (ref_energy + np.finfo(float).eps)
    )
    return derivative, float(normalized)


def successive_relative_changes(
    values: NDArray[np.floating] | list[float] | tuple[float, ...],
) -> NDArray[np.float64]:
    """Return coarse-to-fine relative changes for a metric sequence."""

    sequence = np.asarray(values, dtype=np.float64)
    if sequence.ndim != 1 or sequence.size < 2:
        msg = "values must be a 1D sequence containing at least two entries."
        raise ValueError(msg)
    if not np.all(np.isfinite(sequence)):
        msg = "values must contain only finite entries."
        raise ValueError(msg)
    denominator = np.maximum(np.abs(sequence[1:]), np.finfo(float).eps)
    return (np.abs(sequence[1:] - sequence[:-1]) / denominator).astype(
        np.float64
    )


def _symmetric_eigenvalues(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return eigenvalues of a small real-symmetric matrix without LAPACK."""

    values = np.asarray(matrix, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        msg = "matrix must be square."
        raise ValueError(msg)
    size = values.shape[0]
    if size == 0:
        return np.asarray([], dtype=np.float64)
    tolerance = np.finfo(float).eps * max(
        float(np.max(np.abs(values))), 1.0
    )
    for _ in range(max(20, 20 * size * size)):
        p_index, q_index = 0, 0
        largest = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                magnitude = abs(float(values[row, column]))
                if magnitude > largest:
                    largest = magnitude
                    p_index, q_index = row, column
        if largest <= tolerance:
            break

        app = float(values[p_index, p_index])
        aqq = float(values[q_index, q_index])
        apq = float(values[p_index, q_index])
        tau = (aqq - app) / (2.0 * apq)
        sign = 1.0 if tau >= 0.0 else -1.0
        tangent = sign / (abs(tau) + np.sqrt(1.0 + tau * tau))
        cosine = 1.0 / np.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine

        for index in range(size):
            if index in {p_index, q_index}:
                continue
            aip = float(values[index, p_index])
            aiq = float(values[index, q_index])
            new_ip = cosine * aip - sine * aiq
            new_iq = sine * aip + cosine * aiq
            values[index, p_index] = values[p_index, index] = new_ip
            values[index, q_index] = values[q_index, index] = new_iq
        values[p_index, p_index] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq
        )
        values[q_index, q_index] = (
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        values[p_index, q_index] = values[q_index, p_index] = 0.0
    return np.sort(np.diag(values).astype(np.float64))


def analyze_local_observability(
    reference: NDArray[np.complexfloating],
    derivative_columns: Mapping[str, NDArray[np.complexfloating]],
    parameter_scales: Mapping[str, float],
) -> dict[str, Any]:
    """Analyze a gauge-projected, parameter-scaled local complex Jacobian.

    Complex columns are represented by concatenated real and imaginary parts.
    Each physical derivative is multiplied by its nominal parameter scale,
    projected away from global complex scale, and normalized before column
    correlations and singular values are calculated.
    """

    if not derivative_columns:
        msg = "derivative_columns must not be empty."
        raise ValueError(msg)
    labels = list(derivative_columns)
    vectors: list[NDArray[np.float64]] = []
    column_norms: list[float] = []
    for label in labels:
        if label not in parameter_scales:
            msg = f"Missing parameter scale for {label}."
            raise ValueError(msg)
        scale = float(parameter_scales[label])
        if not np.isfinite(scale) or scale <= 0.0:
            msg = f"Parameter scale for {label} must be finite and positive."
            raise ValueError(msg)
        projected = gauge_project_complex_derivative(
            derivative_columns[label], reference
        )
        scaled = projected * scale
        vector = np.concatenate([scaled.real.ravel(), scaled.imag.ravel()])
        norm = float(np.sqrt(np.sum(vector**2, dtype=np.float64)))
        vectors.append(vector)
        column_norms.append(norm)

    matrix = np.column_stack(vectors)
    norms = np.asarray(column_norms, dtype=np.float64)
    threshold = np.finfo(float).eps * max(float(np.max(norms)), 1.0)
    num_columns = matrix.shape[1]
    gram = np.zeros((num_columns, num_columns), dtype=np.float64)
    for row in range(num_columns):
        for column in range(row, num_columns):
            value = float(
                np.sum(matrix[:, row] * matrix[:, column], dtype=np.float64)
            )
            gram[row, column] = gram[column, row] = value
    correlation = np.zeros_like(gram)
    for row in range(num_columns):
        for column in range(num_columns):
            denominator = norms[row] * norms[column]
            if denominator > threshold:
                correlation[row, column] = gram[row, column] / denominator
    correlation = np.clip(correlation, -1.0, 1.0)
    eigenvalues = _symmetric_eigenvalues(gram)
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    ratio = smallest / largest if largest > 0.0 else 0.0
    condition = largest / smallest if smallest > threshold else float("inf")
    return {
        "parameter_labels": labels,
        "column_norms": norms,
        "normalized_column_correlation": correlation.astype(np.float64),
        "singular_values": singular_values.astype(np.float64),
        "smallest_to_largest_singular_value_ratio": float(ratio),
        "condition_number": float(condition),
        "numerical_rank": int(np.count_nonzero(singular_values > threshold)),
    }
