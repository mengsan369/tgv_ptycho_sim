"""Bounded-memory iterative Helmholtz solver controls for exp040."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_array, csc_matrix
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu


def sparse_storage_bytes(matrix: csc_matrix | csc_array) -> int:
    """Return exact NumPy-array storage used by a CSC matrix."""

    if not isinstance(matrix, (csc_matrix, csc_array)):
        raise TypeError("matrix must be CSC.")
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def superlu_storage_bytes(factor: Any) -> int:
    """Return sparse arrays and permutations retained by a SuperLU factor."""

    return int(
        sparse_storage_bytes(factor.L.tocsc())
        + sparse_storage_bytes(factor.U.tocsc())
        + np.asarray(factor.perm_r).nbytes
        + np.asarray(factor.perm_c).nbytes
    )


@dataclass
class BuiltPreconditioner:
    """Linear operator plus deterministic storage/setup controls."""

    operator: LinearOperator
    controls: dict[str, Any]
    retained: Any


def build_csl_ilu_preconditioner(
    shifted_matrix: csc_matrix,
    *,
    drop_tolerance: float,
    fill_factor: float,
    drop_rule: str,
    permc_spec: str,
) -> BuiltPreconditioner:
    """Build a fill-capped incomplete LU of a complex shifted operator."""

    started = time.perf_counter()
    factor = spilu(
        shifted_matrix,
        drop_tol=float(drop_tolerance),
        fill_factor=float(fill_factor),
        drop_rule=str(drop_rule),
        permc_spec=str(permc_spec),
    )
    setup_elapsed = float(time.perf_counter() - started)

    def apply(vector: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
        return np.asarray(factor.solve(np.asarray(vector)), dtype=np.complex128)

    operator = LinearOperator(
        shifted_matrix.shape, matvec=apply, dtype=np.complex128
    )
    factor_bytes = superlu_storage_bytes(factor)
    controls = {
        "kind": "global_incomplete_lu_of_shifted_operator",
        "drop_tolerance": float(drop_tolerance),
        "fill_factor_limit": float(fill_factor),
        "drop_rule": str(drop_rule),
        "permc_spec": str(permc_spec),
        "shifted_matrix_nnz": int(shifted_matrix.nnz),
        "factor_l_nnz": int(factor.L.nnz),
        "factor_u_nnz": int(factor.U.nnz),
        "factor_fill_ratio": float(
            (factor.L.nnz + factor.U.nnz) / shifted_matrix.nnz
        ),
        "factor_storage_bytes": factor_bytes,
        "retained_shifted_matrix_bytes": 0,
        "total_preconditioner_storage_bytes": factor_bytes,
        "setup_elapsed_s": setup_elapsed,
        "full_global_factorization": False,
        "all_finite": bool(
            np.all(np.isfinite(factor.L.data))
            and np.all(np.isfinite(factor.U.data))
        ),
    }
    return BuiltPreconditioner(operator=operator, controls=controls, retained=factor)


@dataclass(frozen=True)
class _RASBlock:
    core_indices: NDArray[np.int64]
    extended_indices: NDArray[np.int64]
    core_positions: NDArray[np.int64]
    factor: Any


def _rectangular_indices(
    radial_count: int,
    z_start: int,
    z_stop: int,
    r_start: int,
    r_stop: int,
) -> NDArray[np.int64]:
    z_indices = np.arange(z_start, z_stop, dtype=np.int64)
    r_indices = np.arange(r_start, r_stop, dtype=np.int64)
    return (
        z_indices[:, None] * int(radial_count) + r_indices[None, :]
    ).reshape(-1)


def build_two_level_ras_csl_preconditioner(
    shifted_matrix: csc_matrix,
    *,
    active_shape: tuple[int, int],
    core_block_shape_nodes: tuple[int, int],
    overlap_nodes: int,
) -> BuiltPreconditioner:
    """Build constant-coarse multiplicative correction plus shifted RAS."""

    nz, nr = (int(active_shape[0]), int(active_shape[1]))
    block_z, block_r = (
        int(core_block_shape_nodes[0]),
        int(core_block_shape_nodes[1]),
    )
    overlap = int(overlap_nodes)
    if nz * nr != shifted_matrix.shape[0]:
        raise ValueError("active_shape does not match shifted_matrix.")
    if min(block_z, block_r) < 1 or overlap < 0:
        raise ValueError("block shape and overlap must be non-negative integers.")
    started = time.perf_counter()
    blocks: list[_RASBlock] = []
    coarse_rows: list[NDArray[np.int64]] = []
    coarse_columns: list[NDArray[np.int64]] = []
    coarse_values: list[NDArray[np.float64]] = []
    local_factor_bytes = 0
    local_l_nnz = 0
    local_u_nnz = 0
    block_id = 0
    for z_start in range(0, nz, block_z):
        z_stop = min(z_start + block_z, nz)
        extended_z_start = max(0, z_start - overlap)
        extended_z_stop = min(nz, z_stop + overlap)
        for r_start in range(0, nr, block_r):
            r_stop = min(r_start + block_r, nr)
            extended_r_start = max(0, r_start - overlap)
            extended_r_stop = min(nr, r_stop + overlap)
            core_indices = _rectangular_indices(
                nr, z_start, z_stop, r_start, r_stop
            )
            extended_indices = _rectangular_indices(
                nr,
                extended_z_start,
                extended_z_stop,
                extended_r_start,
                extended_r_stop,
            )
            core_positions = np.searchsorted(extended_indices, core_indices)
            if not np.array_equal(
                extended_indices[core_positions], core_indices
            ):
                raise RuntimeError("RAS core is not contained in its overlap.")
            local_matrix = shifted_matrix[extended_indices, :][
                :, extended_indices
            ].tocsc()
            factor = splu(local_matrix, permc_spec="COLAMD")
            local_factor_bytes += superlu_storage_bytes(factor)
            local_l_nnz += int(factor.L.nnz)
            local_u_nnz += int(factor.U.nnz)
            blocks.append(
                _RASBlock(
                    core_indices=core_indices,
                    extended_indices=extended_indices,
                    core_positions=core_positions,
                    factor=factor,
                )
            )
            coarse_rows.append(core_indices)
            coarse_columns.append(
                np.full(core_indices.size, block_id, dtype=np.int64)
            )
            coarse_values.append(
                np.full(
                    core_indices.size,
                    1.0 / np.sqrt(core_indices.size),
                    dtype=np.float64,
                )
            )
            block_id += 1
    prolongation = coo_matrix(
        (
            np.concatenate(coarse_values),
            (np.concatenate(coarse_rows), np.concatenate(coarse_columns)),
        ),
        shape=(shifted_matrix.shape[0], block_id),
        dtype=np.complex128,
    ).tocsc()
    coarse_matrix = (
        prolongation.conjugate().transpose() @ shifted_matrix @ prolongation
    ).tocsc()
    coarse_factor = splu(coarse_matrix, permc_spec="COLAMD")
    coarse_factor_bytes = superlu_storage_bytes(coarse_factor)
    shifted_bytes = sparse_storage_bytes(shifted_matrix)
    prolongation_bytes = sparse_storage_bytes(prolongation)

    def apply(vector: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
        right = np.asarray(vector, dtype=np.complex128)
        coarse_right = prolongation.conjugate().transpose() @ right
        coarse_solution = coarse_factor.solve(coarse_right)
        coarse_correction = np.asarray(
            prolongation @ coarse_solution, dtype=np.complex128
        )
        residual = right - shifted_matrix @ coarse_correction
        local_correction = np.zeros_like(right)
        for block in blocks:
            local = block.factor.solve(residual[block.extended_indices])
            local_correction[block.core_indices] = local[block.core_positions]
        return coarse_correction + local_correction

    operator = LinearOperator(
        shifted_matrix.shape, matvec=apply, dtype=np.complex128
    )
    total_storage = (
        shifted_bytes
        + prolongation_bytes
        + local_factor_bytes
        + coarse_factor_bytes
    )
    controls = {
        "kind": "multiplicative_constant_coarse_plus_ras",
        "active_shape": [nz, nr],
        "core_block_shape_nodes": [block_z, block_r],
        "overlap_nodes": overlap,
        "block_count": len(blocks),
        "coarse_dimension": block_id,
        "local_factor_l_nnz": local_l_nnz,
        "local_factor_u_nnz": local_u_nnz,
        "coarse_factor_l_nnz": int(coarse_factor.L.nnz),
        "coarse_factor_u_nnz": int(coarse_factor.U.nnz),
        "local_factor_storage_bytes": local_factor_bytes,
        "coarse_factor_storage_bytes": coarse_factor_bytes,
        "prolongation_storage_bytes": prolongation_bytes,
        "retained_shifted_matrix_bytes": shifted_bytes,
        "total_preconditioner_storage_bytes": total_storage,
        "setup_elapsed_s": float(time.perf_counter() - started),
        "full_global_factorization": False,
        "all_finite": bool(
            np.all(np.isfinite(coarse_factor.L.data))
            and np.all(np.isfinite(coarse_factor.U.data))
            and all(
                np.all(np.isfinite(block.factor.L.data))
                and np.all(np.isfinite(block.factor.U.data))
                for block in blocks
            )
        ),
    }
    retained = {
        "blocks": blocks,
        "prolongation": prolongation,
        "coarse_factor": coarse_factor,
        "shifted_matrix": shifted_matrix,
    }
    return BuiltPreconditioner(
        operator=operator, controls=controls, retained=retained
    )


def solve_restarted_gmres(
    matrix: csc_matrix,
    rhs: NDArray[np.complexfloating],
    preconditioner: LinearOperator,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    restart: int,
    maximum_inner_iterations: int,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Run restarted GMRES and report both callback and true residuals."""

    right = np.asarray(rhs, dtype=np.complex128)
    restart_value = int(restart)
    maximum = int(maximum_inner_iterations)
    if restart_value < 1 or maximum < restart_value:
        raise ValueError("invalid GMRES restart or maximum iteration count.")
    restart_cycles = int(np.ceil(maximum / restart_value))
    residual_history: list[float] = []

    def callback(value: float) -> None:
        residual_history.append(float(value))

    started = time.perf_counter()
    solution, info = gmres(
        matrix,
        right,
        x0=np.zeros_like(right),
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
        restart=restart_value,
        maxiter=restart_cycles,
        M=preconditioner,
        callback=callback,
        callback_type="pr_norm",
    )
    elapsed = float(time.perf_counter() - started)
    solution = np.asarray(solution, dtype=np.complex128)
    true_residual = matrix @ solution - right
    relative_residual = float(
        np.linalg.norm(true_residual)
        / max(np.linalg.norm(right), np.finfo(float).eps)
    )
    controls = {
        "solver": "scipy_gmres",
        "info": int(info),
        "converged": bool(info == 0),
        "restart": restart_value,
        "maximum_inner_iterations": maximum,
        "inner_iteration_count": len(residual_history),
        "preconditioned_residual_history": np.asarray(
            residual_history, dtype=np.float64
        ),
        "final_preconditioned_residual": (
            float(residual_history[-1]) if residual_history else None
        ),
        "true_relative_residual": relative_residual,
        "solve_elapsed_s": elapsed,
        "krylov_basis_storage_bytes": int(
            matrix.shape[0] * np.dtype(np.complex128).itemsize * (restart_value + 2)
        ),
        "all_finite": bool(
            np.all(np.isfinite(solution))
            and np.all(np.isfinite(true_residual))
            and np.isfinite(relative_residual)
            and np.all(np.isfinite(residual_history))
        ),
    }
    return solution, controls


__all__ = [
    "BuiltPreconditioner",
    "build_csl_ilu_preconditioner",
    "build_two_level_ras_csl_preconditioner",
    "solve_restarted_gmres",
    "sparse_storage_bytes",
    "superlu_storage_bytes",
]
