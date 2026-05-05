"""Sparse least-squares solver for residual static estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from app.services.residual_static_design_matrix import (
    ResidualStaticColumnLayout,
    ResidualStaticModelEvaluation,
    ResidualStaticObservationMatrixTriplets,
    ResidualStaticParameterParts,
    build_residual_static_observation_matrix_triplets,
    evaluate_residual_static_model,
    unpack_residual_static_parameters,
)
from app.services.residual_static_inputs import ResidualStaticSolverInputs


@dataclass(frozen=True)
class ResidualStaticLsmrOptions:
    atol: float = 1.0e-10
    btol: float = 1.0e-10
    conlim: float = 1.0e8
    maxiter: int | None = None


@dataclass(frozen=True)
class ResidualStaticLsmrDiagnostics:
    istop: int
    itn: int
    normr: float
    normar: float
    norma: float
    conda: float
    normx: float


@dataclass(frozen=True)
class ResidualStaticRawLsmrResult:
    parameter_vector: np.ndarray
    diagnostics: ResidualStaticLsmrDiagnostics


@dataclass(frozen=True)
class ResidualStaticLeastSquaresResult:
    parameter_vector: np.ndarray
    parameter_parts: ResidualStaticParameterParts
    model_evaluation: ResidualStaticModelEvaluation
    diagnostics: ResidualStaticLsmrDiagnostics

    layout: ResidualStaticColumnLayout
    used_mask_sorted: np.ndarray
    row_to_sorted_trace_index: np.ndarray

    n_observations: int
    n_model_parameters: int
    rank_deficient_possible: bool


def validate_lsmr_options(
    options: ResidualStaticLsmrOptions,
) -> ResidualStaticLsmrOptions:
    """Validate and normalize LSMR numeric options."""
    if not isinstance(options, ResidualStaticLsmrOptions):
        raise ValueError('options must be a ResidualStaticLsmrOptions instance')
    return ResidualStaticLsmrOptions(
        atol=_coerce_nonnegative_finite_float(options.atol, name='atol'),
        btol=_coerce_nonnegative_finite_float(options.btol, name='btol'),
        conlim=_coerce_positive_finite_float(options.conlim, name='conlim'),
        maxiter=_coerce_optional_positive_int(options.maxiter, name='maxiter'),
    )


def build_csr_matrix_from_residual_static_triplets(
    triplets: ResidualStaticObservationMatrixTriplets,
):
    """Build a SciPy CSR matrix from residual-static COO triplets."""
    n_rows = _coerce_positive_int(triplets.n_rows, name='triplets.n_rows')
    n_cols = _coerce_positive_int(triplets.n_cols, name='triplets.n_cols')
    layout_n_cols = _coerce_positive_int(
        triplets.layout.n_model_parameters,
        name='triplets.layout.n_model_parameters',
    )
    if n_cols != layout_n_cols:
        raise ValueError('triplets.n_cols must match layout.n_model_parameters')

    row_indices = _coerce_1d_integer_int64(
        triplets.row_indices,
        name='triplets.row_indices',
    )
    col_indices = _coerce_1d_integer_int64(
        triplets.col_indices,
        name='triplets.col_indices',
    )
    data = _coerce_1d_real_numeric_float64(triplets.data, name='triplets.data')
    if row_indices.shape != col_indices.shape or row_indices.shape != data.shape:
        raise ValueError('triplet row_indices, col_indices, and data must match')
    _validate_index_range(row_indices, n_unique=n_rows, name='triplets.row_indices')
    _validate_index_range(col_indices, n_unique=n_cols, name='triplets.col_indices')
    _validate_all_finite(data, name='triplets.data')

    rhs_s = _coerce_1d_real_numeric_float64(
        triplets.rhs_s,
        name='triplets.rhs_s',
        expected_shape=(n_rows,),
    )
    _validate_all_finite(rhs_s, name='triplets.rhs_s')
    _coerce_1d_integer_int64(
        triplets.row_to_sorted_trace_index,
        name='triplets.row_to_sorted_trace_index',
        expected_shape=(n_rows,),
    )
    _coerce_1d_bool_array(
        triplets.used_mask_sorted,
        name='triplets.used_mask_sorted',
    )

    return sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_rows, n_cols),
        dtype=np.float64,
    ).tocsr()


def run_sparse_lsmr(
    matrix,
    rhs_s: np.ndarray,
    *,
    options: ResidualStaticLsmrOptions | None = None,
) -> ResidualStaticRawLsmrResult:
    """Run SciPy LSMR against a validated sparse linear system."""
    matrix_csr = _coerce_sparse_matrix_float64_csr(matrix)
    n_rows, n_cols = matrix_csr.shape
    rhs = _coerce_1d_real_numeric_float64(
        rhs_s,
        name='rhs_s',
        expected_shape=(n_rows,),
    )
    _validate_all_finite(rhs, name='rhs_s')
    validated_options = validate_lsmr_options(options or ResidualStaticLsmrOptions())

    try:
        x, istop, itn, normr, normar, norma, conda, normx = sparse_linalg.lsmr(
            matrix_csr,
            rhs,
            damp=0.0,
            atol=validated_options.atol,
            btol=validated_options.btol,
            conlim=validated_options.conlim,
            maxiter=validated_options.maxiter,
            show=False,
        )
    except Exception as exc:
        raise RuntimeError('LSMR solve failed') from exc

    parameter_vector = np.ascontiguousarray(x, dtype=np.float64)
    if parameter_vector.shape != (n_cols,):
        msg = (
            'LSMR parameter_vector shape mismatch: '
            f'expected {(n_cols,)}, got {parameter_vector.shape}'
        )
        raise ValueError(msg)
    _validate_all_finite(parameter_vector, name='parameter_vector')
    return ResidualStaticRawLsmrResult(
        parameter_vector=parameter_vector,
        diagnostics=ResidualStaticLsmrDiagnostics(
            istop=int(istop),
            itn=int(itn),
            normr=float(normr),
            normar=float(normar),
            norma=float(norma),
            conda=float(conda),
            normx=float(normx),
        ),
    )


def solve_residual_static_least_squares(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray | None = None,
    options: ResidualStaticLsmrOptions | None = None,
) -> ResidualStaticLeastSquaresResult:
    """Solve preliminary ungauged residual static delays with sparse LSMR."""
    triplets = build_residual_static_observation_matrix_triplets(
        inputs,
        used_mask_sorted=used_mask_sorted,
    )
    matrix = build_csr_matrix_from_residual_static_triplets(triplets)
    raw_result = run_sparse_lsmr(matrix, triplets.rhs_s, options=options)
    parameter_parts = unpack_residual_static_parameters(
        triplets.layout,
        raw_result.parameter_vector,
    )
    model_evaluation = evaluate_residual_static_model(
        inputs,
        triplets.layout,
        raw_result.parameter_vector,
        residual_mask_sorted=triplets.used_mask_sorted,
    )
    _validate_domain_result(
        parameter_vector=raw_result.parameter_vector,
        parameter_parts=parameter_parts,
        model_evaluation=model_evaluation,
        layout=triplets.layout,
    )

    return ResidualStaticLeastSquaresResult(
        parameter_vector=raw_result.parameter_vector,
        parameter_parts=parameter_parts,
        model_evaluation=model_evaluation,
        diagnostics=raw_result.diagnostics,
        layout=triplets.layout,
        used_mask_sorted=np.ascontiguousarray(triplets.used_mask_sorted, dtype=bool),
        row_to_sorted_trace_index=np.ascontiguousarray(
            triplets.row_to_sorted_trace_index,
            dtype=np.int64,
        ),
        n_observations=int(matrix.shape[0]),
        n_model_parameters=int(matrix.shape[1]),
        rank_deficient_possible=True,
    )


def _validate_domain_result(
    *,
    parameter_vector: np.ndarray,
    parameter_parts: ResidualStaticParameterParts,
    model_evaluation: ResidualStaticModelEvaluation,
    layout: ResidualStaticColumnLayout,
) -> None:
    if parameter_vector.shape != (layout.n_model_parameters,):
        raise ValueError('parameter_vector shape must match layout.n_model_parameters')
    _validate_all_finite(parameter_vector, name='parameter_vector')
    if parameter_parts.source_delay_s.shape != (layout.n_sources,):
        raise ValueError('source_delay_s shape must match layout.n_sources')
    if parameter_parts.receiver_delay_s.shape != (layout.n_receivers,):
        raise ValueError('receiver_delay_s shape must match layout.n_receivers')
    _validate_all_finite(parameter_parts.source_delay_s, name='source_delay_s')
    _validate_all_finite(parameter_parts.receiver_delay_s, name='receiver_delay_s')
    residual_mask = _coerce_1d_bool_array(
        model_evaluation.residual_valid_mask_sorted,
        name='model_evaluation.residual_valid_mask_sorted',
    )
    residual = _coerce_1d_real_numeric_float64(
        model_evaluation.residual_s_sorted,
        name='model_evaluation.residual_s_sorted',
        expected_shape=residual_mask.shape,
    )
    if np.any(~np.isfinite(residual[residual_mask])):
        raise ValueError('residual_s_sorted must be finite for used traces')


def _coerce_sparse_matrix_float64_csr(matrix):
    if not sparse.issparse(matrix):
        raise ValueError('matrix must be a SciPy sparse matrix')
    if len(matrix.shape) != 2:
        raise ValueError('matrix must be 2D')
    n_rows = _coerce_positive_int(matrix.shape[0], name='matrix.shape[0]')
    n_cols = _coerce_positive_int(matrix.shape[1], name='matrix.shape[1]')
    dtype = np.dtype(matrix.dtype)
    if not _is_real_numeric_dtype(dtype):
        raise ValueError('matrix dtype must be real numeric')
    matrix_csr = matrix.tocsr().astype(np.float64, copy=False)
    if matrix_csr.shape != (n_rows, n_cols):
        raise ValueError('matrix shape changed during CSR conversion')
    _validate_all_finite(matrix_csr.data, name='matrix.data')
    return matrix_csr


def _coerce_1d_integer_int64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must contain integer values')
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr, dtype=np.int64)


def _coerce_1d_bool_array(
    values: object,
    *,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if not np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must have bool dtype')
    return np.ascontiguousarray(arr, dtype=bool)


def _coerce_1d_real_numeric_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must have a numeric dtype')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    out = int(value)
    if out <= 0:
        raise ValueError(f'{name} must be greater than 0')
    return out


def _coerce_optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_positive_int(value, name=name)


def _coerce_nonnegative_finite_float(value: object, *, name: str) -> float:
    out = _coerce_finite_float(value, name=name)
    if out < 0.0:
        raise ValueError(f'{name} must be greater than or equal to 0')
    return out


def _coerce_positive_finite_float(value: object, *, name: str) -> float:
    out = _coerce_finite_float(value, name=name)
    if out <= 0.0:
        raise ValueError(f'{name} must be greater than 0')
    return out


def _coerce_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


def _validate_index_range(
    indices: np.ndarray,
    *,
    n_unique: int,
    name: str,
) -> None:
    if indices.size == 0:
        return
    if np.any(indices < 0):
        raise ValueError(f'{name} must be greater than or equal to 0')
    if np.any(indices >= n_unique):
        raise ValueError(f'{name} contains values outside 0..{n_unique - 1}')


def _validate_all_finite(values: np.ndarray, *, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'ResidualStaticLeastSquaresResult',
    'ResidualStaticLsmrDiagnostics',
    'ResidualStaticLsmrOptions',
    'ResidualStaticRawLsmrResult',
    'build_csr_matrix_from_residual_static_triplets',
    'run_sparse_lsmr',
    'solve_residual_static_least_squares',
    'validate_lsmr_options',
]
