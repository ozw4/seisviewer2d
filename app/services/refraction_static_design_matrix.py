"""Sparse design matrix builder for GLI refraction statics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import sparse

from app.api.schemas import RefractionStaticModelRequest
from app.services.refraction_static_types import (
    RefractionStaticDesignMatrix,
    RefractionStaticInputModel,
)

BedrockVelocityMode = Literal['solve_global', 'fixed_global']


class RefractionStaticDesignMatrixError(ValueError):
    """Raised when refraction static design matrix inputs are inconsistent."""


def build_refraction_static_design_matrix(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
) -> RefractionStaticDesignMatrix:
    """Build the physical GLI sparse system from a refraction input model."""
    method = getattr(model, 'method', None)
    if method != 'gli_variable_thickness':
        raise RefractionStaticDesignMatrixError(
            'model.method must be gli_variable_thickness'
        )
    mode = _validate_bedrock_velocity_mode(
        getattr(model, 'bedrock_velocity_mode', None)
    )
    weathering_velocity = _coerce_positive_finite_float(
        getattr(model, 'weathering_velocity_m_s', None),
        name='model.weathering_velocity_m_s',
    )
    fixed_velocity = getattr(model, 'bedrock_velocity_m_s', None)
    if mode == 'fixed_global':
        fixed_velocity = _coerce_positive_finite_float(
            fixed_velocity,
            name='model.bedrock_velocity_m_s',
        )
        if fixed_velocity <= weathering_velocity:
            raise RefractionStaticDesignMatrixError(
                'model.bedrock_velocity_m_s must be greater than '
                'model.weathering_velocity_m_s'
            )
    elif fixed_velocity is not None:
        raise RefractionStaticDesignMatrixError(
            'model.bedrock_velocity_m_s is only allowed when '
            'model.bedrock_velocity_mode is fixed_global'
        )

    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=input_model.pick_time_s_sorted,
        valid_observation_mask_sorted=input_model.valid_observation_mask_sorted,
        source_node_id_sorted=input_model.source_node_id_sorted,
        receiver_node_id_sorted=input_model.receiver_node_id_sorted,
        distance_m_sorted=input_model.distance_m_sorted,
        node_id=input_model.endpoint_table.node_id,
        bedrock_velocity_mode=mode,
        fixed_bedrock_velocity_m_s=fixed_velocity,
        n_traces=input_model.n_traces,
    )


def build_refraction_static_design_matrix_from_arrays(
    *,
    pick_time_s_sorted: np.ndarray,
    valid_observation_mask_sorted: np.ndarray,
    source_node_id_sorted: np.ndarray,
    receiver_node_id_sorted: np.ndarray,
    distance_m_sorted: np.ndarray,
    node_id: np.ndarray,
    bedrock_velocity_mode: BedrockVelocityMode,
    fixed_bedrock_velocity_m_s: float | None = None,
    n_traces: int | None = None,
) -> RefractionStaticDesignMatrix:
    """Build a refraction static design matrix from sorted observation arrays."""
    mode = _validate_bedrock_velocity_mode(bedrock_velocity_mode)
    fixed_velocity: float | None = None
    fixed_slowness: float | None = None
    if mode == 'fixed_global':
        fixed_velocity = _coerce_positive_finite_float(
            fixed_bedrock_velocity_m_s,
            name='fixed_bedrock_velocity_m_s',
        )
        fixed_slowness = float(1.0 / fixed_velocity)
    elif fixed_bedrock_velocity_m_s is not None:
        raise RefractionStaticDesignMatrixError(
            'fixed_bedrock_velocity_m_s is only allowed for fixed_global mode'
        )

    pick_time = _coerce_1d_real_numeric_float64(
        pick_time_s_sorted,
        name='pick_time_s_sorted',
    )
    trace_count = _validate_n_traces(n_traces, default=int(pick_time.shape[0]))
    expected_shape = (trace_count,)
    if pick_time.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            'pick_time_s_sorted shape mismatch: '
            f'expected {expected_shape}, got {pick_time.shape}'
        )
    valid_mask = _coerce_1d_bool_array(
        valid_observation_mask_sorted,
        name='valid_observation_mask_sorted',
        expected_shape=expected_shape,
    )
    source_node_id = _coerce_1d_integer_int64(
        source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_shape,
    )
    receiver_node_id = _coerce_1d_integer_int64(
        receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_shape,
    )
    distance_m = _coerce_1d_real_numeric_float64(
        distance_m_sorted,
        name='distance_m_sorted',
        expected_shape=expected_shape,
    )
    total_node_id = _coerce_unique_node_id(node_id)

    row_trace_index = np.ascontiguousarray(np.flatnonzero(valid_mask), dtype=np.int64)
    n_observations = int(row_trace_index.shape[0])
    if n_observations <= 0:
        raise RefractionStaticDesignMatrixError(
            'at least one valid refraction observation is required'
        )

    observed_pick_time = np.ascontiguousarray(
        pick_time[row_trace_index],
        dtype=np.float64,
    )
    row_source_node_id = np.ascontiguousarray(
        source_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_receiver_node_id = np.ascontiguousarray(
        receiver_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_distance_m = np.ascontiguousarray(
        distance_m[row_trace_index],
        dtype=np.float64,
    )

    _validate_selected_values(
        observed_pick_time=observed_pick_time,
        row_distance_m=row_distance_m,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        node_id=total_node_id,
    )
    active_node_id, inactive_node_id = _split_active_nodes(
        node_id=total_node_id,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
    )
    node_id_to_col = {int(value): idx for idx, value in enumerate(active_node_id.tolist())}
    if len(node_id_to_col) != int(active_node_id.shape[0]):
        raise RefractionStaticDesignMatrixError('active node IDs must be unique')
    source_node_col = _map_node_ids_to_cols(
        row_source_node_id,
        node_id_to_col=node_id_to_col,
        name='row_source_node_id',
    )
    receiver_node_col = _map_node_ids_to_cols(
        row_receiver_node_id,
        node_id_to_col=node_id_to_col,
        name='row_receiver_node_id',
    )

    bedrock_slowness_col = None
    if mode == 'solve_global':
        bedrock_slowness_col = int(active_node_id.shape[0])
    n_parameters = int(active_node_id.shape[0]) + (
        1 if bedrock_slowness_col is not None else 0
    )

    matrix = _build_sparse_matrix(
        source_node_col=source_node_col,
        receiver_node_col=receiver_node_col,
        row_distance_m=row_distance_m,
        bedrock_slowness_col=bedrock_slowness_col,
        n_observations=n_observations,
        n_parameters=n_parameters,
    )
    if mode == 'fixed_global':
        if fixed_slowness is None:
            raise RefractionStaticDesignMatrixError(
                'fixed_global mode requires fixed bedrock slowness'
            )
        rhs_s = np.ascontiguousarray(
            observed_pick_time - row_distance_m * fixed_slowness,
            dtype=np.float64,
        )
    else:
        rhs_s = np.ascontiguousarray(observed_pick_time, dtype=np.float64)

    _validate_matrix_package(
        matrix=matrix,
        rhs_s=rhs_s,
        n_observations=n_observations,
        n_parameters=n_parameters,
    )
    qc = _build_qc(
        method='gli_variable_thickness',
        mode=mode,
        n_traces=trace_count,
        n_observations=n_observations,
        n_total_nodes=int(total_node_id.shape[0]),
        n_active_nodes=int(active_node_id.shape[0]),
        n_parameters=n_parameters,
        matrix=matrix,
        row_distance_m=row_distance_m,
        observed_pick_time=observed_pick_time,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        active_node_id=active_node_id,
        node_id_to_col=node_id_to_col,
        fixed_bedrock_velocity_m_s=fixed_velocity,
        fixed_bedrock_slowness_s_per_m=fixed_slowness,
        slowness_column_present=bedrock_slowness_col is not None,
    )

    return RefractionStaticDesignMatrix(
        matrix=matrix,
        rhs_s=rhs_s,
        observed_pick_time_s=observed_pick_time,
        row_trace_index_sorted=row_trace_index,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        row_distance_m=row_distance_m,
        active_node_id=active_node_id,
        inactive_node_id=inactive_node_id,
        node_id_to_col=node_id_to_col,
        source_node_col=source_node_col,
        receiver_node_col=receiver_node_col,
        bedrock_slowness_col=bedrock_slowness_col,
        bedrock_velocity_mode=mode,
        fixed_bedrock_velocity_m_s=fixed_velocity,
        fixed_bedrock_slowness_s_per_m=fixed_slowness,
        n_total_nodes=int(total_node_id.shape[0]),
        n_active_nodes=int(active_node_id.shape[0]),
        n_observations=n_observations,
        n_parameters=n_parameters,
        qc=qc,
    )


def _build_sparse_matrix(
    *,
    source_node_col: np.ndarray,
    receiver_node_col: np.ndarray,
    row_distance_m: np.ndarray,
    bedrock_slowness_col: int | None,
    n_observations: int,
    n_parameters: int,
) -> sparse.csr_matrix:
    entries_per_row = 3 if bedrock_slowness_col is not None else 2
    row_indices = np.repeat(np.arange(n_observations, dtype=np.int64), entries_per_row)
    col_indices = np.empty(n_observations * entries_per_row, dtype=np.int64)
    data = np.empty(n_observations * entries_per_row, dtype=np.float64)

    col_indices[0::entries_per_row] = source_node_col
    col_indices[1::entries_per_row] = receiver_node_col
    data[0::entries_per_row] = 1.0
    data[1::entries_per_row] = 1.0
    if bedrock_slowness_col is not None:
        col_indices[2::entries_per_row] = bedrock_slowness_col
        data[2::entries_per_row] = row_distance_m

    matrix = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_observations, n_parameters),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _build_qc(
    *,
    method: str,
    mode: BedrockVelocityMode,
    n_traces: int,
    n_observations: int,
    n_total_nodes: int,
    n_active_nodes: int,
    n_parameters: int,
    matrix: sparse.csr_matrix,
    row_distance_m: np.ndarray,
    observed_pick_time: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    active_node_id: np.ndarray,
    node_id_to_col: dict[int, int],
    fixed_bedrock_velocity_m_s: float | None,
    fixed_bedrock_slowness_s_per_m: float | None,
    slowness_column_present: bool,
) -> dict[str, Any]:
    nnz_per_row = np.diff(matrix.indptr)
    matrix_size = int(matrix.shape[0]) * int(matrix.shape[1])
    connectivity = _connectivity_qc(
        active_node_id=active_node_id,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        node_id_to_col=node_id_to_col,
    )
    qc: dict[str, Any] = {
        'method': method,
        'bedrock_velocity_mode': mode,
        'n_traces': int(n_traces),
        'n_observations': int(n_observations),
        'n_total_nodes': int(n_total_nodes),
        'n_active_nodes': int(n_active_nodes),
        'n_inactive_nodes': int(n_total_nodes - n_active_nodes),
        'n_parameters': int(n_parameters),
        'matrix_shape': [int(matrix.shape[0]), int(matrix.shape[1])],
        'matrix_nnz': int(matrix.nnz),
        'matrix_density': float(matrix.nnz / matrix_size) if matrix_size else 0.0,
        'nnz_per_row_min': int(np.min(nnz_per_row)),
        'nnz_per_row_max': int(np.max(nnz_per_row)),
        'nnz_per_row_median': float(np.median(nnz_per_row)),
        'distance_m_min': float(np.min(row_distance_m)),
        'distance_m_max': float(np.max(row_distance_m)),
        'distance_m_median': float(np.median(row_distance_m)),
        'pick_time_s_min': float(np.min(observed_pick_time)),
        'pick_time_s_max': float(np.max(observed_pick_time)),
        'pick_time_s_median': float(np.median(observed_pick_time)),
        'source_receiver_same_node_count': int(
            np.count_nonzero(row_source_node_id == row_receiver_node_id)
        ),
        'inactive_node_count': int(n_total_nodes - n_active_nodes),
        'slowness_column_present': bool(slowness_column_present),
        **connectivity,
    }
    if mode == 'fixed_global':
        qc.update(
            {
                'fixed_bedrock_velocity_m_s': float(fixed_bedrock_velocity_m_s),
                'fixed_bedrock_slowness_s_per_m': float(
                    fixed_bedrock_slowness_s_per_m
                ),
            }
        )
    return qc


def _connectivity_qc(
    *,
    active_node_id: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    node_id_to_col: dict[int, int],
) -> dict[str, int]:
    n_active = int(active_node_id.shape[0])
    parent = np.arange(n_active, dtype=np.int64)
    source_seen = np.zeros(n_active, dtype=bool)
    receiver_seen = np.zeros(n_active, dtype=bool)

    def find(col: int) -> int:
        while int(parent[col]) != col:
            parent[col] = parent[int(parent[col])]
            col = int(parent[col])
        return col

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source_id, receiver_id in zip(
        row_source_node_id.tolist(),
        row_receiver_node_id.tolist(),
        strict=True,
    ):
        source_col = node_id_to_col[int(source_id)]
        receiver_col = node_id_to_col[int(receiver_id)]
        source_seen[source_col] = True
        receiver_seen[receiver_col] = True
        if source_col != receiver_col:
            union(source_col, receiver_col)

    component_roots = {find(col) for col in range(n_active)}
    return {
        'n_source_only_nodes': int(np.count_nonzero(source_seen & ~receiver_seen)),
        'n_receiver_only_nodes': int(np.count_nonzero(receiver_seen & ~source_seen)),
        'n_source_and_receiver_nodes': int(
            np.count_nonzero(source_seen & receiver_seen)
        ),
        'n_connected_components': int(len(component_roots)),
    }


def _validate_selected_values(
    *,
    observed_pick_time: np.ndarray,
    row_distance_m: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    node_id: np.ndarray,
) -> None:
    if np.any(~np.isfinite(observed_pick_time)):
        raise RefractionStaticDesignMatrixError(
            'pick_time_s_sorted must be finite for selected observations'
        )
    if np.any(observed_pick_time < 0.0):
        raise RefractionStaticDesignMatrixError(
            'pick_time_s_sorted must be non-negative for selected observations'
        )
    if np.any(~np.isfinite(row_distance_m)):
        raise RefractionStaticDesignMatrixError(
            'distance_m_sorted must be finite for selected observations'
        )
    if np.any(row_distance_m <= 0.0):
        raise RefractionStaticDesignMatrixError(
            'distance_m_sorted must be greater than 0 for selected observations'
        )

    source_missing = ~np.isin(row_source_node_id, node_id)
    if np.any(source_missing):
        missing = int(row_source_node_id[np.flatnonzero(source_missing)[0]])
        raise RefractionStaticDesignMatrixError(
            f'source_node_id_sorted contains unknown node ID {missing}'
        )
    receiver_missing = ~np.isin(row_receiver_node_id, node_id)
    if np.any(receiver_missing):
        missing = int(row_receiver_node_id[np.flatnonzero(receiver_missing)[0]])
        raise RefractionStaticDesignMatrixError(
            f'receiver_node_id_sorted contains unknown node ID {missing}'
        )


def _split_active_nodes(
    *,
    node_id: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used_node_id = np.unique(np.concatenate((row_source_node_id, row_receiver_node_id)))
    active_mask = np.isin(node_id, used_node_id)
    active_node_id = np.ascontiguousarray(node_id[active_mask], dtype=np.int64)
    inactive_node_id = np.ascontiguousarray(node_id[~active_mask], dtype=np.int64)
    if active_node_id.shape[0] != used_node_id.shape[0]:
        raise RefractionStaticDesignMatrixError('active node IDs must be unique')
    return active_node_id, inactive_node_id


def _map_node_ids_to_cols(
    node_ids: np.ndarray,
    *,
    node_id_to_col: dict[int, int],
    name: str,
) -> np.ndarray:
    try:
        cols = [node_id_to_col[int(node_id)] for node_id in node_ids.tolist()]
    except KeyError as exc:
        raise RefractionStaticDesignMatrixError(
            f'{name} contains a node ID without an active column'
        ) from exc
    return np.ascontiguousarray(cols, dtype=np.int64)


def _validate_matrix_package(
    *,
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    n_observations: int,
    n_parameters: int,
) -> None:
    if matrix.shape != (n_observations, n_parameters):
        raise RefractionStaticDesignMatrixError('refraction design matrix shape mismatch')
    if matrix.format != 'csr':
        raise RefractionStaticDesignMatrixError('refraction design matrix must be CSR')
    if matrix.dtype != np.float64:
        raise RefractionStaticDesignMatrixError(
            'refraction design matrix dtype must be float64'
        )
    if rhs_s.shape != (n_observations,):
        raise RefractionStaticDesignMatrixError('rhs_s shape mismatch')
    if rhs_s.dtype != np.float64:
        raise RefractionStaticDesignMatrixError('rhs_s dtype must be float64')


def _validate_bedrock_velocity_mode(value: object) -> BedrockVelocityMode:
    if value == 'solve_global':
        return 'solve_global'
    if value == 'fixed_global':
        return 'fixed_global'
    raise RefractionStaticDesignMatrixError(
        'model.bedrock_velocity_mode must be solve_global or fixed_global'
    )


def _validate_n_traces(value: int | None, *, default: int) -> int:
    if value is None:
        out = default
    else:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise RefractionStaticDesignMatrixError('input_model.n_traces must be an integer')
        out = int(value)
    if out <= 0:
        raise RefractionStaticDesignMatrixError(
            'input_model.n_traces must be greater than 0'
        )
    return out


def _coerce_unique_node_id(values: np.ndarray) -> np.ndarray:
    node_id = _coerce_1d_integer_int64(values, name='input_model.endpoint_table.node_id')
    if node_id.shape[0] == 0:
        raise RefractionStaticDesignMatrixError(
            'input_model.endpoint_table.node_id must contain at least one node'
        )
    if np.unique(node_id).shape[0] != node_id.shape[0]:
        raise RefractionStaticDesignMatrixError(
            'input_model.endpoint_table.node_id values must be unique'
        )
    return node_id


def _coerce_1d_real_numeric_float64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticDesignMatrixError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if not _is_real_numeric_dtype(arr.dtype):
        raise RefractionStaticDesignMatrixError(f'{name} must have a numeric dtype')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_1d_integer_int64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticDesignMatrixError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise RefractionStaticDesignMatrixError(f'{name} must contain integer values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not _is_real_numeric_dtype(arr.dtype):
        raise RefractionStaticDesignMatrixError(f'{name} must contain integer values')
    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        raise RefractionStaticDesignMatrixError(f'{name} must contain only finite values')
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise RefractionStaticDesignMatrixError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_1d_bool_array(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticDesignMatrixError(f'{name} must be a 1D array')
    if arr.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if not np.issubdtype(arr.dtype, np.bool_):
        raise RefractionStaticDesignMatrixError(f'{name} must have bool dtype')
    return np.ascontiguousarray(arr, dtype=bool)


def _coerce_positive_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionStaticDesignMatrixError(
            f'{name} must be finite and greater than 0'
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticDesignMatrixError(
            f'{name} must be finite and greater than 0'
        ) from exc
    if not np.isfinite(out) or out <= 0.0:
        raise RefractionStaticDesignMatrixError(
            f'{name} must be finite and greater than 0'
        )
    return out


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'RefractionStaticDesignMatrix',
    'RefractionStaticDesignMatrixError',
    'build_refraction_static_design_matrix',
    'build_refraction_static_design_matrix_from_arrays',
]
