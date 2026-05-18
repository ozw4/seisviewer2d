"""Endpoint geometry table builder for static linkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.services.geometry_linkage_validation import GeometryLinkageHeaders
from app.utils.segy_scalars import (
    apply_segy_scalar,
    count_zero_segy_scalars,
    normalize_linear_unit,
)

EndpointKind = Literal['source', 'receiver']


@dataclass(frozen=True)
class EndpointGeometryTable:
    endpoint_kind: EndpointKind
    endpoint_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    first_sorted_trace_index: np.ndarray
    trace_count: np.ndarray


@dataclass(frozen=True)
class EndpointGeometryTables:
    n_traces: int
    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray
    coordinate_scalar_sorted: np.ndarray
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray
    source_endpoints: EndpointGeometryTable
    receiver_endpoints: EndpointGeometryTable
    scalar_zero_count: int


def build_endpoint_geometry_tables(
    headers: GeometryLinkageHeaders,
    *,
    coordinate_unit: Literal['m', 'ft'] = 'm',
) -> EndpointGeometryTables:
    """Build exact-coordinate source and receiver endpoint tables in meters."""
    source_x_raw = _coerce_1d_numeric(headers.source_x, name='source_x')
    source_y_raw = _coerce_1d_numeric(headers.source_y, name='source_y')
    receiver_x_raw = _coerce_1d_numeric(headers.receiver_x, name='receiver_x')
    receiver_y_raw = _coerce_1d_numeric(headers.receiver_y, name='receiver_y')
    scalars = _coerce_1d_integer_scalars(
        headers.coordinate_scalar,
        name='coordinate_scalar',
    )

    expected_shape = source_x_raw.shape
    _validate_matching_shape(source_y_raw, name='source_y', expected_shape=expected_shape)
    _validate_matching_shape(
        receiver_x_raw,
        name='receiver_x',
        expected_shape=expected_shape,
    )
    _validate_matching_shape(
        receiver_y_raw,
        name='receiver_y',
        expected_shape=expected_shape,
    )
    _validate_matching_shape(
        scalars,
        name='coordinate_scalar',
        expected_shape=expected_shape,
    )

    n_traces = int(source_x_raw.shape[0])
    if n_traces <= 0:
        msg = 'geometry linkage headers must contain at least one trace'
        raise ValueError(msg)

    source_x_m = normalize_linear_unit(
        apply_segy_scalar(source_x_raw, scalars),
        coordinate_unit,
    )
    source_y_m = normalize_linear_unit(
        apply_segy_scalar(source_y_raw, scalars),
        coordinate_unit,
    )
    receiver_x_m = normalize_linear_unit(
        apply_segy_scalar(receiver_x_raw, scalars),
        coordinate_unit,
    )
    receiver_y_m = normalize_linear_unit(
        apply_segy_scalar(receiver_y_raw, scalars),
        coordinate_unit,
    )

    source_table, source_inverse = _build_unique_endpoint_table(
        'source',
        source_x_m,
        source_y_m,
    )
    receiver_table, receiver_inverse = _build_unique_endpoint_table(
        'receiver',
        receiver_x_m,
        receiver_y_m,
    )

    return EndpointGeometryTables(
        n_traces=n_traces,
        source_x_m_sorted=np.ascontiguousarray(source_x_m, dtype=np.float64),
        source_y_m_sorted=np.ascontiguousarray(source_y_m, dtype=np.float64),
        receiver_x_m_sorted=np.ascontiguousarray(receiver_x_m, dtype=np.float64),
        receiver_y_m_sorted=np.ascontiguousarray(receiver_y_m, dtype=np.float64),
        coordinate_scalar_sorted=np.ascontiguousarray(scalars, dtype=np.int64),
        source_endpoint_id_sorted=np.ascontiguousarray(source_inverse, dtype=np.int64),
        receiver_endpoint_id_sorted=np.ascontiguousarray(
            receiver_inverse,
            dtype=np.int64,
        ),
        source_endpoints=source_table,
        receiver_endpoints=receiver_table,
        scalar_zero_count=count_zero_segy_scalars(scalars),
    )


def _build_unique_endpoint_table(
    endpoint_kind: EndpointKind,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[EndpointGeometryTable, np.ndarray]:
    xy = np.column_stack((x_m, y_m))
    unique_xy, first_index, inverse, counts = np.unique(
        xy,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    n_endpoints = int(unique_xy.shape[0])
    endpoint_id = np.arange(n_endpoints, dtype=np.int64)
    table = EndpointGeometryTable(
        endpoint_kind=endpoint_kind,
        endpoint_id=endpoint_id,
        x_m=np.ascontiguousarray(unique_xy[:, 0], dtype=np.float64),
        y_m=np.ascontiguousarray(unique_xy[:, 1], dtype=np.float64),
        first_sorted_trace_index=np.ascontiguousarray(first_index, dtype=np.int64),
        trace_count=np.ascontiguousarray(counts, dtype=np.int64),
    )
    return table, np.asarray(inverse, dtype=np.int64)


def _coerce_1d_numeric(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} must have a real numeric dtype'
        raise ValueError(msg)
    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    return arr_f64


def _coerce_1d_integer_scalars(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.integer):
        return np.asarray(arr, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.floating):
        msg = f'{name} must have an integer dtype or integer-valued float dtype'
        raise ValueError(msg)

    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        msg = f'{name} must contain only integer values'
        raise ValueError(msg)

    int64_info = np.iinfo(np.int64)
    if arr_f64.size and (
        float(arr_f64.min()) < int64_info.min
        or float(arr_f64.max()) > int64_info.max
    ):
        msg = f'{name} values are outside the int64 range'
        raise ValueError(msg)
    return np.asarray(arr_f64, dtype=np.int64)


def _validate_matching_shape(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    if values.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {values.shape}'
        raise ValueError(msg)


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'EndpointGeometryTable',
    'EndpointGeometryTables',
    'EndpointKind',
    'build_endpoint_geometry_tables',
]
