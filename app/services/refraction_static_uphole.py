"""Uphole-time resolution for M4 refraction field corrections."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.services.refraction_static_types import (
    REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES,
    RefractionEndpointFieldCorrectionResult,
    RefractionStaticInputModel,
    RefractionUpholeResult,
)

REFRACTION_UPHOLE_QC_JSON_NAME = 'refraction_uphole_qc.json'
REFRACTION_UPHOLE_SOURCES_CSV_NAME = 'refraction_uphole_sources.csv'

_STATUS_DTYPE = '<U32'
_FIELD_STATUS_DTYPE = '<U48'
_ENDPOINT_KEY_DTYPE = object
_DEFAULT_INCONSISTENCY_TOLERANCE_S = 1.0e-6
_UPHOLE_STATUSES = (
    'ok',
    'missing_uphole_time',
    'invalid_uphole_time',
    'inconsistent_uphole_time',
    'exceeds_max_abs_uphole_time',
    'inactive_source_endpoint',
)
_UPHOLE_COLUMNS = (
    'source_endpoint_key',
    'source_endpoint_id',
    'source_node_id',
    'uphole_time_s',
    'uphole_status',
    'uphole_pick_count',
    'uphole_trace_count',
)
_UPHOLE_COMPONENT = 'uphole_shift_s'
_NOT_APPLICABLE_STATUS = 'not_applicable'
_SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'


def resolve_refraction_uphole_for_input_model(
    *,
    input_model: RefractionStaticInputModel,
    uphole_time_sorted: np.ndarray | None,
    mode: str,
    uphole_time_byte: int | None,
    uphole_time_unit: str = 's',
    positive_time_means_delay: bool = True,
    max_abs_uphole_time_s: float = 1.0,
    inconsistency_tolerance_s: float = _DEFAULT_INCONSISTENCY_TOLERANCE_S,
    job_dir: Path | None = None,
) -> RefractionUpholeResult:
    """Resolve trace-order uphole times to source endpoint rows."""
    if input_model.source_endpoint_id_sorted is None:
        raise ValueError(
            'input_model.source_endpoint_id_sorted is required for uphole '
            'resolution'
        )
    result = resolve_refraction_uphole(
        source_endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        source_endpoint_id_sorted=input_model.source_endpoint_id_sorted,
        source_node_id_sorted=input_model.source_node_id_sorted,
        uphole_time_sorted=uphole_time_sorted,
        mode=mode,
        uphole_time_byte=uphole_time_byte,
        uphole_time_unit=uphole_time_unit,
        positive_time_means_delay=positive_time_means_delay,
        max_abs_uphole_time_s=max_abs_uphole_time_s,
        inconsistency_tolerance_s=inconsistency_tolerance_s,
    )
    if job_dir is not None:
        write_refraction_uphole_artifacts(Path(job_dir), result)
    return result


def resolve_refraction_uphole(
    *,
    source_endpoint_key_sorted: np.ndarray,
    source_node_id_sorted: np.ndarray,
    mode: str,
    uphole_time_byte: int | None,
    uphole_time_unit: str = 's',
    uphole_time_sorted: np.ndarray | None = None,
    source_endpoint_id_sorted: np.ndarray | None = None,
    positive_time_means_delay: bool = True,
    max_abs_uphole_time_s: float = 1.0,
    inconsistency_tolerance_s: float = _DEFAULT_INCONSISTENCY_TOLERANCE_S,
) -> RefractionUpholeResult:
    """Aggregate source uphole-time samples by source endpoint."""
    mode_text = _coerce_mode(mode)
    unit_text = _coerce_unit(uphole_time_unit)
    keys = _coerce_1d_string(
        source_endpoint_key_sorted,
        name='source_endpoint_key_sorted',
    )
    n_traces = int(keys.shape[0])
    nodes = _coerce_1d_integer(
        source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=(n_traces,),
    )
    endpoint_ids = _coerce_optional_1d_integer(
        source_endpoint_id_sorted,
        name='source_endpoint_id_sorted',
        expected_shape=(n_traces,),
    )
    if uphole_time_sorted is None:
        uphole = np.full(n_traces, np.nan, dtype=np.float64)
        uphole_array_present = False
    else:
        raw_uphole = _coerce_1d_float(
            uphole_time_sorted,
            name='uphole_time_sorted',
            expected_shape=(n_traces,),
        )
        uphole = _uphole_time_to_seconds(raw_uphole, unit_text)
        uphole_array_present = True
    max_abs = _positive_finite_float(
        max_abs_uphole_time_s,
        name='max_abs_uphole_time_s',
    )
    tolerance = _nonnegative_finite_float(
        inconsistency_tolerance_s,
        name='inconsistency_tolerance_s',
    )

    groups, first_positions = _source_endpoint_groups(keys)
    n_endpoints = int(first_positions.shape[0])
    out_key = keys[first_positions].astype(_ENDPOINT_KEY_DTYPE, copy=False)
    out_id = np.empty(n_endpoints, dtype=np.int64)
    out_node = np.empty(n_endpoints, dtype=np.int64)
    out_uphole = np.full(n_endpoints, np.nan, dtype=np.float64)
    out_status = np.full(n_endpoints, 'ok', dtype=_STATUS_DTYPE)
    out_pick_count = np.zeros(n_endpoints, dtype=np.int64)
    out_trace_count = np.zeros(n_endpoints, dtype=np.int64)
    uphole_required = mode_text != 'none'

    for endpoint_index, endpoint_key in enumerate(out_key.tolist()):
        trace_indices = np.asarray(groups[str(endpoint_key)], dtype=np.int64)
        out_trace_count[endpoint_index] = int(trace_indices.shape[0])
        out_id[endpoint_index] = _endpoint_id_for_group(
            endpoint_ids=endpoint_ids,
            trace_indices=trace_indices,
            fallback=endpoint_index,
        )
        node_values = nodes[trace_indices]
        out_node[endpoint_index] = _node_id_for_group(node_values)

        values = uphole[trace_indices]
        valid_values = values[np.isfinite(values)]
        out_pick_count[endpoint_index] = int(valid_values.shape[0])
        if valid_values.size:
            out_uphole[endpoint_index] = float(np.median(valid_values))

        out_status[endpoint_index] = _uphole_status(
            node_id=int(out_node[endpoint_index]),
            values=values,
            valid_values=valid_values,
            uphole_array_present=uphole_array_present,
            uphole_required=uphole_required,
            max_abs_uphole_time_s=max_abs,
            inconsistency_tolerance_s=tolerance,
        )

    qc = _uphole_qc(
        mode=mode_text,
        uphole_time_byte=uphole_time_byte,
        uphole_time_unit=unit_text,
        positive_time_means_delay=positive_time_means_delay,
        max_abs_uphole_time_s=max_abs,
        inconsistency_tolerance_s=tolerance,
        uphole_time_s=out_uphole,
        status=out_status,
        n_source_endpoints=n_endpoints,
    )
    return RefractionUpholeResult(
        source_endpoint_key=np.ascontiguousarray(out_key, dtype=_ENDPOINT_KEY_DTYPE),
        source_endpoint_id=np.ascontiguousarray(out_id, dtype=np.int64),
        source_node_id=np.ascontiguousarray(out_node, dtype=np.int64),
        uphole_time_s=np.ascontiguousarray(out_uphole, dtype=np.float64),
        uphole_status=np.ascontiguousarray(out_status, dtype=_STATUS_DTYPE),
        uphole_pick_count=np.ascontiguousarray(out_pick_count, dtype=np.int64),
        uphole_trace_count=np.ascontiguousarray(out_trace_count, dtype=np.int64),
        qc=qc,
    )


def write_refraction_uphole_artifacts(
    job_dir: Path,
    result: RefractionUpholeResult,
) -> dict[str, Path]:
    """Write uphole QC JSON and one-row-per-source CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_UPHOLE_QC_JSON_NAME
    source_path = root / REFRACTION_UPHOLE_SOURCES_CSV_NAME
    _write_json_atomic(qc_path, result.qc)
    _write_csv_atomic(source_path, _uphole_rows(result), _UPHOLE_COLUMNS)
    return {
        'qc_json': qc_path,
        'sources_csv': source_path,
    }


def compute_uphole_time_correction(
    uphole_time_s: np.ndarray,
    *,
    status: np.ndarray,
    positive_time_means_delay: bool = True,
    max_abs_uphole_time_s: float | None = None,
) -> RefractionEndpointFieldCorrectionResult:
    """Compute uphole shifts for source endpoints."""
    time_s = _coerce_1d_float(
        uphole_time_s,
        name='uphole_time_s',
        expected_shape=np.asarray(uphole_time_s).shape,
    )
    if time_s.ndim != 1:
        raise ValueError('uphole_time_s must be one-dimensional')
    endpoint_count = int(time_s.shape[0])
    status_array = _coerce_1d_string_status(
        status,
        name='status',
        expected_shape=(endpoint_count,),
    )
    endpoint_id = np.arange(endpoint_count, dtype=np.int64)
    endpoint_key = endpoint_id.astype('<U32')
    return _compute_uphole_time_correction_for_endpoints(
        endpoint_key=endpoint_key,
        endpoint_id=endpoint_id,
        node_id=endpoint_id,
        uphole_time_s=time_s,
        uphole_status=status_array,
        positive_time_means_delay=positive_time_means_delay,
        max_abs_uphole_time_s=max_abs_uphole_time_s,
    )


def compute_uphole_time_correction_from_result(
    result: RefractionUpholeResult,
    *,
    positive_time_means_delay: bool = True,
    max_abs_uphole_time_s: float | None = None,
) -> RefractionEndpointFieldCorrectionResult:
    """Compute uphole field shifts from resolved endpoint uphole rows."""
    return _compute_uphole_time_correction_for_endpoints(
        endpoint_key=result.source_endpoint_key,
        endpoint_id=result.source_endpoint_id,
        node_id=result.source_node_id,
        uphole_time_s=result.uphole_time_s,
        uphole_status=result.uphole_status,
        positive_time_means_delay=positive_time_means_delay,
        max_abs_uphole_time_s=max_abs_uphole_time_s,
    )


def _compute_uphole_time_correction_for_endpoints(
    *,
    endpoint_key: np.ndarray,
    endpoint_id: np.ndarray,
    node_id: np.ndarray,
    uphole_time_s: np.ndarray,
    uphole_status: np.ndarray,
    positive_time_means_delay: bool,
    max_abs_uphole_time_s: float | None,
) -> RefractionEndpointFieldCorrectionResult:
    keys = _coerce_1d_string(endpoint_key, name='endpoint_key')
    endpoint_count = int(keys.shape[0])
    ids = _coerce_1d_integer(
        endpoint_id,
        name='endpoint_id',
        expected_shape=(endpoint_count,),
    )
    nodes = _coerce_1d_integer(
        node_id,
        name='node_id',
        expected_shape=(endpoint_count,),
    )
    time_s = _coerce_1d_float(
        uphole_time_s,
        name='uphole_time_s',
        expected_shape=(endpoint_count,),
    )
    status = _coerce_1d_string_status(
        uphole_status,
        name='uphole_status',
        expected_shape=(endpoint_count,),
    )
    max_abs = (
        None
        if max_abs_uphole_time_s is None
        else _positive_finite_float(
            max_abs_uphole_time_s,
            name='max_abs_uphole_time_s',
        )
    )

    component_status = status.astype(_FIELD_STATUS_DTYPE, copy=True)
    shift = np.full(endpoint_count, np.nan, dtype=np.float64)

    ok = component_status == 'ok'
    missing = ok & np.isnan(time_s)
    component_status[missing] = 'missing_uphole_time'
    invalid = ok & ~np.isnan(time_s) & ~np.isfinite(time_s)
    component_status[invalid] = 'invalid_uphole_time'
    ok = component_status == 'ok'
    if max_abs is not None:
        exceeds = ok & (np.abs(time_s) > max_abs)
        component_status[exceeds] = 'exceeds_max_abs_uphole_time'
        ok = component_status == 'ok'

    sign = -1.0 if bool(positive_time_means_delay) else 1.0
    shift[ok] = sign * time_s[ok]
    total = np.where(component_status == 'ok', shift, np.nan).astype(np.float64)
    zeros = np.zeros(endpoint_count, dtype=np.float64)
    inactive_status = np.full(
        endpoint_count,
        _NOT_APPLICABLE_STATUS,
        dtype=_FIELD_STATUS_DTYPE,
    )
    component_shift_s = {
        'source_depth_shift_s': zeros.copy(),
        'uphole_shift_s': np.ascontiguousarray(shift, dtype=np.float64),
        'manual_static_shift_s': zeros.copy(),
    }
    component_status_by_name = {
        'source_depth_shift_s': inactive_status.copy(),
        'uphole_shift_s': np.ascontiguousarray(
            component_status,
            dtype=_FIELD_STATUS_DTYPE,
        ),
        'manual_static_shift_s': inactive_status.copy(),
    }
    qc = _uphole_shift_qc(
        uphole_time_s=time_s,
        shift_s=shift,
        status=component_status,
        positive_time_means_delay=bool(positive_time_means_delay),
        max_abs_uphole_time_s=max_abs,
        n_source_endpoints=endpoint_count,
    )
    return RefractionEndpointFieldCorrectionResult(
        endpoint_kind=np.full(endpoint_count, 'source', dtype='<U16'),
        endpoint_key=np.ascontiguousarray(keys, dtype=object),
        endpoint_id=np.ascontiguousarray(ids, dtype=np.int64),
        node_id=np.ascontiguousarray(nodes, dtype=np.int64),
        component_shift_s=component_shift_s,
        component_status=component_status_by_name,
        total_field_shift_s=np.ascontiguousarray(total, dtype=np.float64),
        field_static_status=np.ascontiguousarray(
            component_status,
            dtype=_FIELD_STATUS_DTYPE,
        ),
        qc=qc,
    )


def _uphole_shift_qc(
    *,
    uphole_time_s: np.ndarray,
    shift_s: np.ndarray,
    status: np.ndarray,
    positive_time_means_delay: bool,
    max_abs_uphole_time_s: float | None,
    n_source_endpoints: int,
) -> dict[str, Any]:
    finite_time = uphole_time_s[np.isfinite(uphole_time_s)]
    finite_shift = shift_s[np.isfinite(shift_s)]
    formula = _uphole_shift_formula(positive_time_means_delay)
    return {
        'uphole_mode': 'header_time',
        'component_name': _UPHOLE_COMPONENT,
        'component_names': list(REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES),
        'uphole_shift_formula': formula,
        'sign_convention': _SIGN_CONVENTION,
        'positive_time_means_delay': bool(positive_time_means_delay),
        'positive_shift': 'event appears later in corrected data',
        'negative_shift': 'event appears earlier in corrected data',
        'max_abs_uphole_time_s': (
            None if max_abs_uphole_time_s is None else float(max_abs_uphole_time_s)
        ),
        'n_source_endpoints': int(n_source_endpoints),
        'n_ok_uphole_shifts': int(np.count_nonzero(status == 'ok')),
        'n_invalid_uphole_shifts': int(np.count_nonzero(status != 'ok')),
        'status_counts': _status_counts(status),
        'min_uphole_time_s': _finite_stat(finite_time, 'min'),
        'median_uphole_time_s': _finite_stat(finite_time, 'median'),
        'max_uphole_time_s': _finite_stat(finite_time, 'max'),
        'min_uphole_shift_s': _finite_stat(finite_shift, 'min'),
        'median_uphole_shift_s': _finite_stat(finite_shift, 'median'),
        'max_uphole_shift_s': _finite_stat(finite_shift, 'max'),
    }


def _uphole_qc(
    *,
    mode: str,
    uphole_time_byte: int | None,
    uphole_time_unit: str,
    positive_time_means_delay: bool,
    max_abs_uphole_time_s: float,
    inconsistency_tolerance_s: float,
    uphole_time_s: np.ndarray,
    status: np.ndarray,
    n_source_endpoints: int,
) -> dict[str, Any]:
    finite_uphole = uphole_time_s[np.isfinite(uphole_time_s)]
    status_counts = {
        item: int(np.count_nonzero(status == item)) for item in _UPHOLE_STATUSES
    }
    return {
        'uphole_mode': mode,
        'uphole_time_byte': (
            None if uphole_time_byte is None else int(uphole_time_byte)
        ),
        'uphole_time_unit': uphole_time_unit,
        'positive_time_means_delay': bool(positive_time_means_delay),
        'uphole_shift_formula': _uphole_shift_formula(positive_time_means_delay),
        'sign_convention': _SIGN_CONVENTION,
        'max_abs_uphole_time_s': float(max_abs_uphole_time_s),
        'uphole_inconsistency_tolerance_s': float(inconsistency_tolerance_s),
        'n_source_endpoints': int(n_source_endpoints),
        'n_sources_with_uphole': int(finite_uphole.shape[0]),
        'n_missing_uphole': status_counts['missing_uphole_time'],
        'n_invalid_uphole': status_counts['invalid_uphole_time'],
        'n_inconsistent_uphole': status_counts['inconsistent_uphole_time'],
        'n_exceeds_max_abs_uphole': status_counts[
            'exceeds_max_abs_uphole_time'
        ],
        'n_inactive_source_endpoints': status_counts['inactive_source_endpoint'],
        'min_uphole_time_s': _finite_stat(finite_uphole, 'min'),
        'median_uphole_time_s': _finite_stat(finite_uphole, 'median'),
        'max_uphole_time_s': _finite_stat(finite_uphole, 'max'),
        'status_counts': status_counts,
    }


def _uphole_status(
    *,
    node_id: int,
    values: np.ndarray,
    valid_values: np.ndarray,
    uphole_array_present: bool,
    uphole_required: bool,
    max_abs_uphole_time_s: float,
    inconsistency_tolerance_s: float,
) -> str:
    if node_id < 0:
        return 'inactive_source_endpoint'
    if not uphole_array_present:
        return 'missing_uphole_time' if uphole_required else 'ok'

    nonfinite_invalid = np.isinf(values)
    if bool(np.any(nonfinite_invalid)):
        return 'invalid_uphole_time'
    if uphole_required and bool(np.any(np.isnan(values))):
        return 'missing_uphole_time'
    if valid_values.size == 0:
        return 'missing_uphole_time' if uphole_required else 'ok'
    if bool(np.any(np.abs(valid_values) > max_abs_uphole_time_s)):
        return 'exceeds_max_abs_uphole_time'
    if (
        valid_values.size > 1
        and float(np.max(valid_values) - np.min(valid_values))
        > inconsistency_tolerance_s
    ):
        return 'inconsistent_uphole_time'
    return 'ok'


def _source_endpoint_groups(
    keys: np.ndarray,
) -> tuple[dict[str, list[int]], np.ndarray]:
    groups: dict[str, list[int]] = {}
    first_positions: list[int] = []
    for index, raw_key in enumerate(keys.tolist()):
        key = str(raw_key)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
            first_positions.append(index)
        groups[key].append(index)
    return groups, np.asarray(first_positions, dtype=np.int64)


def _endpoint_id_for_group(
    *,
    endpoint_ids: np.ndarray | None,
    trace_indices: np.ndarray,
    fallback: int,
) -> int:
    if endpoint_ids is None:
        return int(fallback)
    return int(endpoint_ids[int(trace_indices[0])])


def _node_id_for_group(node_values: np.ndarray) -> int:
    nonnegative = node_values[node_values >= 0]
    if nonnegative.size:
        return int(nonnegative[0])
    if node_values.size:
        return int(node_values[0])
    return -1


def _uphole_rows(result: RefractionUpholeResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        uphole_time = float(result.uphole_time_s[index])
        rows.append(
            {
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_endpoint_id': int(result.source_endpoint_id[index]),
                'source_node_id': int(result.source_node_id[index]),
                'uphole_time_s': _csv_float(uphole_time),
                'uphole_status': str(result.uphole_status[index]),
                'uphole_pick_count': int(result.uphole_pick_count[index]),
                'uphole_trace_count': int(result.uphole_trace_count[index]),
            }
        )
    return rows


def _uphole_time_to_seconds(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == 's':
        return np.ascontiguousarray(values, dtype=np.float64)
    if unit == 'ms':
        return np.ascontiguousarray(values / 1000.0, dtype=np.float64)
    raise ValueError(f'unsupported uphole_time_unit: {unit!r}')


def _uphole_shift_formula(positive_time_means_delay: bool) -> str:
    if bool(positive_time_means_delay):
        return 'uphole_shift_s = -uphole_time_s'
    return 'uphole_shift_s = +uphole_time_s'


def _coerce_mode(value: object) -> str:
    if value == 'none':
        return 'none'
    if value == 'header_time':
        return 'header_time'
    if value == 'manual_table':
        raise ValueError('uphole mode manual_table is not implemented')
    raise ValueError(f'unsupported uphole mode: {value!r}')


def _coerce_unit(value: object) -> str:
    if value in {'s', 'ms'}:
        return str(value)
    raise ValueError(f'unsupported uphole_time_unit: {value!r}')


def _coerce_1d_string(values: object, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return np.ascontiguousarray(arr.astype(object, copy=False), dtype=object)


def _coerce_1d_string_status(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != expected_shape:
        raise ValueError(f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}')
    return np.ascontiguousarray(arr.astype(_FIELD_STATUS_DTYPE, copy=False))


def _coerce_1d_integer(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != expected_shape:
        raise ValueError(f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{name} must have an integer dtype')
    return np.ascontiguousarray(arr, dtype=np.int64)


def _coerce_optional_1d_integer(
    values: object | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None
    return _coerce_1d_integer(values, name=name, expected_shape=expected_shape)


def _coerce_1d_float(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != expected_shape:
        raise ValueError(f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.number,
    ):
        raise ValueError(f'{name} must have a real numeric dtype')
    if np.iscomplexobj(arr):
        raise ValueError(f'{name} must have a real numeric dtype')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _positive_finite_float(value: object, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise ValueError(f'{name} must be positive')
    return out


def _nonnegative_finite_float(value: object, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out < 0.0:
        raise ValueError(f'{name} must be nonnegative')
    return out


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a finite number')
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


def _finite_stat(values: np.ndarray, stat: str) -> float | None:
    if values.size == 0:
        return None
    if stat == 'min':
        return float(np.min(values))
    if stat == 'median':
        return float(np.median(values))
    if stat == 'max':
        return float(np.max(values))
    raise ValueError(f'unsupported finite stat: {stat}')


def _status_counts(status: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw_status in status.tolist():
        item = str(raw_status)
        out[item] = int(out.get(item, 0) + 1)
    return out


def _csv_float(value: float) -> str:
    if not np.isfinite(value):
        return ''
    return f'{float(value):.17g}'


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='raise')
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


__all__ = [
    'REFRACTION_UPHOLE_QC_JSON_NAME',
    'REFRACTION_UPHOLE_SOURCES_CSV_NAME',
    'compute_uphole_time_correction',
    'compute_uphole_time_correction_from_result',
    'resolve_refraction_uphole',
    'resolve_refraction_uphole_for_input_model',
    'write_refraction_uphole_artifacts',
]
