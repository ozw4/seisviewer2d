"""Source-depth resolution for M4 refraction field corrections."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.services.refraction_static_types import (
    RefractionSourceDepthResult,
    RefractionStaticInputModel,
)

REFRACTION_SOURCE_DEPTH_QC_JSON_NAME = 'refraction_source_depth_qc.json'
REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME = 'refraction_source_depth_sources.csv'

_STATUS_DTYPE = '<U32'
_ENDPOINT_KEY_DTYPE = object
_DEFAULT_INCONSISTENCY_TOLERANCE_M = 0.01
_SOURCE_DEPTH_STATUSES = (
    'ok',
    'missing_source_depth',
    'invalid_source_depth',
    'inconsistent_source_depth',
    'exceeds_max_abs_source_depth',
    'inactive_source_endpoint',
)
_SOURCE_DEPTH_COLUMNS = (
    'source_endpoint_key',
    'source_endpoint_id',
    'source_node_id',
    'source_depth_m',
    'source_depth_status',
    'source_depth_pick_count',
    'source_depth_trace_count',
)


def resolve_refraction_source_depth_for_input_model(
    *,
    input_model: RefractionStaticInputModel,
    mode: str,
    source_depth_byte: int | None,
    positive_down: bool = True,
    max_abs_source_depth_m: float = 100.0,
    inconsistency_tolerance_m: float = _DEFAULT_INCONSISTENCY_TOLERANCE_M,
    job_dir: Path | None = None,
) -> RefractionSourceDepthResult:
    """Resolve trace-order source depths to source endpoint rows."""
    if input_model.source_endpoint_id_sorted is None:
        raise ValueError(
            'input_model.source_endpoint_id_sorted is required for source-depth '
            'resolution'
        )
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        source_endpoint_id_sorted=input_model.source_endpoint_id_sorted,
        source_node_id_sorted=input_model.source_node_id_sorted,
        source_depth_m_sorted=input_model.source_depth_m_sorted,
        mode=mode,
        source_depth_byte=source_depth_byte,
        positive_down=positive_down,
        max_abs_source_depth_m=max_abs_source_depth_m,
        inconsistency_tolerance_m=inconsistency_tolerance_m,
    )
    if job_dir is not None:
        write_refraction_source_depth_artifacts(Path(job_dir), result)
    return result


def resolve_refraction_source_depth(
    *,
    source_endpoint_key_sorted: np.ndarray,
    source_node_id_sorted: np.ndarray,
    mode: str,
    source_depth_byte: int | None,
    source_depth_m_sorted: np.ndarray | None = None,
    source_endpoint_id_sorted: np.ndarray | None = None,
    positive_down: bool = True,
    max_abs_source_depth_m: float = 100.0,
    inconsistency_tolerance_m: float = _DEFAULT_INCONSISTENCY_TOLERANCE_M,
) -> RefractionSourceDepthResult:
    """Aggregate positive-down source depth samples by source endpoint."""
    mode_text = _coerce_mode(mode)
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
    if source_depth_m_sorted is None:
        raw_depth = np.full(n_traces, np.nan, dtype=np.float64)
        depth_array_present = False
    else:
        raw_depth = _coerce_1d_float(
            source_depth_m_sorted,
            name='source_depth_m_sorted',
            expected_shape=(n_traces,),
        )
        depth_array_present = True
    depth = raw_depth if positive_down else -raw_depth
    max_abs = _positive_finite_float(
        max_abs_source_depth_m,
        name='max_abs_source_depth_m',
    )
    tolerance = _nonnegative_finite_float(
        inconsistency_tolerance_m,
        name='inconsistency_tolerance_m',
    )

    groups, first_positions = _source_endpoint_groups(keys)
    n_endpoints = int(first_positions.shape[0])
    out_key = keys[first_positions].astype(_ENDPOINT_KEY_DTYPE, copy=False)
    out_id = np.empty(n_endpoints, dtype=np.int64)
    out_node = np.empty(n_endpoints, dtype=np.int64)
    out_depth = np.full(n_endpoints, np.nan, dtype=np.float64)
    out_status = np.full(n_endpoints, 'ok', dtype=_STATUS_DTYPE)
    out_pick_count = np.zeros(n_endpoints, dtype=np.int64)
    out_trace_count = np.zeros(n_endpoints, dtype=np.int64)
    depth_required = mode_text != 'none'

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

        values = depth[trace_indices]
        finite = np.isfinite(values)
        positive = finite & (values > 0.0)
        valid_values = values[positive]
        out_pick_count[endpoint_index] = int(valid_values.shape[0])
        if valid_values.size:
            out_depth[endpoint_index] = float(np.median(valid_values))

        out_status[endpoint_index] = _source_depth_status(
            node_id=int(out_node[endpoint_index]),
            values=values,
            valid_values=valid_values,
            depth_array_present=depth_array_present,
            depth_required=depth_required,
            max_abs_source_depth_m=max_abs,
            inconsistency_tolerance_m=tolerance,
        )

    qc = _source_depth_qc(
        mode=mode_text,
        source_depth_byte=source_depth_byte,
        positive_down=positive_down,
        max_abs_source_depth_m=max_abs,
        inconsistency_tolerance_m=tolerance,
        source_depth_m=out_depth,
        status=out_status,
        n_source_endpoints=n_endpoints,
    )
    return RefractionSourceDepthResult(
        source_endpoint_key=np.ascontiguousarray(out_key, dtype=_ENDPOINT_KEY_DTYPE),
        source_endpoint_id=np.ascontiguousarray(out_id, dtype=np.int64),
        source_node_id=np.ascontiguousarray(out_node, dtype=np.int64),
        source_depth_m=np.ascontiguousarray(out_depth, dtype=np.float64),
        source_depth_status=np.ascontiguousarray(out_status, dtype=_STATUS_DTYPE),
        source_depth_pick_count=np.ascontiguousarray(out_pick_count, dtype=np.int64),
        source_depth_trace_count=np.ascontiguousarray(out_trace_count, dtype=np.int64),
        qc=qc,
    )


def write_refraction_source_depth_artifacts(
    job_dir: Path,
    result: RefractionSourceDepthResult,
) -> dict[str, Path]:
    """Write source-depth QC JSON and one-row-per-source CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_SOURCE_DEPTH_QC_JSON_NAME
    source_path = root / REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME
    _write_json_atomic(qc_path, result.qc)
    _write_csv_atomic(source_path, _source_depth_rows(result), _SOURCE_DEPTH_COLUMNS)
    return {
        'qc_json': qc_path,
        'sources_csv': source_path,
    }


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


def _source_depth_status(
    *,
    node_id: int,
    values: np.ndarray,
    valid_values: np.ndarray,
    depth_array_present: bool,
    depth_required: bool,
    max_abs_source_depth_m: float,
    inconsistency_tolerance_m: float,
) -> str:
    if node_id < 0:
        return 'inactive_source_endpoint'
    if not depth_array_present:
        return 'missing_source_depth' if depth_required else 'ok'

    finite = np.isfinite(values)
    nonfinite_invalid = np.isinf(values)
    nonpositive = finite & (values <= 0.0)
    if bool(np.any(nonfinite_invalid | nonpositive)):
        return 'invalid_source_depth'
    if depth_required and bool(np.any(np.isnan(values))):
        return 'missing_source_depth'
    if valid_values.size == 0:
        return 'missing_source_depth' if depth_required else 'ok'
    if bool(np.any(np.abs(valid_values) > max_abs_source_depth_m)):
        return 'exceeds_max_abs_source_depth'
    if (
        valid_values.size > 1
        and float(np.max(valid_values) - np.min(valid_values))
        > inconsistency_tolerance_m
    ):
        return 'inconsistent_source_depth'
    return 'ok'


def _source_depth_qc(
    *,
    mode: str,
    source_depth_byte: int | None,
    positive_down: bool,
    max_abs_source_depth_m: float,
    inconsistency_tolerance_m: float,
    source_depth_m: np.ndarray,
    status: np.ndarray,
    n_source_endpoints: int,
) -> dict[str, Any]:
    finite_depth = source_depth_m[np.isfinite(source_depth_m)]
    status_counts = {
        item: int(np.count_nonzero(status == item)) for item in _SOURCE_DEPTH_STATUSES
    }
    return {
        'source_depth_mode': mode,
        'source_depth_byte': None if source_depth_byte is None else int(source_depth_byte),
        'source_depth_positive_down': bool(positive_down),
        'max_abs_source_depth_m': float(max_abs_source_depth_m),
        'source_depth_inconsistency_tolerance_m': float(inconsistency_tolerance_m),
        'n_source_endpoints': int(n_source_endpoints),
        'n_sources_with_depth': int(finite_depth.shape[0]),
        'n_missing_source_depth': status_counts['missing_source_depth'],
        'n_invalid_source_depth': status_counts['invalid_source_depth'],
        'n_inconsistent_source_depth': status_counts['inconsistent_source_depth'],
        'n_exceeds_max_abs_source_depth': status_counts[
            'exceeds_max_abs_source_depth'
        ],
        'n_inactive_source_endpoints': status_counts['inactive_source_endpoint'],
        'min_source_depth_m': _finite_stat(finite_depth, 'min'),
        'median_source_depth_m': _finite_stat(finite_depth, 'median'),
        'max_source_depth_m': _finite_stat(finite_depth, 'max'),
        'status_counts': status_counts,
    }


def _source_depth_rows(
    result: RefractionSourceDepthResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        depth = float(result.source_depth_m[index])
        rows.append(
            {
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_endpoint_id': int(result.source_endpoint_id[index]),
                'source_node_id': int(result.source_node_id[index]),
                'source_depth_m': _csv_float(depth),
                'source_depth_status': str(result.source_depth_status[index]),
                'source_depth_pick_count': int(result.source_depth_pick_count[index]),
                'source_depth_trace_count': int(result.source_depth_trace_count[index]),
            }
        )
    return rows


def _coerce_mode(value: object) -> str:
    if value == 'none':
        return 'none'
    if value == 'weathering_velocity_time':
        return 'weathering_velocity_time'
    raise ValueError(f'unsupported source_depth mode: {value!r}')


def _coerce_1d_string(values: object, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return np.ascontiguousarray(arr.astype(object, copy=False), dtype=object)


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
    'REFRACTION_SOURCE_DEPTH_QC_JSON_NAME',
    'REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME',
    'resolve_refraction_source_depth',
    'resolve_refraction_source_depth_for_input_model',
    'write_refraction_source_depth_artifacts',
]
