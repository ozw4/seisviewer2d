"""All-gather pick-map QC for refraction statics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.statics.refraction.contracts.qc import (
    RefractionStaticPickMapRequest,
    RefractionStaticPickMapResponse,
)
from app.statics.refraction.artifacts import (
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_PICK_MAP_QC_COMPLETED_CACHE_DIR_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
)
from app.statics.refraction.application.job_status import (
    is_ready_status_value,
    normalize_status_value,
)
from app.statics.refraction.application.pick_source_loader import (
    load_npz_refraction_pick_source_from_path,
)
from app.statics.refraction.ports.runtime import RefractionRuntime
from app.services.trace_store_index_validation import validate_sorted_to_original
from app.utils.segy_scalars import apply_segy_scalar, normalize_linear_unit


class RefractionStaticPickMapError(ValueError):
    """Raised when a pick-map bundle cannot be assembled."""


_COMPLETED_PICK_MAP_CACHE_SCHEMA_VERSION = 1


def build_refraction_static_pick_map(
    *,
    req: RefractionStaticPickMapRequest,
    runtime: RefractionRuntime,
    job: dict[str, object] | None = None,
    uploaded_pick_npz_path: Path | None = None,
) -> dict[str, Any]:
    """Build a Plotly-ready all-gather pick-map response."""
    if req.job_id is not None:
        if job is None:
            raise RefractionStaticPickMapError('job metadata is required')
        return _build_completed_job_pick_map(req=req, job=job)
    return _build_pre_statics_pick_map(
        req=req,
        runtime=runtime,
        uploaded_pick_npz_path=uploaded_pick_npz_path,
    )


def _build_pre_statics_pick_map(
    *,
    req: RefractionStaticPickMapRequest,
    runtime: RefractionRuntime,
    uploaded_pick_npz_path: Path | None,
) -> dict[str, Any]:
    if req.file_id is None or req.pick_source is None:
        raise RefractionStaticPickMapError('file_id and pick_source are required')

    reader = runtime.trace_store.get_reader(req.file_id, req.key1_byte, req.key2_byte)
    n_traces = _reader_n_traces(reader)
    n_samples = _reader_n_samples(reader)
    dt_s = _reader_dt(reader, runtime=runtime, file_id=req.file_id)
    sorted_to_original = validate_sorted_to_original(
        np.asarray(reader.get_sorted_to_original()),
        expected_n_traces=n_traces,
        role='reader',
    )
    pick_path = _resolve_pick_path(
        req=req,
        runtime=runtime,
        uploaded_pick_npz_path=uploaded_pick_npz_path,
    )
    loaded = load_npz_refraction_pick_source_from_path(
        pick_path,
        n_traces=n_traces,
        n_samples=n_samples,
        dt_s=dt_s,
        sorted_trace_index=sorted_to_original,
        source_kind=req.pick_source.kind,
        allow_invalid_pick_values=True,
    )

    geometry = req.geometry
    source_id = _header_as_python_values(
        reader.ensure_header(geometry.source_id_byte),
        n_traces=n_traces,
        name='source_id',
    )
    receiver_id = _header_as_python_values(
        reader.ensure_header(geometry.receiver_id_byte),
        n_traces=n_traces,
        name='receiver_id',
    )
    offset_m = _geometry_offset_m(reader=reader, req=req, n_traces=n_traces)

    pick_s = np.asarray(loaded.picks_time_s_sorted, dtype=np.float64)
    valid = np.isfinite(pick_s) & (pick_s >= 0.0)
    records = _empty_pick_map()
    for index in np.flatnonzero(valid):
        pick_ms = float(pick_s[index] * 1000.0)
        records['gather_id'].append(source_id[index])
        records['receiver_number'].append(
            _global_sequential_receiver_number(receiver_id[index])
        )
        records['pick_before_ms'].append(pick_ms)
        records['trace_index'].append(int(index))
        records['shot_id'].append(source_id[index])
        records['source_id'].append(source_id[index])
        records['receiver_id'].append(receiver_id[index])
        records['offset_m'].append(_finite_float_or_none(offset_m[index]))
        records['used_in_statics'].append(None)
        records['pick_after_ms'].append(None)
        records['applied_shift_ms'].append(None)
        records['offset_used'].append(None)

    return _response(
        job_id=None,
        mode='pre_statics',
        status_message=(
            'Pre-statics pick map loaded. Static Correction has not been run, '
            'so After Statics is unavailable.'
        ),
        has_after_statics=False,
        receiver_number_mode=req.geometry.receiver_number_mode,
        pick_map=records,
    )


def _resolve_pick_path(
    *,
    req: RefractionStaticPickMapRequest,
    runtime: RefractionRuntime,
    uploaded_pick_npz_path: Path | None,
) -> Path:
    pick_source = req.pick_source
    if pick_source is None:
        raise RefractionStaticPickMapError('pick_source is required')
    if pick_source.kind == 'uploaded_npz':
        if uploaded_pick_npz_path is None:
            raise RefractionStaticPickMapError(
                'pick_source.kind=uploaded_npz requires multipart pick_npz upload'
            )
        return Path(uploaded_pick_npz_path)
    if req.file_id is None:
        raise RefractionStaticPickMapError('file_id is required')
    if pick_source.kind == 'manual_memmap':
        raise RefractionStaticPickMapError(
            'manual_memmap pick sources are not supported for pick-map QC'
        )
    allowed_job_types = (
        {'batch_apply'} if pick_source.kind == 'batch_predicted_npz' else {'statics'}
    )
    return runtime.artifacts.resolve_artifact(
        job_id=str(pick_source.job_id),
        name=str(pick_source.artifact_name),
        allowed_job_types=allowed_job_types,
        expected_file_id=req.file_id,
        expected_key1_byte=req.key1_byte,
        expected_key2_byte=req.key2_byte,
        reference_label='pick_source',
    )


def _build_completed_job_pick_map(
    *,
    req: RefractionStaticPickMapRequest,
    job: dict[str, object],
) -> dict[str, Any]:
    job_id = str(req.job_id)
    if job.get('statics_kind') != 'refraction':
        raise RefractionStaticPickMapError(f'Job {job_id} is not a refraction statics job')
    if not is_ready_status_value(job.get('status')):
        raise RefractionStaticPickMapError(
            f'Job {job_id} is not complete; current state is '
            f'{normalize_status_value(job.get("status"))}'
        )
    artifacts_dir = _job_artifacts_dir(job, job_id)
    cached = _load_completed_pick_map_cache(artifacts_dir=artifacts_dir, req=req)
    if cached is not None:
        return cached

    payload = _build_completed_job_pick_map_uncached(
        req=req,
        job_id=job_id,
        artifacts_dir=artifacts_dir,
    )
    _write_completed_pick_map_cache(
        artifacts_dir=artifacts_dir,
        req=req,
        payload=payload,
    )
    return payload


def _build_completed_job_pick_map_uncached(
    *,
    req: RefractionStaticPickMapRequest,
    job_id: str,
    artifacts_dir: Path,
) -> dict[str, Any]:
    pick_rows = _read_completed_pick_rows(artifacts_dir)
    shifts_ms = _read_completed_trace_shifts_ms(artifacts_dir)

    records = _empty_pick_map()
    for row in pick_rows:
        before_ms = _required_float(row.get('observed_first_break_time_s')) * 1000.0
        trace_index = _optional_int(row.get('trace_index_sorted'))
        if trace_index is None:
            trace_index = _optional_int(row.get('sorted_trace_index'))
        shift_ms = shifts_ms.get(trace_index) if trace_index is not None else None
        used = _bool_value(row.get('used_in_solve', row.get('used_for_inversion')))
        after_ms = before_ms + shift_ms if shift_ms is not None else None
        offset = _optional_float(row.get('offset_m'))
        source_id = _first_present(row, 'source_id', 'source_endpoint_key')
        if not _include_gather_id_in_request_range(source_id, req):
            continue
        receiver_id = _first_present(row, 'receiver_id', 'receiver_endpoint_key')
        receiver_number = _global_sequential_receiver_number(receiver_id)

        records['gather_id'].append(source_id)
        records['receiver_number'].append(receiver_number)
        records['pick_before_ms'].append(before_ms)
        records['trace_index'].append(trace_index)
        records['shot_id'].append(source_id)
        records['source_id'].append(source_id)
        records['receiver_id'].append(receiver_id)
        records['offset_m'].append(offset)
        records['used_in_statics'].append(used)
        records['pick_after_ms'].append(after_ms)
        records['applied_shift_ms'].append(shift_ms)
        records['offset_used'].append(offset if used else None)

    return _response(
        job_id=job_id,
        mode='completed_job',
        status_message='Static job loaded. Before/After pick comparison is available.',
        has_after_statics=True,
        receiver_number_mode=req.geometry.receiver_number_mode,
        pick_map=records,
    )


def _load_completed_pick_map_cache(
    *,
    artifacts_dir: Path,
    req: RefractionStaticPickMapRequest,
) -> dict[str, Any] | None:
    path = _completed_pick_map_cache_path(artifacts_dir=artifacts_dir, req=req)
    if not path.is_file():
        return None
    try:
        with path.open(encoding='utf-8') as handle:
            payload = json.load(handle)
        return RefractionStaticPickMapResponse.model_validate(payload).model_dump(
            mode='json'
        )
    except Exception:  # noqa: BLE001
        path.unlink(missing_ok=True)
        return None


def _write_completed_pick_map_cache(
    *,
    artifacts_dir: Path,
    req: RefractionStaticPickMapRequest,
    payload: dict[str, Any],
) -> None:
    model = RefractionStaticPickMapResponse.model_validate(payload)
    path = _completed_pick_map_cache_path(artifacts_dir=artifacts_dir, req=req)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(
                model.model_dump(mode='json'),
                handle,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(',', ':'),
            )
            handle.write('\n')
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _completed_pick_map_cache_path(
    *,
    artifacts_dir: Path,
    req: RefractionStaticPickMapRequest,
) -> Path:
    digest = hashlib.sha256(
        json.dumps(
            _completed_pick_map_cache_key(req),
            ensure_ascii=True,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    return (
        artifacts_dir
        / REFRACTION_PICK_MAP_QC_COMPLETED_CACHE_DIR_NAME
        / f'{digest}.json'
    )


def _completed_pick_map_cache_key(
    req: RefractionStaticPickMapRequest,
) -> dict[str, Any]:
    return {
        'job_id': req.job_id,
        'gather_start': req.gather_start,
        'gather_end': req.gather_end,
        'response_schema_version': _COMPLETED_PICK_MAP_CACHE_SCHEMA_VERSION,
        'receiver_number_mode': req.geometry.receiver_number_mode,
        'source_artifact_names': [
            REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
            REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
            REFRACTION_STATIC_SOLUTION_NPZ_NAME,
            REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
        ],
    }


def _include_gather_id_in_request_range(
    gather_id: object,
    req: RefractionStaticPickMapRequest,
) -> bool:
    gather_number = _numeric_or_none(gather_id)
    if gather_number is None:
        return True
    if req.gather_start is not None and gather_number < req.gather_start:
        return False
    if req.gather_end is not None and gather_number > req.gather_end:
        return False
    return True


def _read_completed_pick_rows(artifacts_dir: Path) -> list[dict[str, Any]]:
    npz_path = artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME
    if npz_path.is_file():
        return _read_pick_rows_from_npz(npz_path)
    csv_path = artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME
    if csv_path.is_file():
        with csv_path.open(encoding='utf-8', newline='') as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise RefractionStaticPickMapError(
        f'Refraction pick map requires artifact {REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME}'
    )


def _read_pick_rows_from_npz(path: Path) -> list[dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = ('observed_first_break_time_s',)
            for key in required:
                if key not in data.files:
                    raise RefractionStaticPickMapError(
                        f'Refraction pick map artifact is missing {key}'
                    )
            n_rows = int(np.asarray(data['observed_first_break_time_s']).shape[0])
            rows: list[dict[str, Any]] = []
            for index in range(n_rows):
                row: dict[str, Any] = {}
                for key in data.files:
                    value = np.asarray(data[key])
                    if value.ndim == 0:
                        continue
                    if value.shape[0] != n_rows:
                        continue
                    row[key] = _np_scalar_to_python(value[index])
                rows.append(row)
            return rows
    except RefractionStaticPickMapError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RefractionStaticPickMapError(
            f'Could not read refraction pick map artifact {path.name}'
        ) from exc


def _read_completed_trace_shifts_ms(artifacts_dir: Path) -> dict[int, float]:
    solution_path = artifacts_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME
    if solution_path.is_file():
        return _read_trace_shifts_from_solution_npz(solution_path)
    csv_path = artifacts_dir / REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME
    if csv_path.is_file():
        return _read_trace_shifts_from_component_csv(csv_path)
    return {}


def _read_trace_shifts_from_solution_npz(path: Path) -> dict[int, float]:
    try:
        with np.load(path, allow_pickle=False) as data:
            field = (
                'final_trace_shift_s_sorted'
                if 'final_trace_shift_s_sorted' in data.files
                else 'refraction_trace_shift_s_sorted'
            )
            if field not in data.files:
                return {}
            shifts = np.asarray(data[field], dtype=np.float64)
            return {
                int(index): float(shift_s * 1000.0)
                for index, shift_s in enumerate(shifts)
                if np.isfinite(shift_s)
            }
    except Exception as exc:  # noqa: BLE001
        raise RefractionStaticPickMapError(
            f'Could not read refraction trace shifts from {path.name}'
        ) from exc


def _read_trace_shifts_from_component_csv(path: Path) -> dict[int, float]:
    shifts: dict[int, float] = {}
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            trace_index = _optional_int(row.get('trace_index_sorted'))
            shift_ms = _optional_float(
                row.get('applied_trace_shift_ms', row.get('final_trace_shift_ms'))
            )
            if trace_index is not None and shift_ms is not None:
                shifts[trace_index] = shift_ms
    return shifts


def _geometry_offset_m(
    *,
    reader: Any,
    req: RefractionStaticPickMapRequest,
    n_traces: int,
) -> np.ndarray:
    geometry = req.geometry
    try:
        source_x = _float_header(reader.ensure_header(geometry.source_x_byte), n_traces)
        source_y = _float_header(reader.ensure_header(geometry.source_y_byte), n_traces)
        receiver_x = _float_header(reader.ensure_header(geometry.receiver_x_byte), n_traces)
        receiver_y = _float_header(reader.ensure_header(geometry.receiver_y_byte), n_traces)
        coordinate_scalar = _int_header(
            reader.ensure_header(geometry.coordinate_scalar_byte),
            n_traces,
        )
    except Exception:
        return np.full(n_traces, np.nan, dtype=np.float64)
    try:
        source_x = normalize_linear_unit(
            apply_segy_scalar(source_x, coordinate_scalar),
            geometry.coordinate_unit,
        )
        source_y = normalize_linear_unit(
            apply_segy_scalar(source_y, coordinate_scalar),
            geometry.coordinate_unit,
        )
        receiver_x = normalize_linear_unit(
            apply_segy_scalar(receiver_x, coordinate_scalar),
            geometry.coordinate_unit,
        )
        receiver_y = normalize_linear_unit(
            apply_segy_scalar(receiver_y, coordinate_scalar),
            geometry.coordinate_unit,
        )
    except Exception:
        return np.full(n_traces, np.nan, dtype=np.float64)
    return np.ascontiguousarray(
        np.hypot(source_x - receiver_x, source_y - receiver_y),
        dtype=np.float64,
    )


def _response(
    *,
    job_id: str | None,
    mode: str,
    status_message: str,
    has_after_statics: bool,
    receiver_number_mode: str,
    pick_map: dict[str, list[Any]],
) -> dict[str, Any]:
    return {
        'job_id': job_id,
        'statics_kind': 'refraction',
        'mode': mode,
        'status_message': status_message,
        'has_after_statics': has_after_statics,
        'receiver_number_mode': receiver_number_mode,
        'gather_range': _gather_range(pick_map['gather_id']),
        'pick_map': pick_map,
    }


def _empty_pick_map() -> dict[str, list[Any]]:
    return {
        'gather_id': [],
        'receiver_number': [],
        'pick_before_ms': [],
        'trace_index': [],
        'shot_id': [],
        'source_id': [],
        'receiver_id': [],
        'offset_m': [],
        'used_in_statics': [],
        'pick_after_ms': [],
        'applied_shift_ms': [],
        'offset_used': [],
    }


def _gather_range(values: list[Any]) -> dict[str, int | float | str | None]:
    numeric = [_numeric_or_none(value) for value in values]
    finite = [float(value) for value in numeric if value is not None]
    if finite:
        start = min(finite)
        end = max(finite)
        if start.is_integer() and end.is_integer():
            return {'min': int(start), 'max': int(end)}
        return {'min': start, 'max': end}
    text = [str(value) for value in values if value is not None and str(value)]
    if text:
        return {'min': min(text), 'max': max(text)}
    return {'min': None, 'max': None}


def _job_artifacts_dir(job: dict[str, object], job_id: str) -> Path:
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise RefractionStaticPickMapError(
            f'Job {job_id} metadata is missing artifacts_dir'
        )
    path = Path(raw)
    if not path.is_dir():
        raise RefractionStaticPickMapError(
            f'Job {job_id} artifacts directory is not available'
        )
    return path


def _reader_n_traces(reader: Any) -> int:
    shape = getattr(getattr(reader, 'traces', None), 'shape', ())
    if shape:
        value = int(shape[0])
        if value > 0:
            return value
    meta = getattr(reader, 'meta', {})
    value = int(meta.get('n_traces', 0)) if isinstance(meta, dict) else 0
    if value <= 0:
        raise RefractionStaticPickMapError('TraceStore metadata unavailable: n_traces')
    return value


def _reader_n_samples(reader: Any) -> int:
    shape = getattr(getattr(reader, 'traces', None), 'shape', ())
    if len(shape) >= 2 and int(shape[-1]) > 0:
        return int(shape[-1])
    raise RefractionStaticPickMapError('TraceStore metadata unavailable: n_samples')


def _reader_dt(reader: Any, *, runtime: RefractionRuntime, file_id: str) -> float:
    meta = getattr(reader, 'meta', {})
    if isinstance(meta, dict):
        raw = meta.get('dt')
        if isinstance(raw, (int, float)) and raw > 0:
            return float(raw)
    value = float(runtime.trace_store.get_dt(file_id))
    if value <= 0.0:
        raise RefractionStaticPickMapError('TraceStore metadata unavailable: dt')
    return value


def _header_as_python_values(values: object, *, n_traces: int, name: str) -> list[Any]:
    arr = np.asarray(values)
    if arr.shape != (n_traces,):
        raise RefractionStaticPickMapError(
            f'{name} header shape mismatch: expected {(n_traces,)}, got {arr.shape}'
        )
    return [_np_scalar_to_python(item) for item in arr]


def _float_header(values: object, n_traces: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_traces,):
        raise RefractionStaticPickMapError(
            f'header shape mismatch: expected {(n_traces,)}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _int_header(values: object, n_traces: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape != (n_traces,):
        raise RefractionStaticPickMapError(
            f'header shape mismatch: expected {(n_traces,)}, got {arr.shape}'
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise RefractionStaticPickMapError('header must have an integer dtype')
    return np.ascontiguousarray(arr, dtype=np.int64)


def _required_float(value: object) -> float:
    out = _optional_float(value)
    if out is None:
        raise RefractionStaticPickMapError('completed pick row has invalid pick time')
    return out


def _optional_float(value: object) -> float | None:
    if value is None or value == '':
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _finite_float_or_none(value: object) -> float | None:
    return _optional_float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out) or not out.is_integer():
            return None
        return int(out)


def _numeric_or_none(value: object) -> int | float | None:
    out = _optional_float(value)
    if out is None and isinstance(value, str):
        for part in value.split(':')[1:]:
            out = _optional_float(part)
            if out is not None:
                break
    if out is None:
        return None
    return int(out) if out.is_integer() else out


def _global_sequential_receiver_number(receiver_id: object) -> int | float | None:
    value = _optional_float(receiver_id)
    if value is None and isinstance(receiver_id, str):
        parts = receiver_id.split(':')
        if len(parts) >= 2 and parts[0] == 'receiver':
            value = _optional_float(parts[1])
    if value is None:
        return None
    if value.is_integer():
        return int(value)
    return value


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return False


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != '':
            return value
    return None


def _np_scalar_to_python(value: object) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    'RefractionStaticPickMapError',
    'build_refraction_static_pick_map',
]
