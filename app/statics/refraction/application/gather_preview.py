"""Gather-preview data for refraction statics QC."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.statics.refraction.contracts.gather_preview import (
    RefractionStaticGatherPreviewRequest,
)
from app.statics.refraction.application.job_status import (
    is_ready_status_value,
    normalize_status_value,
)
from app.statics.refraction.artifacts.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.statics.refraction.artifacts import (
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
)
from app.services.trace_store_index_validation import validate_sorted_to_original
from app.statics.refraction.ports.runtime import RefractionRuntime

_MAX_SERVICE_TRACES = 500
_MAX_SERVICE_SAMPLES = 4000
_WINDOW_ENDPOINT = '/get_section_window_bin'


def _coerce_section_f32(arr: np.ndarray, scale: float | None) -> np.ndarray:
    out = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    if not out.flags.writeable:
        out = out.copy()
    if scale is not None:
        out *= float(scale)
    if not out.flags['C_CONTIGUOUS'] or out.dtype != np.float32:
        out = np.ascontiguousarray(out, dtype=np.float32)
    return out


class RefractionStaticGatherPreviewError(ValueError):
    """Raised when gather preview metadata cannot be assembled."""


class RefractionStaticGatherPreviewNotFound(LookupError):
    """Raised when a requested gather target is not present."""


@dataclass(frozen=True)
class _ResolvedWindow:
    key1: int | None
    x0: int
    x1: int
    y0: int
    y1: int
    x_indices: np.ndarray
    y_indices: np.ndarray
    sorted_trace_indices: np.ndarray
    trace_indices: np.ndarray
    key1_values: np.ndarray
    section_x_indices: np.ndarray
    step_x: int
    step_y: int
    requested_trace_count: int
    requested_sample_count: int
    trace_capped: bool
    sample_capped: bool
    regular_section_step_x: int | None


@dataclass(frozen=True)
class _OverlaySelection:
    rows: list[int | None]
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class _TraceShiftSelection:
    shift_field: str
    shifts_s: np.ndarray


def build_refraction_static_gather_preview(
    *,
    job_id: str,
    job: dict[str, object],
    req: RefractionStaticGatherPreviewRequest,
    runtime: RefractionRuntime,
) -> dict[str, Any]:
    """Build bounded preview data from existing TraceStore/artifact data."""
    _validate_job(job_id=job_id, job=job, req=req)
    artifacts_dir = _job_artifacts_dir(job, job_id)
    qc = _read_json_artifact(
        artifacts_dir / REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
    )
    sign_convention = _extract_sign_convention(qc)

    reader = runtime.trace_store.get_reader(req.file_id, req.key1_byte, req.key2_byte)
    dt_s = _reader_dt(reader, runtime=runtime, file_id=req.file_id)
    fit_arrays = _try_read_npz_artifact(
        artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    )
    key1 = _resolve_key1(req, reader=reader, fit_arrays=fit_arrays)
    section_shape = _section_shape(reader, key1) if key1 is not None else None
    resolved = _resolve_window(
        req,
        reader=reader,
        key1=key1,
        dt_s=dt_s,
        section_shape=section_shape,
        fit_arrays=fit_arrays,
    )
    shift_selection = _load_trace_shift_selection(
        artifacts_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        n_traces=int(reader.traces.shape[0]),
    )
    selected_shifts = _select_sorted_trace_shifts(
        shift_selection,
        sorted_trace_indices=resolved.sorted_trace_indices,
    )
    overlay_selection = _select_overlay_rows(
        fit_arrays,
        req=req,
        trace_indices=resolved.sorted_trace_indices,
    )

    raw_samples = _sample_trace_window(reader, resolved=resolved)
    raw_window_ref = _window_ref(
        status='ok',
        source='raw_tracestore',
        file_id=req.file_id,
        req=req,
        resolved=resolved,
    )
    corrected_window_ref, corrected_samples, corrected_samples_source = (
        _corrected_window_ref_and_samples(
            job=job,
            req=req,
            runtime=runtime,
            raw_reader=reader,
            raw_samples_trace_major=raw_samples,
            resolved=resolved,
            selected_shifts=selected_shifts,
            dt_s=dt_s,
        )
    )

    observed = _overlay_float_values(
        fit_arrays,
        overlay_selection.rows,
        field='observed_first_break_time_s',
        enabled='observed_first_break' in req.overlay_layers,
    )
    modeled = _overlay_float_values(
        fit_arrays,
        overlay_selection.rows,
        field='modeled_first_break_time_s',
        enabled='modeled_first_break' in req.overlay_layers,
    )
    residual = _overlay_float_values(
        fit_arrays,
        overlay_selection.rows,
        field='residual_time_s',
        fallback_field='residual_s',
        enabled=True,
    )
    offsets = _overlay_float_values(
        fit_arrays,
        overlay_selection.rows,
        field='offset_m',
        enabled=True,
    )
    reduced_observed, reduced_modeled = _reduced_time_values(
        req=req,
        offsets=offsets,
        observed=observed,
        modeled=modeled,
    )

    return {
        'job_id': job_id,
        'statics_kind': 'refraction',
        'sign_convention': sign_convention,
        'raw_window_ref': raw_window_ref,
        'corrected_window_ref': corrected_window_ref,
        'raw_samples': _samples_to_response(raw_samples),
        'corrected_samples': _samples_to_response(corrected_samples),
        'corrected_samples_source': corrected_samples_source,
        'dt_s': float(dt_s),
        'shape': [int(resolved.y_indices.size), int(resolved.x_indices.size)],
        'window': _window_metadata(req=req, resolved=resolved, dt_s=dt_s),
        'gather': _gather_metadata(req=req, overlay_selection=overlay_selection),
        'x_indices': [int(value) for value in resolved.x_indices.tolist()],
        'trace_indices': [int(value) for value in resolved.trace_indices.tolist()],
        'offset_m': offsets,
        'source_endpoint_key': _overlay_text_values(
            fit_arrays,
            overlay_selection.rows,
            field='source_endpoint_key',
        ),
        'receiver_endpoint_key': _overlay_text_values(
            fit_arrays,
            overlay_selection.rows,
            field='receiver_endpoint_key',
        ),
        'observed_pick_time_s': observed,
        'modeled_pick_time_s': modeled,
        'residual_s': residual,
        'final_trace_shift_s': _float_list(selected_shifts),
        'corrected_observed_pick_time_s': _add_shift(observed, selected_shifts),
        'corrected_modeled_pick_time_s': _add_shift(modeled, selected_shifts),
        'reduced_observed_time_s': reduced_observed,
        'reduced_modeled_time_s': reduced_modeled,
        'overlay_status': {
            'first_break_fit': overlay_selection.status,
            'reason': overlay_selection.reason,
            'artifact': REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
            'shift_field': shift_selection.shift_field,
            'reduction_velocity_m_s': req.reduction_velocity_m_s,
        },
        'artifacts': {
            'qc': REFRACTION_STATIC_QC_JSON_NAME,
            'first_break_fit_qc_npz': REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
            'refraction_static_solution_npz': REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        },
    }


def _validate_job(
    *,
    job_id: str,
    job: dict[str, object],
    req: RefractionStaticGatherPreviewRequest,
) -> None:
    if job.get('statics_kind') != 'refraction':
        raise RefractionStaticGatherPreviewError(
            f'Job {job_id} is not a refraction statics job'
        )
    if not is_ready_status_value(job.get('status')):
        raise RefractionStaticGatherPreviewError(
            f'Job {job_id} is not complete; current state is '
            f'{normalize_status_value(job.get("status"))}'
        )
    job_file_id = _job_text(job, 'file_id', job_id)
    if job_file_id != req.file_id:
        raise RefractionStaticGatherPreviewError(
            f'Request file_id {req.file_id!r} does not match job file_id {job_file_id!r}'
        )
    for field, requested in (
        ('key1_byte', req.key1_byte),
        ('key2_byte', req.key2_byte),
    ):
        job_value = _job_int(job, field, job_id)
        if job_value != requested:
            raise RefractionStaticGatherPreviewError(
                f'Request {field} {requested} does not match job {field} {job_value}'
            )


def _job_artifacts_dir(job: dict[str, object], job_id: str) -> Path:
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise RefractionStaticGatherPreviewError(
            f'Job {job_id} metadata is missing artifacts_dir'
        )
    path = Path(raw)
    if not path.is_dir():
        raise RefractionStaticGatherPreviewError(
            f'Job {job_id} artifacts directory is not available'
        )
    return path


def _job_text(job: dict[str, object], field: str, job_id: str) -> str:
    value = job.get(field)
    if not isinstance(value, str) or not value:
        raise RefractionStaticGatherPreviewError(
            f'Job {job_id} metadata is missing {field}'
        )
    return value


def _job_int(job: dict[str, object], field: str, job_id: str) -> int:
    value = job.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RefractionStaticGatherPreviewError(
            f'Job {job_id} metadata is missing {field}'
        )
    return int(value)


def _read_json_artifact(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RefractionStaticGatherPreviewError(
            f'Refraction gather preview requires artifact {artifact_name}'
        )
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RefractionStaticGatherPreviewError(
            f'Refraction artifact {artifact_name} is not valid JSON'
        ) from exc
    if not isinstance(payload, dict):
        raise RefractionStaticGatherPreviewError(
            f'Refraction artifact {artifact_name} must contain a JSON object'
        )
    return payload


def _extract_sign_convention(qc: dict[str, Any]) -> str:
    raw = qc.get('sign_convention')
    if isinstance(raw, str):
        sign_convention = raw
    elif isinstance(raw, dict) and isinstance(raw.get('trace_shift_s'), str):
        sign_convention = raw['trace_shift_s']
    else:
        raise RefractionStaticGatherPreviewError(
            'Refraction QC artifact is missing sign_convention'
        )
    if sign_convention != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
        raise RefractionStaticGatherPreviewError(
            'Refraction QC artifact has unsupported sign_convention: '
            f'{sign_convention!r}'
        )
    return sign_convention


def _try_read_npz_artifact(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]).copy() for name in data.files}
    except Exception as exc:  # noqa: BLE001
        raise RefractionStaticGatherPreviewError(
            f'Refraction artifact {path.name} could not be read'
        ) from exc


def _read_npz_artifact(path: Path, artifact_name: str) -> dict[str, np.ndarray]:
    arrays = _try_read_npz_artifact(path)
    if arrays is None:
        raise RefractionStaticGatherPreviewError(
            f'Refraction gather preview requires artifact {artifact_name}'
        )
    if not arrays:
        raise RefractionStaticGatherPreviewError(
            f'Refraction artifact {artifact_name} contains no arrays'
        )
    return arrays


def _section_shape(reader: Any, key1: int) -> tuple[int, int]:
    try:
        section = reader.get_section(int(key1)).arr
    except ValueError as exc:
        raise RefractionStaticGatherPreviewNotFound(str(exc)) from exc
    if section.ndim != 2:
        raise RefractionStaticGatherPreviewError('TraceStore section data must be 2D')
    n_traces, n_samples = int(section.shape[0]), int(section.shape[1])
    if n_traces <= 0 or n_samples <= 0:
        raise RefractionStaticGatherPreviewError('TraceStore section is empty')
    return n_traces, n_samples


def _reader_dt(
    reader: Any,
    *,
    runtime: RefractionRuntime,
    file_id: str,
) -> float:
    meta = getattr(reader, 'meta', None)
    raw_dt = meta.get('dt') if isinstance(meta, dict) else None
    if isinstance(raw_dt, int | float):
        dt = float(raw_dt)
        if math.isfinite(dt) and dt > 0.0:
            return dt
    dt = float(runtime.trace_store.get_dt(file_id))
    if not math.isfinite(dt) or dt <= 0.0:
        raise RefractionStaticGatherPreviewError('TraceStore sample interval is invalid')
    return dt


def _resolve_key1(
    req: RefractionStaticGatherPreviewRequest,
    *,
    reader: Any,
    fit_arrays: dict[str, np.ndarray] | None,
) -> int | None:
    if req.key1 is not None:
        return int(req.key1)
    if req.gather_axis == 'section':
        raise RefractionStaticGatherPreviewError(
            'key1 is required for section gather previews'
        )
    if fit_arrays is None:
        raise RefractionStaticGatherPreviewError(
            f'{REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME} is required to resolve '
            f'{req.gather_axis} gather targets'
        )
    matching_traces = _endpoint_trace_indices(
        fit_arrays,
        gather_axis=req.gather_axis,
        endpoint_key=req.endpoint_key,
    )
    key1_header = np.asarray(reader.get_header(req.key1_byte))
    if key1_header.ndim != 1 or key1_header.shape[0] != int(reader.traces.shape[0]):
        raise RefractionStaticGatherPreviewError(
            'TraceStore key1 header does not match trace count'
        )
    if np.any((matching_traces < 0) | (matching_traces >= key1_header.shape[0])):
        raise RefractionStaticGatherPreviewError(
            'first-break QC trace_index_sorted contains out-of-range trace indices'
        )
    key1_values = np.unique(key1_header[matching_traces].astype(np.int64, copy=False))
    if key1_values.size == 1:
        return int(key1_values[0])
    return None


def _resolve_window(
    req: RefractionStaticGatherPreviewRequest,
    *,
    reader: Any,
    key1: int | None,
    dt_s: float,
    section_shape: tuple[int, int] | None,
    fit_arrays: dict[str, np.ndarray] | None,
) -> _ResolvedWindow:
    if key1 is not None:
        if section_shape is None:
            raise RefractionStaticGatherPreviewError('Section shape is required')
        n_section_traces, n_samples = section_shape
        x0 = 0 if req.x0 is None else int(req.x0)
        x1 = n_section_traces - 1 if req.x1 is None else int(req.x1)
        if x0 < 0 or x1 < x0 or x1 >= n_section_traces:
            raise RefractionStaticGatherPreviewError('Trace range out of bounds')
    else:
        if req.gather_axis not in {'source', 'receiver'}:
            raise RefractionStaticGatherPreviewError(
                'key1 is required for section gather previews'
            )
        x0 = 0
        x1 = 0
        n_samples = int(reader.traces.shape[1])
    y0, y1 = _resolve_sample_range(req, dt_s=dt_s, n_samples=n_samples)
    if y1 >= n_samples:
        raise RefractionStaticGatherPreviewError('Sample range out of bounds')

    effective_max_traces = min(int(req.max_traces), _MAX_SERVICE_TRACES)
    effective_max_samples = min(int(req.max_samples), _MAX_SERVICE_SAMPLES)
    key1_header = np.asarray(reader.get_header(req.key1_byte))
    if key1_header.ndim != 1 or key1_header.shape[0] != int(reader.traces.shape[0]):
        raise RefractionStaticGatherPreviewError(
            'TraceStore key1 header does not match trace count'
        )

    section_sorted_trace_indices: np.ndarray | None = None
    if key1 is None:
        if fit_arrays is None:
            raise RefractionStaticGatherPreviewError(
                f'{REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME} is required to validate '
                f'{req.gather_axis} gather targets'
            )
        endpoint_sorted_trace_indices = _endpoint_trace_indices(
            fit_arrays,
            gather_axis=req.gather_axis,
            endpoint_key=req.endpoint_key,
        )
        _validate_sorted_trace_indices(endpoint_sorted_trace_indices, reader=reader)
        candidate_x_indices = np.arange(
            endpoint_sorted_trace_indices.size,
            dtype=np.int64,
        )
        x1 = int(max(candidate_x_indices.size - 1, 0))
    else:
        section_sorted_trace_indices = reader.get_trace_seq_for_value(
            key1,
            align_to='display',
        )
        if section_shape is None or section_sorted_trace_indices.shape[0] != section_shape[0]:
            raise RefractionStaticGatherPreviewError(
                'TraceStore section index does not match section trace count'
            )
        candidate_x_indices = np.arange(x0, x1 + 1, dtype=np.int64)
        if req.gather_axis in {'source', 'receiver'}:
            if fit_arrays is None:
                raise RefractionStaticGatherPreviewError(
                    f'{REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME} is required to validate '
                    f'{req.gather_axis} gather targets'
                )
            endpoint_traces = set(
                int(value)
                for value in _endpoint_trace_indices(
                    fit_arrays,
                    gather_axis=req.gather_axis,
                    endpoint_key=req.endpoint_key,
                ).tolist()
            )
            candidate_x_indices = np.asarray(
                [
                    int(pos)
                    for pos in candidate_x_indices.tolist()
                    if int(section_sorted_trace_indices[int(pos)]) in endpoint_traces
                ],
                dtype=np.int64,
            )
            if candidate_x_indices.size == 0:
                raise RefractionStaticGatherPreviewNotFound(
                    f'Refraction {req.gather_axis} gather was not found in the '
                    f'selected section/window: {req.endpoint_key}'
                )

    if key1 is None:
        section_x_by_sorted = _section_x_index_by_sorted_trace(reader)
        candidate_sorted_trace_indices = endpoint_sorted_trace_indices
        candidate_x_indices = np.asarray(
            candidate_x_indices,
            dtype=np.int64,
        )
    else:
        candidate_sorted_trace_indices = section_sorted_trace_indices

    x_indices, step_x, requested_trace_count, trace_capped = _bounded_positions(
        candidate_x_indices,
        step=int(req.step_x),
        max_count=effective_max_traces,
    )
    step_y, requested_sample_count, sample_capped = _bounded_step(
        start=y0,
        stop=y1,
        step=req.step_y,
        max_count=effective_max_samples,
    )
    y_indices = np.arange(y0, y1 + 1, step_y, dtype=np.int64)
    if x_indices.size == 0 or y_indices.size == 0:
        raise RefractionStaticGatherPreviewError('Requested preview window is empty')
    sorted_trace_indices = np.asarray(candidate_sorted_trace_indices[x_indices], dtype=np.int64)
    sorted_to_original = _reader_sorted_to_original(reader)
    if np.any(
        (sorted_trace_indices < 0)
        | (sorted_trace_indices >= sorted_to_original.shape[0])
    ):
        raise RefractionStaticGatherPreviewError(
            'TraceStore section contains out-of-range sorted trace indices'
        )
    trace_indices = sorted_to_original[sorted_trace_indices]
    selected_key1_values = key1_header[sorted_trace_indices].astype(np.int64, copy=False)
    if key1 is None:
        section_x_indices = section_x_by_sorted[sorted_trace_indices]
    else:
        section_x_indices = x_indices
    return _ResolvedWindow(
        key1=None if key1 is None else int(key1),
        x0=int(x0),
        x1=int(x1),
        y0=int(y0),
        y1=int(y1),
        x_indices=np.ascontiguousarray(x_indices, dtype=np.int64),
        y_indices=np.ascontiguousarray(y_indices, dtype=np.int64),
        sorted_trace_indices=np.ascontiguousarray(
            sorted_trace_indices,
            dtype=np.int64,
        ),
        trace_indices=np.ascontiguousarray(trace_indices, dtype=np.int64),
        key1_values=np.ascontiguousarray(selected_key1_values, dtype=np.int64),
        section_x_indices=np.ascontiguousarray(section_x_indices, dtype=np.int64),
        step_x=step_x,
        step_y=step_y,
        requested_trace_count=requested_trace_count,
        requested_sample_count=requested_sample_count,
        trace_capped=trace_capped,
        sample_capped=sample_capped,
        regular_section_step_x=_regular_positive_step(x_indices),
    )


def _resolve_sample_range(
    req: RefractionStaticGatherPreviewRequest,
    *,
    dt_s: float,
    n_samples: int,
) -> tuple[int, int]:
    if req.y0 is not None and req.y1 is not None:
        return int(req.y0), int(req.y1)
    if req.time_start_s is None or req.time_end_s is None:
        raise RefractionStaticGatherPreviewError(
            'y0/y1 or time_start_s/time_end_s is required'
        )
    y0 = int(math.floor(float(req.time_start_s) / float(dt_s)))
    y1 = int(math.ceil(float(req.time_end_s) / float(dt_s)))
    if y0 < 0 or y1 < y0:
        raise RefractionStaticGatherPreviewError('Time range is invalid')
    if y0 >= int(n_samples) or y1 >= int(n_samples):
        raise RefractionStaticGatherPreviewError('Time range out of bounds')
    return y0, y1


def _bounded_positions(
    positions: np.ndarray,
    *,
    step: int,
    max_count: int,
) -> tuple[np.ndarray, int, int, bool]:
    arr = np.asarray(positions, dtype=np.int64)
    if arr.ndim != 1:
        raise RefractionStaticGatherPreviewError('Trace positions must be 1D')
    if arr.size == 0:
        return arr, int(step), 0, False
    requested = arr[:: int(step)]
    requested_count = int(requested.size)
    if requested_count <= int(max_count):
        return (
            np.ascontiguousarray(requested, dtype=np.int64),
            int(step),
            requested_count,
            False,
        )
    stride_multiplier = int(math.ceil(requested_count / int(max_count)))
    effective_step = int(step) * stride_multiplier
    selected = arr[::effective_step]
    return (
        np.ascontiguousarray(selected, dtype=np.int64),
        effective_step,
        requested_count,
        True,
    )


def _bounded_step(
    *,
    start: int,
    stop: int,
    step: int,
    max_count: int,
) -> tuple[int, int, bool]:
    requested_count = ((int(stop) - int(start)) // int(step)) + 1
    if requested_count <= int(max_count):
        return int(step), int(requested_count), False
    stride_multiplier = int(math.ceil(requested_count / int(max_count)))
    return int(step) * stride_multiplier, int(requested_count), True


def _regular_positive_step(indices: np.ndarray) -> int | None:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        return None
    if arr.size == 1:
        return 1
    diffs = np.diff(arr)
    first = int(diffs[0])
    if first <= 0 or not np.all(diffs == first):
        return None
    return first


def _load_trace_shift_selection(path: Path, *, n_traces: int) -> _TraceShiftSelection:
    arrays = _read_npz_artifact(path, REFRACTION_STATIC_SOLUTION_NPZ_NAME)
    for field in ('final_trace_shift_s_sorted', 'refraction_trace_shift_s_sorted'):
        if field not in arrays:
            continue
        shifts = np.asarray(arrays[field], dtype=np.float64)
        if shifts.shape != (n_traces,):
            raise RefractionStaticGatherPreviewError(
                f'{REFRACTION_STATIC_SOLUTION_NPZ_NAME} {field} shape does not '
                'match TraceStore trace count'
            )
        return _TraceShiftSelection(
            shift_field=field,
            shifts_s=np.ascontiguousarray(shifts, dtype=np.float64),
        )
    raise RefractionStaticGatherPreviewError(
        f'{REFRACTION_STATIC_SOLUTION_NPZ_NAME} is missing trace shift arrays'
    )


def _reader_sorted_to_original(reader: Any) -> np.ndarray:
    try:
        values = reader.get_sorted_to_original()
    except ValueError as exc:
        raise RefractionStaticGatherPreviewError(str(exc)) from exc
    try:
        return validate_sorted_to_original(
            np.asarray(values),
            expected_n_traces=int(reader.traces.shape[0]),
            role='TraceStore',
        )
    except ValueError as exc:
        raise RefractionStaticGatherPreviewError(str(exc)) from exc


def _select_sorted_trace_shifts(
    selection: _TraceShiftSelection,
    *,
    sorted_trace_indices: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(sorted_trace_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise RefractionStaticGatherPreviewError(
            'Selected sorted trace indices must be 1D'
        )
    shifts = np.asarray(selection.shifts_s, dtype=np.float64)
    if np.any((indices < 0) | (indices >= shifts.shape[0])):
        raise RefractionStaticGatherPreviewError(
            f'{selection.shift_field} selection contains out-of-range sorted trace indices'
        )
    return np.ascontiguousarray(shifts[indices], dtype=np.float64)


def _validate_sorted_trace_indices(
    indices: np.ndarray,
    *,
    reader: Any,
) -> None:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim != 1:
        raise RefractionStaticGatherPreviewError(
            'first-break QC trace_index_sorted must be a 1D array'
        )
    if np.any((arr < 0) | (arr >= int(reader.traces.shape[0]))):
        raise RefractionStaticGatherPreviewError(
            'first-break QC trace_index_sorted contains out-of-range trace indices'
        )


def _section_x_index_by_sorted_trace(reader: Any) -> np.ndarray:
    n_traces = int(reader.traces.shape[0])
    out = np.full(n_traces, -1, dtype=np.int64)
    for key1_value in reader.get_key1_values().tolist():
        section_indices = np.asarray(
            reader.get_trace_seq_for_value(int(key1_value), align_to='display'),
            dtype=np.int64,
        )
        if np.any((section_indices < 0) | (section_indices >= n_traces)):
            raise RefractionStaticGatherPreviewError(
                'TraceStore section contains out-of-range sorted trace indices'
            )
        out[section_indices] = np.arange(section_indices.size, dtype=np.int64)
    if np.any(out < 0):
        raise RefractionStaticGatherPreviewError(
            'TraceStore section index does not cover all traces'
        )
    return np.ascontiguousarray(out, dtype=np.int64)


def _select_overlay_rows(
    arrays: dict[str, np.ndarray] | None,
    *,
    req: RefractionStaticGatherPreviewRequest,
    trace_indices: np.ndarray,
) -> _OverlaySelection:
    if arrays is None:
        if req.gather_axis in {'source', 'receiver'}:
            raise RefractionStaticGatherPreviewError(
                f'{REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME} is required to validate '
                f'{req.gather_axis} gather targets'
            )
        return _OverlaySelection(
            rows=[None] * int(trace_indices.size),
            status='unavailable',
            reason='artifact_missing',
        )

    trace_index = _trace_index_array(arrays)
    endpoint_values: np.ndarray | None = None
    if req.gather_axis in {'source', 'receiver'}:
        endpoint_column = f'{req.gather_axis}_endpoint_key'
        endpoint_values = _string_1d(arrays, endpoint_column)
        if endpoint_values.shape != trace_index.shape:
            raise RefractionStaticGatherPreviewError(
                f'{endpoint_column} shape does not match trace_index_sorted'
            )

    wanted = {int(value) for value in trace_indices.tolist()}
    row_by_trace: dict[int, int] = {}
    for row_index, trace in enumerate(trace_index.tolist()):
        trace_int = int(trace)
        if trace_int not in wanted or trace_int in row_by_trace:
            continue
        if endpoint_values is not None and endpoint_values[row_index] != req.endpoint_key:
            continue
        row_by_trace[trace_int] = int(row_index)

    if req.gather_axis in {'source', 'receiver'}:
        if not row_by_trace:
            raise RefractionStaticGatherPreviewNotFound(
                f'Refraction {req.gather_axis} gather was not found in the '
                f'requested window: {req.endpoint_key}'
            )
        missing = [
            int(trace)
            for trace in trace_indices.tolist()
            if int(trace) not in row_by_trace
        ]
        if missing:
            raise RefractionStaticGatherPreviewError(
                'Requested section window includes traces outside the selected '
                f'{req.gather_axis} gather target'
            )

    rows = [row_by_trace.get(int(trace)) for trace in trace_indices.tolist()]
    if any(row is None for row in rows):
        return _OverlaySelection(
            rows=rows,
            status='partial',
            reason='selected_trace_missing',
        )
    return _OverlaySelection(rows=rows, status='available')


def _trace_index_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if 'trace_index_sorted' not in arrays:
        raise RefractionStaticGatherPreviewError(
            'first-break QC artifact is missing trace_index_sorted'
        )
    arr = np.asarray(arrays['trace_index_sorted'])
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise RefractionStaticGatherPreviewError(
            'first-break QC trace_index_sorted must be a 1D integer array'
        )
    return np.asarray(arr, dtype=np.int64)


def _endpoint_trace_indices(
    arrays: dict[str, np.ndarray],
    *,
    gather_axis: str,
    endpoint_key: str | None,
) -> np.ndarray:
    if endpoint_key is None:
        raise RefractionStaticGatherPreviewError(
            'endpoint_key is required for source/receiver gather targets'
        )
    trace_index = _trace_index_array(arrays)
    endpoint_column = f'{gather_axis}_endpoint_key'
    endpoint_values = _string_1d(arrays, endpoint_column)
    if endpoint_values.shape != trace_index.shape:
        raise RefractionStaticGatherPreviewError(
            f'{endpoint_column} shape does not match trace_index_sorted'
        )
    matching = trace_index[endpoint_values == str(endpoint_key)]
    if matching.size == 0:
        raise RefractionStaticGatherPreviewNotFound(
            f'Refraction {gather_axis} gather was not found: {endpoint_key}'
        )
    return np.ascontiguousarray(matching.astype(np.int64, copy=False))


def _sample_trace_window(
    reader: Any,
    *,
    resolved: _ResolvedWindow,
) -> np.ndarray:
    base = reader.traces
    if base.ndim != 2:
        raise RefractionStaticGatherPreviewError('TraceStore data must be 2D')
    sub = base[np.ix_(resolved.sorted_trace_indices, resolved.y_indices)]
    samples = _coerce_section_f32(sub, reader.scale)
    if not np.all(np.isfinite(samples)):
        raise RefractionStaticGatherPreviewError(
            'TraceStore preview samples contain non-finite values'
        )
    return np.ascontiguousarray(samples, dtype=np.float32)


def _shifted_preview_samples(
    reader: Any,
    *,
    resolved: _ResolvedWindow,
    selected_shifts: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    shifts = np.asarray(selected_shifts, dtype=np.float64)
    if shifts.shape != (int(resolved.x_indices.size),):
        raise RefractionStaticGatherPreviewError(
            'Trace shift count does not match preview trace count'
        )
    if not np.all(np.isfinite(shifts)):
        raise RefractionStaticGatherPreviewError(
            'Refraction trace shifts contain non-finite values'
        )
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise RefractionStaticGatherPreviewError('TraceStore sample interval is invalid')

    base = reader.traces
    if base.ndim != 2:
        raise RefractionStaticGatherPreviewError('TraceStore data must be 2D')
    n_samples = int(base.shape[1])
    display_samples = resolved.y_indices.astype(np.float64, copy=False)
    shift_samples = shifts / dt
    source_min = float(np.min(display_samples[0] - shift_samples))
    source_max = float(np.max(display_samples[-1] - shift_samples))
    raw_y0 = max(0, int(math.floor(source_min)) - 1)
    raw_y1 = min(n_samples - 1, int(math.ceil(source_max)) + 1)
    raw_y_indices = np.arange(raw_y0, raw_y1 + 1, dtype=np.int64)
    expanded = base[np.ix_(resolved.sorted_trace_indices, raw_y_indices)]
    expanded = _coerce_section_f32(expanded, reader.scale)
    if not np.all(np.isfinite(expanded)):
        raise RefractionStaticGatherPreviewError(
            'TraceStore preview samples contain non-finite values'
        )

    out = np.zeros((int(resolved.x_indices.size), int(resolved.y_indices.size)), dtype=np.float32)
    max_source = float(expanded.shape[1] - 1)
    for trace_idx, shift in enumerate(shift_samples.tolist()):
        source = display_samples - float(shift) - float(raw_y0)
        valid = (source >= 0.0) & (source <= max_source)
        if not np.any(valid):
            continue
        source_valid = source[valid]
        lo = np.floor(source_valid).astype(np.int64)
        hi = np.minimum(lo + 1, expanded.shape[1] - 1)
        frac = (source_valid - lo).astype(np.float32, copy=False)
        row = expanded[trace_idx]
        out[trace_idx, valid] = row[lo] * (1.0 - frac) + row[hi] * frac
    return np.ascontiguousarray(out, dtype=np.float32)


def _samples_to_response(samples: np.ndarray) -> list[list[float]]:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 2:
        raise RefractionStaticGatherPreviewError('Preview samples must be 2D')
    if not np.all(np.isfinite(arr)):
        raise RefractionStaticGatherPreviewError(
            'Preview samples contain non-finite values'
        )
    transposed = np.ascontiguousarray(arr.T, dtype=np.float32)
    return [
        [float(value) for value in row.tolist()]
        for row in transposed
    ]


def _window_ref(
    *,
    status: str,
    source: str,
    file_id: str,
    req: RefractionStaticGatherPreviewRequest,
    resolved: _ResolvedWindow,
    message: str | None = None,
) -> dict[str, Any]:
    if resolved.key1 is None:
        ref = {
            'status': 'not_available',
            'source': source,
            'message': (
                'selected gather traces span multiple key1 sections and are not '
                'representable as one section-window request'
            ),
        }
        if message is not None:
            ref['message'] = message
        return ref
    if resolved.regular_section_step_x is None:
        ref = {
            'status': 'not_available',
            'source': source,
            'message': (
                'selected gather traces are not representable as one contiguous '
                'section-window request'
            ),
        }
        if message is not None:
            ref['message'] = message
        return ref
    query: dict[str, Any] = {
        'file_id': file_id,
        'key1': int(resolved.key1),
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'x0': int(resolved.x_indices[0]),
        'x1': int(resolved.x_indices[-1]),
        'y0': int(resolved.y0),
        'y1': int(resolved.y1),
        'step_x': int(resolved.regular_section_step_x),
        'step_y': int(resolved.step_y),
        'transpose': True,
        'scaling': req.scaling,
    }
    ref: dict[str, Any] = {
        'status': status,
        'source': source,
        'endpoint': _WINDOW_ENDPOINT,
        'query': query,
    }
    if message is not None:
        ref['message'] = message
    return ref


def _corrected_window_ref_and_samples(
    *,
    job: dict[str, object],
    req: RefractionStaticGatherPreviewRequest,
    runtime: RefractionRuntime,
    raw_reader: Any,
    raw_samples_trace_major: np.ndarray,
    resolved: _ResolvedWindow,
    selected_shifts: np.ndarray,
    dt_s: float,
) -> tuple[dict[str, Any], np.ndarray, str]:
    def shifted_fallback(
        *,
        status: str,
        message: str,
    ) -> tuple[dict[str, Any], np.ndarray, str]:
        shifted = _shifted_preview_samples(
            raw_reader,
            resolved=resolved,
            selected_shifts=selected_shifts,
            dt_s=dt_s,
        )
        return {
            'status': status,
            'source': 'corrected_tracestore',
            'message': message,
        }, shifted, 'raw_tracestore_shifted_on_the_fly'

    corrected_file_id = job.get('corrected_file_id')
    if not isinstance(corrected_file_id, str) or not corrected_file_id:
        return shifted_fallback(
            status='not_registered',
            message='corrected TraceStore was not registered for this refraction job',
        )

    try:
        corrected_reader = runtime.trace_store.get_reader(
            corrected_file_id,
            req.key1_byte,
            req.key2_byte,
        )
        corrected_shape = corrected_reader.traces.shape
    except Exception:  # noqa: BLE001
        return shifted_fallback(
            status='unavailable',
            message=(
                f'Registered corrected TraceStore {corrected_file_id} could not be '
                f'read; returning on-the-fly shifted preview samples'
            ),
        )
    if corrected_shape != raw_reader.traces.shape:
        return shifted_fallback(
            status='shape_mismatch',
            message=(
                f'Registered corrected TraceStore {corrected_file_id} shape does not '
                'match the raw TraceStore; returning on-the-fly shifted '
                'preview samples'
            ),
        )
    corrected_samples = _sample_trace_window(corrected_reader, resolved=resolved)
    if corrected_samples.shape != raw_samples_trace_major.shape:
        return shifted_fallback(
            status='shape_mismatch',
            message=(
                'Registered corrected TraceStore preview sample shape does not match '
                'raw; returning on-the-fly shifted preview samples'
            ),
        )
    return _window_ref(
        status='ok',
        source='corrected_tracestore',
        file_id=corrected_file_id,
        req=req,
        resolved=resolved,
    ), corrected_samples, 'corrected_tracestore'


def _window_metadata(
    *,
    req: RefractionStaticGatherPreviewRequest,
    resolved: _ResolvedWindow,
    dt_s: float,
) -> dict[str, Any]:
    sample_start = int(resolved.y_indices[0])
    sample_stop = int(resolved.y_indices[-1])
    return {
        'key1': None if resolved.key1 is None else int(resolved.key1),
        'key1_values': [int(value) for value in resolved.key1_values.tolist()],
        'section_x_indices': [
            int(value) for value in resolved.section_x_indices.tolist()
        ],
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'x0': int(resolved.x0),
        'x1': int(resolved.x1),
        'y0': int(resolved.y0),
        'y1': int(resolved.y1),
        'requested_step_x': int(req.step_x),
        'requested_step_y': int(req.step_y),
        'effective_step_x': int(resolved.step_x),
        'effective_step_y': int(resolved.step_y),
        'regular_section_step_x': resolved.regular_section_step_x,
        'requested_trace_count': int(resolved.requested_trace_count),
        'returned_trace_count': int(resolved.x_indices.size),
        'requested_sample_count': int(resolved.requested_sample_count),
        'returned_sample_count': int(resolved.y_indices.size),
        'trace_capped': bool(resolved.trace_capped),
        'sample_capped': bool(resolved.sample_capped),
        'service_max_traces': _MAX_SERVICE_TRACES,
        'service_max_samples': _MAX_SERVICE_SAMPLES,
        'transpose': True,
        'scaling': req.scaling,
        'sample_start': sample_start,
        'sample_stop': sample_stop,
        'time_start_s': sample_start * float(dt_s),
        'time_end_s': sample_stop * float(dt_s),
    }


def _gather_metadata(
    *,
    req: RefractionStaticGatherPreviewRequest,
    overlay_selection: _OverlaySelection,
) -> dict[str, Any]:
    return {
        'axis': req.gather_axis,
        'endpoint_key': req.endpoint_key,
        'overlay_status': overlay_selection.status,
        'overlay_reason': overlay_selection.reason,
    }


def _overlay_float_values(
    arrays: dict[str, np.ndarray] | None,
    rows: list[int | None],
    *,
    field: str,
    enabled: bool,
    fallback_field: str | None = None,
) -> list[float | None]:
    if not enabled or arrays is None:
        return [None] * len(rows)
    selected_field = field if field in arrays else fallback_field
    if selected_field is None or selected_field not in arrays:
        return [None] * len(rows)
    values = np.asarray(arrays[selected_field], dtype=np.float64)
    return [
        _json_float(values[row]) if row is not None else None
        for row in rows
    ]


def _overlay_text_values(
    arrays: dict[str, np.ndarray] | None,
    rows: list[int | None],
    *,
    field: str,
) -> list[str | None]:
    if arrays is None or field not in arrays:
        return [None] * len(rows)
    values = np.asarray(arrays[field], dtype=str)
    return [str(values[row]) if row is not None else None for row in rows]


def _reduced_time_values(
    *,
    req: RefractionStaticGatherPreviewRequest,
    offsets: list[float | None],
    observed: list[float | None],
    modeled: list[float | None],
) -> tuple[list[float | None] | None, list[float | None] | None]:
    if (
        req.reduction_velocity_m_s is None
        or 'reduced_time' not in req.overlay_layers
    ):
        return None, None
    velocity = float(req.reduction_velocity_m_s)
    reduced_observed: list[float | None] = []
    reduced_modeled: list[float | None] = []
    for offset, observed_time, modeled_time in zip(
        offsets,
        observed,
        modeled,
        strict=True,
    ):
        moveout_s = None if offset is None else offset / velocity
        reduced_observed.append(
            None
            if observed_time is None or moveout_s is None
            else observed_time - moveout_s
        )
        reduced_modeled.append(
            None
            if modeled_time is None or moveout_s is None
            else modeled_time - moveout_s
        )
    return reduced_observed, reduced_modeled


def _add_shift(
    values: list[float | None],
    shifts_s: np.ndarray,
) -> list[float | None]:
    out: list[float | None] = []
    for value, shift in zip(values, shifts_s.tolist(), strict=True):
        out.append(None if value is None else value + float(shift))
    return out


def _string_1d(arrays: dict[str, np.ndarray], field: str) -> np.ndarray:
    if field not in arrays:
        raise RefractionStaticGatherPreviewError(
            f'first-break QC artifact is missing {field}'
        )
    arr = np.asarray(arrays[field])
    if arr.ndim != 1:
        raise RefractionStaticGatherPreviewError(f'{field} must be a 1D array')
    return np.asarray(arr, dtype=str)


def _float_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        out.append(_json_float(value))
    return out


def _json_float(value: object) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


__all__ = [
    'RefractionStaticGatherPreviewError',
    'RefractionStaticGatherPreviewNotFound',
    'build_refraction_static_gather_preview',
]
