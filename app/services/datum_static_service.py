"""Datum static correction background job service."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any
from uuid import uuid4

import numpy as np

from app.api.schemas import DatumStaticApplyRequest
from app.core.paths import get_trace_store_dir
from app.core.state import AppState
from app.services.corrected_trace_store import (
    TimeShiftedTraceStoreResult,
    build_time_shifted_trace_store,
)
from app.services.datum_static_geometry import (
    DatumStaticGeometryConfig,
    load_datum_static_geometry,
)
from app.services.datum_static_math import compute_datum_static_shifts
from app.services.datum_static_validation import (
    ExistingStaticHeaderConfig,
    validate_existing_static_headers,
    validate_trace_shift_limits,
)
from app.services.job_runner import (
    JobCancelledError,
    JobCompletion,
    JobFailure,
    ensure_job_not_cancelled,
    run_job_with_lifecycle,
)
from app.services.pipeline_artifacts import get_job_dir
from app.services.reader import get_reader
from app.services.static_artifacts import (
    QC_JSON_NAME,
    SOLUTION_NPZ_NAME,
    STATICS_CSV_NAME,
    write_datum_static_artifacts,
)
from app.services.trace_store_registration import (
    register_trace_store,
    trace_store_cache_key,
)

_SAFE_STORE_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')
_CORRECTED_FILE_NAME = 'corrected_file.json'


@dataclass
class _DatumStaticLifecycle:
    created_ts: float
    job_dir: Path | None = None
    corrected_store_path: Path | None = None
    corrected_store_built: bool = False
    corrected_file_id: str | None = None
    key1_byte: int | None = None
    key2_byte: int | None = None


def _resolve_created_ts(state: AppState, job_id: str) -> float:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            return time.time()
        created_ts_obj = job.get('created_ts')
    return (
        float(created_ts_obj)
        if isinstance(created_ts_obj, (int, float))
        else time.time()
    )


def _resolve_job_dir(state: AppState, job_id: str) -> Path:
    with state.lock:
        job = state.jobs.get(job_id)
        artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
    if isinstance(artifacts_dir, str) and artifacts_dir:
        return Path(artifacts_dir)
    return get_job_dir(job_id)


def _set_job_progress_message(
    state: AppState,
    job_id: str,
    *,
    progress: float,
    message: str,
) -> None:
    with state.lock:
        if state.jobs.get(job_id) is None:
            return
        state.jobs.set_progress(job_id, progress)
        state.jobs.set_message(job_id, message)


def _is_cancel_requested(state: AppState, job_id: str) -> bool:
    with state.lock:
        return state.jobs.is_cancel_requested(job_id)


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


def _write_job_meta(
    *,
    job_id: str,
    job_dir: Path,
    req: DatumStaticApplyRequest,
) -> None:
    _write_json_atomic(
        job_dir / 'job_meta.json',
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'datum',
            'source_file_id': req.file_id,
            'key1_byte': req.key1_byte,
            'key2_byte': req.key2_byte,
            'request': req.model_dump(mode='json'),
        },
    )


def _safe_store_name_component(name: str) -> str:
    safe = _SAFE_STORE_NAME_RE.sub('_', name)
    if safe in {'', '.', '..'}:
        raise ValueError('source TraceStore name cannot be made filesystem-safe')
    return safe


def _corrected_store_path(*, source_store_path: Path, job_id: str) -> Path:
    trace_store_dir = get_trace_store_dir()
    trace_store_dir.mkdir(parents=True, exist_ok=True)
    source_name = _safe_store_name_component(source_store_path.name)
    job_prefix = _safe_store_name_component(str(job_id)[:8])
    store_name = f'{source_name}.statics.datum.{job_prefix}'
    path = trace_store_dir / store_name
    if path.exists() or path.is_symlink():
        raise ValueError(f'corrected output path already exists: {path}')
    return path


def _source_store_path(reader: object) -> Path:
    store_dir = getattr(reader, 'store_dir', None)
    try:
        path = Path(store_dir)
    except TypeError as exc:
        raise ValueError('source TraceStore path is not available') from exc
    if not path.is_dir():
        raise ValueError(f'source TraceStore does not exist: {path}')
    return path


def _resolve_dt(state: AppState, file_id: str) -> float:
    dt = float(state.file_registry.get_dt(file_id))
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError('dt must be finite and greater than 0')
    return dt


def _header_bytes_to_materialize(req: DatumStaticApplyRequest) -> tuple[int, ...]:
    values = [
        req.key1_byte,
        req.key2_byte,
        req.geometry.source_elevation_byte,
        req.geometry.receiver_elevation_byte,
        req.geometry.elevation_scalar_byte,
    ]
    if req.geometry.source_depth_byte is not None:
        values.append(req.geometry.source_depth_byte)
    for byte in (
        req.existing_statics.source_static_byte,
        req.existing_statics.receiver_static_byte,
        req.existing_statics.total_static_byte,
    ):
        if byte is not None:
            values.append(byte)
    return tuple(dict.fromkeys(int(value) for value in values))


def _reader_original_segy_path(reader: object) -> str | None:
    meta = getattr(reader, 'meta', None)
    if not isinstance(meta, dict):
        return None
    original = meta.get('original_segy_path')
    return original if isinstance(original, str) else None


def _load_sorted_header(reader: object, byte: int, *, name: str) -> np.ndarray:
    get_header = getattr(reader, 'get_header')
    values = np.asarray(get_header(int(byte)))
    n_traces = int(getattr(reader, 'traces').shape[0])
    if values.shape != (n_traces,):
        raise ValueError(
            f'{name} header byte {byte} shape mismatch: '
            f'expected {(n_traces,)}, got {values.shape}'
        )
    return values


def _build_corrected_file_payload(
    *,
    corrected_file_id: str,
    build_result: TimeShiftedTraceStoreResult,
    source_store_path: Path,
    req: DatumStaticApplyRequest,
    job_id: str,
) -> dict[str, object]:
    return {
        'file_id': corrected_file_id,
        'store_path': str(build_result.store_path),
        'store_name': build_result.store_path.name,
        'derived_from_file_id': req.file_id,
        'derived_from_store_path': str(source_store_path),
        'derived_by': 'datum_static_correction',
        'job_id': job_id,
        'key1_byte': req.key1_byte,
        'key2_byte': req.key2_byte,
        'dt': build_result.dt,
        'n_traces': build_result.n_traces,
        'n_samples': build_result.n_samples,
        'solution_artifact': SOLUTION_NPZ_NAME,
        'qc_artifact': QC_JSON_NAME,
        'statics_csv': STATICS_CSV_NAME,
    }


def _cleanup_corrected_outputs(
    state: AppState,
    lifecycle: _DatumStaticLifecycle,
) -> None:
    if lifecycle.corrected_file_id is not None:
        with state.lock:
            state.file_registry.pop(lifecycle.corrected_file_id, None)
            if lifecycle.key1_byte is not None and lifecycle.key2_byte is not None:
                state.cached_readers.pop(
                    trace_store_cache_key(
                        lifecycle.corrected_file_id,
                        lifecycle.key1_byte,
                        lifecycle.key2_byte,
                    ),
                    None,
                )
    if lifecycle.corrected_store_path is None:
        return
    for tmp_path in lifecycle.corrected_store_path.parent.glob(
        f'{lifecycle.corrected_store_path.name}.tmp-*'
    ):
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path, ignore_errors=True)
    if lifecycle.corrected_store_built and lifecycle.corrected_store_path.exists():
        shutil.rmtree(lifecycle.corrected_store_path, ignore_errors=True)


def _run_datum_static_apply_job_body(
    job_id: str,
    req: DatumStaticApplyRequest,
    state: AppState,
    *,
    lifecycle: _DatumStaticLifecycle,
) -> JobCompletion | None:
    lifecycle.key1_byte = req.key1_byte
    lifecycle.key2_byte = req.key2_byte
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.job_dir = job_dir

    _set_job_progress_message(
        state,
        job_id,
        progress=0.05,
        message='resolving_source_trace_store',
    )
    ensure_job_not_cancelled(state, job_id)
    reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    source_store_path = _source_store_path(reader)
    dt = _resolve_dt(state, req.file_id)
    _write_job_meta(job_id=job_id, job_dir=job_dir, req=req)
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.10,
        message='loading_geometry_headers',
    )
    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(
            source_elevation_byte=req.geometry.source_elevation_byte,
            receiver_elevation_byte=req.geometry.receiver_elevation_byte,
            elevation_scalar_byte=req.geometry.elevation_scalar_byte,
            source_depth_byte=req.geometry.source_depth_byte,
            elevation_unit=req.geometry.elevation_unit,
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.20,
        message='computing_datum_static_shifts',
    )
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=(
            geometry.source_surface_elevation_m_sorted
        ),
        receiver_elevation_m_sorted=geometry.receiver_elevation_m_sorted,
        source_depth_m_sorted=(
            geometry.source_depth_m_sorted
            if req.geometry.source_depth_byte is not None
            else None
        ),
        datum_elevation_m=req.datum.elevation_m,
        replacement_velocity_m_s=req.datum.replacement_velocity_m_s,
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.25,
        message='validating_existing_static_headers',
    )
    existing_check = validate_existing_static_headers(
        reader=reader,
        config=ExistingStaticHeaderConfig(
            policy=req.existing_statics.policy,
            source_static_byte=req.existing_statics.source_static_byte,
            receiver_static_byte=req.existing_statics.receiver_static_byte,
            total_static_byte=req.existing_statics.total_static_byte,
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.30,
        message='validating_shift_limits',
    )
    shift_validation = validate_trace_shift_limits(
        trace_shift_s_sorted=result.trace_shift_s_sorted,
        max_abs_shift_ms=req.apply.max_abs_shift_ms,
        expected_n_traces=geometry.n_traces,
    )
    key1_sorted = _load_sorted_header(reader, req.key1_byte, name='key1')
    key2_sorted = _load_sorted_header(reader, req.key2_byte, name='key2')
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.35,
        message='writing_datum_static_artifacts',
    )
    write_datum_static_artifacts(
        job_dir=job_dir,
        trace_shift_s_sorted=result.trace_shift_s_sorted,
        source_shift_s_sorted=result.source_shift_s_sorted,
        receiver_shift_s_sorted=result.receiver_shift_s_sorted,
        source_surface_elevation_m_sorted=result.source_surface_elevation_m_sorted,
        source_depth_m_sorted=result.source_depth_m_sorted,
        source_depth_used_sorted=result.source_depth_used_sorted,
        source_elevation_m_sorted=result.source_elevation_m_sorted,
        receiver_elevation_m_sorted=result.receiver_elevation_m_sorted,
        key1_sorted=key1_sorted,
        key2_sorted=key2_sorted,
        datum_elevation_m=req.datum.elevation_m,
        replacement_velocity_m_s=req.datum.replacement_velocity_m_s,
        dt=dt,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        source_elevation_byte=req.geometry.source_elevation_byte,
        receiver_elevation_byte=req.geometry.receiver_elevation_byte,
        elevation_scalar_byte=req.geometry.elevation_scalar_byte,
        source_depth_byte=req.geometry.source_depth_byte,
        source_depth_enabled=req.geometry.source_depth_byte is not None,
        elevation_unit=req.geometry.elevation_unit,
        elevation_scalar_zero_count=geometry.elevation_scalar_zero_count,
        existing_static_check=existing_check,
        trace_shift_validation=shift_validation,
        header_source_segy_path=_reader_original_segy_path(reader),
    )
    ensure_job_not_cancelled(state, job_id)

    header_bytes_to_materialize = _header_bytes_to_materialize(req)
    corrected_store_path = _corrected_store_path(
        source_store_path=source_store_path,
        job_id=job_id,
    )
    lifecycle.corrected_store_path = corrected_store_path

    def progress_callback(builder_progress: float, message: str) -> None:
        mapped = 0.40 + (0.50 * max(0.0, min(1.0, float(builder_progress))))
        _set_job_progress_message(
            state,
            job_id,
            progress=mapped,
            message=message,
        )

    def cancel_check() -> bool:
        return _is_cancel_requested(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.40,
        message='building_corrected_trace_store',
    )
    try:
        build_result = build_time_shifted_trace_store(
            source_store_path=source_store_path,
            output_store_path=corrected_store_path,
            trace_shift_s_sorted=result.trace_shift_s_sorted,
            fill_value=req.apply.fill_value,
            output_dtype=req.apply.output_dtype,
            from_file_id=req.file_id,
            header_bytes_to_materialize=header_bytes_to_materialize,
            derived_metadata={
                'statics_kind': 'datum',
                'job_id': job_id,
                'solution_artifact': SOLUTION_NPZ_NAME,
                'qc_artifact': QC_JSON_NAME,
                'statics_csv': STATICS_CSV_NAME,
                'shift_field': 'trace_shift_s_sorted',
                'value_kind': 'applied_event_time_shift_s',
                'datum_elevation_m': req.datum.elevation_m,
                'replacement_velocity_m_s': req.datum.replacement_velocity_m_s,
                'geometry': {
                    'source_elevation_byte': req.geometry.source_elevation_byte,
                    'receiver_elevation_byte': req.geometry.receiver_elevation_byte,
                    'elevation_scalar_byte': req.geometry.elevation_scalar_byte,
                    'source_depth_byte': req.geometry.source_depth_byte,
                    'elevation_unit': req.geometry.elevation_unit,
                },
            },
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    except RuntimeError as exc:
        if 'cancelled' in str(exc).lower() and cancel_check():
            raise JobCancelledError() from exc
        raise
    lifecycle.corrected_store_built = True
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.90,
        message='registering_corrected_trace_store',
    )
    corrected_file_id = str(uuid4())
    register_trace_store(
        state=state,
        file_id=corrected_file_id,
        store_dir=build_result.store_path,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        dt=build_result.dt,
        update_registry=True,
        touch_meta=True,
        preload_header_bytes=header_bytes_to_materialize,
    )
    lifecycle.corrected_file_id = corrected_file_id
    with state.lock:
        state.jobs.set_static_corrected_file(
            job_id,
            corrected_file_id=corrected_file_id,
            corrected_store_path=str(build_result.store_path),
        )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.95,
        message='writing_corrected_file_manifest',
    )
    _write_json_atomic(
        job_dir / _CORRECTED_FILE_NAME,
        _build_corrected_file_payload(
            corrected_file_id=corrected_file_id,
            build_result=build_result,
            source_store_path=source_store_path,
            req=req,
            job_id=job_id,
        ),
    )
    _set_job_progress_message(state, job_id, progress=1.0, message='done')
    return JobCompletion(finished_ts=time.time())


def _handle_datum_static_job_error(
    *,
    state: AppState,
    lifecycle: _DatumStaticLifecycle,
) -> JobFailure:
    _cleanup_corrected_outputs(state, lifecycle)
    return JobFailure(finished_ts=time.time())


def _handle_datum_static_job_cancel(
    *,
    state: AppState,
    lifecycle: _DatumStaticLifecycle,
    exc: JobCancelledError,
) -> JobCompletion:
    _cleanup_corrected_outputs(state, lifecycle)
    finished_ts = float(exc.finished_ts) if exc.finished_ts is not None else time.time()
    return JobCompletion(finished_ts=finished_ts)


def run_datum_static_apply_job(
    job_id: str,
    req: DatumStaticApplyRequest,
    state: AppState,
) -> None:
    """Run one datum static correction job and register the corrected TraceStore."""
    lifecycle = _DatumStaticLifecycle(created_ts=_resolve_created_ts(state, job_id))
    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=lambda: _run_datum_static_apply_job_body(
            job_id,
            req,
            state,
            lifecycle=lifecycle,
        ),
        progress_1_on_done=True,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=lambda _exc: _handle_datum_static_job_error(
            state=state,
            lifecycle=lifecycle,
        ),
        on_cancel=lambda exc: _handle_datum_static_job_cancel(
            state=state,
            lifecycle=lifecycle,
            exc=exc,
        ),
    )


__all__ = ['run_datum_static_apply_job']
