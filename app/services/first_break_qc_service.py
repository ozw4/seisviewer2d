"""First-break QC background job service."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.api.schemas import FirstBreakQcRequest
from app.core.state import AppState
from app.services.first_break_qc_artifacts import (
    FIRST_BREAK_QC_CSV_NAME,
    FIRST_BREAK_QC_JSON_NAME,
    RESIDUAL_BY_KEY1_CSV_NAME,
    write_first_break_qc_artifacts,
)
from app.services.first_break_qc_inputs import build_first_break_qc_inputs
from app.services.first_break_qc_math import compute_first_break_qc_metrics
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.job_runner import (
    JobCancelledError,
    JobCompletion,
    JobFailure,
    ensure_job_not_cancelled,
    run_job_with_lifecycle,
)
from app.services.pick_source_loader import (
    LoadedPickSource,
    load_manual_memmap_pick_source,
    load_npz_pick_source,
)
from app.services.pipeline_artifacts import get_job_dir
from app.services.reader import get_reader
from app.trace_store.reader import TraceStoreSectionReader


@dataclass
class _FirstBreakQcLifecycle:
    created_ts: float
    job_dir: Path | None = None


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
    req: FirstBreakQcRequest,
) -> None:
    pick_source = req.pick_source.model_dump(mode='json', exclude_none=True)
    _write_json_atomic(
        job_dir / 'job_meta.json',
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'first_break_qc',
            'source_file_id': req.file_id,
            'key1_byte': req.key1_byte,
            'key2_byte': req.key2_byte,
            'request': req.model_dump(mode='json'),
            'inputs': {
                'datum_solution': req.datum_solution.model_dump(mode='json'),
                'pick_source': pick_source,
                'offset_byte': req.offset.offset_byte,
            },
            'artifacts': {
                'qc_json': FIRST_BREAK_QC_JSON_NAME,
                'qc_csv': FIRST_BREAK_QC_CSV_NAME,
                'residual_by_key1_csv': RESIDUAL_BY_KEY1_CSV_NAME,
            },
        },
    )


def _reader_n_samples(reader: TraceStoreSectionReader) -> int:
    getter = getattr(reader, 'get_n_samples', None)
    if callable(getter):
        n_samples = int(getter())
    else:
        traces = getattr(reader, 'traces', None)
        shape = getattr(traces, 'shape', ())
        if len(shape) < 2:
            raise ValueError('reader cannot provide number of samples')
        n_samples = int(shape[-1])
    if n_samples <= 0:
        raise ValueError('reader n_samples must be greater than 0')
    return n_samples


def _resolve_dt(state: AppState, file_id: str) -> float:
    dt = float(state.file_registry.get_dt(file_id))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError('dt must be finite and greater than 0')
    return dt


def _resolve_input_artifacts(
    state: AppState,
    req: FirstBreakQcRequest,
) -> tuple[Path, Path | None]:
    solution_path = resolve_job_artifact_path(
        state,
        job_id=req.datum_solution.job_id,
        name=req.datum_solution.name,
        allowed_job_types={'statics'},
        allowed_statics_kinds={'datum'},
        expected_file_id=req.file_id,
        expected_key1_byte=req.key1_byte,
        expected_key2_byte=req.key2_byte,
        reference_label='datum_solution',
    )

    pick_source = req.pick_source
    if pick_source.kind == 'manual_memmap':
        return solution_path, None
    if pick_source.job_id is None or pick_source.name is None:
        raise ValueError('pick_source artifact reference is incomplete')

    if pick_source.kind == 'batch_job_artifact':
        pick_path = resolve_job_artifact_path(
            state,
            job_id=pick_source.job_id,
            name=pick_source.name,
            allowed_job_types={'batch_apply'},
            expected_file_id=req.file_id,
            expected_key1_byte=req.key1_byte,
            expected_key2_byte=req.key2_byte,
            reference_label='pick_source',
        )
        return solution_path, pick_path

    if not pick_source.name.endswith('.npz'):
        raise ValueError('manual_npz_artifact name must end with .npz')
    pick_path = resolve_job_artifact_path(
        state,
        job_id=pick_source.job_id,
        name=pick_source.name,
        allowed_job_types={'statics', 'batch_apply', 'pipeline'},
    )
    return solution_path, pick_path


def _load_pick_source(
    *,
    req: FirstBreakQcRequest,
    reader: TraceStoreSectionReader,
    pick_artifact_path: Path | None,
    expected_dt: float,
    expected_n_samples: int,
    state: AppState,
) -> LoadedPickSource:
    pick_source = req.pick_source
    if pick_source.kind == 'manual_memmap':
        return load_manual_memmap_pick_source(
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            state=state,
        )
    if pick_artifact_path is None:
        raise ValueError('pick source artifact path is required')
    source_kind = 'batch_npz' if pick_source.kind == 'batch_job_artifact' else 'manual_npz'
    return load_npz_pick_source(
        pick_artifact_path,
        reader=reader,
        expected_dt=expected_dt,
        expected_n_samples=expected_n_samples,
        source_kind=source_kind,
    )


def _pick_source_artifact_name(req: FirstBreakQcRequest) -> str | None:
    return None if req.pick_source.kind == 'manual_memmap' else req.pick_source.name


def _run_first_break_qc_job_body(
    job_id: str,
    req: FirstBreakQcRequest,
    state: AppState,
    *,
    lifecycle: _FirstBreakQcLifecycle,
) -> JobCompletion | None:
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.job_dir = job_dir
    _write_job_meta(job_id=job_id, job_dir=job_dir, req=req)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.05,
        message='resolving_source_trace_store',
    )
    reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    dt = _resolve_dt(state, req.file_id)
    n_samples = _reader_n_samples(reader)
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.10,
        message='resolving_input_artifacts',
    )
    solution_path, pick_artifact_path = _resolve_input_artifacts(state, req)
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.25,
        message='loading_pick_source',
    )
    ensure_job_not_cancelled(state, job_id)
    pick_source = _load_pick_source(
        req=req,
        reader=reader,
        pick_artifact_path=pick_artifact_path,
        expected_dt=dt,
        expected_n_samples=n_samples,
        state=state,
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.40,
        message='loading_datum_solution_and_offset',
    )
    ensure_job_not_cancelled(state, job_id)
    inputs = build_first_break_qc_inputs(
        pick_source=pick_source,
        solution_npz_path=solution_path,
        reader=reader,
        offset_byte=req.offset.offset_byte,
        expected_dt=dt,
        expected_n_samples=n_samples,
        expected_key1_byte=req.key1_byte,
        expected_key2_byte=req.key2_byte,
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.60,
        message='computing_first_break_qc',
    )
    ensure_job_not_cancelled(state, job_id)
    metrics = compute_first_break_qc_metrics(
        inputs,
        require_linear_offset_model=req.qc.require_linear_offset_model,
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.85,
        message='writing_first_break_qc_artifacts',
    )
    ensure_job_not_cancelled(state, job_id)
    write_first_break_qc_artifacts(
        job_dir=job_dir,
        inputs=inputs,
        metrics=metrics,
        solution_artifact_name=req.datum_solution.name,
        pick_source_artifact_name=_pick_source_artifact_name(req),
    )
    _set_job_progress_message(state, job_id, progress=1.0, message='done')
    return JobCompletion(finished_ts=time.time())


def _handle_first_break_qc_job_error(
    *,
    lifecycle: _FirstBreakQcLifecycle,
) -> JobFailure:
    return JobFailure(finished_ts=time.time())


def _handle_first_break_qc_job_cancel(
    *,
    lifecycle: _FirstBreakQcLifecycle,
    exc: JobCancelledError,
) -> JobCompletion:
    finished_ts = float(exc.finished_ts) if exc.finished_ts is not None else time.time()
    return JobCompletion(finished_ts=finished_ts)


def run_first_break_qc_job(
    job_id: str,
    req: FirstBreakQcRequest,
    state: AppState,
) -> None:
    """Run one first-break QC job and persist only QC artifacts."""
    lifecycle = _FirstBreakQcLifecycle(created_ts=_resolve_created_ts(state, job_id))
    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=lambda: _run_first_break_qc_job_body(
            job_id,
            req,
            state,
            lifecycle=lifecycle,
        ),
        progress_1_on_done=True,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=lambda _exc: _handle_first_break_qc_job_error(
            lifecycle=lifecycle,
        ),
        on_cancel=lambda exc: _handle_first_break_qc_job_cancel(
            lifecycle=lifecycle,
            exc=exc,
        ),
    )


__all__ = ['run_first_break_qc_job']
