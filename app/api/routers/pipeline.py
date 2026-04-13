"""Pipeline execution and tap retrieval endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Annotated, Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.api._helpers import reject_legacy_key1_query_params, get_state
from app.api.schemas import (
    PipelineAllResponse,
    PipelineJobStatusResponse,
    PipelineSectionResponse,
    PipelineSpec,
)
from app.services.pipeline_artifacts import (
    get_job_dir,
    maybe_cleanup_expired_jobs,
    read_artifact,
    write_artifact,
)
from app.services.fbpick_support import _maybe_attach_fbpick_offsets
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.job_manager import JobManager
from app.services.job_runner import (
    ensure_job_not_cancelled,
    run_job_with_lifecycle,
    request_job_cancel,
    set_job_message,
    set_job_progress,
    start_job_thread,
)
from app.services.pipeline_execution import (
    SectionSourceSpec,
    prepare_pipeline_execution,
    resolve_effective_offset_byte,
    run_pipeline_execution,
)
from app.services.pipeline_taps import build_pipeline_tap_cache_key
from app.services import reader as _reader_service
from app.services.reader import get_reader
from app.core.state import AppState
from app.utils.pipeline import apply_pipeline, pipeline_key
from app.utils.serialization import to_builtin

router = APIRouter()
logger = logging.getLogger(__name__)
coerce_section_f32 = _reader_service.coerce_section_f32


class PipelineAllRequest(BaseModel):
    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    offset_byte: int | None = None
    spec: PipelineSpec
    taps: list[str] = Field(default_factory=list)
    downsample_quicklook: bool = True


def _required_job_str(job: dict[str, object], *, field: str) -> str:
    value = job.get(field)
    if isinstance(value, str) and value:
        return value
    raise HTTPException(
        status_code=500,
        detail=f'Job metadata is inconsistent: {field}',
    )


def _required_job_int(job: dict[str, object], *, field: str) -> int:
    value = job.get(field)
    if isinstance(value, int):
        return value
    raise HTTPException(
        status_code=500,
        detail=f'Job metadata is inconsistent: {field}',
    )


def _optional_job_int(job: dict[str, object], *, field: str) -> int | None:
    value = job.get(field)
    if value is None or isinstance(value, int):
        return value
    raise HTTPException(
        status_code=500,
        detail=f'Job metadata is inconsistent: {field}',
    )


def _run_pipeline_all_job_body(
    job_id: str, req: PipelineAllRequest, pipe_key: str, state: AppState
) -> None:
    reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    key1_values = reader.get_key1_values().tolist()
    total = len(key1_values) or 1
    taps = req.taps
    step_names = [str(step.label or step.name) for step in req.spec.steps]
    for idx, key1 in enumerate(key1_values):
        ensure_job_not_cancelled(state, job_id)
        key1_value = int(key1)
        step_text = ' -> '.join(step_names) if step_names else 'pipeline'
        set_job_message(
            state,
            job_id,
            f'Running section {idx + 1}/{total} (key1={key1_value}). Steps: {step_text}.',
        )
        context = prepare_pipeline_execution(
            spec=req.spec,
            source=SectionSourceSpec(
                file_id=req.file_id,
                key1=key1_value,
                key1_byte=req.key1_byte,
                key2_byte=req.key2_byte,
                offset_byte=req.offset_byte,
            ),
            state=state,
            reader=reader,
            reader_getter=get_reader,
            offset_attacher=_maybe_attach_fbpick_offsets,
        )
        out = run_pipeline_execution(context, taps=taps, apply_fn=apply_pipeline)

        for k, v in out.items():
            val = v
            if req.downsample_quicklook and isinstance(v, np.ndarray):
                val = v[::4, ::4]
            payload_obj = to_builtin(val)
            cache_key = build_pipeline_tap_cache_key(
                file_id=req.file_id,
                key1=key1_value,
                key1_byte=req.key1_byte,
                key2_byte=req.key2_byte,
                pipeline_key=pipe_key,
                window_hash=None,  # all-job artifacts are always full section
                offset_byte=req.offset_byte,  # pass-through (forced by caller)
                tap_label=str(k),
            )
            logger.debug('PIPELINE_CACHE_SET all %s', cache_key)
            with state.lock:
                state.pipeline_tap_cache.set(cache_key, payload_obj)
            write_artifact(
                job_id=job_id,
                key1=key1_value,
                tap_label=str(k),
                payload=payload_obj,
            )

        ensure_job_not_cancelled(state, job_id)
        if not set_job_progress(state, job_id, (idx + 1) / total):
            return


def _run_pipeline_all_job(
    job_id: str, req: PipelineAllRequest, pipe_key: str, state: AppState
) -> None:
    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=lambda: _run_pipeline_all_job_body(job_id, req, pipe_key, state),
    )


@router.post(
    '/pipeline/section',
    response_model=PipelineSectionResponse,
    dependencies=[Depends(reject_legacy_key1_query_params)],
)
def pipeline_section(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1: Annotated[int, Query(...)],
    spec: Annotated[PipelineSpec, Body(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
    offset_byte: Annotated[int | None, Query()] = None,  # pass-throughで受ける
    taps: Annotated[list[str] | None, Body()] = None,
    window: Annotated[dict[str, int | float] | None, Body()] = None,
    list_only: Annotated[bool, Query()] = False,
):
    if list_only and window is not None:
        raise HTTPException(
            status_code=422,
            detail='window is not supported when list_only=true',
        )

    state = get_state(request.app)
    context = prepare_pipeline_execution(
        spec=spec,
        source=SectionSourceSpec(
            file_id=file_id,
            key1=key1,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            offset_byte=offset_byte,
            window=window,
        ),
        state=state,
        reader_getter=get_reader,
        offset_attacher=_maybe_attach_fbpick_offsets,
    )
    window_hash = None
    if context.window_bounds is not None:
        window_hash = hashlib.sha256(
            json.dumps(context.window_bounds, sort_keys=True).encode()
        ).hexdigest()[:8]
    pipe_key = pipeline_key(spec)
    forced_offset_byte = context.effective_offset_byte

    tap_names = taps or []
    if list_only:
        # ★ list_only でもサーバ側で計算して cache に置く（返すのはラベルだけ）
        labels = tap_names if tap_names else ['final']

        # 既存 cache を確認し、無いものだけ計算
        misses: list[str] = []
        for label in labels:
            cache_key = build_pipeline_tap_cache_key(
                file_id=file_id,
                key1=key1,
                key1_byte=key1_byte,
                key2_byte=key2_byte,
                pipeline_key=pipe_key,
                window_hash=None,  # list_only is full-section only
                offset_byte=forced_offset_byte,
                tap_label=label,
            )
            with state.lock:
                payload = state.pipeline_tap_cache.get(cache_key)
            if payload is None:
                misses.append(label)

        if misses:
            out = run_pipeline_execution(context, taps=misses, apply_fn=apply_pipeline)
            for k, v in out.items():
                val = to_builtin(v)
                cache_key = build_pipeline_tap_cache_key(
                    file_id=file_id,
                    key1=key1,
                    key1_byte=key1_byte,
                    key2_byte=key2_byte,
                    pipeline_key=pipe_key,
                    window_hash=None,
                    offset_byte=forced_offset_byte,
                    tap_label=str(k),
                )
                with state.lock:
                    state.pipeline_tap_cache.set(cache_key, val)

        # レスポンスは軽量（ラベル存在のみ）
        return {'taps': dict.fromkeys(labels, True), 'pipeline_key': pipe_key}

    if tap_names:
        taps_out: dict[str, Any] = {}
        misses: list[str] = []

        for tap in tap_names:
            cache_key = build_pipeline_tap_cache_key(
                file_id=file_id,
                key1=key1,
                key1_byte=key1_byte,
                key2_byte=key2_byte,
                pipeline_key=pipe_key,
                window_hash=window_hash,  # window-specific taps have non-None hash
                offset_byte=forced_offset_byte,
                tap_label=tap,
            )
            logger.debug('PIPELINE_CACHE_GET pipelinesection %s', cache_key)
            with state.lock:
                payload = state.pipeline_tap_cache.get(cache_key)
            if payload is not None:
                taps_out[tap] = payload
            else:
                misses.append(tap)

        if misses:
            out = run_pipeline_execution(context, taps=misses, apply_fn=apply_pipeline)
            for k, v in out.items():
                val = to_builtin(v)
                taps_out[k] = val
                cache_key = build_pipeline_tap_cache_key(
                    file_id=file_id,
                    key1=key1,
                    key1_byte=key1_byte,
                    key2_byte=key2_byte,
                    pipeline_key=pipe_key,
                    window_hash=window_hash,
                    offset_byte=forced_offset_byte,
                    tap_label=str(k),
                )
                with state.lock:
                    state.pipeline_tap_cache.set(cache_key, val)

        return {'taps': taps_out, 'pipeline_key': pipe_key}

    out = run_pipeline_execution(context, taps=None, apply_fn=apply_pipeline)
    return {'taps': to_builtin(out), 'pipeline_key': pipe_key}


@router.post('/pipeline/all', response_model=PipelineAllResponse)
def pipeline_all(
    request: Request,
    file_id: Annotated[str, Query(...)],
    spec: Annotated[PipelineSpec, Body(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
    downsample_quicklook: Annotated[bool, Query()] = True,
    offset_byte: Annotated[int | None, Query()] = None,  # pass-throughで受ける
    taps: Annotated[list[str] | None, Body()] = None,
):
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    tap_names = taps or []

    forced_offset_byte = resolve_effective_offset_byte(spec, offset_byte)

    req = PipelineAllRequest(
        file_id=file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        offset_byte=forced_offset_byte,  # ここで確定させて渡す
        spec=spec,
        taps=tap_names,
        downsample_quicklook=downsample_quicklook,
    )

    job_id = str(uuid4())
    pipe_key = pipeline_key(spec)

    with state.lock:
        job_state = state.jobs.create_pipeline_all_job(
            job_id,
            file_id=file_id,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            pipeline_key=pipe_key,
            offset_byte=forced_offset_byte,
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=_run_pipeline_all_job,
        args=(job_id, req, pipe_key, state),
    )

    return {'job_id': job_id, 'state': status}


@router.get('/pipeline/job/{job_id}/status', response_model=PipelineJobStatusResponse)
def pipeline_job_status(request: Request, job_id: str) -> PipelineJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        job_state = JobManager.normalize_status_value(job.get('status', 'unknown'))
        progress = job.get('progress', 0.0)
        message = job.get('message', '')
    return {
        'state': job_state,
        'progress': progress,
        'message': message,
    }


@router.post('/pipeline/job/{job_id}/cancel', response_model=PipelineJobStatusResponse)
def pipeline_job_cancel(request: Request, job_id: str) -> PipelineJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)

    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')

    request_job_cancel(state, job_id)

    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        job_state = JobManager.normalize_status_value(job.get('status', 'unknown'))
        progress = job.get('progress', 0.0)
        message = job.get('message', '')
    return {
        'state': job_state,
        'progress': progress,
        'message': message,
    }


@router.get(
    '/pipeline/job/{job_id}/artifact',
    response_model=Any,
    dependencies=[Depends(reject_legacy_key1_query_params)],
)
def pipeline_job_artifact(
    request: Request,
    job_id: str,
    key1: Annotated[int, Query(...)],
    tap: Annotated[str, Query(...)],
):
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        job_snapshot = dict(job)

    maybe_cleanup_expired_jobs()

    disk_payload = read_artifact(job_id=job_id, key1=key1, tap_label=tap)
    if disk_payload is not None:
        return disk_payload

    job_file_id = _required_job_str(job_snapshot, field='file_id')
    job_pipeline_key = _required_job_str(job_snapshot, field='pipeline_key')
    job_key1_byte = _required_job_int(job_snapshot, field='key1_byte')
    job_key2_byte = _required_job_int(job_snapshot, field='key2_byte')
    job_offset_byte = _optional_job_int(job_snapshot, field='offset_byte')

    cache_key = build_pipeline_tap_cache_key(
        file_id=job_file_id,
        key1=key1,
        key1_byte=job_key1_byte,
        key2_byte=job_key2_byte,
        pipeline_key=job_pipeline_key,
        window_hash=None,  # /pipeline/all artifacts are always full-section taps
        offset_byte=job_offset_byte,
        tap_label=tap,
    )

    # Migration compatibility: fall back to in-memory LRU when disk artifact is absent.
    logger.debug('PIPELINE_CACHE_GET artifact %s', cache_key)

    with state.lock:
        payload = state.pipeline_tap_cache.get(cache_key)
    if payload is None:
        raise HTTPException(status_code=404, detail='Artifact not ready')
    return payload
