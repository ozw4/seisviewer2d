"""Pipeline execution and tap retrieval endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
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
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.fbpick_support import (
    OFFSET_BYTE_FIXED,
    _maybe_attach_fbpick_offsets,
    _spec_uses_offset,
)
from app.services.pipeline_taps import build_pipeline_tap_cache_key
from app.services.reader import coerce_section_f32, get_reader
from app.core.state import AppState
from app.utils.pipeline import apply_pipeline, pipeline_key
from app.utils.utils import to_builtin

router = APIRouter()
logger = logging.getLogger(__name__)


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


def _run_pipeline_all_job(
    job_id: str, req: PipelineAllRequest, pipe_key: str, state: AppState
) -> None:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            return
        job['status'] = 'running'
    try:
        reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
        key1_values = reader.get_key1_values().tolist()
        total = len(key1_values) or 1
        taps = req.taps
        for idx, key1 in enumerate(key1_values):
            key1_value = int(key1)
            view = reader.get_section(key1_value)
            section = coerce_section_f32(view.arr, view.scale)

            dt = float(state.file_registry.get_dt(req.file_id))
            meta = {'dt': dt}

            # offsetをreqからそのまま適用（pass-through）
            meta = _maybe_attach_fbpick_offsets(
                meta,
                spec=req.spec,
                reader=reader,
                key1=key1_value,
                offset_byte=req.offset_byte,  # already forced by caller
                section_shape=(int(section.shape[0]), int(section.shape[1])),
            )

            out = apply_pipeline(section, spec=req.spec, meta=meta, taps=taps)

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

            with state.lock:
                job = state.jobs.get(job_id)
                if job is None:
                    return
                job['progress'] = (idx + 1) / total

        with state.lock:
            job = state.jobs.get(job_id)
            if job is not None:
                job['status'] = 'done'
                job['finished_ts'] = time.time()
    except Exception as e:  # noqa: BLE001
        with state.lock:
            job = state.jobs.get(job_id)
            if job is not None:
                job['status'] = 'error'
                job['message'] = str(e)
                job['finished_ts'] = time.time()


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
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    view = reader.get_section(key1)
    section = coerce_section_f32(view.arr, view.scale)

    trace_slice: slice | None = None
    window_hash = None
    if window:
        tr_min = int(window.get('tr_min', 0))
        tr_max = int(window.get('tr_max', section.shape[0]))
        t_min = int(window.get('t_min', 0))
        t_max = int(window.get('t_max', section.shape[1]))
        section = section[tr_min:tr_max, t_min:t_max]
        trace_slice = slice(tr_min, tr_max)
        clean_window = {
            'tr_min': tr_min,
            'tr_max': tr_max,
            't_min': t_min,
            't_max': t_max,
        }
        window_hash = hashlib.sha256(
            json.dumps(clean_window, sort_keys=True).encode()
        ).hexdigest()[:8]

    dt = float(state.file_registry.get_dt(file_id))
    meta = {'dt': dt}

    # pass-through + 同期フォールバック
    # None で来ていて、かつ FBPICK モードなら固定オフセットに揃える
    forced_offset_byte = (
        OFFSET_BYTE_FIXED
        if (_spec_uses_offset(spec) and offset_byte is None)
        else offset_byte
    )

    meta = _maybe_attach_fbpick_offsets(
        meta,
        spec=spec,
        reader=reader,
        key1=key1,
        offset_byte=forced_offset_byte,
        trace_slice=trace_slice,
        section_shape=(int(section.shape[0]), int(section.shape[1])),
    )

    pipe_key = pipeline_key(spec)

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
            out = apply_pipeline(section, spec=spec, meta=meta, taps=misses)
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
            out = apply_pipeline(section, spec=spec, meta=meta, taps=misses)
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

    out = apply_pipeline(section, spec=spec, meta=meta, taps=None)
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

    # pass-through + 同期フォールバック（sectionと同一ロジック）
    forced_offset_byte = (
        OFFSET_BYTE_FIXED
        if (_spec_uses_offset(spec) and offset_byte is None)
        else offset_byte
    )

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
        state.jobs[job_id] = {
            'status': 'queued',
            'progress': 0.0,
            'message': '',
            'created_ts': time.time(),
            'file_id': file_id,
            'key1_byte': key1_byte,
            'key2_byte': key2_byte,
            'pipeline_key': pipe_key,
            'offset_byte': forced_offset_byte,  # artifact側も同じ値で参照
            'artifacts_dir': str(get_job_dir(job_id)),
        }

    threading.Thread(
        target=_run_pipeline_all_job, args=(job_id, req, pipe_key, state), daemon=True
    ).start()

    with state.lock:
        status = state.jobs[job_id]['status']
    return {'job_id': job_id, 'state': status}


@router.get('/pipeline/job/{job_id}/status', response_model=PipelineJobStatusResponse)
def pipeline_job_status(request: Request, job_id: str) -> PipelineJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        job_state = job.get('status', 'unknown')
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
