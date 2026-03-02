"""First-break inference endpoints."""

from __future__ import annotations

import threading
from typing import Annotated, Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from app.api._helpers import get_state
from app.api.binary_codec import pack_msgpack_gzip
from app.api.schemas import PipelineSpec
from app.core.state import AppState
from app.services.fbpick_support import (
    DEFAULT_FBPICK_MODEL_ID,
    OFFSET_BYTE_FIXED,
    _maybe_attach_fbpick_offsets,
)
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_taps import get_section_from_pipeline_tap
from app.services.reader import coerce_section_f32, get_reader
from app.utils.fbpick_models import (
    list_fbpick_models,
    model_version,
    resolve_model_path,
)
from app.utils.pipeline import apply_pipeline
from app.codec.quantize import quantize_float32

router = APIRouter()


def _effective_channel(channel: str | None) -> str:
    return 'P' if channel is None else channel


def _resolve_model_selection(model_id: str | None) -> tuple[str, bool, str]:
    model_path = resolve_model_path(model_id, require_exists=True)
    chosen_model_id = DEFAULT_FBPICK_MODEL_ID if model_id is None else model_id
    uses_offset = 'offset' in chosen_model_id.lower()
    return chosen_model_id, uses_offset, model_version(model_path)


def _build_fbpick_cache_key(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    offset_byte: int | None,
    tile_h: int,
    tile_w: int,
    overlap: int,
    amp: bool,
    pipeline_key: str | None,
    tap_label: str | None,
    model_id: str,
    model_ver: str,
    channel: str | None = None,
) -> tuple[Any, ...]:
    effective_channel = _effective_channel(channel)
    return (
        file_id,
        int(key1),
        int(key1_byte),
        int(key2_byte),
        offset_byte,
        int(tile_h),
        int(tile_w),
        int(overlap),
        bool(amp),
        pipeline_key,
        tap_label,
        model_id,
        model_ver,
        effective_channel,
        'fbpick',
    )


class FbpickRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1: int
    key1_byte: int = 189
    key2_byte: int = 193
    offset_byte: int | None = None
    tile_h: int = 128
    tile_w: int = 6016
    overlap: int = 32
    amp: bool = True
    pipeline_key: str | None = None
    tap_label: str | None = None
    model_id: str | None = None
    channel: str | None = None


def _run_fbpick_job(job_id: str, req: FbpickRequest, state: AppState) -> None:
    try:
        with state.lock:
            job = state.jobs.get(job_id)
            if job is None:
                return
            state.jobs.set_status(job_id, 'running')
            cache_key = job.get('cache_key')
            file_id_obj = job.get('file_id', req.file_id)
            key1_obj = job.get('key1', req.key1)
            key1_byte_obj = job.get('key1_byte', req.key1_byte)
            key2_byte_obj = job.get('key2_byte', req.key2_byte)
            pipeline_key_obj = job.get('pipeline_key', req.pipeline_key)
            tap_label_obj = job.get('tap_label', req.tap_label)
            offset_byte_obj = job.get('offset_byte', req.offset_byte)
            model_id_obj = job.get('model_id', req.model_id)
            channel_obj = job.get('channel', req.channel)
        if cache_key is None:
            raise RuntimeError('FB pick job metadata is inconsistent: cache_key')
        if not isinstance(file_id_obj, str) or not file_id_obj:
            raise RuntimeError('FB pick job metadata is inconsistent: file_id')
        file_id = file_id_obj

        key1 = int(key1_obj)
        key1_byte = int(key1_byte_obj)
        key2_byte = int(key2_byte_obj)
        pipeline_key = (
            str(pipeline_key_obj) if isinstance(pipeline_key_obj, str) else None
        )
        tap_label = str(tap_label_obj) if isinstance(tap_label_obj, str) else None
        offset_byte = int(offset_byte_obj) if isinstance(offset_byte_obj, int) else None
        model_id = str(model_id_obj) if isinstance(model_id_obj, str) else None
        raw_channel = channel_obj if isinstance(channel_obj, str) else None
        channel = _effective_channel(raw_channel)

        reader: object | None = None
        if offset_byte is not None:
            reader = get_reader(file_id, key1_byte, key2_byte, state=state)
        if pipeline_key and tap_label:
            section = get_section_from_pipeline_tap(
                file_id=file_id,
                key1=key1,
                key1_byte=key1_byte,
                key2_byte=key2_byte,
                pipeline_key=pipeline_key,
                tap_label=tap_label,
                offset_byte=offset_byte,
                state=state,
            )
        else:
            if reader is None:
                reader = get_reader(file_id, key1_byte, key2_byte, state=state)
            view = reader.get_section(key1)
            section = coerce_section_f32(view.arr, view.scale)

        section = np.ascontiguousarray(np.asarray(section, dtype=np.float32))
        spec = PipelineSpec(
            steps=[
                {
                    'kind': 'analyzer',
                    'name': 'fbpick',
                    'params': {
                        'tile': (req.tile_h, req.tile_w),
                        'overlap': req.overlap,
                        'amp': req.amp,
                        'model_id': model_id,
                        'channel': channel,
                    },
                }
            ]
        )
        meta: dict[str, Any] = {}
        if reader is not None:
            meta = _maybe_attach_fbpick_offsets(
                meta,
                spec=spec,
                reader=reader,
                key1=key1,
                offset_byte=offset_byte,
                section_shape=(int(section.shape[0]), int(section.shape[1])),
            )
        out = apply_pipeline(section, spec=spec, meta=meta, taps=None)
        prob = out['fbpick']['prob']
        scale, q = quantize_float32(prob, fixed_scale=127.0)
        obj = {
            'scale': scale,
            'shape': q.shape,
            'data': q.tobytes(),
        }
        packed_payload = pack_msgpack_gzip(obj)
        with state.lock:
            state.fbpick_cache[cache_key] = packed_payload
        with state.lock:
            state.jobs.mark_done(job_id)
    except Exception as e:  # noqa: BLE001
        with state.lock:
            state.jobs.mark_error(job_id, str(e))


@router.post('/fbpick_section_bin')
def fbpick_section_bin(req: FbpickRequest, request: Request):
    state = get_state(request.app)
    cleanup_in_memory_state(state)

    chosen_model_id, uses_offset, model_ver = _resolve_model_selection(req.model_id)
    forced_offset_byte = OFFSET_BYTE_FIXED if uses_offset else None

    pipeline_key = req.pipeline_key
    tap_label = req.tap_label
    channel = _effective_channel(req.channel)
    key1 = req.key1
    cache_key = _build_fbpick_cache_key(
        file_id=req.file_id,
        key1=key1,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        offset_byte=forced_offset_byte,
        tile_h=req.tile_h,
        tile_w=req.tile_w,
        overlap=req.overlap,
        amp=req.amp,
        pipeline_key=pipeline_key,
        tap_label=tap_label,
        model_id=chosen_model_id,
        model_ver=model_ver,
        channel=channel,
    )
    with state.lock:
        payload = state.fbpick_cache.get(cache_key)
        cache_hit = payload is not None
    job_id = str(uuid4())
    with state.lock:
        job_state = state.jobs.create_fbpick_job(
            job_id,
            cache_key=cache_key,
            file_id=req.file_id,
            key1=key1,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            pipeline_key=pipeline_key,
            tap_label=tap_label,
            offset_byte=forced_offset_byte,
            model_id=chosen_model_id,
            channel=channel,
        )
        payload = state.fbpick_cache.get(cache_key)
        cache_hit = payload is not None
    assert not any(isinstance(value, np.ndarray) for value in job_state.values())

    req2 = req.model_copy(
        update={
            'offset_byte': forced_offset_byte,
            'model_id': chosen_model_id,
            'channel': channel,
        }
    )

    if cache_hit:
        with state.lock:
            state.jobs.mark_done(job_id)
    else:
        threading.Thread(
            target=_run_fbpick_job, args=(job_id, req2, state), daemon=True
        ).start()
    with state.lock:
        job = state.jobs.get(job_id)
        status = job.get('status', 'unknown') if job is not None else 'unknown'
    return {'job_id': job_id, 'status': status}


@router.get('/fbpick_models')
def get_fbpick_models() -> dict[str, object]:
    models = list_fbpick_models()
    default_model_id: str | None = None
    if models:
        model_ids = [str(item['id']) for item in models]
        if DEFAULT_FBPICK_MODEL_ID in model_ids:
            default_model_id = DEFAULT_FBPICK_MODEL_ID
        else:
            default_model_id = model_ids[0]
    return {'default_model_id': default_model_id, 'models': models}


@router.get('/fbpick_job_status')
def fbpick_job_status(request: Request, job_id: Annotated[str, Query(...)]):
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        status = job.get('status', 'unknown')
        message = job.get('message', '')
    return {'status': status, 'message': message}


@router.get('/get_fbpick_section_bin')
def get_fbpick_section_bin(request: Request, job_id: Annotated[str, Query(...)]):
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Result not ready')
        status = job.get('status')
        if status == 'expired':
            raise HTTPException(status_code=410, detail='Result expired')
        if status != 'done':
            raise HTTPException(status_code=404, detail='Result not ready')
        cache_key = job.get('cache_key')
        payload = state.fbpick_cache.get(cache_key)
        if payload is None:
            state.jobs.mark_expired(job_id)
            raise HTTPException(status_code=410, detail='Result expired')
    return Response(
        payload,
        media_type='application/octet-stream',
        headers={'Content-Encoding': 'gzip'},
    )
