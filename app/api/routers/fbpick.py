"""First-break inference endpoints."""

from __future__ import annotations

import gzip
import threading
from typing import Any
from uuid import uuid4

import msgpack
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response as FastAPIResponse
from fastapi.responses import Response

from app.api._helpers import (
        OFFSET_BYTE_FIXED,
        USE_FBPICK_OFFSET,
        PipelineTapNotFoundError,
        _maybe_attach_fbpick_offsets,
        fbpick_cache,
        get_reader,
        get_section_from_pipeline_tap,
        jobs,
        load_section_by_indices,
        make_cache_key,
)
from app.api.schemas import FbpickRequest, PipelineSpec
from app.utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from app.utils.pipeline import apply_pipeline
from app.utils.utils import (
        SegySectionReader,
        TraceStoreSectionReader,
        quantize_float32,
)
from app.utils.key_resolver import resolve_indices_slice_on_demand

router = APIRouter()

WARNING_DEPRECATED_IDX = '299 - key1_idx is deprecated; use key1_value'


def _run_fbpick_job(job_id: str, req: FbpickRequest) -> None:
        job = jobs[job_id]
        job['status'] = 'running'
        try:
                forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

                cache_key = job['cache_key']
                section_override = job.pop('section_override', None)
                reader: SegySectionReader | TraceStoreSectionReader | None = get_reader(
                        req.file_id, req.key1_byte, req.key2_byte
                )
                effective_length = req.length if req.length is not None else 1_000_000_000
                gather_idx = resolve_indices_slice_on_demand(
                        reader, req.key1_value, req.start, effective_length
                )
                if gather_idx.size == 0:
                        raise ValueError('Requested gather has no traces')
                if section_override is not None:
                        section = np.asarray(section_override, dtype=np.float32)
                elif req.pipeline_key and req.tap_label:
                        section = get_section_from_pipeline_tap(
                                file_id=req.file_id,
                                key1_idx=req.key1_idx,
                                key1_byte=req.key1_byte,
                                pipeline_key=req.pipeline_key,
                                tap_label=req.tap_label,
                                offset_byte=forced_offset_byte,
                                key1_value=req.key1_value,
                                start=req.start,
                                length=int(gather_idx.size),
                        )
                        section = np.asarray(section, dtype=np.float32)
                        available = section.shape[0]
                        window_end = req.start + int(gather_idx.size)
                        if available == int(gather_idx.size):
                                pass
                        elif available >= window_end:
                                section = section[req.start:window_end]
                        else:
                                raise ValueError('Pipeline tap shorter than requested gather window')
                else:
                        section = load_section_by_indices(reader, gather_idx)
                section = np.ascontiguousarray(section, dtype=np.float32)
                spec = PipelineSpec(
                        steps=[
                                {
                                        'kind': 'analyzer',
                                        'name': 'fbpick',
                                        'params': {
                                                'tile': (req.tile_h, req.tile_w),
                                                'overlap': req.overlap,
                                                'amp': req.amp,
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
                                key1_idx=None,
                                offset_byte=forced_offset_byte,
                                indices=gather_idx,
                        )
                out = apply_pipeline(section, spec=spec, meta=meta, taps=None)
                prob = out['fbpick']['prob']
                scale, q = quantize_float32(prob, fixed_scale=127.0)
                payload = msgpack.packb(
                        {
                                'scale': scale,
                                'shape': q.shape,
                                'data': q.tobytes(),
                        }
                )
                fbpick_cache[cache_key] = gzip.compress(payload)
                job['status'] = 'done'
        except Exception as e:  # noqa: BLE001
                job['status'] = 'error'
                job['message'] = str(e)


@router.post('/fbpick_section_bin')
def fbpick_section_bin(req: FbpickRequest, response: FastAPIResponse):
        if not FBPICK_MODEL_PATH.exists():
                raise HTTPException(status_code=409, detail='FB pick model weights not found')

        forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

        pipeline_key = req.pipeline_key
        tap_label = req.tap_label
        reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
        effective_length = req.length if req.length is not None else 1_000_000_000
        try:
                gather_idx = resolve_indices_slice_on_demand(
                        reader, req.key1_value, req.start, effective_length
                )
        except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if gather_idx.size == 0:
                raise HTTPException(status_code=422, detail='Requested gather has no traces')

        cache_spec = {
                'kb1': req.key1_byte,
                'k2b': req.key2_byte,
                'offset': forced_offset_byte,
                'tile': (req.tile_h, req.tile_w),
                'overlap': req.overlap,
                'amp': bool(req.amp),
                'pipe': pipeline_key,
                'tap': tap_label,
                'kind': 'fbpick',
        }
        cache_key = make_cache_key(
                req.file_id,
                req.key1_value,
                req.start,
                int(gather_idx.size),
                cache_spec,
        )
        section_override: np.ndarray | None = None
        wants_pipeline = bool(pipeline_key and tap_label)
        if cache_key not in fbpick_cache and wants_pipeline:
                try:
                        section_override = get_section_from_pipeline_tap(
                                file_id=req.file_id,
                                key1_idx=req.key1_idx,
                                key1_byte=req.key1_byte,
                                pipeline_key=pipeline_key,
                                tap_label=tap_label,
                                offset_byte=forced_offset_byte,
                                key1_value=req.key1_value,
                                start=req.start,
                                length=int(gather_idx.size),
                        )
                        available = section_override.shape[0]
                        window_end = req.start + int(gather_idx.size)
                        if available == int(gather_idx.size):
                                pass
                        elif available >= window_end:
                                section_override = section_override[req.start:window_end]
                        else:
                                raise ValueError('Pipeline tap shorter than requested gather window')
                except PipelineTapNotFoundError as exc:
                        raise HTTPException(status_code=409, detail=str(exc)) from exc
                except (TypeError, ValueError) as exc:
                        raise HTTPException(status_code=409, detail=str(exc)) from exc
        job_id = str(uuid4())
        job_state: dict[str, object] = {'status': 'queued', 'cache_key': cache_key}
        if section_override is not None:
                job_state['section_override'] = section_override
        jobs[job_id] = job_state

        req2 = req.copy(update={'offset_byte': forced_offset_byte})

        if cache_key in fbpick_cache:
                jobs[job_id]['status'] = 'done'
        else:
                threading.Thread(
                        target=_run_fbpick_job, args=(job_id, req2), daemon=True
                ).start()
        if req.used_deprecated_idx:
                response.headers['Warning'] = WARNING_DEPRECATED_IDX
        return {'job_id': job_id, 'status': jobs[job_id]['status']}


@router.get('/fbpick_job_status')
def fbpick_job_status(job_id: str = Query(...)):
        job = jobs.get(job_id)
        if job is None:
                raise HTTPException(status_code=404, detail='Job ID not found')
        return {'status': job.get('status', 'unknown'), 'message': job.get('message', '')}


@router.get('/get_fbpick_section_bin')
def get_fbpick_section_bin(job_id: str = Query(...)):
        job = jobs.get(job_id)
        if job is None or job.get('status') != 'done':
                raise HTTPException(status_code=404, detail='Result not ready')
        cache_key = job.get('cache_key')
        payload = fbpick_cache.get(cache_key)
        if payload is None:
                raise HTTPException(status_code=404, detail='Result missing')
        return Response(
                payload,
                media_type='application/octet-stream',
                headers={'Content-Encoding': 'gzip'},
        )

