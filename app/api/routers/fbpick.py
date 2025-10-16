"""First-break inference endpoints."""

from __future__ import annotations

import gzip
import threading
from typing import Any
from uuid import uuid4

import msgpack
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from app.api._helpers import (
        OFFSET_BYTE_FIXED,
        USE_FBPICK_OFFSET,
        PipelineTapNotFoundError,
        _maybe_attach_fbpick_offsets,
        fbpick_cache,
        get_reader,
        get_section_from_pipeline_tap,
        jobs,
)
from app.api.schemas import PipelineSpec
from app.utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from app.utils.pipeline import apply_pipeline
from app.utils.utils import (
        SegySectionReader,
        TraceStoreSectionReader,
        quantize_float32,
)

router = APIRouter()


class FbpickRequest(BaseModel):
        file_id: str
        # The actual key1 header value specifying the section to process.  This
        # replaces the previous key1_idx index-based parameter.  When calling
        # the fbpick endpoints from the UI or tests, supply the actual header
        # value obtained from the key1 values array (e.g. key1Values[idx]).
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


def _run_fbpick_job(job_id: str, req: FbpickRequest) -> None:
        job = jobs[job_id]
        job['status'] = 'running'
        try:
                forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

                cache_key = job['cache_key']
                section_override = job.pop('section_override', None)
                reader: SegySectionReader | TraceStoreSectionReader | None = None
                if USE_FBPICK_OFFSET and forced_offset_byte is not None:
                        reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
                if section_override is not None:
                        section = np.asarray(section_override, dtype=np.float32)
                elif req.pipeline_key and req.tap_label:
                        # For pipeline taps, fetch the section using the header value
                        section = get_section_from_pipeline_tap(
                                file_id=req.file_id,
                                key1=req.key1,
                                key1_byte=req.key1_byte,
                                pipeline_key=req.pipeline_key,
                                tap_label=req.tap_label,
                                offset_byte=forced_offset_byte,
                        )
                        section = np.asarray(section, dtype=np.float32)
                else:
                        if reader is None:
                                reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
                        # Use the header value directly to get the section
                        section = np.asarray(reader.get_section(req.key1), dtype=np.float32)
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
                                key1=req.key1,
                                offset_byte=forced_offset_byte,
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
def fbpick_section_bin(req: FbpickRequest):
        if not FBPICK_MODEL_PATH.exists():
                raise HTTPException(status_code=409, detail='FB pick model weights not found')

        forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

        pipeline_key = req.pipeline_key
        tap_label = req.tap_label
        # Use the header value instead of the index as part of the cache key
        cache_key = (
                req.file_id,
                req.key1,
                req.key1_byte,
                req.key2_byte,
                forced_offset_byte,
                req.tile_h,
                req.tile_w,
                req.overlap,
                bool(req.amp),
                pipeline_key,
                tap_label,
                'fbpick',
        )
        section_override: np.ndarray | None = None
        wants_pipeline = bool(pipeline_key and tap_label)
        if cache_key not in fbpick_cache and wants_pipeline:
                try:
                        # Fetch the section for the tap using the header value
                        section_override = get_section_from_pipeline_tap(
                                file_id=req.file_id,
                                key1=req.key1,
                                key1_byte=req.key1_byte,
                                pipeline_key=pipeline_key,
                                tap_label=tap_label,
                                offset_byte=forced_offset_byte,
                        )
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

