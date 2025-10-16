"""Pipeline execution and tap retrieval endpoints."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Annotated, Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.api._helpers import (
        OFFSET_BYTE_FIXED,
        USE_FBPICK_OFFSET,
        _maybe_attach_fbpick_offsets,
        get_reader,
        jobs,
        load_section_by_indices,
        make_cache_key,
        pipeline_tap_cache,
)
from app.api.schemas import (
        PipelineSectionQuery,
        PipelineAllResponse,
        PipelineJobStatusResponse,
        PipelineSectionResponse,
        PipelineSpec,
)
from app.utils.pipeline import apply_pipeline, pipeline_key
from app.utils.utils import to_builtin
from app.utils.key_resolver import resolve_indices_slice_on_demand

router = APIRouter()

WARNING_DEPRECATED_IDX = '299 - key1_idx is deprecated; use key1_value'


class PipelineAllRequest(BaseModel):
        file_id: str
        key1_byte: int = 189
        key2_byte: int = 193
        offset_byte: int | None = None
        spec: PipelineSpec
        taps: list[str] = Field(default_factory=list)
        downsample_quicklook: bool = True


def _run_pipeline_all_job(job_id: str, req: PipelineAllRequest, pipe_key: str) -> None:
        job = jobs[job_id]
        job['status'] = 'running'
        try:
        reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
        key1_vals = reader.get_key1_values().tolist()
        total = len(key1_vals) or 1
        taps = req.taps
        for idx, key1_val in enumerate(key1_vals):
                gather_idx = resolve_indices_slice_on_demand(reader, key1_val, 0, 1_000_000_000)
                section = load_section_by_indices(reader, gather_idx)
                section = np.ascontiguousarray(section, dtype=np.float32)
                dt = 0.002
                if hasattr(reader, 'meta'):
                        dt = getattr(reader, 'meta', {}).get('dt', dt)
                meta = {'dt': dt}
                meta = _maybe_attach_fbpick_offsets(
                        meta,
                        spec=req.spec,
                        reader=reader,
                        key1_idx=None,
                        offset_byte=req.offset_byte,  # already forced by caller
                        indices=gather_idx,
                )
                out = apply_pipeline(section, spec=req.spec, meta=meta, taps=taps)
                for k, v in out.items():
                        val = v
                        if req.downsample_quicklook and isinstance(v, np.ndarray):
                                val = v[::4, ::4]
                        cache_spec = {
                                'kb1': req.key1_byte,
                                'pipe': pipe_key,
                                'offset': req.offset_byte,
                                'tap': k,
                                'window': None,
                        }
                        cache_key = make_cache_key(
                                req.file_id,
                                key1_val,
                                0,
                                int(gather_idx.size),
                                cache_spec,
                        )
                        pipeline_tap_cache.set(cache_key, to_builtin(val))
                job['progress'] = (idx + 1) / total
                job['status'] = 'done'
        except Exception as e:  # noqa: BLE001
                job['status'] = 'error'
                job['message'] = str(e)


@router.post('/pipeline/section', response_model=PipelineSectionResponse)
def pipeline_section(
        q: PipelineSectionQuery = Depends(),
        response: Response,
        spec: PipelineSpec = Body(...),
        taps: list[str] | None = Body(default=None),
        window: dict[str, int | float] | None = Body(default=None),
):
        reader = get_reader(q.file_id, q.key1_byte, q.key2_byte)
        effective_length = q.length if q.length is not None else 1_000_000_000
        try:
                idx = resolve_indices_slice_on_demand(
                        reader, q.key1_value, q.start, effective_length
                )
        except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        section = load_section_by_indices(reader, idx)
        section = np.ascontiguousarray(section, dtype=np.float32)
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
        dt = 0.002
        if hasattr(reader, 'meta'):
                dt = getattr(reader, 'meta', {}).get('dt', dt)
        meta = {'dt': dt}

        forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

        meta = _maybe_attach_fbpick_offsets(
                meta,
                spec=spec,
                reader=reader,
                key1_idx=None,
                offset_byte=forced_offset_byte,
                trace_slice=trace_slice,
                indices=idx,
        )
        pipe_key = pipeline_key(spec)
        tap_names = taps or []
        section_len = int(section.shape[0])
        base_spec = {
                'kb1': q.key1_byte,
                'pipe': pipe_key,
                'offset': forced_offset_byte,
                'window': window_hash,
        }
        if tap_names:
                taps_out: dict[str, Any] = {}
                misses: list[tuple[str, dict[str, object], str]] = []
                for tap in tap_names:
                        cache_spec = dict(base_spec, tap=tap)
                        cache_key = make_cache_key(
                                q.file_id,
                                q.key1_value,
                                q.start,
                                section_len,
                                cache_spec,
                        )
                        payload = pipeline_tap_cache.get(cache_key)
                        if payload is not None:
                                taps_out[tap] = payload
                        else:
                                misses.append((tap, cache_spec, cache_key))
                if misses:
                        missing_labels = [tap for tap, _, _ in misses]
                        out = apply_pipeline(section, spec=spec, meta=meta, taps=missing_labels)
                        for tap, cache_spec, cache_key in misses:
                                val = to_builtin(out[tap])
                                taps_out[tap] = val
                                pipeline_tap_cache.set(cache_key, val)
                if q.used_deprecated_idx:
                        response.headers['Warning'] = WARNING_DEPRECATED_IDX
                return {'taps': taps_out, 'pipeline_key': pipe_key}
        out = apply_pipeline(section, spec=spec, meta=meta, taps=None)
        if q.used_deprecated_idx:
                response.headers['Warning'] = WARNING_DEPRECATED_IDX
        return {'taps': to_builtin(out), 'pipeline_key': pipe_key}


@router.post('/pipeline/all', response_model=PipelineAllResponse)
def pipeline_all(
        file_id: str = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
        downsample_quicklook: Annotated[bool, Query()] = True,
        offset_byte: int | None = Query(None),
        spec: PipelineSpec = Body(...),
        taps: list[str] | None = Body(default=None),
):
        tap_names = taps or []

        forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

        req = PipelineAllRequest(
                file_id=file_id,
                key1_byte=key1_byte,
                key2_byte=key2_byte,
                offset_byte=forced_offset_byte,
                spec=spec,
                taps=tap_names,
                downsample_quicklook=downsample_quicklook,
        )
        job_id = str(uuid4())
        pipe_key = pipeline_key(spec)
        jobs[job_id] = {
                'status': 'queued',
                'progress': 0.0,
                'message': '',
                'file_id': file_id,
                'key1_byte': key1_byte,
                'pipeline_key': pipe_key,
                'offset_byte': forced_offset_byte,
        }
        threading.Thread(
                target=_run_pipeline_all_job, args=(job_id, req, pipe_key), daemon=True
        ).start()
        return {'job_id': job_id, 'state': jobs[job_id]['status']}


@router.get('/pipeline/job/{job_id}/status', response_model=PipelineJobStatusResponse)
def pipeline_job_status(job_id: str) -> PipelineJobStatusResponse:
        job = jobs.get(job_id)
        if job is None:
                raise HTTPException(status_code=404, detail='Job ID not found')
        return {
                'state': job.get('status', 'unknown'),
                'progress': job.get('progress', 0.0),
                'message': job.get('message', ''),
        }


@router.get('/pipeline/job/{job_id}/artifact', response_model=Any)
def pipeline_job_artifact(
        job_id: str,
        key1_idx: int = Query(...),
        tap: str = Query(...),
):
        job = jobs.get(job_id)
        if job is None:
                raise HTTPException(status_code=404, detail='Job ID not found')
        file_id = job.get('file_id')
        if not isinstance(file_id, str):
                raise HTTPException(status_code=404, detail='Job metadata incomplete')
        key1_byte = int(job.get('key1_byte', 189))
        pipe_key = job.get('pipeline_key')
        offset_byte = job.get('offset_byte')

        query = PipelineSectionQuery(
                file_id=file_id,
                key1_byte=key1_byte,
                key1_idx=key1_idx,
        )
        reader = get_reader(file_id, query.key1_byte, query.key2_byte)
        gather_idx = resolve_indices_slice_on_demand(reader, query.key1_value, query.start, 1_000_000_000)
        cache_spec = {
                'kb1': query.key1_byte,
                'pipe': pipe_key,
                'offset': offset_byte,
                'tap': tap,
                'window': None,
        }
        cache_key = make_cache_key(
                file_id,
                query.key1_value,
                query.start,
                int(gather_idx.size),
                cache_spec,
        )
        payload = pipeline_tap_cache.get(cache_key)
        if payload is None:
                raise HTTPException(status_code=404, detail='Artifact not ready')
        return payload

