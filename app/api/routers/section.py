"""Section retrieval and binary I/O endpoints."""

from __future__ import annotations

import gzip
from typing import Any

import msgpack
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from app.api._helpers import (
        EXPECTED_SECTION_NDIM,
        OFFSET_BYTE_FIXED,
        USE_FBPICK_OFFSET,
        PipelineTapNotFoundError,
        get_reader,
        get_section_from_pipeline_tap,
        window_section_cache,
)
from app.utils.segy_meta import get_dt_for_file
from app.utils.utils import quantize_float32

router = APIRouter()


@router.get('/get_key1_values')
def get_key1_values(
        file_id: str = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
):
        reader = get_reader(file_id, key1_byte, key2_byte)
        return JSONResponse(content={'values': reader.get_key1_values().tolist()})


@router.get('/get_section')
def get_section(
        file_id: str = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
        key1_idx: int = Query(...),
):
        try:
                reader = get_reader(file_id, key1_byte, key2_byte)
                section = reader.get_section(key1_idx)
                return JSONResponse(content={'section': section})
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e


@router.get('/get_section_bin')
def get_section_bin(
        file_id: str = Query(...),
        key1_idx: int = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
):
        try:
                reader = get_reader(file_id, key1_byte, key2_byte)
                section = np.array(reader.get_section(key1_idx), dtype=np.float32)
                scale, q = quantize_float32(section)
                obj = {
                        'scale': scale,
                        'shape': q.shape,
                        'data': q.tobytes(),
                        'dt': get_dt_for_file(file_id),
                }
                payload = msgpack.packb(obj)
                return Response(
                        gzip.compress(payload),
                        media_type='application/octet-stream',
                        headers={'Content-Encoding': 'gzip'},
                )
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e


@router.get('/get_section_window_bin')
def get_section_window_bin(
        file_id: str = Query(...),
        key1_idx: int = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
        offset_byte: int | None = Query(None),
        x0: int = Query(...),
        x1: int = Query(...),
        y0: int = Query(...),
        y1: int = Query(...),
        step_x: int = Query(1, ge=1),
        step_y: int = Query(1, ge=1),
        pipeline_key: str | None = Query(None),
        tap_label: str | None = Query(None),
):
        forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else offset_byte

        cache_key = (
                file_id,
                key1_idx,
                key1_byte,
                key2_byte,
                forced_offset_byte,
                x0,
                x1,
                y0,
                y1,
                step_x,
                step_y,
                pipeline_key,
                tap_label,
        )

        cached_payload = window_section_cache.get(cache_key)
        if cached_payload is not None:
                return Response(
                        cached_payload,
                        media_type='application/octet-stream',
                        headers={'Content-Encoding': 'gzip'},
                )

        try:
                if pipeline_key and tap_label:
                        section = get_section_from_pipeline_tap(
                                file_id=file_id,
                                key1_idx=key1_idx,
                                key1_byte=key1_byte,
                                pipeline_key=pipeline_key,
                                tap_label=tap_label,
                                offset_byte=forced_offset_byte,
                        )
                else:
                        reader = get_reader(file_id, key1_byte, key2_byte)
                        section = np.array(reader.get_section(key1_idx), dtype=np.float32)
        except PipelineTapNotFoundError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        section = np.ascontiguousarray(section, dtype=np.float32)
        if section.ndim != EXPECTED_SECTION_NDIM:
                raise HTTPException(status_code=500, detail='Section data must be 2D')

        n_traces, n_samples = section.shape
        if not (0 <= x0 <= x1 < n_traces):
                raise HTTPException(status_code=400, detail='Trace range out of bounds')
        if not (0 <= y0 <= y1 < n_samples):
                raise HTTPException(status_code=400, detail='Sample range out of bounds')
        if step_x < 1 or step_y < 1:
                raise HTTPException(status_code=400, detail='Steps must be >= 1')

        sub = section[x0 : x1 + 1 : step_x, y0 : y1 + 1 : step_y]
        if sub.size == 0:
                raise HTTPException(status_code=400, detail='Requested window is empty')

        window_view = np.ascontiguousarray(sub.T, dtype=np.float32)
        scale, q = quantize_float32(window_view)
        obj: dict[str, Any] = {
                'scale': scale,
                'shape': window_view.shape,
                'data': q.tobytes(),
                'dt': get_dt_for_file(file_id),
        }
        payload = msgpack.packb(obj)
        compressed = gzip.compress(payload)
        window_section_cache.set(cache_key, compressed)
        return Response(
                compressed,
                media_type='application/octet-stream',
                headers={'Content-Encoding': 'gzip'},
        )

