"""Manual pick CRUD and export endpoints."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask

from app.api._helpers import reject_legacy_key1_query_params, get_state
from app.services.reader import get_reader
from app.services.registry import _filename_for_file_id
from app.services.section_index import get_ntraces_for, get_trace_seq_for_value
from app.utils.pick_cache_file1d_mem import (
    clear_all,
    clear_by_traceseq,
    clear_section,
    set_many_by_traceseq,
    set_by_traceseq,
    to_pairs_for_section,
)
from app.utils.segy_meta import get_dt_for_file

router = APIRouter()


def _read_scalar(npz: np.lib.npyio.NpzFile, key: str):
    if key not in npz.files:
        raise HTTPException(400, f'Missing required key in npz: {key}')
    arr = np.asarray(npz[key])
    if arr.shape != ():
        raise HTTPException(400, f'npz key {key} must be scalar')
    return arr.item()


class PickPostModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    file_id: str
    trace: int
    time: float
    key1: int
    key1_byte: int
    key2_byte: int = 193


@router.get('/picks', dependencies=[Depends(reject_legacy_key1_query_params)])
async def get_picks(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1: Annotated[int, Query(...)],
    key1_byte: Annotated[int, Query(...)],
    key2_byte: Annotated[int, Query()] = 193,
) -> dict[str, list[dict[str, int | float]]]:
    state = get_state(request.app)
    fname = _filename_for_file_id(file_id)
    if not fname:
        raise HTTPException(404, 'file_id not found')
    ntr = get_ntraces_for(file_id, key1_byte, key2_byte, state=state)
    sec_map = get_trace_seq_for_value(file_id, key1, key1_byte, key2_byte, state=state)
    picks = await asyncio.to_thread(to_pairs_for_section, fname, ntr, sec_map)
    return {'picks': picks}


@router.post('/picks')
async def post_pick(m: PickPostModel, request: Request) -> dict[str, str]:
    state = get_state(request.app)
    fname = _filename_for_file_id(m.file_id)
    if not fname:
        raise HTTPException(404, 'file_id not found')
    ntr = get_ntraces_for(m.file_id, m.key1_byte, m.key2_byte, state=state)
    sec_map = get_trace_seq_for_value(
        m.file_id, m.key1, m.key1_byte, m.key2_byte, state=state
    )
    if not (0 <= m.trace < sec_map.size):
        raise HTTPException(400, 'trace out of range for section')
    trace_seq = int(sec_map[m.trace])
    await asyncio.to_thread(set_by_traceseq, fname, ntr, trace_seq, float(m.time))
    return {'status': 'ok'}


@router.get('/export_manual_picks_all_npz')
async def export_manual_picks_all_npz(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
) -> FileResponse:
    state = get_state(request.app)
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    key1_values = reader.get_key1_values()
    if key1_values is None or len(key1_values) == 0:
        raise HTTPException(status_code=409, detail='No key1 values found for file')

    key1_list = [int(v) for v in np.asarray(key1_values).ravel()]

    file_name = _filename_for_file_id(file_id)
    if not file_name:
        raise HTTPException(status_code=404, detail='Filename not found for file_id')

    ntraces = get_ntraces_for(file_id, key1_byte, key2_byte, state=state)
    sec_maps: list[np.ndarray] = []
    counts: list[int] = []
    for key1 in key1_list:
        sec_map = get_trace_seq_for_value(
            file_id, key1, key1_byte, key2_byte, state=state
        )
        sec_maps.append(sec_map)
        counts.append(int(sec_map.size))

    width = max(counts, default=0)
    if width <= 0:
        raise HTTPException(
            status_code=409, detail='No traces found for provided key1 values'
        )

    mat = np.full((len(key1_list), width), -1, dtype=np.int32)

    dt = get_dt_for_file(file_id)
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise HTTPException(
            status_code=409, detail='Invalid or missing sample interval (dt) for file'
        )
    dt = float(dt)

    n_samples: int | None = None
    if hasattr(reader, 'get_n_samples'):
        n_samples_raw = reader.get_n_samples()
        if n_samples_raw is not None:
            n_samples = int(n_samples_raw)

    for i, sec_map in enumerate(sec_maps):
        row_picks = await asyncio.to_thread(
            to_pairs_for_section, file_name, ntraces, sec_map
        )
        if not row_picks:
            continue

        latest_by_trace: dict[int, float] = {}
        for pick in row_picks:
            trace_val = pick.get('trace') if isinstance(pick, dict) else None
            time_val = pick.get('time') if isinstance(pick, dict) else None
            if not isinstance(trace_val, (int, np.integer)):
                continue
            if not isinstance(time_val, (int, float, np.integer, np.floating)):
                continue
            trace_idx = int(trace_val)
            time_sec = float(time_val)
            if not np.isfinite(time_sec):
                continue
            latest_by_trace[trace_idx] = time_sec

        row_width = counts[i] if i < len(counts) else width
        if row_width <= 0 or not latest_by_trace:
            continue

        for trace_idx, time_sec in latest_by_trace.items():
            idx_val = round(time_sec / dt)
            if n_samples is not None and n_samples > 0:
                idx_val = max(0, min(idx_val, n_samples - 1))
            if idx_val < 0:
                continue
            if trace_idx < 0 or trace_idx >= width or trace_idx >= row_width:
                continue
            mat[i, trace_idx] = idx_val

    # ---- 一時ファイルは with 内で確実に書いて閉じる（SIM115対応 & 安定）
    source_sha256 = None
    meta = getattr(reader, 'meta', None)
    if isinstance(meta, dict):
        source_sha256 = meta.get('source_sha256')
    payload: dict[str, object] = {
        'picks_idx': mat,
        'key1_values': np.asarray(key1_list, dtype=np.int64),
        'dt': np.float64(dt),
        'key1_byte': np.int32(key1_byte),
        'key2_byte': np.int32(key2_byte),
        'file_id': np.asarray(str(file_id)),
    }
    if isinstance(source_sha256, str) and source_sha256:
        payload['source_sha256'] = np.asarray(source_sha256)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
        np.savez_compressed(tmp, **payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    safe_base = re.sub(r'[^-_.a-zA-Z0-9]', '_', Path(file_name).stem) or 'file'
    download_name = f'pvec_idx_all_{safe_base}.npz'

    background = BackgroundTask(lambda path=tmp_path: path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type='application/octet-stream',
        filename=download_name,
        background=background,
    )


@router.delete('/picks', dependencies=[Depends(reject_legacy_key1_query_params)])
async def delete_pick(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1: Annotated[int, Query(...)],
    key1_byte: Annotated[int, Query(...)],
    key2_byte: Annotated[int, Query()] = 193,
    trace: Annotated[int | None, Query()] = None,
) -> dict[str, str]:
    state = get_state(request.app)
    fname = _filename_for_file_id(file_id)
    if not fname:
        raise HTTPException(404, 'file_id not found')
    ntr = get_ntraces_for(file_id, key1_byte, key2_byte, state=state)
    sec_map = get_trace_seq_for_value(file_id, key1, key1_byte, key2_byte, state=state)

    if trace is None:
        await asyncio.to_thread(clear_section, fname, ntr, sec_map)
    else:
        if not (0 <= trace < sec_map.size):
            raise HTTPException(400, 'trace out of range for section')
        trace_seq = int(sec_map[trace])
        await asyncio.to_thread(clear_by_traceseq, fname, ntr, trace_seq)
    return {'status': 'ok'}


@router.post('/import_manual_picks_all_npz')
async def import_manual_picks_all_npz(
    request: Request,
    file_id: str,
    file: Annotated[UploadFile, File()],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
    mode: Annotated[str, Query()] = 'replace',
) -> dict[str, str | int]:
    if mode not in {'replace', 'merge'}:
        raise HTTPException(400, 'mode must be replace or merge')

    state = get_state(request.app)
    file_name = _filename_for_file_id(file_id)
    if not file_name:
        raise HTTPException(404, 'file_id not found')

    body = await file.read()
    if len(body) == 0:
        raise HTTPException(400, 'uploaded npz file is empty')

    with np.load(BytesIO(body), allow_pickle=False) as npz:
        if 'picks_idx' not in npz.files:
            raise HTTPException(400, 'Missing required key in npz: picks_idx')
        if 'key1_values' not in npz.files:
            raise HTTPException(400, 'Missing required key in npz: key1_values')

        picks_idx = np.asarray(npz['picks_idx'])
        key1_values = np.asarray(npz['key1_values'])
        npz_dt = _read_scalar(npz, 'dt')
        npz_key1_byte = _read_scalar(npz, 'key1_byte')
        npz_key2_byte = _read_scalar(npz, 'key2_byte')
        npz_file_id = _read_scalar(npz, 'file_id')

        if not isinstance(npz_file_id, str):
            raise HTTPException(400, 'npz file_id must be string scalar')
        if npz_file_id != file_id:
            raise HTTPException(409, 'file_id mismatch between query and npz')

        if not isinstance(npz_key1_byte, (int, np.integer)):
            raise HTTPException(400, 'npz key1_byte must be int scalar')
        if not isinstance(npz_key2_byte, (int, np.integer)):
            raise HTTPException(400, 'npz key2_byte must be int scalar')
        if int(npz_key1_byte) != key1_byte:
            raise HTTPException(409, 'key1_byte mismatch between query and npz')
        if int(npz_key2_byte) != key2_byte:
            raise HTTPException(409, 'key2_byte mismatch between query and npz')

        if not isinstance(npz_dt, (int, float, np.integer, np.floating)):
            raise HTTPException(400, 'npz dt must be numeric scalar')
        cur_dt_raw = get_dt_for_file(file_id)
        if not isinstance(cur_dt_raw, (int, float, np.integer, np.floating)):
            raise HTTPException(409, 'Current dt is missing or invalid for file')
        if abs(float(npz_dt) - float(cur_dt_raw)) > 1e-9:
            raise HTTPException(409, 'dt mismatch between current file and npz')
        dt = float(npz_dt)

        if picks_idx.ndim != 2:
            raise HTTPException(400, 'picks_idx must be a 2D integer array')
        if key1_values.ndim != 1:
            raise HTTPException(400, 'key1_values must be a 1D integer array')
        if not np.issubdtype(picks_idx.dtype, np.integer):
            raise HTTPException(400, 'picks_idx must be integer dtype')
        if not np.issubdtype(key1_values.dtype, np.integer):
            raise HTTPException(400, 'key1_values must be integer dtype')

        n_sections, width = int(picks_idx.shape[0]), int(picks_idx.shape[1])
        if key1_values.shape[0] != n_sections:
            raise HTTPException(409, 'picks_idx rows and key1_values length must match')

        reader = get_reader(file_id, key1_byte, key2_byte, state=state)
        cur_key1_values = reader.get_key1_values()
        if cur_key1_values is None or len(cur_key1_values) == 0:
            raise HTTPException(409, 'No key1 values found for current file')

        reader_meta = getattr(reader, 'meta', None)
        source_sha256_cur = None
        if isinstance(reader_meta, dict):
            source_sha256_cur = reader_meta.get('source_sha256')
        if 'source_sha256' in npz.files and source_sha256_cur is not None:
            source_sha256_npz = _read_scalar(npz, 'source_sha256')
            if not isinstance(source_sha256_npz, str):
                raise HTTPException(400, 'npz source_sha256 must be string scalar')
            if source_sha256_npz != source_sha256_cur:
                raise HTTPException(
                    409, 'source_sha256 mismatch between current file and npz'
                )

        ntraces = get_ntraces_for(file_id, key1_byte, key2_byte, state=state)

        sec_maps: list[np.ndarray] = []
        for val in key1_values:
            key1 = int(val)
            sec_map = get_trace_seq_for_value(
                file_id, key1, key1_byte, key2_byte, state=state
            )
            sec_maps.append(sec_map)
            if sec_map.size > width:
                raise HTTPException(
                    409,
                    f'import width is too small for key1={key1}: {width} < {sec_map.size}',
                )

    if mode == 'replace':
        await asyncio.to_thread(clear_all, file_name, ntraces)

    trace_seq_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    inserted = 0
    skipped_negative = 0
    picks_idx_i64 = np.asarray(picks_idx, dtype=np.int64)
    for i, sec_map in enumerate(sec_maps):
        sec_size = int(sec_map.size)
        if sec_size == 0:
            continue
        row_vals = picks_idx_i64[i, :sec_size]
        negative_mask = row_vals < 0
        skipped_negative += int(np.count_nonzero(negative_mask))
        keep_mask = row_vals >= 0
        if not np.any(keep_mask):
            continue
        trace_seq_chunks.append(np.asarray(sec_map[keep_mask], dtype=np.int64))
        time_chunks.append(np.asarray(row_vals[keep_mask] * dt, dtype=np.float32))

    if trace_seq_chunks:
        trace_seq_all = np.concatenate(trace_seq_chunks)
        time_all = np.concatenate(time_chunks)
        inserted = await asyncio.to_thread(
            set_many_by_traceseq, file_name, ntraces, trace_seq_all, time_all
        )

    return {
        'status': 'ok',
        'mode': mode,
        'sections': n_sections,
        'inserted': inserted,
        'skipped_negative': skipped_negative,
        'cleared': 'all' if mode == 'replace' else 0,
    }
