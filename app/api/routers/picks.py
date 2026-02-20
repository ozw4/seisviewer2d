"""Manual pick CRUD and export endpoints."""

from __future__ import annotations

import asyncio
import io
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask

from app.api._helpers import get_state, reject_legacy_key1_query_params
from app.services.reader import get_reader
from app.services.registry import _filename_for_file_id
from app.services.section_index import get_ntraces_for, get_trace_seq_for_value
from app.utils.pick_cache_file1d_mem import (
    clear_by_traceseq,
    clear_section,
    load_all,
    open_for_write,
    set_by_traceseq,
    to_pairs_for_section,
)
from app.utils.segy_meta import get_dt_for_file

router = APIRouter()


def _validated_n_samples(reader) -> int:
    n_samples_raw = reader.get_n_samples()
    if not isinstance(n_samples_raw, (int, np.integer)):
        raise HTTPException(status_code=409, detail='Invalid or missing n_samples')
    n_samples = int(n_samples_raw)
    if n_samples <= 0:
        raise HTTPException(status_code=409, detail='Invalid or missing n_samples')
    return n_samples


def _validated_sorted_to_original(reader, n_traces: int) -> np.ndarray:
    try:
        sorted_to_original = np.asarray(reader.get_sorted_to_original(), dtype=np.int64)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if sorted_to_original.shape != (n_traces,):
        raise HTTPException(status_code=409, detail='sorted_to_original shape mismatch')
    if sorted_to_original.size:
        if (
            int(sorted_to_original.min()) < 0
            or int(sorted_to_original.max()) >= n_traces
        ):
            raise HTTPException(
                status_code=409, detail='sorted_to_original out of range'
            )
    return sorted_to_original


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


@router.get('/export_manual_picks_npz')
async def export_manual_picks_npz(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
) -> FileResponse:
    file_name = _filename_for_file_id(file_id)
    if not file_name:
        raise HTTPException(status_code=404, detail='Filename not found for file_id')
    state = get_state(request.app)
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    n_traces = get_ntraces_for(file_id, key1_byte, key2_byte, state=state)
    n_samples = _validated_n_samples(reader)
    dt_raw = get_dt_for_file(file_id)
    if (
        not isinstance(dt_raw, (int, float, np.integer, np.floating))
        or float(dt_raw) <= 0
    ):
        raise HTTPException(
            status_code=409,
            detail='Invalid or missing sample interval (dt) for file',
        )
    dt = float(dt_raw)
    p_sorted = load_all(file_name, n_traces)
    sorted_to_original = _validated_sorted_to_original(reader, n_traces)
    p_orig = np.full((n_traces,), np.nan, dtype=np.float32)
    p_orig[sorted_to_original] = p_sorted

    payload: dict[str, object] = {
        'picks_time_s': p_orig,
        'n_traces': np.int64(n_traces),
        'n_samples': np.int64(n_samples),
        'dt': np.float64(dt),
        'format_version': np.int64(1),
        'exported_at': np.asarray(datetime.now(timezone.utc).isoformat()),
        'export_app': np.asarray('seisviewer2d'),
        'source_hint': np.asarray(str(file_name)),
    }
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
        np.savez(tmp, **payload)
        tmp_path = Path(tmp.name)

    safe_base = re.sub(r'[^-_.a-zA-Z0-9]', '_', Path(file_name).stem) or 'file'
    download_name = f'manual_picks_time_v1_{safe_base}.npz'
    background = BackgroundTask(lambda path=tmp_path: path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type='application/octet-stream',
        filename=download_name,
        background=background,
    )


@router.post('/import_manual_picks_npz')
async def import_manual_picks_npz(
    request: Request,
    file_id: Annotated[str, Query(...)],
    file: Annotated[UploadFile, File(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
    mode: Annotated[str, Query()] = 'replace',
) -> dict[str, int | str]:
    if mode not in {'replace', 'merge'}:
        raise HTTPException(status_code=400, detail='mode must be replace or merge')
    file_name = _filename_for_file_id(file_id)
    if not file_name:
        raise HTTPException(status_code=404, detail='Filename not found for file_id')
    state = get_state(request.app)
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    n_traces = int(get_ntraces_for(file_id, key1_byte, key2_byte, state=state))
    n_samples = _validated_n_samples(reader)
    dt_raw = get_dt_for_file(file_id)
    if (
        not isinstance(dt_raw, (int, float, np.integer, np.floating))
        or float(dt_raw) <= 0
    ):
        raise HTTPException(
            status_code=409,
            detail='Invalid or missing sample interval (dt) for file',
        )
    dt = float(dt_raw)

    blob = await file.read()
    try:
        npz = np.load(io.BytesIO(blob), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f'Invalid npz: {exc}') from exc

    with npz:
        for key in ('picks_time_s', 'n_traces', 'n_samples', 'dt'):
            if key not in npz.files:
                raise HTTPException(status_code=400, detail=f'Missing key: {key}')
        picks_time_s = npz['picks_time_s']
        if picks_time_s.ndim != 1:
            raise HTTPException(status_code=400, detail='picks_time_s must be 1D')
        if picks_time_s.shape[0] != n_traces:
            raise HTTPException(status_code=409, detail='picks_time_s length mismatch')
        if not np.issubdtype(picks_time_s.dtype, np.floating):
            raise HTTPException(
                status_code=400, detail='picks_time_s must be float dtype'
            )
        n_traces_npz = int(np.asarray(npz['n_traces']).item())
        n_samples_npz = int(np.asarray(npz['n_samples']).item())
        dt_npz = float(np.asarray(npz['dt']).item())
        if n_traces_npz != n_traces:
            raise HTTPException(status_code=409, detail='n_traces mismatch')
        if n_samples_npz != n_samples:
            raise HTTPException(status_code=409, detail='n_samples mismatch')
        if abs(dt_npz - dt) > 1e-9:
            raise HTTPException(status_code=409, detail='dt mismatch')

        p = picks_time_s.astype(np.float32, copy=True)
        p[~np.isfinite(p)] = np.nan
        dropped_neg = int(np.count_nonzero(p < 0))
        p[p < 0] = np.nan
        tmax = np.float32((n_samples - 1) * dt)
        clamped_mask = p > tmax
        clamped_hi = int(np.count_nonzero(clamped_mask))
        p[clamped_mask] = tmax
        sorted_to_original = _validated_sorted_to_original(reader, n_traces)
        p_sorted = p[sorted_to_original]

        mm = open_for_write(file_name, n_traces)
        if mode == 'replace':
            mm[:] = p_sorted
            applied = int(n_traces)
        else:
            mask = ~np.isnan(p_sorted)
            mm[mask] = p_sorted[mask]
            applied = int(np.count_nonzero(mask))
        mm.flush()
        del mm

    return {
        'status': 'ok',
        'mode': mode,
        'applied': applied,
        'dropped_neg': dropped_neg,
        'clamped_hi': clamped_hi,
    }


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
