"""Manual pick CRUD and export endpoints."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask

from app.api._helpers import get_state, reject_legacy_key1_query_params
from app.services.reader import get_reader
from app.services.registry import _filename_for_file_id
from app.services.section_index import get_ntraces_for, get_trace_seq_for_value
from app.utils.manual_pick_csr import (
    csr_to_single_pick_times,
    empty_csr,
    picks_time_s_to_csr,
)
from app.utils.pick_cache_file1d_mem import (
    clear_by_traceseq,
    clear_section,
    load_all,
    open_for_write,
    set_by_traceseq,
    to_pairs_for_section,
)

router = APIRouter()
logger = logging.getLogger(__name__)
_GRSTAT_FIXED_KEY1_BYTE = 9
_NUMBA_CACHE_DIR = '/tmp/numba'


def _validated_n_samples(reader) -> int:
    n_samples_raw = reader.get_n_samples()
    if not isinstance(n_samples_raw, (int, np.integer)):
        raise HTTPException(status_code=409, detail='Invalid or missing n_samples')
    n_samples = int(n_samples_raw)
    if n_samples <= 0:
        raise HTTPException(status_code=409, detail='Invalid or missing n_samples')
    return n_samples


def _validated_dt(state, file_id: str) -> float:
    dt_raw = state.file_registry.get_dt(file_id)
    if (
        not isinstance(dt_raw, (int, float, np.integer, np.floating))
        or float(dt_raw) <= 0
    ):
        raise HTTPException(
            status_code=409,
            detail='Invalid or missing sample interval (dt) for file',
        )
    return float(dt_raw)


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


class _PreparedManualPickExport(NamedTuple):
    file_name: str
    reader: object
    n_traces: int
    n_samples: int
    dt: float
    p_sorted: np.ndarray
    sorted_to_original: np.ndarray
    p_orig: np.ndarray


def _prepare_manual_pick_export(
    request: Request,
    *,
    file_id: str,
    key1_byte: int,
    key2_byte: int,
) -> _PreparedManualPickExport:
    state = get_state(request.app)
    file_name = _filename_for_file_id(file_id, file_registry=state.file_registry)
    if not file_name:
        raise HTTPException(status_code=404, detail='Filename not found for file_id')
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    n_traces = get_ntraces_for(file_id, key1_byte, key2_byte, state=state)
    n_samples = _validated_n_samples(reader)
    dt = _validated_dt(state, file_id)
    p_sorted = load_all(file_name, n_traces)
    sorted_to_original = _validated_sorted_to_original(reader, n_traces)
    p_orig = np.full((n_traces,), np.nan, dtype=np.float32)
    p_orig[sorted_to_original] = p_sorted
    return _PreparedManualPickExport(
        file_name=file_name,
        reader=reader,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        p_sorted=p_sorted,
        sorted_to_original=sorted_to_original,
        p_orig=p_orig,
    )


def _load_numpy2fbcrd():
    os.environ.setdefault('NUMBA_CACHE_DIR', _NUMBA_CACHE_DIR)
    try:
        from seisai_pick.pickio.io_grstat import numpy2fbcrd
    except Exception as exc:  # noqa: BLE001
        logger.exception('Failed to import numpy2fbcrd for grstat txt export')
        raise HTTPException(
            status_code=500,
            detail='Failed to load grstat exporter dependency',
        ) from exc
    return numpy2fbcrd


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
    fname = _filename_for_file_id(file_id, file_registry=state.file_registry)
    if not fname:
        raise HTTPException(404, 'file_id not found')
    ntr = get_ntraces_for(file_id, key1_byte, key2_byte, state=state)
    sec_map = get_trace_seq_for_value(file_id, key1, key1_byte, key2_byte, state=state)
    picks = await asyncio.to_thread(to_pairs_for_section, fname, ntr, sec_map)
    return {'picks': picks}


@router.post('/picks')
async def post_pick(m: PickPostModel, request: Request) -> dict[str, str]:
    state = get_state(request.app)
    fname = _filename_for_file_id(m.file_id, file_registry=state.file_registry)
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
    prepared = _prepare_manual_pick_export(
        request,
        file_id=file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    p_indptr, p_data = picks_time_s_to_csr(
        prepared.p_orig,
        dt=prepared.dt,
        n_samples=prepared.n_samples,
    )
    s_indptr, s_data = empty_csr(prepared.n_traces)

    payload: dict[str, object] = {
        'manual_pick_format': np.asarray('seisai_csr'),
        'picks_time_s': prepared.p_orig,
        'n_traces': np.int64(prepared.n_traces),
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
        'n_samples': np.int64(prepared.n_samples),
        'dt': np.float64(prepared.dt),
        'sorted_to_original': prepared.sorted_to_original.astype(np.int64, copy=False),
        'format_version': np.int64(1),
        'exported_at': np.asarray(datetime.now(timezone.utc).isoformat()),
        'export_app': np.asarray('seisviewer2d'),
        'source': np.asarray(str(prepared.file_name)),
        'source_hint': np.asarray(str(prepared.file_name)),
    }
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
        np.savez(tmp, **payload)
        tmp_path = Path(tmp.name)

    safe_base = re.sub(r'[^-_.a-zA-Z0-9]', '_', Path(prepared.file_name).stem) or 'file'
    download_name = f'manual_picks_time_v1_{safe_base}.npz'
    background = BackgroundTask(lambda path=tmp_path: path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type='application/octet-stream',
        filename=download_name,
        background=background,
    )


@router.get('/export_manual_picks_grstat_txt')
async def export_manual_picks_grstat_txt(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key2_byte: Annotated[int, Query()] = 193,
) -> FileResponse:
    prepared = _prepare_manual_pick_export(
        request,
        file_id=file_id,
        key1_byte=_GRSTAT_FIXED_KEY1_BYTE,
        key2_byte=key2_byte,
    )

    ffid_values = np.asarray(prepared.reader.get_key1_values(), dtype=np.int64)
    if ffid_values.ndim != 1:
        raise HTTPException(status_code=409, detail='Invalid FFID header values')
    if ffid_values.size == 0:
        raise HTTPException(status_code=409, detail='No FFID values available')

    trace_seq_rows: list[np.ndarray] = []
    max_channels = 0
    for rec_no in ffid_values:
        try:
            trace_seq = np.asarray(
                prepared.reader.get_trace_seq_for_value(
                    int(rec_no), align_to='display'
                ),
                dtype=np.int64,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if trace_seq.ndim != 1:
            raise HTTPException(status_code=409, detail='Invalid trace index shape')
        if np.any(trace_seq < 0) or np.any(trace_seq >= prepared.n_traces):
            raise HTTPException(status_code=409, detail='Trace index out of range')
        trace_seq_rows.append(trace_seq)
        max_channels = max(max_channels, int(trace_seq.size))

    if max_channels <= 0:
        raise HTTPException(
            status_code=409, detail='No traces available for FFID export'
        )

    fbnum = np.zeros((int(ffid_values.size), max_channels), dtype=np.float32)
    for row_idx, trace_seq in enumerate(trace_seq_rows):
        if trace_seq.size == 0:
            continue
        trace_times = np.asarray(prepared.p_sorted[trace_seq], dtype=np.float64)
        valid_mask = np.isfinite(trace_times) & (trace_times > 0.0)
        samples = np.zeros((trace_seq.size,), dtype=np.float32)
        if np.any(valid_mask):
            samples_valid = (trace_times[valid_mask] / float(prepared.dt)).astype(
                np.float32, copy=False
            )
            # grstat writer treats 0 as no-pick and emits sentinel (-9999).
            out_of_range = samples_valid >= float(prepared.n_samples)
            if np.any(out_of_range):
                logger.warning(
                    'grstat export: samples over n_samples converted to no-pick: count=%d n_samples=%d',
                    int(np.count_nonzero(out_of_range)),
                    int(prepared.n_samples),
                )
                samples_valid[out_of_range] = np.float32(0.0)
            samples[valid_mask] = samples_valid
        fbnum[row_idx, : trace_seq.size] = samples

    numpy2fbcrd = _load_numpy2fbcrd()
    dt_ms = float(prepared.dt) * 1000.0
    gather_numbers = [int(v) for v in ffid_values.tolist()]

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
        tmp_path = Path(tmp.name)

    try:
        await asyncio.to_thread(
            numpy2fbcrd,
            dt=dt_ms,
            fbnum=fbnum,
            gather_range=gather_numbers,
            output_name=str(tmp_path),
            header_comment='manual first-break picks exported by seisviewer2d',
        )
    except ValueError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409, detail=f'grstat export failed: {exc}'
        ) from exc
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        logger.exception('Unexpected grstat txt export failure')
        raise HTTPException(
            status_code=500,
            detail=f'grstat export failed: {type(exc).__name__}',
        ) from exc

    safe_base = re.sub(r'[^-_.a-zA-Z0-9]', '_', Path(prepared.file_name).stem) or 'file'
    download_name = f'manual_picks_grstat_v1_{safe_base}.txt'
    background = BackgroundTask(lambda path=tmp_path: path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type='text/plain',
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
    state = get_state(request.app)
    file_name = _filename_for_file_id(file_id, file_registry=state.file_registry)
    if not file_name:
        raise HTTPException(status_code=404, detail='Filename not found for file_id')
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    n_traces = int(get_ntraces_for(file_id, key1_byte, key2_byte, state=state))
    n_samples = _validated_n_samples(reader)
    dt = _validated_dt(state, file_id)

    blob = await file.read()
    try:
        npz = np.load(io.BytesIO(blob), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f'Invalid npz: {exc}') from exc

    with npz:
        has_p_indptr = 'p_indptr' in npz.files
        has_p_data = 'p_data' in npz.files
        if has_p_indptr or has_p_data:
            if not has_p_indptr:
                raise HTTPException(status_code=400, detail='Missing key: p_indptr')
            if not has_p_data:
                raise HTTPException(status_code=400, detail='Missing key: p_data')
            if 'n_traces' not in npz.files:
                raise HTTPException(status_code=400, detail='Missing key: n_traces')
            n_traces_npz = int(np.asarray(npz['n_traces']).item())
            if n_traces_npz != n_traces:
                raise HTTPException(status_code=409, detail='n_traces mismatch')
            if 'n_samples' in npz.files:
                n_samples_npz = int(np.asarray(npz['n_samples']).item())
                if n_samples_npz != n_samples:
                    raise HTTPException(status_code=409, detail='n_samples mismatch')
            if 'dt' in npz.files:
                dt_npz = float(np.asarray(npz['dt']).item())
                if abs(dt_npz - dt) > 1e-9:
                    raise HTTPException(status_code=409, detail='dt mismatch')
            try:
                p, traces_with_multiple = csr_to_single_pick_times(
                    npz['p_indptr'],
                    npz['p_data'],
                    n_traces=n_traces,
                    dt=dt,
                    n_samples=n_samples,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f'Invalid CSR: {exc}'
                ) from exc
            if traces_with_multiple > 0:
                logger.warning(
                    'Manual-pick CSR import: 複数pickがあるため最小indexを採用: traces=%d',
                    traces_with_multiple,
                )
            dropped_neg = 0
            clamped_hi = 0
        else:
            for key in ('picks_time_s', 'n_traces', 'n_samples', 'dt'):
                if key not in npz.files:
                    raise HTTPException(status_code=400, detail=f'Missing key: {key}')
            picks_time_s = npz['picks_time_s']
            if picks_time_s.ndim != 1:
                raise HTTPException(status_code=400, detail='picks_time_s must be 1D')
            if picks_time_s.shape[0] != n_traces:
                raise HTTPException(
                    status_code=409, detail='picks_time_s length mismatch'
                )
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
    fname = _filename_for_file_id(file_id, file_registry=state.file_registry)
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
