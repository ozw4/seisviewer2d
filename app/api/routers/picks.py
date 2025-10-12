"""Manual pick CRUD and export endpoints."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.api._helpers import _filename_for_file_id, get_reader
from app.utils import picks_by_name
from app.utils.picks import add_pick, delete_pick, list_picks, store
from app.utils.segy_meta import get_dt_for_file
from app.utils.utils import SegySectionReader, TraceStoreSectionReader

router = APIRouter()


class Pick(BaseModel):
        file_id: str
        trace: int
        time: float
        key1_idx: int
        key1_byte: int


@router.post('/picks')
async def post_pick(pick: Pick) -> dict[str, str]:
        add_pick(pick.file_id, pick.trace, pick.time, pick.key1_idx, pick.key1_byte)
        await asyncio.to_thread(store.save)
        fname = _filename_for_file_id(pick.file_id)
        if fname:
                try:
                        picks_by_name.add_pick(
                                fname,
                                pick.trace,
                                pick.time,
                                pick.key1_idx,
                                pick.key1_byte,
                        )
                        await asyncio.to_thread(picks_by_name.save)
                except Exception as e:  # noqa: BLE001
                        print(f'[picks mirror] filename save failed: {e!s}')
        return {'status': 'ok'}


@router.get('/picks')
async def get_pick(
        file_id: str = Query(...),
        key1_idx: int = Query(...),
        key1_byte: int = Query(...),
) -> dict[str, list[dict[str, int | float]]]:
        return {'picks': list_picks(file_id, key1_idx, key1_byte)}


@router.get('/picks/by-filename')
async def get_picks_by_filename(
        file_name: str = Query(...),
        key1_idx: int = Query(...),
        key1_byte: int = Query(...),
) -> dict[str, list[dict[str, int | float]]]:
        return {'picks': picks_by_name.list_picks(file_name, key1_idx, key1_byte)}


@router.get('/export_manual_picks_all_npy')
async def export_manual_picks_all_npy(
        file_id: str = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
) -> FileResponse:
        reader = get_reader(file_id, key1_byte, key2_byte)
        key1_values = reader.get_key1_values()
        if key1_values is None or len(key1_values) == 0:
                raise HTTPException(status_code=409, detail='No key1 values found for file')

        try:
                key1_list = [int(v) for v in np.asarray(key1_values).ravel()]
        except Exception:  # noqa: BLE001
                key1_list = [int(v) for v in key1_values]

        file_name = _filename_for_file_id(file_id)
        if not file_name:
                raise HTTPException(status_code=404, detail='Filename not found for file_id')

        key1_header: np.ndarray | None
        if isinstance(reader, SegySectionReader):
                key1_header = np.asarray(reader.key1s)
        elif isinstance(reader, TraceStoreSectionReader):
                key1_header = np.asarray(reader._get_header(reader.key1_byte))
        else:
                key1_header = None

        if key1_header is None:
                raise HTTPException(status_code=500, detail='Unable to determine trace counts for file')

        counts = [int(np.count_nonzero(key1_header == val)) for val in key1_list]
        width = max(counts, default=0)
        if width <= 0:
                raise HTTPException(status_code=409, detail='No traces found for provided key1 values')

        mat = np.full((len(key1_list), width), -1, dtype=np.int32)

        dt = get_dt_for_file(file_id)
        if not isinstance(dt, (int, float)) or dt <= 0:
                raise HTTPException(status_code=409, detail='Invalid or missing sample interval (dt) for file')
        dt = float(dt)

        n_samples: int | None = None
        if isinstance(reader, TraceStoreSectionReader):
                traces = getattr(reader, 'traces', None)
                if isinstance(traces, np.ndarray) and traces.ndim >= 2:
                        n_samples = int(traces.shape[-1])

        for i, key1_val in enumerate(key1_list):
                row_picks = picks_by_name.list_picks(
                        file_name,
                        key1_idx=key1_val,
                        key1_byte=key1_byte,
                )
                if not row_picks:
                        continue

                latest_by_trace: dict[int, float] = {}
                for pick in row_picks:
                        trace_val = pick.get('trace') if isinstance(pick, dict) else None
                        time_val = pick.get('time') if isinstance(pick, dict) else None
                        try:
                                trace_idx = int(trace_val)
                                time_sec = float(time_val)
                        except (TypeError, ValueError):
                                continue
                        if not np.isfinite(time_sec):
                                continue
                        latest_by_trace[trace_idx] = time_sec

                row_width = counts[i] if i < len(counts) else width
                if row_width <= 0 or not latest_by_trace:
                        continue

                for trace_idx, time_sec in latest_by_trace.items():
                        idx = int(round(time_sec / dt))
                        if n_samples is not None and n_samples > 0:
                                idx = max(0, min(idx, n_samples - 1))
                        if idx < 0:
                                continue
                        if trace_idx < 0 or trace_idx >= width or trace_idx >= row_width:
                                continue
                        mat[i, trace_idx] = idx

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        try:
                np.save(tmp, mat)
                tmp.flush()
                os.fsync(tmp.fileno())
        finally:
                tmp_path = Path(tmp.name)
                tmp.close()

        safe_base = re.sub(r'[^-_.a-zA-Z0-9]', '_', Path(file_name).stem) or 'file'
        download_name = f'pvec_idx_all_{safe_base}.npy'

        background = BackgroundTask(lambda path=tmp_path: path.unlink(missing_ok=True))
        return FileResponse(
                tmp_path,
                media_type='application/octet-stream',
                filename=download_name,
                background=background,
        )


@router.delete('/picks')
async def delete_pick_route(
        file_id: str = Query(...),
        trace: int | None = Query(None),
        key1_idx: int = Query(...),
        key1_byte: int = Query(...),
) -> dict[str, str]:
        delete_pick(file_id, trace, key1_idx, key1_byte)
        await asyncio.to_thread(store.save)
        fname = _filename_for_file_id(file_id)
        if fname:
                try:
                        picks_by_name.delete_pick(fname, trace, key1_idx, key1_byte)
                        await asyncio.to_thread(picks_by_name.save)
                except Exception as e:  # noqa: BLE001
                        print(f'[picks mirror] filename save failed: {e!s}')
        return {'status': 'ok'}

