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
from app.api.routers.section import get_ntraces_for, get_trace_seq_for
from app.utils.pick_cache_file1d_mem import (
    clear_by_traceseq,
    clear_section,
    set_by_traceseq,
    to_pairs_for_section,
)
from app.utils.segy_meta import get_dt_for_file
from app.utils.utils import TraceStoreSectionReader

router = APIRouter()


class PickPostModel(BaseModel):
    file_id: str
    trace: int
    time: float
    # The actual key1 header value specifying the section.
    key1: int
    key1_byte: int


@router.get("/picks")
async def get_picks(
    file_id: str = Query(...),
    key1: int = Query(...),
    key1_byte: int = Query(...),
) -> dict[str, list[dict[str, int | float]]]:
    fname = _filename_for_file_id(file_id)
    if not fname:
        raise HTTPException(404, "file_id not found")
    ntr = get_ntraces_for(file_id)
    # Use the header value directly when computing the trace sequence
    sec_map = get_trace_seq_for(file_id, key1, key1_byte)
    picks = await asyncio.to_thread(to_pairs_for_section, fname, ntr, sec_map)
    return {"picks": picks}


@router.post("/picks")
async def post_pick(m: PickPostModel) -> dict[str, str]:
    fname = _filename_for_file_id(m.file_id)
    if not fname:
        raise HTTPException(404, "file_id not found")
    ntr = get_ntraces_for(m.file_id)
    # Use the header value directly
    sec_map = get_trace_seq_for(m.file_id, m.key1, m.key1_byte)
    if not (0 <= m.trace < sec_map.size):
        raise HTTPException(400, "trace out of range for section")
    trace_seq = int(sec_map[m.trace])
    await asyncio.to_thread(set_by_traceseq, fname, ntr, trace_seq, float(m.time))
    return {"status": "ok"}


@router.get("/export_manual_picks_all_npy")
async def export_manual_picks_all_npy(
    file_id: str = Query(...),
    key1_byte: int = Query(189),
    key2_byte: int = Query(193),
) -> FileResponse:
    reader = get_reader(file_id, key1_byte, key2_byte)
    key1_values = reader.get_key1_values()
    if key1_values is None or len(key1_values) == 0:
        raise HTTPException(status_code=409, detail="No key1 values found for file")

    try:
        key1_list = [int(v) for v in np.asarray(key1_values).ravel()]
    except Exception:  # noqa: BLE001
        key1_list = [int(v) for v in key1_values]

    file_name = _filename_for_file_id(file_id)
    if not file_name:
        raise HTTPException(status_code=404, detail="Filename not found for file_id")

    ntraces = get_ntraces_for(file_id)
    sec_maps: list[np.ndarray] = []
    counts: list[int] = []
    # Compute trace sequence arrays for each key1 header value.
    for key1_val in key1_list:
        sec_map = get_trace_seq_for(file_id, key1_val, key1_byte)
        sec_maps.append(sec_map)
        counts.append(int(sec_map.size))

    width = max(counts, default=0)
    if width <= 0:
        raise HTTPException(status_code=409, detail="No traces found for provided key1 values")

    mat = np.full((len(key1_list), width), -1, dtype=np.int32)

    dt = get_dt_for_file(file_id)
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise HTTPException(status_code=409, detail="Invalid or missing sample interval (dt) for file")
    dt = float(dt)

    n_samples: int | None = None
    if isinstance(reader, TraceStoreSectionReader):
        traces = getattr(reader, "traces", None)
        if isinstance(traces, np.ndarray) and traces.ndim >= 2:
            n_samples = int(traces.shape[-1])

    for i, sec_map in enumerate(sec_maps):
        row_picks = await asyncio.to_thread(
            to_pairs_for_section, file_name, ntraces, sec_map
        )
        if not row_picks:
            continue

        latest_by_trace: dict[int, float] = {}
        for pick in row_picks:
            trace_val = pick.get("trace") if isinstance(pick, dict) else None
            time_val = pick.get("time") if isinstance(pick, dict) else None
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
            idx_val = int(round(time_sec / dt))
            if n_samples is not None and n_samples > 0:
                idx_val = max(0, min(idx_val, n_samples - 1))
            if idx_val < 0:
                continue
            if trace_idx < 0 or trace_idx >= width or trace_idx >= row_width:
                continue
            mat[i, trace_idx] = idx_val

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    try:
        np.save(tmp, mat)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp_path = Path(tmp.name)
        tmp.close()

    safe_base = re.sub(r"[^-_.a-zA-Z0-9]", "_", Path(file_name).stem) or "file"
    download_name = f"pvec_idx_all_{safe_base}.npy"

    background = BackgroundTask(lambda path=tmp_path: path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type="application/octet-stream",
        filename=download_name,
        background=background,
    )


@router.delete("/picks")
async def delete_pick(
    file_id: str = Query(...),
    trace: int | None = Query(None),
    key1: int = Query(...),
    key1_byte: int = Query(...),
) -> dict[str, str]:
    fname = _filename_for_file_id(file_id)
    if not fname:
        raise HTTPException(404, "file_id not found")
    ntr = get_ntraces_for(file_id)
    # Use the header value directly
    sec_map = get_trace_seq_for(file_id, key1, key1_byte)

    if trace is None:
        await asyncio.to_thread(clear_section, fname, ntr, sec_map)
    else:
        if not (0 <= trace < sec_map.size):
            raise HTTPException(400, "trace out of range for section")
        trace_seq = int(sec_map[trace])
        await asyncio.to_thread(clear_by_traceseq, fname, ntr, trace_seq)
    return {"status": "ok"}

