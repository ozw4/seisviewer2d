"""Reader and section conversion helpers."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.core.state import AppState
from app.services.errors import ConflictError, NotFoundError
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file
from app.utils.utils import TraceStoreSectionReader

EXPECTED_SECTION_NDIM = 2


def coerce_section_f32(arr: NDArray, scale: float | None) -> NDArray[np.float32]:
    out = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    if not out.flags.writeable:
        out = out.copy()
    if scale is not None:
        if not out.flags.writeable:
            out = out.copy()
        if out.dtype != np.float32:
            out = out.astype(np.float32, copy=False)
        out *= float(scale)
    if not out.flags['C_CONTIGUOUS'] or out.dtype != np.float32:
        out = np.ascontiguousarray(out, dtype=np.float32)
    return out


def get_reader(
    file_id: str,
    key1_byte: int,
    key2_byte: int,
    *,
    state: AppState,
) -> TraceStoreSectionReader:
    cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
    reader = state.cached_readers.get(cache_key)
    if reader is None:
        rec = FILE_REGISTRY.get(file_id)
        if rec is None:
            raise NotFoundError('File ID not found')
        store_path = rec.get('store_path') if isinstance(rec, dict) else None
        if not isinstance(store_path, str):
            raise ConflictError('trace store not built')
        p = Path(store_path)
        if not p.is_dir():
            raise ConflictError('trace store not built')
        reader = TraceStoreSectionReader(p, key1_byte, key2_byte)
        state.cached_readers[cache_key] = reader
    dt_val = get_dt_for_file(file_id)
    meta_attr = getattr(reader, 'meta', None)
    if isinstance(meta_attr, dict):
        if not isinstance(meta_attr.get('dt'), (int, float)) or meta_attr['dt'] <= 0:
            meta_attr['dt'] = dt_val
    else:
        with suppress(Exception):
            reader.meta = {'dt': dt_val}
    return reader


def get_raw_section(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    state: AppState,
) -> np.ndarray:
    """Load the RAW seismic section as ``float32``."""
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    view = reader.get_section(key1)
    base = view.arr
    arr = coerce_section_f32(base, view.scale)
    if arr.ndim != EXPECTED_SECTION_NDIM:
        msg = f'Raw section expected 2D data, got {arr.ndim}D'
        raise ValueError(msg)
    return arr


__all__ = [
    'EXPECTED_SECTION_NDIM',
    'coerce_section_f32',
    'get_raw_section',
    'get_reader',
]
