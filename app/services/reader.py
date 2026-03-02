"""Reader and section conversion helpers."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.core.state import AppState
from app.services.errors import ConflictError
from app.trace_store.reader import TraceStoreSectionReader

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
    with state.lock:
        reader = state.cached_readers.get(cache_key)
    if reader is None:
        store_path = state.file_registry.get_store_path(file_id)
        p = Path(store_path)
        if not p.is_dir():
            raise ConflictError('trace store not built')
        fresh_reader = TraceStoreSectionReader(p, key1_byte, key2_byte)
        with state.lock:
            reader = state.cached_readers.get(cache_key)
            if reader is None:
                state.cached_readers[cache_key] = fresh_reader
                reader = fresh_reader
    dt_val = state.file_registry.get_dt(file_id)
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
