"""Trace store reader registration helpers."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path

from app.core.state import AppState
from app.trace_store.reader import TraceStoreSectionReader


def trace_store_cache_key(file_id: str, key1_byte: int, key2_byte: int) -> str:
    return f'{file_id}_{key1_byte}_{key2_byte}'


def touch_trace_store_meta(store_dir: str | Path) -> None:
    meta_path = Path(store_dir) / 'meta.json'
    if not meta_path.exists():
        return
    meta_path.touch()


def register_trace_store(
    *,
    state: AppState,
    file_id: str,
    store_dir: str | Path,
    key1_byte: int,
    key2_byte: int,
    dt: float | None = None,
    update_registry: bool = True,
    touch_meta: bool = True,
    preload_header_bytes: Iterable[int] = (),
) -> TraceStoreSectionReader:
    store_path = Path(store_dir)
    reader = TraceStoreSectionReader(store_path, key1_byte, key2_byte)
    cache_key = trace_store_cache_key(file_id, key1_byte, key2_byte)
    with state.lock:
        state.cached_readers[cache_key] = reader

    threading.Thread(target=reader.preload_all_sections, daemon=True).start()

    header_bytes = dict.fromkeys((key1_byte, key2_byte, *preload_header_bytes))
    for header_byte in header_bytes:
        threading.Thread(
            target=reader.ensure_header,
            args=(header_byte,),
            daemon=True,
        ).start()

    if touch_meta:
        touch_trace_store_meta(store_path)
    if update_registry:
        state.file_registry.update(file_id, store_path=str(store_path), dt=dt)
    return reader


__all__ = [
    'register_trace_store',
    'touch_trace_store_meta',
    'trace_store_cache_key',
]
