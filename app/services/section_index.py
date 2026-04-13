"""Section-index helpers used by picks/export flows."""

from __future__ import annotations

import numpy as np

from app.core.state import AppState
from app.services.errors import ConflictError
from app.services.reader import get_reader


def get_ntraces_for(
    file_id: str,
    key1_byte: int,
    key2_byte: int = 193,
    *,
    state: AppState,
) -> int:
    """Return total number of traces for ``file_id`` using a concrete reader."""
    reader = get_reader(file_id, int(key1_byte), int(key2_byte), state=state)

    ntraces = getattr(reader, 'ntraces', None)
    if ntraces is None:
        meta = getattr(reader, 'meta', None)
        if isinstance(meta, dict):
            ntraces = meta.get('n_traces')
    if ntraces is None and hasattr(reader, 'traces'):
        ntraces = getattr(reader.traces, 'shape', (None,))[0]
    if ntraces is None and hasattr(reader, 'key1s'):
        ntraces = len(reader.key1s)
    if ntraces is None:
        raise ConflictError('Reader cannot provide number of traces')
    return int(ntraces)


def get_trace_seq_for_value(
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int = 193,
    *,
    state: AppState,
) -> np.ndarray:
    """Return display-aligned trace sequence for ``key1`` of ``file_id``."""
    reader = get_reader(file_id, int(key1_byte), int(key2_byte), state=state)
    getter = getattr(reader, 'get_trace_seq_for_value', None)
    if not callable(getter):
        raise ConflictError('Reader cannot provide trace sequence information')
    seq = getter(int(key1), align_to='display')
    return np.asarray(seq, dtype=np.int64)


__all__ = ['get_ntraces_for', 'get_trace_seq_for_value']
