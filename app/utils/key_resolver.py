from __future__ import annotations

from typing import Any

import numpy as np


def ensure_key1_values(reader) -> np.ndarray:
    """Ensure the key1 header is materialized on the given reader and return it
    as a 1D NumPy array in Trseq order. This avoids races with async header preparation.
    """
    # If the reader exposes ensure_header/key1_byte, call it once to materialize.
    ensure = getattr(reader, "ensure_header", None)
    if callable(ensure):
        ensure(getattr(reader, "key1_byte"))
    vals = np.asarray(reader.get_header(reader.key1_byte))
    if vals.ndim != 1:
        raise ValueError("key1 header must be a 1D array")
    return vals


def resolve_indices_slice_on_demand(
    reader,
    key1_value: Any,
    start: int,
    length: int,
) -> np.ndarray:
    """Return absolute trace indices for the virtual gather defined by key1_value,
    sliced as [start : start + length). Uses np.flatnonzero each call.
    - No caching, no registry edits, no precomputed maps.
    - Indices are returned in ascending order (Trseq).

    Raises:
      KeyError   if the value does not exist in the header.
      ValueError if the requested window is out of range.

    """
    vals = ensure_key1_values(reader)
    idx_all = np.flatnonzero(vals == key1_value)  # equality is fine if UI echoes exact value
    total = idx_all.size
    if total == 0:
        raise KeyError(f"unknown key1_value: {key1_value}")
    if length < 1 or start < 0 or start >= total:
        raise ValueError(f"range out of gather: start={start}, len={length}, total={total}")
    end = min(total, start + length)
    return idx_all[start:end]


def indices_to_runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """(Optional helper for future phases)
    Convert absolute indices into continuous runs [(s, e), ...] with e exclusive.
    Useful when batching memmap reads to minimize random I/O.
    """
    if idx.size == 0:
        return []
    d = np.diff(idx)
    cut = np.nonzero(d != 1)[0] + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [idx.size]))
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]
