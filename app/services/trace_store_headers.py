"""Helpers for TraceStore header materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from app.services.trace_store_index_validation import validate_sorted_to_original


def _non_empty_path_value(
    derived: Mapping[str, Any],
    field: str,
) -> str | None:
    if field not in derived:
        return None
    value = derived[field]
    if not isinstance(value, str):
        msg = f'derived.{field} must be a string when used as a header source'
        raise ValueError(msg)
    if not value.strip():
        return None
    return value


def resolve_header_source_store_path(
    meta: Mapping[str, Any],
    *,
    target_store_dir: str | Path,
) -> Path | None:
    """Return the derived TraceStore header source path, if one is configured."""
    derived = meta.get('derived')
    if not isinstance(derived, Mapping):
        return None

    value = _non_empty_path_value(derived, 'header_source_store_path')
    if value is None:
        value = _non_empty_path_value(derived, 'from_store_path')
    if value is None:
        return None

    source_path = Path(value)
    if not source_path.is_absolute():
        msg = (
            'TraceStore header source path must be absolute: '
            f'{source_path!s} for {Path(target_store_dir)!s}'
        )
        raise ValueError(msg)
    return source_path


def _load_sorted_to_original(
    store_dir: Path,
    *,
    role: str,
    expected_n_traces: int,
) -> np.ndarray:
    index_path = store_dir / 'index.npz'
    if not index_path.exists():
        msg = f'{role} TraceStore index.npz is missing: {index_path}'
        raise ValueError(msg)

    with np.load(index_path, allow_pickle=False) as index:
        if 'sorted_to_original' not in index.files:
            msg = f'{role} TraceStore index.npz is missing sorted_to_original: {index_path}'
            raise ValueError(msg)
        return validate_sorted_to_original(
            index['sorted_to_original'],
            expected_n_traces=expected_n_traces,
            role=role,
        )


def validate_same_sorted_trace_order(
    *,
    target_store_dir: str | Path,
    source_store_dir: str | Path,
    expected_n_traces: int,
) -> None:
    """Validate that target and source stores use the same sorted trace order."""
    target_sorted = _load_sorted_to_original(
        Path(target_store_dir),
        role='target',
        expected_n_traces=expected_n_traces,
    )
    source_sorted = _load_sorted_to_original(
        Path(source_store_dir),
        role='source',
        expected_n_traces=expected_n_traces,
    )
    if not np.array_equal(target_sorted, source_sorted):
        msg = 'target and source sorted_to_original arrays do not match'
        raise ValueError(msg)


def write_header_array_atomic(
    *,
    store_dir: str | Path,
    header_byte: int,
    values: np.ndarray,
    expected_n_traces: int,
) -> np.ndarray:
    """Atomically write a validated int32 header array and return it as a memmap."""
    arr = np.asarray(values)
    expected_shape = (int(expected_n_traces),)
    if arr.ndim != 1:
        msg = f'header byte {header_byte} values must be 1-dimensional'
        raise ValueError(msg)
    if arr.shape != expected_shape:
        msg = (
            f'header byte {header_byte} shape mismatch: '
            f'expected {expected_shape}, got {arr.shape}'
        )
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = f'header byte {header_byte} values must have an integer dtype'
        raise ValueError(msg)

    int32_info = np.iinfo(np.int32)
    if arr.size:
        min_value = int(arr.min())
        max_value = int(arr.max())
        if min_value < int32_info.min or max_value > int32_info.max:
            msg = f'header byte {header_byte} values are outside the int32 range'
            raise ValueError(msg)

    out = np.ascontiguousarray(arr, dtype=np.int32)
    store_path = Path(store_dir)
    path = store_path / f'headers_byte_{header_byte}.npy'
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')

    try:
        with tmp_path.open('wb') as handle:
            np.save(handle, out)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return np.load(path, mmap_mode='r')


__all__ = [
    'resolve_header_source_store_path',
    'validate_same_sorted_trace_order',
    'write_header_array_atomic',
]
