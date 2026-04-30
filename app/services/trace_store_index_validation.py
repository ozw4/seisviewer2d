"""Validation helpers for TraceStore index metadata."""

from __future__ import annotations

import numpy as np


def validate_sorted_to_original(
    values: np.ndarray,
    *,
    expected_n_traces: int,
    role: str,
) -> np.ndarray:
    """Return ``sorted_to_original`` as int64 after permutation validation."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{role} sorted_to_original must be 1-dimensional'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = f'{role} sorted_to_original must have an integer dtype'
        raise ValueError(msg)

    expected_shape = (int(expected_n_traces),)
    if arr.shape != expected_shape:
        msg = (
            f'{role} sorted_to_original shape mismatch: '
            f'expected {expected_shape}, got {arr.shape}'
        )
        raise ValueError(msg)

    out = np.ascontiguousarray(arr, dtype=np.int64)
    if out.size == 0:
        return out

    if int(out.min()) < 0 or int(out.max()) >= int(expected_n_traces):
        msg = (
            f'{role} sorted_to_original contains indices outside 0..'
            f'{int(expected_n_traces) - 1}'
        )
        raise ValueError(msg)

    expected = np.arange(int(expected_n_traces), dtype=np.int64)
    if not np.array_equal(np.sort(out), expected):
        msg = (
            f'{role} sorted_to_original must be a permutation of '
            f'0..{int(expected_n_traces) - 1}'
        )
        raise ValueError(msg)

    return out


__all__ = ['validate_sorted_to_original']
