"""Trace-order mapping helpers for refraction statics application code."""

from __future__ import annotations

import numpy as np


def original_to_sorted_position(
    sorted_to_original: np.ndarray,
) -> np.ndarray:
    """Return an inverse map from original trace ID to sorted position."""
    mapping = np.asarray(sorted_to_original)
    if mapping.ndim != 1:
        raise ValueError('sorted_to_original must be a 1D integer array')
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError('sorted_to_original must be a 1D integer array')

    sorted_to_original_i64 = np.ascontiguousarray(mapping, dtype=np.int64)
    n_traces = int(sorted_to_original_i64.shape[0])
    if np.any(sorted_to_original_i64 < 0) or np.any(sorted_to_original_i64 >= n_traces):
        raise ValueError(
            'sorted_to_original must be a complete permutation of 0..n_traces-1'
        )
    if np.unique(sorted_to_original_i64).shape[0] != n_traces:
        raise ValueError(
            'sorted_to_original must be a complete permutation of 0..n_traces-1'
        )

    original_to_sorted = np.empty(n_traces, dtype=np.int64)
    original_to_sorted[sorted_to_original_i64] = np.arange(n_traces, dtype=np.int64)
    return np.ascontiguousarray(original_to_sorted, dtype=np.int64)


def sorted_positions_for_original_trace_ids(
    *,
    sorted_to_original: np.ndarray,
    original_trace_id: np.ndarray,
) -> np.ndarray:
    """Map original trace IDs to positions in sorted trace order."""
    original = np.asarray(original_trace_id)
    if original.ndim != 1:
        raise ValueError('original_trace_id must be a 1D integer array')
    if not np.issubdtype(original.dtype, np.integer):
        raise ValueError('original_trace_id must be a 1D integer array')

    inverse = original_to_sorted_position(sorted_to_original)
    original_i64 = np.ascontiguousarray(original, dtype=np.int64)
    if np.any(original_i64 < 0) or np.any(original_i64 >= inverse.shape[0]):
        raise ValueError('original_trace_id contains out-of-range trace IDs')
    return np.ascontiguousarray(inverse[original_i64], dtype=np.int64)
