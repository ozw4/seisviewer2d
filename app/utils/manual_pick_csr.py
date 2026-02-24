"""Manual-pick CSR conversion helpers."""

from __future__ import annotations

import numpy as np


def build_csr_from_lists(
    picks_by_trace: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR arrays from per-trace pick lists."""
    n_traces = len(picks_by_trace)
    indptr = np.zeros((n_traces + 1,), dtype=np.int64)
    flat: list[np.ndarray] = []
    nnz = 0
    for trace_idx, picks in enumerate(picks_by_trace):
        arr = np.asarray(picks, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError('Each trace pick list must be 1D')
        flat.append(arr)
        nnz += int(arr.size)
        indptr[trace_idx + 1] = nnz
    if nnz == 0:
        data = np.empty((0,), dtype=np.int64)
    else:
        data = np.concatenate(flat).astype(np.int64, copy=False)
    return indptr, data


def lists_from_csr(
    indptr: np.ndarray, data: np.ndarray, n_traces: int
) -> list[list[int]]:
    """Expand CSR arrays into per-trace pick lists."""
    n_traces_int, indptr_arr, data_arr = _validated_csr_arrays(indptr, data, n_traces)

    out: list[list[int]] = []
    for trace_idx in range(n_traces_int):
        start = int(indptr_arr[trace_idx])
        stop = int(indptr_arr[trace_idx + 1])
        out.append([int(v) for v in data_arr[start:stop]])
    return out


def picks_time_s_to_lists(
    picks_time_s: np.ndarray, *, dt: float, n_samples: int
) -> list[list[int]]:
    """Convert 1D picks_time_s into per-trace sample-index lists."""
    if dt <= 0:
        raise ValueError('dt must be > 0')
    n_samples_int = int(n_samples)
    if n_samples_int <= 1:
        raise ValueError('n_samples must be > 1')

    picks_arr = np.asarray(picks_time_s, dtype=np.float64)
    if picks_arr.ndim != 1:
        raise ValueError('picks_time_s must be 1D')

    out: list[list[int]] = []
    for value in picks_arr:
        if not np.isfinite(value):
            out.append([])
            continue
        idx = int(np.rint(float(value) / float(dt)))
        if idx <= 0 or idx >= n_samples_int:
            out.append([])
            continue
        out.append([idx])
    return out


def picks_time_s_to_csr(
    picks_time_s: np.ndarray, *, dt: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1D picks_time_s into CSR arrays."""
    return build_csr_from_lists(
        picks_time_s_to_lists(picks_time_s, dt=dt, n_samples=n_samples)
    )


def empty_csr(n_traces: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an empty CSR structure for ``n_traces`` rows."""
    n_traces_int = int(n_traces)
    if n_traces_int < 0:
        raise ValueError('n_traces must be >= 0')
    return np.zeros((n_traces_int + 1,), dtype=np.int64), np.empty((0,), dtype=np.int64)


def csr_to_single_pick_times(
    indptr: np.ndarray,
    data: np.ndarray,
    *,
    n_traces: int,
    dt: float,
    n_samples: int,
) -> tuple[np.ndarray, int]:
    """Convert CSR picks into 1D pick times with one pick per trace.

    When a trace has multiple valid picks, the minimum sample index is selected.
    """
    if dt <= 0:
        raise ValueError('dt must be > 0')
    n_traces_int, indptr_arr, data_arr = _validated_csr_arrays(indptr, data, n_traces)
    dt_float = float(dt)
    n_samples_int = int(n_samples)
    if n_samples_int <= 1:
        raise ValueError('n_samples must be > 1')

    picks_time_s = np.full((n_traces_int,), np.nan, dtype=np.float32)
    traces_with_multiple = 0
    for trace_idx in range(n_traces_int):
        start = int(indptr_arr[trace_idx])
        stop = int(indptr_arr[trace_idx + 1])
        trace_data = data_arr[start:stop]
        if trace_data.size == 0:
            continue
        valid_mask = (trace_data >= 1) & (trace_data < n_samples_int)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count == 0:
            continue
        if valid_count > 1:
            traces_with_multiple += 1
        pick_idx = int(trace_data[valid_mask].min())
        picks_time_s[trace_idx] = np.float32(pick_idx * dt_float)
    return picks_time_s, traces_with_multiple


def _validated_csr_arrays(
    indptr: np.ndarray, data: np.ndarray, n_traces: int
) -> tuple[int, np.ndarray, np.ndarray]:
    n_traces_int = int(n_traces)
    if n_traces_int < 0:
        raise ValueError('n_traces must be >= 0')

    indptr_raw = np.asarray(indptr)
    data_raw = np.asarray(data)
    if indptr_raw.ndim != 1:
        raise ValueError('indptr must be 1D')
    if data_raw.ndim != 1:
        raise ValueError('data must be 1D')
    if not np.issubdtype(indptr_raw.dtype, np.integer):
        raise ValueError('indptr must be integer dtype')
    if not np.issubdtype(data_raw.dtype, np.integer):
        raise ValueError('data must be integer dtype')

    indptr_arr = indptr_raw.astype(np.int64, copy=False)
    data_arr = data_raw.astype(np.int64, copy=False)
    if indptr_arr.shape[0] != (n_traces_int + 1):
        raise ValueError('indptr length mismatch')
    if indptr_arr.size and int(indptr_arr[0]) != 0:
        raise ValueError('indptr[0] must be 0')
    if np.any(indptr_arr < 0):
        raise ValueError('indptr must be non-negative')
    if np.any(indptr_arr[1:] < indptr_arr[:-1]):
        raise ValueError('indptr must be monotonic')
    if indptr_arr.size and int(indptr_arr[-1]) != int(data_arr.shape[0]):
        raise ValueError('indptr[-1] must match data length')

    return n_traces_int, indptr_arr, data_arr


__all__ = [
    'build_csr_from_lists',
    'csr_to_single_pick_times',
    'empty_csr',
    'lists_from_csr',
    'picks_time_s_to_csr',
    'picks_time_s_to_lists',
]
