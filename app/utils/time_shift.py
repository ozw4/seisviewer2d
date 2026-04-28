"""Trace-wise time-shift utilities."""

from __future__ import annotations

import numpy as np


def shift_traces_linear(
    traces: np.ndarray,
    shifts_s: np.ndarray,
    dt: float,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply per-trace event-time shifts with linear interpolation."""
    arr = np.asarray(traces)
    if arr.ndim != 2:
        raise ValueError('traces must be a 2D array')

    n_traces, n_samples = arr.shape
    if n_traces <= 0:
        raise ValueError('traces must contain at least one trace')
    if n_samples <= 0:
        raise ValueError('traces must contain at least one sample')

    shifts = np.asarray(shifts_s, dtype=np.float64)
    if shifts.shape != (n_traces,):
        raise ValueError('shifts_s shape must match trace count')

    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError('dt must be finite and greater than 0')
    if not np.all(np.isfinite(shifts)):
        raise ValueError('shifts_s must contain only finite values')

    fill = float(fill_value)
    if not np.isfinite(fill):
        raise ValueError('fill_value must be finite')

    arr_f32 = np.asarray(arr, dtype=np.float32)
    out = np.full((n_traces, n_samples), fill, dtype=np.float32)
    sample_indices = np.arange(n_samples, dtype=np.float64)
    max_source = float(n_samples - 1)

    for trace_idx, shift_s in enumerate(shifts):
        source = sample_indices - (float(shift_s) / dt_value)
        valid = (source >= 0.0) & (source <= max_source)
        if not np.any(valid):
            continue

        source_valid = source[valid]
        lo = np.floor(source_valid).astype(np.int64)
        hi = np.minimum(lo + 1, n_samples - 1)
        frac = (source_valid - lo).astype(np.float32, copy=False)
        row = arr_f32[trace_idx]
        out[trace_idx, valid] = row[lo] * (1.0 - frac) + row[hi] * frac

    return np.ascontiguousarray(out, dtype=np.float32)


__all__ = ['shift_traces_linear']
