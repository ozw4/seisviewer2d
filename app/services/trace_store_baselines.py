"""Reusable helpers for TraceStore raw baseline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.baseline_artifacts import (
    build_raw_baseline_payload,
    build_trace_spans_by_key1,
    write_raw_baseline_artifacts,
)

ZERO_STD_EPS = 1e-12


@dataclass(frozen=True)
class RawBaselineStats:
    mu_traces: np.ndarray
    sigma_traces: np.ndarray
    zero_var_mask: np.ndarray
    mu_sections: np.ndarray
    sigma_sections: np.ndarray
    trace_spans_by_key1: dict[str, list[list[int]]]


def _ensure_1d_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        msg = f'{name} must be 1-dimensional'
        raise ValueError(msg)
    return arr


def _ensure_finite_f64(name: str, value: np.ndarray) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must contain finite numeric values'
        raise ValueError(msg) from exc
    if not np.all(np.isfinite(arr)):
        msg = f'{name} must contain finite values'
        raise ValueError(msg)
    return arr


def _ensure_index_i64(name: str, value: np.ndarray) -> np.ndarray:
    arr = _ensure_1d_array(name, value)
    if np.issubdtype(arr.dtype, np.floating):
        arr_f64 = _ensure_finite_f64(name, arr)
        if not np.all(arr_f64 == np.trunc(arr_f64)):
            msg = f'{name} must contain integer values'
            raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = f'{name} must have an integer dtype'
        raise ValueError(msg)
    return np.ascontiguousarray(arr, dtype=np.int64)


def _validate_section_spans(
    *,
    key1_offsets: np.ndarray,
    key1_counts: np.ndarray,
    n_traces: int,
) -> None:
    if np.any(key1_offsets < 0):
        msg = 'key1_offsets must not contain negative values'
        raise ValueError(msg)
    if np.any(key1_counts <= 0):
        msg = 'key1_counts must contain positive values'
        raise ValueError(msg)
    if key1_offsets.size > 1 and np.any(np.diff(key1_offsets) <= 0):
        msg = 'key1_offsets must be ascending'
        raise ValueError(msg)

    expected_offsets = np.empty_like(key1_offsets)
    expected_offsets[0] = 0
    if key1_offsets.size > 1:
        expected_offsets[1:] = key1_offsets[:-1] + key1_counts[:-1]
    if not np.array_equal(key1_offsets, expected_offsets):
        msg = 'section spans must be contiguous'
        raise ValueError(msg)

    last_end = int(key1_offsets[-1] + key1_counts[-1])
    if last_end != int(n_traces):
        msg = 'section spans must cover all traces'
        raise ValueError(msg)


def compute_raw_baseline_stats(
    *,
    key1_values: np.ndarray,
    key1_offsets: np.ndarray,
    key1_counts: np.ndarray,
    trace_sum: np.ndarray,
    trace_sumsq: np.ndarray,
    n_samples: int,
) -> RawBaselineStats:
    if n_samples <= 0:
        msg = 'n_samples must be positive'
        raise ValueError(msg)

    trace_sum_arr = _ensure_1d_array('trace_sum', trace_sum)
    trace_sumsq_arr = _ensure_1d_array('trace_sumsq', trace_sumsq)
    if trace_sum_arr.shape != trace_sumsq_arr.shape:
        msg = 'trace_sum and trace_sumsq must have matching shapes'
        raise ValueError(msg)
    if trace_sum_arr.size == 0:
        msg = 'trace_sum and trace_sumsq must not be empty'
        raise ValueError(msg)
    trace_sum_f64 = _ensure_finite_f64('trace_sum', trace_sum_arr)
    trace_sumsq_f64 = _ensure_finite_f64('trace_sumsq', trace_sumsq_arr)

    key1_values_i64 = _ensure_index_i64('key1_values', key1_values)
    key1_offsets_i64 = _ensure_index_i64('key1_offsets', key1_offsets)
    key1_counts_i64 = _ensure_index_i64('key1_counts', key1_counts)
    if (
        key1_values_i64.shape != key1_offsets_i64.shape
        or key1_values_i64.shape != key1_counts_i64.shape
    ):
        msg = 'key1_values, key1_offsets, and key1_counts must have matching shapes'
        raise ValueError(msg)
    if key1_values_i64.size == 0:
        msg = 'key1 sections must not be empty'
        raise ValueError(msg)

    _validate_section_spans(
        key1_offsets=key1_offsets_i64,
        key1_counts=key1_counts_i64,
        n_traces=int(trace_sum_f64.size),
    )

    n_samples_f64 = float(n_samples)
    mu_traces = trace_sum_f64 / n_samples_f64
    trace_var = np.maximum(
        (trace_sumsq_f64 / n_samples_f64) - np.square(mu_traces),
        0.0,
    )
    sigma_traces = np.sqrt(trace_var)
    zero_var_mask = sigma_traces <= ZERO_STD_EPS
    if zero_var_mask.any():
        sigma_traces = sigma_traces.copy()
        sigma_traces[zero_var_mask] = 1.0

    section_sum = np.add.reduceat(trace_sum_f64, key1_offsets_i64)
    section_sumsq = np.add.reduceat(trace_sumsq_f64, key1_offsets_i64)
    total_samples = key1_counts_i64.astype(np.float64, copy=False) * n_samples_f64
    mu_sections = section_sum / total_samples
    section_var = np.maximum(
        (section_sumsq / total_samples) - np.square(mu_sections),
        0.0,
    )
    sigma_sections = np.sqrt(section_var)

    return RawBaselineStats(
        mu_traces=np.asarray(mu_traces, dtype=np.float32),
        sigma_traces=np.asarray(sigma_traces, dtype=np.float32),
        zero_var_mask=np.asarray(zero_var_mask, dtype=bool),
        mu_sections=np.asarray(mu_sections, dtype=np.float32),
        sigma_sections=np.asarray(sigma_sections, dtype=np.float32),
        trace_spans_by_key1=build_trace_spans_by_key1(
            key1_values_i64,
            key1_offsets_i64,
            key1_counts_i64,
        ),
    )


def write_trace_store_raw_baseline_artifacts(
    *,
    store_path: str | Path,
    key1_byte: int,
    key2_byte: int,
    dtype_base: str,
    dt: float | None,
    key1_values: np.ndarray,
    key1_offsets: np.ndarray,
    key1_counts: np.ndarray,
    trace_sum: np.ndarray,
    trace_sumsq: np.ndarray,
    n_samples: int,
    source_sha256: str | None,
) -> dict[str, Any]:
    stats = compute_raw_baseline_stats(
        key1_values=key1_values,
        key1_offsets=key1_offsets,
        key1_counts=key1_counts,
        trace_sum=trace_sum,
        trace_sumsq=trace_sumsq,
        n_samples=n_samples,
    )
    payload = build_raw_baseline_payload(
        dtype_base=dtype_base,
        dt=dt,
        key1_values=key1_values,
        mu_sections=stats.mu_sections,
        sigma_sections=stats.sigma_sections,
        mu_traces=stats.mu_traces,
        sigma_traces=stats.sigma_traces,
        zero_var_mask=stats.zero_var_mask,
        trace_spans_by_key1=stats.trace_spans_by_key1,
        source_sha256=source_sha256,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        serialize_arrays=False,
    )
    write_raw_baseline_artifacts(
        store_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        payload=payload,
    )
    return payload


__all__ = [
    'RawBaselineStats',
    'ZERO_STD_EPS',
    'compute_raw_baseline_stats',
    'write_trace_store_raw_baseline_artifacts',
]
