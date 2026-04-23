from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.paths import get_trace_store_dir, get_upload_dir

DEFAULT_ORIGINAL_NAME = 'perf_ci.sgy'
DEFAULT_KEY1_BYTE = 189
DEFAULT_KEY2_BYTE = 193
DEFAULT_DT_SECONDS = 0.002
N_SECTIONS = 4
TRACES_PER_SECTION = 320
N_SAMPLES = 1536
ZERO_STD_EPS = 1e-12


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    return int(raw)


def _compute_trace_spans_by_key1(
    key1_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[list[int]]]]:
    unique_values, offsets, counts = np.unique(
        key1_values.astype(np.int64, copy=False),
        return_index=True,
        return_counts=True,
    )
    trace_spans_by_key1 = {
        str(int(value)): [[int(offset), int(offset + count)]]
        for value, offset, count in zip(unique_values, offsets, counts, strict=True)
    }
    return (
        unique_values.astype(np.int32, copy=False),
        offsets.astype(np.int64, copy=False),
        counts.astype(np.int64, copy=False),
        trace_spans_by_key1,
    )


def _build_traces() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    section_idx = np.arange(N_SECTIONS, dtype=np.float32)[:, None, None]
    trace_idx = np.arange(TRACES_PER_SECTION, dtype=np.float32)[None, :, None]
    sample_idx = np.arange(N_SAMPLES, dtype=np.float32)[None, None, :]

    first_break = 140.0 + section_idx * 110.0 + trace_idx * 0.35
    secondary = first_break + 170.0 + section_idx * 12.0

    primary_wavelet = np.exp(-np.square((sample_idx - first_break) / 18.0))
    primary_wavelet *= np.sin((sample_idx - first_break) * 0.12)

    secondary_wavelet = np.exp(-np.square((sample_idx - secondary) / 34.0))
    secondary_wavelet *= np.sin((sample_idx - secondary) * 0.07)

    background = 0.08 * np.sin(sample_idx * 0.010 + section_idx * 0.55)
    background = background + 0.03 * np.cos(trace_idx * 0.18 - sample_idx * 0.004)
    background = background + 0.02 * np.sin((sample_idx + trace_idx * 1.7) * 0.015)

    gain = 1.0 + section_idx * 0.08 + trace_idx * 0.0005
    traces = gain * (primary_wavelet + 0.6 * secondary_wavelet) + background
    traces = traces.reshape(N_SECTIONS * TRACES_PER_SECTION, N_SAMPLES)

    key1_values = np.repeat(
        np.arange(1001, 1001 + N_SECTIONS, dtype=np.int32),
        TRACES_PER_SECTION,
    )
    key2_values = np.tile(
        np.arange(1, TRACES_PER_SECTION + 1, dtype=np.int32),
        N_SECTIONS,
    )
    return traces.astype(np.float32, copy=False), key1_values, key2_values


def _compute_baseline_payload(
    *,
    traces: np.ndarray,
    key1_values: np.ndarray,
    trace_spans_by_key1: dict[str, list[list[int]]],
    source_sha256: str,
    key1_byte: int,
    key2_byte: int,
) -> dict[str, object]:
    unique_key1, inverse = np.unique(
        key1_values.astype(np.int64, copy=False),
        return_inverse=True,
    )
    trace_mean = traces.mean(axis=1, dtype=np.float64)
    trace_std = np.sqrt(np.maximum(traces.var(axis=1, dtype=np.float64), 0.0))
    zero_mask = trace_std <= ZERO_STD_EPS
    if zero_mask.any():
        trace_std = trace_std.copy()
        trace_std[zero_mask] = 1.0

    trace_sum = traces.sum(axis=1, dtype=np.float64)
    trace_sumsq = np.einsum('ij,ij->i', traces, traces, dtype=np.float64)
    trace_counts = np.bincount(inverse, minlength=int(unique_key1.size)).astype(np.float64)
    total_samples = trace_counts * float(traces.shape[1])
    section_mean = np.bincount(
        inverse, weights=trace_sum, minlength=int(unique_key1.size)
    ) / total_samples
    section_mean_sq = np.bincount(
        inverse, weights=trace_sumsq, minlength=int(unique_key1.size)
    ) / total_samples
    section_std = np.sqrt(np.maximum(section_mean_sq - np.square(section_mean), 0.0))

    return {
        'stage': 'raw',
        'ddof': 0,
        'method': 'mean_std',
        'dtype_base': 'float32',
        'dt': DEFAULT_DT_SECONDS,
        'key1_values': unique_key1.astype(np.int64, copy=False).tolist(),
        'mu_section_by_key1': section_mean.astype(np.float32, copy=False).tolist(),
        'sigma_section_by_key1': section_std.astype(np.float32, copy=False).tolist(),
        'mu_traces': trace_mean.astype(np.float32, copy=False).tolist(),
        'sigma_traces': trace_std.astype(np.float32, copy=False).tolist(),
        'zero_var_mask': zero_mask.astype(bool, copy=False).tolist(),
        'trace_spans_by_key1': trace_spans_by_key1,
        'source_sha256': source_sha256,
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'key1_byte': int(key1_byte),
        'key2_byte': int(key2_byte),
    }


def main() -> None:
    original_name = os.environ.get('SV_PERF_ORIGINAL_NAME', DEFAULT_ORIGINAL_NAME).strip()
    key1_byte = _read_int_env('SV_PERF_KEY1_BYTE', DEFAULT_KEY1_BYTE)
    key2_byte = _read_int_env('SV_PERF_KEY2_BYTE', DEFAULT_KEY2_BYTE)

    upload_dir = get_upload_dir()
    trace_store_dir = get_trace_store_dir()
    raw_path = upload_dir / original_name
    store_dir = trace_store_dir / original_name

    upload_dir.mkdir(parents=True, exist_ok=True)
    trace_store_dir.mkdir(parents=True, exist_ok=True)

    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = b'SEISVIEWER2D_PERF_DATASET\n'
    raw_path.write_bytes(raw_bytes)
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    traces, key1_values, key2_values = _build_traces()
    unique_key1, key1_offsets, key1_counts, trace_spans_by_key1 = (
        _compute_trace_spans_by_key1(key1_values)
    )
    sorted_to_original = np.arange(traces.shape[0], dtype=np.int64)

    np.save(store_dir / 'traces.npy', traces)
    np.save(store_dir / f'headers_byte_{key1_byte}.npy', key1_values)
    np.save(store_dir / f'headers_byte_{key2_byte}.npy', key2_values)
    np.savez(
        store_dir / 'index.npz',
        key1_values=unique_key1,
        key1_offsets=key1_offsets,
        key1_counts=key1_counts,
        sorted_to_original=sorted_to_original,
    )

    meta = {
        'original_name': original_name,
        'original_segy_path': str(raw_path),
        'original_size': int(len(raw_bytes)),
        'source_sha256': source_sha256,
        'dtype': 'float32',
        'dt': DEFAULT_DT_SECONDS,
        'n_traces': int(traces.shape[0]),
        'n_samples': int(traces.shape[1]),
        'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
    }
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')

    baseline_payload = _compute_baseline_payload(
        traces=traces,
        key1_values=key1_values,
        trace_spans_by_key1=trace_spans_by_key1,
        source_sha256=source_sha256,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    (store_dir / 'baseline_raw.json').write_text(
        json.dumps(baseline_payload),
        encoding='utf-8',
    )

    output = {
        'raw_path': str(raw_path),
        'store_dir': str(store_dir),
        'original_name': original_name,
        'key1_byte': key1_byte,
        'key2_byte': key2_byte,
        'n_sections': N_SECTIONS,
        'traces_per_section': TRACES_PER_SECTION,
        'n_traces': int(traces.shape[0]),
        'n_samples': int(traces.shape[1]),
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
