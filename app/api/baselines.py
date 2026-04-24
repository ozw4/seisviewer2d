"""Baseline computation helpers for raw SEG-Y statistics."""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.api._helpers import get_state
from app.core.state import AppState
from app.services.file_registry import FileRegistry
from app.services.reader import get_reader
from app.utils.baseline_artifacts import (
    BASELINE_STAGE_RAW,
    LEGACY_BASELINE_FILENAME_RAW,
    build_legacy_baseline_path,
    build_raw_baseline_payload,
    merge_baseline_payload,
    read_split_baseline_payload,
    SplitBaselineArtifactsError,
    write_raw_baseline_artifacts,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

BASELINE_FILENAME_RAW = LEGACY_BASELINE_FILENAME_RAW
BASELINE_LOCK_NAME = '.baseline_raw.lock'
ZERO_STD_EPS = 1e-12
WAIT_FOR_LOCK_SECONDS = 5.0
BASELINE_SOURCE_PRECOMPUTED = 'precomputed'
BASELINE_SOURCE_LEGACY_JSON = 'legacy-json'
BASELINE_SOURCE_COMPUTED_FALLBACK = 'computed-fallback'


class BaselineComputationError(RuntimeError):
    """Raised when the raw baseline cannot be generated."""


@dataclass(frozen=True)
class _TraceStoreArtifacts:
    """Resolved trace-store artifacts used for baseline computation."""

    store_path: Path
    meta: dict[str, Any]
    key1_values: np.ndarray
    key1_offsets: np.ndarray
    key1_counts: np.ndarray


def _resolve_store_path(file_id: str, file_registry: FileRegistry) -> Path:
    """Return the trace-store directory for ``file_id``."""
    rec = file_registry.get_record(file_id)
    if not isinstance(rec, dict):
        raise BaselineComputationError('TraceStore metadata missing')
    store_path = rec.get('store_path')
    if not isinstance(store_path, str):
        raise BaselineComputationError('TraceStore path missing')
    path = Path(store_path)
    if not path.is_dir():
        raise BaselineComputationError('TraceStore directory not found')
    return path


def _load_meta(store_path: Path) -> dict[str, Any]:
    meta_path = store_path / 'meta.json'
    if not meta_path.is_file():
        raise BaselineComputationError('TraceStore meta.json missing')
    return json.loads(meta_path.read_text(encoding='utf-8'))


def _load_index_arrays(store_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_path = store_path / 'index.npz'
    if not index_path.is_file():
        raise BaselineComputationError('TraceStore index.npz missing')
    with np.load(index_path, allow_pickle=False) as index_data:
        key1_values = np.asarray(index_data['key1_values'], dtype=np.int64)
        key1_offsets = np.asarray(index_data['key1_offsets'], dtype=np.int64)
        key1_counts = np.asarray(index_data['key1_counts'], dtype=np.int64)
    return key1_values, key1_offsets, key1_counts


def _load_trace_store_artifacts(
    file_id: str, file_registry: FileRegistry
) -> _TraceStoreArtifacts:
    store_path = _resolve_store_path(file_id, file_registry)
    meta = _load_meta(store_path)
    key1_values, key1_offsets, key1_counts = _load_index_arrays(store_path)
    if key1_values.size != key1_counts.size:
        raise BaselineComputationError('TraceStore index arrays are inconsistent')
    return _TraceStoreArtifacts(
        store_path=store_path,
        meta=meta,
        key1_values=key1_values,
        key1_offsets=key1_offsets,
        key1_counts=key1_counts,
    )


def _resolve_key1_groups(
    *,
    reader: Any,
    traces: np.ndarray,
    key1_byte: int,
) -> tuple[np.ndarray, np.ndarray]:
    get_header = getattr(reader, 'get_header', None)
    if not callable(get_header):
        raise BaselineComputationError(
            'TraceStore reader cannot provide headers for requested key1 byte'
        )
    header = np.asarray(get_header(int(key1_byte)), dtype=np.int64)
    if header.ndim != 1:
        raise BaselineComputationError('TraceStore header array must be 1D')
    if header.shape[0] != traces.shape[0]:
        raise BaselineComputationError(
            'TraceStore header array does not match trace count'
        )
    key1_values, inverse = np.unique(header, return_inverse=True)
    key1_values = np.ascontiguousarray(key1_values, dtype=np.int64)
    inverse = inverse.astype(np.int64, copy=False)
    return key1_values, inverse


def _spans_from_inverse(
    key1_values: np.ndarray,
    inverse: np.ndarray,
    n_groups: int,
) -> dict[str, list[list[int]]]:
    if inverse.ndim != 1:
        raise BaselineComputationError('inverse must be 1D')
    if key1_values.ndim != 1:
        raise BaselineComputationError('key1_values must be 1D')
    if int(key1_values.size) != int(n_groups):
        raise BaselineComputationError('key1_values size does not match n_groups')
    N = int(inverse.shape[0])
    if N == 0:
        return {}
    spans: list[list[list[int]]] = [[] for _ in range(int(n_groups))]
    start = 0
    curr = int(inverse[0])
    for i in range(1, N + 1):
        if i == N or int(inverse[i]) != curr:
            spans[curr].append([start, i])
            if i < N:
                start = i
                curr = int(inverse[i])
    return {
        str(int(key1_values[group])): spans[group] for group in range(int(n_groups))
    }


def _load_json_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise BaselineComputationError(f'Corrupted baseline payload: {path}') from exc
    if not isinstance(payload, dict):
        raise BaselineComputationError(f'Baseline payload must be an object: {path}')
    return payload


def _baseline_payload_matches(
    payload: dict[str, Any],
    *,
    expected_sha: str | None,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> bool:
    if payload.get('stage') != BASELINE_STAGE_RAW:
        return False
    if payload.get('source_sha256') != expected_sha:
        return False
    stored_key1 = payload.get('key1_byte')
    stored_key2 = payload.get('key2_byte')
    if stored_key1 is None or stored_key2 is None:
        return False
    return int(stored_key1) == int(expected_key1_byte) and int(stored_key2) == int(
        expected_key2_byte
    )


def _strip_payload_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    slim = dict(payload)
    slim.pop('mu_traces', None)
    slim.pop('sigma_traces', None)
    slim.pop('zero_var_mask', None)
    return slim


def _json_ready_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return merge_baseline_payload(
        _strip_payload_arrays(payload),
        mu_traces=np.asarray(payload.get('mu_traces'), dtype=np.float32),
        sigma_traces=np.asarray(payload.get('sigma_traces'), dtype=np.float32),
        zero_var_mask=np.asarray(payload.get('zero_var_mask'), dtype=bool),
    )


def _load_split_baseline_if_valid(
    store_path: Path,
    *,
    expected_sha: str | None,
    expected_key1_byte: int,
    expected_key2_byte: int,
    include_arrays: bool,
) -> dict[str, Any] | None:
    try:
        resolved = read_split_baseline_payload(
            store_path,
            stage=BASELINE_STAGE_RAW,
            key1_byte=expected_key1_byte,
            key2_byte=expected_key2_byte,
            include_arrays=include_arrays,
        )
    except SplitBaselineArtifactsError:
        return None
    if resolved is None:
        return None
    payload, _cache_key = resolved
    if not _baseline_payload_matches(
        payload,
        expected_sha=expected_sha,
        expected_key1_byte=expected_key1_byte,
        expected_key2_byte=expected_key2_byte,
    ):
        return None
    if include_arrays:
        return payload
    return _strip_payload_arrays(payload)


def _load_legacy_baseline_if_valid(
    store_path: Path,
    *,
    expected_sha: str | None,
    expected_key1_byte: int,
    expected_key2_byte: int,
    include_arrays: bool,
) -> dict[str, Any] | None:
    baseline_path = build_legacy_baseline_path(store_path)
    if not baseline_path.is_file():
        return None
    payload = _load_json_payload(baseline_path)
    if not _baseline_payload_matches(
        payload,
        expected_sha=expected_sha,
        expected_key1_byte=expected_key1_byte,
        expected_key2_byte=expected_key2_byte,
    ):
        return None
    if include_arrays:
        return payload
    return _strip_payload_arrays(payload)


def _load_existing_baseline(
    store_path: Path,
    *,
    expected_sha: str | None,
    expected_key1_byte: int,
    expected_key2_byte: int,
    include_arrays: bool,
) -> tuple[dict[str, Any], str] | None:
    payload = _load_split_baseline_if_valid(
        store_path,
        expected_sha=expected_sha,
        expected_key1_byte=expected_key1_byte,
        expected_key2_byte=expected_key2_byte,
        include_arrays=include_arrays,
    )
    if payload is not None:
        return payload, BASELINE_SOURCE_PRECOMPUTED
    payload = _load_legacy_baseline_if_valid(
        store_path,
        expected_sha=expected_sha,
        expected_key1_byte=expected_key1_byte,
        expected_key2_byte=expected_key2_byte,
        include_arrays=include_arrays,
    )
    if payload is not None:
        return payload, BASELINE_SOURCE_LEGACY_JSON
    return None


def _acquire_lock_or_wait(store_path: Path) -> bool:
    lock_path = store_path / BASELINE_LOCK_NAME
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - start >= WAIT_FOR_LOCK_SECONDS:
                return False
            time.sleep(0.05)
            continue
        else:
            os.close(fd)
            return True


def _release_lock(store_path: Path) -> None:
    lock_path = store_path / BASELINE_LOCK_NAME
    with suppress(FileNotFoundError):
        lock_path.unlink()


def _compute_section_stats(
    *,
    traces: np.ndarray,
    inverse: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    if traces.ndim != 2:
        raise BaselineComputationError('Trace array must be 2D')
    if inverse.ndim != 1:
        raise BaselineComputationError('inverse must be 1D')
    if int(traces.shape[0]) != int(inverse.shape[0]):
        raise BaselineComputationError('inverse length does not match trace count')
    trace_sum = traces.sum(axis=1, dtype=np.float64)
    trace_sumsq = np.einsum('ij,ij->i', traces, traces, dtype=np.float64)
    group_sum = np.bincount(inverse, weights=trace_sum, minlength=int(n_groups))
    group_sumsq = np.bincount(inverse, weights=trace_sumsq, minlength=int(n_groups))
    n_samples = float(traces.shape[1])
    trace_counts = np.bincount(inverse, minlength=int(n_groups)).astype(np.float64)
    total_samples = trace_counts * n_samples
    mean = group_sum / total_samples
    mean_sq = group_sumsq / total_samples
    var = np.maximum(mean_sq - np.square(mean), 0.0)
    std = np.sqrt(var)
    return mean, std


def _compute_trace_stats(
    traces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = traces.mean(axis=1, dtype=np.float64)
    var = traces.var(axis=1, dtype=np.float64)
    std = np.sqrt(np.maximum(var, 0.0))
    zero_mask = std <= ZERO_STD_EPS
    if zero_mask.any():
        std = std.copy()
        std[zero_mask] = 1.0
    return mean, std, zero_mask


def _prepare_payload(
    *,
    file_id: str,
    file_registry: FileRegistry,
    artifacts: _TraceStoreArtifacts,
    key1_byte: int,
    key2_byte: int,
    key1_values: np.ndarray,
    trace_spans_by_key1: dict[str, list[list[int]]],
    mu_traces: np.ndarray,
    sigma_traces: np.ndarray,
    zero_mask: np.ndarray,
    mu_sections: np.ndarray,
    sigma_sections: np.ndarray,
) -> dict[str, Any]:
    meta = artifacts.meta
    source_sha = meta.get('source_sha256')
    dt_val = float(file_registry.get_dt(file_id))
    return build_raw_baseline_payload(
        dtype_base=str(meta.get('dtype', '')),
        dt=dt_val,
        key1_values=key1_values,
        mu_sections=mu_sections,
        sigma_sections=sigma_sections,
        mu_traces=mu_traces,
        sigma_traces=sigma_traces,
        zero_var_mask=zero_mask,
        trace_spans_by_key1=trace_spans_by_key1,
        source_sha256=(str(source_sha) if isinstance(source_sha, str) else None),
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        serialize_arrays=False,
    )


def _compute_baseline(
    *,
    file_id: str,
    file_registry: FileRegistry,
    artifacts: _TraceStoreArtifacts,
    key1_byte: int,
    key2_byte: int,
    state: AppState,
) -> dict[str, Any]:
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    traces = getattr(reader, 'traces', None)
    if not isinstance(traces, np.ndarray):
        raise BaselineComputationError('TraceStore reader did not expose traces array')
    if traces.ndim != 2:
        raise BaselineComputationError('Trace array must be 2D')
    key1_values, inverse = _resolve_key1_groups(
        reader=reader,
        traces=traces,
        key1_byte=key1_byte,
    )
    n_groups = int(key1_values.size)
    if n_groups == 0:
        raise BaselineComputationError('TraceStore returned no key1 groups')
    mu_traces, sigma_traces, zero_mask = _compute_trace_stats(traces)
    mu_sections, sigma_sections = _compute_section_stats(
        traces=traces,
        inverse=inverse,
        n_groups=n_groups,
    )
    if not np.all(np.isfinite(mu_traces)):
        raise BaselineComputationError('Per-trace mean produced non-finite values')
    if not np.all(np.isfinite(sigma_traces)):
        raise BaselineComputationError('Per-trace sigma produced non-finite values')
    if not np.all(np.isfinite(mu_sections)):
        raise BaselineComputationError('Section mean produced non-finite values')
    if not np.all(np.isfinite(sigma_sections)):
        raise BaselineComputationError('Section sigma produced non-finite values')
    trace_spans_by_key1 = _spans_from_inverse(key1_values, inverse, n_groups)
    return _prepare_payload(
        file_id=file_id,
        file_registry=file_registry,
        artifacts=artifacts,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        key1_values=key1_values,
        trace_spans_by_key1=trace_spans_by_key1,
        mu_traces=mu_traces,
        sigma_traces=sigma_traces,
        zero_mask=zero_mask,
        mu_sections=mu_sections,
        sigma_sections=sigma_sections,
    )


def get_or_create_raw_baseline(
    *,
    file_id: str,
    key1_byte: int,
    key2_byte: int,
    app: FastAPI,
    include_arrays: bool = True,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the cached raw baseline for ``file_id`` computing it if required."""
    state = get_state(app)
    artifacts = _load_trace_store_artifacts(file_id, state.file_registry)
    meta = artifacts.meta
    expected_sha = meta.get('source_sha256')
    existing = _load_existing_baseline(
        artifacts.store_path,
        expected_sha=expected_sha,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
        include_arrays=include_arrays,
    )
    if existing is not None:
        payload, source = existing
        if status is not None:
            status['source'] = source
        if include_arrays:
            return _json_ready_payload(payload)
        return payload
    lock_acquired = _acquire_lock_or_wait(artifacts.store_path)
    if not lock_acquired:
        existing = _load_existing_baseline(
            artifacts.store_path,
            expected_sha=expected_sha,
            expected_key1_byte=key1_byte,
            expected_key2_byte=key2_byte,
            include_arrays=include_arrays,
        )
        if existing is not None:
            payload, source = existing
            if status is not None:
                status['source'] = source
            if include_arrays:
                return _json_ready_payload(payload)
            return payload
        raise BaselineComputationError('Baseline computation is already in progress')
    try:
        cache_key = f'{file_id}_{int(key1_byte)}_{int(key2_byte)}'
        with state.lock:
            state.cached_readers.pop(cache_key, None)
        payload = _compute_baseline(
            file_id=file_id,
            file_registry=state.file_registry,
            artifacts=artifacts,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            state=state,
        )
        write_raw_baseline_artifacts(
            artifacts.store_path,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            payload=payload,
        )
        if status is not None:
            status['source'] = BASELINE_SOURCE_COMPUTED_FALLBACK
        if include_arrays:
            return _json_ready_payload(payload)
        return _strip_payload_arrays(payload)
    finally:
        _release_lock(artifacts.store_path)


__all__ = [
    'BASELINE_SOURCE_COMPUTED_FALLBACK',
    'BASELINE_SOURCE_LEGACY_JSON',
    'BASELINE_SOURCE_PRECOMPUTED',
    'BASELINE_STAGE_RAW',
    'BaselineComputationError',
    'get_or_create_raw_baseline',
]
