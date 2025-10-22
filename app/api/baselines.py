"""Baseline computation helpers for raw SEG-Y statistics."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.api._helpers import cached_readers, get_reader
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file

BASELINE_STAGE_RAW = 'raw'
BASELINE_FILENAME_RAW = 'baseline_raw.json'
BASELINE_LOCK_NAME = '.baseline_raw.lock'
ZERO_STD_EPS = 1e-12
WAIT_FOR_LOCK_SECONDS = 5.0


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


def _resolve_store_path(file_id: str) -> Path:
	"""Return the trace-store directory for ``file_id``."""
	rec = FILE_REGISTRY.get(file_id)
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


def _load_trace_store_artifacts(file_id: str) -> _TraceStoreArtifacts:
	store_path = _resolve_store_path(file_id)
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


def _resolve_key1_partition(
	*,
	artifacts: _TraceStoreArtifacts,
	reader: Any,
	traces: np.ndarray,
	key1_byte: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	meta_key1 = None
	meta = artifacts.meta
	if isinstance(meta, dict):
		key_bytes = meta.get('key_bytes')
		if isinstance(key_bytes, dict):
			meta_key1 = key_bytes.get('key1')
	if meta_key1 is not None and int(meta_key1) == int(key1_byte):
		return artifacts.key1_values, artifacts.key1_offsets, artifacts.key1_counts
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
	key1_values, key1_offsets, key1_counts = np.unique(
		header, return_index=True, return_counts=True
	)
	order = np.argsort(key1_offsets, kind='stable')
	key1_values = np.ascontiguousarray(key1_values[order], dtype=np.int64)
	key1_offsets = np.ascontiguousarray(key1_offsets[order], dtype=np.int64)
	key1_counts = np.ascontiguousarray(key1_counts[order], dtype=np.int64)
	return key1_values, key1_offsets, key1_counts


def _baseline_path(store_path: Path) -> Path:
	return store_path / BASELINE_FILENAME_RAW


def _load_baseline_if_valid(
	store_path: Path,
	*,
	expected_sha: str | None,
	expected_key1_byte: int | None,
) -> dict[str, Any] | None:
	baseline_path = _baseline_path(store_path)
	if not baseline_path.is_file():
		return None
	try:
		payload = json.loads(baseline_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise BaselineComputationError(
			f'Corrupted baseline payload: {baseline_path}'
		) from exc
	if payload.get('stage') != BASELINE_STAGE_RAW:
		return None
	if payload.get('source_sha256') != expected_sha:
		return None
	if expected_key1_byte is not None:
		stored_key1 = payload.get('key1_byte')
		if stored_key1 is None or int(stored_key1) != int(expected_key1_byte):
			return None
	return payload


@contextmanager
def _baseline_lock(store_path: Path):
	lock_path = store_path / BASELINE_LOCK_NAME
	fd = None
	try:
		fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
		yield
	finally:
		if fd is not None:
			os.close(fd)
		with suppress(FileNotFoundError):
			lock_path.unlink()


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


def _ensure_nonempty_counts(key1_counts: np.ndarray) -> None:
	if np.any(key1_counts <= 0):
		raise BaselineComputationError('TraceStore contains empty key1 sections')


def _ensure_trace_alignment(key1_counts: np.ndarray, n_traces: int) -> None:
	if int(np.sum(key1_counts, dtype=np.int64)) != n_traces:
		raise BaselineComputationError('TraceStore index does not match trace array')


def _trace_index_map(
	key1_values: np.ndarray, key1_offsets: np.ndarray, key1_counts: np.ndarray
) -> dict[str, list[int]]:
	mapping: dict[str, list[int]] = {}
	for value, offset, count in zip(
		key1_values, key1_offsets, key1_counts, strict=True
	):
		start = int(offset)
		stop = start + int(count)
		mapping[str(int(value))] = [start, stop]
	return mapping


def _compute_section_stats(
	*,
	traces: np.ndarray,
	key1_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	trace_sum = traces.sum(axis=1, dtype=np.float64)
	trace_sumsq = np.einsum('ij,ij->i', traces, traces, dtype=np.float64)
	group_ids = np.repeat(
		np.arange(key1_counts.size, dtype=np.int64), key1_counts.astype(np.int64)
	)
	group_sum = np.bincount(group_ids, weights=trace_sum, minlength=key1_counts.size)
	group_sumsq = np.bincount(
		group_ids, weights=trace_sumsq, minlength=key1_counts.size
	)
	n_samples = traces.shape[1]
	total_samples = key1_counts.astype(np.float64) * float(n_samples)
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
	artifacts: _TraceStoreArtifacts,
	key1_byte: int,
	key2_byte: int,
	key1_values: np.ndarray,
	key1_offsets: np.ndarray,
	key1_counts: np.ndarray,
	mu_traces: np.ndarray,
	sigma_traces: np.ndarray,
	zero_mask: np.ndarray,
	mu_sections: np.ndarray,
	sigma_sections: np.ndarray,
) -> dict[str, Any]:
	meta = artifacts.meta
	source_sha = meta.get('source_sha256')
	dt_val = float(get_dt_for_file(file_id))
	return {
		'stage': BASELINE_STAGE_RAW,
		'ddof': 0,
		'method': 'mean_std',
		'dtype_base': str(meta.get('dtype', '')),
		'dt': dt_val,
		'key1_values': key1_values.astype(np.int64).tolist(),
		'mu_section_by_key1': mu_sections.astype(np.float32, copy=False).tolist(),
		'sigma_section_by_key1': sigma_sections.astype(np.float32, copy=False).tolist(),
		'mu_traces': mu_traces.astype(np.float32, copy=False).tolist(),
		'sigma_traces': sigma_traces.astype(np.float32, copy=False).tolist(),
		'zero_var_mask': zero_mask.astype(bool, copy=False).tolist(),
		'trace_index_map': _trace_index_map(key1_values, key1_offsets, key1_counts),
		'source_sha256': source_sha,
		'computed_at': datetime.now(timezone.utc).isoformat(),
		'key1_byte': int(key1_byte),
		'key2_byte': int(key2_byte),
	}


def _write_baseline(store_path: Path, payload: dict[str, Any]) -> None:
	baseline_path = _baseline_path(store_path)
	temp_path = baseline_path.with_suffix('.tmp')
	temp_path.write_text(json.dumps(payload), encoding='utf-8')
	temp_path.replace(baseline_path)


def _compute_baseline(
	*,
	file_id: str,
	artifacts: _TraceStoreArtifacts,
	key1_byte: int,
	key2_byte: int,
) -> dict[str, Any]:
	reader = get_reader(file_id, key1_byte, key2_byte)
	traces = getattr(reader, 'traces', None)
	if not isinstance(traces, np.ndarray):
		raise BaselineComputationError('TraceStore reader did not expose traces array')
	if traces.ndim != 2:
		raise BaselineComputationError('Trace array must be 2D')
	key1_values, key1_offsets, key1_counts = _resolve_key1_partition(
		artifacts=artifacts,
		reader=reader,
		traces=traces,
		key1_byte=key1_byte,
	)
	_ensure_nonempty_counts(key1_counts)
	_ensure_trace_alignment(key1_counts, int(traces.shape[0]))
	mu_traces, sigma_traces, zero_mask = _compute_trace_stats(traces)
	mu_sections, sigma_sections = _compute_section_stats(
		traces=traces, key1_counts=key1_counts
	)
	if not np.all(np.isfinite(mu_traces)):
		raise BaselineComputationError('Per-trace mean produced non-finite values')
	if not np.all(np.isfinite(sigma_traces)):
		raise BaselineComputationError('Per-trace sigma produced non-finite values')
	if not np.all(np.isfinite(mu_sections)):
		raise BaselineComputationError('Section mean produced non-finite values')
	if not np.all(np.isfinite(sigma_sections)):
		raise BaselineComputationError('Section sigma produced non-finite values')
	return _prepare_payload(
		file_id=file_id,
		artifacts=artifacts,
		key1_byte=key1_byte,
		key2_byte=key2_byte,
		key1_values=key1_values,
		key1_offsets=key1_offsets,
		key1_counts=key1_counts,
		mu_traces=mu_traces,
		sigma_traces=sigma_traces,
		zero_mask=zero_mask,
		mu_sections=mu_sections,
		sigma_sections=sigma_sections,
	)


def get_or_create_raw_baseline(
	*, file_id: str, key1_byte: int, key2_byte: int
) -> dict[str, Any]:
	"""Return the cached raw baseline for ``file_id`` computing it if required."""
	artifacts = _load_trace_store_artifacts(file_id)
	meta = artifacts.meta
	expected_sha = meta.get('source_sha256')
	baseline = _load_baseline_if_valid(
		artifacts.store_path,
		expected_sha=expected_sha,
		expected_key1_byte=key1_byte,
	)
	if baseline is not None:
		return baseline
	lock_acquired = _acquire_lock_or_wait(artifacts.store_path)
	if not lock_acquired:
		baseline = _load_baseline_if_valid(
			artifacts.store_path,
			expected_sha=expected_sha,
			expected_key1_byte=key1_byte,
		)
		if baseline is not None:
			return baseline
		raise BaselineComputationError('Baseline computation is already in progress')
	try:
		cache_key = f'{file_id}_{int(key1_byte)}_{int(key2_byte)}'
		cached_readers.pop(cache_key, None)
		payload = _compute_baseline(
			file_id=file_id,
			artifacts=artifacts,
			key1_byte=key1_byte,
			key2_byte=key2_byte,
		)
		_write_baseline(artifacts.store_path, payload)
		return payload
	finally:
		_release_lock(artifacts.store_path)


__all__ = [
	'BASELINE_STAGE_RAW',
	'BaselineComputationError',
	'get_or_create_raw_baseline',
]
