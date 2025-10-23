"""Helpers for SEG-Y metadata such as sampling interval."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

HEADER_SAMPLE_INTERVAL_OFFSET = 3200 + 16
SAMPLE_INTERVAL_BYTES = 2
MICROSECONDS_PER_SECOND = 1_000_000.0

# ファイル毎のメタ情報を保持
FILE_REGISTRY: dict[str, dict[str, Any]] = {}

BASELINE_FILENAME_RAW = 'baseline_raw.json'

logger = logging.getLogger(__name__)

NORM_EPS = np.float32(float(os.getenv('NORM_EPS', '1e-6')))

_BASELINE_CACHE: dict[str, dict[str, Any]] = {}


def read_segy_dt_seconds(path: str) -> float | None:
	"""Return the SEG-Y sampling interval in seconds, if available."""
	sample_path = Path(path)
	try:
		with sample_path.open('rb') as f:
			f.seek(HEADER_SAMPLE_INTERVAL_OFFSET)
			raw = f.read(SAMPLE_INTERVAL_BYTES)
		if len(raw) != SAMPLE_INTERVAL_BYTES:
			return None
		us = int.from_bytes(raw, byteorder='big', signed=False)
		if us <= 0:
			return None
		return us / MICROSECONDS_PER_SECOND
	except Exception:  # noqa: BLE001
		return None


def get_dt_for_file(file_id: str) -> float:
	"""Resolve the sampling interval in seconds for ``file_id``."""
	rec = FILE_REGISTRY.get(file_id)
	if not isinstance(rec, dict):
		rec = {}

	dt_val = rec.get('dt')
	if isinstance(dt_val, (int, float)) and dt_val > 0:
		return float(dt_val)

	# 1) 直接パスが分かっていればヘッダから読む
	path = rec.get('path')

	# 2) trace store から meta.json を参照して復元
	if not path:
		store_path = rec.get('store_path')
		if isinstance(store_path, str):
			meta_path = Path(store_path) / 'meta.json'
			try:
				meta = json.loads(meta_path.read_text())
			except Exception:  # noqa: BLE001
				meta = None
			if isinstance(meta, dict):
				meta_dt = meta.get('dt')
				if isinstance(meta_dt, (int, float)) and meta_dt > 0:
					rec['dt'] = float(meta_dt)
					FILE_REGISTRY[file_id] = rec
					return float(meta_dt)
				original = meta.get('original_segy_path')
				if isinstance(original, str):
					path = original
					rec['path'] = path

	dt = read_segy_dt_seconds(path) if path else None
	if not dt:
		raise RuntimeError('dt not found')
	rec['dt'] = dt
	FILE_REGISTRY[file_id] = rec
	return dt


def load_baseline(store_dir: str | Path) -> dict[str, Any]:
	"""Return cached baseline statistics for ``store_dir``."""
	store_path = Path(store_dir)
	baseline_path = store_path / BASELINE_FILENAME_RAW
	cache_key = str(baseline_path.resolve())
	entry = _BASELINE_CACHE.get(cache_key)
	if entry is not None:
		return entry
	if not baseline_path.is_file():
		raise FileNotFoundError(f'baseline payload not found: {baseline_path}')
	payload = json.loads(baseline_path.read_text(encoding='utf-8'))
	key1_values = np.ascontiguousarray(
		np.asarray(payload.get('key1_values'), dtype=np.int64)
	)
	mu_section = np.ascontiguousarray(
		np.asarray(payload.get('mu_section_by_key1'), dtype=np.float32)
	)
	sigma_section = np.ascontiguousarray(
		np.asarray(payload.get('sigma_section_by_key1'), dtype=np.float32)
	)
	if (
		key1_values.shape[0] != mu_section.shape[0]
		or mu_section.shape != sigma_section.shape
	):
		raise ValueError('Baseline section statistics are inconsistent')
	if not np.all(np.isfinite(mu_section)):
		raise ValueError('Baseline section mean contains non-finite values')
	if not np.all(np.isfinite(sigma_section)):
		raise ValueError('Baseline section std contains non-finite values')
	section_clamp_mask = np.ascontiguousarray(sigma_section <= NORM_EPS, dtype=bool)
	if section_clamp_mask.any():
		logger.info(
			'Clamped %d section std values to eps (%s)',
			int(section_clamp_mask.sum()),
			baseline_path,
		)
	safe_sigma_section = sigma_section.copy()
	np.maximum(safe_sigma_section, NORM_EPS, out=safe_sigma_section)
	inv_sigma_section = np.empty_like(safe_sigma_section, dtype=np.float32)
	np.reciprocal(safe_sigma_section, out=inv_sigma_section)
	mu_traces = np.ascontiguousarray(
		np.asarray(payload.get('mu_traces'), dtype=np.float32)
	)
	sigma_traces = np.ascontiguousarray(
		np.asarray(payload.get('sigma_traces'), dtype=np.float32)
	)
	if mu_traces.shape != sigma_traces.shape:
		raise ValueError('Baseline trace statistics are inconsistent')
	if not np.all(np.isfinite(mu_traces)):
		raise ValueError('Baseline trace mean contains non-finite values')
	if not np.all(np.isfinite(sigma_traces)):
		raise ValueError('Baseline trace std contains non-finite values')
	trace_clamp_mask = np.ascontiguousarray(sigma_traces <= NORM_EPS, dtype=bool)
	if trace_clamp_mask.any():
		logger.info(
			'Clamped %d trace std values to eps (%s)',
			int(trace_clamp_mask.sum()),
			baseline_path,
		)
	safe_sigma_traces = sigma_traces.copy()
	np.maximum(safe_sigma_traces, NORM_EPS, out=safe_sigma_traces)
	inv_sigma_traces = np.empty_like(safe_sigma_traces, dtype=np.float32)
	np.reciprocal(safe_sigma_traces, out=inv_sigma_traces)
	raw_spans = payload.get('trace_spans_by_key1') or {}
	trace_spans: dict[int, list[tuple[int, int]]] = {}
	for key_str, ranges in raw_spans.items():
		key_int = int(key_str)
		span_list: list[tuple[int, int]] = []
		for span in ranges:
			if not isinstance(span, list) or len(span) != 2:
				raise ValueError('Baseline trace span entry malformed')
			start, end = int(span[0]), int(span[1])
			if start < 0 or end < start or end > mu_traces.shape[0]:
				raise ValueError('Baseline trace span is out of bounds')
			span_list.append((start, end))
		trace_spans[key_int] = span_list
	entry = {
		'store_key': str(store_path.resolve()),
		'key1_values': key1_values,
		'key1_index': {int(val): idx for idx, val in enumerate(key1_values.tolist())},
		'section_mean': mu_section,
		'section_inv_std': inv_sigma_section,
		'section_clamp_mask': section_clamp_mask,
		'trace_mean': mu_traces,
		'trace_inv_std': inv_sigma_traces,
		'trace_clamp_mask': trace_clamp_mask,
		'trace_spans': trace_spans,
	}
	_BASELINE_CACHE[cache_key] = entry
	return entry
