"""Shared helpers and state for API routers."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import HTTPException
from numpy.typing import NDArray

from app.utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file, load_baseline
from app.utils.utils import TraceStoreSectionReader

if TYPE_CHECKING:
	from app.api.schemas import PipelineSpec

USE_FBPICK_OFFSET = 'offset' in FBPICK_MODEL_PATH.name.lower()
OFFSET_BYTE_FIXED: int = 37

EXPECTED_SECTION_NDIM = 2

logger = logging.getLogger(__name__)

NORM_EPS = np.float32(float(os.getenv('NORM_EPS', '1e-6')))

_TRACE_STATS_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, int]] = {}


def coerce_section_f32(arr: NDArray, scale: float | None) -> NDArray[np.float32]:
	out = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
	if scale is not None:
		if not out.flags.writeable:
			out = out.copy()
		if out.dtype != np.float32:
			out = out.astype(np.float32, copy=False)
		out *= float(scale)
	if not out.flags['C_CONTIGUOUS'] or out.dtype != np.float32:
		out = np.ascontiguousarray(out, dtype=np.float32)
	return out


def apply_scaling_from_baseline(
	arr: NDArray[np.float32],
	scaling: str | None,
	file_id: str,
	key1_val: int,
	store_dir: str | Path,
	*,
	x0: int,
	x1: int,
	step_x: int,
) -> NDArray[np.float32]:
	"""Normalize ``arr`` in-place using baseline statistics."""
	mode = 'amax' if scaling is None else str(scaling)
	mode = mode.lower()
	if mode not in {'amax', 'tracewise'}:
		raise HTTPException(status_code=400, detail='Unsupported scaling mode')
	if not np.all(np.isfinite(arr)):
		raise HTTPException(
			status_code=422, detail='Section contains non-finite values'
		)
	try:
		baseline = load_baseline(store_dir)
	except FileNotFoundError as exc:
		raise HTTPException(
			status_code=500, detail='Baseline statistics not found'
		) from exc
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(
			status_code=500, detail='Failed to load baseline statistics'
		) from exc
	key1_int = int(key1_val)
	index = baseline['key1_index'].get(key1_int)
	if index is None:
		raise HTTPException(
			status_code=422, detail='Baseline statistics missing for key1 value'
		)
	if mode == 'amax':
		mean = baseline['section_mean'][index]
		inv_std = baseline['section_inv_std'][index]
		if not (np.isfinite(mean) and np.isfinite(inv_std)):
			raise HTTPException(
				status_code=422, detail='Baseline statistics contain non-finite values'
			)
		arr -= mean
		arr *= inv_std
		if bool(baseline['section_clamp_mask'][index]):
			logger.info(
				'Section std clamped to eps for file_id=%s key1=%s', file_id, key1_int
			)
		return arr
	spans = baseline['trace_spans'].get(key1_int)
	if spans is None:
		raise HTTPException(
			status_code=422, detail='Baseline trace spans missing for key1 value'
		)
	store_key = baseline['store_key']
	idx_full_key = (store_key, key1_int, 'idx_full')
	idx_full = _TRACE_STATS_CACHE.get(idx_full_key)
	if idx_full is None:
		idx_full = np.concatenate(
			[np.arange(int(s), int(e), dtype=np.int64) for (s, e) in spans]
		)
		_TRACE_STATS_CACHE[idx_full_key] = (idx_full, None, 0)  # slot reuse
	else:
		idx_full = idx_full[0]  # unpack tuple

	# Window/stride selection within the section
	n_traces = int(arr.shape[0])
	if x0 < 0 or x1 < x0 or step_x < 1:
		raise HTTPException(status_code=400, detail='Invalid window parameters')
	sel = np.arange(0, idx_full.shape[0], dtype=np.int64)[x0 : x1 + 1 : step_x]
	if sel.shape[0] != n_traces:
		raise HTTPException(status_code=422, detail='Trace count mismatch for window')
	sel_global = idx_full[sel]

	# Gather per-trace stats for the selected window (cache by x0,x1,step_x)
	trace_key = (store_key, key1_int, x0, x1, step_x)
	cached = _TRACE_STATS_CACHE.get(trace_key)
	if cached is None:
		trace_mean = baseline['trace_mean']
		trace_inv = baseline['trace_inv_std']
		mean_vec = trace_mean[sel_global].astype(np.float32, copy=False)
		inv_vec = trace_inv[sel_global].astype(np.float32, copy=False)
		_TRACE_STATS_CACHE[trace_key] = (mean_vec, inv_vec, 0)
		cached = _TRACE_STATS_CACHE[trace_key]
	mean_vec, inv_vec, _ = cached
	if not np.all(np.isfinite(mean_vec)) or not np.all(np.isfinite(inv_vec)):
		raise HTTPException(
			status_code=422,
			detail='Baseline trace statistics contain non-finite values',
		)
	arr -= mean_vec[:, None]
	arr *= inv_vec[:, None]
	return arr


class LRUCache(OrderedDict):
	"""A tiny ordered cache used for in-memory tap storage."""

	def __init__(self, capacity: int = 16):
		super().__init__()
		self.capacity = capacity

	def get(self, key):
		if key in self:
			self.move_to_end(key)
			return super().__getitem__(key)
		return None

	def set(self, key, value):
		if key in self:
			self.move_to_end(key)
		super().__setitem__(key, value)
		if len(self) > self.capacity:
			self.popitem(last=False)


cached_readers: dict[str, TraceStoreSectionReader] = {}
fbpick_cache: dict[tuple, bytes] = {}
jobs: dict[str, dict[str, object]] = {}
pipeline_tap_cache = LRUCache(16)
window_section_cache = LRUCache(32)


class PipelineTapNotFoundError(LookupError):
	"""Raised when a requested pipeline tap output is unavailable."""


def _spec_uses_fbpick(spec: PipelineSpec) -> bool:
	"""Return ``True`` when ``spec`` contains an fbpick analyzer step."""
	return any(step.kind == 'analyzer' and step.name == 'fbpick' for step in spec.steps)


def _maybe_attach_fbpick_offsets(
	meta: dict[str, Any],
	*,
	spec: PipelineSpec,
	reader: TraceStoreSectionReader,
	key1_val: int,
	offset_byte: int | None,
	trace_slice: slice | None = None,
) -> dict[str, Any]:
	"""Add offset metadata when the fbpick model expects it."""
	if not USE_FBPICK_OFFSET or offset_byte is None:
		return meta
	if not _spec_uses_fbpick(spec):
		return meta
	get_offsets = getattr(reader, 'get_offsets_for_section', None)
	if get_offsets is None:
		return meta
	offsets = get_offsets(key1_val, offset_byte)
	if trace_slice is not None:
		offsets = offsets[trace_slice]
	offsets = np.ascontiguousarray(offsets, dtype=np.float32)
	if not meta:
		return {'offsets': offsets}
	meta_with_offsets = dict(meta)
	meta_with_offsets['offsets'] = offsets
	return meta_with_offsets


def _update_file_registry(
	file_id: str,
	*,
	path: str | None = None,
	store_path: str | None = None,
	dt: float | None = None,
) -> None:
	rec = FILE_REGISTRY.get(file_id) or {}
	if path:
		rec['path'] = path
	if store_path:
		rec['store_path'] = store_path
	if isinstance(dt, (int, float)) and dt > 0:
		rec['dt'] = float(dt)
	FILE_REGISTRY[file_id] = rec


def _filename_for_file_id(file_id: str) -> str | None:
	rec = FILE_REGISTRY.get(file_id) or {}
	path = rec.get('path') or rec.get('store_path')
	return Path(path).name if path else None


def _pipeline_payload_to_array(payload: object, *, tap_label: str) -> np.ndarray:
	"""Convert a cached pipeline payload into a 2D ``float32`` array."""
	data_obj = payload
	if isinstance(payload, dict):
		for key in ('data', 'prob', 'values'):
			if key in payload:
				data_obj = payload[key]
				break
		else:
			msg = f'Pipeline tap {tap_label!r} payload missing data field'
			raise ValueError(msg)

	arr = np.asarray(data_obj, dtype=np.float32)
	if arr.ndim != EXPECTED_SECTION_NDIM:
		msg = f'Pipeline tap {tap_label!r} expected 2D data, got {arr.ndim}D'
		raise ValueError(msg)
	return np.ascontiguousarray(arr)


def get_section_from_pipeline_tap(
	*,
	file_id: str,
	key1_val: int,
	key1_byte: int,
	pipeline_key: str,
	tap_label: str,
	offset_byte: int | None = None,
) -> np.ndarray:
	"""Return the cached pipeline tap output as a ``float32`` array."""
	arr, _ = get_section_and_meta_from_pipeline_tap(
		file_id=file_id,
		key1_val=key1_val,
		key1_byte=key1_byte,
		pipeline_key=pipeline_key,
		tap_label=tap_label,
		offset_byte=offset_byte,
	)
	return arr


def get_section_and_meta_from_pipeline_tap(
	*,
	file_id: str,
	key1_val: int,
	key1_byte: int,
	pipeline_key: str,
	tap_label: str,
	offset_byte: int | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
	"""Return tap payload as ``float32`` along with optional metadata."""
	base_key = (file_id, key1_val, key1_byte, pipeline_key, None, offset_byte)
	payload = pipeline_tap_cache.get((*base_key, tap_label))
	if payload is None:
		msg = (
			f'Pipeline tap {tap_label!r} for pipeline {pipeline_key!r} '
			f'and key1={key1_val} is not available. '
			'Please re-run the pipeline.'
		)
		raise PipelineTapNotFoundError(msg)
	arr = _pipeline_payload_to_array(payload, tap_label=tap_label)
	meta = None
	if isinstance(payload, dict):
		meta_obj = payload.get('meta')
		if isinstance(meta_obj, dict):
			meta = meta_obj
	return arr, meta


def get_reader(file_id: str, key1_byte: int, key2_byte: int) -> TraceStoreSectionReader:
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	reader = cached_readers.get(cache_key)
	if reader is None:
		rec = FILE_REGISTRY.get(file_id)
		if rec is None:
			raise HTTPException(status_code=404, detail='File ID not found')
		store_path = rec.get('store_path') if isinstance(rec, dict) else None
		if not isinstance(store_path, str):
			raise HTTPException(status_code=409, detail='trace store not built')
		p = Path(store_path)
		if not p.is_dir():
			raise HTTPException(status_code=409, detail='trace store not built')
		reader = TraceStoreSectionReader(p, key1_byte, key2_byte)
		cached_readers[cache_key] = reader
	dt_val = get_dt_for_file(file_id)
	meta_attr = getattr(reader, 'meta', None)
	if isinstance(meta_attr, dict):
		if not isinstance(meta_attr.get('dt'), (int, float)) or meta_attr['dt'] <= 0:
			meta_attr['dt'] = dt_val
	else:
		with suppress(Exception):
			reader.meta = {'dt': dt_val}
	return reader


def get_raw_section(
	*, file_id: str, key1_val: int, key1_byte: int, key2_byte: int
) -> np.ndarray:
	"""Load the RAW seismic section as ``float32``."""
	reader = get_reader(file_id, key1_byte, key2_byte)
	view = reader.get_section(key1_val)
	base = view.arr
	arr = coerce_section_f32(base, view.scale)
	if arr.ndim != EXPECTED_SECTION_NDIM:
		msg = f'Raw section expected 2D data, got {arr.ndim}D'
		raise ValueError(msg)
	return arr


__all__ = [
	'EXPECTED_SECTION_NDIM',
	'OFFSET_BYTE_FIXED',
	'USE_FBPICK_OFFSET',
	'LRUCache',
	'PipelineTapNotFoundError',
	'_filename_for_file_id',
	'_maybe_attach_fbpick_offsets',
	'_pipeline_payload_to_array',
	'_spec_uses_fbpick',
	'_update_file_registry',
	'cached_readers',
	'coerce_section_f32',
	'fbpick_cache',
	'get_raw_section',
	'get_reader',
	'get_section_and_meta_from_pipeline_tap',
	'get_section_from_pipeline_tap',
	'jobs',
	'pipeline_tap_cache',
	'window_section_cache',
]
