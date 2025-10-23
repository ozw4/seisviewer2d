"""Section retrieval and binary I/O endpoints."""

from __future__ import annotations

import contextlib
import gzip
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import msgpack
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

if TYPE_CHECKING:
	from numpy.typing import NDArray
else:  # pragma: no cover - runtime alias for type checkers
	NDArray = np.ndarray

from app.api._helpers import (
	EXPECTED_SECTION_NDIM,
	OFFSET_BYTE_FIXED,
	USE_FBPICK_OFFSET,
	PipelineTapNotFoundError,
	apply_scaling_from_baseline,
	coerce_section_f32,
	get_reader,
	get_section_from_pipeline_tap,
	window_section_cache,
)
from app.api.baselines import (
	BASELINE_STAGE_RAW,
	BaselineComputationError,
	get_or_create_raw_baseline,
)
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file
from app.utils.utils import (
	SectionView,
	quantize_float32,
)

router = APIRouter()


class SectionMeta(BaseModel):
	shape: list[int]
	dt: float
	dtype: str | None = None
	scale: float | None = None


def get_ntraces_for(
	file_id: str, key1_byte: int | None = None, key2_byte: int | None = None
) -> int:
	"""Return total number of traces for ``file_id``.
	Always uses ``get_reader`` (lazy-safe). Falls back to registry meta if needed.
	"""
	ent = FILE_REGISTRY.get(file_id)
	if ent is None:
		raise KeyError(f'file_id not found: {file_id}')

	kb1 = 189 if key1_byte is None else int(key1_byte)
	kb2 = 193 if key2_byte is None else int(key2_byte)

	# Prefer an actual reader (lazy open). If it fails, try registry meta.
	with contextlib.suppress(Exception):
		reader = get_reader(file_id, kb1, kb2)
		ntraces = getattr(reader, 'ntraces', None)
		if ntraces is None:
			meta = getattr(reader, 'meta', None)
			if isinstance(meta, dict):
				ntraces = meta.get('n_traces')
		if ntraces is None and hasattr(reader, 'traces'):
			ntraces = getattr(reader.traces, 'shape', (None,))[0]
		if ntraces is None and hasattr(reader, 'key1s'):
			ntraces = len(reader.key1s)
		if ntraces is not None:
			return int(ntraces)

	# Fallback: registry meta (if present)
	meta = getattr(ent, 'meta', None)
	if isinstance(ent, dict) and meta is None:
		meta = ent.get('meta')
	if isinstance(meta, dict) and 'n_traces' in meta:
		return int(meta['n_traces'])

	raise AttributeError('Unable to determine number of traces for file')


def get_trace_seq_for_value(
	file_id: str, key1_val: int, key1_byte: int
) -> NDArray[np.int64]:
	"""Return display-aligned trace ordering for ``key1_val`` of ``file_id``."""
	key2_byte = 193
	ent = FILE_REGISTRY.get(file_id)
	if ent is None:
		raise KeyError(f'file_id not found: {file_id}')
	maybe_reader = getattr(ent, 'reader', None)
	if maybe_reader is None and isinstance(ent, dict):
		maybe_reader = ent.get('reader')
	if maybe_reader is not None:
		key2_byte = int(getattr(maybe_reader, 'key2_byte', 193))

	reader = get_reader(file_id, int(key1_byte), key2_byte)
	target_val = int(key1_val)

	get_trace_seq = getattr(reader, 'get_trace_seq_for_section', None)
	if callable(get_trace_seq):
		seq = get_trace_seq(target_val, align_to='display')
		return np.asarray(seq, dtype=np.int64)

	get_header = getattr(reader, 'get_header', None)
	if callable(get_header):
		key1_headers = np.asarray(get_header(int(key1_byte)), dtype=np.int64)
		indices = np.flatnonzero(key1_headers == target_val)
		if indices.size == 0:
			msg = f'Key1 value {target_val} not found'
			raise ValueError(msg)
		key2_src = int(getattr(reader, 'key2_byte', key2_byte))
		key2_headers = np.asarray(get_header(key2_src), dtype=np.int64)
		order = np.argsort(key2_headers[indices], kind='stable')
		return np.asarray(indices[order], dtype=np.int64)

	msg = 'Reader cannot provide trace sequence information'
	raise AttributeError(msg)


@router.get('/get_key1_values')
def get_key1_values(
	file_id: Annotated[str, Query(...)],
	key1_byte: Annotated[int, Query()] = 189,
	key2_byte: Annotated[int, Query()] = 193,
) -> JSONResponse:
	"""Return the available key1 header values for ``file_id``."""
	reader = get_reader(file_id, key1_byte, key2_byte)
	values = reader.get_key1_values()
	payload = values.tolist() if isinstance(values, np.ndarray) else list(values)
	return JSONResponse(content={'values': payload})


@router.get('/get_section')
def get_section(
	file_id: Annotated[str, Query(...)],
	key1_val: Annotated[int, Query(...)],
	key1_byte: Annotated[int, Query()] = 189,
	key2_byte: Annotated[int, Query()] = 193,
) -> JSONResponse:
	"""Return the section for the ``key1_val`` trace grouping."""
	try:
		reader = get_reader(file_id, key1_byte, key2_byte)
		view = reader.get_section(key1_val)
		payload = view.arr.tolist()
		return JSONResponse(content={'section': payload})
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/section/stats')
def get_section_stats(
	file_id: Annotated[str, Query(...)],
	baseline: Annotated[str, Query(...)],
	key1_idx: Annotated[int | None, Query()] = None,
	key1_byte: Annotated[int, Query()] = 189,
	key2_byte: Annotated[int, Query()] = 193,
	step_x: Annotated[int | None, Query()] = None,
	step_y: Annotated[int | None, Query()] = None,
) -> JSONResponse:
	baseline_value = baseline.lower().strip()
	if baseline_value != BASELINE_STAGE_RAW:
		raise HTTPException(status_code=400, detail='Only baseline=raw is supported')
	for name, value in (('step_x', step_x), ('step_y', step_y)):
		if value is not None and int(value) != 1:
			raise HTTPException(
				status_code=400,
				detail=f'{name} must equal 1 for raw baseline',
			)
	try:
		payload = get_or_create_raw_baseline(
			file_id=file_id,
			key1_byte=int(key1_byte),
			key2_byte=int(key2_byte),
		)
	except BaselineComputationError as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc
	response_payload = dict(payload)
	if key1_idx is not None:
		key1_val = int(key1_idx)
		key1_values = response_payload.get('key1_values') or []
		try:
			pos = key1_values.index(key1_val)
		except ValueError as exc:
			raise HTTPException(
				status_code=404,
				detail=f'key1_idx {key1_val} not found in baseline',
			) from exc
		trace_spans_map = response_payload.get('trace_spans_by_key1') or {}
		trace_spans = trace_spans_map.get(str(key1_val))
		if trace_spans is None:
			trace_spans = trace_spans_map.get(str(int(key1_val)))
		if trace_spans is None:
			trace_spans = []
		selected = {
			'key1_value': key1_val,
			'mu_section': response_payload['mu_section_by_key1'][pos],
			'sigma_section': response_payload['sigma_section_by_key1'][pos],
			'trace_spans': trace_spans,
		}
		if len(trace_spans) == 1:
			selected['trace_range'] = trace_spans[0]
		response_payload['selected_key1'] = selected
	return JSONResponse(content=response_payload)


@router.get('/get_section_meta', response_model=SectionMeta)
def get_section_meta(
	file_id: Annotated[str, Query(...)],
	key1_byte: Annotated[int, Query()] = 189,
	key2_byte: Annotated[int, Query()] = 193,
) -> SectionMeta:
	reader = get_reader(file_id, key1_byte, key2_byte)

	# セクション形状を実データから最短経路で確定
	values = reader.get_key1_values()
	first_val = int(values[0])
	n_traces = int(reader.get_trace_seq_for_value(first_val, align_to='display').size)
	n_samples = int(reader.get_n_samples())

	dtype = str(reader.dtype) if reader.dtype is not None else None
	scale = float(reader.scale) if isinstance(reader.scale, (int, float)) else None
	dt_val = float(get_dt_for_file(file_id))
	_ = get_or_create_raw_baseline(
		file_id=file_id, key1_byte=key1_byte, key2_byte=key2_byte
	)

	return SectionMeta(
		shape=[n_traces, n_samples],  # セクション内 [traces, samples]
		dt=dt_val,
		dtype=dtype,
		scale=scale,
	)


@router.get('/get_section_window_bin')
def get_section_window_bin(
	file_id: Annotated[str, Query(...)],
	key1_val: Annotated[int, Query(...)],
	x0: Annotated[int, Query(...)],
	x1: Annotated[int, Query(...)],
	y0: Annotated[int, Query(...)],
	y1: Annotated[int, Query(...)],
	key1_byte: Annotated[int, Query()] = 189,
	key2_byte: Annotated[int, Query()] = 193,
	offset_byte: Annotated[int | None, Query()] = None,
	step_x: Annotated[int, Query(ge=1)] = 1,
	step_y: Annotated[int, Query(ge=1)] = 1,
	transpose: Annotated[bool, Query()] = True,
	pipeline_key: Annotated[str | None, Query()] = None,
	tap_label: Annotated[str | None, Query()] = None,
	scaling: Annotated[
		Literal['amax', 'tracewise'] | None, Query(description='Normalization mode')
	] = None,
) -> Response:
	"""Return a quantized window of a section, optionally via a pipeline tap."""
	mode = 'amax' if scaling is None else scaling
	mode = mode.lower()
	if mode not in {'amax', 'tracewise'}:
		raise HTTPException(status_code=400, detail='Unsupported scaling mode')
	forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else offset_byte
	cache_key = (
		file_id,
		key1_val,
		key1_byte,
		key2_byte,
		forced_offset_byte,
		x0,
		x1,
		y0,
		y1,
		step_x,
		step_y,
		transpose,
		pipeline_key,
		tap_label,
		mode,
	)

	cached_payload = window_section_cache.get(cache_key)
	if cached_payload is not None:
		return Response(
			cached_payload,
			media_type='application/octet-stream',
			headers={'Content-Encoding': 'gzip'},
		)

	reader = None
	try:
		if pipeline_key and tap_label:
			section_arr = get_section_from_pipeline_tap(
				file_id=file_id,
				key1_val=key1_val,
				key1_byte=key1_byte,
				pipeline_key=pipeline_key,
				tap_label=tap_label,
				offset_byte=forced_offset_byte,
			)
			section_view = SectionView(
				arr=section_arr, dtype=section_arr.dtype, scale=None
			)
		else:
			reader = get_reader(file_id, key1_byte, key2_byte)
			section_view = reader.get_section(key1_val)
	except PipelineTapNotFoundError as exc:
		raise HTTPException(status_code=409, detail=str(exc)) from exc
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	base = section_view.arr
	if base.ndim != EXPECTED_SECTION_NDIM:
		raise HTTPException(status_code=500, detail='Section data must be 2D')

	n_traces, n_samples = base.shape
	if not (0 <= x0 <= x1 < n_traces):
		raise HTTPException(status_code=400, detail='Trace range out of bounds')
	if not (0 <= y0 <= y1 < n_samples):
		raise HTTPException(status_code=400, detail='Sample range out of bounds')
	if step_x < 1 or step_y < 1:
		raise HTTPException(status_code=400, detail='Steps must be >= 1')

	sub = base[x0 : x1 + 1 : step_x, y0 : y1 + 1 : step_y]
	if sub.size == 0:
		raise HTTPException(status_code=400, detail='Requested window is empty')

	prepared = coerce_section_f32(sub, section_view.scale)
	registry_entry = FILE_REGISTRY.get(file_id)
	store_dir: str | None = None
	if isinstance(registry_entry, dict):
		maybe_store = registry_entry.get('store_path')
		if isinstance(maybe_store, str):
			store_dir = maybe_store
	else:
		maybe_store = getattr(registry_entry, 'store_path', None)
		if isinstance(maybe_store, (str, Path)):
			store_dir = str(maybe_store)
	if store_dir is None and reader is not None:
		maybe_store = getattr(reader, 'store_dir', None)
		if isinstance(maybe_store, (str, Path)):
			store_dir = str(maybe_store)
	if store_dir is None:
		raise HTTPException(status_code=500, detail='Trace store path unavailable')
	prepared = apply_scaling_from_baseline(
		prepared,
		mode,
		file_id,
		key1_val,
		store_dir,
	)
	view = prepared.T if transpose else prepared
	window_view = np.ascontiguousarray(view, dtype=np.float32)
	scale_val, q = quantize_float32(window_view)
	obj: dict[str, Any] = {
		'scale': scale_val,
		'shape': window_view.shape,
		'data': q.tobytes(),
		'dt': get_dt_for_file(file_id),
	}
	payload = msgpack.packb(obj)
	compressed = gzip.compress(payload)
	window_section_cache.set(cache_key, compressed)
	return Response(
		compressed,
		media_type='application/octet-stream',
		headers={'Content-Encoding': 'gzip'},
	)
