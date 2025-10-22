"""Section retrieval and binary I/O endpoints."""

from __future__ import annotations

import contextlib
import gzip
import logging
from typing import TYPE_CHECKING, Annotated, Any

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
	coerce_section_f32,
	get_reader,
	get_section_from_pipeline_tap,
	window_section_cache,
)
from app.utils.raw_stats import ensure_raw_baseline
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file
from app.utils.utils import (
	SectionView,
	quantize_float32,
)

router = APIRouter()
logger = logging.getLogger(__name__)

RAW_BASELINE_DDOF = 0


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

	return SectionMeta(
		shape=[n_traces, n_samples],  # セクション内 [traces, samples]
		dt=dt_val,
		dtype=dtype,
		scale=scale,
	)


@router.get('/get_section_bin')
def get_section_bin(
	file_id: Annotated[str, Query(...)],
	key1_val: Annotated[int, Query(...)],
	key1_byte: Annotated[int, Query()] = 189,
	key2_byte: Annotated[int, Query()] = 193,
) -> Response:
	"""Return a quantized, binary section payload."""
	try:
		reader = get_reader(file_id, key1_byte, key2_byte)
		view = reader.get_section(key1_val)
		base = view.arr
		if base.ndim != EXPECTED_SECTION_NDIM:
			raise HTTPException(status_code=500, detail='Section data must be 2D')
		prepared = coerce_section_f32(base, view.scale)
		scale_val, q = quantize_float32(prepared)
		obj = {
			'scale': scale_val,
			'shape': prepared.shape,
			'data': q.tobytes(),
			'dt': get_dt_for_file(file_id),
		}
		payload = msgpack.packb(obj)
		return Response(
			gzip.compress(payload),
			media_type='application/octet-stream',
			headers={'Content-Encoding': 'gzip'},
		)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/section/stats')
def get_section_stats(
	file_id: Annotated[str, Query(...)],
	key1_val: Annotated[int, Query(...)],
	baseline: Annotated[str, Query()] = 'raw',
	key1_byte: Annotated[int, Query()] = 189,
	key2_byte: Annotated[int, Query()] = 193,
) -> JSONResponse:
	"""Return baseline statistics for visualization."""

	baseline_norm = baseline.lower()
	if baseline_norm != 'raw':
		msg = "Only baseline='raw' is supported"
		raise HTTPException(status_code=400, detail=msg)

	try:
		reader = get_reader(file_id, key1_byte, key2_byte)
	except HTTPException:
		raise
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	try:
		view = reader.get_section(key1_val)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	arr = coerce_section_f32(view.arr, view.scale)
	if arr.ndim != EXPECTED_SECTION_NDIM:
		msg = f'Section data must be 2D, got {arr.ndim}D'
		raise HTTPException(status_code=500, detail=msg)

	dt_val = float(get_dt_for_file(file_id))
	dtype_base = (
		str(view.dtype) if isinstance(view.dtype, np.dtype) else str(view.dtype)
	)

	stats = ensure_raw_baseline(
		reader=reader,
		key1_val=int(key1_val),
		section=arr,
		dtype_base=dtype_base,
		dt=dt_val,
		ddof=RAW_BASELINE_DDOF,
	)

	logger.info(
		'section.stats stage=%s key1=%s ddof=%s source=%s',
		stats.stage,
		key1_val,
		stats.ddof,
		stats.source_sha256,
	)

	return JSONResponse(content=stats.to_payload())


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
) -> Response:
	"""Return a quantized window of a section, optionally via a pipeline tap."""
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
	)

	cached_payload = window_section_cache.get(cache_key)
	if cached_payload is not None:
		return Response(
			cached_payload,
			media_type='application/octet-stream',
			headers={'Content-Encoding': 'gzip'},
		)

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
