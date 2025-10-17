"""Section retrieval and binary I/O endpoints."""

from __future__ import annotations

import contextlib
import gzip
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

if TYPE_CHECKING:
	from numpy.typing import NDArray
else:  # pragma: no cover - runtime alias for type checkers
	NDArray = np.ndarray

from app.api._helpers import (
	EXPECTED_SECTION_NDIM,
	OFFSET_BYTE_FIXED,
	USE_FBPICK_OFFSET,
	PipelineTapNotFoundError,
	get_reader,
	get_section_from_pipeline_tap,
	window_section_cache,
)
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file
from app.utils.utils import SegySectionReader, TraceStoreSectionReader, quantize_float32

router = APIRouter()


def _resolve_reader(entry: object) -> SegySectionReader | TraceStoreSectionReader:
	reader = getattr(entry, 'reader', None)
	if reader is None and isinstance(entry, dict):
		reader = entry.get('reader')
	if reader is None:
		msg = 'FILE_REGISTRY entry is missing a reader instance'
		raise KeyError(msg)
	if not isinstance(reader, (SegySectionReader, TraceStoreSectionReader)):
		msg = 'FILE_REGISTRY reader is of an unexpected type'
		raise TypeError(msg)
	return reader


def _key1_values_array(
	reader: SegySectionReader | TraceStoreSectionReader,
) -> NDArray[np.int64]:
	vals: Any = getattr(reader, 'unique_key1', None)
	if vals is not None:
		vals = np.asarray(vals)
	else:
		get_key1_values = getattr(reader, 'get_key1_values', None)
		if callable(get_key1_values):
			vals = np.asarray(get_key1_values())
	if vals is None:
		msg = 'Reader does not expose key1 values'
		raise AttributeError(msg)
	return np.asarray(vals, dtype=np.int64)


def _key1_value_for_index(
	reader: SegySectionReader | TraceStoreSectionReader, key1_idx: int
) -> int:
	vals = _key1_values_array(reader)
	if key1_idx < 0 or key1_idx >= vals.size:
		msg = 'key1_idx out of range'
		raise IndexError(msg)
	return int(vals[key1_idx])


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


def get_trace_seq_for(file_id: str, key1_idx: int, key1_byte: int) -> NDArray[np.int64]:
	"""Return display-aligned trace ordering for ``key1_idx`` of ``file_id``.
	Avoids touching FILE_REGISTRY.reader directly; always goes through get_reader (lazy-safe).
	"""
	# Try to pick a sensible key2_byte: use configured one if available, else default 193.
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

	key1_val = _key1_value_for_index(reader, key1_idx)

	get_trace_seq = getattr(reader, 'get_trace_seq_for_section', None)
	if callable(get_trace_seq):
		seq = get_trace_seq(key1_val, align_to='display')
		return np.asarray(seq, dtype=np.int64)

	if isinstance(reader, TraceStoreSectionReader):
		key1s = np.asarray(reader.get_header(reader.key1_byte), dtype=np.int64)
		indices = np.flatnonzero(key1s == key1_val)
		if indices.size == 0:
			msg = f'Key1 value {key1_val} not found'
			raise ValueError(msg)
		key2s = np.asarray(reader.get_header(reader.key2_byte)[indices])
		order = np.argsort(key2s, kind='stable')
		return np.asarray(indices[order], dtype=np.int64)

	msg = 'Reader cannot provide trace sequence information'
	raise AttributeError(msg)


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
	file_id: str = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
) -> JSONResponse:
	"""Return the available key1 header values for ``file_id``."""
	reader = get_reader(file_id, key1_byte, key2_byte)
	values = reader.get_key1_values()
	payload = values.tolist() if isinstance(values, np.ndarray) else list(values)
	return JSONResponse(content={'values': payload})


@router.get('/get_section')
def get_section(
	file_id: str = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
	key1_idx: int = Query(...),
) -> JSONResponse:
	"""Return the section for the ``key1_idx`` trace grouping."""
	try:
		reader = get_reader(file_id, key1_byte, key2_byte)
		key1_val = _key1_value_for_index(reader, key1_idx)
		section = reader.get_section(key1_val)
		payload = section.tolist() if isinstance(section, np.ndarray) else section
		return JSONResponse(content={'section': payload})
	except IndexError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/get_section_bin')
def get_section_bin(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
) -> Response:
	"""Return a quantized, binary section payload."""
	try:
		reader = get_reader(file_id, key1_byte, key2_byte)
		key1_val = key1_idx
		# key1_val = _key1_value_for_index(reader, key1_idx)
		section = np.array(reader.get_section(key1_val), dtype=np.float32)
		scale, q = quantize_float32(section)
		obj = {
			'scale': scale,
			'shape': q.shape,
			'data': q.tobytes(),
			'dt': get_dt_for_file(file_id),
		}
		payload = msgpack.packb(obj)
		return Response(
			gzip.compress(payload),
			media_type='application/octet-stream',
			headers={'Content-Encoding': 'gzip'},
		)
	except IndexError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/get_section_window_bin')
def get_section_window_bin(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
	offset_byte: int | None = Query(None),
	x0: int = Query(...),
	x1: int = Query(...),
	y0: int = Query(...),
	y1: int = Query(...),
	step_x: int = Query(1, ge=1),
	step_y: int = Query(1, ge=1),
	pipeline_key: str | None = Query(None),
	tap_label: str | None = Query(None),
) -> Response:
	"""Return a quantized window of a section, optionally via a pipeline tap."""
	forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else offset_byte
	key1_val = key1_idx

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
			section = get_section_from_pipeline_tap(
				file_id=file_id,
				key1_val=key1_val,
				key1_byte=key1_byte,
				pipeline_key=pipeline_key,
				tap_label=tap_label,
				offset_byte=forced_offset_byte,
			)
		else:
			reader = get_reader(file_id, key1_byte, key2_byte)
			# key1_val = _key1_value_for_index(reader, key1_idx)
			section = np.array(reader.get_section(key1_val), dtype=np.float32)
	except IndexError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except PipelineTapNotFoundError as exc:
		raise HTTPException(status_code=409, detail=str(exc)) from exc

	section = np.ascontiguousarray(section, dtype=np.float32)
	if section.ndim != EXPECTED_SECTION_NDIM:
		raise HTTPException(status_code=500, detail='Section data must be 2D')

	n_traces, n_samples = section.shape
	if not (0 <= x0 <= x1 < n_traces):
		raise HTTPException(status_code=400, detail='Trace range out of bounds')
	if not (0 <= y0 <= y1 < n_samples):
		raise HTTPException(status_code=400, detail='Sample range out of bounds')
	if step_x < 1 or step_y < 1:
		raise HTTPException(status_code=400, detail='Steps must be >= 1')

	sub = section[x0 : x1 + 1 : step_x, y0 : y1 + 1 : step_y]
	if sub.size == 0:
		raise HTTPException(status_code=400, detail='Requested window is empty')

	window_view = np.ascontiguousarray(sub.T, dtype=np.float32)
	scale, q = quantize_float32(window_view)
	obj: dict[str, Any] = {
		'scale': scale,
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
