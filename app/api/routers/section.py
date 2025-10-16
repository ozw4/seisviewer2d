"""Section retrieval and binary I/O endpoints."""

from __future__ import annotations

import contextlib
import gzip
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse

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
        load_section_by_indices,
        make_cache_key,
        window_section_cache,
)
from app.api.schemas import SectionBinQuery, SectionQuery, SectionWindowBinQuery
from app.utils.key_resolver import resolve_indices_slice_on_demand
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file
from app.utils.utils import SegySectionReader, TraceStoreSectionReader, quantize_float32

router = APIRouter()


WARNING_DEPRECATED_IDX = '299 - key1_idx is deprecated; use key1_value'


def _resolve_indices_for_request(
        reader: SegySectionReader | TraceStoreSectionReader,
        key1_value: object,
        start: int,
        length: int | None,
) -> np.ndarray:
        effective_length = length if length is not None else 1_000_000_000
        try:
                return resolve_indices_slice_on_demand(reader, key1_value, start, effective_length)
        except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc


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
        q: SectionQuery = Depends(),
) -> JSONResponse:
        """Return the section for the requested key1 gather."""
        reader = get_reader(q.file_id, q.key1_byte, q.key2_byte)
        idx = _resolve_indices_for_request(reader, q.key1_value, q.start, q.length)
        section = load_section_by_indices(reader, idx)
        if section.ndim != EXPECTED_SECTION_NDIM:
                raise HTTPException(status_code=500, detail='Section data must be 2D')
        payload = section.tolist()
        resp = JSONResponse(content={'section': payload})
        if q.used_deprecated_idx:
                resp.headers['Warning'] = WARNING_DEPRECATED_IDX
        return resp


@router.get('/get_section_bin')
def get_section_bin(
        q: SectionBinQuery = Depends(),
) -> Response:
        """Return a quantized, binary section payload."""
        reader = get_reader(q.file_id, q.key1_byte, q.key2_byte)
        idx = _resolve_indices_for_request(reader, q.key1_value, q.start, q.length)
        section = load_section_by_indices(reader, idx)
        if section.ndim != EXPECTED_SECTION_NDIM:
                raise HTTPException(status_code=500, detail='Section data must be 2D')
        section = np.ascontiguousarray(section, dtype=np.float32)
        scale, q_arr = quantize_float32(section)
        obj = {
                'scale': scale,
                'shape': q_arr.shape,
                'data': q_arr.tobytes(),
                'dt': get_dt_for_file(q.file_id),
        }
        payload = msgpack.packb(obj)
        resp = Response(
                gzip.compress(payload),
                media_type='application/octet-stream',
                headers={'Content-Encoding': 'gzip'},
        )
        if q.used_deprecated_idx:
                resp.headers['Warning'] = WARNING_DEPRECATED_IDX
        return resp


@router.get('/get_section_window_bin')
def get_section_window_bin(
        q: SectionWindowBinQuery = Depends(),
) -> Response:
        """Return a quantized window of a section, optionally via a pipeline tap."""
        forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else q.offset_byte

        reader = get_reader(q.file_id, q.key1_byte, q.key2_byte)
        idx = _resolve_indices_for_request(reader, q.key1_value, q.start, q.length)
        trace_count = int(idx.size)

        cache_spec = {
                'kb1': q.key1_byte,
                'k2b': q.key2_byte,
                'offset': forced_offset_byte,
                'y0': q.y0,
                'y1': q.y1,
                'sx': q.step_x,
                'sy': q.step_y,
                'pipe': q.pipeline_key,
                'tap': q.tap_label,
        }
        cache_key = make_cache_key(q.file_id, q.key1_value, q.start, trace_count, cache_spec)

        cached_payload = window_section_cache.get(cache_key)
        if cached_payload is not None:
                resp = Response(
                        cached_payload,
                        media_type='application/octet-stream',
                        headers={'Content-Encoding': 'gzip'},
                )
                if q.used_deprecated_idx:
                        resp.headers['Warning'] = WARNING_DEPRECATED_IDX
                return resp

        try:
                if q.pipeline_key and q.tap_label:
                        section = get_section_from_pipeline_tap(
                                file_id=q.file_id,
                                key1_idx=q.key1_idx,
                                key1_byte=q.key1_byte,
                                pipeline_key=q.pipeline_key,
                                tap_label=q.tap_label,
                                offset_byte=forced_offset_byte,
                                key1_value=q.key1_value,
                                start=q.start,
                                length=trace_count if trace_count else None,
                        )
                        section = np.asarray(section, dtype=np.float32)
                        if section.ndim != EXPECTED_SECTION_NDIM:
                                raise HTTPException(status_code=500, detail='Section data must be 2D')
                        available = section.shape[0]
                        window_end = q.start + trace_count
                        if available == trace_count:
                                pass
                        elif available >= window_end:
                                section = section[q.start:window_end]
                        else:
                                raise HTTPException(
                                        status_code=422,
                                        detail='Pipeline tap output shorter than requested window',
                                )
                else:
                        section = load_section_by_indices(reader, idx)
        except PipelineTapNotFoundError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        section = np.ascontiguousarray(section, dtype=np.float32)
        if section.ndim != EXPECTED_SECTION_NDIM:
                raise HTTPException(status_code=500, detail='Section data must be 2D')

        n_traces, n_samples = section.shape
        if n_traces == 0:
                raise HTTPException(status_code=422, detail='Requested window is empty')

        y0 = q.y0
        y1 = (n_samples - 1) if q.y1 is None else q.y1
        if not (0 <= y0 <= y1 < n_samples):
                raise HTTPException(status_code=400, detail='Sample range out of bounds')

        traces = section[:: q.step_x]
        window = traces[:, y0 : y1 + 1 : q.step_y]
        if window.size == 0:
                raise HTTPException(status_code=400, detail='Requested window is empty')

        window_view = np.ascontiguousarray(window.T, dtype=np.float32)
        scale, q_arr = quantize_float32(window_view)
        obj: dict[str, Any] = {
                'scale': scale,
                'shape': window_view.shape,
                'data': q_arr.tobytes(),
                'dt': get_dt_for_file(q.file_id),
        }
        payload = msgpack.packb(obj)
        compressed = gzip.compress(payload)
        window_section_cache.set(cache_key, compressed)

        resp = Response(
                compressed,
                media_type='application/octet-stream',
                headers={'Content-Encoding': 'gzip'},
        )
        if q.used_deprecated_idx:
                resp.headers['Warning'] = WARNING_DEPRECATED_IDX
        return resp
