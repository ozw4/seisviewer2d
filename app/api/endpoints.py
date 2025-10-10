# ruff: noqa: ANN001, ANN201, ANN204, D100, D101, D102, D103, FAST002, PLR0913,B008
# endpoint.py
import asyncio
import gzip
import hashlib
import json
import pathlib
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import msgpack
import numpy as np
import segyio
from api.schemas import (
	PipelineAllResponse,
	PipelineJobStatusResponse,
	PipelineSectionResponse,
	PipelineSpec,
)
from fastapi import (
	APIRouter,
	Body,
	File,
	Form,  # 忘れずにインポート
	HTTPException,
	Query,
	UploadFile,
)
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from utils import picks_by_name
from utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from utils.picks import add_pick, delete_pick, list_picks, store
from utils.pipeline import apply_pipeline, pipeline_key
from utils.segy_meta import FILE_REGISTRY, get_dt_for_file, read_segy_dt_seconds
from utils.utils import (
	SegySectionReader,
	TraceStoreSectionReader,
	quantize_float32,
	to_builtin,
)

# Offset-enabled model detection from filename (e.g., "*offset*.pth")
USE_FBPICK_OFFSET = 'offset' in FBPICK_MODEL_PATH.name.lower()
# Always use this SEG-Y trace header byte as "offset" when offset-enabled model is active
OFFSET_BYTE_FIXED: int = 37

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = UPLOAD_DIR / 'processed'

TRACE_DIR = PROCESSED_DIR / 'traces'
TRACE_DIR.mkdir(parents=True, exist_ok=True)


cached_readers: dict[str, SegySectionReader | TraceStoreSectionReader] = {}
SEGYS: dict[str, str] = {}


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


# Private caches for FB pick sections and asynchronous jobs
fbpick_cache: dict[tuple, bytes] = {}
jobs: dict[str, dict[str, object]] = {}


def _filename_for_file_id(file_id: str) -> str | None:
	rec = FILE_REGISTRY.get(file_id) or {}
	path = rec.get('path') or rec.get('store_path')
	return Path(path).name if path else None


class LRUCache(OrderedDict):
	def __init__(self, capacity: int = 16):
		"""Initialize the cache with a maximum size."""
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


pipeline_tap_cache = LRUCache(16)
window_section_cache = LRUCache(32)

EXPECTED_SECTION_NDIM = 2


def _spec_uses_fbpick(spec: PipelineSpec) -> bool:
	"""Return True when ``spec`` contains an fbpick analyzer step."""
	return any(step.kind == 'analyzer' and step.name == 'fbpick' for step in spec.steps)


def _maybe_attach_fbpick_offsets(
	meta: dict[str, Any],
	*,
	spec: PipelineSpec,
	reader: SegySectionReader | TraceStoreSectionReader,
	key1_idx: int,
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
	offsets = get_offsets(key1_idx, offset_byte)
	if trace_slice is not None:
		offsets = offsets[trace_slice]
	offsets = np.ascontiguousarray(offsets, dtype=np.float32)
	if not meta:
		return {'offsets': offsets}
	meta_with_offsets = dict(meta)
	meta_with_offsets['offsets'] = offsets
	return meta_with_offsets


class PipelineTapNotFoundError(LookupError):
	"""Raised when a requested pipeline tap output is unavailable."""


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


def get_raw_section(
	*, file_id: str, key1_idx: int, key1_byte: int, key2_byte: int
) -> np.ndarray:
	"""Load the RAW seismic section as ``float32``."""
	reader = get_reader(file_id, key1_byte, key2_byte)
	section = reader.get_section(key1_idx)
	arr = np.asarray(section, dtype=np.float32)
	if arr.ndim != EXPECTED_SECTION_NDIM:
		msg = f'Raw section expected 2D data, got {arr.ndim}D'
		raise ValueError(msg)
	return np.ascontiguousarray(arr)


def get_section_from_pipeline_tap(
	*,
	file_id: str,
	key1_idx: int,
	key1_byte: int,
	pipeline_key: str,
	tap_label: str,
	offset_byte: int | None = None,
) -> np.ndarray:
	"""Return the cached pipeline tap output as a ``float32`` array."""
	base_key = (file_id, key1_idx, key1_byte, pipeline_key, None, offset_byte)
	payload = pipeline_tap_cache.get((*base_key, tap_label))
	if payload is None:
		msg = (
			f'Pipeline tap {tap_label!r} for pipeline {pipeline_key!r} '
			f'and key1={key1_idx} is not available. '
			'Please re-run the pipeline.'
		)
		raise PipelineTapNotFoundError(msg)
	return _pipeline_payload_to_array(payload, tap_label=tap_label)


def get_reader(
	file_id: str, key1_byte: int, key2_byte: int
) -> SegySectionReader | TraceStoreSectionReader:
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	if cache_key not in cached_readers:
		if file_id not in SEGYS:
			raise HTTPException(status_code=404, detail='File ID not found')
		path = SEGYS[file_id]
		p = Path(path)
		if p.is_dir():
			reader = TraceStoreSectionReader(p, key1_byte, key2_byte)
		else:
			reader = SegySectionReader(path, key1_byte, key2_byte)
		cached_readers[cache_key] = reader
	reader = cached_readers[cache_key]
	dt_val = get_dt_for_file(file_id)
	meta_attr = getattr(reader, 'meta', None)
	if isinstance(meta_attr, dict):
		if not isinstance(meta_attr.get('dt'), (int, float)) or meta_attr['dt'] <= 0:
			meta_attr['dt'] = dt_val
	else:
		try:
			reader.meta = {'dt': dt_val}
		except Exception:  # noqa: BLE001
			pass
	return reader


class Pick(BaseModel):
	file_id: str
	trace: int
	time: float
	key1_idx: int
	key1_byte: int


class FbpickRequest(BaseModel):
	file_id: str
	key1_idx: int
	key1_byte: int = 189
	key2_byte: int = 193
	offset_byte: int | None = None
	tile_h: int = 128
	tile_w: int = 6016
	overlap: int = 32
	amp: bool = True
	pipeline_key: str | None = None
	tap_label: str | None = None


class PipelineAllRequest(BaseModel):
	file_id: str
	key1_byte: int = 189
	key2_byte: int = 193
	offset_byte: int | None = None
	spec: PipelineSpec
	taps: list[str] = Field(default_factory=list)
	downsample_quicklook: bool = True


def _run_fbpick_job(job_id: str, req: FbpickRequest) -> None:
	job = jobs[job_id]
	job['status'] = 'running'
	try:
		# Force offset_byte if offset-enabled model is active
		forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

		cache_key = job['cache_key']
		section_override = job.pop('section_override', None)
		reader: SegySectionReader | TraceStoreSectionReader | None = None
		if USE_FBPICK_OFFSET and forced_offset_byte is not None:
			reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
		if section_override is not None:
			section = np.asarray(section_override, dtype=np.float32)
		elif req.pipeline_key and req.tap_label:
			section = get_section_from_pipeline_tap(
				file_id=req.file_id,
				key1_idx=req.key1_idx,
				key1_byte=req.key1_byte,
				pipeline_key=req.pipeline_key,
				tap_label=req.tap_label,
				offset_byte=forced_offset_byte,
			)
			section = np.asarray(section, dtype=np.float32)
		else:
			if reader is None:
				reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
			section = np.asarray(reader.get_section(req.key1_idx), dtype=np.float32)
		section = np.ascontiguousarray(section, dtype=np.float32)
		spec = PipelineSpec(
			steps=[
				{
					'kind': 'analyzer',
					'name': 'fbpick',
					'params': {
						'tile': (req.tile_h, req.tile_w),
						'overlap': req.overlap,
						'amp': req.amp,
					},
				}
			]
		)
		meta: dict[str, Any] = {}
		if reader is not None:
			meta = _maybe_attach_fbpick_offsets(
				meta,
				spec=spec,
				reader=reader,
				key1_idx=req.key1_idx,
				offset_byte=forced_offset_byte,
			)
		out = apply_pipeline(section, spec=spec, meta=meta, taps=None)
		prob = out['fbpick']['prob']
		scale, q = quantize_float32(prob, fixed_scale=127.0)
		payload = msgpack.packb(
			{
				'scale': scale,
				'shape': q.shape,
				'data': q.tobytes(),
			}
		)
		fbpick_cache[cache_key] = gzip.compress(payload)
		job['status'] = 'done'
	except Exception as e:  # noqa: BLE001
		job['status'] = 'error'
		job['message'] = str(e)


def _run_pipeline_all_job(job_id: str, req: PipelineAllRequest, pipe_key: str) -> None:
	job = jobs[job_id]
	job['status'] = 'running'
	try:
		reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
		key1_vals = reader.get_key1_values().tolist()
		total = len(key1_vals) or 1
		taps = req.taps
		for idx, key1_val in enumerate(key1_vals):
			section = np.array(reader.get_section(int(key1_val)), dtype=np.float32)
			dt = 0.002
			if hasattr(reader, 'meta'):
				dt = getattr(reader, 'meta', {}).get('dt', dt)
			meta = {'dt': dt}
			meta = _maybe_attach_fbpick_offsets(
				meta,
				spec=req.spec,
				reader=reader,
				key1_idx=int(key1_val),
				offset_byte=req.offset_byte,  # already forced by caller
			)
			out = apply_pipeline(section, spec=req.spec, meta=meta, taps=taps)
			base_key = (
				req.file_id,
				int(key1_val),
				req.key1_byte,
				pipe_key,
				None,
				req.offset_byte,  # forced
			)
			for k, v in out.items():
				val = v
				if req.downsample_quicklook and isinstance(v, np.ndarray):
					val = v[::4, ::4]
				pipeline_tap_cache.set((*base_key, k), to_builtin(val))
			job['progress'] = (idx + 1) / total
		job['status'] = 'done'
	except Exception as e:  # noqa: BLE001
		job['status'] = 'error'
		job['message'] = str(e)


@router.get('/get_key1_values')
def get_key1_values(
	file_id: str = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
):
	reader = get_reader(file_id, key1_byte, key2_byte)
	return JSONResponse(content={'values': reader.get_key1_values().tolist()})


@router.post('/open_segy')
async def open_segy(
	original_name: str = Form(...),
	key1_byte: int = Form(189),
	key2_byte: int = Form(193),
):
	safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', original_name)
	store_dir = TRACE_DIR / safe_name
	meta_path = store_dir / 'meta.json'
	if not meta_path.exists():
		raise HTTPException(
			status_code=404,
			detail=f'Trace store not found for {original_name}',
		)
	print(f'Opening existing trace store for {original_name}')
	file_id = str(uuid4())
	reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
	SEGYS[file_id] = str(store_dir)
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	cached_readers[cache_key] = reader
	threading.Thread(target=reader.preload_all_sections, daemon=True).start()
	for b in {key1_byte, key2_byte}:
		threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
	segy_path = (
		reader.meta.get('original_segy_path') if isinstance(reader.meta, dict) else None
	)
	dt_meta = None
	if isinstance(reader.meta, dict):
		dt_meta = reader.meta.get('dt')
	if (
		dt_meta is None or not isinstance(dt_meta, (int, float)) or dt_meta <= 0
	) and isinstance(segy_path, str):
		dt_meta = read_segy_dt_seconds(segy_path)
	_update_file_registry(
		file_id,
		path=segy_path if isinstance(segy_path, str) else None,
		store_path=str(store_dir),
		dt=dt_meta,
	)
	return {'file_id': file_id, 'reused_trace_store': True}


@router.post('/upload_segy')
async def upload_segy(
	file: UploadFile = File(...),
	key1_byte: int = Form(189),
	key2_byte: int = Form(193),
):
	if not file.filename:
		raise HTTPException(
			status_code=400, detail='Uploaded file must have a filename'
		)
	print(f'Uploading file: {file.filename}')
	safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', file.filename)
	store_dir = TRACE_DIR / safe_name
	meta_path = store_dir / 'meta.json'
	file_id = str(uuid4())

	if meta_path.exists():
		print(f'Reusing trace store for {file.filename}')
		reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
		SEGYS[file_id] = str(store_dir)
		cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
		cached_readers[cache_key] = reader
		threading.Thread(target=reader.preload_all_sections, daemon=True).start()
		for b in {key1_byte, key2_byte}:
			threading.Thread(
				target=reader.ensure_header, args=(b,), daemon=True
			).start()
		segy_path = (
			reader.meta.get('original_segy_path')
			if isinstance(reader.meta, dict)
			else None
		)
		dt_meta = None
		if isinstance(reader.meta, dict):
			dt_meta = reader.meta.get('dt')
		if (
			dt_meta is None or not isinstance(dt_meta, (int, float)) or dt_meta <= 0
		) and isinstance(segy_path, str):
			dt_meta = read_segy_dt_seconds(segy_path)
		_update_file_registry(
			file_id,
			path=segy_path if isinstance(segy_path, str) else None,
			store_path=str(store_dir),
			dt=dt_meta,
		)
		return {'file_id': file_id, 'reused_trace_store': True}

	raw_path = UPLOAD_DIR / safe_name
	data = await file.read()
	await asyncio.to_thread(raw_path.write_bytes, data)
	store_dir.mkdir(parents=True, exist_ok=True)
	traces_tmp = store_dir / 'traces.npy.tmp'
	dt_seconds = read_segy_dt_seconds(str(raw_path)) or 0.002

	with segyio.open(raw_path, 'r', ignore_geometry=True) as segy:
		segy.mmap()
		n_traces = segy.tracecount
		n_samples = len(segy.trace[0])
	mm = np.lib.format.open_memmap(
		traces_tmp,
		mode='w+',
		dtype=np.float32,
		shape=(n_traces, n_samples),
	)
	for i in range(n_traces):
		tr = segy.trace[i].astype(np.float32)
		mean = tr.mean()
		std = tr.std()
		if std == 0:
			std = 1.0
		mm[i] = (tr - mean) / std
	del mm
	traces_tmp.replace(store_dir / 'traces.npy')
	meta = {
		'n_traces': int(n_traces),
		'n_samples': int(n_samples),
		'original_segy_path': str(raw_path),
		'version': 1,
		'normalized': True,
		'dt': dt_seconds,
	}
	tmp_meta = store_dir / 'meta.json.tmp'
	tmp_meta.write_text(json.dumps(meta))
	tmp_meta.replace(meta_path)

	reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
	SEGYS[file_id] = str(store_dir)
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	cached_readers[cache_key] = reader
	threading.Thread(target=reader.preload_all_sections, daemon=True).start()
	for b in {key1_byte, key2_byte}:
		threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
	_update_file_registry(
		file_id,
		path=str(raw_path),
		store_path=str(store_dir),
		dt=dt_seconds,
	)
	return {'file_id': file_id, 'reused_trace_store': False}


@router.get('/get_section')
def get_section(
	file_id: str = Query(...),
	key1_byte: int = Query(189),  # デフォルト設定
	key2_byte: int = Query(193),
	key1_idx: int = Query(...),
):
	try:
		reader = get_reader(file_id, key1_byte, key2_byte)
		section = reader.get_section(key1_idx)
		return JSONResponse(content={'section': section})

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e)) from e


@router.get('/get_section_bin')
def get_section_bin(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
):
	try:
		reader = get_reader(file_id, key1_byte, key2_byte)
		section = np.array(reader.get_section(key1_idx), dtype=np.float32)
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
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e)) from e


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
):
	# Force offset_byte for pipeline tap cache alignment when offset-enabled
	forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else offset_byte

	cache_key = (
		file_id,
		key1_idx,
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
				key1_idx=key1_idx,
				key1_byte=key1_byte,
				pipeline_key=pipeline_key,
				tap_label=tap_label,
				offset_byte=forced_offset_byte,
			)
		else:
			reader = get_reader(file_id, key1_byte, key2_byte)
			section = np.array(reader.get_section(key1_idx), dtype=np.float32)
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
	obj = {
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


@router.post('/fbpick_section_bin')
def fbpick_section_bin(req: FbpickRequest):
	if not FBPICK_MODEL_PATH.exists():
		raise HTTPException(status_code=409, detail='FB pick model weights not found')

	# Force offset_byte if offset-enabled model is active
	forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

	pipeline_key = req.pipeline_key
	tap_label = req.tap_label
	cache_key = (
		req.file_id,
		req.key1_idx,
		req.key1_byte,
		req.key2_byte,
		forced_offset_byte,
		req.tile_h,
		req.tile_w,
		req.overlap,
		bool(req.amp),
		pipeline_key,
		tap_label,
		'fbpick',
	)
	section_override: np.ndarray | None = None
	wants_pipeline = bool(pipeline_key and tap_label)
	if cache_key not in fbpick_cache and wants_pipeline:
		try:
			section_override = get_section_from_pipeline_tap(
				file_id=req.file_id,
				key1_idx=req.key1_idx,
				key1_byte=req.key1_byte,
				pipeline_key=pipeline_key,
				tap_label=tap_label,
				offset_byte=forced_offset_byte,
			)
		except PipelineTapNotFoundError as exc:
			raise HTTPException(status_code=409, detail=str(exc)) from exc
		except (TypeError, ValueError) as exc:
			raise HTTPException(status_code=409, detail=str(exc)) from exc
	job_id = str(uuid4())
	job_state: dict[str, object] = {'status': 'queued', 'cache_key': cache_key}
	if section_override is not None:
		job_state['section_override'] = section_override
	jobs[job_id] = job_state

	# Ensure the worker sees the forced value
	req2 = req.copy(update={'offset_byte': forced_offset_byte})

	if cache_key in fbpick_cache:
		jobs[job_id]['status'] = 'done'
	else:
		threading.Thread(
			target=_run_fbpick_job, args=(job_id, req2), daemon=True
		).start()
	return {'job_id': job_id, 'status': jobs[job_id]['status']}


@router.get('/fbpick_job_status')
def fbpick_job_status(job_id: str = Query(...)):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail='Job ID not found')
	return {'status': job.get('status', 'unknown'), 'message': job.get('message', '')}


@router.get('/get_fbpick_section_bin')
def get_fbpick_section_bin(job_id: str = Query(...)):
	job = jobs.get(job_id)
	if job is None or job.get('status') != 'done':
		raise HTTPException(status_code=404, detail='Result not ready')
	cache_key = job.get('cache_key')
	payload = fbpick_cache.get(cache_key)
	if payload is None:
		raise HTTPException(status_code=404, detail='Result missing')
	return Response(
		payload,
		media_type='application/octet-stream',
		headers={'Content-Encoding': 'gzip'},
	)


@router.post('/pipeline/section', response_model=PipelineSectionResponse)
def pipeline_section(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
	offset_byte: int | None = Query(None),
	spec: PipelineSpec = Body(...),
	taps: list[str] | None = Body(default=None),
	window: dict[str, int | float] | None = Body(default=None),
):
	reader = get_reader(file_id, key1_byte, key2_byte)
	section = np.array(reader.get_section(key1_idx), dtype=np.float32)
	trace_slice: slice | None = None
	window_hash = None
	if window:
		tr_min = int(window.get('tr_min', 0))
		tr_max = int(window.get('tr_max', section.shape[0]))
		t_min = int(window.get('t_min', 0))
		t_max = int(window.get('t_max', section.shape[1]))
		section = section[tr_min:tr_max, t_min:t_max]
		trace_slice = slice(tr_min, tr_max)
		clean_window = {
			'tr_min': tr_min,
			'tr_max': tr_max,
			't_min': t_min,
			't_max': t_max,
		}
		window_hash = hashlib.sha256(
			json.dumps(clean_window, sort_keys=True).encode()
		).hexdigest()[:8]
	dt = 0.002
	if hasattr(reader, 'meta'):
		dt = getattr(reader, 'meta', {}).get('dt', dt)
	meta = {'dt': dt}

	# Force offset_byte for fbpick use
	forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

	meta = _maybe_attach_fbpick_offsets(
		meta,
		spec=spec,
		reader=reader,
		key1_idx=key1_idx,
		offset_byte=forced_offset_byte,
		trace_slice=trace_slice,
	)
	pipe_key = pipeline_key(spec)
	tap_names = taps or []
	if tap_names:
		base_key = (
			file_id,
			key1_idx,
			key1_byte,
			pipe_key,
			window_hash,
			forced_offset_byte,
		)
		taps_out: dict[str, object] = {}
		misses: list[str] = []
		for tap in tap_names:
			payload = pipeline_tap_cache.get((*base_key, tap))
			if payload is not None:
				taps_out[tap] = payload
			else:
				misses.append(tap)
		if misses:
			out = apply_pipeline(section, spec=spec, meta=meta, taps=misses)
			for k, v in out.items():
				val = to_builtin(v)
				taps_out[k] = val
				pipeline_tap_cache.set((*base_key, k), val)
		return {'taps': taps_out, 'pipeline_key': pipe_key}
	out = apply_pipeline(section, spec=spec, meta=meta, taps=None)
	return {'taps': to_builtin(out), 'pipeline_key': pipe_key}


@router.post('/pipeline/all', response_model=PipelineAllResponse)
def pipeline_all(
	file_id: str = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
	downsample_quicklook: Annotated[bool, Query()] = True,
	offset_byte: int | None = Query(None),
	spec: PipelineSpec = Body(...),
	taps: list[str] | None = Body(default=None),
):
	tap_names = taps or []

	# Force offset_byte for the entire pipeline if applicable
	forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None

	req = PipelineAllRequest(
		file_id=file_id,
		key1_byte=key1_byte,
		key2_byte=key2_byte,
		offset_byte=forced_offset_byte,
		spec=spec,
		taps=tap_names,
		downsample_quicklook=downsample_quicklook,
	)
	job_id = str(uuid4())
	pipe_key = pipeline_key(spec)
	jobs[job_id] = {
		'status': 'queued',
		'progress': 0.0,
		'message': '',
		'file_id': file_id,
		'key1_byte': key1_byte,
		'pipeline_key': pipe_key,
		'offset_byte': forced_offset_byte,
	}
	threading.Thread(
		target=_run_pipeline_all_job, args=(job_id, req, pipe_key), daemon=True
	).start()
	return {'job_id': job_id, 'state': jobs[job_id]['status']}


@router.get('/pipeline/job/{job_id}/status', response_model=PipelineJobStatusResponse)  # noqa: FAST001
def pipeline_job_status(job_id: str) -> PipelineJobStatusResponse:
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail='Job ID not found')
	return {
		'state': job.get('status', 'unknown'),
		'progress': job.get('progress', 0.0),
		'message': job.get('message', ''),
	}


@router.get('/pipeline/job/{job_id}/artifact', response_model=Any)
def pipeline_job_artifact(
	job_id: str,
	key1_idx: int = Query(...),
	tap: str = Query(...),
):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail='Job ID not found')
	base_key = (
		job.get('file_id'),
		key1_idx,
		job.get('key1_byte'),
		job.get('pipeline_key'),
		None,
		job.get('offset_byte'),
	)
	payload = pipeline_tap_cache.get((*base_key, tap))
	if payload is None:
		raise HTTPException(status_code=404, detail='Artifact not ready')
	return payload


@router.post('/picks')
async def post_pick(pick: Pick) -> dict[str, str]:
	add_pick(pick.file_id, pick.trace, pick.time, pick.key1_idx, pick.key1_byte)
	await asyncio.to_thread(store.save)
	fname = _filename_for_file_id(pick.file_id)
	if fname:
		try:
			picks_by_name.add_pick(
				fname,
				pick.trace,
				pick.time,
				pick.key1_idx,
				pick.key1_byte,
			)
			await asyncio.to_thread(picks_by_name.save)
		except Exception:  # noqa: BLE001
			pass
	return {'status': 'ok'}


@router.get('/picks')
async def get_pick(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	key1_byte: int = Query(...),
) -> dict[str, list[dict[str, int | float]]]:
	return {'picks': list_picks(file_id, key1_idx, key1_byte)}


@router.get('/picks/by-filename')
async def get_picks_by_filename(
	file_name: str = Query(...),
	key1_idx: int = Query(...),
	key1_byte: int = Query(...),
) -> dict[str, list[dict[str, int | float]]]:
	return {'picks': picks_by_name.list_picks(file_name, key1_idx, key1_byte)}


@router.delete('/picks')
async def delete_pick_route(
	file_id: str = Query(...),
	trace: int | None = Query(None),
	key1_idx: int = Query(...),
	key1_byte: int = Query(...),
) -> dict[str, str]:
	delete_pick(file_id, trace, key1_idx, key1_byte)
	await asyncio.to_thread(store.save)
	fname = _filename_for_file_id(file_id)
	if fname:
		try:
			picks_by_name.delete_pick(fname, trace, key1_idx, key1_byte)
			await asyncio.to_thread(picks_by_name.save)
		except Exception:  # noqa: BLE001
			pass
	return {'status': 'ok'}
