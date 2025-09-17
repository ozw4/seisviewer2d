# endpoint.py
import asyncio
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import threading
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import msgpack
import numpy as np
import segyio
from api.schemas import (
	BandpassParams,
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
from utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from utils.picks import add_pick, delete_pick, list_picks, store
from utils.pipeline import apply_pipeline, pipeline_key
from utils.utils import (
	SegySectionReader,
	TraceStoreSectionReader,
	quantize_float32,
	to_builtin,
)

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = UPLOAD_DIR / 'processed'
DENOISE_DIR = PROCESSED_DIR / 'denoise'
DENOISE_DIR.mkdir(parents=True, exist_ok=True)

TRACE_DIR = PROCESSED_DIR / 'traces'
TRACE_DIR.mkdir(parents=True, exist_ok=True)

LATEST_DIR = DENOISE_DIR / 'latest'
LATEST_DIR.mkdir(parents=True, exist_ok=True)


def _denoise_path(file_id: str, key1_idx: int, param_hash: str) -> Path:
	safe_id = str(file_id).replace('/', '_')
	return DENOISE_DIR / safe_id / param_hash / f'{key1_idx}.bin.gz'


def _denoise_latest_path(file_id: str, key1_idx: int) -> Path:
	safe_id = str(file_id).replace('/', '_')
	return LATEST_DIR / safe_id / f'{key1_idx}.bin.gz'


cached_readers: dict[str, SegySectionReader | TraceStoreSectionReader] = {}
SEGYS: dict[str, str] = {}

# Private caches for denoised sections, band-passed sections and asynchronous jobs
denoise_cache: dict[tuple, bytes] = {}
bandpass_cache: dict[tuple[str, int, str], bytes] = {}
fbpick_cache: dict[tuple, bytes] = {}
jobs: dict[str, dict[str, object]] = {}


class LRUCache(OrderedDict):
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


pipeline_tap_cache = LRUCache(16)
window_section_cache = LRUCache(32)


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
	if arr.ndim != 2:
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
	if arr.ndim != 2:
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
) -> np.ndarray:
	"""Return the cached pipeline tap output as a ``float32`` array."""
	base_key = (file_id, key1_idx, key1_byte, pipeline_key, None)
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
	return cached_readers[cache_key]


class Pick(BaseModel):
	file_id: str
	trace: int
	time: float
	key1_idx: int
	key1_byte: int


class BandpassRequest(BandpassParams):
	file_id: str
	key1_idx: int
	key1_byte: int = 189
	key2_byte: int = 193


class BandpassApplyRequest(BandpassParams):
	file_id: str
	scope: Literal['display', 'all_key1', 'by_header']
	key1_idx: int | None = None
	group_header_byte: int | None = None
	key1_byte: int = 189
	key2_byte: int = 193


class DenoiseRequest(BaseModel):
	file_id: str
	key1_idx: int
	key1_byte: int = 189
	key2_byte: int = 193
	chunk_h: int = 128
	overlap: int = 32
	mask_ratio: float = 0.5
	noise_std: float = 1.0
	mask_noise_mode: Literal['replace', 'add'] = 'replace'
	passes_batch: int = 4


class DenoiseApplyRequest(BaseModel):
	file_id: str
	scope: Literal['display', 'all_key1', 'by_header']
	key1_idx: int | None = None
	group_header_byte: int | None = None
	key1_byte: int = 189
	key2_byte: int = 193
	chunk_h: int = 128
	overlap: int = 32
	mask_ratio: float = 0.5
	noise_std: float = 1.0
	mask_noise_mode: Literal['replace', 'add'] = 'replace'
	passes_batch: int = 4


class FbpickRequest(BaseModel):
	file_id: str
	key1_idx: int
	key1_byte: int = 189
	key2_byte: int = 193
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
	spec: PipelineSpec
	taps: list[str] = Field(default_factory=list)
	downsample_quicklook: bool = True


def _run_denoise_job(job_id: str, req: DenoiseApplyRequest) -> None:
	job = jobs[job_id]
	try:
		reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
		if req.scope == 'display':
			if req.key1_idx is None:
				msg = 'key1_idx is required for display scope'
				raise ValueError(msg)
			key1_vals = [req.key1_idx]
		elif req.scope == 'all_key1':
			key1_vals = reader.get_key1_values().tolist()
		else:
			msg = 'by_header scope not implemented'
			raise ValueError(msg)
		total = len(key1_vals) or 1
		params = {
			'chunk_h': req.chunk_h,
			'overlap': req.overlap,
			'mask_ratio': req.mask_ratio,
			'noise_std': req.noise_std,
			'mask_noise_mode': req.mask_noise_mode,
			'passes_batch': req.passes_batch,
		}
		param_hash = hashlib.sha256(
			json.dumps(params, sort_keys=True).encode('utf-8')
		).hexdigest()
		spec = PipelineSpec(
			steps=[
				{
					'kind': 'transform',
					'name': 'denoise',
					'params': {
						'chunk_h': req.chunk_h,
						'overlap': req.overlap,
						'mask_ratio': req.mask_ratio,
						'noise_std': req.noise_std,
						'mask_noise_mode': req.mask_noise_mode,
						'passes_batch': req.passes_batch,
					},
				}
			]
		)
		for idx, key1_val in enumerate(key1_vals):
			cache_key = (req.file_id, int(key1_val), param_hash)
			if cache_key in denoise_cache:
				job['progress'] = (idx + 1) / total
				continue
			section = np.array(reader.get_section(int(key1_val)), dtype=np.float32)
			out = apply_pipeline(section, spec=spec, meta={}, taps=['denoise'])
			denoised = out['denoise']['data']
			scale, q = quantize_float32(denoised)
			payload = msgpack.packb(
				{
					'scale': scale,
					'shape': q.shape,
					'data': q.tobytes(),
				}
			)
			gz = gzip.compress(payload)
			p = _denoise_path(req.file_id, int(key1_val), param_hash)
			p.parent.mkdir(parents=True, exist_ok=True)
			p.write_bytes(gz)
			p_latest = _denoise_latest_path(req.file_id, int(key1_val))
			p_latest.parent.mkdir(parents=True, exist_ok=True)
			tmp = p_latest.with_suffix('.tmp')
			tmp.write_bytes(gz)
			tmp.replace(p_latest)
			denoise_cache[cache_key] = gz
			denoise_cache[(req.file_id, int(key1_val))] = gz
			try:
				base = DENOISE_DIR / str(req.file_id).replace('/', '_')
				for child in base.iterdir():
					if child.is_dir() and child.name not in {param_hash, 'latest'}:
						shutil.rmtree(child, ignore_errors=True)
			except Exception:
				pass
			job['progress'] = (idx + 1) / total
		job['status'] = 'done'
	except Exception as e:
		job['status'] = 'error'
		job['message'] = str(e)


def _run_bandpass_job(job_id: str, req: BandpassApplyRequest) -> None:
	job = jobs[job_id]
	try:
		reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
		if req.scope == 'display':
			if req.key1_idx is None:
				msg = 'key1_idx is required for display scope'
				raise ValueError(msg)
			key1_vals = [req.key1_idx]
		elif req.scope == 'all_key1':
			key1_vals = reader.get_key1_values().tolist()
		else:
			msg = 'by_header scope not implemented'
			raise ValueError(msg)
		total = len(key1_vals) or 1
		params = {
			'low_hz': req.low_hz,
			'high_hz': req.high_hz,
			'dt': req.dt,
			'taper': req.taper,
		}
		param_hash = hashlib.sha256(
			json.dumps(params, sort_keys=True).encode('utf-8')
		).hexdigest()
		spec = PipelineSpec(
			steps=[
				{
					'kind': 'transform',
					'name': 'bandpass',
					'params': {
						'low_hz': req.low_hz,
						'high_hz': req.high_hz,
						'dt': req.dt,
						'taper': req.taper,
					},
				}
			]
		)
		for idx, key1_val in enumerate(key1_vals):
			cache_key = (req.file_id, int(key1_val), param_hash)
			if cache_key in bandpass_cache:
				job['progress'] = (idx + 1) / total
				continue
			section = np.array(reader.get_section(int(key1_val)), dtype=np.float32)
			out = apply_pipeline(section, spec=spec, meta={}, taps=['bandpass'])
			filtered = out['bandpass']['data']
			scale, q = quantize_float32(filtered)
			payload = msgpack.packb(
				{
					'scale': scale,
					'shape': q.shape,
					'data': q.tobytes(),
				}
			)
			bandpass_cache[cache_key] = gzip.compress(payload)
			job['progress'] = (idx + 1) / total
		job['status'] = 'done'
	except Exception as e:
		job['status'] = 'error'
		job['message'] = str(e)


def _run_fbpick_job(job_id: str, req: FbpickRequest) -> None:
	job = jobs[job_id]
	job['status'] = 'running'
	try:
		cache_key = job['cache_key']
		section_override = job.pop('section_override', None)
		if section_override is not None:
			section = np.asarray(section_override, dtype=np.float32)
		elif req.pipeline_key and req.tap_label:
			section = get_section_from_pipeline_tap(
				file_id=req.file_id,
				key1_idx=req.key1_idx,
				key1_byte=req.key1_byte,
				pipeline_key=req.pipeline_key,
				tap_label=req.tap_label,
			)
		else:
			section = get_raw_section(
				file_id=req.file_id,
				key1_idx=req.key1_idx,
				key1_byte=req.key1_byte,
				key2_byte=req.key2_byte,
			)
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
		out = apply_pipeline(section, spec=spec, meta={}, taps=None)
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
	except Exception as e:
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
			out = apply_pipeline(section, spec=req.spec, meta=meta, taps=taps)
			base_key = (
				req.file_id,
				int(key1_val),
				req.key1_byte,
				pipe_key,
				None,
			)
			for k, v in out.items():
				val = v
				if req.downsample_quicklook and isinstance(v, np.ndarray):
					val = v[::4, ::4]
				pipeline_tap_cache.set((*base_key, k), to_builtin(val))
			job['progress'] = (idx + 1) / total
		job['status'] = 'done'
	except Exception as e:
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
		return {'file_id': file_id, 'reused_trace_store': True}

	raw_path = UPLOAD_DIR / safe_name
	with open(raw_path, 'wb') as f:
		f.write(await file.read())
	store_dir.mkdir(parents=True, exist_ok=True)
	traces_tmp = store_dir / 'traces.npy.tmp'
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
	os.replace(traces_tmp, store_dir / 'traces.npy')
	meta = {
		'n_traces': int(n_traces),
		'n_samples': int(n_samples),
		'original_segy_path': str(raw_path),
		'version': 1,
		'normalized': True,
	}
	tmp_meta = store_dir / 'meta.json.tmp'
	tmp_meta.write_text(json.dumps(meta))
	os.replace(tmp_meta, meta_path)

	reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
	SEGYS[file_id] = str(store_dir)
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	cached_readers[cache_key] = reader
	threading.Thread(target=reader.preload_all_sections, daemon=True).start()
	for b in {key1_byte, key2_byte}:
		threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
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
		raise HTTPException(status_code=500, detail=str(e))


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
                payload = msgpack.packb(
                        {
                                'scale': scale,
                                'shape': q.shape,
                                'data': q.tobytes(),
                        }
                )
                return Response(
                        gzip.compress(payload),
                        media_type='application/octet-stream',
                        headers={'Content-Encoding': 'gzip'},
                )
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


@router.get('/get_section_window_bin')
def get_section_window_bin(
        file_id: str = Query(...),
        key1_idx: int = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
        x0: int = Query(...),
        x1: int = Query(...),
        y0: int = Query(...),
        y1: int = Query(...),
        step_x: int = Query(1, ge=1),
        step_y: int = Query(1, ge=1),
        pipeline_key: str | None = Query(None),
        tap_label: str | None = Query(None),
):
        cache_key = (
                file_id,
                key1_idx,
                key1_byte,
                key2_byte,
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
                        )
                else:
                        reader = get_reader(file_id, key1_byte, key2_byte)
                        section = np.array(reader.get_section(key1_idx), dtype=np.float32)
        except PipelineTapNotFoundError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        section = np.ascontiguousarray(section, dtype=np.float32)
        if section.ndim != 2:
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
        payload = msgpack.packb(
                {
                        'scale': scale,
                        'shape': window_view.shape,
                        'data': q.tobytes(),
                }
        )
        compressed = gzip.compress(payload)
        window_section_cache.set(cache_key, compressed)
        return Response(
                compressed,
                media_type='application/octet-stream',
                headers={'Content-Encoding': 'gzip'},
        )


@router.post('/bandpass_section_bin')
def bandpass_section_bin(req: BandpassRequest):
        try:
                reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
                section = np.array(reader.get_section(req.key1_idx), dtype=np.float32)
		spec = PipelineSpec(
			steps=[
				{
					'kind': 'transform',
					'name': 'bandpass',
					'params': {
						'low_hz': req.low_hz,
						'high_hz': req.high_hz,
						'dt': req.dt,
						'taper': req.taper,
					},
				}
			]
		)
		out = apply_pipeline(section, spec=spec, meta={}, taps=['bandpass'])
		filtered = out['bandpass']['data']
		scale, q = quantize_float32(filtered)
		payload = msgpack.packb(
			{
				'scale': scale,
				'shape': q.shape,
				'data': q.tobytes(),
			}
		)
		return Response(
			gzip.compress(payload),
			media_type='application/octet-stream',
			headers={'Content-Encoding': 'gzip'},
		)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post('/bandpass_apply')
def bandpass_apply(req: BandpassApplyRequest):
	if req.scope == 'by_header':
		raise HTTPException(status_code=400, detail='by_header scope not implemented')
	job_id = str(uuid4())
	jobs[job_id] = {'status': 'running', 'progress': 0.0, 'message': ''}
	threading.Thread(target=_run_bandpass_job, args=(job_id, req), daemon=True).start()
	return {'job_id': job_id}


@router.get('/bandpass_job_status')
def bandpass_job_status(job_id: str = Query(...)):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail='Job ID not found')
	return {
		'status': job.get('status', 'unknown'),
		'progress': job.get('progress', 0.0),
		'message': job.get('message', ''),
	}


@router.get('/get_bandpassed_section_bin')
def get_bandpassed_section_bin(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	low_hz: float = Query(...),
	high_hz: float = Query(...),
	dt: float = Query(0.002),
	taper: float = Query(0.0),
):
	params = {'low_hz': low_hz, 'high_hz': high_hz, 'dt': dt, 'taper': taper}
	param_hash = hashlib.sha256(
		json.dumps(params, sort_keys=True).encode('utf-8')
	).hexdigest()
	cache_key = (file_id, key1_idx, param_hash)
	payload = bandpass_cache.get(cache_key)
	if payload is None:
		raise HTTPException(status_code=404, detail='Section not processed')
	return Response(
		payload,
		media_type='application/octet-stream',
		headers={'Content-Encoding': 'gzip'},
	)


@router.post('/denoise_section_bin')
def denoise_section_bin(req: DenoiseRequest):
	try:
		reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
		section = np.array(reader.get_section(req.key1_idx), dtype=np.float32)
		spec = PipelineSpec(
			steps=[
				{
					'kind': 'transform',
					'name': 'denoise',
					'params': {
						'chunk_h': req.chunk_h,
						'overlap': req.overlap,
						'mask_ratio': req.mask_ratio,
						'noise_std': req.noise_std,
						'mask_noise_mode': req.mask_noise_mode,
						'passes_batch': req.passes_batch,
					},
				}
			]
		)
		out = apply_pipeline(section, spec=spec, meta={}, taps=['denoise'])
		denoised = out['denoise']['data']
		scale, q = quantize_float32(denoised)
		payload = msgpack.packb(
			{
				'scale': scale,
				'shape': q.shape,
				'data': q.tobytes(),
			}
		)
		params = {
			'chunk_h': req.chunk_h,
			'overlap': req.overlap,
			'mask_ratio': req.mask_ratio,
			'noise_std': req.noise_std,
			'mask_noise_mode': req.mask_noise_mode,
			'passes_batch': req.passes_batch,
		}
		param_hash = hashlib.sha256(
			json.dumps(params, sort_keys=True).encode('utf-8')
		).hexdigest()
		cache_key = (req.file_id, req.key1_idx, param_hash)
		gz = gzip.compress(payload)
		p = _denoise_path(req.file_id, req.key1_idx, param_hash)
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_bytes(gz)
		p_latest = _denoise_latest_path(req.file_id, req.key1_idx)
		p_latest.parent.mkdir(parents=True, exist_ok=True)
		tmp = p_latest.with_suffix('.tmp')
		tmp.write_bytes(gz)
		tmp.replace(p_latest)
		denoise_cache[cache_key] = gz
		denoise_cache[(req.file_id, req.key1_idx)] = gz
		try:
			base = DENOISE_DIR / str(req.file_id).replace('/', '_')
			for child in base.iterdir():
				if child.is_dir() and child.name not in {param_hash, 'latest'}:
					shutil.rmtree(child, ignore_errors=True)
		except Exception:
			pass
		return Response(
			gz,
			media_type='application/octet-stream',
			headers={'Content-Encoding': 'gzip'},
		)
	except Exception as e:
		traceback.print_exc(file=sys.stderr)
		raise HTTPException(status_code=500, detail=str(e))


@router.post('/denoise_apply')
def denoise_apply(req: DenoiseApplyRequest):
	if req.scope == 'by_header':
		raise HTTPException(status_code=400, detail='by_header scope not implemented')
	job_id = str(uuid4())
	jobs[job_id] = {'status': 'running', 'progress': 0.0, 'message': ''}
	threading.Thread(target=_run_denoise_job, args=(job_id, req), daemon=True).start()
	return {'job_id': job_id}


@router.get('/denoise_job_status')
def denoise_job_status(job_id: str = Query(...)):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail='Job ID not found')
	return {
		'status': job.get('status', 'unknown'),
		'progress': job.get('progress', 0.0),
		'message': job.get('message', ''),
	}


@router.get('/get_denoised_section_bin')
def get_denoised_section_bin(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	chunk_h: int = Query(128),
	overlap: int = Query(32),
	mask_ratio: float = Query(0.5),
	noise_std: float = Query(1.0),
	mask_noise_mode: Literal['replace', 'add'] = Query('replace'),
	passes_batch: int = Query(4),
):
	cache_key_latest = (file_id, key1_idx)
	payload = denoise_cache.get(cache_key_latest)
	if payload is None:
		p_latest = _denoise_latest_path(file_id, key1_idx)
		if p_latest.exists():
			payload = p_latest.read_bytes()
			denoise_cache[cache_key_latest] = payload
	if payload is None:
		params = {
			'chunk_h': chunk_h,
			'overlap': overlap,
			'mask_ratio': mask_ratio,
			'noise_std': noise_std,
			'mask_noise_mode': mask_noise_mode,
			'passes_batch': passes_batch,
		}
		param_hash = hashlib.sha256(
			json.dumps(params, sort_keys=True).encode('utf-8')
		).hexdigest()
		cache_key = (file_id, key1_idx, param_hash)
		payload = denoise_cache.get(cache_key)
		if payload is None:
			p = _denoise_path(file_id, key1_idx, param_hash)
			if p.exists():
				payload = p.read_bytes()
				denoise_cache[(file_id, key1_idx, param_hash)] = payload
		if payload is None:
			raise HTTPException(status_code=404, detail='Section not processed')
	return Response(
		payload,
		media_type='application/octet-stream',
		headers={'Content-Encoding': 'gzip'},
	)


@router.post('/fbpick_section_bin')
def fbpick_section_bin(req: FbpickRequest):
	if not FBPICK_MODEL_PATH.exists():
		raise HTTPException(status_code=409, detail='FB pick model weights not found')
	pipeline_key = req.pipeline_key
	tap_label = req.tap_label
	cache_key = (
		req.file_id,
		req.key1_idx,
		req.key1_byte,
		req.key2_byte,
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
		assert pipeline_key is not None
		assert tap_label is not None
		try:
			section_override = get_section_from_pipeline_tap(
				file_id=req.file_id,
				key1_idx=req.key1_idx,
				key1_byte=req.key1_byte,
				pipeline_key=pipeline_key,
				tap_label=tap_label,
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
	if cache_key in fbpick_cache:
		jobs[job_id]['status'] = 'done'
	else:
		threading.Thread(
			target=_run_fbpick_job, args=(job_id, req), daemon=True
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
	spec: PipelineSpec = Body(...),
	taps: list[str] | None = Body(default=None),
	window: dict[str, int | float] | None = Body(default=None),
):
	reader = get_reader(file_id, key1_byte, key2_byte)
	section = np.array(reader.get_section(key1_idx), dtype=np.float32)
	window_hash = None
	if window:
		tr_min = int(window.get('tr_min', 0))
		tr_max = int(window.get('tr_max', section.shape[0]))
		t_min = int(window.get('t_min', 0))
		t_max = int(window.get('t_max', section.shape[1]))
		section = section[tr_min:tr_max, t_min:t_max]
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
	pipe_key = pipeline_key(spec)
	tap_names = taps or []
	if tap_names:
		base_key = (file_id, key1_idx, key1_byte, pipe_key, window_hash)
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
	spec: PipelineSpec = Body(...),
	taps: list[str] | None = Body(default=None),
	downsample_quicklook: bool = Query(True),
):
	tap_names = taps or []
	req = PipelineAllRequest(
		file_id=file_id,
		key1_byte=key1_byte,
		key2_byte=key2_byte,
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
	}
	threading.Thread(
		target=_run_pipeline_all_job, args=(job_id, req, pipe_key), daemon=True
	).start()
	return {'job_id': job_id, 'state': jobs[job_id]['status']}


@router.get('/pipeline/job/{job_id}/status', response_model=PipelineJobStatusResponse)
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
	)
	payload = pipeline_tap_cache.get((*base_key, tap))
	if payload is None:
		raise HTTPException(status_code=404, detail='Artifact not ready')
	return payload


@router.post('/picks')
async def post_pick(pick: Pick) -> dict[str, str]:
	add_pick(pick.file_id, pick.trace, pick.time, pick.key1_idx, pick.key1_byte)
	await asyncio.to_thread(store.save)
	return {'status': 'ok'}


@router.get('/picks')
async def get_pick(
	file_id: str = Query(...),
	key1_idx: int = Query(...),
	key1_byte: int = Query(...),
) -> dict[str, list[dict[str, int | float]]]:
	return {'picks': list_picks(file_id, key1_idx, key1_byte)}


@router.delete('/picks')
async def delete_pick_route(
	file_id: str = Query(...),
	trace: int | None = Query(None),
	key1_idx: int = Query(...),
	key1_byte: int = Query(...),
) -> dict[str, str]:
	delete_pick(file_id, trace, key1_idx, key1_byte)
	await asyncio.to_thread(store.save)
	return {'status': 'ok'}
