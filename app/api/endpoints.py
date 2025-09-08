# endpoint.py
import asyncio
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import threading
from pathlib import Path
from typing import Literal
from uuid import uuid4

import msgpack
import numpy as np
import segyio
import torch
from fastapi import (
	APIRouter,
	File,
	Form,  # 忘れずにインポート
	HTTPException,
	Query,
	UploadFile,
)
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from utils.bandpass import bandpass_np
from utils.denoise import denoise_tensor
from utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from utils.fbpick import infer_prob_map
from utils.picks import add_pick, delete_pick, list_picks, store
from utils.utils import (
	SegySectionReader,
	TraceStoreSectionReader,
	quantize_float32,
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
jobs: dict[str, dict[str, float | str]] = {}


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


class BandpassRequest(BaseModel):
	file_id: str
	key1_idx: int
	key1_byte: int = 189
	key2_byte: int = 193
	low_hz: float
	high_hz: float
	dt: float = 0.002
	taper: float = 0.0


class BandpassApplyRequest(BaseModel):
	file_id: str
	scope: Literal['display', 'all_key1', 'by_header']
	key1_idx: int | None = None
	group_header_byte: int | None = None
	key1_byte: int = 189
	key2_byte: int = 193
	low_hz: float
	high_hz: float
	dt: float = 0.002
	taper: float = 0.0


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
        tile_w: int = 6000
        overlap: int = 32
        amp: bool = True


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
		for idx, key1_val in enumerate(key1_vals):
			cache_key = (req.file_id, int(key1_val), param_hash)
			if cache_key in denoise_cache:
				job['progress'] = (idx + 1) / total
				continue
			section = np.array(reader.get_section(int(key1_val)), dtype=np.float32)
			xt = torch.from_numpy(section).unsqueeze(0).unsqueeze(0)
			yt = denoise_tensor(
				xt,
				chunk_h=req.chunk_h,
				overlap=req.overlap,
				mask_ratio=req.mask_ratio,
				noise_std=req.noise_std,
				mask_noise_mode=req.mask_noise_mode,
				passes_batch=req.passes_batch,
			)
			denoised = yt.squeeze(0).squeeze(0).numpy()
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
		for idx, key1_val in enumerate(key1_vals):
			cache_key = (req.file_id, int(key1_val), param_hash)
			if cache_key in bandpass_cache:
				job['progress'] = (idx + 1) / total
				continue
			section = np.array(reader.get_section(int(key1_val)), dtype=np.float32)
			filtered = bandpass_np(
				section,
				low_hz=req.low_hz,
				high_hz=req.high_hz,
				dt=req.dt,
				taper=req.taper,
			)
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
		reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
		section = np.array(reader.get_section(req.key1_idx), dtype=np.float32)
		prob = infer_prob_map(
			section,
			amp=req.amp,
			tile=(req.tile_h, req.tile_w),
			overlap=req.overlap,
		)
		scale, q = quantize_float32(prob, fixed_scale=127.0)
		payload = msgpack.packb({
			'scale': scale,
			'shape': q.shape,
			'data': q.tobytes(),
		})
		fbpick_cache[cache_key] = gzip.compress(payload)
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


@router.post('/bandpass_section_bin')
def bandpass_section_bin(req: BandpassRequest):
	try:
		reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
		section = np.array(reader.get_section(req.key1_idx), dtype=np.float32)
		filtered = bandpass_np(
			section,
			low_hz=req.low_hz,
			high_hz=req.high_hz,
			dt=req.dt,
			taper=req.taper,
		)
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
		xt = torch.from_numpy(section).unsqueeze(0).unsqueeze(0)
		yt = denoise_tensor(
			xt,
			chunk_h=req.chunk_h,
			overlap=req.overlap,
			mask_ratio=req.mask_ratio,
			noise_std=req.noise_std,
			mask_noise_mode=req.mask_noise_mode,
			passes_batch=req.passes_batch,
		)
		denoised = yt.squeeze(0).squeeze(0).numpy()
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
		import sys
		import traceback

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
	cache_key = (
		req.file_id,
		req.key1_idx,
		req.key1_byte,
		req.tile_h,
		req.tile_w,
		req.overlap,
		bool(req.amp),
		'fbpick',
	)
	job_id = str(uuid4())
	jobs[job_id] = {'status': 'queued', "cache_key": cache_key}
	if cache_key in fbpick_cache:
		jobs[job_id]['status'] = 'done'
	else:
		threading.Thread(target=_run_fbpick_job, args=(job_id, req), daemon=True).start()
	return {'job_id': job_id, "status": jobs[job_id]['status']}


@router.get('/fbpick_job_status')
def fbpick_job_status(job_id: str = Query(...)):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail='Job ID not found')
	return {'status': job.get('status', 'unknown'), "message": job.get('message', "")}


@router.get('/get_fbpick_section_bin')
def get_fbpick_section_bin(job_id: str = Query(...)):
	job = jobs.get(job_id)
	if job is None or job.get('status') != 'done':
		raise HTTPException(status_code=404, detail='Result not ready')
	cache_key = job.get('cache_key')
	payload = fbpick_cache.get(cache_key)
	if payload is None:
		raise HTTPException(status_code=404, detail='Result missing')
	return Response(payload, media_type='application/octet-stream', headers={'Content-Encoding': "gzip"})


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
