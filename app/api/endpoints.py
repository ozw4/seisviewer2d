# endpoint.py
import asyncio
import gzip
import hashlib
import json
import pathlib
import threading
from typing import Literal
from uuid import uuid4

import msgpack
import numpy as np
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
from utils.denoise import denoise_tensor
from utils.picks import add_pick, delete_pick, list_picks, store
from utils.utils import SegySectionReader, quantize_float32

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

cached_readers: dict[str, SegySectionReader] = {}
SEGYS: dict[str, str] = {}

# Private caches for denoised sections and asynchronous jobs
denoise_cache: dict[tuple[str, int, str], bytes] = {}
jobs: dict[str, dict[str, float | str]] = {}


def get_reader(file_id: str, key1_byte: int, key2_byte: int) -> SegySectionReader:
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	if cache_key not in cached_readers:
		if file_id not in SEGYS:
			raise HTTPException(status_code=404, detail='File ID not found')
		path = SEGYS[file_id]
		cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)
	return cached_readers[cache_key]


class Pick(BaseModel):
	file_id: str
	trace: int
	time: float
	key1_idx: int
	key1_byte: int


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
			denoise_cache[cache_key] = gzip.compress(payload)
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
	ext = pathlib.Path(file.filename).suffix.lower()
	file_id = str(uuid4())
	dest_path = UPLOAD_DIR / f'{file_id}{ext}'
	with open(dest_path, 'wb') as f:
		f.write(await file.read())

	SEGYS[file_id] = str(dest_path)

	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	print(f'Creating cache key: {cache_key}')

	reader = SegySectionReader(str(dest_path), key1_byte, key2_byte)
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	cached_readers[cache_key] = reader

	threading.Thread(target=reader.preload_all_sections, daemon=True).start()
	return {'file_id': file_id}


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
		return Response(
			gzip.compress(payload),
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
		raise HTTPException(status_code=404, detail='Section not processed')
	return Response(
		payload,
		media_type='application/octet-stream',
		headers={'Content-Encoding': 'gzip'},
	)


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
