# endpoint.py
import gzip
import pathlib
import threading
from uuid import uuid4

import msgpack
import numpy as np
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
from utils.picking import pick_first_arrival
from utils.utils import SegySectionReader, quantize_float32

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

cached_readers: dict[str, SegySectionReader] = {}
SEGYS: dict[str, str] = {}
PICK_STORE: dict[str, dict[int, dict[int, int | None]]] = {}

class AutoPickParams(BaseModel):
	"""Parameters for automatic picking."""

	file_id: str
	key1_idx: int
	trace_idx: int | None = None
	win_sta: int = 20
	win_lta: int = 100
	threshold: float = 3.0
	key1_byte: int = 189
	key2_byte: int = 193

def _store_pick(file_id: str, key1_idx: int, trace_idx: int, pick: int | None) -> None:
	PICK_STORE.setdefault(file_id, {}).setdefault(key1_idx, {})[trace_idx] = pick

def _auto_pick_task(params: AutoPickParams) -> None:
	cache_key = f'{params.file_id}_{params.key1_byte}_{params.key2_byte}'
	if cache_key not in cached_readers:
		if params.file_id not in SEGYS:
			return
		path = SEGYS[params.file_id]
		cached_readers[cache_key] = SegySectionReader(
		path, params.key1_byte, params.key2_byte
	)
	reader = cached_readers[cache_key]
	section = reader.get_section(params.key1_idx)
	if params.trace_idx is not None:
		traces = [(params.trace_idx, section[params.trace_idx])]
	else:
		traces = list(enumerate(section))
	for idx, tr in traces:
		pick = pick_first_arrival(
				np.array(tr, dtype=np.float32),
				params.win_sta,
				params.win_lta,
				params.threshold,
			)
		_store_pick(params.file_id, params.key1_idx, idx, pick)

@router.post('/auto_pick')
def auto_pick(params: AutoPickParams) -> dict[str, str]:
	"""Trigger automatic picking in background."""
	threading.Thread(target=_auto_pick_task, args=(params,), daemon=True).start()
	return {'status': 'started'}



@router.get('/get_key1_values')
def get_key1_values(
	file_id: str = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
):
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	if cache_key not in cached_readers:
		if file_id not in SEGYS:
			raise HTTPException(status_code=404, detail='File ID not found')
		path = SEGYS[file_id]
		cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)
	reader = cached_readers[cache_key]
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
		# 複合キーを作る（file_id, key1_byte, key2_byte）
		cache_key = f'{file_id}_{key1_byte}_{key2_byte}'

		# キャッシュにreaderがなければ初期化
		if cache_key not in cached_readers:
			if file_id not in SEGYS:
				raise HTTPException(status_code=404, detail='File ID not found')
			path = SEGYS[file_id]
			cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)

		reader = cached_readers[cache_key]
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
        cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
        if cache_key not in cached_readers:
            if file_id not in SEGYS:
                raise HTTPException(status_code=404, detail='File ID not found')
            path = SEGYS[file_id]
            cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)
        reader = cached_readers[cache_key]
        section = np.array(reader.get_section(key1_idx), dtype=np.float32)
        scale, q = quantize_float32(section)
        payload = msgpack.packb({
            'scale': scale,
            'shape': q.shape,
            'data': q.tobytes(),
        })
        return Response(
            gzip.compress(payload),
            media_type='application/octet-stream',
            headers={'Content-Encoding': 'gzip'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
