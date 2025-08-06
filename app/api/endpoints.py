# endpoint.py
import asyncio
import gzip
import pathlib
import threading
from uuid import uuid4

import msgpack
import numpy as np
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
from utils.picks import PickStore
from utils.utils import SegySectionReader, quantize_float32

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

cached_readers: dict[str, SegySectionReader] = {}
SEGYS: dict[str, str] = {}
pick_store = PickStore()


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


@router.post('/picks')
async def add_picks(
        file_id: str = Query(...),
        picks: dict[int, float] = Body(...),
):
        updated = await asyncio.to_thread(pick_store.add_or_update, file_id, picks)
        return {'file_id': file_id, 'picks': updated}


@router.get('/picks')
async def get_picks(file_id: str = Query(...)):
        picks = await asyncio.to_thread(pick_store.get, file_id)
        return {'file_id': file_id, 'picks': picks}


@router.delete('/picks')
async def delete_picks(
        file_id: str = Query(...),
        trace_idx: int | None = Query(None),
):
        await asyncio.to_thread(pick_store.delete, file_id, trace_idx)
        return {'status': 'ok'}
