# endpoint.py
import asyncio
import gzip
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
        raise HTTPException(status_code=500, detail=str(e))


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
