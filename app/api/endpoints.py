# endpoint.py
import math
import pathlib
import threading
from uuid import uuid4

import numpy as np
from fastapi import (
        APIRouter,
        File,
        Form,  # 忘れずにインポート
        HTTPException,
        Query,
        UploadFile,
)
from fastapi.responses import JSONResponse
from utils.utils import SegySectionReader

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

cached_readers: dict[str, SegySectionReader] = {}
SEGYS: dict[str, str] = {}
cached_sections: dict[str, dict[int, list[list[float]]]] = {}


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

    def _preload() -> None:
        reader.preload_all_sections()
        cached_sections[cache_key] = dict(reader.section_cache)

    threading.Thread(target=_preload, daemon=True).start()
    return {'file_id': file_id}


@router.get('/get_section')
def get_section(
        file_id: str = Query(...),
        key1_byte: int = Query(189),  # デフォルト設定
        key2_byte: int = Query(193),
        key1_idx: int = Query(...),
        start_trace: int = Query(0),
        end_trace: int = Query(-1),
        target_px: int = Query(1000),
):
        try:
                cache_key = f"{file_id}_{key1_byte}_{key2_byte}"

                if cache_key not in cached_sections:
                        section_cache: dict[int, list[list[float]]] = {}
                        cached_sections[cache_key] = section_cache
                else:
                        section_cache = cached_sections[cache_key]

                if key1_idx not in section_cache:
                        if cache_key not in cached_readers:
                                if file_id not in SEGYS:
                                        raise HTTPException(status_code=404, detail='File ID not found')
                                path = SEGYS[file_id]
                                cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)

                        reader = cached_readers[cache_key]
                        section_cache[key1_idx] = reader.get_section(key1_idx)

                section_full = np.array(section_cache[key1_idx], dtype='float32')
                total_traces = section_full.shape[0]
                if end_trace < 0 or end_trace >= total_traces:
                        end_trace = total_traces - 1
                start_trace = max(0, start_trace)

                visible_traces = end_trace - start_trace + 1
                factor = int(math.ceil(visible_traces / (target_px * 10))) or 1
                section_ds = section_full[start_trace : end_trace + 1 : factor, ::factor]
                return JSONResponse(
                        content={
                                'section': section_ds.tolist(),
                                'factor': factor,
                        }
                )

        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


@router.post('/preload_sections')
def preload_sections(
        file_id: str = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
):
        cache_key = f"{file_id}_{key1_byte}_{key2_byte}"
        if cache_key not in cached_readers:
                if file_id not in SEGYS:
                        raise HTTPException(status_code=404, detail='File ID not found')
                path = SEGYS[file_id]
                cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)

        reader = cached_readers[cache_key]
        reader.preload_all_sections()
        cached_sections[cache_key] = dict(reader.section_cache)
        return {'status': 'preloaded'}
