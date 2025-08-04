# endpoint.py
import json
import pathlib
import threading
from uuid import uuid4

import redis
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

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)


@router.get('/get_key1_values')
def get_key1_values(
        file_id: str = Query(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
):
        try:
                raw = redis_client.get(f'{file_id}:key1_values')
                if raw is None:
                        raise HTTPException(status_code=404, detail='File ID not found')
                values = json.loads(raw)
                return JSONResponse(content={'values': values})
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


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

        reader = SegySectionReader(str(dest_path), key1_byte, key2_byte)

        def preload():
                key1_vals = reader.get_key1_values()
                redis_client.set(
                        f'{file_id}:key1_values', json.dumps(key1_vals.tolist())
                )
                for key1_val in key1_vals:
                        section = reader.get_section(key1_val)
                        redis_client.set(
                                f'{file_id}:{key1_val}', json.dumps(section)
                        )

        threading.Thread(target=preload, daemon=True).start()
        return {'file_id': file_id}


@router.get('/get_section')
def get_section(
        file_id: str = Query(...),
        key1_idx: int = Query(...),
        key1_byte: int = Query(189),  # 未使用
        key2_byte: int = Query(193),  # 未使用
):
        try:
                raw = redis_client.get(f'{file_id}:{key1_idx}')
                if raw is None:
                        raise HTTPException(status_code=404, detail='Section not found')
                section = json.loads(raw)
                return JSONResponse(content={'section': section})
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
