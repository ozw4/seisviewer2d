# endpoint.py
import json
import pathlib
import threading
from uuid import uuid4

import redis
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from utils.utils import SegySectionReader

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

cached_readers: dict[str, SegySectionReader] = {}
SEGYS: dict[str, str] = {}


@router.get("/get_key1_values")
def get_key1_values(
    file_id: str = Query(...),
    key1_byte: int = Query(189),
    key2_byte: int = Query(193),
):
    cache_key = f"{file_id}_{key1_byte}_{key2_byte}"
    redis_key = f"key1_values:{cache_key}"
    cached = redis_client.get(redis_key)
    if cached:
        return JSONResponse(content={"values": json.loads(cached)})

    if cache_key not in cached_readers:
        if file_id not in SEGYS:
            raise HTTPException(status_code=404, detail="File ID not found")
        path = SEGYS[file_id]
        cached_readers[cache_key] = SegySectionReader(
            path, file_id, key1_byte, key2_byte, redis_client
        )
    reader = cached_readers[cache_key]
    values = reader.get_key1_values().tolist()
    redis_client.set(redis_key, json.dumps(values))
    return JSONResponse(content={"values": values})


@router.post("/upload_segy")
async def upload_segy(
    file: UploadFile = File(...),
    key1_byte: int = Form(189),
    key2_byte: int = Form(193),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    print(f"Uploading file: {file.filename}")
    ext = pathlib.Path(file.filename).suffix.lower()
    file_id = str(uuid4())
    dest_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    SEGYS[file_id] = str(dest_path)

    cache_key = f"{file_id}_{key1_byte}_{key2_byte}"
    print(f"Creating cache key: {cache_key}")

    reader = SegySectionReader(
        str(dest_path), file_id, key1_byte, key2_byte, redis_client
    )
    cached_readers[cache_key] = reader

    redis_key = f"key1_values:{cache_key}"
    redis_client.set(redis_key, json.dumps(reader.get_key1_values().tolist()))

    threading.Thread(target=reader.preload_all_sections, daemon=True).start()
    return {"file_id": file_id}


@router.get("/get_section")
def get_section(
    file_id: str = Query(...),
    key1_byte: int = Query(189),
    key2_byte: int = Query(193),
    key1_idx: int = Query(...),
):
    try:
        cache_key = f"{file_id}_{key1_byte}_{key2_byte}"
        redis_key = f"section:{file_id}:{key1_byte}:{key2_byte}:{key1_idx}"
        cached = redis_client.get(redis_key)
        if cached:
            return JSONResponse(content={"section": json.loads(cached)})

        if cache_key not in cached_readers:
            if file_id not in SEGYS:
                raise HTTPException(status_code=404, detail="File ID not found")
            path = SEGYS[file_id]
            cached_readers[cache_key] = SegySectionReader(
                path, file_id, key1_byte, key2_byte, redis_client
            )

        reader = cached_readers[cache_key]
        section = reader.get_section(key1_idx)
        return JSONResponse(content={"section": section})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

