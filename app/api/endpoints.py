"""FastAPI endpoints for working with SEG-Y files."""

from __future__ import annotations

import pathlib
import threading
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from utils.utils import SegySectionReader

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

cached_readers: dict[str, SegySectionReader] = {}
SEGYS: dict[str, str] = {}


@router.get("/get_key1_values")
def get_key1_values(
    file_id: Annotated[str, Query(...)],
    key1_byte: Annotated[int, Query(189)],
    key2_byte: Annotated[int, Query(193)],
) -> JSONResponse:
    """Return all unique key1 values for the given file."""
    cache_key = f"{file_id}_{key1_byte}_{key2_byte}"
    if cache_key not in cached_readers:
        if file_id not in SEGYS:
            raise HTTPException(status_code=404, detail="File ID not found")
        path = SEGYS[file_id]
        cached_readers[cache_key] = SegySectionReader(
            path, file_id, key1_byte, key2_byte
        )
    reader = cached_readers[cache_key]
    return JSONResponse(content={"values": reader.get_key1_values().tolist()})


@router.post("/upload_segy")
async def upload_segy(
    file: Annotated[UploadFile, File(...)],
    key1_byte: Annotated[int, Form(189)],
    key2_byte: Annotated[int, Form(193)],
) -> dict[str, str]:
    """Upload a SEG-Y file and begin preloading its sections."""
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Uploaded file must have a filename"
        )
    ext = pathlib.Path(file.filename).suffix.lower()
    file_id = str(uuid4())
    dest_path = UPLOAD_DIR / f"{file_id}{ext}"
    dest_path.write_bytes(await file.read())

    SEGYS[file_id] = str(dest_path)
    cache_key = f"{file_id}_{key1_byte}_{key2_byte}"
    reader = SegySectionReader(str(dest_path), file_id, key1_byte, key2_byte)
    cached_readers[cache_key] = reader
    threading.Thread(target=reader.preload_all_sections, daemon=True).start()
    return {"file_id": file_id}


@router.get("/get_section")
def get_section(
    file_id: Annotated[str, Query(...)],
    key1_byte: Annotated[int, Query(189)],
    key2_byte: Annotated[int, Query(193)],
    key1_idx: Annotated[int, Query(...)],
) -> JSONResponse:
    """Return a cached or freshly read section for the given index."""
    cache_key = f"{file_id}_{key1_byte}_{key2_byte}"
    if cache_key not in cached_readers:
        if file_id not in SEGYS:
            raise HTTPException(status_code=404, detail="File ID not found")
        path = SEGYS[file_id]
        cached_readers[cache_key] = SegySectionReader(
            path, file_id, key1_byte, key2_byte
        )
    reader = cached_readers[cache_key]
    section = reader.get_section(key1_idx)
    return JSONResponse(content={"section": section})
