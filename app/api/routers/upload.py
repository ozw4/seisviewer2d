"""Upload and registration endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.api._helpers import get_state
from app.core.paths import (
    get_processed_upload_dir,
    get_trace_store_dir,
    get_upload_dir,
)
from app.core.state import AppState
from app.utils.ingest import SegyIngestor
from app.utils.segy_meta import FILE_REGISTRY
from app.utils.utils import TraceStoreSectionReader

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = get_upload_dir()
PROCESSED_DIR = get_processed_upload_dir()
TRACE_DIR = get_trace_store_dir()
UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024


def _trace_store_complete(store_dir: Path, key1_byte: int, key2_byte: int) -> bool:
    """Return True only when a trace store exists AND its metadata indicates
    it was built for the requested key bytes. Header files are not part of
    the completion predicate (they are generated on demand).
    """
    if not store_dir.is_dir():
        return False
    meta_path = store_dir / 'meta.json'
    if not (
        (store_dir / 'traces.npy').exists()
        and (store_dir / 'index.npz').exists()
        and meta_path.exists()
    ):
        return False
    meta = _load_trace_store_meta(meta_path)
    if not isinstance(meta, dict):
        return False
    kb = meta.get('key_bytes')
    return (
        isinstance(kb, dict)
        and kb.get('key1') == key1_byte
        and kb.get('key2') == key2_byte
    )


def _load_trace_store_meta(meta_path: Path) -> dict | None:
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    return meta


def _trace_store_matches_source(
    store_dir: Path,
    key1_byte: int,
    key2_byte: int,
    *,
    source_sha256: str | None = None,
    source_size: int | None = None,
) -> dict | None:
    if not _trace_store_complete(store_dir, key1_byte, key2_byte):
        return None
    meta_path = store_dir / 'meta.json'
    meta = _load_trace_store_meta(meta_path)
    if meta is None:
        return None
    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        return None
    if key_bytes.get('key1') != key1_byte or key_bytes.get('key2') != key2_byte:
        return None
    # --- Prefer exact identity via content hash when available ---
    meta_hash = meta.get('source_sha256')
    if isinstance(source_sha256, str) and isinstance(meta_hash, str):
        return meta if meta_hash == source_sha256 else None

    # --- Fallback: size match only (for legacy stores without hash) ---
    if isinstance(source_size, int):
        original_size = meta.get('original_size')
        if isinstance(original_size, int) and original_size == source_size:
            return meta
        return None
    return meta


def _archive_trace_store(store_dir: Path) -> None:
    if not store_dir.exists():
        return
    archive_dir = store_dir.parent / f'{store_dir.name}.old-{uuid4().hex}'
    store_dir.rename(archive_dir)


def _register_trace_store(
    file_id: str,
    store_dir: Path,
    key1_byte: int,
    key2_byte: int,
    *,
    state: AppState,
) -> TraceStoreSectionReader:
    reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
    cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
    state.cached_readers[cache_key] = reader
    threading.Thread(target=reader.preload_all_sections, daemon=True).start()
    for b in {key1_byte, key2_byte}:
        threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
    return reader


@router.post('/open_segy')
async def open_segy(
    request: Request,
    original_name: Annotated[str, Form(...)],
    key1_byte: Annotated[int, Form()] = 189,
    key2_byte: Annotated[int, Form()] = 193,
):
    state = get_state(request.app)
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', original_name)
    store_dir = TRACE_DIR / safe_name
    meta_path = store_dir / 'meta.json'
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f'Trace store not found for {original_name}',
        )
    logger.info('Opening existing trace store for %s', original_name)
    file_id = str(uuid4())
    reused = _trace_store_complete(store_dir, key1_byte, key2_byte)
    if reused:
        meta = json.loads(meta_path.read_text())
    else:
        meta = json.loads(meta_path.read_text())
        segy_path = meta.get('original_segy_path') if isinstance(meta, dict) else None
        if not isinstance(segy_path, str):
            raise HTTPException(
                status_code=500,
                detail='Trace store incomplete and SEG-Y path unavailable',
            )
        meta = await asyncio.to_thread(
            SegyIngestor.from_segy,
            segy_path,
            store_dir,
            key1_byte,
            key2_byte,
        )
    _register_trace_store(file_id, store_dir, key1_byte, key2_byte, state=state)
    if isinstance(meta, dict):
        FILE_REGISTRY[file_id] = {
            'store_path': str(store_dir),
            'dt': meta.get('dt'),
        }
    return {'file_id': file_id, 'reused_trace_store': reused}


@router.post('/upload_segy')
async def upload_segy(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    key1_byte: Annotated[int, Form()] = 189,
    key2_byte: Annotated[int, Form()] = 193,
):
    state = get_state(request.app)
    if not file.filename:
        raise HTTPException(
            status_code=400, detail='Uploaded file must have a filename'
        )
    logger.info('Uploading file: %s', file.filename)
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', file.filename)
    store_dir = TRACE_DIR / safe_name
    file_id = str(uuid4())
    raw_path = UPLOAD_DIR / safe_name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = raw_path.with_suffix(raw_path.suffix + f'.{uuid4().hex}.tmp')
    hasher = hashlib.sha256()
    with tmp_path.open('wb') as temp_file:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            temp_file.write(chunk)
    tmp_path.replace(raw_path)
    source_sha256 = hasher.hexdigest()
    try:
        source_size = raw_path.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    meta: dict | None = None
    reused = False
    if store_dir.exists():
        meta = _trace_store_matches_source(
            store_dir,
            key1_byte,
            key2_byte,
            source_sha256=source_sha256,
            source_size=source_size,
        )
        if meta is not None:
            reused = True
        else:
            try:
                _archive_trace_store(store_dir)
            except OSError as exc:
                msg = f'Unable to archive existing trace store: {exc}'
                raise HTTPException(status_code=409, detail=msg) from exc

    if reused and meta is not None:
        logger.info('Reusing trace store for %s', file.filename)
        _register_trace_store(file_id, store_dir, key1_byte, key2_byte, state=state)
        FILE_REGISTRY[file_id] = {
            'store_path': str(store_dir),
            'dt': meta.get('dt'),
        }
        return {'file_id': file_id, 'reused_trace_store': True}

    store_dir.mkdir(parents=True, exist_ok=True)
    meta = await asyncio.to_thread(
        SegyIngestor.from_segy,
        str(raw_path),
        store_dir,
        key1_byte,
        key2_byte,
        source_sha256=source_sha256,
    )
    _register_trace_store(file_id, store_dir, key1_byte, key2_byte, state=state)
    if isinstance(meta, dict):
        FILE_REGISTRY[file_id] = {
            'store_path': str(store_dir),
            'dt': meta.get('dt'),
        }
    else:
        FILE_REGISTRY[file_id] = {'store_path': str(store_dir)}
    return {'file_id': file_id, 'reused_trace_store': False}


@router.get('/file_info')
async def file_info(file_id: Annotated[str, Query()]) -> dict[str, str]:
    """Return basename for a given ``file_id``."""
    rec = FILE_REGISTRY.get(file_id) or {}
    path = rec.get('path') or rec.get('store_path')
    if not path:
        raise HTTPException(status_code=404, detail='Unknown file_id')

    name = Path(str(path).replace('\\', '/')).name
    return {'file_name': name}
