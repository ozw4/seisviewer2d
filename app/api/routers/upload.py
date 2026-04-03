"""Upload and registration endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
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
TRACE_STORE_REQUIRED_FILES = ('meta.json', 'traces.npy', 'index.npz')
ARCHIVED_TRACE_STORE_MARKER = '.old-'


def _trace_store_required_files_present(store_dir: Path) -> bool:
    return all((store_dir / name).exists() for name in TRACE_STORE_REQUIRED_FILES)


def _trace_store_complete(store_dir: Path, key1_byte: int, key2_byte: int) -> bool:
    """Return True only when a trace store exists AND its metadata indicates
    it was built for the requested key bytes. Header files are not part of
    the completion predicate (they are generated on demand).
    """
    if not store_dir.is_dir():
        return False
    meta_path = store_dir / 'meta.json'
    if not _trace_store_required_files_present(store_dir):
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


def _is_archived_trace_store_name(name: str) -> bool:
    return ARCHIVED_TRACE_STORE_MARKER in name


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_dt(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    return None


def _trace_store_display_name(store_dir: Path, meta: dict) -> str:
    original_path = meta.get('original_segy_path')
    if isinstance(original_path, str) and original_path:
        normalized = original_path.replace('\\', '/')
        basename = Path(normalized).name
        if basename:
            return basename
    return store_dir.name


def _trace_store_updated_timestamp(store_dir: Path) -> float | None:
    try:
        timestamps = [store_dir.stat().st_mtime]
        for name in TRACE_STORE_REQUIRED_FILES:
            timestamps.append((store_dir / name).stat().st_mtime)
    except OSError:
        return None
    return max(timestamps) if timestamps else None


def _build_recent_dataset_entry(
    store_dir: Path,
) -> tuple[float, dict[str, object]] | None:
    if not store_dir.is_dir() or _is_archived_trace_store_name(store_dir.name):
        return None
    if not _trace_store_required_files_present(store_dir):
        return None

    meta = _load_trace_store_meta(store_dir / 'meta.json')
    if meta is None:
        return None

    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        return None

    key1_byte = _coerce_int(key_bytes.get('key1'))
    key2_byte = _coerce_int(key_bytes.get('key2'))
    n_traces = _coerce_int(meta.get('n_traces'))
    n_samples = _coerce_int(meta.get('n_samples'))
    if key1_byte is None or key2_byte is None:
        return None
    if n_traces is None or n_traces <= 0:
        return None
    if n_samples is None or n_samples <= 0:
        return None

    updated_ts = _trace_store_updated_timestamp(store_dir)
    if updated_ts is None:
        return None

    original_size = _coerce_int(meta.get('original_size'))
    if original_size is not None and original_size < 0:
        original_size = None

    updated_at = datetime.fromtimestamp(updated_ts, tz=timezone.utc).isoformat()
    entry = {
        'name': _trace_store_display_name(store_dir, meta),
        'key1_byte': key1_byte,
        'key2_byte': key2_byte,
        'dt': _coerce_dt(meta.get('dt')),
        'n_traces': n_traces,
        'n_samples': n_samples,
        'original_size': original_size,
        'updated_at': updated_at,
    }
    return updated_ts, entry


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
    with state.lock:
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


@router.get('/recent_datasets')
async def recent_datasets() -> dict[str, list[dict[str, object]]]:
    if not TRACE_DIR.exists() or not TRACE_DIR.is_dir():
        return {'datasets': []}

    try:
        store_dirs = list(TRACE_DIR.iterdir())
    except OSError as exc:
        logger.warning('Failed to list recent trace stores in %s: %s', TRACE_DIR, exc)
        return {'datasets': []}

    datasets: list[tuple[float, dict[str, object]]] = []
    for store_dir in store_dirs:
        item = _build_recent_dataset_entry(store_dir)
        if item is not None:
            datasets.append(item)

    datasets.sort(key=lambda item: item[0], reverse=True)
    return {'datasets': [item[1] for item in datasets]}


@router.get('/file_info')
async def file_info(file_id: Annotated[str, Query()]) -> dict[str, str]:
    """Return basename for a given ``file_id``."""
    rec = FILE_REGISTRY.get(file_id) or {}
    path = rec.get('path') or rec.get('store_path')
    if not path:
        raise HTTPException(status_code=404, detail='Unknown file_id')

    name = Path(str(path).replace('\\', '/')).name
    return {'file_name': name}
