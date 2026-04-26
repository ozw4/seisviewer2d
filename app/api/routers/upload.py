"""Upload and registration endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import threading
from dataclasses import dataclass
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
from app.utils.baseline_artifacts import has_split_baseline_artifacts
from app.utils.header_qc import inspect_segy_header_qc
from app.trace_store.reader import TraceStoreSectionReader

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = get_upload_dir()
PROCESSED_DIR = get_processed_upload_dir()
TRACE_DIR = get_trace_store_dir()
UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class SavedUpload:
    original_name: str
    safe_name: str
    raw_path: Path
    source_sha256: str
    source_size: int


def _safe_upload_name(filename: str) -> str:
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    if safe_name in {'', '.', '..'}:
        raise HTTPException(
            status_code=400,
            detail='Uploaded file must have a safe filename',
        )
    return safe_name


def _staged_upload_dir() -> Path:
    return UPLOAD_DIR / 'staged'


def _cleanup_staged_upload(raw_path: Path) -> None:
    try:
        raw_path.unlink(missing_ok=True)
    except OSError:
        logger.warning('Unable to delete staged SEG-Y file: %s', raw_path)

    staged_dir = raw_path.parent
    if staged_dir.parent != _staged_upload_dir():
        return
    shutil.rmtree(staged_dir, ignore_errors=True)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(UPLOAD_CHUNK_SIZE), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _promote_staged_segy_to_raw(
    *,
    staged_path: Path,
    safe_name: str,
    source_sha256: str,
) -> Path:
    raw_dir = UPLOAD_DIR / 'raw' / source_sha256
    raw_path = raw_dir / safe_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        if raw_path.is_file() and _sha256_file(raw_path) == source_sha256:
            return raw_path
        raw_path = raw_dir / f'{raw_path.stem}.{uuid4().hex}{raw_path.suffix}'

    tmp_path = raw_path.with_suffix(raw_path.suffix + f'.{uuid4().hex}.tmp')
    try:
        shutil.copy2(staged_path, tmp_path)
        tmp_path.replace(raw_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f'Unable to promote staged SEG-Y: {exc}',
        ) from exc
    return raw_path


def _trace_store_complete(store_dir: Path, key1_byte: int, key2_byte: int) -> bool:
    """Return True only when the requested trace store artifacts are present."""
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
    if not (
        isinstance(kb, dict)
        and kb.get('key1') == key1_byte
        and kb.get('key2') == key2_byte
    ):
        return False
    source_sha256 = meta.get('source_sha256')
    if source_sha256 is not None and not isinstance(source_sha256, str):
        return False
    return has_split_baseline_artifacts(
        store_dir,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=source_sha256,
    )


def _load_trace_store_meta(meta_path: Path) -> dict | None:
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    return meta


def _ensure_trace_store_meta_raw_path(
    store_dir: Path,
    meta: dict,
    *,
    raw_path: Path,
    source_sha256: str,
    source_size: int,
) -> dict:
    updated = dict(meta)
    changed = False
    raw_path_text = str(raw_path)

    if updated.get('original_segy_path') != raw_path_text:
        updated['original_segy_path'] = raw_path_text
        changed = True
    if updated.get('original_size') != source_size:
        updated['original_size'] = source_size
        changed = True
    if updated.get('source_sha256') != source_sha256:
        updated['source_sha256'] = source_sha256
        changed = True

    try:
        original_mtime = raw_path.stat().st_mtime
    except OSError:
        original_mtime = None
    if original_mtime is not None and updated.get('original_mtime') != original_mtime:
        updated['original_mtime'] = original_mtime
        changed = True

    if not changed:
        return meta

    meta_path = store_dir / 'meta.json'
    tmp_path = meta_path.with_suffix(meta_path.suffix + f'.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(json.dumps(updated), encoding='utf-8')
        tmp_path.replace(meta_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=f'Unable to update trace store metadata: {exc}',
        ) from exc
    return updated


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


def _touch_trace_store_meta(store_dir: Path) -> None:
    meta_path = store_dir / 'meta.json'
    if not meta_path.exists():
        return
    meta_path.touch()


def _trace_store_summary(store_dir: Path) -> dict | None:
    meta_path = store_dir / 'meta.json'
    meta = _load_trace_store_meta(meta_path)
    if meta is None:
        logger.warning('Skipping trace store with unreadable metadata: %s', store_dir)
        return None

    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        logger.warning(
            'Skipping trace store with invalid key byte metadata: %s', store_dir
        )
        return None

    key1_byte = key_bytes.get('key1')
    key2_byte = key_bytes.get('key2')
    if not isinstance(key1_byte, int) or not isinstance(key2_byte, int):
        logger.warning('Skipping trace store with missing key bytes: %s', store_dir)
        return None

    if not _trace_store_complete(store_dir, key1_byte, key2_byte):
        logger.warning(
            'Skipping incomplete trace store in recent datasets: %s', store_dir
        )
        return None

    try:
        last_used_ts = meta_path.stat().st_mtime
    except OSError:
        logger.warning(
            'Skipping trace store with unreadable metadata mtime: %s', store_dir
        )
        return None

    original_name = meta.get('original_name')
    if not isinstance(original_name, str) or not original_name:
        original_name = store_dir.name

    summary = {
        'original_name': original_name,
        'store_name': store_dir.name,
        'key1_byte': key1_byte,
        'key2_byte': key2_byte,
        'last_used_ts': last_used_ts,
    }
    for key in ('dt', 'n_traces', 'n_samples'):
        value = meta.get(key)
        if isinstance(value, (int, float)):
            summary[key] = value
    return summary


def _list_recent_dataset_summaries() -> list[dict]:
    summaries: list[dict] = []
    if not TRACE_DIR.exists():
        return summaries

    for child in TRACE_DIR.iterdir():
        if not child.is_dir():
            continue
        summary = _trace_store_summary(child)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=lambda item: item['last_used_ts'], reverse=True)
    return summaries


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


async def _save_upload_file(
    file: UploadFile,
    safe_name: str,
    *,
    raw_path: Path,
) -> SavedUpload:
    original_name = file.filename or safe_name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = raw_path.with_suffix(raw_path.suffix + f'.{uuid4().hex}.tmp')
    hasher = hashlib.sha256()
    try:
        with tmp_path.open('wb') as temp_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                temp_file.write(chunk)
        tmp_path.replace(raw_path)
        source_size = raw_path.stat().st_size
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SavedUpload(
        original_name=original_name,
        safe_name=safe_name,
        raw_path=raw_path,
        source_sha256=hasher.hexdigest(),
        source_size=int(source_size),
    )


async def _ingest_saved_segy(
    *,
    request: Request,
    original_name: str,
    safe_name: str,
    raw_path: Path,
    source_sha256: str,
    source_size: int,
    key1_byte: int,
    key2_byte: int,
) -> dict:
    state = get_state(request.app)
    store_dir = TRACE_DIR / safe_name
    file_id = str(uuid4())

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
        meta = _ensure_trace_store_meta_raw_path(
            store_dir,
            meta,
            raw_path=raw_path,
            source_sha256=source_sha256,
            source_size=source_size,
        )
        logger.info('Reusing trace store for %s', original_name)
        _register_trace_store(file_id, store_dir, key1_byte, key2_byte, state=state)
        _touch_trace_store_meta(store_dir)
        state.file_registry.update(
            file_id,
            store_path=str(store_dir),
            dt=meta.get('dt'),
        )
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
    _touch_trace_store_meta(store_dir)
    if isinstance(meta, dict):
        state.file_registry.update(
            file_id,
            store_path=str(store_dir),
            dt=meta.get('dt'),
        )
    else:
        state.file_registry.update(file_id, store_path=str(store_dir))
    return {'file_id': file_id, 'reused_trace_store': False}


def _selected_header_qc_summary(
    header_qc: object,
    *,
    key1_byte: int,
    key2_byte: int,
) -> dict | None:
    if not isinstance(header_qc, dict):
        return None

    pairs = header_qc.get('recommended_pairs')
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            if (
                pair.get('key1_byte') == key1_byte
                and pair.get('key2_byte') == key2_byte
            ):
                warnings = pair.get('warnings')
                return {
                    'selected_pair_score': pair.get('score'),
                    'confidence': pair.get('confidence', 'unknown'),
                    'warnings': warnings if isinstance(warnings, list) else [],
                }

    warnings = header_qc.get('warnings')
    return {
        'selected_pair_score': None,
        'confidence': 'unknown',
        'warnings': warnings if isinstance(warnings, list) else [],
    }


@router.get('/recent_datasets')
async def recent_datasets():
    return {'datasets': _list_recent_dataset_summaries()}


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
        source_sha256 = (
            meta.get('source_sha256')
            if isinstance(meta, dict) and isinstance(meta.get('source_sha256'), str)
            else None
        )
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
            source_sha256=source_sha256,
        )
    _register_trace_store(file_id, store_dir, key1_byte, key2_byte, state=state)
    _touch_trace_store_meta(store_dir)
    if isinstance(meta, dict):
        state.file_registry.update(
            file_id,
            store_path=str(store_dir),
            dt=meta.get('dt'),
        )
    return {'file_id': file_id, 'reused_trace_store': reused}


@router.post('/upload_segy')
async def upload_segy(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    key1_byte: Annotated[int, Form()] = 189,
    key2_byte: Annotated[int, Form()] = 193,
):
    if not file.filename:
        raise HTTPException(
            status_code=400, detail='Uploaded file must have a filename'
        )
    logger.info('Uploading file: %s', file.filename)
    safe_name = _safe_upload_name(file.filename)
    saved = await _save_upload_file(file, safe_name, raw_path=UPLOAD_DIR / safe_name)
    return await _ingest_saved_segy(
        request=request,
        original_name=saved.original_name,
        safe_name=saved.safe_name,
        raw_path=saved.raw_path,
        source_sha256=saved.source_sha256,
        source_size=saved.source_size,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )


@router.post('/stage_segy')
async def stage_segy(
    request: Request,
    file: Annotated[UploadFile, File(...)],
):
    state = get_state(request.app)
    if not file.filename:
        raise HTTPException(
            status_code=400, detail='Uploaded file must have a filename'
        )

    staged_id = uuid4().hex
    safe_name = _safe_upload_name(file.filename)
    raw_path = _staged_upload_dir() / staged_id / safe_name
    saved = await _save_upload_file(file, safe_name, raw_path=raw_path)
    try:
        header_qc = await asyncio.to_thread(inspect_segy_header_qc, saved.raw_path)
    except Exception as exc:  # noqa: BLE001
        _cleanup_staged_upload(saved.raw_path)
        raise HTTPException(
            status_code=422,
            detail=f'Unable to inspect SEG-Y headers: {exc}',
        ) from exc

    record = {
        'original_name': saved.original_name,
        'safe_name': saved.safe_name,
        'raw_path': str(saved.raw_path),
        'source_sha256': saved.source_sha256,
        'source_size': saved.source_size,
        'header_qc': header_qc,
    }
    with state.lock:
        state.staged_uploads.set(staged_id, record)

    return {
        'staged_id': staged_id,
        'file': {
            'original_name': saved.original_name,
            'safe_name': saved.safe_name,
            'size': saved.source_size,
            'sha256': saved.source_sha256,
        },
        **header_qc,
    }


@router.post('/ingest_staged_segy')
async def ingest_staged_segy(
    request: Request,
    staged_id: Annotated[str, Form(...)],
    key1_byte: Annotated[int, Form()] = 189,
    key2_byte: Annotated[int, Form()] = 193,
):
    state = get_state(request.app)
    with state.lock:
        staged = state.staged_uploads.get(staged_id)
    if staged is None:
        raise HTTPException(status_code=404, detail='Staged SEG-Y not found')
    if key1_byte == key2_byte:
        raise HTTPException(
            status_code=400,
            detail='key1_byte and key2_byte must be different',
        )
    if not isinstance(staged, dict):
        raise HTTPException(status_code=404, detail='Staged SEG-Y not found')

    raw_path = Path(str(staged.get('raw_path', '')))
    if not raw_path.is_file():
        raise HTTPException(status_code=410, detail='Staged SEG-Y file is missing')

    safe_name = str(staged.get('safe_name'))
    source_sha256 = str(staged.get('source_sha256'))
    source_size = int(staged.get('source_size'))
    durable_raw_path = _promote_staged_segy_to_raw(
        staged_path=raw_path,
        safe_name=safe_name,
        source_sha256=source_sha256,
    )
    result = await _ingest_saved_segy(
        request=request,
        original_name=str(staged.get('original_name') or staged.get('safe_name')),
        safe_name=safe_name,
        raw_path=durable_raw_path,
        source_sha256=source_sha256,
        source_size=source_size,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    with state.lock:
        state.staged_uploads.pop(staged_id, None)
    _cleanup_staged_upload(raw_path)
    summary = _selected_header_qc_summary(
        staged.get('header_qc'),
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    if summary is not None:
        result['header_qc'] = summary
    return result


@router.get('/file_info')
async def file_info(
    request: Request,
    file_id: Annotated[str, Query()],
) -> dict[str, str]:
    """Return basename for a given ``file_id``."""
    state = get_state(request.app)
    name = state.file_registry.filename(file_id)
    if name is None:
        raise HTTPException(status_code=404, detail='Unknown file_id')
    return {'file_name': name}
