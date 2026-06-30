"""Upload and registration endpoints."""

from __future__ import annotations

import asyncio
import logging
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
from app.services.segy_upload_storage import (
    SavedUpload,
    cleanup_discarded_staged_upload as storage_cleanup_discarded_staged_upload,
    cleanup_staged_upload as storage_cleanup_staged_upload,
    cleanup_staged_uploads as storage_cleanup_staged_uploads,
    ensure_staged_upload_cleanup_callback as storage_ensure_staged_upload_cleanup_callback,
    promote_staged_segy_to_raw,
    save_upload_file,
    sha256_file,
    staged_upload_dir,
)
from app.services.segy_ingest_service import ingest_saved_segy
from app.services.segy_open_service import open_existing_trace_store
from app.trace_store.catalog import list_recent_dataset_summaries
from app.trace_store.naming import (
    bounded_direct_import_raw_name,
    content_addressed_compare_store_name,
    safe_upload_name,
)
from app.utils.header_qc import inspect_segy_header_qc

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = get_upload_dir()
PROCESSED_DIR = get_processed_upload_dir()
TRACE_DIR = get_trace_store_dir()


def _staged_upload_dir() -> Path:
    return staged_upload_dir(UPLOAD_DIR)


def _cleanup_staged_upload(raw_path: Path) -> None:
    storage_cleanup_staged_upload(raw_path, upload_dir=UPLOAD_DIR)


def _cleanup_discarded_staged_upload(_key, value, _reason: str) -> None:
    storage_cleanup_discarded_staged_upload(
        _key,
        value,
        _reason,
        upload_dir=UPLOAD_DIR,
    )


def _ensure_staged_upload_cleanup_callback(state: AppState) -> None:
    storage_ensure_staged_upload_cleanup_callback(state, upload_dir=UPLOAD_DIR)


def cleanup_staged_uploads(
    state: AppState,
    *,
    force: bool = False,
    now_ts: float | None = None,
) -> int:
    return storage_cleanup_staged_uploads(
        state,
        upload_dir=UPLOAD_DIR,
        force=force,
        now_ts=now_ts,
    )


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _promote_staged_segy_to_raw(
    *,
    staged_path: Path,
    safe_name: str,
    source_sha256: str,
) -> Path:
    return promote_staged_segy_to_raw(
        staged_path=staged_path,
        safe_name=safe_name,
        source_sha256=source_sha256,
        upload_dir=UPLOAD_DIR,
    )


async def _save_upload_file(
    file: UploadFile,
    safe_name: str,
    *,
    raw_path: Path,
) -> SavedUpload:
    return await save_upload_file(file, safe_name, raw_path=raw_path)


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
    return {'datasets': list_recent_dataset_summaries(TRACE_DIR)}


@router.post('/open_segy')
async def open_segy(
    request: Request,
    original_name: Annotated[str | None, Form()] = None,
    store_name: Annotated[str | None, Form()] = None,
    key1_byte: Annotated[int, Form()] = 189,
    key2_byte: Annotated[int, Form()] = 193,
):
    state = get_state(request.app)
    form = await request.form()
    store_name_was_provided = 'store_name' in form
    return await open_existing_trace_store(
        state=state,
        trace_dir=TRACE_DIR,
        original_name=original_name,
        store_name=store_name,
        store_name_was_provided=store_name_was_provided,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )


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
    safe_name = safe_upload_name(file.filename)
    saved = await _save_upload_file(file, safe_name, raw_path=UPLOAD_DIR / safe_name)
    state = get_state(request.app)
    return await ingest_saved_segy(
        state=state,
        trace_dir=TRACE_DIR,
        original_name=saved.original_name,
        safe_name=saved.safe_name,
        raw_path=saved.raw_path,
        source_sha256=saved.source_sha256,
        source_size=saved.source_size,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )


@router.post('/compare/raw/import')
async def import_compare_raw(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    key1_byte: Annotated[int, Form(...)],
    key2_byte: Annotated[int, Form(...)],
):
    state = get_state(request.app)
    cleanup_staged_uploads(state)
    if not file.filename:
        raise HTTPException(
            status_code=400, detail='Uploaded file must have a filename'
        )
    if key1_byte == key2_byte:
        raise HTTPException(
            status_code=400,
            detail='key1_byte and key2_byte must be different',
        )

    staged_id = uuid4().hex
    safe_name = safe_upload_name(file.filename)
    raw_safe_name = bounded_direct_import_raw_name(safe_name)
    raw_path = _staged_upload_dir() / staged_id / raw_safe_name
    saved: SavedUpload | None = None
    try:
        saved = await _save_upload_file(file, raw_safe_name, raw_path=raw_path)
        try:
            header_qc = await asyncio.to_thread(inspect_segy_header_qc, saved.raw_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f'Unable to inspect SEG-Y headers: {exc}',
            ) from exc
        header_qc_summary = _selected_header_qc_summary(
            header_qc,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
        if header_qc_summary is None:
            raise HTTPException(
                status_code=422,
                detail='Unable to summarize selected SEG-Y headers',
            )

        durable_raw_path = _promote_staged_segy_to_raw(
            staged_path=saved.raw_path,
            safe_name=raw_safe_name,
            source_sha256=saved.source_sha256,
        )
        store_name = content_addressed_compare_store_name(
            safe_name=safe_name,
            source_sha256=saved.source_sha256,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
        result = await ingest_saved_segy(
            state=state,
            trace_dir=TRACE_DIR,
            original_name=saved.original_name,
            safe_name=safe_name,
            raw_path=durable_raw_path,
            source_sha256=saved.source_sha256,
            source_size=saved.source_size,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            store_name=store_name,
            allow_archive_existing=False,
        )
        return {
            'file_id': result['file_id'],
            'display_name': saved.original_name,
            'original_name': saved.original_name,
            'safe_name': safe_name,
            'store_name': result['store_name'],
            'source_sha256': saved.source_sha256,
            'source_size': saved.source_size,
            'key1_byte': key1_byte,
            'key2_byte': key2_byte,
            'reused_trace_store': result['reused_trace_store'],
            'header_qc': header_qc_summary,
        }
    finally:
        _cleanup_staged_upload(saved.raw_path if saved is not None else raw_path)


@router.post('/stage_segy')
async def stage_segy(
    request: Request,
    file: Annotated[UploadFile, File(...)],
):
    state = get_state(request.app)
    cleanup_staged_uploads(state)
    if not file.filename:
        raise HTTPException(
            status_code=400, detail='Uploaded file must have a filename'
        )

    staged_id = uuid4().hex
    safe_name = safe_upload_name(file.filename)
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
    cleanup_staged_uploads(state)
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
    result = await ingest_saved_segy(
        state=state,
        trace_dir=TRACE_DIR,
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
