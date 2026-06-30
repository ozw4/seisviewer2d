"""Upload and registration endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.api._helpers import get_state
from app.core.paths import (
    get_processed_upload_dir,
    get_trace_store_dir,
    get_upload_dir,
)
from app.core.state import AppState
from app.services.compare_raw_import_service import import_compare_raw_source
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
from app.services.staged_segy_upload_service import (
    ingest_staged_segy_upload,
    stage_segy_upload,
)
from app.trace_store.catalog import list_recent_dataset_summaries
from app.trace_store.naming import safe_upload_name

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
    return await import_compare_raw_source(
        state=state,
        upload_dir=UPLOAD_DIR,
        trace_dir=TRACE_DIR,
        file=file,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )


@router.post('/stage_segy')
async def stage_segy(
    request: Request,
    file: Annotated[UploadFile, File(...)],
):
    state = get_state(request.app)
    return await stage_segy_upload(state=state, upload_dir=UPLOAD_DIR, file=file)


@router.post('/ingest_staged_segy')
async def ingest_staged_segy(
    request: Request,
    staged_id: Annotated[str, Form(...)],
    key1_byte: Annotated[int, Form()] = 189,
    key2_byte: Annotated[int, Form()] = 193,
):
    state = get_state(request.app)
    return await ingest_staged_segy_upload(
        state=state,
        upload_dir=UPLOAD_DIR,
        trace_dir=TRACE_DIR,
        staged_id=staged_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )


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
