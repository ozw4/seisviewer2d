"""Upload and registration endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.api._helpers import get_state
from app.core.paths import (
    get_processed_upload_dir,
    get_trace_store_dir,
    get_upload_dir,
)
from app.services.compare_raw_import_service import import_compare_raw_source
from app.services.segy_open_service import open_existing_trace_store
from app.services.segy_upload_service import upload_segy_file
from app.services.staged_segy_upload_service import (
    ingest_staged_segy_upload,
    stage_segy_upload,
)
from app.trace_store.catalog import list_recent_dataset_summaries

router = APIRouter()

UPLOAD_DIR = get_upload_dir()
PROCESSED_DIR = get_processed_upload_dir()
TRACE_DIR = get_trace_store_dir()


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
    state = get_state(request.app)
    return await upload_segy_file(
        state=state,
        upload_dir=UPLOAD_DIR,
        trace_dir=TRACE_DIR,
        file=file,
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
