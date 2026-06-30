"""Application service for direct compare raw SEG-Y imports."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.state import AppState
from app.services.segy_header_qc_summary import selected_header_qc_summary
from app.services.segy_ingest_service import ingest_saved_segy
from app.services.segy_upload_storage import (
    SavedUpload,
    cleanup_staged_upload,
    cleanup_staged_uploads,
    promote_staged_segy_to_raw,
    save_upload_file,
    staged_upload_dir,
)
from app.trace_store.naming import (
    bounded_direct_import_raw_name,
    content_addressed_compare_store_name,
    safe_upload_name,
)
from app.utils.header_qc import inspect_segy_header_qc


async def import_compare_raw_source(
    *,
    state: AppState,
    upload_dir: Path,
    trace_dir: Path,
    file: UploadFile,
    key1_byte: int,
    key2_byte: int,
) -> dict:
    cleanup_staged_uploads(state, upload_dir=upload_dir)
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
    raw_path = staged_upload_dir(upload_dir) / staged_id / raw_safe_name
    saved: SavedUpload | None = None
    try:
        saved = await save_upload_file(file, raw_safe_name, raw_path=raw_path)
        try:
            header_qc = await asyncio.to_thread(inspect_segy_header_qc, saved.raw_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f'Unable to inspect SEG-Y headers: {exc}',
            ) from exc
        header_qc_summary = selected_header_qc_summary(
            header_qc,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
        if header_qc_summary is None:
            raise HTTPException(
                status_code=422,
                detail='Unable to summarize selected SEG-Y headers',
            )

        durable_raw_path = promote_staged_segy_to_raw(
            staged_path=saved.raw_path,
            safe_name=raw_safe_name,
            source_sha256=saved.source_sha256,
            upload_dir=upload_dir,
        )
        store_name = content_addressed_compare_store_name(
            safe_name=safe_name,
            source_sha256=saved.source_sha256,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
        result = await ingest_saved_segy(
            state=state,
            trace_dir=trace_dir,
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
        cleanup_staged_upload(
            saved.raw_path if saved is not None else raw_path,
            upload_dir=upload_dir,
        )
