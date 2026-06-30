"""Application service for direct SEG-Y uploads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from app.core.state import AppState
from app.services.segy_ingest_service import ingest_saved_segy
from app.services.segy_upload_storage import save_upload_file
from app.trace_store.naming import safe_upload_name

logger = logging.getLogger(__name__)


async def upload_segy_file(
    *,
    state: AppState,
    upload_dir: Path,
    trace_dir: Path,
    file: UploadFile,
    key1_byte: int,
    key2_byte: int,
) -> dict[str, Any]:
    """Save an uploaded SEG-Y file and ingest or reuse its TraceStore."""
    if not file.filename:
        raise HTTPException(
            status_code=400, detail='Uploaded file must have a filename'
        )

    logger.info('Uploading file: %s', file.filename)
    safe_name = safe_upload_name(file.filename)
    saved = await save_upload_file(file, safe_name, raw_path=upload_dir / safe_name)
    return await ingest_saved_segy(
        state=state,
        trace_dir=trace_dir,
        original_name=saved.original_name,
        safe_name=saved.safe_name,
        raw_path=saved.raw_path,
        source_sha256=saved.source_sha256,
        source_size=saved.source_size,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )


__all__ = ['upload_segy_file']
