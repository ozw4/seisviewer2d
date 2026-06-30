"""Existing TraceStore open application service."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.core.state import AppState
from app.services.trace_store_registration import register_trace_store
from app.trace_store.catalog import (
    ensure_trace_store_meta_key_bytes,
    load_trace_store_meta,
    trace_store_complete,
)
from app.trace_store.naming import safe_store_name, safe_upload_name
from app.utils.ingest import SegyIngestor

logger = logging.getLogger(__name__)


async def open_existing_trace_store(
    *,
    state: AppState,
    trace_dir: Path,
    original_name: str | None,
    store_name: str | None,
    store_name_was_provided: bool,
    key1_byte: int,
    key2_byte: int,
) -> dict:
    if store_name_was_provided:
        if not isinstance(store_name, str):
            raise HTTPException(status_code=400, detail='Trace store name is unsafe')
        safe_store = safe_store_name(store_name)
        store_dir = trace_dir / safe_store
        requested_name = store_name
    elif original_name is not None:
        safe_name = safe_upload_name(original_name)
        store_dir = trace_dir / safe_name
        requested_name = original_name
    else:
        raise HTTPException(
            status_code=400,
            detail='store_name or original_name is required',
        )

    meta_path = store_dir / 'meta.json'
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f'Trace store not found for {requested_name}',
        )

    logger.info('Opening existing trace store for %s', store_dir.name)
    file_id = str(uuid4())
    reused = trace_store_complete(store_dir, key1_byte, key2_byte)
    meta = load_trace_store_meta(meta_path)
    if meta is None:
        raise HTTPException(
            status_code=500,
            detail='Trace store metadata is unreadable',
        )
    if store_name_was_provided:
        ensure_trace_store_meta_key_bytes(
            meta,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )

    if not reused:
        segy_path = meta.get('original_segy_path')
        meta_source_sha256 = meta.get('source_sha256')
        source_sha256 = (
            meta_source_sha256 if isinstance(meta_source_sha256, str) else None
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

    response_original_name = meta.get('original_name')
    if not isinstance(response_original_name, str) or not response_original_name:
        response_original_name = original_name or store_dir.name
    source_sha256 = meta.get('source_sha256')

    register_trace_store(
        state=state,
        file_id=file_id,
        store_dir=store_dir,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        dt=meta.get('dt'),
        update_registry=True,
        touch_meta=True,
    )
    response = {
        'file_id': file_id,
        'reused_trace_store': reused,
        'store_name': store_dir.name,
        'original_name': response_original_name,
        'key1_byte': key1_byte,
        'key2_byte': key2_byte,
    }
    if isinstance(source_sha256, str):
        response['source_sha256'] = source_sha256
    return response
