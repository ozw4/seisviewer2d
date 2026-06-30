"""SEG-Y TraceStore ingest application service."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.core.state import AppState
from app.services.trace_store_registration import register_trace_store
from app.trace_store.catalog import (
    archive_trace_store,
    ensure_trace_store_meta_identity,
    trace_store_matches_source,
)
from app.trace_store.naming import safe_store_name
from app.utils.ingest import SegyIngestor

logger = logging.getLogger(__name__)


async def ingest_saved_segy(
    *,
    state: AppState,
    trace_dir: Path,
    original_name: str,
    safe_name: str,
    raw_path: Path,
    source_sha256: str,
    source_size: int,
    key1_byte: int,
    key2_byte: int,
    store_name: str | None = None,
    allow_archive_existing: bool = True,
) -> dict:
    resolved_store_name = safe_name if store_name is None else store_name
    store_dir = trace_dir / safe_store_name(resolved_store_name)
    file_id = str(uuid4())

    meta: dict | None = None
    reused = False
    if store_dir.exists():
        meta = trace_store_matches_source(
            store_dir,
            key1_byte,
            key2_byte,
            source_sha256=source_sha256,
            source_size=source_size,
        )
        if meta is not None:
            reused = True
        else:
            if not allow_archive_existing:
                raise HTTPException(
                    status_code=409,
                    detail='Trace store already exists for a different source or key bytes',
                )
            try:
                archive_trace_store(store_dir)
            except OSError as exc:
                msg = f'Unable to archive existing trace store: {exc}'
                raise HTTPException(status_code=409, detail=msg) from exc

    if reused and meta is not None:
        meta = ensure_trace_store_meta_identity(
            store_dir,
            meta,
            raw_path=raw_path,
            original_name=original_name,
            display_name=original_name,
            store_name=store_dir.name,
            source_sha256=source_sha256,
            source_size=source_size,
        )
        logger.info('Reusing trace store for %s', original_name)
        register_trace_store(
            state=state,
            file_id=file_id,
            store_dir=store_dir,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            dt=meta.get('dt') if isinstance(meta, dict) else None,
            update_registry=True,
            touch_meta=True,
        )
        return {
            'file_id': file_id,
            'reused_trace_store': True,
            'store_name': store_dir.name,
        }

    store_dir.mkdir(parents=True, exist_ok=True)
    meta = await asyncio.to_thread(
        SegyIngestor.from_segy,
        str(raw_path),
        store_dir,
        key1_byte,
        key2_byte,
        source_sha256=source_sha256,
    )
    meta = ensure_trace_store_meta_identity(
        store_dir,
        meta,
        raw_path=raw_path,
        original_name=original_name,
        display_name=original_name,
        store_name=store_dir.name,
        source_sha256=source_sha256,
        source_size=source_size,
    )
    register_trace_store(
        state=state,
        file_id=file_id,
        store_dir=store_dir,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        dt=meta.get('dt') if isinstance(meta, dict) else None,
        update_registry=True,
        touch_meta=True,
    )
    return {
        'file_id': file_id,
        'reused_trace_store': False,
        'store_name': store_dir.name,
    }
