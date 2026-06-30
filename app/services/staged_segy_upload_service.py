"""Application service for staged SEG-Y upload and ingest."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.state import AppState
from app.services.segy_header_qc_summary import selected_header_qc_summary
from app.services.segy_ingest_service import ingest_saved_segy
from app.services.segy_upload_storage import (
    cleanup_staged_upload,
    cleanup_staged_uploads,
    promote_staged_segy_to_raw,
    save_upload_file,
    staged_upload_dir,
)
from app.trace_store.naming import safe_upload_name


def inspect_segy_header_qc(path: str | Path) -> dict:
    from app.utils.header_qc import inspect_segy_header_qc as _inspect

    return _inspect(path)


async def stage_segy_upload(
    *,
    state: AppState,
    upload_dir: Path,
    file: UploadFile,
) -> dict:
    cleanup_staged_uploads(state, upload_dir=upload_dir)
    if not file.filename:
        raise HTTPException(
            status_code=400, detail='Uploaded file must have a filename'
        )

    staged_id = uuid4().hex
    safe_name = safe_upload_name(file.filename)
    raw_path = staged_upload_dir(upload_dir) / staged_id / safe_name
    saved = await save_upload_file(file, safe_name, raw_path=raw_path)
    try:
        header_qc = await asyncio.to_thread(inspect_segy_header_qc, saved.raw_path)
    except Exception as exc:  # noqa: BLE001
        cleanup_staged_upload(saved.raw_path, upload_dir=upload_dir)
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


async def ingest_staged_segy_upload(
    *,
    state: AppState,
    upload_dir: Path,
    trace_dir: Path,
    staged_id: str,
    key1_byte: int,
    key2_byte: int,
) -> dict:
    cleanup_staged_uploads(state, upload_dir=upload_dir)
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
    durable_raw_path = promote_staged_segy_to_raw(
        staged_path=raw_path,
        safe_name=safe_name,
        source_sha256=source_sha256,
        upload_dir=upload_dir,
    )
    result = await ingest_saved_segy(
        state=state,
        trace_dir=trace_dir,
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
    cleanup_staged_upload(raw_path, upload_dir=upload_dir)
    summary = selected_header_qc_summary(
        staged.get('header_qc'),
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    if summary is not None:
        result['header_qc'] = summary
    return result


__all__ = [
    'ingest_staged_segy_upload',
    'stage_segy_upload',
]
