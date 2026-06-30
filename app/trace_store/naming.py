"""TraceStore and upload naming helpers."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

CONTENT_ADDRESSED_STORE_NAME_MAX_CHARS = 180
DIRECT_IMPORT_RAW_NAME_MAX_CHARS = 180


def safe_upload_name(filename: str) -> str:
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    if safe_name in {'', '.', '..'}:
        raise HTTPException(
            status_code=400,
            detail='Uploaded file must have a safe filename',
        )
    return safe_name


def safe_store_name(store_name: str) -> str:
    if store_name in {'', '.', '..'}:
        raise HTTPException(status_code=400, detail='Trace store name is unsafe')
    if Path(store_name).is_absolute():
        raise HTTPException(status_code=400, detail='Trace store name is unsafe')
    if '/' in store_name or '\\' in store_name:
        raise HTTPException(status_code=400, detail='Trace store name is unsafe')
    safe_name = safe_upload_name(store_name)
    if safe_name != store_name:
        raise HTTPException(status_code=400, detail='Trace store name is unsafe')
    return safe_name


def bounded_direct_import_raw_name(safe_name: str) -> str:
    if len(safe_name) <= DIRECT_IMPORT_RAW_NAME_MAX_CHARS:
        return safe_name

    path = Path(safe_name)
    suffix = path.suffix
    if len(suffix) >= DIRECT_IMPORT_RAW_NAME_MAX_CHARS:
        suffix = ''
    stem_limit = DIRECT_IMPORT_RAW_NAME_MAX_CHARS - len(suffix)
    stem = path.stem or 'segy'
    stem = stem[:stem_limit] or 'segy'
    return safe_upload_name(f'{stem}{suffix}')


def content_addressed_compare_store_name(
    *,
    safe_name: str,
    source_sha256: str,
    key1_byte: int,
    key2_byte: int,
) -> str:
    source_hash = source_sha256.lower()
    if not re.fullmatch(r'[0-9a-f]{64}', source_hash):
        raise HTTPException(status_code=400, detail='Source sha256 is invalid')
    suffix = f'__k{key1_byte}_{key2_byte}__sha256_{source_hash}'
    stem_limit = CONTENT_ADDRESSED_STORE_NAME_MAX_CHARS - len(suffix)
    if stem_limit < 1:
        raise HTTPException(status_code=400, detail='Content-addressed name is invalid')
    safe_stem = safe_upload_name(Path(safe_name).stem or 'segy')
    safe_stem = safe_stem[:stem_limit] or 'segy'
    store_name = f'{safe_stem}{suffix}'
    return safe_store_name(store_name)
