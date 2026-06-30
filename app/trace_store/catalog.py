"""TraceStore catalog and metadata helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.utils.baseline_artifacts import has_split_baseline_artifacts

logger = logging.getLogger(__name__)


def trace_store_complete(store_dir: Path, key1_byte: int, key2_byte: int) -> bool:
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
    meta = load_trace_store_meta(meta_path)
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


def load_trace_store_meta(meta_path: Path) -> dict | None:
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    return meta


def ensure_trace_store_meta_key_bytes(
    meta: dict,
    *,
    key1_byte: int,
    key2_byte: int,
) -> None:
    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        raise HTTPException(
            status_code=409,
            detail='TraceStore key bytes do not match requested key bytes',
        )
    if key_bytes.get('key1') != key1_byte or key_bytes.get('key2') != key2_byte:
        raise HTTPException(
            status_code=409,
            detail='TraceStore key bytes do not match requested key bytes',
        )


def ensure_trace_store_meta_identity(
    store_dir: Path,
    meta: dict,
    *,
    raw_path: Path,
    original_name: str,
    display_name: str,
    store_name: str,
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
    if updated.get('source_size') != source_size:
        updated['source_size'] = source_size
        changed = True
    if updated.get('source_sha256') != source_sha256:
        updated['source_sha256'] = source_sha256
        changed = True
    if updated.get('original_name') != original_name:
        updated['original_name'] = original_name
        changed = True
    if updated.get('display_name') != display_name:
        updated['display_name'] = display_name
        changed = True
    if updated.get('store_name') != store_name:
        updated['store_name'] = store_name
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


def trace_store_matches_source(
    store_dir: Path,
    key1_byte: int,
    key2_byte: int,
    *,
    source_sha256: str | None = None,
    source_size: int | None = None,
) -> dict | None:
    if not trace_store_complete(store_dir, key1_byte, key2_byte):
        return None
    meta_path = store_dir / 'meta.json'
    meta = load_trace_store_meta(meta_path)
    if meta is None:
        return None
    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        return None
    if key_bytes.get('key1') != key1_byte or key_bytes.get('key2') != key2_byte:
        return None
    # Prefer exact identity via content hash when available.
    meta_hash = meta.get('source_sha256')
    if isinstance(source_sha256, str) and isinstance(meta_hash, str):
        return meta if meta_hash == source_sha256 else None

    # Fallback: size match only for legacy stores without hash.
    if isinstance(source_size, int):
        original_size = meta.get('original_size')
        if isinstance(original_size, int) and original_size == source_size:
            return meta
        return None
    return meta


def archive_trace_store(store_dir: Path) -> None:
    if not store_dir.exists():
        return
    archive_dir = store_dir.parent / f'{store_dir.name}.old-{uuid4().hex}'
    store_dir.rename(archive_dir)


def trace_store_summary(store_dir: Path) -> dict | None:
    meta_path = store_dir / 'meta.json'
    meta = load_trace_store_meta(meta_path)
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

    if not trace_store_complete(store_dir, key1_byte, key2_byte):
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
    display_name = meta.get('display_name')
    if not isinstance(display_name, str) or not display_name:
        display_name = original_name or store_dir.name

    summary = {
        'original_name': original_name,
        'display_name': display_name,
        'store_name': store_dir.name,
        'key1_byte': key1_byte,
        'key2_byte': key2_byte,
        'last_used_ts': last_used_ts,
    }
    source_sha256 = meta.get('source_sha256')
    if isinstance(source_sha256, str):
        summary['source_sha256'] = source_sha256
    source_size = meta.get('source_size')
    if not isinstance(source_size, int):
        source_size = meta.get('original_size')
    if isinstance(source_size, int):
        summary['source_size'] = source_size
    for key in ('dt', 'n_traces', 'n_samples'):
        value = meta.get(key)
        if isinstance(value, (int, float)):
            summary[key] = value
    return summary


def list_recent_dataset_summaries(trace_dir: Path) -> list[dict]:
    summaries: list[dict] = []
    if not trace_dir.exists():
        return summaries

    for child in trace_dir.iterdir():
        if not child.is_dir():
            continue
        summary = trace_store_summary(child)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=lambda item: item['last_used_ts'], reverse=True)
    return summaries
