"""Storage helpers for uploaded SEG-Y files."""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.state import AppState
from app.services.staged_upload_cleanup import (
    cleanup_staged_upload as cleanup_staged_upload_dir,
)
from app.services.staged_upload_cleanup import cleanup_stale_staged_upload_dirs

UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024
STAGED_UPLOAD_CLEANUP_INTERVAL_SEC = 600
_last_staged_cleanup_ts = 0.0


@dataclass(frozen=True)
class SavedUpload:
    original_name: str
    safe_name: str
    raw_path: Path
    source_sha256: str
    source_size: int


def staged_upload_dir(upload_dir: Path) -> Path:
    return upload_dir / 'staged'


def cleanup_staged_upload(raw_path: Path, *, upload_dir: Path) -> None:
    cleanup_staged_upload_dir(raw_path, staged_root=staged_upload_dir(upload_dir))


def cleanup_discarded_staged_upload(
    _key,
    value,
    _reason: str,
    *,
    upload_dir: Path,
) -> None:
    if not isinstance(value, dict):
        return
    raw_path = value.get('raw_path')
    if not isinstance(raw_path, (str, Path)):
        return
    cleanup_staged_upload(Path(raw_path), upload_dir=upload_dir)


def ensure_staged_upload_cleanup_callback(
    state: AppState,
    *,
    upload_dir: Path,
) -> None:
    def _cleanup_discarded_staged_upload(_key, value, _reason: str) -> None:
        cleanup_discarded_staged_upload(
            _key,
            value,
            _reason,
            upload_dir=upload_dir,
        )

    state.staged_uploads.set_on_discard(_cleanup_discarded_staged_upload)


def cleanup_staged_uploads(
    state: AppState,
    *,
    upload_dir: Path,
    force: bool = False,
    now_ts: float | None = None,
) -> int:
    """Purge expired staged records and stale staged directories."""
    global _last_staged_cleanup_ts

    now = time.time() if now_ts is None else float(now_ts)
    with state.lock:
        ensure_staged_upload_cleanup_callback(state, upload_dir=upload_dir)
        removed = state.staged_uploads.purge_expired()
        active_ids = set(state.staged_uploads.keys())
        ttl_sec = state.staged_uploads.ttl_sec

    if not force and now - _last_staged_cleanup_ts < STAGED_UPLOAD_CLEANUP_INTERVAL_SEC:
        return removed

    _last_staged_cleanup_ts = now
    removed += cleanup_stale_staged_upload_dirs(
        staged_root=staged_upload_dir(upload_dir),
        ttl_sec=ttl_sec,
        active_ids=active_ids,
        now_ts=now,
    )
    return removed


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(UPLOAD_CHUNK_SIZE), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def promote_staged_segy_to_raw(
    *,
    staged_path: Path,
    safe_name: str,
    source_sha256: str,
    upload_dir: Path,
) -> Path:
    raw_dir = upload_dir / 'raw' / source_sha256
    raw_path = raw_dir / safe_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        if raw_path.is_file() and sha256_file(raw_path) == source_sha256:
            return raw_path
        raw_path = raw_dir / f'{raw_path.stem}.{uuid4().hex}{raw_path.suffix}'

    tmp_path = raw_path.with_suffix(raw_path.suffix + f'.{uuid4().hex}.tmp')
    try:
        shutil.copy2(staged_path, tmp_path)
        tmp_path.replace(raw_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f'Unable to promote staged SEG-Y: {exc}',
        ) from exc
    return raw_path


async def save_upload_file(
    file: UploadFile,
    safe_name: str,
    *,
    raw_path: Path,
) -> SavedUpload:
    original_name = file.filename or safe_name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = raw_path.with_suffix(raw_path.suffix + f'.{uuid4().hex}.tmp')
    hasher = hashlib.sha256()
    try:
        with tmp_path.open('wb') as temp_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                temp_file.write(chunk)
        tmp_path.replace(raw_path)
        source_size = raw_path.stat().st_size
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SavedUpload(
        original_name=original_name,
        safe_name=safe_name,
        raw_path=raw_path,
        source_sha256=hasher.hexdigest(),
        source_size=int(source_size),
    )


__all__ = [
    'STAGED_UPLOAD_CLEANUP_INTERVAL_SEC',
    'UPLOAD_CHUNK_SIZE',
    'SavedUpload',
    'cleanup_discarded_staged_upload',
    'cleanup_staged_upload',
    'cleanup_staged_uploads',
    'ensure_staged_upload_cleanup_callback',
    'promote_staged_segy_to_raw',
    'save_upload_file',
    'sha256_file',
    'staged_upload_dir',
]
