"""Uploaded static-correction artifact helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.services.refraction_static_artifacts import UPLOADED_REFRACTION_PICKS_NPZ_NAME

_UPLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_UPLOADED_PICK_NPZ_BYTES = 256 * 1024 * 1024
_ACCEPTED_NPZ_CONTENT_TYPES = {
    'application/x-npz',
    'application/zip',
    'application/x-zip-compressed',
}


def _validate_refraction_pick_upload(pick_npz: UploadFile) -> None:
    filename = pick_npz.filename or ''
    content_type = (pick_npz.content_type or '').split(';', 1)[0].strip().lower()
    if filename.lower().endswith('.npz'):
        return
    if content_type in _ACCEPTED_NPZ_CONTENT_TYPES:
        return
    raise HTTPException(
        status_code=422,
        detail='pick_npz must be an .npz upload',
    )


def _store_refraction_pick_upload(
    *,
    pick_npz: UploadFile,
    job_dir: Path,
) -> tuple[Path, int]:
    job_dir.mkdir(parents=True, exist_ok=True)
    target_path = job_dir / UPLOADED_REFRACTION_PICKS_NPZ_NAME
    tmp_path = target_path.with_name(f'{target_path.name}.{uuid4().hex}.tmp')
    total_size = 0
    try:
        with tmp_path.open('wb') as handle:
            while True:
                chunk = pick_npz.file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > _MAX_UPLOADED_PICK_NPZ_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail='pick_npz exceeds maximum upload size',
                    )
                handle.write(chunk)
        if total_size == 0:
            raise HTTPException(status_code=422, detail='pick_npz is empty')
        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target_path, total_size
