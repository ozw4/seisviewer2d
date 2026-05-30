"""Shared API helpers for background-job artifact files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fastapi import HTTPException

from app.core.state import AppState
from app.services.job_artifact_refs import resolve_job_artifact_path


def job_artifacts_dir_or_404(job: Mapping[str, object]) -> Path:
    """Return a job's artifact directory or raise the route-compatible 404."""
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise HTTPException(status_code=404, detail='Job artifacts not found')

    artifacts_dir = Path(raw)
    if not artifacts_dir.is_dir():
        raise HTTPException(status_code=404, detail='Job artifacts not found')
    return artifacts_dir


def list_job_artifact_files(
    job: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Return top-level regular artifact files using the existing route shape."""
    artifacts_dir = job_artifacts_dir_or_404(job)
    files = []
    for file_path in sorted(artifacts_dir.iterdir(), key=lambda path: path.name):
        if not file_path.is_file():
            continue
        files.append(
            {
                'name': file_path.name,
                'size_bytes': int(file_path.stat().st_size),
            }
        )
    return {'files': files}


def resolve_download_artifact_or_http_error(
    state: AppState,
    *,
    job_id: str,
    name: str,
    allowed_job_types: set[str],
) -> Path:
    """Resolve a downloadable artifact path and map resolver errors for routes."""
    try:
        return resolve_job_artifact_path(
            state,
            job_id=job_id,
            name=name,
            allowed_job_types=allowed_job_types,
        )
    except ValueError as exc:
        message = str(exc)
        if 'artifact name must be a plain file name' in message:
            raise HTTPException(status_code=400, detail='Invalid file name') from exc
        if (
            'has no artifacts_dir' in message
            or 'artifacts_dir is not a directory' in message
        ):
            raise HTTPException(status_code=404, detail='Job artifacts not found') from exc
        if 'job_id not found' in message or 'unsupported job_type' in message:
            raise HTTPException(status_code=404, detail='Job ID not found') from exc
        if 'job artifact not found' in message:
            raise HTTPException(status_code=404, detail='File not found') from exc
        raise HTTPException(status_code=404, detail='File not found') from exc


__all__ = [
    'job_artifacts_dir_or_404',
    'list_job_artifact_files',
    'resolve_download_artifact_or_http_error',
]
