"""Factory for common background-job lifecycle API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from app.api._helpers import get_state
from app.api.job_artifacts import (
    list_job_artifact_files,
    resolve_download_artifact_or_http_error,
)
from app.api.job_routes import (
    cancel_job_and_get_status_payload,
    get_job_or_404,
    job_status_payload,
)
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs


def build_job_lifecycle_router(
    *,
    route_prefix: str,
    allowed_job_types: set[str],
    status_response_model: type[Any],
    files_response_model: type[Any],
) -> APIRouter:
    """Build shared status, cancel, file-list, and download job routes."""
    router = APIRouter()
    allowed_types = set(allowed_job_types)

    @router.get(
        f'{route_prefix}/{{job_id}}/status',
        response_model=status_response_model,
    )
    def job_status(request: Request, job_id: str) -> Any:
        state = get_state(request.app)
        cleanup_in_memory_state(state)
        job = get_job_or_404(state, job_id, allowed_job_types=allowed_types)
        return job_status_payload(job)

    @router.post(
        f'{route_prefix}/{{job_id}}/cancel',
        response_model=status_response_model,
    )
    def job_cancel(request: Request, job_id: str) -> Any:
        state = get_state(request.app)
        cleanup_in_memory_state(state)
        return cancel_job_and_get_status_payload(
            state,
            job_id,
            allowed_job_types=allowed_types,
        )

    @router.get(
        f'{route_prefix}/{{job_id}}/files',
        response_model=files_response_model,
    )
    def job_files(request: Request, job_id: str) -> Any:
        state = get_state(request.app)
        cleanup_in_memory_state(state)
        job = get_job_or_404(state, job_id, allowed_job_types=allowed_types)

        maybe_cleanup_expired_jobs()
        return list_job_artifact_files(job)

    @router.get(f'{route_prefix}/{{job_id}}/download')
    def job_download(
        request: Request,
        job_id: str,
        name: str = Query(...),
    ) -> FileResponse:
        state = get_state(request.app)
        cleanup_in_memory_state(state)
        get_job_or_404(state, job_id, allowed_job_types=allowed_types)

        maybe_cleanup_expired_jobs()
        file_path = resolve_download_artifact_or_http_error(
            state,
            job_id=job_id,
            name=name,
            allowed_job_types=allowed_types,
        )

        return FileResponse(path=file_path, filename=Path(file_path).name)

    return router


__all__ = ['build_job_lifecycle_router']
