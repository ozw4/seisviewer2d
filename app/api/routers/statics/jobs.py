"""Static correction job lifecycle APIs."""

from __future__ import annotations

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
from app.contracts.statics.common import (
    StaticJobFilesResponse,
    StaticJobStatusResponse,
)
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs

router = APIRouter()


@router.get(
    '/statics/job/{job_id}/status',
    response_model=StaticJobStatusResponse,
)
def static_job_status(request: Request, job_id: str) -> StaticJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = get_job_or_404(state, job_id, allowed_job_types={'statics'})
    return job_status_payload(job)


@router.post(
    '/statics/job/{job_id}/cancel',
    response_model=StaticJobStatusResponse,
)
def static_job_cancel(request: Request, job_id: str) -> StaticJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    return cancel_job_and_get_status_payload(
        state,
        job_id,
        allowed_job_types={'statics'},
    )


@router.get(
    '/statics/job/{job_id}/files',
    response_model=StaticJobFilesResponse,
)
def static_job_files(request: Request, job_id: str) -> StaticJobFilesResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = get_job_or_404(state, job_id, allowed_job_types={'statics'})

    maybe_cleanup_expired_jobs()
    return list_job_artifact_files(job)


@router.get('/statics/job/{job_id}/download')
def static_job_download(
    request: Request,
    job_id: str,
    name: str = Query(...),
) -> FileResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    get_job_or_404(state, job_id, allowed_job_types={'statics'})

    maybe_cleanup_expired_jobs()
    file_path = resolve_download_artifact_or_http_error(
        state,
        job_id=job_id,
        name=name,
        allowed_job_types={'statics'},
    )

    return FileResponse(path=file_path, filename=name)
