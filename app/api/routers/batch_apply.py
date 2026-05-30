"""Batch apply job APIs (scaffold for asynchronous processing)."""

from __future__ import annotations

import threading

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
from app.contracts.batch import (
    BatchApplyRequest,
    BatchApplyResponse,
    BatchJobFilesResponse,
    BatchJobStatusResponse,
)
from app.services.batch_apply_service import run_batch_apply_job
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.job_runner import start_job_thread
from app.services.pipeline_artifacts import get_job_dir, maybe_cleanup_expired_jobs
from uuid import uuid4

router = APIRouter()


@router.post('/batch/apply', response_model=BatchApplyResponse)
def batch_apply(req: BatchApplyRequest, request: Request) -> BatchApplyResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    job_id = str(uuid4())
    with state.lock:
        job_state = state.jobs.create_batch_apply_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_batch_apply_job,
        args=(job_id, req, state),
    )

    return {'job_id': job_id, 'state': status}


@router.get('/batch/job/{job_id}/status', response_model=BatchJobStatusResponse)
def batch_job_status(request: Request, job_id: str) -> BatchJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = get_job_or_404(state, job_id, allowed_job_types={'batch_apply'})
    return job_status_payload(job)


@router.post('/batch/job/{job_id}/cancel', response_model=BatchJobStatusResponse)
def batch_job_cancel(request: Request, job_id: str) -> BatchJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    return cancel_job_and_get_status_payload(
        state,
        job_id,
        allowed_job_types={'batch_apply'},
    )


@router.get('/batch/job/{job_id}/files', response_model=BatchJobFilesResponse)
def batch_job_files(request: Request, job_id: str) -> BatchJobFilesResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = get_job_or_404(state, job_id, allowed_job_types={'batch_apply'})

    maybe_cleanup_expired_jobs()
    return list_job_artifact_files(job)


@router.get('/batch/job/{job_id}/download')
def batch_job_download(
    request: Request,
    job_id: str,
    name: str = Query(...),
) -> FileResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    get_job_or_404(state, job_id, allowed_job_types={'batch_apply'})

    maybe_cleanup_expired_jobs()
    file_path = resolve_download_artifact_or_http_error(
        state,
        job_id=job_id,
        name=name,
        allowed_job_types={'batch_apply'},
    )

    return FileResponse(path=file_path, filename=name)
