"""Batch apply job APIs (scaffold for asynchronous processing)."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.api._helpers import get_state
from app.api.schemas import (
    BatchApplyRequest,
    BatchApplyResponse,
    BatchJobFilesResponse,
    BatchJobStatusResponse,
)
from app.services.batch_apply_service import run_batch_apply_job
from app.services.in_memory_cleanup import cleanup_in_memory_state
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

    threading.Thread(
        target=run_batch_apply_job,
        args=(job_id, req, state),
        daemon=True,
    ).start()

    return {'job_id': job_id, 'state': status}


@router.get('/batch/job/{job_id}/status', response_model=BatchJobStatusResponse)
def batch_job_status(request: Request, job_id: str) -> BatchJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)

    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        job_state = job.get('status', 'unknown')
        progress = job.get('progress', 0.0)
        message = job.get('message', '')
    return {
        'state': job_state,
        'progress': progress,
        'message': message,
    }


@router.get('/batch/job/{job_id}/files', response_model=BatchJobFilesResponse)
def batch_job_files(request: Request, job_id: str) -> BatchJobFilesResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    with state.lock:
        job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job ID not found')

    maybe_cleanup_expired_jobs()
    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail='Job artifacts not found')

    files = []
    for file_path in sorted(job_dir.iterdir()):
        if not file_path.is_file():
            continue
        files.append(
            {
                'name': file_path.name,
                'size_bytes': int(file_path.stat().st_size),
            }
        )
    return {'files': files}


@router.get('/batch/job/{job_id}/download')
def batch_job_download(
    request: Request,
    job_id: str,
    name: str = Query(...),
) -> FileResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    with state.lock:
        job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job ID not found')

    if Path(name).name != name:
        raise HTTPException(status_code=400, detail='Invalid file name')

    maybe_cleanup_expired_jobs()
    file_path = get_job_dir(job_id) / name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail='File not found')

    return FileResponse(path=file_path, filename=name)
