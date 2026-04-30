"""Static correction job APIs."""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.api._helpers import get_state
from app.api.schemas import (
    DatumStaticApplyRequest,
    DatumStaticApplyResponse,
    StaticJobFilesResponse,
    StaticJobStatusResponse,
)
from app.core.state import AppState
from app.services.datum_static_service import run_datum_static_apply_job
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.job_manager import JobManager
from app.services.job_runner import request_job_cancel, start_job_thread
from app.services.pipeline_artifacts import get_job_dir, maybe_cleanup_expired_jobs

router = APIRouter()


def _get_static_job_or_404(state: AppState, job_id: str) -> dict[str, object]:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        if job.get('job_type') != 'statics':
            raise HTTPException(status_code=404, detail='Job ID not found')
        return dict(job)


def _static_job_artifacts_dir(job: dict[str, object]) -> Path:
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise HTTPException(
            status_code=500,
            detail='Job metadata is inconsistent: artifacts_dir',
        )
    return Path(raw)


def _static_job_status_payload(
    job: dict[str, object],
) -> StaticJobStatusResponse:
    progress = job.get('progress', 0.0)
    message = job.get('message', '')
    return {
        'state': JobManager.normalize_status_value(job.get('status', 'unknown')),
        'progress': float(progress) if isinstance(progress, (int, float)) else 0.0,
        'message': message if isinstance(message, str) else '',
    }


@router.post('/statics/datum/apply', response_model=DatumStaticApplyResponse)
def datum_static_apply(
    req: DatumStaticApplyRequest,
    request: Request,
) -> DatumStaticApplyResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    job_id = str(uuid4())
    with state.lock:
        job_state = state.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='datum',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_datum_static_apply_job,
        args=(job_id, req, state),
    )

    return {'job_id': job_id, 'state': status}


@router.get(
    '/statics/job/{job_id}/status',
    response_model=StaticJobStatusResponse,
)
def static_job_status(request: Request, job_id: str) -> StaticJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, job_id)
    return _static_job_status_payload(job)


@router.post(
    '/statics/job/{job_id}/cancel',
    response_model=StaticJobStatusResponse,
)
def static_job_cancel(request: Request, job_id: str) -> StaticJobStatusResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    _get_static_job_or_404(state, job_id)

    request_job_cancel(state, job_id)

    job = _get_static_job_or_404(state, job_id)
    return _static_job_status_payload(job)


@router.get(
    '/statics/job/{job_id}/files',
    response_model=StaticJobFilesResponse,
)
def static_job_files(request: Request, job_id: str) -> StaticJobFilesResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, job_id)

    maybe_cleanup_expired_jobs()
    artifacts_dir = _static_job_artifacts_dir(job)
    if not artifacts_dir.is_dir():
        raise HTTPException(status_code=404, detail='Job artifacts not found')

    files = []
    for file_path in sorted(artifacts_dir.iterdir()):
        if not file_path.is_file():
            continue
        files.append(
            {
                'name': file_path.name,
                'size_bytes': int(file_path.stat().st_size),
            }
        )
    return {'files': files}


@router.get('/statics/job/{job_id}/download')
def static_job_download(
    request: Request,
    job_id: str,
    name: str = Query(...),
) -> FileResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, job_id)

    if not name or Path(name).name != name:
        raise HTTPException(status_code=400, detail='Invalid file name')

    maybe_cleanup_expired_jobs()
    artifacts_dir = _static_job_artifacts_dir(job)
    if not artifacts_dir.is_dir():
        raise HTTPException(status_code=404, detail='Job artifacts not found')
    file_path = artifacts_dir / name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail='File not found')

    return FileResponse(path=file_path, filename=name)
