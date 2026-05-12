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
    FirstBreakQcJobResponse,
    FirstBreakQcRequest,
    RefractionStaticApplyRequest,
    RefractionStaticApplyResponse,
    RefractionStaticExportJobRequest,
    RefractionStaticExportJobResponse,
    RefractionStaticTableApplyRequest,
    RefractionStaticTableApplyResponse,
    ResidualStaticApplyRequest,
    ResidualStaticApplyResponse,
    StaticLinkageBuildRequest,
    StaticLinkageBuildResponse,
    StaticJobFilesResponse,
    StaticJobStatusResponse,
    TimeTermStaticApplyRequest,
    TimeTermStaticApplyResponse,
)
from app.core.state import AppState
from app.services.datum_static_service import run_datum_static_apply_job
from app.services.first_break_qc_service import run_first_break_qc_job
from app.services.geometry_linkage_service import run_geometry_linkage_build_job
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.job_manager import JobManager
from app.services.job_runner import request_job_cancel, start_job_thread
from app.services.pipeline_artifacts import get_job_dir, maybe_cleanup_expired_jobs
from app.services.refraction_static_export_service import (
    RefractionStaticExportSourceJobNotFound,
    RefractionStaticExportValidationError,
    resolve_refraction_static_export_formats,
    run_refraction_static_export_job,
    validate_refraction_static_export_source_job,
)
from app.services.refraction_static_service import run_refraction_static_apply_job
from app.services.refraction_static_table_apply_service import (
    run_refraction_static_table_apply_job,
)
from app.services.residual_static_service import run_residual_static_apply_job
from app.services.time_term_static_service import run_time_term_static_apply_job

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


@router.post('/statics/first-break/qc', response_model=FirstBreakQcJobResponse)
def first_break_qc(
    req: FirstBreakQcRequest,
    request: Request,
) -> FirstBreakQcJobResponse:
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
            statics_kind='first_break_qc',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_first_break_qc_job,
        args=(job_id, req, state),
    )

    return {'job_id': job_id, 'state': status}


@router.post(
    '/statics/linkage/build',
    response_model=StaticLinkageBuildResponse,
)
def static_linkage_build(
    req: StaticLinkageBuildRequest,
    request: Request,
) -> StaticLinkageBuildResponse:
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
            statics_kind='geometry_linkage',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_geometry_linkage_build_job,
        args=(job_id, req, state),
    )

    return {'job_id': job_id, 'state': status}


@router.post(
    '/statics/residual/apply',
    response_model=ResidualStaticApplyResponse,
)
def residual_static_apply(
    req: ResidualStaticApplyRequest,
    request: Request,
) -> ResidualStaticApplyResponse:
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
            statics_kind='residual',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_residual_static_apply_job,
        args=(job_id, req, state),
    )

    return {'job_id': job_id, 'state': status}


@router.post(
    '/statics/time-term/apply',
    response_model=TimeTermStaticApplyResponse,
)
def time_term_static_apply(
    req: TimeTermStaticApplyRequest,
    request: Request,
) -> TimeTermStaticApplyResponse:
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
            statics_kind='time_term',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_time_term_static_apply_job,
        args=(job_id, req, state),
    )

    return {'job_id': job_id, 'state': status}


@router.post(
    '/statics/refraction/apply',
    response_model=RefractionStaticApplyResponse,
    response_model_exclude_none=True,
)
def refraction_static_apply(
    req: RefractionStaticApplyRequest,
    request: Request,
) -> RefractionStaticApplyResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    job_id = str(uuid4())
    requested_formats = resolve_refraction_static_export_formats(req.export)
    with state.lock:
        job_state = state.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        if req.export.enabled:
            job_state['export_formats'] = list(requested_formats)
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_refraction_static_apply_job,
        args=(job_id, req, state),
    )

    response: RefractionStaticApplyResponse = {'job_id': job_id, 'state': status}
    if req.export.enabled:
        response['requested_formats'] = list(requested_formats)
    return response


@router.post(
    '/statics/refraction/static-table/apply',
    response_model=RefractionStaticTableApplyResponse,
)
def refraction_static_table_apply(
    req: RefractionStaticTableApplyRequest,
    request: Request,
) -> RefractionStaticTableApplyResponse:
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
            statics_kind='refraction_static_table_apply',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_refraction_static_table_apply_job,
        args=(job_id, req, state),
    )

    return {'job_id': job_id, 'state': status}


@router.post(
    '/statics/refraction/export',
    response_model=RefractionStaticExportJobResponse,
)
def refraction_static_export(
    req: RefractionStaticExportJobRequest,
    request: Request,
) -> RefractionStaticExportJobResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    try:
        source = validate_refraction_static_export_source_job(req=req, state=state)
    except RefractionStaticExportSourceJobNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RefractionStaticExportValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    job_id = str(uuid4())
    with state.lock:
        job_state = state.jobs.create_static_job(
            job_id,
            file_id=source.source_file_id,
            key1_byte=source.key1_byte,
            key2_byte=source.key2_byte,
            statics_kind='refraction_export',
            artifacts_dir=str(get_job_dir(job_id)),
        )
        job_state['source_job_id'] = source.source_job_id
        job_state['export_formats'] = list(source.requested_formats)
        status = job_state['status']

    start_job_thread(
        thread_factory=threading.Thread,
        target=run_refraction_static_export_job,
        args=(job_id, req, state),
    )

    return {
        'job_id': job_id,
        'state': status,
        'source_job_id': source.source_job_id,
        'requested_formats': list(source.requested_formats),
    }


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
    _get_static_job_or_404(state, job_id)

    maybe_cleanup_expired_jobs()
    try:
        file_path = resolve_job_artifact_path(
            state,
            job_id=job_id,
            name=name,
            allowed_job_types={'statics'},
        )
    except ValueError as exc:
        if 'artifact name must be a plain file name' in str(exc):
            raise HTTPException(status_code=400, detail='Invalid file name') from exc
        if 'artifacts_dir is not a directory' in str(exc):
            raise HTTPException(status_code=404, detail='Job artifacts not found') from exc
        if 'job artifact not found' in str(exc):
            raise HTTPException(status_code=404, detail='File not found') from exc
        raise HTTPException(status_code=404, detail='File not found') from exc

    return FileResponse(path=file_path, filename=name)
