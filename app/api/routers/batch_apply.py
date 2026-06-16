"""Batch apply job APIs (scaffold for asynchronous processing)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api._helpers import get_state
from app.api.job_lifecycle_router import build_job_lifecycle_router
from app.contracts.batch import (
    BatchApplyRequest,
    BatchApplyResponse,
    BatchJobFilesResponse,
    BatchJobStatusResponse,
)
from app.services.batch_apply_service import run_batch_apply_job
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.jobs import launch_managed_job
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs

router = APIRouter()
router.include_router(
    build_job_lifecycle_router(
        route_prefix='/batch/job',
        allowed_job_types={'batch_apply'},
        status_response_model=BatchJobStatusResponse,
        files_response_model=BatchJobFilesResponse,
    )
)


@router.post('/batch/apply', response_model=BatchApplyResponse)
def batch_apply(req: BatchApplyRequest, request: Request) -> BatchApplyResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    launched = launch_managed_job(
        state,
        create_job=lambda job_id, artifacts_dir: state.jobs.create_batch_apply_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            artifacts_dir=str(artifacts_dir),
        ),
        target=run_batch_apply_job,
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}
