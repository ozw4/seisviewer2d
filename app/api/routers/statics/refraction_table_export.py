"""Refraction static table apply and export job APIs."""

from __future__ import annotations

from collections.abc import MutableMapping

from fastapi import APIRouter, HTTPException, Request

from app.api._helpers import get_state
from app.api.routers.statics.launch import launch_static_job, static_router_job_target
from app.contracts.statics.refraction.export import (
    RefractionStaticExportJobRequest,
    RefractionStaticExportJobResponse,
)
from app.contracts.statics.refraction.table_apply import (
    RefractionStaticTableApplyRequest,
    RefractionStaticTableApplyResponse,
)
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs
from app.services.refraction_static_export_service import (
    RefractionStaticExportSourceJobNotFound,
    RefractionStaticExportValidationError,
    validate_refraction_static_export_source_job,
)
from app.services.refraction_static_table_apply_service import (
    run_refraction_static_table_apply_job,
)

router = APIRouter()


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

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='refraction_static_table_apply',
        target=run_refraction_static_table_apply_job,
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}


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

    def _after_create(job_state: MutableMapping[str, object]) -> None:
        job_state['source_job_id'] = source.source_job_id
        job_state['export_formats'] = list(source.requested_formats)

    launched = launch_static_job(
        state=state,
        file_id=source.source_file_id,
        key1_byte=source.key1_byte,
        key2_byte=source.key2_byte,
        statics_kind='refraction_export',
        target=static_router_job_target('run_refraction_static_export_job'),
        target_args=lambda job_id: (job_id, req, state),
        after_create=_after_create,
    )

    return {
        'job_id': launched.job_id,
        'state': launched.state,
        'source_job_id': source.source_job_id,
        'requested_formats': list(source.requested_formats),
    }
