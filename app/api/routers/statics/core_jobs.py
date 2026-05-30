"""Core non-refraction static correction job launch APIs."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api._helpers import get_state
from app.api.routers.statics.launch import launch_static_job
from app.contracts.statics.datum import (
    DatumStaticApplyRequest,
    DatumStaticApplyResponse,
)
from app.contracts.statics.first_break_qc import (
    FirstBreakQcJobResponse,
    FirstBreakQcRequest,
)
from app.contracts.statics.geometry_linkage import (
    StaticLinkageBuildRequest,
    StaticLinkageBuildResponse,
)
from app.contracts.statics.residual import (
    ResidualStaticApplyRequest,
    ResidualStaticApplyResponse,
)
from app.contracts.statics.time_term import (
    TimeTermStaticApplyRequest,
    TimeTermStaticApplyResponse,
)
from app.services.datum_static_service import run_datum_static_apply_job
from app.services.first_break_qc_service import run_first_break_qc_job
from app.services.geometry_linkage_service import run_geometry_linkage_build_job
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs
from app.services.residual_static_service import run_residual_static_apply_job
from app.services.time_term_static_service import run_time_term_static_apply_job

router = APIRouter()


@router.post('/statics/datum/apply', response_model=DatumStaticApplyResponse)
def datum_static_apply(
    req: DatumStaticApplyRequest,
    request: Request,
) -> DatumStaticApplyResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='datum',
        target=run_datum_static_apply_job,
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}


@router.post('/statics/first-break/qc', response_model=FirstBreakQcJobResponse)
def first_break_qc(
    req: FirstBreakQcRequest,
    request: Request,
) -> FirstBreakQcJobResponse:
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='first_break_qc',
        target=run_first_break_qc_job,
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}


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

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='geometry_linkage',
        target=run_geometry_linkage_build_job,
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}


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

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='residual',
        target=run_residual_static_apply_job,
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}


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

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='time_term',
        target=run_time_term_static_apply_job,
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}
