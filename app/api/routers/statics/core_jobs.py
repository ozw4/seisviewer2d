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
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs
from app.services.static_job_targets import get_static_job_target

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
        target=get_static_job_target('datum'),
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
        target=get_static_job_target('first_break_qc'),
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
        target=get_static_job_target('geometry_linkage'),
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
        target=get_static_job_target('residual'),
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
        target=get_static_job_target('time_term'),
        target_args=lambda job_id: (job_id, req, state),
    )

    return {'job_id': launched.job_id, 'state': launched.state}
