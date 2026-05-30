"""Static correction job APIs."""

from __future__ import annotations

import json
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.api._helpers import get_state
from app.api.routers.statics.launch import launch_static_job, static_router_job_target
from app.api.routers.statics.uploads import (
    _store_refraction_pick_upload,
    _validate_refraction_pick_upload,
)
from app.contracts.statics.refraction.export import (
    RefractionStaticExportJobRequest,
    RefractionStaticExportJobResponse,
)
from app.contracts.statics.refraction.gather_preview import (
    RefractionStaticGatherPreviewRequest,
    RefractionStaticGatherPreviewResponse,
)
from app.contracts.statics.refraction.qc import (
    RefractionStaticStationStructureRequest,
    RefractionStaticStationStructureResponse,
    RefractionStaticQcBundleRequest,
    RefractionStaticQcBundleResponse,
    RefractionStaticQcDrilldownRequest,
    RefractionStaticQcDrilldownResponse,
    RefractionStaticQcEndpointSearchRequest,
    RefractionStaticQcEndpointSearchResponse,
    RefractionStaticPickMapRequest,
    RefractionStaticPickMapResponse,
)
from app.contracts.statics.refraction.table_apply import (
    RefractionStaticTableApplyRequest,
    RefractionStaticTableApplyResponse,
)
from app.core.state import AppState
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs
from app.services.refraction_static_export_service import (
    RefractionStaticExportSourceJobNotFound,
    RefractionStaticExportValidationError,
    validate_refraction_static_export_source_job,
)
from app.services.refraction_static_gather_preview import (
    RefractionStaticGatherPreviewError,
    RefractionStaticGatherPreviewNotFound,
    build_refraction_static_gather_preview,
)
from app.services.refraction_static_qc_bundle import (
    RefractionStaticQcBundleError,
    build_refraction_static_qc_bundle,
)
from app.services.refraction_static_qc_drilldown import (
    RefractionStaticQcDrilldownError,
    RefractionStaticQcDrilldownNotFound,
    build_refraction_static_qc_drilldown,
)
from app.services.refraction_static_qc_endpoint_search import (
    RefractionStaticQcEndpointSearchError,
    build_refraction_static_qc_endpoint_search,
)
from app.services.refraction_static_station_structure import (
    RefractionStaticStationStructureError,
    build_refraction_static_station_structure,
)
from app.services.refraction_static_pick_map import (
    RefractionStaticPickMapError,
    build_refraction_static_pick_map,
)
from app.services.refraction_static_table_apply_service import (
    run_refraction_static_table_apply_job,
)

router = APIRouter()


def _get_static_job_or_404(state: AppState, job_id: str) -> dict[str, object]:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        if job.get('job_type') != 'statics':
            raise HTTPException(status_code=404, detail='Job ID not found')
        return dict(job)


def _parse_refraction_pick_map_request_json(
    request_json: str,
) -> RefractionStaticPickMapRequest:
    try:
        return RefractionStaticPickMapRequest.model_validate_json(request_json)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json()),
        ) from exc


@router.post(
    '/statics/refraction/qc',
    response_model=RefractionStaticQcBundleResponse,
)
def refraction_static_qc_bundle(
    req: RefractionStaticQcBundleRequest,
    request: Request,
) -> RefractionStaticQcBundleResponse:
    """Return a compact QC bundle assembled from refraction static artifacts."""
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, req.job_id)
    try:
        return build_refraction_static_qc_bundle(
            job_id=req.job_id,
            job=job,
            req=req,
        )
    except RefractionStaticQcBundleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    '/statics/refraction/qc/endpoints',
    response_model=RefractionStaticQcEndpointSearchResponse,
    response_model_exclude_none=True,
)
def refraction_static_qc_endpoint_search(
    req: RefractionStaticQcEndpointSearchRequest,
    request: Request,
) -> RefractionStaticQcEndpointSearchResponse:
    """Return searchable source/receiver endpoint selector records."""
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, req.job_id)
    try:
        return build_refraction_static_qc_endpoint_search(
            job_id=req.job_id,
            job=job,
            req=req,
        )
    except RefractionStaticQcEndpointSearchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    '/statics/refraction/qc/pick-map',
    response_model=RefractionStaticPickMapResponse,
    response_model_exclude_none=True,
)
async def refraction_static_qc_pick_map(
    request: Request,
) -> RefractionStaticPickMapResponse:
    """Return all-gather before/after refraction pick-map arrays."""
    content_type = request.headers.get('content-type', '').lower()
    uploaded_pick_path: Path | None = None
    temp_dir_handle: tempfile.TemporaryDirectory[str] | None = None
    try:
        if content_type.startswith('multipart/form-data'):
            form = await request.form()
            request_json = form.get('request_json')
            pick_npz = form.get('pick_npz')
            if not isinstance(request_json, str):
                raise HTTPException(
                    status_code=422,
                    detail='request_json form field is required',
                )
            if not hasattr(pick_npz, 'filename') or not hasattr(pick_npz, 'file'):
                raise HTTPException(
                    status_code=422,
                    detail='pick_npz form file is required',
                )
            req = _parse_refraction_pick_map_request_json(request_json)
            _validate_refraction_pick_upload(pick_npz)
            temp_dir_handle = tempfile.TemporaryDirectory(prefix='refraction-pick-map-')
            temp_dir = Path(temp_dir_handle.name)
            uploaded_pick_path, _size_bytes = _store_refraction_pick_upload(
                pick_npz=pick_npz,
                job_dir=temp_dir,
            )
        else:
            try:
                req = RefractionStaticPickMapRequest.model_validate(
                    await request.json()
                )
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=json.loads(exc.json()),
                ) from exc

        state = get_state(request.app)
        cleanup_in_memory_state(state)
        job = _get_static_job_or_404(state, req.job_id) if req.job_id else None
        try:
            return build_refraction_static_pick_map(
                req=req,
                state=state,
                job=job,
                uploaded_pick_npz_path=uploaded_pick_path,
            )
        except RefractionStaticPickMapError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        if temp_dir_handle is not None:
            temp_dir_handle.cleanup()


@router.post(
    '/statics/refraction/qc/drilldown',
    response_model=RefractionStaticQcDrilldownResponse,
    response_model_exclude_none=True,
)
def refraction_static_qc_drilldown(
    req: RefractionStaticQcDrilldownRequest,
    request: Request,
) -> RefractionStaticQcDrilldownResponse:
    """Return detailed QC rows for a selected refraction endpoint or cell."""
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, req.job_id)
    try:
        return build_refraction_static_qc_drilldown(
            job_id=req.job_id,
            job=job,
            req=req,
        )
    except RefractionStaticQcDrilldownNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RefractionStaticQcDrilldownError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    '/statics/refraction/qc/station-structure',
    response_model=RefractionStaticStationStructureResponse,
)
def refraction_static_qc_station_structure(
    req: RefractionStaticStationStructureRequest,
    request: Request,
) -> RefractionStaticStationStructureResponse:
    """Return source/receiver station-structure QC series."""
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, req.job_id)
    try:
        return build_refraction_static_station_structure(
            job_id=req.job_id,
            job=job,
            req=req,
        )
    except RefractionStaticStationStructureError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    '/statics/refraction/qc/gather-preview',
    response_model=RefractionStaticGatherPreviewResponse,
    response_model_exclude_none=True,
)
def refraction_static_gather_preview(
    req: RefractionStaticGatherPreviewRequest,
    request: Request,
) -> RefractionStaticGatherPreviewResponse:
    """Return bounded before/after gather samples and overlays for refraction QC."""
    state = get_state(request.app)
    cleanup_in_memory_state(state)
    job = _get_static_job_or_404(state, req.job_id)
    try:
        return build_refraction_static_gather_preview(
            job_id=req.job_id,
            job=job,
            req=req,
            state=state,
        )
    except RefractionStaticGatherPreviewNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RefractionStaticGatherPreviewError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
