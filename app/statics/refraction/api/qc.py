"""Refraction static QC APIs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.api._helpers import get_state
from app.api.routers.statics.common import _get_static_job_or_404
from app.statics.refraction.api.uploads import (
    _store_refraction_pick_upload,
    _validate_refraction_pick_upload,
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
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.statics.refraction.application.gather_preview import (
    RefractionStaticGatherPreviewError,
    RefractionStaticGatherPreviewNotFound,
    build_refraction_static_gather_preview,
)
from app.statics.refraction.application.qc_bundle import (
    RefractionStaticQcBundleError,
    build_refraction_static_qc_bundle,
)
from app.statics.refraction.application.qc_drilldown import (
    RefractionStaticQcDrilldownError,
    RefractionStaticQcDrilldownNotFound,
    build_refraction_static_qc_drilldown,
)
from app.statics.refraction.application.qc_endpoint_search import (
    RefractionStaticQcEndpointSearchError,
    build_refraction_static_qc_endpoint_search,
)
from app.statics.refraction.application.station_structure import (
    RefractionStaticStationStructureError,
    build_refraction_static_station_structure,
)
from app.statics.refraction.application.pick_map import (
    RefractionStaticPickMapError,
    build_refraction_static_pick_map,
)

router = APIRouter()


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
