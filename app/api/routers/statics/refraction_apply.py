"""Refraction static apply and uploaded-picks validation APIs."""

from __future__ import annotations

import json
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from app.api._helpers import get_state
from app.api.routers.statics.launch import launch_static_job, static_router_job_target
from app.api.routers.statics.uploads import (
    _store_refraction_pick_upload,
    _validate_refraction_pick_upload,
)
from app.contracts.statics.refraction.apply import (
    RefractionStaticApplyRequest,
    RefractionStaticApplyResponse,
)
from app.services.in_memory_cleanup import cleanup_in_memory_state
from app.services.pipeline_artifacts import maybe_cleanup_expired_jobs
from app.statics.refraction.artifacts import UPLOADED_REFRACTION_PICKS_NPZ_NAME
from app.statics.refraction.application.export_service import (
    resolve_refraction_static_export_formats,
)
from app.services.refraction_static_validation_service import (
    validate_refraction_static_inputs_with_picks,
)

router = APIRouter()


def _parse_refraction_apply_request_json(
    request_json: str,
) -> RefractionStaticApplyRequest:
    try:
        return RefractionStaticApplyRequest.model_validate_json(request_json)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json()),
        ) from exc


@router.post(
    '/statics/refraction/apply',
    response_model=RefractionStaticApplyResponse,
    response_model_exclude_none=True,
)
def refraction_static_apply(
    req: RefractionStaticApplyRequest,
    request: Request,
) -> RefractionStaticApplyResponse:
    if req.pick_source.kind == 'uploaded_npz':
        raise HTTPException(
            status_code=422,
            detail=(
                'pick_source.kind uploaded_npz requires multipart '
                '/statics/refraction/apply-with-picks'
            ),
        )

    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    requested_formats = resolve_refraction_static_export_formats(req.export)

    def _after_create(job_state: MutableMapping[str, object]) -> None:
        if req.export.enabled:
            job_state['export_formats'] = list(requested_formats)

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='refraction',
        target=static_router_job_target('run_refraction_static_apply_job'),
        target_args=lambda job_id: (job_id, req, state),
        after_create=_after_create,
    )

    response: RefractionStaticApplyResponse = {
        'job_id': launched.job_id,
        'state': launched.state,
    }
    if req.export.enabled:
        response['requested_formats'] = list(requested_formats)
    return response


@router.post(
    '/statics/refraction/apply-with-picks',
    response_model=RefractionStaticApplyResponse,
    response_model_exclude_none=True,
)
def refraction_static_apply_with_picks(
    request: Request,
    request_json: Annotated[str, Form(...)],
    pick_npz: Annotated[UploadFile, File(...)],
) -> RefractionStaticApplyResponse:
    req = _parse_refraction_apply_request_json(request_json)
    if req.pick_source.kind != 'uploaded_npz':
        raise HTTPException(
            status_code=422,
            detail='pick_source.kind must be uploaded_npz',
        )
    _validate_refraction_pick_upload(pick_npz)

    state = get_state(request.app)
    cleanup_in_memory_state(state)
    maybe_cleanup_expired_jobs()

    stored_path: Path | None = None
    upload_metadata = {
        'original_filename': pick_npz.filename or '',
        'stored_name': UPLOADED_REFRACTION_PICKS_NPZ_NAME,
    }

    requested_formats = resolve_refraction_static_export_formats(req.export)

    def _pre_create(_job_id: str, artifacts_dir: Path) -> None:
        nonlocal stored_path
        stored_path, _size_bytes = _store_refraction_pick_upload(
            pick_npz=pick_npz,
            job_dir=artifacts_dir,
        )

    def _after_create(job_state: MutableMapping[str, object]) -> None:
        job_state['pick_source'] = {
            'kind': 'uploaded_npz',
            **upload_metadata,
        }
        if req.export.enabled:
            job_state['export_formats'] = list(requested_formats)

    def _target_args(job_id: str) -> tuple[Any, ...]:
        if stored_path is None:
            raise RuntimeError('uploaded refraction picks were not stored')
        return (job_id, req, state, stored_path, upload_metadata)

    launched = launch_static_job(
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        statics_kind='refraction',
        target=static_router_job_target('run_refraction_static_apply_job'),
        target_args=_target_args,
        pre_create=_pre_create,
        after_create=_after_create,
    )

    response: RefractionStaticApplyResponse = {
        'job_id': launched.job_id,
        'state': launched.state,
    }
    if req.export.enabled:
        response['requested_formats'] = list(requested_formats)
    return response


@router.post(
    '/statics/refraction/validate-with-picks',
    response_model_exclude_none=True,
)
def refraction_static_validate_with_picks(
    request: Request,
    request_json: Annotated[str, Form(...)],
    pick_npz: Annotated[UploadFile, File(...)],
) -> dict[str, object]:
    req = _parse_refraction_apply_request_json(request_json)
    if req.pick_source.kind != 'uploaded_npz':
        raise HTTPException(
            status_code=422,
            detail='pick_source.kind must be uploaded_npz',
        )
    _validate_refraction_pick_upload(pick_npz)

    state = get_state(request.app)
    cleanup_in_memory_state(state)

    with tempfile.TemporaryDirectory(prefix='refraction-validate-') as tmp:
        temp_dir = Path(tmp)
        stored_path, _size_bytes = _store_refraction_pick_upload(
            pick_npz=pick_npz,
            job_dir=temp_dir,
        )
        upload_metadata = {
            'original_filename': pick_npz.filename or '',
            'stored_name': UPLOADED_REFRACTION_PICKS_NPZ_NAME,
        }
        return validate_refraction_static_inputs_with_picks(
            req=req,
            state=state,
            pick_npz_path=stored_path,
            uploaded_pick_metadata=upload_metadata,
        )
