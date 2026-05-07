"""Refraction static correction background job service."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from app.api.schemas import RefractionStaticApplyRequest
from app.core.state import AppState
from app.services.job_runner import JobCompletion, JobFailure, run_job_with_lifecycle
from app.services.refraction_static_bedrock import (
    estimate_global_bedrock_slowness_from_first_breaks,
)

_REQUEST_JSON_NAME = 'refraction_static_request.json'
_NOT_IMPLEMENTED_MESSAGE = (
    'Refraction statics weathering-thickness conversion is not implemented yet.'
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _set_job_progress_message(
    state: AppState,
    job_id: str,
    *,
    progress: float,
    message: str,
) -> None:
    with state.lock:
        if state.jobs.get(job_id) is None:
            return
        state.jobs.set_progress(job_id, progress)
        state.jobs.set_message(job_id, message)


def _resolve_job_dir(state: AppState, job_id: str) -> Path:
    with state.lock:
        job = state.jobs.get(job_id)
        artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
    if not isinstance(artifacts_dir, str) or not artifacts_dir:
        raise ValueError('job artifacts_dir is not available')
    return Path(artifacts_dir)


def _run_refraction_static_apply_job_body(
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
) -> JobCompletion | None:
    _set_job_progress_message(
        state,
        job_id,
        progress=0.05,
        message='writing_refraction_static_request',
    )
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        job_dir / _REQUEST_JSON_NAME,
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'refraction',
            'source_file_id': req.file_id,
            'key1_byte': req.key1_byte,
            'key2_byte': req.key2_byte,
            'request': req.model_dump(mode='json'),
        },
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.20,
        message='estimating_refraction_bedrock_velocity',
    )
    estimate_global_bedrock_slowness_from_first_breaks(
        req=req,
        state=state,
        job_dir=job_dir,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.50,
        message='refraction_bedrock_velocity_solved',
    )
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


def _handle_refraction_static_job_error(_exc: Exception) -> JobFailure:
    return JobFailure(finished_ts=time.time())


def run_refraction_static_apply_job(
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
) -> None:
    """Start the refraction statics job lifecycle."""

    def worker() -> JobCompletion | None:
        return _run_refraction_static_apply_job_body(
            job_id=job_id,
            req=req,
            state=state,
        )

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=False,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=_handle_refraction_static_job_error,
    )


__all__ = ['run_refraction_static_apply_job']
